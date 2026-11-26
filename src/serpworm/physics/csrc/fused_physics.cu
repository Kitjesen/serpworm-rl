// Lossless FP64 fused CUDA physics core (V3) for N-segment pre-curved steel-strip robots.
//
// Key architectural upgrades in V3 while preserving exact IEEE-754 FP64 mechanics:
//   1. Sub-warp register shuffle Schur reduction + broadcast:
//      STRIPS_PER_SEG = 8 partitions every 32-thread warp into 4 exact 8-lane sub-warps.
//      Accumulates all 78 lower-triangular doubles of the 12x12 Schur complement and
//      12 RHS doubles via 3-step __shfl_down_sync(..., delta={4,2,1}, width=8), solves
//      the 12x12 plate system in-place on lane 0, and broadcasts vp[12] + seg_ok back
//      to all 8 lanes via __shfl_sync(..., 0, 8) with zero __syncthreads() barriers.
//   2. Packed 21-scalar in-place 6x6 Cholesky + cached diagonal reciprocals +
//      branch-free 3-stage element pipeline (prologue / interior / epilogue):
//      eliminates 1.25 KB/thread local-memory stack frames, eliminates ~450 redundant
//      FP64 divisions per strip, and keeps peak register pressure below 255 regs/thread.
//   3. 16-scalar sparse beam element assembly + __ldg read-only loads + coalesced
//      strip_buf streaming + compile-time unrolled <16> and <8> templates + CUDA Graph capture.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int STRIPS_PER_SEG = 8;
constexpr int SEGS_PER_BLOCK = 8;
constexpr int SEG_THREADS = SEGS_PER_BLOCK * STRIPS_PER_SEG;  // 64 threads/block (2 warps, 8 segments)
constexpr int BUF_STRIDE = 78;       // 6 (b_tilde) + 36 (C0_tilde) + 36 (U_tilde)
constexpr int SHM_SEG_DOUBLES = 90;  // 78 (lower-tri 12x12 S) + 12 (rhs), accessed by lane 0 only

// ---------------------------------------------------------------------------
// Small FP64 3D / quaternion / SE(3) helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ double dot3(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ double norm3(const double a[3]) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

__device__ __forceinline__ void unit3(const double a[3], double out[3]) {
    double n = norm3(a);
    double inv = 1.0 / (n > 1e-15 ? n : 1e-15);
    out[0] = a[0] * inv;
    out[1] = a[1] * inv;
    out[2] = a[2] * inv;
}

__device__ __forceinline__ void unit4(const double q[4], double out[4]) {
    double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    double inv = 1.0 / (n > 1e-15 ? n : 1e-15);
    out[0] = q[0] * inv;
    out[1] = q[1] * inv;
    out[2] = q[2] * inv;
    out[3] = q[3] * inv;
}

__device__ __forceinline__ void qmul(const double a[4], const double b[4], double out[4]) {
    double x = a[3] * b[0] + b[3] * a[0] + (a[1] * b[2] - a[2] * b[1]);
    double y = a[3] * b[1] + b[3] * a[1] + (a[2] * b[0] - a[0] * b[2]);
    double z = a[3] * b[2] + b[3] * a[2] + (a[0] * b[1] - a[1] * b[0]);
    double w = a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
    out[0] = x; out[1] = y; out[2] = z; out[3] = w;
}

__device__ __forceinline__ void qconj(const double q[4], double out[4]) {
    out[0] = -q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = q[3];
}

__device__ __forceinline__ void qrot(const double q[4], const double v[3], double out[3]) {
    double tx = 2.0 * (q[1] * v[2] - q[2] * v[1]);
    double ty = 2.0 * (q[2] * v[0] - q[0] * v[2]);
    double tz = 2.0 * (q[0] * v[1] - q[1] * v[0]);
    out[0] = v[0] + q[3] * tx + (q[1] * tz - q[2] * ty);
    out[1] = v[1] + q[3] * ty + (q[2] * tx - q[0] * tz);
    out[2] = v[2] + q[3] * tz + (q[0] * ty - q[1] * tx);
}

__device__ __forceinline__ void qmat(const double q[4], double R[9]) {
    double x = q[0], y = q[1], z = q[2], w = q[3];
    double xx = x * x, yy = y * y, zz = z * z;
    double xy = x * y, xz = x * z, yz = y * z;
    double wx = w * x, wy = w * y, wz = w * z;
    R[0] = 1.0 - 2.0 * (yy + zz); R[1] = 2.0 * (xy - wz);       R[2] = 2.0 * (xz + wy);
    R[3] = 2.0 * (xy + wz);       R[4] = 1.0 - 2.0 * (xx + zz); R[5] = 2.0 * (yz - wx);
    R[6] = 2.0 * (xz - wy);       R[7] = 2.0 * (yz + wx);       R[8] = 1.0 - 2.0 * (xx + yy);
}

__device__ __forceinline__ void qlog(const double q_in[4], double out[3]) {
    double s = q_in[3] < 0.0 ? -1.0 : 1.0;
    double x = s * q_in[0], y = s * q_in[1], z = s * q_in[2], w = s * q_in[3];
    double n = sqrt(x * x + y * y + z * z);
    double denom = n > 1e-15 ? n : 1e-15;
    double scale = 2.0 * atan2(n, w) / denom;
    out[0] = x * scale;
    out[1] = y * scale;
    out[2] = z * scale;
}

__device__ __forceinline__ void qalign(const double q[4], const double t[3], double out[4]) {
    double ty = 2.0 * q[2];
    double tz = -2.0 * q[1];
    double a[3] = {
        1.0 + (q[1] * tz - q[2] * ty),
        q[3] * ty - q[0] * tz,
        q[3] * tz + q[0] * ty
    };
    double c = dot3(a, t);
    double cr[3];
    cross3(a, t, cr);
    double raw[4] = {cr[0], cr[1], cr[2], 1.0 + c};
    double r[4];
    unit4(raw, r);
    qmul(r, q, out);
}

__device__ __forceinline__ void midpoint_frame(
    const double p0[3], const double p3[3],
    const double q0[4], const double q1[4],
    double L, double center[3], double r_out[4]
) {
    double ex[3] = {1.0, 0.0, 0.0};
    double t0[3], t1[3];
    qrot(q0, ex, t0);
    qrot(q1, ex, t1);
    double p1[3], p2[3], diff03[3], d01[3], d12[3];
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        p1[k] = p0[k] + L * t0[k] / 3.0;
        p2[k] = p3[k] - L * t1[k] / 3.0;
        diff03[k] = p3[k] - p0[k];
        d01[k] = p1[k] - p0[k];
        d12[k] = p2[k] - p1[k];
    }
    bool compressed = (dot3(d01, d12) < 0.0) && (norm3(diff03) < 0.4 * L);
    if (compressed) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) center[k] = 0.5 * (p0[k] + p3[k]);
        #pragma unroll
        for (int k = 0; k < 4; ++k) r_out[k] = q0[k];
        return;
    }
    double traw[3], t[3];
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        center[k] = (p0[k] + 3.0 * p1[k] + 3.0 * p2[k] + p3[k]) * 0.125;
        traw[k] = -0.75 * p0[k] - 0.75 * p1[k] + 0.75 * p2[k] + 0.75 * p3[k];
    }
    unit3(traw, t);
    double r0[4], r1[4];
    qalign(q0, t, r0);
    qalign(q1, t, r1);
    double d = r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2] + r0[3] * r1[3];
    double s = d < 0.0 ? -1.0 : 1.0;
    double rsum[4] = {r0[0] + s * r1[0], r0[1] + s * r1[1], r0[2] + s * r1[2], r0[3] + s * r1[3]};
    double rmid[4];
    unit4(rsum, rmid);
    qalign(rmid, t, r_out);
}

__device__ __forceinline__ void se3_integrate(double x[7], const double v[6], double h) {
    x[0] += h * v[0];
    x[1] += h * v[1];
    x[2] += h * v[2];
    double ox = 0.5 * h * v[3];
    double oy = 0.5 * h * v[4];
    double oz = 0.5 * h * v[5];
    double theta2 = ox * ox + oy * oy + oz * oz;
    double theta = sqrt(theta2);
    double sinc_val = (theta < 1e-8) ? (1.0 - theta2 / 6.0) : (sin(theta) / theta);
    double exp_q[4] = {ox * sinc_val, oy * sinc_val, oz * sinc_val, cos(theta)};
    double q_new[4];
    qmul(exp_q, x + 3, q_new);
    unit4(q_new, x + 3);
}

// ---------------------------------------------------------------------------
// In-place branch-free SPD Cholesky solvers (packed 6x6 register + 12x12 SRAM)
// ---------------------------------------------------------------------------

// In-place 6x6 SPD Cholesky on packed lower-triangular L_tri[21] in registers.
// Strict lower triangle stores L[i][j] (j < i); diagonal slots store 1.0 / L[i][i].
__device__ __forceinline__ bool chol_6x6_packed(double L_tri[21]) {
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int row_off = (i * (i + 1)) >> 1;
        double diag = L_tri[row_off + i];
        #pragma unroll
        for (int k = 0; k < i; ++k) {
            double lik = L_tri[row_off + k];
            diag -= lik * lik;
        }
        if (!(diag > 0.0) || !isfinite(diag)) return false;
        double lii = sqrt(diag);
        double inv_lii = 1.0 / lii;
        L_tri[row_off + i] = inv_lii;
        #pragma unroll
        for (int j = i + 1; j < 6; ++j) {
            const int j_off = (j * (j + 1)) >> 1;
            double sum = L_tri[j_off + i];
            #pragma unroll
            for (int k = 0; k < i; ++k) {
                sum -= L_tri[j_off + k] * L_tri[row_off + k];
            }
            L_tri[j_off + i] = sum * inv_lii;
        }
    }
    return true;
}

__device__ __forceinline__ void solve_L_6x1_inplace(const double L_tri[21], double b[6]) {
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int row_off = (i * (i + 1)) >> 1;
        double s = b[i];
        #pragma unroll
        for (int k = 0; k < i; ++k) s -= L_tri[row_off + k] * b[k];
        b[i] = s * L_tri[row_off + i];
    }
}

__device__ __forceinline__ void solve_LT_6x1_inplace(const double L_tri[21], double y[6]) {
    #pragma unroll
    for (int i = 5; i >= 0; --i) {
        const int row_off = (i * (i + 1)) >> 1;
        double s = y[i];
        #pragma unroll
        for (int k = i + 1; k < 6; ++k) {
            const int k_off = (k * (k + 1)) >> 1;
            s -= L_tri[k_off + i] * y[k];
        }
        y[i] = s * L_tri[row_off + i];
    }
}

__device__ __forceinline__ void solve_L_6x6_inplace(const double L_tri[21], double B[6][6]) {
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int row_off = (i * (i + 1)) >> 1;
        double inv_lii = L_tri[row_off + i];
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            double s = B[i][c];
            #pragma unroll
            for (int k = 0; k < i; ++k) s -= L_tri[row_off + k] * B[k][c];
            B[i][c] = s * inv_lii;
        }
    }
}

__device__ __forceinline__ void solve_LT_6x6_inplace(const double L_tri[21], double Y[6][6]) {
    #pragma unroll
    for (int i = 5; i >= 0; --i) {
        const int row_off = (i * (i + 1)) >> 1;
        double inv_lii = L_tri[row_off + i];
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            double s = Y[i][c];
            #pragma unroll
            for (int k = i + 1; k < 6; ++k) {
                const int k_off = (k * (k + 1)) >> 1;
                s -= L_tri[k_off + i] * Y[k][c];
            }
            Y[i][c] = s * inv_lii;
        }
    }
}

// Zero-stack-frame in-place 12x12 Cholesky solve on S[0..77] (lower-tri) and b = S + 78 (RHS -> sol)
__device__ bool chol_solve_12x12_packed(double* __restrict__ S) {
    double* b = S + 78;
    for (int i = 0; i < 12; ++i) {
        int row_off = (i * (i + 1)) >> 1;
        double diag = S[row_off + i];
        for (int k = 0; k < i; ++k) {
            double lik = S[row_off + k];
            diag -= lik * lik;
        }
        if (!(diag > 0.0) || !isfinite(diag)) return false;
        double lii = sqrt(diag);
        double inv_lii = 1.0 / lii;
        S[row_off + i] = inv_lii;
        for (int j = i + 1; j < 12; ++j) {
            int j_off = (j * (j + 1)) >> 1;
            double sum = S[j_off + i];
            for (int k = 0; k < i; ++k) {
                sum -= S[j_off + k] * S[row_off + k];
            }
            S[j_off + i] = sum * inv_lii;
        }
    }
    for (int i = 0; i < 12; ++i) {
        int row_off = (i * (i + 1)) >> 1;
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= S[row_off + k] * b[k];
        b[i] = s * S[row_off + i];
    }
    for (int i = 11; i >= 0; --i) {
        int row_off = (i * (i + 1)) >> 1;
        double s = b[i];
        for (int k = i + 1; k < 12; ++k) {
            int k_off = (k * (k + 1)) >> 1;
            s -= S[k_off + i] * b[k];
        }
        b[i] = s * S[row_off + i];
    }
    return true;
}

__device__ __forceinline__ double subwarp_reduce_sum8(unsigned mask, double val) {
    val += __shfl_down_sync(mask, val, 4, 8);
    val += __shfl_down_sync(mask, val, 2, 8);
    val += __shfl_down_sync(mask, val, 1, 8);
    return val;
}

// ---------------------------------------------------------------------------
// Sparse 16-scalar beam element assembly helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void sparse_matvec_12(const double p[16], const double x[12], double scale, double acc[12]) {
    acc[0]  += scale * (p[0] * x[0] + p[8] * x[6]);
    acc[1]  += scale * (p[1] * x[1] + p[6] * x[5] + p[9]  * x[7] + p[14] * x[11]);
    acc[2]  += scale * (p[2] * x[2] + p[7] * x[4] + p[10] * x[8] + p[15] * x[10]);

    acc[3]  += scale * (p[3] * x[3] + p[11] * x[9]);
    acc[4]  += scale * (p[7] * x[2] + p[4]  * x[4] - p[15] * x[8] + p[12] * x[10]);
    acc[5]  += scale * (p[6] * x[1] + p[5]  * x[5] - p[14] * x[7] + p[13] * x[11]);

    acc[6]  += scale * (p[8]  * x[0] + p[0] * x[6]);
    acc[7]  += scale * (p[9]  * x[1] - p[14] * x[5] + p[1] * x[7] - p[6] * x[11]);
    acc[8]  += scale * (p[10] * x[2] - p[15] * x[4] + p[2] * x[8] - p[7] * x[10]);

    acc[9]  += scale * (p[11] * x[3] + p[3] * x[9]);
    acc[10] += scale * (p[15] * x[2] + p[12] * x[4] - p[7] * x[8] + p[4] * x[10]);
    acc[11] += scale * (p[14] * x[1] + p[13] * x[5] - p[6] * x[7] + p[5] * x[11]);
}

__device__ __forceinline__ void rot_diag_3x3(const double R[9], double d0, double d1, double d2, double out[3][3]) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        const double* rr = R + r * 3;
        #pragma unroll
        for (int c = r; c < 3; ++c) {
            const double* rc = R + c * 3;
            double v = d0 * rr[0] * rc[0] + d1 * rr[1] * rc[1] + d2 * rr[2] * rc[2];
            out[r][c] = v;
            out[c][r] = v;
        }
    }
}

__device__ __forceinline__ void rot_two_3x3(const double R[9], double b12, double b21, double out[3][3]) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        double r1 = R[r * 3 + 1], r2 = R[r * 3 + 2];
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            out[r][c] = b12 * r1 * R[c * 3 + 2] + b21 * r2 * R[c * 3 + 1];
        }
    }
}

template <bool HAS_OFF0, bool HAS_OFF1>
__device__ __forceinline__ void compute_element_frame_and_rhs(
    const double x0[7], const double x1[7],
    const double v0[6], const double v1[6],
    const double* __restrict__ tr0,
    const double* __restrict__ tr1,
    const double* __restrict__ rest_loc,
    const double* __restrict__ K_pack,
    const double* __restrict__ M_pack,
    double L, double h, double m_c, double k_c,
    double R[9], double off0[3], double off1[3], double rhs_e[12], double A_loc[16]
) {
    double q0[4], q1[4], p0[3], p1[3];
    if constexpr (HAS_OFF0) {
        qmul(x0 + 3, tr0 + 3, q0);
        qrot(x0 + 3, tr0, off0);
        p0[0] = x0[0] + off0[0]; p0[1] = x0[1] + off0[1]; p0[2] = x0[2] + off0[2];
    } else {
        #pragma unroll
        for (int k = 0; k < 4; ++k) q0[k] = x0[3 + k];
        p0[0] = x0[0]; p0[1] = x0[1]; p0[2] = x0[2];
    }
    if constexpr (HAS_OFF1) {
        qmul(x1 + 3, tr1 + 3, q1);
        qrot(x1 + 3, tr1, off1);
        p1[0] = x1[0] + off1[0]; p1[1] = x1[1] + off1[1]; p1[2] = x1[2] + off1[2];
    } else {
        #pragma unroll
        for (int k = 0; k < 4; ++k) q1[k] = x1[3 + k];
        p1[0] = x1[0]; p1[1] = x1[1]; p1[2] = x1[2];
    }

    double center[3], r[4], inv_r[4];
    midpoint_frame(p0, p1, q0, q1, L, center, r);
    qconj(r, inv_r);
    qmat(r, R);

    double dp0[3] = {p0[0] - center[0], p0[1] - center[1], p0[2] - center[2]};
    double dp1[3] = {p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]};
    double dq0[4], dq1[4], u[12], du[12];
    qrot(inv_r, dp0, u + 0);
    qmul(inv_r, q0, dq0); qlog(dq0, u + 3);
    qrot(inv_r, dp1, u + 6);
    qmul(inv_r, q1, dq1); qlog(dq1, u + 9);
    #pragma unroll
    for (int k = 0; k < 12; ++k) du[k] = u[k] - __ldg(rest_loc + k);

    double vg0[6] = {v0[0], v0[1], v0[2] - 9.81 * h, v0[3], v0[4], v0[5]};
    double vg1[6] = {v1[0], v1[1], v1[2] - 9.81 * h, v1[3], v1[4], v1[5]};
    double lv0[3] = {vg0[0], vg0[1], vg0[2]};
    double lv1[3] = {vg1[0], vg1[1], vg1[2]};
    if constexpr (HAS_OFF0) {
        double cr[3]; cross3(vg0 + 3, off0, cr);
        lv0[0] += cr[0]; lv0[1] += cr[1]; lv0[2] += cr[2];
    }
    if constexpr (HAS_OFF1) {
        double cr[3]; cross3(vg1 + 3, off1, cr);
        lv1[0] += cr[0]; lv1[1] += cr[1]; lv1[2] += cr[2];
    }

    double Tvg[12];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        Tvg[0 + i] = R[i] * lv0[0] + R[3 + i] * lv0[1] + R[6 + i] * lv0[2];
        Tvg[3 + i] = R[i] * vg0[3] + R[3 + i] * vg0[4] + R[6 + i] * vg0[5];
        Tvg[6 + i] = R[i] * lv1[0] + R[3 + i] * lv1[1] + R[6 + i] * lv1[2];
        Tvg[9 + i] = R[i] * vg1[3] + R[3 + i] * vg1[4] + R[6 + i] * vg1[5];
    }

    double mp[16], kp[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        mp[k] = __ldg(M_pack + k);
        kp[k] = __ldg(K_pack + k);
        A_loc[k] = m_c * mp[k] + k_c * kp[k];
    }

    double floc[12] = {0.0};
    sparse_matvec_12(mp, Tvg, 1.0, floc);
    sparse_matvec_12(kp, du, -h, floc);

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const double* rr = R + i * 3;
        rhs_e[0 + i] = rr[0] * floc[0] + rr[1] * floc[1] + rr[2] * floc[2];
        rhs_e[3 + i] = rr[0] * floc[3] + rr[1] * floc[4] + rr[2] * floc[5];
        rhs_e[6 + i] = rr[0] * floc[6] + rr[1] * floc[7] + rr[2] * floc[8];
        rhs_e[9 + i] = rr[0] * floc[9] + rr[1] * floc[10] + rr[2] * floc[11];
    }
    if constexpr (HAS_OFF0) {
        double cr[3]; cross3(off0, rhs_e + 0, cr);
        rhs_e[3] += cr[0]; rhs_e[4] += cr[1]; rhs_e[5] += cr[2];
    }
    if constexpr (HAS_OFF1) {
        double cr[3]; cross3(off1, rhs_e + 6, cr);
        rhs_e[9] += cr[0]; rhs_e[10] += cr[1]; rhs_e[11] += cr[2];
    }
}

// Add un-sheared A00 lower triangle (for HAS_OFF0 == false) directly into D_tri[21]
__device__ __forceinline__ void add_A00_unsheared_tri(
    const double R[9], const double A_loc[16], double D_tri[21]
) {
    double Q00[3][3], Q11[3][3], Q01[3][3];
    rot_diag_3x3(R, A_loc[0], A_loc[1], A_loc[2], Q00);
    rot_diag_3x3(R, A_loc[3], A_loc[4], A_loc[5], Q11);
    rot_two_3x3(R, A_loc[6], A_loc[7], Q01);

    int idx = 0;
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c <= r; ++c) {
            D_tri[idx++] += Q00[r][c];
        }
    }
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            D_tri[idx++] += Q01[c][r];
        }
        #pragma unroll
        for (int c = 0; c <= r; ++c) {
            D_tri[idx++] += Q11[r][c];
        }
    }
}

// Assemble A01[6][6] from R and A_loc[8..15]
__device__ __forceinline__ void build_A01_unsheared(
    const double R[9], const double A_loc[16], double A01[6][6]
) {
    double Q02[3][3], Q13[3][3], Q03[3][3];
    rot_diag_3x3(R, A_loc[8],  A_loc[9],  A_loc[10], Q02);
    rot_diag_3x3(R, A_loc[11], A_loc[12], A_loc[13], Q13);
    rot_two_3x3(R, A_loc[14], A_loc[15], Q03);

    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            A01[r][c]         = Q02[r][c];
            A01[r][3 + c]     = Q03[r][c];
            A01[3 + r][c]     = -Q03[c][r];
            A01[3 + r][3 + c] = Q13[r][c];
        }
    }
}

// Set D_tri[21] = un-sheared A11_tri - (W^T @ W)_tri
__device__ __forceinline__ void set_A11_minus_WtW_tri(
    const double R[9], const double A_loc[16],
    const double W[6][6], double D_tri[21]
) {
    double Q00[3][3], Q11[3][3], Q01[3][3];
    rot_diag_3x3(R, A_loc[0], A_loc[1], A_loc[2], Q00);
    rot_diag_3x3(R, A_loc[3], A_loc[4], A_loc[5], Q11);
    rot_two_3x3(R, A_loc[6], A_loc[7], Q01);

    int idx = 0;
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c <= r; ++c) {
            double sw = 0.0;
            #pragma unroll
            for (int k = 0; k < 6; ++k) sw += W[k][r] * W[k][c];
            D_tri[idx++] = Q00[r][c] - sw;
        }
    }
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int rr = 3 + r;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            double sw = 0.0;
            #pragma unroll
            for (int k = 0; k < 6; ++k) sw += W[k][rr] * W[k][c];
            D_tri[idx++] = -Q01[c][r] - sw;
        }
        #pragma unroll
        for (int c = 0; c <= r; ++c) {
            const int cc = 3 + c;
            double sw = 0.0;
            #pragma unroll
            for (int k = 0; k < 6; ++k) sw += W[k][rr] * W[k][cc];
            D_tri[idx++] = Q11[r][c] - sw;
        }
    }
}

// ---------------------------------------------------------------------------
// Analytical C1 Pipe/Bend/S-Curve/Right-Angle-L-Duct & 600-Node Strip Contact
// Exact bit-for-bit mirror of serpworm/physics/pipe_geometry.py
// ---------------------------------------------------------------------------

__device__ __forceinline__ void accumulate_surface_contact_3d(
    const double vel[3],
    double pen,
    const double n[3],
    double k_w,
    double c_w,
    double mu_w,
    double h,
    int implicit_c,
    double f_acc[3],
    double k_tri6[6],
    double& fn_acc
) {
    if (pen <= 0.0 || k_w <= 0.0) return;
    const double vn = vel[0] * n[0] + vel[1] * n[1] + vel[2] * n[2];
    const double raw_fn = k_w * pen - c_w * vn;
    if (raw_fn <= 0.0) return;
    fn_acc += raw_fn;

    const double vt0 = vel[0] - vn * n[0];
    const double vt1 = vel[1] - vn * n[1];
    const double vt2 = vel[2] - vn * n[2];
    const double vt_mag = sqrt(vt0 * vt0 + vt1 * vt1 + vt2 * vt2);
    const double v_reg = 0.015;
    const double th = tanh(vt_mag / v_reg);
    const double inv_vt = (vt_mag > 1e-12) ? (th / vt_mag) : (1.0 / v_reg);
    const double fscale = -mu_w * raw_fn * inv_vt;

    f_acc[0] += raw_fn * n[0] + fscale * vt0;
    f_acc[1] += raw_fn * n[1] + fscale * vt1;
    f_acc[2] += raw_fn * n[2] + fscale * vt2;

    if (implicit_c) {
        const double ct = (mu_w * raw_fn / v_reg) * fmax(1.0 - th * th, 0.0);
        f_acc[0] += c_w * vn * n[0] + ct * vt0;
        f_acc[1] += c_w * vn * n[1] + ct * vt1;
        f_acc[2] += c_w * vn * n[2] + ct * vt2;

        const double wn = h * c_w + (h * h) * k_w;
        const double wt = h * ct;
        const double dw = wn - wt;
        k_tri6[0] += dw * n[0] * n[0] + wt;
        k_tri6[1] += dw * n[1] * n[0];
        k_tri6[2] += dw * n[1] * n[1] + wt;
        k_tri6[3] += dw * n[2] * n[0];
        k_tri6[4] += dw * n[2] * n[1];
        k_tri6[5] += dw * n[2] * n[2] + wt;
    }
}

__device__ __forceinline__ void eval_pipe_contact_cuda(
    const double pos[3],
    const double vel[3],
    const double pc[12],
    double node_radius,
    double h,
    int implicit_c,
    double k_scale,
    double c_wall,
    double f_acc[3],
    double k_tri6[6],
    double& fn_acc
) {
    const double ptype = pc[0];
    const double k_w = pc[10] * k_scale;
    if (ptype < 0.5 || k_w <= 0.0) return;

    const double y0 = pc[1];
    const double r_in = pc[2];
    const double x_en = pc[3];
    const double x_b = pc[4];
    const double R_b = fmax(pc[5], 1e-3);
    const double ang = pc[6];
    const double sgn = (pc[7] >= 0.0) ? 1.0 : -1.0;
    const double L_ex = pc[8];
    const double z_c = pc[9];
    const double r_fil = fmax(pc[11], 1e-4);

    const double px = pos[0], py = pos[1], pz = pos[2];
    const double funnel = 0.08;
    const bool in_fun = (px <= x_en + funnel) && (px > x_en);
    double u_fun = (x_en + funnel - px) / funnel;
    if (u_fun < 0.0) u_fun = 0.0;
    if (u_fun > 1.0) u_fun = 1.0;
    const double r_flare = in_fun ? (0.055 * (1.0 - u_fun)) : 0.0;
    const double r_eff = fmax(r_in + r_flare - node_radius, 0.005);

    if (ptype < 3.5 || (ptype > 4.5 && ptype < 5.5) || (ptype > 7.5 && ptype < 8.5)) {
        double qx = 0.0, qy = 0.0;
        bool active_c = false;
        double r_eff_c = r_eff;

        if (ptype < 1.5) {
            if (px <= x_en + funnel && px >= x_b) {
                qx = px;
                qy = y0;
                active_c = true;
            }
        } else if (ptype < 2.5) {
            const double cy2 = y0 + sgn * R_b;
            const double ax_a = fmin(fmax(px, x_b), x_en + funnel);
            const double ay_a = y0;
            const double d2_a = (px - ax_a) * (px - ax_a) + (py - ay_a) * (py - ay_a);

            const double ux = px - x_b;
            const double uy = py - cy2;
            double alpha = atan2(-ux, -sgn * uy);
            if (alpha < 0.0) alpha = 0.0;
            if (alpha > 1.5707963267948966) alpha = 1.5707963267948966;
            const double ax_b = x_b - R_b * sin(alpha);
            const double ay_b = cy2 - sgn * R_b * cos(alpha);
            const double d2_b = (px - ax_b) * (px - ax_b) + (py - ay_b) * (py - ay_b);

            const double ax_c = x_b - R_b;
            double sy_c = sgn * (py - cy2);
            if (sy_c < 0.0) sy_c = 0.0;
            if (sy_c > L_ex) sy_c = L_ex;
            const double ay_c = cy2 + sgn * sy_c;
            const double d2_c = (px - ax_c) * (px - ax_c) + (py - ay_c) * (py - ay_c);

            qx = ax_a; qy = ay_a;
            double bd2 = d2_a;
            if (d2_b < bd2) { qx = ax_b; qy = ay_b; bd2 = d2_b; }
            if (d2_c < bd2) { qx = ax_c; qy = ay_c; }

            const bool past_in = (px > x_en + funnel) && (d2_a <= d2_b) && (d2_a <= d2_c);
            const bool past_out = (sgn * (py - cy2) > L_ex) && (d2_c <= d2_a) && (d2_c <= d2_b);
            active_c = (!past_in) && (!past_out);
        } else if (ptype < 3.5) {
            double phi = ang;
            if (phi < 0.1) phi = 0.1;
            if (phi > 1.5707963267948966) phi = 1.5707963267948966;
            const double dx_a = R_b * sin(phi);
            const double dy_a = R_b * (1.0 - cos(phi));

            const double sx1 = fmin(fmax(px, x_b), x_en + funnel);
            const double sy1 = y0;
            const double d1 = (px - sx1) * (px - sx1) + (py - sy1) * (py - sy1);

            const double c1x = x_b, c1y = y0 + sgn * R_b;
            double a1 = atan2(-(px - c1x), -sgn * (py - c1y));
            if (a1 < 0.0) a1 = 0.0;
            if (a1 > phi) a1 = phi;
            const double sx2 = c1x - R_b * sin(a1);
            const double sy2 = c1y - sgn * R_b * cos(a1);
            const double d2 = (px - sx2) * (px - sx2) + (py - sy2) * (py - sy2);

            const double c2x = x_b - 2.0 * dx_a;
            const double c2y = y0 + sgn * (2.0 * dy_a - R_b);
            double a2 = atan2(px - c2x, sgn * (py - c2y));
            if (a2 < 0.0) a2 = 0.0;
            if (a2 > phi) a2 = phi;
            const double sx3 = c2x + R_b * sin(a2);
            const double sy3 = c2y + sgn * R_b * cos(a2);
            const double d3 = (px - sx3) * (px - sx3) + (py - sy3) * (py - sy3);

            const double y_ex = y0 + sgn * 2.0 * dy_a;
            const double sx4 = fmin(fmax(px, c2x - L_ex), c2x);
            const double sy4 = y_ex;
            const double d4 = (px - sx4) * (px - sx4) + (py - sy4) * (py - sy4);

            qx = sx1; qy = sy1;
            double bd = d1;
            if (d2 < bd) { qx = sx2; qy = sy2; bd = d2; }
            if (d3 < bd) { qx = sx3; qy = sy3; bd = d3; }
            if (d4 < bd) { qx = sx4; qy = sy4; }

            const bool past_in = (px > x_en + funnel) && (d1 <= d2) && (d1 <= d3) && (d1 <= d4);
            const bool past_out = (px < c2x - L_ex) && (d4 <= d1) && (d4 <= d2) && (d4 <= d3);
            active_c = (!past_in) && (!past_out);
        } else if (ptype < 5.5) {
            const double cy5 = y0 + sgn * R_b;
            const double y_ret = y0 + 2.0 * sgn * R_b;
            const double ax_a = fmin(fmax(px, x_b), x_en + funnel);
            const double ay_a = y0;
            const double d2_a = (px - ax_a) * (px - ax_a) + (py - ay_a) * (py - ay_a);

            const double ux5 = px - x_b;
            const double uy5 = sgn * (py - cy5);
            const double alpha5 = atan2(fmax(-ux5, 0.0), -uy5);
            const double ax_b = x_b - R_b * sin(alpha5);
            const double ay_b = cy5 - sgn * R_b * cos(alpha5);
            const double d2_b = (px - ax_b) * (px - ax_b) + (py - ay_b) * (py - ay_b);

            const double ax_c = fmin(fmax(px, x_b), x_b + L_ex);
            const double ay_c = y_ret;
            const double d2_c = (px - ax_c) * (px - ax_c) + (py - ay_c) * (py - ay_c);

            qx = ax_a; qy = ay_a;
            double bd2 = d2_a;
            if (d2_b < bd2) { qx = ax_b; qy = ay_b; bd2 = d2_b; }
            if (d2_c < bd2) { qx = ax_c; qy = ay_c; }

            const bool past_in = (px > x_en + funnel) && (d2_a <= d2_b) && (d2_a <= d2_c);
            const bool past_out = (px > x_b + L_ex) && (d2_c <= d2_a) && (d2_c <= d2_b);
            active_c = (!past_in) && (!past_out);
            const double rf5 = (sgn * (py - y0) < R_b) ? r_flare : 0.0;
            r_eff_c = fmax(r_in + rf5 - node_radius, 0.005);
        } else {
            double phi = ang;
            if (phi < 0.1) phi = 0.1;
            if (phi > 1.35) phi = 1.35;
            const double dx_a = R_b * sin(phi);
            const double dy_a = R_b * (1.0 - cos(phi));

            const double sx1 = fmin(fmax(px, x_b), x_en + funnel);
            const double sy1 = y0;
            const double d1 = (px - sx1) * (px - sx1) + (py - sy1) * (py - sy1);

            const double c1x = x_b, c1y = y0 + sgn * R_b;
            double a1 = atan2(-(px - c1x), -sgn * (py - c1y));
            if (a1 < 0.0) a1 = 0.0;
            if (a1 > phi) a1 = phi;
            const double sx2 = c1x - R_b * sin(a1);
            const double sy2 = c1y - sgn * R_b * cos(a1);
            const double d2 = (px - sx2) * (px - sx2) + (py - sy2) * (py - sy2);

            const double c2x = x_b - 2.0 * dx_a;
            const double c2y = y0 + sgn * (2.0 * dy_a - R_b);
            double a2 = atan2(px - c2x, sgn * (py - c2y));
            if (a2 < -phi) a2 = -phi;
            if (a2 > phi) a2 = phi;
            const double sx3 = c2x + R_b * sin(a2);
            const double sy3 = c2y + sgn * R_b * cos(a2);
            const double d3 = (px - sx3) * (px - sx3) + (py - sy3) * (py - sy3);

            const double c3x = x_b - 4.0 * dx_a;
            const double c3y = y0 + sgn * R_b;
            double a3 = atan2(-(px - c3x), -sgn * (py - c3y));
            if (a3 < -phi) a3 = -phi;
            if (a3 > 0.0) a3 = 0.0;
            const double sx4 = c3x - R_b * sin(a3);
            const double sy4 = c3y - sgn * R_b * cos(a3);
            const double d4 = (px - sx4) * (px - sx4) + (py - sy4) * (py - sy4);

            const double sx5 = fmin(fmax(px, c3x - L_ex), c3x);
            const double sy5 = y0;
            const double d5 = (px - sx5) * (px - sx5) + (py - sy5) * (py - sy5);

            qx = sx1; qy = sy1;
            double bd = d1;
            if (d2 < bd) { qx = sx2; qy = sy2; bd = d2; }
            if (d3 < bd) { qx = sx3; qy = sy3; bd = d3; }
            if (d4 < bd) { qx = sx4; qy = sy4; bd = d4; }
            if (d5 < bd) { qx = sx5; qy = sy5; }

            const bool past_in = (px > x_en + funnel) && (d1 <= d2) && (d1 <= d3) && (d1 <= d4) && (d1 <= d5);
            const bool past_out = (px < c3x - L_ex) && (d5 <= d1) && (d5 <= d2) && (d5 <= d3) && (d5 <= d4);
            active_c = (!past_in) && (!past_out);
        }

        if (active_c) {
            const double dx = px - qx;
            const double dy = py - qy;
            const double dz = pz - z_c;
            const double dr = sqrt(dx * dx + dy * dy + dz * dz + 1e-24);
            const double pen = dr - r_eff_c;
            if (pen > 0.0) {
                const double inv_dr = 1.0 / fmax(dr, 1e-12);
                const double n_cyl[3] = {-dx * inv_dr, -dy * inv_dr, -dz * inv_dr};
                accumulate_surface_contact_3d(vel, pen, n_cyl, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
            }
        }
    } else if (ptype < 4.5) {
        const double lx = px - x_b;
        const double ly = sgn * (py - y0);
        const double lx_en = x_en - x_b;
        const double hw_core = fmax(r_in - node_radius, 0.005);
        const double hw = hw_core + r_flare;
        if (lx <= lx_en + funnel && ly <= L_ex && lx >= -hw_core - 0.15 && ly >= -hw_core - 0.15) {
            if (-hw_core - lx > 0.0) {
                const double n_end[3] = {1.0, 0.0, 0.0};
                accumulate_surface_contact_3d(vel, -hw_core - lx, n_end, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
            }
            if (-hw - ly > 0.0) {
                const double n_out[3] = {0.0, sgn, 0.0};
                accumulate_surface_contact_3d(vel, -hw - ly, n_out, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
            }
            const double cx_f = r_in + r_fil;
            const double cy_f = r_in + r_fil;
            const double R_cf = r_fil + node_radius;
            if (lx > cx_f) {
                if (ly - hw > 0.0) {
                    const double n_ew[3] = {0.0, -sgn, 0.0};
                    accumulate_surface_contact_3d(vel, ly - hw, n_ew, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                }
            } else if (ly > cy_f) {
                if (lx - hw_core > 0.0) {
                    const double n_xw[3] = {-1.0, 0.0, 0.0};
                    accumulate_surface_contact_3d(vel, lx - hw_core, n_xw, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                }
            } else if (lx > r_in - node_radius - r_fil && ly > r_in - node_radius - r_fil) {
                const double dfx = lx - cx_f;
                const double dfy = ly - cy_f;
                const double d_f = sqrt(dfx * dfx + dfy * dfy + 1e-24);
                const double pen_cf = R_cf - d_f;
                if (pen_cf > 0.0) {
                    const double inv_df = 1.0 / fmax(d_f, 1e-12);
                    const double n_cf[3] = {dfx * inv_df, sgn * dfy * inv_df, 0.0};
                    accumulate_surface_contact_3d(vel, pen_cf, n_cf, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                }
            }
            const double pen_top = pz - (z_c - node_radius);
            if (pen_top > 0.0) {
                const double n_top[3] = {0.0, 0.0, -1.0};
                accumulate_surface_contact_3d(vel, pen_top, n_top, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
            }
        }
    } else if (ptype < 6.5) {
        // Type 6: Smooth C1 Conical Taper / Reducer Pipe
        const double L_tap = fmax(R_b, 0.10);
        const double r_out = fmax(r_fil, 0.080);
        const double x_end6 = x_b - L_tap - L_ex;
        if (px <= x_en + funnel && px >= x_end6) {
            double r_loc, dr_dx;
            if (px >= x_b) {
                r_loc = r_in + r_flare;
                dr_dx = 0.0;
            } else if (px > x_b - L_tap) {
                const double u = (x_b - px) / L_tap;
                r_loc = r_in + 0.5 * (r_out - r_in) * (1.0 - cos(3.14159265358979323846 * u));
                dr_dx = -0.5 * (r_out - r_in) * (3.14159265358979323846 / L_tap) * sin(3.14159265358979323846 * u);
            } else {
                r_loc = r_out;
                dr_dx = 0.0;
            }
            const double dy6 = py - y0;
            const double dz6 = pz - z_c;
            const double rho6 = sqrt(dy6 * dy6 + dz6 * dz6 + 1e-24);
            const double r_eff6 = fmax(r_loc - node_radius, 0.005);
            const double inv_sl = 1.0 / sqrt(1.0 + dr_dx * dr_dx);
            const double pen6 = (rho6 - r_eff6) * inv_sl;
            if (pen6 > 0.0) {
                const double inv_rho6 = 1.0 / fmax(rho6, 1e-12);
                const double n6[3] = {dr_dx * inv_sl, -(dy6 * inv_rho6) * inv_sl, -(dz6 * inv_rho6) * inv_sl};
                accumulate_surface_contact_3d(vel, pen6, n6, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
            }
        }
    } else if (ptype < 7.5) {
        // Type 7: Z-Shaped Double 90-deg Sharp Square Chicane
        const double lx = px - x_b;
        const double ly = sgn * (py - y0);
        const double lx_en = x_en - x_b;
        const double L_mid = fmax(R_b, 0.50);
        const double ly_half = 0.5 * L_mid;
        const double hw_core = fmax(r_in - node_radius, 0.005);
        const double hw = hw_core + r_flare;
        const double cx_f = r_in + r_fil;
        const double cy_f = r_in + r_fil;
        const double R_cf = r_fil + node_radius;
        bool in_z = false;
        if (ly <= ly_half) {
            if (lx <= lx_en + funnel && lx >= -hw_core - 0.15 && ly >= -hw_core - 0.15) {
                in_z = true;
                if (-hw_core - lx > 0.0) {
                    const double n_e[3] = {1.0, 0.0, 0.0};
                    accumulate_surface_contact_3d(vel, -hw_core - lx, n_e, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                }
                if (-hw - ly > 0.0) {
                    const double n_o[3] = {0.0, sgn, 0.0};
                    accumulate_surface_contact_3d(vel, -hw - ly, n_o, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                }
                if (lx > cx_f) {
                    if (ly - hw > 0.0) {
                        const double n_i[3] = {0.0, -sgn, 0.0};
                        accumulate_surface_contact_3d(vel, ly - hw, n_i, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                    }
                } else if (ly > cy_f) {
                    if (lx - hw_core > 0.0) {
                        const double n_i[3] = {-1.0, 0.0, 0.0};
                        accumulate_surface_contact_3d(vel, lx - hw_core, n_i, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                    }
                } else if (lx > r_in - node_radius - r_fil && ly > r_in - node_radius - r_fil) {
                    const double dfx = lx - cx_f, dfy = ly - cy_f;
                    const double d_f = sqrt(dfx * dfx + dfy * dfy + 1e-24);
                    if (R_cf - d_f > 0.0) {
                        const double inv_df = 1.0 / fmax(d_f, 1e-12);
                        const double n_c[3] = {dfx * inv_df, sgn * dfy * inv_df, 0.0};
                        accumulate_surface_contact_3d(vel, R_cf - d_f, n_c, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                    }
                }
            }
        } else {
            const double ux = -lx;
            const double uy = L_mid - ly;
            if (ux <= L_ex && ux >= -hw_core - 0.15 && uy >= -hw_core - 0.15) {
                in_z = true;
                if (-hw_core - uy > 0.0) {
                    const double n_e[3] = {0.0, -sgn, 0.0};
                    accumulate_surface_contact_3d(vel, -hw_core - uy, n_e, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                }
                if (-hw_core - ux > 0.0) {
                    const double n_o[3] = {-1.0, 0.0, 0.0};
                    accumulate_surface_contact_3d(vel, -hw_core - ux, n_o, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                }
                if (uy > cy_f) {
                    if (ux - hw_core > 0.0) {
                        const double n_i[3] = {1.0, 0.0, 0.0};
                        accumulate_surface_contact_3d(vel, ux - hw_core, n_i, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                    }
                } else if (ux > cx_f) {
                    if (uy - hw_core > 0.0) {
                        const double n_i[3] = {0.0, sgn, 0.0};
                        accumulate_surface_contact_3d(vel, uy - hw_core, n_i, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                    }
                } else if (ux > r_in - node_radius - r_fil && uy > r_in - node_radius - r_fil) {
                    const double dfu = ux - cx_f, dfv = uy - cy_f;
                    const double d_f = sqrt(dfu * dfu + dfv * dfv + 1e-24);
                    if (R_cf - d_f > 0.0) {
                        const double inv_df = 1.0 / fmax(d_f, 1e-12);
                        const double n_c[3] = {-dfu * inv_df, -sgn * dfv * inv_df, 0.0};
                        accumulate_surface_contact_3d(vel, R_cf - d_f, n_c, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
                    }
                }
            }
        }
        if (in_z) {
            const double pen_top = pz - (z_c - node_radius);
            if (pen_top > 0.0) {
                const double n_top[3] = {0.0, 0.0, -1.0};
                accumulate_surface_contact_3d(vel, pen_top, n_top, k_w, c_wall, 0.18, h, implicit_c, f_acc, k_tri6, fn_acc);
            }
        }
    }
}

__device__ __forceinline__ void eval_strip_node_contact_cuda(
    const double pos[3],
    const double vel[3],
    const double gg[5],
    const double pc[12],
    int ground_contact,
    int strip_node_contact,
    double h,
    double r_cur[6],
    double D_cur_tri[21],
    double& sw_sum,
    double& min_z
) {
    const bool has_pipe = (pc[0] > 0.5) && (pc[10] > 0.0);
    if (!strip_node_contact && !has_pipe) return;

    double f_c[3] = {0.0, 0.0, 0.0};
    double k_c[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double fn_g = 0.0;

    // 1. Strip-to-ground contact protection using exact ground_geom surface
    const double wx = pos[0], wy = pos[1], wz = pos[2];
    double z_g = gg[0] * wx + gg[1] * wy;
    double dz_dx = gg[0], dz_dy = gg[1];
    if (gg[2] != 0.0) {
        const double sx = sin(gg[3] * wx), cx = cos(gg[3] * wx);
        const double sy = sin(gg[4] * wy), cy = cos(gg[4] * wy);
        z_g += gg[2] * sx * cy;
        dz_dx += gg[2] * gg[3] * cx * cy;
        dz_dy -= gg[2] * gg[4] * sx * sy;
    }
    double up_raw[3] = {-dz_dx, -dz_dy, 1.0};
    double up_g[3]; unit3(up_raw, up_g);
    const double clr_g = (wz - z_g) * up_g[2];
    if (clr_g < min_z) min_z = clr_g;
    if (ground_contact) {
        const double pen_g = 0.002 - clr_g;
        if (pen_g > 0.0) {
            accumulate_surface_contact_3d(vel, pen_g, up_g, 2500.0, 8.0, 0.35, h, 1, f_c, k_c, fn_g);
        }
    }

    // 2. Strip-to-pipe-wall contact (straight / 90-deg arc / S-curve / sharp 90-deg L-duct)
    if (has_pipe) {
        double fn_p = 0.0;
        eval_pipe_contact_cuda(pos, vel, pc, 0.002, h, 1, 0.20, 4.0, f_c, k_c, fn_p);
        sw_sum += fn_p;
    }

    r_cur[0] += h * f_c[0];
    r_cur[1] += h * f_c[1];
    r_cur[2] += h * f_c[2];
    #pragma unroll
    for (int i = 0; i < 6; ++i) D_cur_tri[i] += k_c[i];
}

// ---------------------------------------------------------------------------
// Kernel 1: Fine-grained external plate loads (P = 2*count threads per env)
// ---------------------------------------------------------------------------

constexpr int ENVS_PER_EXT_BLK = 8;
constexpr int MAX_P = 20;  // supports up to count=10 segments (28 KB static smem <= 48 KB)

__global__ void external_kernel_v3(
    int B, int count, int D, double h, int ground_contact, double stop_gap,
    int crank_mode, double horn_r, double guide_d, double slack_m, double k_ten, double c_ten,
    const double* __restrict__ x,          // (B, D, 7)
    const double* __restrict__ v,          // (B, D, 6)
    double* __restrict__ servo,            // (B, P)
    const double* __restrict__ target,     // (B, P)
    const double* __restrict__ yaw_tgt,    // (B, count-1)
    double* __restrict__ spin,             // (B, P, 2)
    double* __restrict__ angle,            // (B, P, 2)
    double* __restrict__ memory,           // (B, P, 2, 3)
    double* __restrict__ tension,          // (B, count, 4)
    double* __restrict__ support,          // (B, P, 2)
    double* __restrict__ last_slip,        // (B, P, 2, 2)
    double* __restrict__ last_yaw_torque,  // (B, count-1)
    double* __restrict__ last_yaw_cap,     // (B, count-1)
    double* __restrict__ plate_ext,        // (B, P, 6)
    const double* __restrict__ wheel_mu,
    const double* __restrict__ yaw_kp,
    const double* __restrict__ yaw_kd,
    const double* __restrict__ yaw_torque_limit,
    const double* __restrict__ yaw_speed_limit,
    const double* __restrict__ servo_torque_limit,
    const double* __restrict__ servo_speed_limit,
    const double* __restrict__ mounts,     // (P, 2, 3)
    const double* __restrict__ anchors,    // (4, 3)
    const double* __restrict__ guides,     // (4, 3)
    const double* __restrict__ conn_arm,   // (3,)
    int contact_model_id,
    int implicit_contact,
    double ground_kn,
    double ground_cn,
    double ground_hc_k,
    double ground_hc_alpha,
    double ground_soil_exp,
    double ground_soil_unload_ratio,
    double ground_rut_relax_m,
    double ground_rut_lat_k,
    double wheel_mu_lat_ratio,
    double wheel_rolling,
    double wheel_patch_radius,
    double wheel_pivot_mu,
    double ratchet_backlash,
    double* __restrict__ ratchet_play,     // (B, P, 2)
    double* __restrict__ soil_sinkage,     // (B, P, 2)
    double* __restrict__ plate_K_contact,  // (B, P, 21)
    const double* __restrict__ env_terrain,// (B, 5): [roll, hc_k, unload_r, rut_lat_k, mu_lat_r]
    const double* __restrict__ wall_cfg,   // (B, 5): [y_center, y_allow, x_entry, x_exit, k_wall]
    double* __restrict__ last_wall_force,  // (B, P)
    const double* __restrict__ env_dynamics,// (B, 7): [backlash, static_mu_r, patch_r, capstan_mt, slack, yaw_sf, sv_db]
    const double* __restrict__ ground_geom, // (B, 5): [slope_x, slope_y, wave_amp, wave_kx, wave_ky]
    const double* __restrict__ pipe_cfg,    // (B, 12): generalized analytical pipe/bend/S/L-duct
    double wheel_mass_kg
) {
    const int P = 2 * count;
    const int local_env = threadIdx.y;
    const int p = threadIdx.x;
    const int b = blockIdx.x * ENVS_PER_EXT_BLK + local_env;

    __shared__ double sh_Rp[ENVS_PER_EXT_BLK][MAX_P][9];
    __shared__ double sh_takeup[ENVS_PER_EXT_BLK][MAX_P];
    __shared__ double sh_sv_vel[ENVS_PER_EXT_BLK][MAX_P];
    __shared__ double sh_f_seg[ENVS_PER_EXT_BLK][MAX_P][6];
    __shared__ double sh_f_conn[ENVS_PER_EXT_BLK][MAX_P][6];

    const bool active_thread = (b < B) && (p < P);

    const double* xb = active_thread ? (x + (int64_t)b * D * 7) : nullptr;
    const double* vb = active_thread ? (v + (int64_t)b * D * 6) : nullptr;
    const double* dyn = active_thread ? (env_dynamics + (int64_t)b * 7) : nullptr;

    if (active_thread) {
        qmat(xb + p * 7 + 3, sh_Rp[local_env][p]);
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            sh_f_seg[local_env][p][c] = 0.0;
            sh_f_conn[local_env][p][c] = 0.0;
        }

        // 1. Each plate thread updates its own tendon servo & takeup in parallel (with deadband + Capstan pull load)
        const double radius = horn_r, guide = guide_d;
        const double takeup_offset = guide - radius;
        const double s_tlim = servo_torque_limit[b];
        const double s_slim = servo_speed_limit[b];
        double cap_mt = dyn[3];
        if (cap_mt < 0.0) cap_mt = 0.0;
        if (cap_mt > 0.6) cap_mt = 0.6;
        double sv_db = dyn[6];
        if (sv_db < 0.0) sv_db = 0.0;

        double sv = servo[b * P + p];
        double chord = sqrt(guide * guide + radius * radius - 2.0 * guide * radius * cos(sv));
        if (chord < 1e-9) chord = 1e-9;
        double lever;
        if (crank_mode) {
            lever = radius * guide * fabs(sin(sv)) / chord;
            if (lever > radius) lever = radius;
        } else {
            const double alpha = acos(radius / guide);
            lever = (sv <= alpha) ? (radius * guide * sin(sv) / chord) : radius;
        }
        int seg = p >> 1;
        int pair = p & 1;
        double tsum = tension[(b * count + seg) * 4 + 2 * pair] +
                      tension[(b * count + seg) * 4 + 2 * pair + 1];
        double load = tsum * lever * exp(cap_mt);
        double raw_err = target[b * P + p] - sv;
        double sgn_e = (raw_err > 0.0) ? 1.0 : ((raw_err < 0.0) ? -1.0 : 0.0);
        double eff_err = sgn_e * fmax(fabs(raw_err) - sv_db, 0.0);
        double inc = eff_err;
        double max_inc = s_slim * h;
        if (inc > max_inc) inc = max_inc;
        if (inc < -max_inc) inc = -max_inc;
        if (inc > 0.0) {
            double scale = 1.0 - load / s_tlim;
            if (scale < 0.0) scale = 0.0;
            if (scale > 1.0) scale = 1.0;
            inc *= scale;
        }
        sh_sv_vel[local_env][p] = inc / h;
        sv += inc;
        servo[b * P + p] = sv;
        chord = sqrt(guide * guide + radius * radius - 2.0 * guide * radius * cos(sv));
        if (crank_mode) {
            sh_takeup[local_env][p] = chord - takeup_offset;
        } else {
            const double alpha = acos(radius / guide);
            const double tang_base = sqrt(guide * guide - radius * radius);
            sh_takeup[local_env][p] = ((sv <= alpha) ? chord : (tang_base + radius * (sv - alpha))) - takeup_offset;
        }
    }
    __syncthreads();

    // 2. Even threads compute segment tendons + stop; odd threads compute yaw connectors in parallel
    if (active_thread) {
        if ((p & 1) == 0) {
            int seg = p >> 1;
            int pa = 2 * seg;
            int pb = 2 * seg + 1;
            const double* xa = xb + pa * 7;
            const double* xbp = xb + pb * 7;
            const double* va = vb + pa * 6;
            const double* vbp = vb + pb * 6;
            double fa[6] = {0.0}, fbp[6] = {0.0};

            for (int k = 0; k < 4; ++k) {
                double ar[3], br[3];
                qrot(xa + 3, anchors + k * 3, ar);
                qrot(xbp + 3, guides + k * 3, br);
                double delta[3] = {
                    xbp[0] + br[0] - xa[0] - ar[0],
                    xbp[1] + br[1] - xa[1] - ar[1],
                    xbp[2] + br[2] - xa[2] - ar[2]
                };
                double len = norm3(delta);
                double inv_l = 1.0 / (len > 1e-15 ? len : 1e-15);
                double dir[3] = {delta[0] * inv_l, delta[1] * inv_l, delta[2] * inv_l};

                double cra[3], crb[3];
                cross3(va + 3, ar, cra);
                cross3(vbp + 3, br, crb);
                double dv[3] = {
                    (vbp[0] + crb[0]) - (va[0] + cra[0]),
                    (vbp[1] + crb[1]) - (va[1] + cra[1]),
                    (vbp[2] + crb[2]) - (va[2] + cra[2])
                };
                double env_slack = dyn[4];
                double cap_mt_t = dyn[3];
                if (cap_mt_t < 0.0) cap_mt_t = 0.0;
                if (cap_mt_t > 0.6) cap_mt_t = 0.6;
                double rest = (0.101 + env_slack) - sh_takeup[local_env][2 * seg + (k >> 1)];
                double ext = len - rest;
                double spd = dot3(dv, dir);
                double c_eff = c_ten + 12.0 * cap_mt_t;
                double ten = 0.0;
                if (ext >= 0.0) {
                    ten = k_ten * ext + c_eff * spd;
                    if (ten < 0.0) ten = 0.0;
                    if (cap_mt_t > 0.0) {
                        double sv_v = sh_sv_vel[local_env][2 * seg + (k >> 1)];
                        double pull_frac = (sv_v > 0.0) ? tanh(sv_v / 0.35) : 0.0;
                        ten *= exp(-cap_mt_t * pull_frac);
                    }
                }
                tension[(b * count + seg) * 4 + k] = ten;
                double force[3] = {ten * dir[0], ten * dir[1], ten * dir[2]};
                double ta[3], tb[3];
                cross3(ar, force, ta);
                cross3(br, force, tb);
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    fa[c] += force[c];  fbp[c] -= force[c];
                    fa[3 + c] += ta[c]; fbp[3 + c] -= tb[c];
                }
            }

            double sep[3] = {xbp[0] - xa[0], xbp[1] - xa[1], xbp[2] - xa[2]};
            const double* Rpa = sh_Rp[local_env][pa];
            double axis[3] = {-Rpa[0], -Rpa[3], -Rpa[6]};
            double gap = dot3(sep, axis);
            if (gap < stop_gap) {
                double mag = 50000.0 * (stop_gap - gap);
                double push[3] = {mag * axis[0], mag * axis[1], mag * axis[2]};
                double tp[3]; cross3(sep, push, tp);
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    fbp[c] += push[c];
                    fa[c] -= push[c];
                    fa[3 + c] -= tp[c];
                }
            }
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                sh_f_seg[local_env][pa][c] = fa[c];
                sh_f_seg[local_env][pb][c] = fbp[c];
            }
        } else {
            int j = p >> 1;
            if (j < count - 1) {
                int pa = 2 * j + 1;
                int pb = 2 * j + 2;
                const double* xa = xb + pa * 7;
                const double* xbp = xb + pb * 7;
                const double* va = vb + pa * 6;
                const double* vbp = vb + pb * 6;
                const double* Rpa = sh_Rp[local_env][pa];
                const double* Rpb = sh_Rp[local_env][pb];
                double fa[6] = {0.0}, fbp[6] = {0.0};

                double arm[3]; qrot(xa + 3, conn_arm, arm);
                double sep[3] = {xbp[0] - xa[0], xbp[1] - xa[1], xbp[2] - xa[2]};
                double err[3] = {sep[0] - arm[0], sep[1] - arm[1], sep[2] - arm[2]};
                double cr[3]; cross3(va + 3, sep, cr);
                double relv[3] = {vbp[0] - va[0] - cr[0], vbp[1] - va[1] - cr[1], vbp[2] - va[2] - cr[2]};
                double force[3] = {
                    -10000.0 * err[0] - 10.0 * relv[0],
                    -10000.0 * err[1] - 10.0 * relv[1],
                    -10000.0 * err[2] - 10.0 * relv[2]
                };
                double tf[3]; cross3(sep, force, tf);
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    fbp[c] += force[c]; fa[c] -= force[c]; fa[3 + c] -= tf[c];
                }

                double r00 = Rpa[0] * Rpb[0] + Rpa[3] * Rpb[3] + Rpa[6] * Rpb[6];
                double r10 = Rpa[1] * Rpb[0] + Rpa[4] * Rpb[3] + Rpa[7] * Rpb[6];
                double yaw_cur = atan2(r10, r00);
                double axis[3] = {Rpa[2], Rpa[5], Rpa[8]};
                double axis_b[3] = {Rpb[2], Rpb[5], Rpb[8]};
                double omega[3] = {vbp[3] - va[3], vbp[4] - va[4], vbp[5] - va[5]};
                double rate = dot3(omega, axis);

                const double kp = yaw_kp[b];
                const double kd = yaw_kd[b];
                const double tlim = yaw_torque_limit[b];
                const double slim = yaw_speed_limit[b];

                double yaw_sf = dyn[5];
                if (yaw_sf < 0.0) yaw_sf = 0.0;
                double raw_yt = kp * (yaw_tgt[b * (count - 1) + j] - yaw_cur) - kd * rate;
                double sgn_yt = (raw_yt > 0.0) ? 1.0 : ((raw_yt < 0.0) ? -1.0 : 0.0);
                double yt = sgn_yt * fmax(fabs(raw_yt) - yaw_sf, 0.0);
                if (yt > tlim) yt = tlim;
                if (yt < -tlim) yt = -tlim;
                double cap = tlim;
                if (slim > 0.0) {
                    if (yt * rate > 0.0) {
                        double s = 1.0 - fabs(rate) / slim;
                        if (s < 0.0) s = 0.0;
                        if (s > 1.0) s = 1.0;
                        cap = tlim * s;
                    }
                    if (yt > cap) yt = cap;
                    if (yt < -cap) yt = -cap;
                }
                last_yaw_torque[b * (count - 1) + j] = yt;
                last_yaw_cap[b * (count - 1) + j] = cap;

                double align_c[3]; cross3(axis_b, axis, align_c);
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    double tq = yt * axis[c] + 20.0 * align_c[c] - 0.1 * (omega[c] - axis[c] * rate);
                    fbp[3 + c] += tq;
                    fa[3 + c] -= tq;
                }
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    sh_f_conn[local_env][pa][c] = fa[c];
                    sh_f_conn[local_env][pb][c] = fbp[c];
                }
            }
        }
    }
    __syncthreads();

    // 3. Each plate thread computes its 2 passive wheels in parallel & writes plate_ext[b, p]
    if (active_thread) {
        const double wrad = 0.018;
        const double winertia = 0.5 * wheel_mass_kg * wrad * wrad;
        const double wstiff = 2000.0, wdamp = 8.0, wvisc = 2e-6;
        const double wgain = wstiff * h + wdamp;
        const double coast_decay = fmax(0.0, 1.0 - h * wvisc / winertia);
        const double mu = wheel_mu[b];
        const double* terr = env_terrain + (int64_t)b * 5;
        const double env_roll = terr[0];
        const double env_hc_k = terr[1];
        const double env_unload_r = terr[2];
        const double env_rut_lat_k = terr[3];
        const double env_mu_lat_r = terr[4];
        const double env_backlash = dyn[0];
        const double env_static_mu_r = dyn[1];
        const double env_patch_r = dyn[2];
        const double* gg = ground_geom + (int64_t)b * 5;
        const double g_sx = gg[0], g_sy = gg[1], g_amp = gg[2], g_kx = gg[3], g_ky = gg[4];

        const double* xp = xb + p * 7;
        const double* vp = vb + p * 6;
        const double* Rpp = sh_Rp[local_env][p];

        double fp[6];
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            fp[c] = sh_f_seg[local_env][p][c] + sh_f_conn[local_env][p][c];
        }

        double axle[3] = {Rpp[1], Rpp[4], Rpp[7]};
        double rim_raw[3] = {-axle[0] * axle[2], -axle[1] * axle[2], 1.0 - axle[2] * axle[2]};
        double rim_u[3]; unit3(rim_raw, rim_u);
        double rim[3] = {-wrad * rim_u[0], -wrad * rim_u[1], -wrad * rim_u[2]};

        if (contact_model_id == 0 && !implicit_contact) {
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                int64_t wid = ((int64_t)b * P + p) * 2 + s;
                double offset[3]; qrot(xp + 3, mounts + (p * 2 + s) * 3, offset);
                double arm[3] = {offset[0] + rim[0], offset[1] + rim[1], offset[2] + rim[2]};
                double cr[3]; cross3(vp + 3, arm, cr);
                double vel[3] = {vp[0] + cr[0], vp[1] + cr[1], vp[2] + cr[2]};

                double wx = xp[0] + arm[0], wy = xp[1] + arm[1], wz = xp[2] + arm[2];
                double z_g = g_sx * wx + g_sy * wy;
                double dz_dx = g_sx, dz_dy = g_sy;
                if (g_amp != 0.0) {
                    double s_x = sin(g_kx * wx), c_x = cos(g_kx * wx);
                    double s_y = sin(g_ky * wy), c_y = cos(g_ky * wy);
                    z_g += g_amp * s_x * c_y;
                    dz_dx += g_amp * g_kx * c_x * c_y;
                    dz_dy -= g_amp * g_ky * s_x * s_y;
                }
                double up_raw[3] = {-dz_dx, -dz_dy, 1.0};
                double up_g[3]; unit3(up_raw, up_g);
                double fwd_raw[3]; cross3(up_g, axle, fwd_raw);
                double fwd[3]; unit3(fwd_raw, fwd);
                double lat[3]; cross3(up_g, fwd, lat);

                double pen = (z_g - wz) * up_g[2];
                double vn = dot3(up_g, vel);
                bool active = (pen > 0.0) && (ground_contact != 0);
                double normal = active ? fmax(ground_kn * pen - ground_cn * vn, 0.0) : 0.0;

                double tv0 = dot3(fwd, vel);
                double tv1 = dot3(lat, vel);
                double* mem_w = memory + wid * 3;
                double m0 = dot3(fwd, mem_w);
                double m1 = dot3(lat, mem_w);
                double sp = spin[wid];

                double b0 = -wstiff * m0 - wgain * tv0;
                double b1 = -wstiff * m1 - wgain * tv1;
                double limit = env_roll * normal * wrad;
                double om = (winertia * sp / h - wrad * b0 - limit) / (winertia / h + wrad * wrad * wgain + wvisc);
                if (om < 0.0) om = 0.0;
                double f0 = b0 + wgain * wrad * om;
                double f1 = b1;
                double mag = hypot(f0, f1);
                double cap = mu * normal;
                bool sliding = mag > cap;
                double scale = sliding ? (cap / (mag > 1e-15 ? mag : 1e-15)) : 1.0;
                f0 *= scale; f1 *= scale;

                om = (winertia * sp / h - wrad * f0 - limit) / (winertia / h + wvisc);
                if (om < 0.0) om = 0.0;
                double reaction = winertia * (om - sp) / h + wrad * f0;
                double sl0 = tv0 - wrad * om;
                double sl1 = tv1;

                double nm0 = sliding ? -(f0 + wdamp * sl0) / wstiff : (m0 + h * sl0);
                double nm1 = sliding ? -(f1 + wdamp * sl1) / wstiff : (m1 + h * sl1);

                last_slip[wid * 2 + 0] = sl0;
                last_slip[wid * 2 + 1] = sl1;
                if (active) {
                    mem_w[0] = nm0 * fwd[0] + nm1 * lat[0];
                    mem_w[1] = nm0 * fwd[1] + nm1 * lat[1];
                    mem_w[2] = nm0 * fwd[2] + nm1 * lat[2];
                    spin[wid] = om;
                } else {
                    mem_w[0] = mem_w[1] = mem_w[2] = 0.0;
                    spin[wid] = sp * coast_decay;
                }
                angle[wid] += h * spin[wid];
                support[wid] = normal;

                if (active) {
                    double Fw[3] = {
                        f0 * fwd[0] + f1 * lat[0] + normal * up_g[0],
                        f0 * fwd[1] + f1 * lat[1] + normal * up_g[1],
                        f0 * fwd[2] + f1 * lat[2] + normal * up_g[2]
                    };
                    double rt[3], toff[3];
                    cross3(rim, Fw, rt);
                    cross3(offset, Fw, toff);
                    double rt_ax = dot3(rt, axle);
                    #pragma unroll
                    for (int c = 0; c < 3; ++c) {
                        fp[c] += Fw[c];
                        fp[3 + c] += toff[c] + rt[c] - axle[c] * rt_ax + reaction * axle[c];
                    }
                }
            }

            // Per-env generalized analytical pipe or legacy lateral wall contact (explicit path)
            {
                const double* pc = pipe_cfg + (int64_t)b * 12;
                double fn_w = 0.0;
                if (pc[0] > 0.5) {
                    double f_gp[3] = {0.0, 0.0, 0.0};
                    double k_unused[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    eval_pipe_contact_cuda(xp, vp, pc, 0.074, h, 0, 1.0, 18.0, f_gp, k_unused, fn_w);
                    fp[0] += f_gp[0]; fp[1] += f_gp[1]; fp[2] += f_gp[2];
                } else {
                    const double* wc = wall_cfg + (int64_t)b * 5;
                    const double y_c = wc[0], y_al = wc[1], x_en = wc[2], x_ex = wc[3], k_w = wc[4];
                    if (k_w > 0.0 && y_al < 10.0) {
                        const double px_w = xp[0];
                        const bool in_fun = (px_w <= x_en + 0.08) && (px_w > x_en);
                        const bool in_cor = (px_w <= x_en) && (px_w >= x_ex);
                        if (in_fun || in_cor) {
                            double u_fun = (x_en + 0.08 - px_w) / 0.08;
                            if (u_fun < 0.0) u_fun = 0.0;
                            if (u_fun > 1.0) u_fun = 1.0;
                            const double y_lim = in_fun ? (y_al + 0.055 * (1.0 - u_fun)) : y_al;
                            const double dy_w = xp[1] - y_c;
                            const double pen_w = fabs(dy_w) - y_lim;
                            if (pen_w > 0.0) {
                                const double sgn_w = (dy_w >= 0.0) ? 1.0 : -1.0;
                                fn_w = fmax(k_w * pen_w + 18.0 * (vp[1] * sgn_w), 0.0);
                                if (fn_w > 0.0) {
                                    fp[1] += -sgn_w * fn_w;
                                    fp[0] += -0.35 * fn_w * tanh(vp[0] / 0.015);
                                }
                            }
                        }
                    }
                }
                last_wall_force[(int64_t)b * P + p] = fn_w;
            }
        } else {
            const double inv_dt_I = winertia / h;
            const double denom_coup = inv_dt_I + wrad * wrad * wgain + wvisc;
            const double denom_sat = inv_dt_I + wvisc;
            double kc_tri[21];
            if (implicit_contact) {
                #pragma unroll
                for (int k = 0; k < 21; ++k) kc_tri[k] = 0.0;
            }

            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                int64_t wid = ((int64_t)b * P + p) * 2 + s;
                double offset[3]; qrot(xp + 3, mounts + (p * 2 + s) * 3, offset);
                double arm[3] = {offset[0] + rim[0], offset[1] + rim[1], offset[2] + rim[2]};
                double cr[3]; cross3(vp + 3, arm, cr);
                double vel[3] = {vp[0] + cr[0], vp[1] + cr[1], vp[2] + cr[2]};

                double wx = xp[0] + arm[0], wy = xp[1] + arm[1], wz = xp[2] + arm[2];
                double z_g = g_sx * wx + g_sy * wy;
                double dz_dx = g_sx, dz_dy = g_sy;
                if (g_amp != 0.0) {
                    double s_x = sin(g_kx * wx), c_x = cos(g_kx * wx);
                    double s_y = sin(g_ky * wy), c_y = cos(g_ky * wy);
                    z_g += g_amp * s_x * c_y;
                    dz_dx += g_amp * g_kx * c_x * c_y;
                    dz_dy -= g_amp * g_ky * s_x * s_y;
                }
                double up_raw[3] = {-dz_dx, -dz_dy, 1.0};
                double up_g[3]; unit3(up_raw, up_g);
                double fwd_raw[3]; cross3(up_g, axle, fwd_raw);
                double fwd[3]; unit3(fwd_raw, fwd);
                double lat[3]; cross3(up_g, fwd, lat);

                double pen = (z_g - wz) * up_g[2];
                double vn = dot3(up_g, vel);
                double tv0 = dot3(fwd, vel);
                double tv1 = dot3(lat, vel);

                double normal = 0.0, new_rut = 0.0, kn_tan = 0.0, cn_tan = 0.0;
                bool active = false;
                if (contact_model_id == 0) {
                    double raw_n = ground_kn * pen - ground_cn * vn;
                    active = (pen > 0.0) && (raw_n > 0.0) && (ground_contact != 0);
                    if (active) {
                        normal = raw_n;
                        kn_tan = ground_kn;
                        cn_tan = ground_cn;
                    }
                } else if (ground_contact != 0) {
                    double v_horiz = hypot(tv0, tv1);
                    double pen_pos = fmax(pen, 0.0);
                    double rut_rel = 0.0;
                    if (env_unload_r > 1.0) {
                        double inv_relax = 1.0 / fmax(ground_rut_relax_m, 1e-6);
                        rut_rel = soil_sinkage[wid] * exp(-v_horiz * h * inv_relax);
                        double plastic_frac = 1.0 - 1.0 / env_unload_r;
                        new_rut = fmax(rut_rel, plastic_frac * pen_pos);
                    }
                    bool in_contact = (pen > rut_rel) && (pen > 0.0);
                    double f_el = 0.0, df_el = 0.0;
                    if (in_contact) {
                        double pen_pm1 = pow(pen, ground_soil_exp - 1.0);
                        double pen_pm2 = pen_pm1 / pen;
                        double virgin_el = env_hc_k * pen * pen_pm1;
                        if (env_unload_r > 1.0) {
                            double dpen = fmax(pen - rut_rel, 0.0);
                            double unload_el = env_unload_r * env_hc_k * dpen * pen_pm1;
                            bool is_unload = unload_el < virgin_el;
                            f_el = is_unload ? unload_el : virgin_el;
                            df_el = is_unload
                                ? (env_unload_r * env_hc_k * (pen_pm1 + (ground_soil_exp - 1.0) * dpen * pen_pm2))
                                : (ground_soil_exp * env_hc_k * pen_pm1);
                        } else {
                            f_el = virgin_el;
                            df_el = ground_soil_exp * env_hc_k * pen_pm1;
                        }
                    }
                    double damp_raw = 1.0 - ground_hc_alpha * vn;
                    if (in_contact && (damp_raw > 0.0)) {
                        normal = f_el * damp_raw;
                        kn_tan = df_el * damp_raw;
                        cn_tan = ground_hc_alpha * f_el;
                    }
                    active = (normal > 0.0);
                }
                soil_sinkage[wid] = new_rut;

                double* mem_w = memory + wid * 3;
                double m0 = dot3(fwd, mem_w);
                double m1 = dot3(lat, mem_w);
                double sp = spin[wid];
                double play = ratchet_play[wid];

                double b0 = -wstiff * m0 - wgain * tv0;
                double b1 = -wstiff * m1 - wgain * tv1;
                double limit = env_roll * normal * wrad;
                double omega_min = (env_backlash > 0.0) ? (-play / h) : 0.0;

                double n_drive = inv_dt_I * sp - wrad * b0;
                double sgn_nd = (n_drive > 0.0) ? 1.0 : ((n_drive < 0.0) ? -1.0 : 0.0);
                double drive_net = sgn_nd * fmax(fabs(n_drive) - limit, 0.0);
                double omega_unc = drive_net / denom_coup;
                double om = fmax(omega_unc, omega_min);

                double f0 = b0 + wgain * wrad * om;
                double f1 = b1;

                double stribeck = 1.0;
                if (env_static_mu_r > 1.0) {
                    double vs_est = hypot(tv0 - wrad * om, tv1);
                    double ratio = vs_est / 0.03;
                    stribeck = 1.0 + (env_static_mu_r - 1.0) * exp(-(ratio * ratio));
                }

                double cap_fwd = mu * stribeck * normal;
                double cap_lat = (mu * env_mu_lat_r) * stribeck * normal;
                bool has_normal = (cap_fwd > 1e-15) && (cap_lat > 1e-15);
                double u0 = has_normal ? (f0 / cap_fwd) : 0.0;
                double u1 = has_normal ? (f1 / cap_lat) : 0.0;
                double ell = hypot(u0, u1);
                bool sliding = (!has_normal) || (ell > 1.0);
                double scale = has_normal ? fmin(1.0, 1.0 / fmax(ell, 1e-15)) : 0.0;
                f0 *= scale;
                f1 *= scale;

                double n_drive_sat = inv_dt_I * sp - wrad * f0;
                double sgn_nds = (n_drive_sat > 0.0) ? 1.0 : ((n_drive_sat < 0.0) ? -1.0 : 0.0);
                double drive_sat_net = sgn_nds * fmax(fabs(n_drive_sat) - limit, 0.0);
                om = fmax(drive_sat_net / denom_sat, omega_min);

                double reaction = inv_dt_I * (om - sp) + wrad * f0;
                double sl0 = tv0 - wrad * om;
                double sl1 = tv1;

                double nm0 = sliding ? -(f0 + wdamp * sl0) / wstiff : (m0 + h * sl0);
                double nm1 = sliding ? -(f1 + wdamp * sl1) / wstiff : (m1 + h * sl1);

                double new_play = (env_backlash > 0.0) ? fmin(fmax(play + om * h, 0.0), env_backlash) : 0.0;

                double f_rut_lat = 0.0, c_rut_lat = 0.0;
                if (env_rut_lat_k > 0.0) {
                    double th_lat = tanh(tv1 / 0.02);
                    double rut_act = has_normal ? new_rut : 0.0;
                    f_rut_lat = -(env_rut_lat_k * rut_act) * th_lat;
                    c_rut_lat = (env_rut_lat_k * rut_act / 0.02) * fmax(1.0 - th_lat * th_lat, 0.0);
                }
                double f1_tot = f1 + f_rut_lat;

                double tau_z = 0.0, c_piv_tan = 0.0;
                if ((env_patch_r > 0.0) && (wheel_pivot_mu > 0.0)) {
                    double piv_cap = (2.0 / 3.0) * wheel_pivot_mu * env_patch_r * normal;
                    double th_wz = tanh(vp[5] / 0.15);
                    tau_z = -piv_cap * th_wz;
                    c_piv_tan = (piv_cap / 0.15) * fmax(1.0 - th_wz * th_wz, 0.0);
                }

                bool free_rolling = (fabs(n_drive) > limit) && (omega_unc > omega_min);
                double g_fwd = free_rolling ? (wgain * (denom_sat / denom_coup)) : wgain;
                double ct_fwd_tan = g_fwd * scale;
                double ct_lat_tan = wgain * scale + c_rut_lat;

                last_slip[wid * 2 + 0] = sl0;
                last_slip[wid * 2 + 1] = sl1;
                if (active) {
                    mem_w[0] = nm0 * fwd[0] + nm1 * lat[0];
                    mem_w[1] = nm0 * fwd[1] + nm1 * lat[1];
                    mem_w[2] = nm0 * fwd[2] + nm1 * lat[2];
                    spin[wid] = om;
                    ratchet_play[wid] = new_play;
                } else {
                    mem_w[0] = mem_w[1] = mem_w[2] = 0.0;
                    spin[wid] = sp * coast_decay;
                }
                angle[wid] += h * spin[wid];
                support[wid] = normal;

                if (active) {
                    double Fw[3] = {
                        f0 * fwd[0] + f1_tot * lat[0] + normal * up_g[0],
                        f0 * fwd[1] + f1_tot * lat[1] + normal * up_g[1],
                        f0 * fwd[2] + f1_tot * lat[2] + normal * up_g[2]
                    };
                    double rt[3], toff[3];
                    cross3(rim, Fw, rt);
                    cross3(offset, Fw, toff);
                    double rt_ax = dot3(rt, axle);
                    #pragma unroll
                    for (int c = 0; c < 3; ++c) {
                        fp[c] += Fw[c];
                        fp[3 + c] += toff[c] + rt[c] - axle[c] * rt_ax + reaction * axle[c];
                    }
                    fp[5] += tau_z;

                    if (implicit_contact) {
                        double cr_j0[3], cr_j1[3], cr_j2[3];
                        cross3(arm, fwd, cr_j0);
                        cross3(arm, lat, cr_j1);
                        cross3(arm, up_g, cr_j2);
                        double j0[6] = {fwd[0], fwd[1], fwd[2], cr_j0[0], cr_j0[1], cr_j0[2]};
                        double j1[6] = {lat[0], lat[1], lat[2], cr_j1[0], cr_j1[1], cr_j1[2]};
                        double j2[6] = {up_g[0], up_g[1], up_g[2], cr_j2[0], cr_j2[1], cr_j2[2]};

                        double s0 = ct_fwd_tan * tv0;
                        double s1 = ct_lat_tan * tv1;
                        double s2 = cn_tan * vn;
                        #pragma unroll
                        for (int r = 0; r < 6; ++r) {
                            fp[r] += s0 * j0[r] + s1 * j1[r] + s2 * j2[r];
                        }
                        fp[5] += c_piv_tan * vp[5];

                        double w0 = h * ct_fwd_tan;
                        double w1 = h * ct_lat_tan;
                        double w2 = h * cn_tan + h * h * kn_tan;
                        double wp = h * c_piv_tan;
                        int k = 0;
                        #pragma unroll
                        for (int r = 0; r < 6; ++r) {
                            #pragma unroll
                            for (int c = 0; c <= r; ++c) {
                                kc_tri[k++] += w0 * j0[r] * j0[c] + w1 * j1[r] * j1[c] + w2 * j2[r] * j2[c];
                            }
                        }
                        kc_tri[20] += wp;
                    }
                }
            }

            // Per-env generalized analytical pipe or legacy lateral wall contact (exact parity with gpu_core.py)
            {
                const double* pc = pipe_cfg + (int64_t)b * 12;
                double fn_w = 0.0;
                if (pc[0] > 0.5) {
                    double f_gp[3] = {0.0, 0.0, 0.0};
                    double k_gp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    eval_pipe_contact_cuda(xp, vp, pc, 0.074, h, implicit_contact, 1.0, 18.0, f_gp, k_gp, fn_w);
                    fp[0] += f_gp[0]; fp[1] += f_gp[1]; fp[2] += f_gp[2];
                    if (implicit_contact) {
                        #pragma unroll
                        for (int k = 0; k < 6; ++k) kc_tri[k] += k_gp[k];
                    }
                } else {
                    const double* wc = wall_cfg + (int64_t)b * 5;
                    const double y_c = wc[0], y_al = wc[1], x_en = wc[2], x_ex = wc[3], k_w = wc[4];
                    if (k_w > 0.0 && y_al < 10.0) {
                        const double px_w = xp[0];
                        const bool in_fun = (px_w <= x_en + 0.08) && (px_w > x_en);
                        const bool in_cor = (px_w <= x_en) && (px_w >= x_ex);
                        if (in_fun || in_cor) {
                            double u_fun = (x_en + 0.08 - px_w) / 0.08;
                            if (u_fun < 0.0) u_fun = 0.0;
                            if (u_fun > 1.0) u_fun = 1.0;
                            const double y_lim = in_fun ? (y_al + 0.055 * (1.0 - u_fun)) : y_al;
                            const double dy_w = xp[1] - y_c;
                            const double pen_w = fabs(dy_w) - y_lim;
                            if (pen_w > 0.0) {
                                const double sgn_w = (dy_w >= 0.0) ? 1.0 : -1.0;
                                fn_w = fmax(k_w * pen_w + 18.0 * (vp[1] * sgn_w), 0.0);
                                if (fn_w > 0.0) {
                                    const double th_vx = tanh(vp[0] / 0.015);
                                    fp[1] += -sgn_w * fn_w;
                                    fp[0] += -0.35 * fn_w * th_vx;
                                    if (implicit_contact) {
                                        fp[1] += 18.0 * vp[1];
                                        const double ct_x = (0.35 * fn_w / 0.015) * fmax(1.0 - th_vx * th_vx, 0.0);
                                        fp[0] += ct_x * vp[0];
                                        kc_tri[2] += h * 18.0 + (h * h) * k_w;
                                        kc_tri[0] += h * ct_x;
                                    }
                                }
                            }
                        }
                    }
                }
                last_wall_force[(int64_t)b * P + p] = fn_w;
            }

            if (implicit_contact) {
                double* out_kc = plate_K_contact + ((int64_t)b * P + p) * 21;
                #pragma unroll
                for (int k = 0; k < 21; ++k) out_kc[k] = kc_tri[k];
            }
        }

        double* out_ext = plate_ext + ((int64_t)b * P + p) * 6;
        #pragma unroll
        for (int c = 0; c < 6; ++c) out_ext[c] = fp[c];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 (V3): Zero-barrier Sub-Warp Shuffle Schur + Packed Cholesky + SE(3)
// Template E_FIXED > 0 unrolls compile-time E=8 or E=16; E_FIXED == 0 handles arbitrary E.
// ---------------------------------------------------------------------------

template <int E_FIXED>
__global__ __launch_bounds__(SEG_THREADS, 4)
void segment_solve_kernel_v3(
    int B, int count, int S, int E_rt, int D, double h,
    int implicit_contact,
    double* __restrict__ x,                      // (B, D, 7)
    double* __restrict__ v,                      // (B, D, 6)
    const double* __restrict__ plate_ext,        // (B, P, 6)
    const double* __restrict__ plate_K_contact,  // (B, P, 21)
    const double* __restrict__ mass_scale,       // (B,)
    const int32_t* __restrict__ strip_nodes,     // (S, E+1)
    const double* __restrict__ lengths,          // (S, E)
    const double* __restrict__ transforms,       // (S, E, 2, 7)
    const double* __restrict__ rest_local,       // (S, E, 12)
    const double* __restrict__ K_pack,           // (S, E, 16)
    const double* __restrict__ M_pack,           // (S, E, 16)
    const double* __restrict__ A_pack,           // (S, E, 16)
    const double* __restrict__ plate_inertia,    // (3,)
    double* __restrict__ strip_buf,              // (E-2, 78, B*S) coalesced!
    bool* __restrict__ failed,                   // (1,)
    double rayleigh_alpha,
    const double* __restrict__ plate_masses,     // (B, P)
    const double* __restrict__ rayleigh_beta,    // (B,)
    int ground_contact,
    int strip_node_contact,
    const double* __restrict__ ground_geom,      // (B, 5)
    const double* __restrict__ pipe_cfg,         // (B, 12)
    double* __restrict__ last_strip_wall_force,  // (B, S)
    double* __restrict__ last_strip_min_z        // (B, S)
) {
    const int E = (E_FIXED > 0) ? E_FIXED : E_rt;
    const int N_int = E - 1;

    // sh_seg[seg_in_blk] is accessed exclusively by lane == 0; sh_end0[k][tid] exclusively by tid.
    // Zero __syncthreads() or __syncwarp() barriers are required anywhere in this kernel.
    __shared__ double sh_seg[SEGS_PER_BLOCK][SHM_SEG_DOUBLES];
    __shared__ double sh_end0[27][SEG_THREADS];

    const int tid = threadIdx.x;
    const int64_t total_strips = (int64_t)B * S;
    const int64_t gtid = (int64_t)blockIdx.x * blockDim.x + tid;
    if (gtid >= total_strips) return;

    // Exact active mask for this 32-thread warp (0xffffffffu on all full warps; safe on tail warps when B*S % 32 != 0)
    const int64_t warp_base = (int64_t)blockIdx.x * blockDim.x + (tid & ~31);
    const int64_t warp_rem = total_strips - warp_base;
    const unsigned warp_mask = (warp_rem >= 32) ? 0xffffffffu : ((1u << (int)warp_rem) - 1u);

    const int lane = tid & 7;         // 0..7 strip lane within 8-lane segment sub-warp
    const int seg_in_blk = tid >> 3;  // 0..(SEGS_PER_BLOCK-1) segment within block
    const int b = (int)(gtid / S);
    const int s = (int)(gtid % S);
    const int seg = s >> 3;
    const int P = 2 * count;
    const int p0 = 2 * seg;
    const int p1 = 2 * seg + 1;

    double* xb = x + (int64_t)b * D * 7;
    double* vb = v + (int64_t)b * D * 6;
    const int32_t* sn = strip_nodes + s * (E + 1);
    const double m_c = 1.0 + rayleigh_alpha * h;
    const double k_c = rayleigh_beta[b] * h + h * h;
    const double* gg_b = ground_geom + (int64_t)b * 5;
    const double* pc_b = pipe_cfg + (int64_t)b * 12;
    double strip_wall_sum = 0.0;
    double strip_min_z = 1e6;

    bool ok = true;

    double D_cur_tri[21];
    double r_cur[6];
    double G0_cur[6][6];

    // -----------------------------------------------------------------------
    // Stage 1 (Prologue): Element e = 0 (front plate p0 -> first interior node)
    // -----------------------------------------------------------------------
    {
        int n0 = sn[0], n1 = sn[1];
        double R[9], off0[3], off1_unused[3], rhs0[12], A0[16];
        compute_element_frame_and_rhs<true, false>(
            xb + n0 * 7, xb + n1 * 7, vb + n0 * 6, vb + n1 * 6,
            transforms + ((s * E + 0) * 2 + 0) * 7,
            nullptr,
            rest_local + (s * E + 0) * 12,
            K_pack + (s * E + 0) * 16,
            M_pack + (s * E + 0) * 16,
            __ldg(lengths + s * E + 0), h, m_c, k_c,
            R, off0, off1_unused, rhs0, A0
        );

        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            sh_end0[21 + r][tid] = rhs0[r];
            r_cur[r] = rhs0[6 + r];
        }

        double Q00[3][3], Q11[3][3], Q01[3][3];
        rot_diag_3x3(R, A0[0], A0[1], A0[2], Q00);
        rot_diag_3x3(R, A0[3], A0[4], A0[5], Q11);
        rot_two_3x3(R, A0[6], A0[7], Q01);

        // Un-sheared A11_0 lower triangle -> D_cur_tri
        {
            int idx = 0;
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c <= r; ++c) D_cur_tri[idx++] = Q00[r][c];
            }
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c < 3; ++c) D_cur_tri[idx++] = -Q01[c][r];
                #pragma unroll
                for (int c = 0; c <= r; ++c) D_cur_tri[idx++] = Q11[r][c];
            }
        }

        // A00_0 with front-plate offset shear -> sh_end0[0..20]
        {
            double A00[6][6];
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    A00[r][c]         = Q00[r][c];
                    A00[r][3 + c]     = Q01[r][c];
                    A00[3 + r][c]     = Q01[c][r];
                    A00[3 + r][3 + c] = Q11[r][c];
                }
            }
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                double rlin[3] = {A00[r][0], A00[r][1], A00[r][2]};
                double cr[3]; cross3(off0, rlin, cr);
                A00[r][3] += cr[0]; A00[r][4] += cr[1]; A00[r][5] += cr[2];
            }
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                double c0[3] = {A00[0][c], A00[1][c], A00[2][c]};
                double cr0[3]; cross3(off0, c0, cr0);
                A00[3][c] += cr0[0]; A00[4][c] += cr0[1]; A00[5][c] += cr0[2];
            }
            int k = 0;
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                #pragma unroll
                for (int c = 0; c <= r; ++c) sh_end0[k++][tid] = A00[r][c];
            }
        }

        // A01_0 with front-plate left-shear -> G0_cur = A01_0^T
        {
            double A01[6][6];
            build_A01_unsheared(R, A0, A01);
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                double c1[3] = {A01[0][c], A01[1][c], A01[2][c]};
                double cr1[3]; cross3(off0, c1, cr1);
                A01[3][c] += cr1[0]; A01[4][c] += cr1[1]; A01[5][c] += cr1[2];
            }
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                #pragma unroll
                for (int c = 0; c < 6; ++c) G0_cur[r][c] = A01[c][r];
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stage 2 (Interior Sweep): i = 0 .. N_int - 2 (elements e_next = 1 .. E - 2)
    // Pure interior elements: both has_off0 and has_off1 are compile-time false!
    // -----------------------------------------------------------------------
    #pragma unroll 1
    for (int i = 0; i < N_int - 1; ++i) {
        const int e_next = i + 1;
        const int n0 = sn[e_next], n1 = sn[e_next + 1];

        double R[9], off0_unused[3], off1_unused[3], rhs_n[12], rhs_tail[6], Ae[16];
        compute_element_frame_and_rhs<false, false>(
            xb + n0 * 7, xb + n1 * 7, vb + n0 * 6, vb + n1 * 6,
            nullptr, nullptr,
            rest_local + (s * E + e_next) * 12,
            K_pack + (s * E + e_next) * 16,
            M_pack + (s * E + e_next) * 16,
            __ldg(lengths + s * E + e_next), h, m_c, k_c,
            R, off0_unused, off1_unused, rhs_n, Ae
        );

        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            r_cur[r] += rhs_n[r];
            rhs_tail[r] = rhs_n[6 + r];
        }
        add_A00_unsheared_tri(R, Ae, D_cur_tri);
        eval_strip_node_contact_cuda(
            xb + n0 * 7, vb + n0 * 6, gg_b, pc_b, ground_contact, strip_node_contact, h,
            r_cur, D_cur_tri, strip_wall_sum, strip_min_z
        );

        if (!chol_6x6_packed(D_cur_tri)) ok = false;
        solve_L_6x1_inplace(D_cur_tri, r_cur);
        solve_L_6x6_inplace(D_cur_tri, G0_cur);

        // On-the-fly front-plate Schur update: endA00 -= Y0_i^T @ Y0_i, endr0 -= Y0_i^T @ y_i
        {
            int tri_idx = 0;
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                double sr = 0.0;
                #pragma unroll
                for (int k = 0; k < 6; ++k) sr += G0_cur[k][r] * r_cur[k];
                sh_end0[21 + r][tid] -= sr;
                #pragma unroll
                for (int c = 0; c <= r; ++c) {
                    double sa = 0.0;
                    #pragma unroll
                    for (int k = 0; k < 6; ++k) sa += G0_cur[k][r] * G0_cur[k][c];
                    sh_end0[tri_idx++][tid] -= sa;
                }
            }
        }

        // Compute W_i = L_i^{-1} A01_n in-place and stream [b_tilde, C0_tilde, U_tilde] to strip_buf
        double W_i[6][6];
        build_A01_unsheared(R, Ae, W_i);
        solve_L_6x6_inplace(D_cur_tri, W_i);

        {
            double* base = strip_buf + (int64_t)i * BUF_STRIDE * total_strips + gtid;
            int idx = 0;
            double col[6];

            #pragma unroll
            for (int r = 0; r < 6; ++r) col[r] = r_cur[r];
            solve_LT_6x1_inplace(D_cur_tri, col);
            #pragma unroll
            for (int r = 0; r < 6; ++r) base[(int64_t)(idx++) * total_strips] = col[r];

            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                #pragma unroll
                for (int r = 0; r < 6; ++r) col[r] = G0_cur[r][c];
                solve_LT_6x1_inplace(D_cur_tri, col);
                #pragma unroll
                for (int r = 0; r < 6; ++r) base[(int64_t)(idx++) * total_strips] = col[r];
            }

            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                #pragma unroll
                for (int r = 0; r < 6; ++r) col[r] = W_i[r][c];
                solve_LT_6x1_inplace(D_cur_tri, col);
                #pragma unroll
                for (int r = 0; r < 6; ++r) base[(int64_t)(idx++) * total_strips] = col[r];
            }
        }

        // Advance D_cur_tri, r_cur, and G0_cur to interior node i + 1
        set_A11_minus_WtW_tri(R, Ae, W_i, D_cur_tri);

        {
            double r_next[6];
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                double sy = 0.0;
                #pragma unroll
                for (int k = 0; k < 6; ++k) sy += W_i[k][r] * r_cur[k];
                r_next[r] = rhs_tail[r] - sy;
            }
            #pragma unroll
            for (int r = 0; r < 6; ++r) r_cur[r] = r_next[r];
        }

        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            double gcol[6];
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                double sg = 0.0;
                #pragma unroll
                for (int k = 0; k < 6; ++k) sg += W_i[k][r] * G0_cur[k][c];
                gcol[r] = -sg;
            }
            #pragma unroll
            for (int r = 0; r < 6; ++r) G0_cur[r][c] = gcol[r];
        }
    }

    // -----------------------------------------------------------------------
    // Stage 3 (Epilogue): Last interior node i = N_int - 1 (element e_last = E - 1)
    // + Sub-warp register shuffle Schur reduction + 12x12 plate solve + broadcast
    // -----------------------------------------------------------------------
    double A01_last[6][6];
    {
        const int e_last = E - 1;
        const int n0 = sn[e_last], n1 = sn[e_last + 1];

        double R[9], off0_unused[3], off1[3], rhs_n[12], rhs_tail[6], Al[16];
        compute_element_frame_and_rhs<false, true>(
            xb + n0 * 7, xb + n1 * 7, vb + n0 * 6, vb + n1 * 6,
            nullptr,
            transforms + ((s * E + e_last) * 2 + 1) * 7,
            rest_local + (s * E + e_last) * 12,
            K_pack + (s * E + e_last) * 16,
            M_pack + (s * E + e_last) * 16,
            __ldg(lengths + s * E + e_last), h, m_c, k_c,
            R, off0_unused, off1, rhs_n, Al
        );

        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            r_cur[r] += rhs_n[r];
            rhs_tail[r] = rhs_n[6 + r];
        }
        add_A00_unsheared_tri(R, Al, D_cur_tri);
        eval_strip_node_contact_cuda(
            xb + n0 * 7, vb + n0 * 6, gg_b, pc_b, ground_contact, strip_node_contact, h,
            r_cur, D_cur_tri, strip_wall_sum, strip_min_z
        );
        if (strip_node_contact || (pc_b[0] > 0.5 && pc_b[10] > 0.0)) {
            last_strip_wall_force[(int64_t)b * S + s] = strip_wall_sum;
            last_strip_min_z[(int64_t)b * S + s] = strip_min_z;
        }

        if (!chol_6x6_packed(D_cur_tri)) ok = false;
        solve_L_6x1_inplace(D_cur_tri, r_cur);
        solve_L_6x6_inplace(D_cur_tri, G0_cur);

        // Final front-plate Schur accumulation
        {
            int tri_idx = 0;
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                double sr = 0.0;
                #pragma unroll
                for (int k = 0; k < 6; ++k) sr += G0_cur[k][r] * r_cur[k];
                sh_end0[21 + r][tid] -= sr;
                #pragma unroll
                for (int c = 0; c <= r; ++c) {
                    double sa = 0.0;
                    #pragma unroll
                    for (int k = 0; k < 6; ++k) sa += G0_cur[k][r] * G0_cur[k][c];
                    sh_end0[tri_idx++][tid] -= sa;
                }
            }
        }

        // Build A01_last with back-plate right-shear and solve Y1_last = L_last^{-1} A01_last in-place
        build_A01_unsheared(R, Al, A01_last);
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            double r0[3] = {A01_last[r][0], A01_last[r][1], A01_last[r][2]};
            double cr0[3]; cross3(off1, r0, cr0);
            A01_last[r][3] += cr0[0]; A01_last[r][4] += cr0[1]; A01_last[r][5] += cr0[2];
        }
        solve_L_6x6_inplace(D_cur_tri, A01_last);

        double* seg_sm = sh_seg[seg_in_blk];

        // Sub-warp shuffle reduction (1/2): rows 0..5 (endA00_tri[0..20] and endr0[0..5])
        {
            int t0 = 0;
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                int row_off = (r * (r + 1)) >> 1;
                #pragma unroll
                for (int c = 0; c <= r; ++c) {
                    double v_sum = subwarp_reduce_sum8(warp_mask, sh_end0[t0++][tid]);
                    if (lane == 0) seg_sm[row_off + c] = v_sum;
                }
                double r_sum = subwarp_reduce_sum8(warp_mask, sh_end0[21 + r][tid]);
                if (lane == 0) seg_sm[78 + r] = r_sum;
            }
        }

        // Sub-warp shuffle reduction (2/2): rows 6..11 (endA10, endr1, and stash Y1^T Y1 in freed sh_end0)
        {
            int t1 = 0;
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                int row_idx = 6 + r;
                int row_off = (row_idx * (row_idx + 1)) >> 1;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    double sa10 = 0.0;
                    #pragma unroll
                    for (int k = 0; k < 6; ++k) sa10 += A01_last[k][r] * G0_cur[k][c];
                    double v_sum = subwarp_reduce_sum8(warp_mask, -sa10);
                    if (lane == 0) seg_sm[row_off + c] = v_sum;
                }
                double sr1 = 0.0;
                #pragma unroll
                for (int k = 0; k < 6; ++k) sr1 += A01_last[k][r] * r_cur[k];
                double r_sum = subwarp_reduce_sum8(warp_mask, rhs_tail[r] - sr1);
                if (lane == 0) seg_sm[78 + 6 + r] = r_sum;

                #pragma unroll
                for (int c = 0; c <= r; ++c) {
                    double sa11 = 0.0;
                    #pragma unroll
                    for (int k = 0; k < 6; ++k) sa11 += A01_last[k][r] * A01_last[k][c];
                    sh_end0[t1++][tid] = sa11;
                }
            }
        }

        // Convert y_last -> b_last, Y0_last -> C0_last, Y1_last -> C1_last in-place (frees D_cur_tri)
        solve_LT_6x1_inplace(D_cur_tri, r_cur);
        solve_LT_6x6_inplace(D_cur_tri, G0_cur);
        solve_LT_6x6_inplace(D_cur_tri, A01_last);

        // Assemble A11_last with back-plate shear and shuffle-reduce endA11_tri = A11_last - Y1^T Y1
        {
            double Q00[3][3], Q11[3][3], Q01[3][3], A11_last[6][6];
            rot_diag_3x3(R, Al[0], Al[1], Al[2], Q00);
            rot_diag_3x3(R, Al[3], Al[4], Al[5], Q11);
            rot_two_3x3(R, Al[6], Al[7], Q01);

            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                #pragma unroll
                for (int c = 0; c < 3; ++c) {
                    A11_last[r][c]         = Q00[r][c];
                    A11_last[r][3 + c]     = -Q01[r][c];
                    A11_last[3 + r][c]     = -Q01[c][r];
                    A11_last[3 + r][3 + c] = Q11[r][c];
                }
            }
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                double r1[3] = {A11_last[r][0], A11_last[r][1], A11_last[r][2]};
                double cr1[3]; cross3(off1, r1, cr1);
                A11_last[r][3] += cr1[0]; A11_last[r][4] += cr1[1]; A11_last[r][5] += cr1[2];
            }
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                double c1[3] = {A11_last[0][c], A11_last[1][c], A11_last[2][c]};
                double cr1[3]; cross3(off1, c1, cr1);
                A11_last[3][c] += cr1[0]; A11_last[4][c] += cr1[1]; A11_last[5][c] += cr1[2];
            }

            int t1 = 0;
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                int row_idx = 6 + r;
                int row_off = (row_idx * (row_idx + 1)) >> 1;
                #pragma unroll
                for (int c = 0; c <= r; ++c) {
                    double v_sum = subwarp_reduce_sum8(warp_mask, A11_last[r][c] - sh_end0[t1++][tid]);
                    if (lane == 0) seg_sm[row_off + 6 + c] = v_sum;
                }
            }
        }
    }

    // Lane 0 adds plate mass/inertia + external loads, solves 12x12 SPD Cholesky in-place, and integrates plates
    double vp[12] = {0.0};
    int lane0_ok = 1;
    if (lane == 0) {
        double* seg_sm = sh_seg[seg_in_blk];
        const double ms = mass_scale[b];
        const double m0 = plate_masses[(int64_t)b * P + p0] * ms;
        const double m1 = plate_masses[(int64_t)b * P + p1] * ms;
        const double s0 = m0 / 0.28;
        const double s1 = m1 / 0.28;
        const double ray = 1.0 + rayleigh_alpha * h;
        const double diag_m0[6] = {m0, m0, m0, plate_inertia[0] * s0, plate_inertia[1] * s0, plate_inertia[2] * s0};
        const double diag_m1[6] = {m1, m1, m1, plate_inertia[0] * s1, plate_inertia[1] * s1, plate_inertia[2] * s1};

        const double* vp0_old = vb + p0 * 6;
        const double* vp1_old = vb + p1 * 6;
        const double* ep0 = plate_ext + ((int64_t)b * P + p0) * 6;
        const double* ep1 = plate_ext + ((int64_t)b * P + p1) * 6;

        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            double gc = (c == 2) ? -9.81 * h : 0.0;
            int d0_idx = ((c * (c + 1)) >> 1) + c;
            int c1 = 6 + c;
            int d1_idx = ((c1 * (c1 + 1)) >> 1) + c1;
            seg_sm[d0_idx] += ray * diag_m0[c];
            seg_sm[d1_idx] += ray * diag_m1[c];
            seg_sm[78 + c] += h * ep0[c] + diag_m0[c] * (vp0_old[c] + gc);
            seg_sm[78 + 6 + c] += h * ep1[c] + diag_m1[c] * (vp1_old[c] + gc);
        }

        if (implicit_contact) {
            const double* kc0 = plate_K_contact + ((int64_t)b * P + p0) * 21;
            const double* kc1 = plate_K_contact + ((int64_t)b * P + p1) * 21;
            int k = 0;
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                const int row1 = 6 + r;
                const int row1_off = (row1 * (row1 + 1)) >> 1;
                #pragma unroll
                for (int c = 0; c <= r; ++c) {
                    seg_sm[k] += kc0[k];
                    seg_sm[row1_off + 6 + c] += kc1[k];
                    ++k;
                }
            }
        }

        if (!chol_solve_12x12_packed(seg_sm)) lane0_ok = 0;

        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            double v0c = seg_sm[78 + c];
            double v1c = seg_sm[78 + 6 + c];
            vp[c] = v0c;
            vp[6 + c] = v1c;
            vb[p0 * 6 + c] = v0c;
            vb[p1 * 6 + c] = v1c;
        }
        se3_integrate(xb + p0 * 7, vp, h);
        se3_integrate(xb + p1 * 7, vp + 6, h);
    }

    // Register-level sub-warp broadcast of vp[0..11] + seg_ok from lane 0 to all 8 strip lanes!
    #pragma unroll
    for (int c = 0; c < 12; ++c) {
        vp[c] = __shfl_sync(warp_mask, vp[c], 0, 8);
    }
    lane0_ok = __shfl_sync(warp_mask, lane0_ok, 0, 8);
    const bool seg_ok = ok && (lane0_ok != 0);

    // Back-substitute interior velocities from i = N_int - 1 down to 0 and integrate SE(3)
    double v_cur[6];
    #pragma unroll
    for (int r = 0; r < 6; ++r) {
        double val = r_cur[r];
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            val -= G0_cur[r][c] * vp[c] + A01_last[r][c] * vp[6 + c];
        }
        v_cur[r] = val;
    }
    {
        int nid_last = sn[N_int];
        #pragma unroll
        for (int r = 0; r < 6; ++r) vb[nid_last * 6 + r] = v_cur[r];
        se3_integrate(xb + nid_last * 7, v_cur, h);
    }

    for (int i = N_int - 2; i >= 0; --i) {
        const double* base = strip_buf + (int64_t)i * BUF_STRIDE * total_strips + gtid;
        double v_next[6];
        int idx = 0;
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            v_next[r] = base[(int64_t)(idx++) * total_strips];
        }
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            double vpc = vp[c];
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                v_next[r] -= base[(int64_t)(idx++) * total_strips] * vpc;
            }
        }
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            double vcc = v_cur[c];
            #pragma unroll
            for (int r = 0; r < 6; ++r) {
                v_next[r] -= base[(int64_t)(idx++) * total_strips] * vcc;
            }
        }
        int nid = sn[i + 1];
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            v_cur[r] = v_next[r];
            vb[nid * 6 + r] = v_next[r];
        }
        se3_integrate(xb + nid * 7, v_cur, h);
    }

    if (!seg_ok) {
        *failed = true;
    }
}

}  // namespace

void fused_advance_steps_cuda(
    int64_t steps, int64_t count, double dt, bool ground_contact, double stop_gap,
    bool crank_mode, double horn_r, double guide_d, double slack_m, double k_ten, double c_ten,
    torch::Tensor x,
    torch::Tensor v,
    torch::Tensor servo,
    torch::Tensor target,
    torch::Tensor yaw_tgt,
    torch::Tensor spin,
    torch::Tensor angle,
    torch::Tensor memory,
    torch::Tensor tension,
    torch::Tensor support,
    torch::Tensor last_slip,
    torch::Tensor last_yaw_torque,
    torch::Tensor last_yaw_cap,
    torch::Tensor failed,
    torch::Tensor wheel_mu,
    torch::Tensor mass_scale,
    torch::Tensor yaw_kp,
    torch::Tensor yaw_kd,
    torch::Tensor yaw_torque_limit,
    torch::Tensor yaw_speed_limit,
    torch::Tensor servo_torque_limit,
    torch::Tensor servo_speed_limit,
    torch::Tensor strip_nodes,
    torch::Tensor lengths,
    torch::Tensor transforms,
    torch::Tensor rest_local,
    torch::Tensor K_pack,
    torch::Tensor M_pack,
    torch::Tensor A_pack,
    torch::Tensor mounts,
    torch::Tensor anchors,
    torch::Tensor guides,
    torch::Tensor conn_arm,
    torch::Tensor plate_inertia,
    torch::Tensor plate_ext,
    torch::Tensor strip_buf,
    int64_t contact_model_id,
    bool implicit_contact,
    double ground_kn,
    double ground_cn,
    double ground_hc_k,
    double ground_hc_alpha,
    double ground_soil_exp,
    double ground_soil_unload_ratio,
    double ground_rut_relax_m,
    double ground_rut_lat_k,
    double wheel_mu_lat_ratio,
    double wheel_rolling,
    double wheel_patch_radius,
    double wheel_pivot_mu,
    double ratchet_backlash,
    torch::Tensor ratchet_play,
    torch::Tensor soil_sinkage,
    torch::Tensor plate_K_contact,
    torch::Tensor env_terrain,
    torch::Tensor wall_cfg,
    torch::Tensor last_wall_force,
    double rayleigh_alpha,
    torch::Tensor env_dynamics,
    torch::Tensor plate_masses,
    torch::Tensor rayleigh_beta,
    torch::Tensor ground_geom,
    torch::Tensor pipe_cfg,
    bool strip_node_contact,
    torch::Tensor last_strip_wall_force,
    torch::Tensor last_strip_min_z,
    double wheel_mass_kg
) {
    const int B = (int)x.size(0);
    const int D = (int)x.size(1);
    const int S = (int)strip_nodes.size(0);
    const int E = (int)strip_nodes.size(1) - 1;
    const int P = 2 * (int)count;
    const int64_t total_strips = (int64_t)B * S;
    const int cm_id = (int)contact_model_id;
    const int imp_c = implicit_contact ? 1 : 0;
    const int gc_i = ground_contact ? 1 : 0;
    const int snc_i = strip_node_contact ? 1 : 0;

    const dim3 ext_threads(P, ENVS_PER_EXT_BLK);
    const int ext_blocks = (B + ENVS_PER_EXT_BLK - 1) / ENVS_PER_EXT_BLK;

    const int seg_blocks = (int)((total_strips + SEG_THREADS - 1) / SEG_THREADS);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    for (int64_t st = 0; st < steps; ++st) {
        external_kernel_v3<<<ext_blocks, ext_threads, 0, stream>>>(
            B, (int)count, D, dt, gc_i, stop_gap,
            crank_mode ? 1 : 0, horn_r, guide_d, slack_m, k_ten, c_ten,
            x.data_ptr<double>(),
            v.data_ptr<double>(),
            servo.data_ptr<double>(),
            target.data_ptr<double>(),
            yaw_tgt.data_ptr<double>(),
            spin.data_ptr<double>(),
            angle.data_ptr<double>(),
            memory.data_ptr<double>(),
            tension.data_ptr<double>(),
            support.data_ptr<double>(),
            last_slip.data_ptr<double>(),
            last_yaw_torque.data_ptr<double>(),
            last_yaw_cap.data_ptr<double>(),
            plate_ext.data_ptr<double>(),
            wheel_mu.data_ptr<double>(),
            yaw_kp.data_ptr<double>(),
            yaw_kd.data_ptr<double>(),
            yaw_torque_limit.data_ptr<double>(),
            yaw_speed_limit.data_ptr<double>(),
            servo_torque_limit.data_ptr<double>(),
            servo_speed_limit.data_ptr<double>(),
            mounts.data_ptr<double>(),
            anchors.data_ptr<double>(),
            guides.data_ptr<double>(),
            conn_arm.data_ptr<double>(),
            cm_id, imp_c,
            ground_kn, ground_cn,
            ground_hc_k, ground_hc_alpha, ground_soil_exp, ground_soil_unload_ratio,
            ground_rut_relax_m, ground_rut_lat_k,
            wheel_mu_lat_ratio, wheel_rolling, wheel_patch_radius, wheel_pivot_mu,
            ratchet_backlash,
            ratchet_play.data_ptr<double>(),
            soil_sinkage.data_ptr<double>(),
            plate_K_contact.data_ptr<double>(),
            env_terrain.data_ptr<double>(),
            wall_cfg.data_ptr<double>(),
            last_wall_force.data_ptr<double>(),
            env_dynamics.data_ptr<double>(),
            ground_geom.data_ptr<double>(),
            pipe_cfg.data_ptr<double>(),
            wheel_mass_kg
        );

        if (E == 16) {
            segment_solve_kernel_v3<16><<<seg_blocks, SEG_THREADS, 0, stream>>>(
                B, (int)count, S, E, D, dt, imp_c,
                x.data_ptr<double>(), v.data_ptr<double>(),
                plate_ext.data_ptr<double>(), plate_K_contact.data_ptr<double>(),
                mass_scale.data_ptr<double>(),
                strip_nodes.data_ptr<int32_t>(), lengths.data_ptr<double>(),
                transforms.data_ptr<double>(), rest_local.data_ptr<double>(),
                K_pack.data_ptr<double>(), M_pack.data_ptr<double>(), A_pack.data_ptr<double>(),
                plate_inertia.data_ptr<double>(), strip_buf.data_ptr<double>(),
                failed.data_ptr<bool>(),
                rayleigh_alpha, plate_masses.data_ptr<double>(), rayleigh_beta.data_ptr<double>(),
                gc_i, snc_i, ground_geom.data_ptr<double>(), pipe_cfg.data_ptr<double>(),
                last_strip_wall_force.data_ptr<double>(), last_strip_min_z.data_ptr<double>()
            );
        } else if (E == 8) {
            segment_solve_kernel_v3<8><<<seg_blocks, SEG_THREADS, 0, stream>>>(
                B, (int)count, S, E, D, dt, imp_c,
                x.data_ptr<double>(), v.data_ptr<double>(),
                plate_ext.data_ptr<double>(), plate_K_contact.data_ptr<double>(),
                mass_scale.data_ptr<double>(),
                strip_nodes.data_ptr<int32_t>(), lengths.data_ptr<double>(),
                transforms.data_ptr<double>(), rest_local.data_ptr<double>(),
                K_pack.data_ptr<double>(), M_pack.data_ptr<double>(), A_pack.data_ptr<double>(),
                plate_inertia.data_ptr<double>(), strip_buf.data_ptr<double>(),
                failed.data_ptr<bool>(),
                rayleigh_alpha, plate_masses.data_ptr<double>(), rayleigh_beta.data_ptr<double>(),
                gc_i, snc_i, ground_geom.data_ptr<double>(), pipe_cfg.data_ptr<double>(),
                last_strip_wall_force.data_ptr<double>(), last_strip_min_z.data_ptr<double>()
            );
        } else {
            segment_solve_kernel_v3<0><<<seg_blocks, SEG_THREADS, 0, stream>>>(
                B, (int)count, S, E, D, dt, imp_c,
                x.data_ptr<double>(), v.data_ptr<double>(),
                plate_ext.data_ptr<double>(), plate_K_contact.data_ptr<double>(),
                mass_scale.data_ptr<double>(),
                strip_nodes.data_ptr<int32_t>(), lengths.data_ptr<double>(),
                transforms.data_ptr<double>(), rest_local.data_ptr<double>(),
                K_pack.data_ptr<double>(), M_pack.data_ptr<double>(), A_pack.data_ptr<double>(),
                plate_inertia.data_ptr<double>(), strip_buf.data_ptr<double>(),
                failed.data_ptr<bool>(),
                rayleigh_alpha, plate_masses.data_ptr<double>(), rayleigh_beta.data_ptr<double>(),
                gc_i, snc_i, ground_geom.data_ptr<double>(), pipe_cfg.data_ptr<double>(),
                last_strip_wall_force.data_ptr<double>(), last_strip_min_z.data_ptr<double>()
            );
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("advance_steps", &fused_advance_steps_cuda, "Lossless FP64 fused physics V3 advance (CUDA)");
}
