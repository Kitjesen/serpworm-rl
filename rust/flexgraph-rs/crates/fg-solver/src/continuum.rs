//! Dual Continuum Formulations for `flexgraph-rs`:
//! 1. **Cubic-Hermite Corotational 3D Euler-Bernoulli / Timoshenko Beam** (exact `<1e-10` FP64
//!    parity with `serpworm.physics.gpu_core`, plus optional analytic `beam_strain_jacobian`
//!    and Kirchhoff-Love / Vlasov thin-shell stiffening).
//! 2. **Discrete Elastic Rod (DER) / Cosserat Rod** with parallel-transport Bishop frames,
//!    holonomy-aware discrete twist, anisotropic material curvature, exact 11x11 analytical
//!    Hessians (following Bergou 2008/2010 & Jawed / Py-DiSMech 2025), virtual-ghost rigid
//!    clamp coupling, and $O(E)$ symmetric banded Schur condensation (`bw = 10`).

use crate::schur::{BeamSchurContribution, LinearSolveError, SymmetricBandedLdlt};
use fg_ir::{BeamDofMap, BeamEdgeSpec, BeamFormulation};
use fg_math::{
    beam_frame_transform, beam_local_strain, beam_strain_jacobian, compute_bishop_frames,
    der_bending_energy_grad_hess, der_stretching_energy_grad_hess, der_twisting_energy_grad_hess,
    local_beam_matrices, material_curvature, material_frame, parallel_transport, reference_twist,
    Mat12, Mat3, Mat6, Quat, Se3, Vec12, Vec3, Vec6,
};

/// Precomputed rest-state data for a single corotational beam edge.
#[derive(Clone, Debug)]
pub struct CorotationalBeamData {
    pub beam_id: usize,
    pub body_a: usize,
    pub body_b: Option<usize>,
    pub n_elements: usize,
    pub lengths: Vec<f64>,
    pub elem_transforms: Vec<[Se3; 2]>,
    pub k_local: Vec<Mat12>,
    pub m_local: Vec<Mat12>,
    pub rest_local: Vec<Vec12>,
    pub rayleigh_alpha: f64,
    pub rayleigh_beta: f64,
}

/// Assembled per-element world matrices and nodal elastic wrenches for a corotational beam.
#[derive(Clone, Debug)]
pub struct CorotationalBeamEval {
    pub energy: f64,
    /// Per-element `12x12` elastic wrenches `f_el = -T^T K (u - u_0)` (or `-J^T K (u - u_0)`).
    pub elem_elastic: Vec<Vec12>,
    /// Per-element world stiffness `K_w = T^T K_loc T` (or `J^T K_loc J`).
    pub elem_k: Vec<Mat12>,
    /// Per-element world consistent mass `M_w = T^T M_loc T`.
    pub elem_m: Vec<Mat12>,
}

impl CorotationalBeamData {
    /// Precompute element transforms, local `K`/`M` matrices, and rest corotational strains
    /// matching `serpworm.physics.gpu_core` (`local_matrices_torch` + `rest_local` + `prestress_frac`).
    pub fn from_spec(
        spec: &BeamEdgeSpec,
        map: &BeamDofMap,
        rest_poses: &[Se3],
        stable_log_rest: bool,
    ) -> Self {
        let n_el = spec.n_elements;
        let lengths = spec.resolved_lengths();
        let timoshenko = matches!(spec.formulation, BeamFormulation::CorotationalTimoshenko);

        let elem_transforms: Vec<[Se3; 2]> = if let Some(ref ts) = spec.element_transforms {
            ts.clone()
        } else {
            let mut ts = vec![[Se3::IDENTITY; 2]; n_el];
            ts[0][0] = spec.clamp_a;
            if spec.body_b.is_some() {
                ts[n_el - 1][1] = spec.clamp_b.unwrap_or(Se3::IDENTITY);
            }
            ts
        };

        let mut k_local = Vec::with_capacity(n_el);
        let mut m_local = Vec::with_capacity(n_el);
        let mut rest_local = Vec::with_capacity(n_el);

        let pf = spec.prestress_frac.clamp(0.0, 1.0);
        for e in 0..n_el {
            let l = lengths[e];
            let (k, m) = local_beam_matrices(
                l,
                &spec.section,
                &spec.material,
                spec.thin_shell_compat,
                timoshenko,
            );
            k_local.push(k);
            m_local.push(m);

            let n0 = map.chain_node_ids[e];
            let n1 = map.chain_node_ids[e + 1];
            let x0 = rest_poses[n0];
            let x1 = rest_poses[n1];
            let tf0 = elem_transforms[e][0];
            let tf1 = elem_transforms[e][1];

            let q0 = x0.rot.mul(tf0.rot);
            let q1 = x1.rot.mul(tf1.rot);
            let p0 = x0.pos + x0.rot.rotate(tf0.pos);
            let p1 = x1.pos + x1.rot.rotate(tf1.pos);

            let (u_rest, _) = beam_local_strain(p0, q0, p1, q1, l, stable_log_rest);
            let u0 = if pf > 0.0 {
                let mut flat = [0.0; 12];
                flat[0] = -0.5 * l;
                flat[6] = 0.5 * l;
                u_rest * (1.0 - pf) + Vec12(flat) * pf
            } else {
                u_rest
            };
            rest_local.push(u0);
        }

        Self {
            beam_id: spec.id,
            body_a: spec.body_a,
            body_b: spec.body_b,
            n_elements: n_el,
            lengths,
            elem_transforms,
            k_local,
            m_local,
            rest_local,
            rayleigh_alpha: spec.material.rayleigh_alpha,
            rayleigh_beta: spec.material.rayleigh_beta,
        }
    }

    /// Evaluate a single corotational element `e` at endpoint poses `(x0, x1)`, returning
    /// `(energy_e, f_el_12, k_w_12x12, m_w_12x12)`.
    pub fn evaluate_element(
        &self,
        e: usize,
        x0: Se3,
        x1: Se3,
        use_strain_jacobian: bool,
    ) -> (f64, Vec12, Mat12, Mat12) {
        let tf0 = self.elem_transforms[e][0];
        let tf1 = self.elem_transforms[e][1];

        let q0 = x0.rot.mul(tf0.rot);
        let q1 = x1.rot.mul(tf1.rot);
        let offset0 = x0.rot.rotate(tf0.pos);
        let offset1 = x1.rot.rotate(tf1.pos);
        let p0 = x0.pos + offset0;
        let p1 = x1.pos + offset1;
        let l = self.lengths[e];

        let (u, frame) = beam_local_strain(p0, q0, p1, q1, l, use_strain_jacobian);
        let du = u - self.rest_local[e];
        let k_du = self.k_local[e].mul_vec(du);
        let energy_e = 0.5 * du.dot(k_du);

        let t_mat = beam_frame_transform(frame.rot, offset0, offset1);
        let m_w = self.m_local[e].congruent_transform(&t_mat);

        let (f_el, k_w) = if use_strain_jacobian {
            let j_mat = beam_strain_jacobian(p0, q0, offset0, p1, q1, offset1, l);
            (
                -(j_mat.transpose().mul_vec(k_du)),
                self.k_local[e].congruent_transform(&j_mat),
            )
        } else {
            (
                -(t_mat.transpose().mul_vec(k_du)),
                self.k_local[e].congruent_transform(&t_mat),
            )
        };
        (energy_e, f_el, k_w, m_w)
    }

    /// Evaluate corotational beam elastic energy, element wrenches, and world `K`/`M` matrices.
    ///
    /// - `use_strain_jacobian = false`: uses frozen-frame Plücker transform `T` + `log_legacy`
    ///   (exact `< 1e-10` FP64 parity with `serpworm.physics.gpu_core`).
    /// - `use_strain_jacobian = true`: uses exact analytic `beam_strain_jacobian` `J` + `Quat::log`.
    pub fn evaluate(
        &self,
        map: &BeamDofMap,
        poses: &[Se3],
        use_strain_jacobian: bool,
    ) -> CorotationalBeamEval {
        let mut energy = 0.0;
        let mut elem_elastic = Vec::with_capacity(self.n_elements);
        let mut elem_k = Vec::with_capacity(self.n_elements);
        let mut elem_m = Vec::with_capacity(self.n_elements);

        for e in 0..self.n_elements {
            let n0 = map.chain_node_ids[e];
            let n1 = map.chain_node_ids[e + 1];
            let (en_e, f_el, k_w, m_w) =
                self.evaluate_element(e, poses[n0], poses[n1], use_strain_jacobian);
            energy += en_e;
            elem_elastic.push(f_el);
            elem_k.push(k_w);
            elem_m.push(m_w);
        }

        CorotationalBeamEval {
            energy,
            elem_elastic,
            elem_k,
            elem_m,
        }
    }
}

/// Precomputed rest geometry and material stiffnesses for a Discrete Elastic Rod (DER) edge.
#[derive(Clone, Debug)]
pub struct DerBeamData {
    pub beam_id: usize,
    pub body_a: usize,
    pub body_b: Option<usize>,
    pub clamp_a: Se3,
    pub clamp_b: Option<Se3>,
    pub n_elements: usize,
    pub rest_lengths: Vec<f64>,
    /// Voronoi lengths at interior vertices `1..n_elements` (len `n_elements - 1`).
    pub voronoi_lengths: Vec<f64>,
    /// Rest curvatures `[kappa1, kappa2]` at interior vertices `1..n_elements`.
    pub rest_kappa: Vec<[f64; 2]>,
    /// Rest twist at interior vertices `1..n_elements`.
    pub rest_twist: Vec<f64>,
    /// Rest curvature and twist at clamp A hinge (joining virtual ghost edge `-1` and edge `0`).
    pub clamp_a_rest_kappa: [f64; 2],
    pub clamp_a_rest_twist: f64,
    /// Rest curvature and twist at clamp B hinge when `body_b.is_some()`.
    pub clamp_b_rest_kappa: [f64; 2],
    pub clamp_b_rest_twist: f64,
    /// Stretching stiffness `EA`, bending stiffnesses `[E_y * Iy, E * Iz]`, torsional stiffness `GJ`.
    pub ea: f64,
    pub ei: [f64; 2],
    pub gj: f64,
    /// Lumped translational mass per free vertex (len `num_free_verts`).
    pub free_vertex_mass: Vec<f64>,
    /// Lumped rotational twist inertia per edge `rho * (Iy + Iz) * L_e` (len `n_elements`).
    pub edge_twist_inertia: Vec<f64>,
    /// Lumped translational mass contributed to endpoint A (and B if clamped-clamped).
    pub end_a_mass: f64,
    pub end_b_mass: f64,
    pub rayleigh_alpha: f64,
    pub rayleigh_beta: f64,
}

/// Dynamic state of a Discrete Elastic Rod (time-parallel Bishop frames, edge twists, ref twists).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DerBeamState {
    pub bishop_t: Vec<Vec3>,
    pub bishop_u: Vec<Vec3>,
    pub theta: Vec<f64>,
    pub theta_vel: Vec<f64>,
    pub ref_twist: Vec<f64>,
    pub clamp_a_ref_twist: f64,
    pub clamp_b_ref_twist: f64,
}

/// Cached banded $LDL^\top$ factor for a single DER beam (interleaved DOFs, half-bandwidth 10),
/// condensing all free vertex positions and edge twist angles onto rigid clamp endpoints `(body_a, body_b)`.
#[derive(Clone, Debug)]
pub struct DerBeamFactor {
    pub beam_id: usize,
    pub n_elements: usize,
    pub n_free_verts: usize,
    pub is_cantilever: bool,
    pub n_dof: usize,
    banded: SymmetricBandedLdlt,
    /// `H_{I, a}` coupling columns (len `n_dof` each, stored for multi-RHS `condense_rhs`).
    h_ia: [Vec<f64>; 6],
    /// `H_{I, b}` coupling columns (len `n_dof` each; empty if cantilever).
    h_ib: [Vec<f64>; 6],
    /// `z0 = H_{II}^{-1} b_I` (len `n_dof`).
    z0: Vec<f64>,
    /// `z_a[col]` = `H_{II}^{-1} H_{I, a}[:, col]` (`6` columns, each len `n_dof`).
    z_a: [Vec<f64>; 6],
    /// `z_b[col]` = `H_{II}^{-1} H_{I, b}[:, col]` (`6` columns, each len `n_dof`; empty if cantilever).
    z_b: [Vec<f64>; 6],
}

/// Output of [`DerBeamData::assemble_and_condense`] including exact dynamic interior residuals
/// for nonlinear Newton-Raphson convergence monitoring.
#[derive(Clone, Debug)]
pub struct DerAssembleOutput {
    pub energy: f64,
    pub factor: DerBeamFactor,
    pub contrib: BeamSchurContribution,
    pub end_a_rhs: Vec6,
    pub end_b_rhs: Vec6,
    /// Free vertex dynamic residuals `r_verts = -rhs_i_verts` (`0` at exact equilibrium).
    pub res_verts: Vec<Vec3>,
    /// Edge twist dynamic residuals `r_thetas = -rhs_i_thetas` (`0` at exact equilibrium).
    pub res_thetas: Vec<f64>,
}

/// Half-bandwidth of the interleaved DER DOF ordering `[theta_0, x_1, theta_1, x_2, theta_2, ...]`.
pub const DER_HALF_BANDWIDTH: usize = 10;

/// Dense `11x11` DER bending/twisting stencil matrix.
type Mat11 = [[f64; 11]; 11];

/// Cached `(H_+, G_H)` pair for exact Daleckii-Krein differentiation of `project_spd_11x11`.
type SpdDaleckiiPair = (Mat11, Mat11);

/// Symmetric Jacobi eigendecomposition of an `11x11` matrix `h`.
/// Returns `(eigenvalues, V)` such that `h ≈ V * diag(eigenvalues) * V^T` with `V^T V = I`.
fn sym_eigen_11x11(h: &Mat11) -> ([f64; 11], Mat11) {
    let mut a = [[0.0; 11]; 11];
    for i in 0..11 {
        for j in 0..11 {
            a[i][j] = 0.5 * (h[i][j] + h[j][i]);
        }
    }
    let mut v = [[0.0; 11]; 11];
    for i in 0..11 {
        v[i][i] = 1.0;
    }

    let mut scale = 0.0_f64;
    for i in 0..11 {
        for j in 0..11 {
            scale = scale.max(a[i][j].abs());
        }
    }
    let tol = 1e-15 * scale.max(1e-30);

    for _ in 0..800 {
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..11 {
            for j in (i + 1)..11 {
                let val = a[i][j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off <= tol {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..11 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        for row in &mut v {
            let vrp = row[p];
            let vrq = row[q];
            row[p] = c * vrp - s * vrq;
            row[q] = s * vrp + c * vrq;
        }
    }

    let evals = std::array::from_fn(|k| a[k][k]);
    (evals, v)
}

/// Exact Daleckii-Krein spectral projection & cotangent for `H_+ = proj_spd(H, 0.0)`:
/// returns `(H_+, G_H)` such that for any smooth symmetric perturbation `dH`,
/// `l^T d(H_+) v = sum_{i,j} G_H[i][j] * dH[i][j]` without straddling zero-eigenvalue kinks.
fn daleckii_krein_spd_11x11(h: &Mat11, l: &[f64; 11], v: &[f64; 11]) -> SpdDaleckiiPair {
    let (evals, vecs) = sym_eigen_11x11(h);
    let mut scale = 0.0_f64;
    for &ev in &evals {
        scale = scale.max(ev.abs());
    }

    let mut h_plus = [[0.0; 11]; 11];
    for k in 0..11 {
        let lam = evals[k].max(0.0);
        if lam == 0.0 {
            continue;
        }
        for i in 0..11 {
            let vik = vecs[i][k] * lam;
            for j in 0..11 {
                h_plus[i][j] += vik * vecs[j][k];
            }
        }
    }

    let mut lt = [0.0; 11];
    let mut vt = [0.0; 11];
    for k in 0..11 {
        let mut ls = 0.0;
        let mut vs = 0.0;
        for i in 0..11 {
            ls += vecs[i][k] * l[i];
            vs += vecs[i][k] * v[i];
        }
        lt[k] = ls;
        vt[k] = vs;
    }

    let mut g_tilde = [[0.0; 11]; 11];
    for a in 0..11 {
        let la = evals[a];
        let fa = la.max(0.0);
        for b in 0..11 {
            let lb = evals[b];
            let s_ab = 0.5 * (lt[a] * vt[b] + vt[a] * lt[b]);
            if s_ab == 0.0 {
                continue;
            }
            let f1 = if la > 0.0 && lb > 0.0 {
                1.0
            } else if la <= 0.0 && lb <= 0.0 {
                0.0
            } else {
                let denom = la - lb;
                if denom.abs() > 1e-30 * scale.max(1.0) {
                    (fa - lb.max(0.0)) / denom
                } else {
                    0.0
                }
            };
            g_tilde[a][b] = s_ab * f1;
        }
    }

    // Transform back: G_H = V * g_tilde * V^T
    let mut tmp = [[0.0; 11]; 11];
    for i in 0..11 {
        for b in 0..11 {
            let mut sum = 0.0;
            for a in 0..11 {
                sum += vecs[i][a] * g_tilde[a][b];
            }
            tmp[i][b] = sum;
        }
    }
    let mut g_h = [[0.0; 11]; 11];
    for i in 0..11 {
        for j in 0..11 {
            let mut sum = 0.0;
            for b in 0..11 {
                sum += tmp[i][b] * vecs[j][b];
            }
            g_h[i][j] = sum;
        }
    }
    (h_plus, g_h)
}

/// Cached unperturbed SPD projection & Daleckii-Krein cotangents for exact, smooth
/// differentiation of DER elastic contractions when `project_der_hessian_spd == true`.
pub(crate) struct DerElasticVjpCache {
    stretch_clamp_zero: Vec<bool>,
    interior_spd: Vec<SpdDaleckiiPair>,
    clamp_a_spd: SpdDaleckiiPair,
    clamp_b_spd: Option<SpdDaleckiiPair>,
}

/// Symmetric Jacobi SPD projection for an `11x11` DER bending/twisting Hessian block.
pub fn project_spd_11x11(h: &[[f64; 11]; 11], min_eig: f64) -> [[f64; 11]; 11] {
    let (evals, v) = sym_eigen_11x11(h);
    let mut out = [[0.0; 11]; 11];
    for k in 0..11 {
        let lam = evals[k].max(min_eig);
        if lam == 0.0 {
            continue;
        }
        for i in 0..11 {
            let vik = v[i][k] * lam;
            for j in 0..11 {
                out[i][j] += vik * v[j][k];
            }
        }
    }
    out
}

impl DerBeamData {
    /// Interleaved scalar index of edge twist `theta_e` (`e in 0..n_elements`).
    #[inline]
    pub fn theta_dof(e: usize) -> usize {
        4 * e
    }

    /// Starting scalar index of free vertex `v` (`v = 1..=n_free_verts`, occupying 3 consecutive slots).
    #[inline]
    pub fn vert_dof_start(v: usize) -> usize {
        debug_assert!(v >= 1);
        4 * (v - 1) + 1
    }

    /// Total free scalar DOFs `3 * n_free_verts + n_elements`.
    #[inline]
    pub fn n_free_dof(&self) -> usize {
        let n_free_verts = if self.body_b.is_some() {
            self.n_elements.saturating_sub(1)
        } else {
            self.n_elements
        };
        3 * n_free_verts + self.n_elements
    }

    /// Build rest DER data and initial Bishop-frame state from `BeamEdgeSpec`.
    pub fn from_spec(spec: &BeamEdgeSpec) -> (Self, DerBeamState) {
        let n_el = spec.n_elements;
        let lengths = spec.resolved_lengths();
        let props = spec.section.properties();
        let e_mod = spec.material.youngs_modulus;
        let nu = spec.material.poisson_ratio;
        let g_mod = spec.material.shear_modulus();
        let rho = spec.material.density;

        let e_y = if spec.thin_shell_compat {
            e_mod / (1.0 - nu * nu)
        } else {
            e_mod
        };
        let j_eff = if spec.thin_shell_compat {
            match props.thin_width {
                Some(w) => props.j_torsion * (1.0 + (w * w) / (12.0 * 0.145 * 0.145 * (1.0 - nu))),
                None => props.j_torsion,
            }
        } else {
            props.j_torsion
        };

        let ea = e_mod * props.area;
        // Material director m1 aligns with local Y (width), m2 with local Z (thickness normal).
        // Bending around m1 deflects out-of-plane (Iy), bending around m2 deflects in-plane (Iz).
        let ei = [e_y * props.iy, e_mod * props.iz];
        let gj = g_mod * j_eff;

        let rest_pts: Vec<Vec3> = spec.centerline_rest_poses.iter().map(|s| s.pos).collect();
        let u0_init = spec.centerline_rest_poses[0].rot.rotate(Vec3::EY);
        let bishop = compute_bishop_frames(&rest_pts, u0_init);
        let bishop_t: Vec<Vec3> = bishop.iter().map(|b| b.0).collect();
        let bishop_u: Vec<Vec3> = bishop.iter().map(|b| b.1).collect();

        // Compute initial material twist angles so initial material frames align with centerline rest poses
        let mut theta = vec![0.0; n_el];
        for e in 0..n_el {
            let target_m1 = spec.centerline_rest_poses[e].rot.rotate(Vec3::EY);
            let t = bishop_t[e];
            let u = bishop_u[e];
            let v = t.cross(u);
            let proj = (target_m1 - target_m1.dot(t) * t).normalized_or(u);
            theta[e] = proj.dot(v).atan2(proj.dot(u));
        }

        let mut voronoi_lengths = Vec::with_capacity(n_el.saturating_sub(1));
        let mut rest_kappa = Vec::with_capacity(n_el.saturating_sub(1));
        let mut rest_twist = Vec::with_capacity(n_el.saturating_sub(1));
        let mut ref_twist = Vec::with_capacity(n_el.saturating_sub(1));

        let pf_scale = 1.0 - spec.prestress_frac.clamp(0.0, 1.0);
        for i in 1..n_el {
            let vl = 0.5 * (lengths[i - 1] + lengths[i]);
            voronoi_lengths.push(vl);
            let e0 = rest_pts[i] - rest_pts[i - 1];
            let e1 = rest_pts[i + 1] - rest_pts[i];
            let kb = fg_math::curvature_binormal(e0, e1);
            let v0 = bishop_t[i - 1].cross(bishop_u[i - 1]);
            let v1 = bishop_t[i].cross(bishop_u[i]);
            let (m1_0, m2_0) = material_frame(bishop_t[i - 1], bishop_u[i - 1], v0, theta[i - 1]);
            let (m1_1, m2_1) = material_frame(bishop_t[i], bishop_u[i], v1, theta[i]);
            let k = material_curvature(kb, m1_0, m2_0, m1_1, m2_1);
            let rt = reference_twist(bishop_u[i - 1], bishop_t[i - 1], bishop_u[i], bishop_t[i], 0.0);
            ref_twist.push(rt);
            rest_kappa.push([k[0] * pf_scale, k[1] * pf_scale]);
            rest_twist.push((theta[i] - theta[i - 1] + rt) * pf_scale);
        }

        // Clamp A rest curvature & twist (virtual ghost edge from clamp_a orientation into edge 0)
        let q_ca = spec.centerline_rest_poses[0].rot;
        let ta_ghost = q_ca.rotate(Vec3::EX);
        let m1_a = q_ca.rotate(Vec3::EY);
        let m2_a = q_ca.rotate(Vec3::EZ);
        let kb_a = fg_math::curvature_binormal(ta_ghost * lengths[0], rest_pts[1] - rest_pts[0]);
        let (m1_0, m2_0) = material_frame(
            bishop_t[0],
            bishop_u[0],
            bishop_t[0].cross(bishop_u[0]),
            theta[0],
        );
        let k_a = material_curvature(kb_a, m1_a, m2_a, m1_0, m2_0);
        let clamp_a_ref_twist = reference_twist(m1_a, ta_ghost, bishop_u[0], bishop_t[0], 0.0);
        let clamp_a_rest_kappa = [k_a[0] * pf_scale, k_a[1] * pf_scale];
        let clamp_a_rest_twist = (theta[0] + clamp_a_ref_twist) * pf_scale;

        let (clamp_b_rest_kappa, clamp_b_rest_twist, clamp_b_ref_twist) = if spec.body_b.is_some() {
            let q_cb = spec.centerline_rest_poses[n_el].rot;
            let tb_ghost = q_cb.rotate(Vec3::EX);
            let m1_b = q_cb.rotate(Vec3::EY);
            let m2_b = q_cb.rotate(Vec3::EZ);
            let last = n_el - 1;
            let kb_b = fg_math::curvature_binormal(
                rest_pts[n_el] - rest_pts[last],
                tb_ghost * lengths[last],
            );
            let (m1_l, m2_l) = material_frame(
                bishop_t[last],
                bishop_u[last],
                bishop_t[last].cross(bishop_u[last]),
                theta[last],
            );
            let k_b = material_curvature(kb_b, m1_l, m2_l, m1_b, m2_b);
            let rt_b = reference_twist(bishop_u[last], bishop_t[last], m1_b, tb_ghost, 0.0);
            (
                [k_b[0] * pf_scale, k_b[1] * pf_scale],
                (-theta[last] + rt_b) * pf_scale,
                rt_b,
            )
        } else {
            ([0.0, 0.0], 0.0, 0.0)
        };

        let n_free_verts = spec.num_free_beam_nodes();
        let mut free_vertex_mass = Vec::with_capacity(n_free_verts);
        for v in 1..=n_free_verts {
            let vl = if v < n_el {
                0.5 * (lengths[v - 1] + lengths[v])
            } else {
                0.5 * lengths[n_el - 1]
            };
            free_vertex_mass.push((rho * props.area * vl).max(1e-9));
        }
        let edge_twist_inertia: Vec<f64> = lengths
            .iter()
            .map(|&l| (rho * (props.iy + props.iz) * l).max(1e-12))
            .collect();
        let end_a_mass = 0.5 * rho * props.area * lengths[0];
        let end_b_mass = if spec.body_b.is_some() {
            0.5 * rho * props.area * lengths[n_el - 1]
        } else {
            0.0
        };

        let data = Self {
            beam_id: spec.id,
            body_a: spec.body_a,
            body_b: spec.body_b,
            clamp_a: spec.clamp_a,
            clamp_b: spec.clamp_b,
            n_elements: n_el,
            rest_lengths: lengths,
            voronoi_lengths,
            rest_kappa,
            rest_twist,
            clamp_a_rest_kappa,
            clamp_a_rest_twist,
            clamp_b_rest_kappa,
            clamp_b_rest_twist,
            ea,
            ei,
            gj,
            free_vertex_mass,
            edge_twist_inertia,
            end_a_mass,
            end_b_mass,
            rayleigh_alpha: spec.material.rayleigh_alpha,
            rayleigh_beta: spec.material.rayleigh_beta,
        };
        let state = DerBeamState {
            bishop_t,
            bishop_u,
            theta,
            theta_vel: vec![0.0; n_el],
            ref_twist,
            clamp_a_ref_twist,
            clamp_b_ref_twist,
        };
        (data, state)
    }

    /// Compute current world vertex positions `x[0..=E]` and clamp frames for this DER rod.
    pub fn vertex_positions(&self, map: &BeamDofMap, poses: &[Se3]) -> Vec<Vec3> {
        let mut pts = Vec::with_capacity(self.n_elements + 1);
        let xa = poses[self.body_a];
        pts.push(xa.pos + xa.rot.rotate(self.clamp_a.pos));
        for &nid in &map.free_node_ids {
            pts.push(poses[nid].pos);
        }
        if let Some(bb) = self.body_b {
            let xb = poses[bb];
            let cb = self.clamp_b.unwrap_or(Se3::IDENTITY);
            pts.push(xb.pos + xb.rot.rotate(cb.pos));
        }
        pts
    }

    /// Update time-parallel Bishop frames and unwrapped reference twists after a pose/twist update,
    /// and synchronize free beam node SE(3) orientations `(t, m1, m2)` with the DER material frames.
    pub fn update_bishop_frames_and_sync_poses(
        &self,
        map: &BeamDofMap,
        poses: &mut [Se3],
        state: &mut DerBeamState,
    ) {
        let pts = self.vertex_positions(map, poses);
        for e in 0..self.n_elements {
            let t_new = (pts[e + 1] - pts[e]).normalized_or(state.bishop_t[e]);
            let u_new = parallel_transport(state.bishop_u[e], state.bishop_t[e], t_new);
            state.bishop_t[e] = t_new;
            state.bishop_u[e] = u_new;
        }
        for i in 1..self.n_elements {
            state.ref_twist[i - 1] = reference_twist(
                state.bishop_u[i - 1],
                state.bishop_t[i - 1],
                state.bishop_u[i],
                state.bishop_t[i],
                state.ref_twist[i - 1],
            );
        }
        let q_ca = poses[self.body_a].rot.mul(self.clamp_a.rot);
        let ta_ghost = q_ca.rotate(Vec3::EX);
        let m1_a = q_ca.rotate(Vec3::EY);
        state.clamp_a_ref_twist = reference_twist(
            m1_a,
            ta_ghost,
            state.bishop_u[0],
            state.bishop_t[0],
            state.clamp_a_ref_twist,
        );
        if let Some(bb) = self.body_b {
            let cb = self.clamp_b.unwrap_or(Se3::IDENTITY);
            let q_cb = poses[bb].rot.mul(cb.rot);
            let tb_ghost = q_cb.rotate(Vec3::EX);
            let m1_b = q_cb.rotate(Vec3::EY);
            let last = self.n_elements - 1;
            state.clamp_b_ref_twist = reference_twist(
                state.bishop_u[last],
                state.bishop_t[last],
                m1_b,
                tb_ghost,
                state.clamp_b_ref_twist,
            );
        }

        // Sync SE(3) orientation of free beam nodes from adjacent edge material frames
        for (idx, &nid) in map.free_node_ids.iter().enumerate() {
            let e = idx.min(self.n_elements - 1);
            let t = state.bishop_t[e];
            let u = state.bishop_u[e];
            let v = t.cross(u);
            let (m1, m2) = material_frame(t, u, v, state.theta[e]);
            let r = Mat3::from_cols(t, m1, m2);
            poses[nid].rot = Quat::from_mat3(r);
        }
    }

    /// Evaluate exact DER elastic energy, negative elastic gradients (`-grad E`), and assemble
    /// the $O(E)$ banded system + exact rigid clamp Schur condensation `(DerBeamFactor, BeamSchurContribution)`.
    ///
    /// Solves the linearized implicit step:
    /// `(c_m * M + c_k * K_der + K_contact) * delta_y = rhs_I`,
    /// where for linearly implicit Euler (`v_new` formulation): `c_m = 1 + h * alpha`, `c_k = h * beta + h^2`,
    /// and for Newton-Raphson (`delta_x` formulation): `c_m = c_a + alpha * c_v`, `c_k = 1 + beta * c_v`.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_and_condense(
        &self,
        map: &BeamDofMap,
        poses: &[Se3],
        vels: &[Vec6],
        accels: Option<&[Vec6]>,
        theta_accels: Option<&[f64]>,
        state: &DerBeamState,
        gravity: Vec3,
        interior_ext_force: &[Vec3],
        interior_contact_k: &[Mat3],
        dt: f64,
        c_m: f64,
        c_k: f64,
        is_velocity_form: bool,
        project_spd: bool,
    ) -> Result<DerAssembleOutput, LinearSolveError> {
        let n_el = self.n_elements;
        let n_free_verts = map.free_node_ids.len();
        let is_cantilever = self.body_b.is_none();
        let n_dof = self.n_free_dof();

        let pts = self.vertex_positions(map, poses);
        let mut cur_t = Vec::with_capacity(n_el);
        let mut cur_u = Vec::with_capacity(n_el);
        let mut m1 = Vec::with_capacity(n_el);
        let mut m2 = Vec::with_capacity(n_el);

        for e in 0..n_el {
            let t = (pts[e + 1] - pts[e]).normalized_or(state.bishop_t[e]);
            let u = parallel_transport(state.bishop_u[e], state.bishop_t[e], t);
            let v = t.cross(u);
            let (d1, d2) = material_frame(t, u, v, state.theta[e]);
            cur_t.push(t);
            cur_u.push(u);
            m1.push(d1);
            m2.push(d2);
        }

        let mut banded = SymmetricBandedLdlt::zeros(n_dof, DER_HALF_BANDWIDTH);
        let mut grad_i = vec![0.0; n_dof];
        let mut h_ia: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n_dof]);
        let mut h_ib: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n_dof]);
        let mut h_aa = Mat6::ZERO;
        let mut h_ab = Mat6::ZERO;
        let mut h_bb = Mat6::ZERO;
        let mut grad_a = Vec6::ZERO;
        let mut grad_b = Vec6::ZERO;
        let mut total_energy = 0.0;

        // Helper mapping clamp vertex kinematics onto rigid body 6D twists:
        // dx_p = dv_lin - [offset]_x d_omega  =>  J_p = [I_3, -[offset]_x] (3x6)
        let xa = poses[self.body_a];
        let off_a = xa.rot.rotate(self.clamp_a.pos);
        let q_ca = xa.rot.mul(self.clamp_a.rot);
        let ta_ghost = q_ca.rotate(Vec3::EX);
        let m1_a = q_ca.rotate(Vec3::EY);
        let m2_a = q_ca.rotate(Vec3::EZ);
        let off_a_ghost = off_a - self.rest_lengths[0] * ta_ghost;

        let (off_b, tb_ghost, m1_b, m2_b, off_b_ghost) = if let Some(bb) = self.body_b {
            let xb = poses[bb];
            let cb = self.clamp_b.unwrap_or(Se3::IDENTITY);
            let ob = xb.rot.rotate(cb.pos);
            let q_cb = xb.rot.mul(cb.rot);
            let tb = q_cb.rotate(Vec3::EX);
            (
                ob,
                tb,
                q_cb.rotate(Vec3::EY),
                q_cb.rotate(Vec3::EZ),
                ob + self.rest_lengths[n_el - 1] * tb,
            )
        } else {
            (Vec3::ZERO, Vec3::EX, Vec3::EY, Vec3::EZ, Vec3::ZERO)
        };

        let point_jac_3x6 = |offset: Vec3| -> [[f64; 6]; 3] {
            let s = (-offset.skew()).0;
            [
                [1.0, 0.0, 0.0, s[0][0], s[0][1], s[0][2]],
                [0.0, 1.0, 0.0, s[1][0], s[1][1], s[1][2]],
                [0.0, 0.0, 1.0, s[2][0], s[2][1], s[2][2]],
            ]
        };
        let jac_x0_a = point_jac_3x6(off_a);
        let jac_xm1_a = point_jac_3x6(off_a_ghost);
        let jac_xe_b = point_jac_3x6(off_b);
        let jac_xep1_b = point_jac_3x6(off_b_ghost);

        // Pre-pack interior velocities for consistent Rayleigh stiffness damping in Newton mode
        let add_rayleigh_beta_res = !is_velocity_form && self.rayleigh_beta > 0.0;
        let va_vel = vels[self.body_a];
        let vb_vel = self.body_b.map_or(Vec6::ZERO, |bb| vels[bb]);
        let va_pt_vel = va_vel.lin() + va_vel.ang().cross(off_a);
        let vb_pt_vel = vb_vel.lin() + vb_vel.ang().cross(off_b);
        let mut vel_i = vec![0.0; n_dof];
        if add_rayleigh_beta_res {
            for v in 1..=n_free_verts {
                let s = Self::vert_dof_start(v);
                let vl = vels[map.free_node_ids[v - 1]].lin();
                vel_i[s] = vl.x();
                vel_i[s + 1] = vl.y();
                vel_i[s + 2] = vl.z();
            }
            for e in 0..n_el {
                vel_i[Self::theta_dof(e)] = state.theta_vel[e];
            }
        }

        // --- 1. Stretching springs across all E edges ---
        for e in 0..n_el {
            let se = der_stretching_energy_grad_hess(
                pts[e],
                pts[e + 1],
                self.rest_lengths[e],
                self.ea,
                project_spd,
            );
            total_energy += se.energy;

            // Endpoints of edge e: (v0, v1) where v=0 is body_a, v=E (if clamped) is body_b, else free vert
            let v0_free = e >= 1;
            let v1_free = e < n_free_verts;
            let mut g0 = se.grad.lin();
            let mut g1 = se.grad.ang();
            if add_rayleigh_beta_res {
                let v0 = if v0_free {
                    vels[map.free_node_ids[e - 1]].lin()
                } else {
                    va_pt_vel
                };
                let v1 = if v1_free {
                    vels[map.free_node_ids[e]].lin()
                } else {
                    vb_pt_vel
                };
                g0 += (se.hess.block(0, 0).mul_vec(v0) + se.hess.block(0, 1).mul_vec(v1))
                    * self.rayleigh_beta;
                g1 += (se.hess.block(1, 0).mul_vec(v0) + se.hess.block(1, 1).mul_vec(v1))
                    * self.rayleigh_beta;
            }
            let k00 = se.hess.block(0, 0) * c_k;
            let k01 = se.hess.block(0, 1) * c_k;
            let k10 = se.hess.block(1, 0) * c_k;
            let k11 = se.hess.block(1, 1) * c_k;

            // Accumulate gradients
            if v0_free {
                let s0 = Self::vert_dof_start(e);
                for r in 0..3 {
                    grad_i[s0 + r] += g0[r];
                }
            } else {
                grad_a += Vec6::from_parts(g0, off_a.cross(g0));
            }
            if v1_free {
                let s1 = Self::vert_dof_start(e + 1);
                for r in 0..3 {
                    grad_i[s1 + r] += g1[r];
                }
            } else {
                grad_b += Vec6::from_parts(g1, off_b.cross(g1));
            }

            // Accumulate Hessians (scaled by c_k)
            match (v0_free, v1_free) {
                (true, true) => {
                    let s0 = Self::vert_dof_start(e);
                    let s1 = Self::vert_dof_start(e + 1);
                    for r in 0..3 {
                        for c in 0..3 {
                            banded.add_from_full_sym(s0 + r, s0 + c, k00.0[r][c]);
                            banded.add_from_full_sym(s1 + r, s1 + c, k11.0[r][c]);
                            banded.add_from_full_sym(s1 + r, s0 + c, k10.0[r][c]);
                        }
                    }
                }
                (false, true) => {
                    // v0 on body_a, v1 free
                    let s1 = Self::vert_dof_start(e + 1);
                    h_aa += jt_k_j_3x6(&jac_x0_a, k00, &jac_x0_a);
                    for r in 0..3 {
                        for c in 0..3 {
                            banded.add_from_full_sym(s1 + r, s1 + c, k11.0[r][c]);
                        }
                        for col in 0..6 {
                            let mut val = 0.0;
                            for k in 0..3 {
                                val += k10.0[r][k] * jac_x0_a[k][col];
                            }
                            h_ia[col][s1 + r] += val;
                        }
                    }
                }
                (true, false) => {
                    // v0 free, v1 on body_b
                    let s0 = Self::vert_dof_start(e);
                    h_bb += jt_k_j_3x6(&jac_xe_b, k11, &jac_xe_b);
                    for r in 0..3 {
                        for c in 0..3 {
                            banded.add_from_full_sym(s0 + r, s0 + c, k00.0[r][c]);
                        }
                        for col in 0..6 {
                            let mut val = 0.0;
                            for k in 0..3 {
                                val += k01.0[r][k] * jac_xe_b[k][col];
                            }
                            h_ib[col][s0 + r] += val;
                        }
                    }
                }
                (false, false) => {
                    // Single-element clamped-clamped DER rod (n_elements == 1)
                    h_aa += jt_k_j_3x6(&jac_x0_a, k00, &jac_x0_a);
                    h_bb += jt_k_j_3x6(&jac_xe_b, k11, &jac_xe_b);
                    let ab = jt_k_j_3x6(&jac_x0_a, k01, &jac_xe_b);
                    h_ab += ab;
                }
            }
        }

        // Helper to scatter an 11-DOF stencil `[x0(0..3), x1(3..6), x2(6..9), th0(9), th1(10)]`
        // where each local DOF maps either to an interior scalar index or to a 6D row on body_a / body_b.
        #[derive(Copy, Clone)]
        enum StencilDof {
            Interior(usize),
            BodyA([f64; 6]),
            BodyB([f64; 6]),
        }

        let scatter_stencil11 = |map11: &[StencilDof; 11],
                                 g11: &[f64; 11],
                                 h11: &[[f64; 11]; 11],
                                 banded: &mut SymmetricBandedLdlt,
                                 grad_i: &mut [f64],
                                 h_ia: &mut [Vec<f64>; 6],
                                 h_ib: &mut [Vec<f64>; 6],
                                 h_aa: &mut Mat6,
                                 h_ab: &mut Mat6,
                                 h_bb: &mut Mat6,
                                 grad_a: &mut Vec6,
                                 grad_b: &mut Vec6| {
            // Gradient scatter (plus optional Rayleigh stiffness damping beta * H * v in Newton mode)
            let mut v11 = [0.0; 11];
            if add_rayleigh_beta_res {
                for j in 0..11 {
                    v11[j] = match map11[j] {
                        StencilDof::Interior(idx) => vel_i[idx],
                        StencilDof::BodyA(jrow) => (0..6).map(|c| jrow[c] * va_vel.0[c]).sum(),
                        StencilDof::BodyB(jrow) => (0..6).map(|c| jrow[c] * vb_vel.0[c]).sum(),
                    };
                }
            }
            for i in 0..11 {
                let mut gi = g11[i];
                if add_rayleigh_beta_res {
                    let mut hv = 0.0;
                    for j in 0..11 {
                        hv += h11[i][j] * v11[j];
                    }
                    gi += self.rayleigh_beta * hv;
                }
                if gi == 0.0 {
                    continue;
                }
                match map11[i] {
                    StencilDof::Interior(idx) => grad_i[idx] += gi,
                    StencilDof::BodyA(jrow) => {
                        for c in 0..6 {
                            grad_a.0[c] += jrow[c] * gi;
                        }
                    }
                    StencilDof::BodyB(jrow) => {
                        for c in 0..6 {
                            grad_b.0[c] += jrow[c] * gi;
                        }
                    }
                }
            }
            // Hessian scatter (scaled by c_k)
            for i in 0..11 {
                for j in 0..11 {
                    let hij = h11[i][j] * c_k;
                    if hij == 0.0 {
                        continue;
                    }
                    match (map11[i], map11[j]) {
                        (StencilDof::Interior(ri), StencilDof::Interior(cj)) => {
                            banded.add_from_full_sym(ri, cj, hij);
                        }
                        (StencilDof::Interior(ri), StencilDof::BodyA(jrow)) => {
                            for c in 0..6 {
                                h_ia[c][ri] += hij * jrow[c];
                            }
                        }
                        (StencilDof::Interior(ri), StencilDof::BodyB(jrow)) => {
                            for c in 0..6 {
                                h_ib[c][ri] += hij * jrow[c];
                            }
                        }
                        (StencilDof::BodyA(ra), StencilDof::BodyA(ca)) => {
                            for r in 0..6 {
                                for c in 0..6 {
                                    h_aa.0[r][c] += ra[r] * hij * ca[c];
                                }
                            }
                        }
                        (StencilDof::BodyA(ra), StencilDof::BodyB(cb)) => {
                            for r in 0..6 {
                                for c in 0..6 {
                                    h_ab.0[r][c] += ra[r] * hij * cb[c];
                                }
                            }
                        }
                        (StencilDof::BodyB(rb), StencilDof::BodyB(cb)) => {
                            for r in 0..6 {
                                for c in 0..6 {
                                    h_bb.0[r][c] += rb[r] * hij * cb[c];
                                }
                            }
                        }
                        // Symmetric counterparts handled by (Interior, BodyA/B) and (BodyA, BodyB)
                        (StencilDof::BodyA(_), StencilDof::Interior(_))
                        | (StencilDof::BodyB(_), StencilDof::Interior(_))
                        | (StencilDof::BodyB(_), StencilDof::BodyA(_)) => {}
                    }
                }
            }
        };

        let make_vert_dofs = |v: usize| -> [StencilDof; 3] {
            if v == 0 {
                [
                    StencilDof::BodyA(jac_x0_a[0]),
                    StencilDof::BodyA(jac_x0_a[1]),
                    StencilDof::BodyA(jac_x0_a[2]),
                ]
            } else if v <= n_free_verts {
                let s = Self::vert_dof_start(v);
                [
                    StencilDof::Interior(s),
                    StencilDof::Interior(s + 1),
                    StencilDof::Interior(s + 2),
                ]
            } else {
                [
                    StencilDof::BodyB(jac_xe_b[0]),
                    StencilDof::BodyB(jac_xe_b[1]),
                    StencilDof::BodyB(jac_xe_b[2]),
                ]
            }
        };

        // --- 2. Interior bending & twisting springs (vertices i = 1..n_elements-1) ---
        for i in 1..n_el {
            let vl = self.voronoi_lengths[i - 1];
            let rt = reference_twist(cur_u[i - 1], cur_t[i - 1], cur_u[i], cur_t[i], state.ref_twist[i - 1]);
            let be = der_bending_energy_grad_hess(
                pts[i - 1],
                pts[i],
                pts[i + 1],
                m1[i - 1],
                m2[i - 1],
                m1[i],
                m2[i],
                self.rest_kappa[i - 1],
                vl,
                self.ei,
            );
            let te = der_twisting_energy_grad_hess(
                pts[i - 1],
                pts[i],
                pts[i + 1],
                state.theta[i - 1],
                state.theta[i],
                rt,
                self.rest_twist[i - 1],
                vl,
                self.gj,
            );
            total_energy += be.energy + te.energy;

            let mut g11 = [0.0; 11];
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                g11[r] = be.grad[r] + te.grad[r];
                for c in 0..11 {
                    h11[r][c] = be.hess[r][c] + te.hess[r][c];
                }
            }
            if project_spd {
                h11 = project_spd_11x11(&h11, 0.0);
            }

            let d0 = make_vert_dofs(i - 1);
            let d1 = make_vert_dofs(i);
            let d2 = make_vert_dofs(i + 1);
            let map11 = [
                d0[0],
                d0[1],
                d0[2],
                d1[0],
                d1[1],
                d1[2],
                d2[0],
                d2[1],
                d2[2],
                StencilDof::Interior(Self::theta_dof(i - 1)),
                StencilDof::Interior(Self::theta_dof(i)),
            ];
            scatter_stencil11(
                &map11,
                &g11,
                &h11,
                &mut banded,
                &mut grad_i,
                &mut h_ia,
                &mut h_ib,
                &mut h_aa,
                &mut h_ab,
                &mut h_bb,
                &mut grad_a,
                &mut grad_b,
            );
        }

        // --- 3. Clamp A boundary bending-twisting spring (ghost vertex -1, vertex 0, vertex 1) ---
        {
            let x_m1 = pts[0] - self.rest_lengths[0] * ta_ghost;
            let rt_a = reference_twist(
                m1_a,
                ta_ghost,
                cur_u[0],
                cur_t[0],
                state.clamp_a_ref_twist,
            );
            let vl_a = self.rest_lengths[0];
            let be_a = der_bending_energy_grad_hess(
                x_m1,
                pts[0],
                pts[1],
                m1_a,
                m2_a,
                m1[0],
                m2[0],
                self.clamp_a_rest_kappa,
                vl_a,
                self.ei,
            );
            let te_a = der_twisting_energy_grad_hess(
                x_m1,
                pts[0],
                pts[1],
                0.0,
                state.theta[0],
                rt_a,
                self.clamp_a_rest_twist,
                vl_a,
                self.gj,
            );
            total_energy += be_a.energy + te_a.energy;

            let mut g11 = [0.0; 11];
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                g11[r] = be_a.grad[r] + te_a.grad[r];
                for c in 0..11 {
                    h11[r][c] = be_a.hess[r][c] + te_a.hess[r][c];
                }
            }
            if project_spd {
                h11 = project_spd_11x11(&h11, 0.0);
            }
            let d1 = make_vert_dofs(1);
            let th_a_row = [0.0, 0.0, 0.0, ta_ghost[0], ta_ghost[1], ta_ghost[2]];
            let map11 = [
                StencilDof::BodyA(jac_xm1_a[0]),
                StencilDof::BodyA(jac_xm1_a[1]),
                StencilDof::BodyA(jac_xm1_a[2]),
                StencilDof::BodyA(jac_x0_a[0]),
                StencilDof::BodyA(jac_x0_a[1]),
                StencilDof::BodyA(jac_x0_a[2]),
                d1[0],
                d1[1],
                d1[2],
                StencilDof::BodyA(th_a_row),
                StencilDof::Interior(Self::theta_dof(0)),
            ];
            scatter_stencil11(
                &map11,
                &g11,
                &h11,
                &mut banded,
                &mut grad_i,
                &mut h_ia,
                &mut h_ib,
                &mut h_aa,
                &mut h_ab,
                &mut h_bb,
                &mut grad_a,
                &mut grad_b,
            );
        }

        // --- 4. Clamp B boundary bending-twisting spring (when body_b is Some) ---
        if self.body_b.is_some() {
            let last = n_el - 1;
            let x_ep1 = pts[n_el] + self.rest_lengths[last] * tb_ghost;
            let rt_b = reference_twist(
                cur_u[last],
                cur_t[last],
                m1_b,
                tb_ghost,
                state.clamp_b_ref_twist,
            );
            let vl_b = self.rest_lengths[last];
            let be_b = der_bending_energy_grad_hess(
                pts[last],
                pts[n_el],
                x_ep1,
                m1[last],
                m2[last],
                m1_b,
                m2_b,
                self.clamp_b_rest_kappa,
                vl_b,
                self.ei,
            );
            let te_b = der_twisting_energy_grad_hess(
                pts[last],
                pts[n_el],
                x_ep1,
                state.theta[last],
                0.0,
                rt_b,
                self.clamp_b_rest_twist,
                vl_b,
                self.gj,
            );
            total_energy += be_b.energy + te_b.energy;

            let mut g11 = [0.0; 11];
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                g11[r] = be_b.grad[r] + te_b.grad[r];
                for c in 0..11 {
                    h11[r][c] = be_b.hess[r][c] + te_b.hess[r][c];
                }
            }
            if project_spd {
                h11 = project_spd_11x11(&h11, 0.0);
            }
            let d_prev = make_vert_dofs(last);
            let th_b_row = [0.0, 0.0, 0.0, tb_ghost[0], tb_ghost[1], tb_ghost[2]];
            let map11 = [
                d_prev[0],
                d_prev[1],
                d_prev[2],
                StencilDof::BodyB(jac_xe_b[0]),
                StencilDof::BodyB(jac_xe_b[1]),
                StencilDof::BodyB(jac_xe_b[2]),
                StencilDof::BodyB(jac_xep1_b[0]),
                StencilDof::BodyB(jac_xep1_b[1]),
                StencilDof::BodyB(jac_xep1_b[2]),
                StencilDof::Interior(Self::theta_dof(last)),
                StencilDof::BodyB(th_b_row),
            ];
            scatter_stencil11(
                &map11,
                &g11,
                &h11,
                &mut banded,
                &mut grad_i,
                &mut h_ia,
                &mut h_ib,
                &mut h_aa,
                &mut h_ab,
                &mut h_bb,
                &mut grad_a,
                &mut grad_b,
            );
        }

        // --- 5. Add lumped mass, contact stiffness, and build interior RHS ---
        let mut rhs_i = vec![0.0; n_dof];
        for v in 1..=n_free_verts {
            let s = Self::vert_dof_start(v);
            let m = self.free_vertex_mass[v - 1];
            let nid = map.free_node_ids[v - 1];
            let vel = vels[nid].lin();
            let f_ext = interior_ext_force
                .get(v - 1)
                .copied()
                .unwrap_or(Vec3::ZERO);
            let k_con = interior_contact_k
                .get(v - 1)
                .copied()
                .unwrap_or(Mat3::ZERO);
            for r in 0..3 {
                banded.add_lower(s + r, s + r, c_m * m);
                for c in 0..=r {
                    banded.add_lower(s + r, s + c, k_con.0[r][c]);
                }
                if is_velocity_form {
                    rhs_i[s + r] = m * (vel[r] + dt * gravity[r]) + dt * (f_ext[r] - grad_i[s + r]);
                } else {
                    let a_lin = accels.map_or(0.0, |aa| aa[nid][r]);
                    rhs_i[s + r] =
                        m * (gravity[r] - a_lin - self.rayleigh_alpha * vel[r]) + f_ext[r]
                            - grad_i[s + r];
                }
            }
        }
        for e in 0..n_el {
            let idx = Self::theta_dof(e);
            let j_rot = self.edge_twist_inertia[e];
            banded.add_lower(idx, idx, c_m * j_rot);
            if is_velocity_form {
                rhs_i[idx] = j_rot * state.theta_vel[e] - dt * grad_i[idx];
            } else {
                let a_th = theta_accels.and_then(|ta| ta.get(e).copied()).unwrap_or(0.0);
                rhs_i[idx] =
                    -j_rot * (a_th + self.rayleigh_alpha * state.theta_vel[e]) - grad_i[idx];
            }
        }

        // Lumped endpoint masses onto rigid clamps
        h_aa += jt_k_j_3x6(&jac_x0_a, Mat3::diag([c_m * self.end_a_mass; 3]), &jac_x0_a);
        if self.body_b.is_some() {
            h_bb += jt_k_j_3x6(&jac_xe_b, Mat3::diag([c_m * self.end_b_mass; 3]), &jac_xe_b);
        }

        // --- 6. Capture interior residuals and factor banded H_II in O(E) ---
        let mut res_verts = Vec::with_capacity(n_free_verts);
        for v in 1..=n_free_verts {
            let s = Self::vert_dof_start(v);
            res_verts.push(Vec3::new(-rhs_i[s], -rhs_i[s + 1], -rhs_i[s + 2]));
        }
        let mut res_thetas = Vec::with_capacity(n_el);
        for e in 0..n_el {
            res_thetas.push(-rhs_i[Self::theta_dof(e)]);
        }

        banded.factorize(self.beam_id)?;

        let mut z0 = rhs_i;
        banded.solve_in_place(&mut z0);

        let mut z_a: [Vec<f64>; 6] = std::array::from_fn(|c| h_ia[c].clone());
        for col in &mut z_a {
            banded.solve_in_place(col);
        }
        let z_b: [Vec<f64>; 6] = if is_cantilever {
            std::array::from_fn(|_| Vec::new())
        } else {
            let mut zb: [Vec<f64>; 6] = std::array::from_fn(|c| h_ib[c].clone());
            for col in &mut zb {
                banded.solve_in_place(col);
            }
            zb
        };

        // Condensed Schur blocks: S_aa = H_aa - H_ia^T Z_a, etc.
        let mut s_aa = h_aa;
        let mut s_ab = h_ab;
        let mut s_bb = h_bb;
        let mut r_a = Vec6::ZERO;
        let mut r_b = Vec6::ZERO;

        for c1 in 0..6 {
            let mut ra_dot = 0.0;
            for k in 0..n_dof {
                ra_dot += h_ia[c1][k] * z0[k];
            }
            r_a.0[c1] = -ra_dot;
            for c2 in 0..6 {
                let mut dot_aa = 0.0;
                for k in 0..n_dof {
                    dot_aa += h_ia[c1][k] * z_a[c2][k];
                }
                s_aa.0[c1][c2] -= dot_aa;
            }
        }
        if !is_cantilever {
            for c1 in 0..6 {
                let mut rb_dot = 0.0;
                for k in 0..n_dof {
                    rb_dot += h_ib[c1][k] * z0[k];
                }
                r_b.0[c1] = -rb_dot;
                for c2 in 0..6 {
                    let mut dot_ab = 0.0;
                    let mut dot_bb = 0.0;
                    for k in 0..n_dof {
                        dot_ab += h_ia[c1][k] * z_b[c2][k];
                        dot_bb += h_ib[c1][k] * z_b[c2][k];
                    }
                    s_ab.0[c1][c2] -= dot_ab;
                    s_bb.0[c1][c2] -= dot_bb;
                }
            }
        }

        // Elastic endpoint wrenches (and lumped endpoint inertia/gravity)
        let mut end_a_elastic = -grad_a;
        let mut end_b_elastic = -grad_b;
        if is_velocity_form {
            let mom_a = self.end_a_mass * (va_pt_vel + dt * gravity);
            end_a_elastic = Vec6::from_parts(mom_a, off_a.cross(mom_a)) + dt * (-grad_a);
            if self.body_b.is_some() {
                let mom_b = self.end_b_mass * (vb_pt_vel + dt * gravity);
                end_b_elastic = Vec6::from_parts(mom_b, off_b.cross(mom_b)) + dt * (-grad_b);
            }
        } else {
            let aa_pt = accels.map_or(Vec3::ZERO, |aa| {
                aa[self.body_a].lin() + aa[self.body_a].ang().cross(off_a)
            });
            let fa_inert = self.end_a_mass * (gravity - aa_pt - self.rayleigh_alpha * va_pt_vel);
            end_a_elastic += Vec6::from_parts(fa_inert, off_a.cross(fa_inert));
            if let Some(bb) = self.body_b {
                let ab_pt = accels.map_or(Vec3::ZERO, |aa| {
                    aa[bb].lin() + aa[bb].ang().cross(off_b)
                });
                let fb_inert = self.end_b_mass * (gravity - ab_pt - self.rayleigh_alpha * vb_pt_vel);
                end_b_elastic += Vec6::from_parts(fb_inert, off_b.cross(fb_inert));
            }
        }

        let h_ib_stored = if is_cantilever {
            std::array::from_fn(|_| Vec::new())
        } else {
            h_ib
        };
        let factor = DerBeamFactor {
            beam_id: self.beam_id,
            n_elements: n_el,
            n_free_verts,
            is_cantilever,
            n_dof,
            banded,
            h_ia,
            h_ib: h_ib_stored,
            z0,
            z_a,
            z_b,
        };
        let contrib = BeamSchurContribution {
            beam_id: self.beam_id,
            body_a: self.body_a,
            body_b: self.body_b,
            s_aa,
            s_ab,
            s_ba: s_ab.transpose(),
            s_bb,
            r_a,
            r_b,
        };
        Ok(DerAssembleOutput {
            energy: total_energy,
            factor,
            contrib,
            end_a_rhs: end_a_elastic,
            end_b_rhs: end_b_elastic,
            res_verts,
            res_thetas,
        })
    }

    /// Evaluate exact $O(E)$ scalar elastic contractions `(lambda^T * grad_E, lambda^T * H_E * y_next)`
    /// for IFT reverse-mode adjoints without allocating or factoring the banded matrix.
    ///
    /// Matches [`Self::assemble_and_condense`] bit-for-bit on stretching, interior bending/twisting,
    /// and rigid clamp A/B boundary stencils (including `project_spd`).
    #[allow(clippy::too_many_arguments)]
    pub fn eval_elastic_contraction(
        &self,
        map: &BeamDofMap,
        poses: &[Se3],
        state: &DerBeamState,
        lambda_nodes: &[Vec6],
        lambda_thetas: &[f64],
        v_next_nodes: &[Vec6],
        v_next_thetas: &[f64],
        project_spd: bool,
    ) -> (f64, f64) {
        self.eval_elastic_contraction_subset(
            map,
            poses,
            state,
            lambda_nodes,
            lambda_thetas,
            v_next_nodes,
            v_next_thetas,
            project_spd,
            None,
            0..self.n_elements,
            1..self.n_elements,
            true,
            self.body_b.is_some(),
        )
    }

    /// Precompute unperturbed SPD projections and exact Daleckii-Krein cotangents across all
    /// DER stencils at `(poses, state)` so pose/twist VJPs differentiate only the smooth
    /// unprojected analytical Hessians (zero zero-eigenvalue kink straddling, 5-10x faster).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_elastic_vjp_cache(
        &self,
        map: &BeamDofMap,
        poses: &[Se3],
        state: &DerBeamState,
        lambda_nodes: &[Vec6],
        lambda_thetas: &[f64],
        v_next_nodes: &[Vec6],
        v_next_thetas: &[f64],
    ) -> DerElasticVjpCache {
        let n_el = self.n_elements;
        let n_free_verts = map.free_node_ids.len();

        let xa = poses[self.body_a];
        let off_a = xa.rot.rotate(self.clamp_a.pos);
        let pos_a = xa.pos + off_a;
        let q_ca = xa.rot.mul(self.clamp_a.rot);
        let ta_ghost = q_ca.rotate(Vec3::EX);
        let m1_a = q_ca.rotate(Vec3::EY);
        let m2_a = q_ca.rotate(Vec3::EZ);
        let off_a_ghost = off_a - self.rest_lengths[0] * ta_ghost;

        let (pos_b, off_b, tb_ghost, m1_b, m2_b, off_b_ghost) = if let Some(bb) = self.body_b {
            let xb = poses[bb];
            let cb = self.clamp_b.unwrap_or(Se3::IDENTITY);
            let ob = xb.rot.rotate(cb.pos);
            let q_cb = xb.rot.mul(cb.rot);
            let tb = q_cb.rotate(Vec3::EX);
            (
                xb.pos + ob,
                ob,
                tb,
                q_cb.rotate(Vec3::EY),
                q_cb.rotate(Vec3::EZ),
                ob + self.rest_lengths[n_el - 1] * tb,
            )
        } else {
            (
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::EX,
                Vec3::EY,
                Vec3::EZ,
                Vec3::ZERO,
            )
        };

        let vert_pos = |v: usize| -> Vec3 {
            if v == 0 {
                pos_a
            } else if v <= n_free_verts {
                poses[map.free_node_ids[v - 1]].pos
            } else {
                pos_b
            }
        };

        let edge_frame = |e: usize| -> (Vec3, Vec3, Vec3, Vec3) {
            let p0 = vert_pos(e);
            let p1 = vert_pos(e + 1);
            let t = (p1 - p0).normalized_or(state.bishop_t[e]);
            let u = parallel_transport(state.bishop_u[e], state.bishop_t[e], t);
            let v = t.cross(u);
            let (d1, d2) = material_frame(t, u, v, state.theta[e]);
            (t, u, d1, d2)
        };

        let lam_a = lambda_nodes[self.body_a];
        let vn_a = v_next_nodes[self.body_a];
        let lam_b = self.body_b.map_or(Vec6::ZERO, |bb| lambda_nodes[bb]);
        let vn_b = self.body_b.map_or(Vec6::ZERO, |bb| v_next_nodes[bb]);

        let lam_pt_a = lam_a.lin() + lam_a.ang().cross(off_a);
        let vn_pt_a = vn_a.lin() + vn_a.ang().cross(off_a);
        let lam_pt_am1 = lam_a.lin() + lam_a.ang().cross(off_a_ghost);
        let vn_pt_am1 = vn_a.lin() + vn_a.ang().cross(off_a_ghost);
        let lam_pt_b = lam_b.lin() + lam_b.ang().cross(off_b);
        let vn_pt_b = vn_b.lin() + vn_b.ang().cross(off_b);
        let lam_pt_bp1 = lam_b.lin() + lam_b.ang().cross(off_b_ghost);
        let vn_pt_bp1 = vn_b.lin() + vn_b.ang().cross(off_b_ghost);

        let vert_lam_vn = |v: usize| -> (Vec3, Vec3) {
            if v == 0 {
                (lam_pt_a, vn_pt_a)
            } else if v <= n_free_verts {
                let nid = map.free_node_ids[v - 1];
                (lambda_nodes[nid].lin(), v_next_nodes[nid].lin())
            } else {
                (lam_pt_b, vn_pt_b)
            }
        };

        // 1. Stretching compression clamping flags at base point
        let mut stretch_clamp_zero = Vec::with_capacity(n_el);
        for e in 0..n_el {
            let l = (vert_pos(e + 1) - vert_pos(e)).norm().max(1e-15);
            let strain = l / self.rest_lengths[e] - 1.0;
            stretch_clamp_zero.push(self.ea * strain / l <= 0.0);
        }

        // 2. Interior hinges
        let mut interior_spd = Vec::with_capacity(n_el.saturating_sub(1));
        for i in 1..n_el {
            let vl = self.voronoi_lengths[i - 1];
            let (t_prev, u_prev, m1_prev, m2_prev) = edge_frame(i - 1);
            let (t_curr, u_curr, m1_curr, m2_curr) = edge_frame(i);
            let p_prev = vert_pos(i - 1);
            let p_curr = vert_pos(i);
            let p_next = vert_pos(i + 1);
            let rt = reference_twist(u_prev, t_prev, u_curr, t_curr, state.ref_twist[i - 1]);
            let be = der_bending_energy_grad_hess(
                p_prev,
                p_curr,
                p_next,
                m1_prev,
                m2_prev,
                m1_curr,
                m2_curr,
                self.rest_kappa[i - 1],
                vl,
                self.ei,
            );
            let te = der_twisting_energy_grad_hess(
                p_prev,
                p_curr,
                p_next,
                state.theta[i - 1],
                state.theta[i],
                rt,
                self.rest_twist[i - 1],
                vl,
                self.gj,
            );
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                for c in 0..11 {
                    h11[r][c] = be.hess[r][c] + te.hess[r][c];
                }
            }
            let (l0, v0) = vert_lam_vn(i - 1);
            let (l1, v1) = vert_lam_vn(i);
            let (l2, v2) = vert_lam_vn(i + 1);
            let l11 = [
                l0[0],
                l0[1],
                l0[2],
                l1[0],
                l1[1],
                l1[2],
                l2[0],
                l2[1],
                l2[2],
                lambda_thetas[i - 1],
                lambda_thetas[i],
            ];
            let v11 = [
                v0[0],
                v0[1],
                v0[2],
                v1[0],
                v1[1],
                v1[2],
                v2[0],
                v2[1],
                v2[2],
                v_next_thetas[i - 1],
                v_next_thetas[i],
            ];
            interior_spd.push(daleckii_krein_spd_11x11(&h11, &l11, &v11));
        }

        // 3. Clamp A boundary
        let clamp_a_spd = {
            let p0 = vert_pos(0);
            let p1 = vert_pos(1);
            let x_m1 = p0 - self.rest_lengths[0] * ta_ghost;
            let (t0, u0, m1_0, m2_0) = edge_frame(0);
            let rt_a = reference_twist(m1_a, ta_ghost, u0, t0, state.clamp_a_ref_twist);
            let vl_a = self.rest_lengths[0];
            let be_a = der_bending_energy_grad_hess(
                x_m1,
                p0,
                p1,
                m1_a,
                m2_a,
                m1_0,
                m2_0,
                self.clamp_a_rest_kappa,
                vl_a,
                self.ei,
            );
            let te_a = der_twisting_energy_grad_hess(
                x_m1,
                p0,
                p1,
                0.0,
                state.theta[0],
                rt_a,
                self.clamp_a_rest_twist,
                vl_a,
                self.gj,
            );
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                for c in 0..11 {
                    h11[r][c] = be_a.hess[r][c] + te_a.hess[r][c];
                }
            }
            let (l1, v1) = vert_lam_vn(1);
            let l11 = [
                lam_pt_am1[0],
                lam_pt_am1[1],
                lam_pt_am1[2],
                lam_pt_a[0],
                lam_pt_a[1],
                lam_pt_a[2],
                l1[0],
                l1[1],
                l1[2],
                lam_a.ang().dot(ta_ghost),
                lambda_thetas[0],
            ];
            let v11 = [
                vn_pt_am1[0],
                vn_pt_am1[1],
                vn_pt_am1[2],
                vn_pt_a[0],
                vn_pt_a[1],
                vn_pt_a[2],
                v1[0],
                v1[1],
                v1[2],
                vn_a.ang().dot(ta_ghost),
                v_next_thetas[0],
            ];
            daleckii_krein_spd_11x11(&h11, &l11, &v11)
        };

        // 4. Clamp B boundary
        let clamp_b_spd = self.body_b.map(|_| {
            let last = n_el - 1;
            let p_last = vert_pos(last);
            let p_end = vert_pos(n_el);
            let x_ep1 = p_end + self.rest_lengths[last] * tb_ghost;
            let (t_last, u_last, m1_last, m2_last) = edge_frame(last);
            let rt_b = reference_twist(u_last, t_last, m1_b, tb_ghost, state.clamp_b_ref_twist);
            let vl_b = self.rest_lengths[last];
            let be_b = der_bending_energy_grad_hess(
                p_last,
                p_end,
                x_ep1,
                m1_last,
                m2_last,
                m1_b,
                m2_b,
                self.clamp_b_rest_kappa,
                vl_b,
                self.ei,
            );
            let te_b = der_twisting_energy_grad_hess(
                p_last,
                p_end,
                x_ep1,
                state.theta[last],
                0.0,
                rt_b,
                self.clamp_b_rest_twist,
                vl_b,
                self.gj,
            );
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                for c in 0..11 {
                    h11[r][c] = be_b.hess[r][c] + te_b.hess[r][c];
                }
            }
            let (l_prev, v_prev) = vert_lam_vn(last);
            let l11 = [
                l_prev[0],
                l_prev[1],
                l_prev[2],
                lam_pt_b[0],
                lam_pt_b[1],
                lam_pt_b[2],
                lam_pt_bp1[0],
                lam_pt_bp1[1],
                lam_pt_bp1[2],
                lambda_thetas[last],
                lam_b.ang().dot(tb_ghost),
            ];
            let v11 = [
                v_prev[0],
                v_prev[1],
                v_prev[2],
                vn_pt_b[0],
                vn_pt_b[1],
                vn_pt_b[2],
                vn_pt_bp1[0],
                vn_pt_bp1[1],
                vn_pt_bp1[2],
                v_next_thetas[last],
                vn_b.ang().dot(tb_ghost),
            ];
            daleckii_krein_spd_11x11(&h11, &l11, &v11)
        });

        DerElasticVjpCache {
            stretch_clamp_zero,
            interior_spd,
            clamp_a_spd,
            clamp_b_spd,
        }
    }

    /// Evaluate exact zero-allocation elastic contractions `(lambda^T * grad_E, lambda^T * H_E * y_next)`
    /// over a caller-specified subset of stretching edges, interior hinges, and clamp boundaries
    /// (enables strictly $O(E)$ DER elastic pose/twist/Bishop-frame VJPs in `adjoint.rs`!).
    ///
    /// When `spd_cache` is `Some(cache)`, evaluates the exact smooth Daleckii-Krein surrogate
    /// `l(x)^T H_{+,0} v(x) + tr(G_{H,0} H(x))` so differentiation never straddles `max(lam, 0)` kinks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_elastic_contraction_subset(
        &self,
        map: &BeamDofMap,
        poses: &[Se3],
        state: &DerBeamState,
        lambda_nodes: &[Vec6],
        lambda_thetas: &[f64],
        v_next_nodes: &[Vec6],
        v_next_thetas: &[f64],
        project_spd: bool,
        spd_cache: Option<&DerElasticVjpCache>,
        stretch_range: std::ops::Range<usize>,
        interior_range: std::ops::Range<usize>,
        include_clamp_a: bool,
        include_clamp_b: bool,
    ) -> (f64, f64) {
        let n_el = self.n_elements;
        let n_free_verts = map.free_node_ids.len();

        let xa = poses[self.body_a];
        let off_a = xa.rot.rotate(self.clamp_a.pos);
        let pos_a = xa.pos + off_a;
        let q_ca = xa.rot.mul(self.clamp_a.rot);
        let ta_ghost = q_ca.rotate(Vec3::EX);
        let m1_a = q_ca.rotate(Vec3::EY);
        let m2_a = q_ca.rotate(Vec3::EZ);
        let off_a_ghost = off_a - self.rest_lengths[0] * ta_ghost;

        let (pos_b, off_b, tb_ghost, m1_b, m2_b, off_b_ghost) = if let Some(bb) = self.body_b {
            let xb = poses[bb];
            let cb = self.clamp_b.unwrap_or(Se3::IDENTITY);
            let ob = xb.rot.rotate(cb.pos);
            let q_cb = xb.rot.mul(cb.rot);
            let tb = q_cb.rotate(Vec3::EX);
            (
                xb.pos + ob,
                ob,
                tb,
                q_cb.rotate(Vec3::EY),
                q_cb.rotate(Vec3::EZ),
                ob + self.rest_lengths[n_el - 1] * tb,
            )
        } else {
            (
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::EX,
                Vec3::EY,
                Vec3::EZ,
                Vec3::ZERO,
            )
        };

        let vert_pos = |v: usize| -> Vec3 {
            if v == 0 {
                pos_a
            } else if v <= n_free_verts {
                poses[map.free_node_ids[v - 1]].pos
            } else {
                pos_b
            }
        };

        let edge_frame = |e: usize| -> (Vec3, Vec3, Vec3, Vec3) {
            let p0 = vert_pos(e);
            let p1 = vert_pos(e + 1);
            let t = (p1 - p0).normalized_or(state.bishop_t[e]);
            let u = parallel_transport(state.bishop_u[e], state.bishop_t[e], t);
            let v = t.cross(u);
            let (d1, d2) = material_frame(t, u, v, state.theta[e]);
            (t, u, d1, d2)
        };

        let lam_a = lambda_nodes[self.body_a];
        let vn_a = v_next_nodes[self.body_a];
        let lam_b = self.body_b.map_or(Vec6::ZERO, |bb| lambda_nodes[bb]);
        let vn_b = self.body_b.map_or(Vec6::ZERO, |bb| v_next_nodes[bb]);

        let lam_pt_a = lam_a.lin() + lam_a.ang().cross(off_a);
        let vn_pt_a = vn_a.lin() + vn_a.ang().cross(off_a);
        let lam_pt_am1 = lam_a.lin() + lam_a.ang().cross(off_a_ghost);
        let vn_pt_am1 = vn_a.lin() + vn_a.ang().cross(off_a_ghost);

        let lam_pt_b = lam_b.lin() + lam_b.ang().cross(off_b);
        let vn_pt_b = vn_b.lin() + vn_b.ang().cross(off_b);
        let lam_pt_bp1 = lam_b.lin() + lam_b.ang().cross(off_b_ghost);
        let vn_pt_bp1 = vn_b.lin() + vn_b.ang().cross(off_b_ghost);

        let vert_lam_vn = |v: usize| -> (Vec3, Vec3) {
            if v == 0 {
                (lam_pt_a, vn_pt_a)
            } else if v <= n_free_verts {
                let nid = map.free_node_ids[v - 1];
                (lambda_nodes[nid].lin(), v_next_nodes[nid].lin())
            } else {
                (lam_pt_b, vn_pt_b)
            }
        };

        let mut grad_dot = 0.0_f64;
        let mut hess_dot = 0.0_f64;

        // 1. Stretching springs
        for e in stretch_range {
            let p0 = vert_pos(e);
            let p1 = vert_pos(e + 1);
            let (l0, v0) = vert_lam_vn(e);
            let (l1, v1) = vert_lam_vn(e + 1);
            let l6 = Vec6::from_parts(l0, l1);
            let v6 = Vec6::from_parts(v0, v1);
            if let Some(cache) = spd_cache {
                let se = der_stretching_energy_grad_hess(p0, p1, self.rest_lengths[e], self.ea, false);
                grad_dot += l6.dot(se.grad);
                if cache.stretch_clamp_zero[e] {
                    let edge = p1 - p0;
                    let len = edge.norm().max(1e-15);
                    let t = edge / len;
                    let k_par = self.ea / self.rest_lengths[e];
                    let dl = (l1 - l0).dot(t);
                    let dv = (v1 - v0).dot(t);
                    hess_dot += k_par * dl * dv;
                } else {
                    hess_dot += l6.dot(se.hess.mul_vec(v6));
                }
            } else {
                let se = der_stretching_energy_grad_hess(
                    p0,
                    p1,
                    self.rest_lengths[e],
                    self.ea,
                    project_spd,
                );
                grad_dot += l6.dot(se.grad);
                hess_dot += l6.dot(se.hess.mul_vec(v6));
            }
        }

        let accum_stencil11 = |l11: &[f64; 11],
                               v11: &[f64; 11],
                               g11: &[f64; 11],
                               h11: &Mat11,
                               cached_spd: Option<&SpdDaleckiiPair>,
                               grad_dot: &mut f64,
                               hess_dot: &mut f64| {
            if let Some((h_plus_0, g_h_0)) = cached_spd {
                for r in 0..11 {
                    let lr = l11[r];
                    if lr != 0.0 {
                        *grad_dot += lr * g11[r];
                        let mut hv = 0.0;
                        for c in 0..11 {
                            hv += h_plus_0[r][c] * v11[c];
                        }
                        *hess_dot += lr * hv;
                    }
                    let mut tr_row = 0.0;
                    for c in 0..11 {
                        tr_row += g_h_0[r][c] * h11[r][c];
                    }
                    *hess_dot += tr_row;
                }
            } else {
                for r in 0..11 {
                    let lr = l11[r];
                    if lr == 0.0 {
                        continue;
                    }
                    *grad_dot += lr * g11[r];
                    let mut hv = 0.0;
                    for c in 0..11 {
                        hv += h11[r][c] * v11[c];
                    }
                    *hess_dot += lr * hv;
                }
            }
        };

        // 2. Interior bending & twisting stencils
        for i in interior_range {
            let vl = self.voronoi_lengths[i - 1];
            let (t_prev, u_prev, m1_prev, m2_prev) = edge_frame(i - 1);
            let (t_curr, u_curr, m1_curr, m2_curr) = edge_frame(i);
            let p_prev = vert_pos(i - 1);
            let p_curr = vert_pos(i);
            let p_next = vert_pos(i + 1);
            let rt = reference_twist(u_prev, t_prev, u_curr, t_curr, state.ref_twist[i - 1]);
            let be = der_bending_energy_grad_hess(
                p_prev,
                p_curr,
                p_next,
                m1_prev,
                m2_prev,
                m1_curr,
                m2_curr,
                self.rest_kappa[i - 1],
                vl,
                self.ei,
            );
            let te = der_twisting_energy_grad_hess(
                p_prev,
                p_curr,
                p_next,
                state.theta[i - 1],
                state.theta[i],
                rt,
                self.rest_twist[i - 1],
                vl,
                self.gj,
            );
            let mut g11 = [0.0; 11];
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                g11[r] = be.grad[r] + te.grad[r];
                for c in 0..11 {
                    h11[r][c] = be.hess[r][c] + te.hess[r][c];
                }
            }
            let cached = spd_cache.map(|c| &c.interior_spd[i - 1]);
            if project_spd && cached.is_none() {
                h11 = project_spd_11x11(&h11, 0.0);
            }
            let (l0, v0) = vert_lam_vn(i - 1);
            let (l1, v1) = vert_lam_vn(i);
            let (l2, v2) = vert_lam_vn(i + 1);
            let l11 = [
                l0[0],
                l0[1],
                l0[2],
                l1[0],
                l1[1],
                l1[2],
                l2[0],
                l2[1],
                l2[2],
                lambda_thetas[i - 1],
                lambda_thetas[i],
            ];
            let v11 = [
                v0[0],
                v0[1],
                v0[2],
                v1[0],
                v1[1],
                v1[2],
                v2[0],
                v2[1],
                v2[2],
                v_next_thetas[i - 1],
                v_next_thetas[i],
            ];
            accum_stencil11(&l11, &v11, &g11, &h11, cached, &mut grad_dot, &mut hess_dot);
        }

        // 3. Clamp A boundary bending-twisting stencil
        if include_clamp_a {
            let p0 = vert_pos(0);
            let p1 = vert_pos(1);
            let x_m1 = p0 - self.rest_lengths[0] * ta_ghost;
            let (t0, u0, m1_0, m2_0) = edge_frame(0);
            let rt_a = reference_twist(m1_a, ta_ghost, u0, t0, state.clamp_a_ref_twist);
            let vl_a = self.rest_lengths[0];
            let be_a = der_bending_energy_grad_hess(
                x_m1,
                p0,
                p1,
                m1_a,
                m2_a,
                m1_0,
                m2_0,
                self.clamp_a_rest_kappa,
                vl_a,
                self.ei,
            );
            let te_a = der_twisting_energy_grad_hess(
                x_m1,
                p0,
                p1,
                0.0,
                state.theta[0],
                rt_a,
                self.clamp_a_rest_twist,
                vl_a,
                self.gj,
            );
            let mut g11 = [0.0; 11];
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                g11[r] = be_a.grad[r] + te_a.grad[r];
                for c in 0..11 {
                    h11[r][c] = be_a.hess[r][c] + te_a.hess[r][c];
                }
            }
            let cached = spd_cache.map(|c| &c.clamp_a_spd);
            if project_spd && cached.is_none() {
                h11 = project_spd_11x11(&h11, 0.0);
            }
            let (l1, v1) = vert_lam_vn(1);
            let l11 = [
                lam_pt_am1[0],
                lam_pt_am1[1],
                lam_pt_am1[2],
                lam_pt_a[0],
                lam_pt_a[1],
                lam_pt_a[2],
                l1[0],
                l1[1],
                l1[2],
                lam_a.ang().dot(ta_ghost),
                lambda_thetas[0],
            ];
            let v11 = [
                vn_pt_am1[0],
                vn_pt_am1[1],
                vn_pt_am1[2],
                vn_pt_a[0],
                vn_pt_a[1],
                vn_pt_a[2],
                v1[0],
                v1[1],
                v1[2],
                vn_a.ang().dot(ta_ghost),
                v_next_thetas[0],
            ];
            accum_stencil11(&l11, &v11, &g11, &h11, cached, &mut grad_dot, &mut hess_dot);
        }

        // 4. Clamp B boundary bending-twisting stencil
        if include_clamp_b && self.body_b.is_some() {
            let last = n_el - 1;
            let p_last = vert_pos(last);
            let p_end = vert_pos(n_el);
            let x_ep1 = p_end + self.rest_lengths[last] * tb_ghost;
            let (t_last, u_last, m1_last, m2_last) = edge_frame(last);
            let rt_b = reference_twist(u_last, t_last, m1_b, tb_ghost, state.clamp_b_ref_twist);
            let vl_b = self.rest_lengths[last];
            let be_b = der_bending_energy_grad_hess(
                p_last,
                p_end,
                x_ep1,
                m1_last,
                m2_last,
                m1_b,
                m2_b,
                self.clamp_b_rest_kappa,
                vl_b,
                self.ei,
            );
            let te_b = der_twisting_energy_grad_hess(
                p_last,
                p_end,
                x_ep1,
                state.theta[last],
                0.0,
                rt_b,
                self.clamp_b_rest_twist,
                vl_b,
                self.gj,
            );
            let mut g11 = [0.0; 11];
            let mut h11 = [[0.0; 11]; 11];
            for r in 0..11 {
                g11[r] = be_b.grad[r] + te_b.grad[r];
                for c in 0..11 {
                    h11[r][c] = be_b.hess[r][c] + te_b.hess[r][c];
                }
            }
            let cached = spd_cache.and_then(|c| c.clamp_b_spd.as_ref());
            if project_spd && cached.is_none() {
                h11 = project_spd_11x11(&h11, 0.0);
            }
            let (l_prev, v_prev) = vert_lam_vn(last);
            let l11 = [
                l_prev[0],
                l_prev[1],
                l_prev[2],
                lam_pt_b[0],
                lam_pt_b[1],
                lam_pt_b[2],
                lam_pt_bp1[0],
                lam_pt_bp1[1],
                lam_pt_bp1[2],
                lambda_thetas[last],
                lam_b.ang().dot(tb_ghost),
            ];
            let v11 = [
                v_prev[0],
                v_prev[1],
                v_prev[2],
                vn_pt_b[0],
                vn_pt_b[1],
                vn_pt_b[2],
                vn_pt_bp1[0],
                vn_pt_bp1[1],
                vn_pt_bp1[2],
                v_next_thetas[last],
                vn_b.ang().dot(tb_ghost),
            ];
            accum_stencil11(&l11, &v11, &g11, &h11, cached, &mut grad_dot, &mut hess_dot);
        }

        (grad_dot, hess_dot)
    }
}

impl DerBeamFactor {
    /// Condense an arbitrary new interior vertex translational RHS vector using cached
    /// banded factors (zero re-factorization, used for exact Delassus compliance solves).
    pub fn condense_rhs(&self, rhs_int_verts: &[Vec3]) -> (Vec<f64>, Vec6, Vec6) {
        self.condense_rhs_full(rhs_int_verts, &[])
    }

    /// Condense both interior vertex translational RHS and edge twist RHS using cached
    /// banded factors (zero re-factorization, used for exact DER IFT reverse-mode adjoints).
    pub fn condense_rhs_full(
        &self,
        rhs_int_verts: &[Vec3],
        rhs_int_thetas: &[f64],
    ) -> (Vec<f64>, Vec6, Vec6) {
        let mut z0 = vec![0.0; self.n_dof];
        for (li, v) in rhs_int_verts.iter().take(self.n_free_verts).enumerate() {
            let s = DerBeamData::vert_dof_start(li + 1);
            z0[s] = v.x();
            z0[s + 1] = v.y();
            z0[s + 2] = v.z();
        }
        for (e, &t) in rhs_int_thetas.iter().take(self.n_elements).enumerate() {
            z0[DerBeamData::theta_dof(e)] = t;
        }
        self.banded.solve_in_place(&mut z0);
        let mut r_a = Vec6::ZERO;
        let mut r_b = Vec6::ZERO;
        for c in 0..6 {
            let mut dot_a = 0.0;
            for k in 0..self.n_dof {
                dot_a += self.h_ia[c][k] * z0[k];
            }
            r_a.0[c] = -dot_a;
        }
        if !self.is_cantilever {
            for c in 0..6 {
                let mut dot_b = 0.0;
                for k in 0..self.n_dof {
                    dot_b += self.h_ib[c][k] * z0[k];
                }
                r_b.0[c] = -dot_b;
            }
        }
        (z0, r_a, r_b)
    }

    /// Recover all free vertex velocities/increments (`Vec<Vec3>`) and edge twist
    /// velocities/increments (`Vec<f64>`) given rigid endpoint twists `v_a`, `v_b`.
    pub fn back_substitute(&self, v_a: Vec6, v_b: Vec6) -> (Vec<Vec3>, Vec<f64>) {
        self.back_substitute_with_z0(&self.z0, v_a, v_b)
    }

    /// Back-substitute using a custom `z0` returned by [`Self::condense_rhs`].
    pub fn back_substitute_with_z0(
        &self,
        z0: &[f64],
        v_a: Vec6,
        v_b: Vec6,
    ) -> (Vec<Vec3>, Vec<f64>) {
        let mut y = z0.to_vec();
        for c in 0..6 {
            let va_c = v_a[c];
            if va_c != 0.0 {
                for k in 0..self.n_dof {
                    y[k] -= self.z_a[c][k] * va_c;
                }
            }
        }
        if !self.is_cantilever {
            for c in 0..6 {
                let vb_c = v_b[c];
                if vb_c != 0.0 {
                    for k in 0..self.n_dof {
                        y[k] -= self.z_b[c][k] * vb_c;
                    }
                }
            }
        }

        let mut vert_sol = Vec::with_capacity(self.n_free_verts);
        for v in 1..=self.n_free_verts {
            let s = DerBeamData::vert_dof_start(v);
            vert_sol.push(Vec3::new(y[s], y[s + 1], y[s + 2]));
        }
        let mut theta_sol = Vec::with_capacity(self.n_elements);
        for e in 0..self.n_elements {
            theta_sol.push(y[DerBeamData::theta_dof(e)]);
        }
        (vert_sol, theta_sol)
    }

    /// Solve `H_II * x = b` directly using cached banded factors.
    pub fn solve_interior_only(&self, b: &mut [f64]) {
        self.banded.solve_in_place(b);
    }
}

/// Compute `J_l^T * K * J_r` for `3x6` Jacobians and `3x3` matrix `K`.
fn jt_k_j_3x6(jl: &[[f64; 6]; 3], k: Mat3, jr: &[[f64; 6]; 3]) -> Mat6 {
    let mut kj = [[0.0; 6]; 3];
    for i in 0..3 {
        for c in 0..6 {
            kj[i][c] = k.0[i][0] * jr[0][c] + k.0[i][1] * jr[1][c] + k.0[i][2] * jr[2][c];
        }
    }
    let mut out = [[0.0; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            out[r][c] = jl[0][r] * kj[0][c] + jl[1][r] * kj[1][c] + jl[2][r] * kj[2][c];
        }
    }
    Mat6(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fg_ir::{RigidBodySpec, RigidFlexGraph};
    use fg_math::{BeamMaterial, BeamSection};

    #[test]
    fn test_corotational_energy_gradient_matches_finite_difference() {
        let spec = BeamEdgeSpec {
            id: 0,
            name: "strip".into(),
            body_a: 0,
            clamp_a: Se3::new(Vec3::new(0.01, 0.02, -0.005), Quat::from_axis_angle(Vec3::EY, 0.15)),
            body_b: Some(1),
            clamp_b: Some(Se3::new(
                Vec3::new(-0.01, 0.015, 0.005),
                Quat::from_axis_angle(Vec3::EY, -0.15),
            )),
            n_elements: 4,
            formulation: BeamFormulation::CorotationalEulerBernoulli,
            section: BeamSection::Rectangular {
                width: 0.016,
                thickness: 0.0002,
            },
            material: BeamMaterial::SPRING_STEEL,
            thin_shell_compat: true,
            prestress_frac: 0.0,
            centerline_rest_poses: (0..=4)
                .map(|i| {
                    let u = i as f64 / 4.0;
                    Se3::new(Vec3::new(u * 0.12, 0.02, 0.02 * 4.0 * u * (1.0 - u)), Quat::IDENTITY)
                })
                .collect(),
            element_lengths: Vec::new(),
            element_transforms: None,
            interior_node_contact_radius: None,
            enable_self_contact: false,
        };
        let graph = RigidFlexGraph {
            name: "fd_beam".into(),
            gravity: Vec3::ZERO,
            bodies: vec![
                RigidBodySpec {
                    id: 0,
                    name: "b0".into(),
                    is_fixed: false,
                    mass: 0.28,
                    inertia: Mat3::diag([1e-3, 1e-3, 1e-3]),
                    com_local: Vec3::ZERO,
                    rest_pose: Se3::IDENTITY,
                    collision_radius: 0.01,
                    rayleigh_alpha: 0.05,
                    tags: Vec::new(),
                },
                RigidBodySpec {
                    id: 1,
                    name: "b1".into(),
                    is_fixed: false,
                    mass: 0.28,
                    inertia: Mat3::diag([1e-3, 1e-3, 1e-3]),
                    com_local: Vec3::ZERO,
                    rest_pose: Se3::new(Vec3::new(0.12, 0.0, 0.0), Quat::IDENTITY),
                    collision_radius: 0.01,
                    rayleigh_alpha: 0.05,
                    tags: Vec::new(),
                },
            ],
            beams: vec![spec.clone()],
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let analysis = graph.analyze().unwrap();
        let map = &analysis.indexing.beam_maps[0];
        let mut rest_poses = vec![Se3::IDENTITY; analysis.indexing.total_nodes];
        rest_poses[0] = graph.bodies[0].rest_pose;
        rest_poses[1] = graph.bodies[1].rest_pose;
        for (i, &nid) in map.free_node_ids.iter().enumerate() {
            rest_poses[nid] = spec.centerline_rest_poses[i + 1];
        }
        let data = CorotationalBeamData::from_spec(&spec, map, &rest_poses, true);

        // Perturb all nodes nonlinearly
        let mut poses = rest_poses.clone();
        poses[1].pos += Vec3::new(-0.012, 0.008, 0.006);
        poses[1].rot = Quat::from_axis_angle(Vec3::new(0.3, 0.7, -0.5), 0.22);
        for (i, &nid) in map.free_node_ids.iter().enumerate() {
            poses[nid].pos += Vec3::new(0.002 * (i as f64 + 1.0), -0.003, 0.005);
            poses[nid].rot = Quat::from_axis_angle(Vec3::new(-0.2, 0.9, 0.1), 0.1 * (i as f64 + 1.0));
        }

        let eval = data.evaluate(map, &poses, true);
        let mut f_analytic = vec![Vec6::ZERO; analysis.indexing.total_nodes];
        for e in 0..data.n_elements {
            f_analytic[map.chain_node_ids[e]] += eval.elem_elastic[e].head();
            f_analytic[map.chain_node_ids[e + 1]] += eval.elem_elastic[e].tail();
        }

        let eps = 1e-6;
        let mut max_rel_err = 0.0_f64;
        for nid in 0..analysis.indexing.total_nodes {
            for d in 0..6 {
                let mut twist = Vec6::ZERO;
                twist[d] = 1.0;
                let mut p_plus = poses.clone();
                let mut p_minus = poses.clone();
                p_plus[nid] = poses[nid].integrate_decoupled(twist, eps);
                p_minus[nid] = poses[nid].integrate_decoupled(twist, -eps);
                let u_plus = data.evaluate(map, &p_plus, true).energy;
                let u_minus = data.evaluate(map, &p_minus, true).energy;
                let f_fd = -(u_plus - u_minus) / (2.0 * eps);
                let f_an = f_analytic[nid][d];
                let scale = f_fd.abs().max(f_an.abs()).max(1.0);
                max_rel_err = max_rel_err.max((f_fd - f_an).abs() / scale);
            }
        }
        assert!(
            max_rel_err < 1e-6,
            "corotational analytic strain Jacobian gradient FD rel err = {max_rel_err:e}"
        );
    }

    #[test]
    fn test_der_beam_condensation_and_rigid_invariance() {
        let spec = BeamEdgeSpec {
            id: 0,
            name: "catheter_der".into(),
            body_a: 0,
            clamp_a: Se3::IDENTITY,
            body_b: None, // cantilever tip!
            clamp_b: None,
            n_elements: 6,
            formulation: BeamFormulation::DiscreteElasticRod,
            section: BeamSection::Circular { radius: 0.002 },
            material: BeamMaterial::NITINOL,
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: (0..=6)
                .map(|i| Se3::new(Vec3::new(i as f64 * 0.025, 0.0, 0.0), Quat::IDENTITY))
                .collect(),
            element_lengths: Vec::new(),
            element_transforms: None,
            interior_node_contact_radius: None,
            enable_self_contact: false,
        };
        let graph = RigidFlexGraph {
            name: "der_cantilever".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![RigidBodySpec {
                id: 0,
                name: "base".into(),
                is_fixed: true,
                mass: 1.0,
                inertia: Mat3::diag([1e-2, 1e-2, 1e-2]),
                com_local: Vec3::ZERO,
                rest_pose: Se3::IDENTITY,
                collision_radius: 0.01,
                rayleigh_alpha: 0.05,
                tags: Vec::new(),
            }],
            beams: vec![spec.clone()],
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let analysis = graph.analyze().unwrap();
        let map = &analysis.indexing.beam_maps[0];
        let (der_data, mut der_state) = DerBeamData::from_spec(&spec);

        let mut poses = vec![Se3::IDENTITY; analysis.indexing.total_nodes];
        for (i, &nid) in map.free_node_ids.iter().enumerate() {
            poses[nid] = spec.centerline_rest_poses[i + 1];
        }
        // Deflect the tip and twist the rod
        poses[map.free_node_ids[5]].pos += Vec3::new(-0.005, 0.012, -0.018);
        der_state.theta[3] = 0.25;
        let vels = vec![Vec6::ZERO; analysis.indexing.total_nodes];
        let ext = vec![Vec3::ZERO; map.free_node_ids.len()];
        let con_k = vec![Mat3::ZERO; map.free_node_ids.len()];

        let dt = 0.001;
        let out = der_data
            .assemble_and_condense(
                map,
                &poses,
                &vels,
                None,
                None,
                &der_state,
                graph.gravity,
                &ext,
                &con_k,
                dt,
                1.0 + dt * der_data.rayleigh_alpha,
                dt * der_data.rayleigh_beta + dt * dt,
                true,
                true,
            )
            .expect("DER assemble ok");
        assert!(out.energy > 0.0 && out.energy.is_finite());
        assert!(out.contrib.s_aa.0[0][0] > 0.0);
        let (v_verts, v_thetas) = out.factor.back_substitute(Vec6::ZERO, Vec6::ZERO);
        assert_eq!(v_verts.len(), 6);
        assert_eq!(v_thetas.len(), 6);
        assert!(v_verts.iter().all(|v| v.norm().is_finite()));
        assert!(v_thetas.iter().all(|t| t.is_finite()));

        // Verify SPD projection clamps negative eigenvalues cleanly
        let mut indef = [[0.0; 11]; 11];
        for i in 0..11 {
            indef[i][i] = if i % 2 == 0 { 5.0 } else { -3.0 };
        }
        indef[0][1] = 2.0;
        indef[1][0] = 2.0;
        let spd = project_spd_11x11(&indef, 0.0);
        // Check x^T spd x >= 0 for random vectors
        for trial in 0..20 {
            let mut x = [0.0; 11];
            for (i, val) in x.iter_mut().enumerate() {
                *val = ((trial * 7 + i * 13) as f64).sin();
            }
            let mut q = 0.0;
            for i in 0..11 {
                for j in 0..11 {
                    q += x[i] * spd[i][j] * x[j];
                }
            }
            assert!(q >= -1e-12, "projected 11x11 matrix not PSD: q={q}");
        }

        // Verify Daleckii-Krein spectral cotangent of project_spd_11x11 vs central FD on indefinite H
        let l_vec: [f64; 11] = std::array::from_fn(|i| ((i as f64 + 1.0) * 0.7).sin());
        let v_vec: [f64; 11] = std::array::from_fn(|i| ((i as f64 + 2.0) * 1.1).cos());
        let (h_plus, g_h) = daleckii_krein_spd_11x11(&indef, &l_vec, &v_vec);
        let mut dir = [[0.0; 11]; 11];
        for i in 0..11 {
            for j in i..11 {
                let d = ((i * 5 + j * 3 + 1) as f64).sin();
                dir[i][j] = d;
                dir[j][i] = d;
            }
        }
        let mut an_deriv = 0.0;
        for i in 0..11 {
            for j in 0..11 {
                assert!((h_plus[i][j] - spd[i][j]).abs() < 1e-12);
                an_deriv += g_h[i][j] * dir[i][j];
            }
        }
        let eps = 1e-6;
        let eval_proj = |step: f64| -> f64 {
            let mut h_pert = indef;
            for i in 0..11 {
                for j in 0..11 {
                    h_pert[i][j] += step * dir[i][j];
                }
            }
            let hp = project_spd_11x11(&h_pert, 0.0);
            let mut sum = 0.0;
            for i in 0..11 {
                for j in 0..11 {
                    sum += l_vec[i] * hp[i][j] * v_vec[j];
                }
            }
            sum
        };
        let fd_deriv = (eval_proj(eps) - eval_proj(-eps)) / (2.0 * eps);
        let dk_rel = (an_deriv - fd_deriv).abs() / fd_deriv.abs().max(1.0);
        assert!(
            dk_rel < 1e-8,
            "Daleckii-Krein SPD derivative mismatch: an={an_deriv:e} fd={fd_deriv:e} rel={dk_rel:e}"
        );
    }
}
