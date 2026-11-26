//! Fixed-size FP64 linear algebra types (`Vec3`, `Vec6`, `Vec12`, `Mat3`, `Mat6`, `Mat12`),
//! direct small-block solvers (Cholesky / partial-pivoting LU), and symmetric Jacobi
//! eigendecomposition with SPD projection.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign};

/// 3D FP64 column vector `[x, y, z]`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Vec3(pub [f64; 3]);

impl Vec3 {
    pub const ZERO: Self = Self([0.0, 0.0, 0.0]);
    pub const EX: Self = Self([1.0, 0.0, 0.0]);
    pub const EY: Self = Self([0.0, 1.0, 0.0]);
    pub const EZ: Self = Self([0.0, 0.0, 1.0]);

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self([x, y, z])
    }

    #[inline]
    pub const fn x(self) -> f64 {
        self.0[0]
    }
    #[inline]
    pub const fn y(self) -> f64 {
        self.0[1]
    }
    #[inline]
    pub const fn z(self) -> f64 {
        self.0[2]
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1] + self.0[2] * rhs.0[2]
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self([
            self.0[1] * rhs.0[2] - self.0[2] * rhs.0[1],
            self.0[2] * rhs.0[0] - self.0[0] * rhs.0[2],
            self.0[0] * rhs.0[1] - self.0[1] * rhs.0[0],
        ])
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n > 1e-15 {
            self / n
        } else {
            self / 1e-15
        }
    }

    #[inline]
    pub fn normalized_or(self, fallback: Self) -> Self {
        let n = self.norm();
        if n > 1e-15 {
            self / n
        } else {
            fallback
        }
    }

    /// Skew-symmetric cross-product matrix `[v]_x` such that `[v]_x u = v x u`.
    #[inline]
    pub const fn skew(self) -> Mat3 {
        let [x, y, z] = self.0;
        Mat3([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    }

    /// Outer product `self * rhs^T`.
    #[inline]
    pub fn outer(self, rhs: Self) -> Mat3 {
        let a = self.0;
        let b = rhs.0;
        Mat3([
            [a[0] * b[0], a[0] * b[1], a[0] * b[2]],
            [a[1] * b[0], a[1] * b[1], a[1] * b[2]],
            [a[2] * b[0], a[2] * b[1], a[2] * b[2]],
        ])
    }

    #[inline]
    pub fn max_abs(self) -> f64 {
        self.0[0].abs().max(self.0[1].abs()).max(self.0[2].abs())
    }
}

impl From<[f64; 3]> for Vec3 {
    #[inline]
    fn from(v: [f64; 3]) -> Self {
        Self(v)
    }
}

impl From<Vec3> for [f64; 3] {
    #[inline]
    fn from(v: Vec3) -> Self {
        v.0
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0[0] += rhs.0[0];
        self.0[1] += rhs.0[1];
        self.0[2] += rhs.0[2];
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0[0] -= rhs.0[0];
        self.0[1] -= rhs.0[1];
        self.0[2] -= rhs.0[2];
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        let inv = 1.0 / s;
        Self([self.0[0] * inv, self.0[1] * inv, self.0[2] * inv])
    }
}

/// 6D spatial vector `[linear (0..3); angular (3..6)]`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Vec6(pub [f64; 6]);

impl Vec6 {
    pub const ZERO: Self = Self([0.0; 6]);

    #[inline]
    pub const fn from_parts(lin: Vec3, ang: Vec3) -> Self {
        Self([lin.0[0], lin.0[1], lin.0[2], ang.0[0], ang.0[1], ang.0[2]])
    }

    #[inline]
    pub const fn lin(self) -> Vec3 {
        Vec3([self.0[0], self.0[1], self.0[2]])
    }

    #[inline]
    pub const fn ang(self) -> Vec3 {
        Vec3([self.0[3], self.0[4], self.0[5]])
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        let mut s = 0.0;
        for i in 0..6 {
            s += self.0[i] * rhs.0[i];
        }
        s
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn outer(self, rhs: Self) -> Mat6 {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.0[i] * rhs.0[j];
            }
        }
        Mat6(out)
    }

    #[inline]
    pub fn max_abs(self) -> f64 {
        self.0.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    }
}

impl From<[f64; 6]> for Vec6 {
    #[inline]
    fn from(v: [f64; 6]) -> Self {
        Self(v)
    }
}

impl Index<usize> for Vec6 {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}

impl IndexMut<usize> for Vec6 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}

impl Add for Vec6 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = [0.0; 6];
        for i in 0..6 {
            out[i] = self.0[i] + rhs.0[i];
        }
        Self(out)
    }
}

impl AddAssign for Vec6 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            self.0[i] += rhs.0[i];
        }
    }
}

impl Sub for Vec6 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [0.0; 6];
        for i in 0..6 {
            out[i] = self.0[i] - rhs.0[i];
        }
        Self(out)
    }
}

impl SubAssign for Vec6 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            self.0[i] -= rhs.0[i];
        }
    }
}

impl Neg for Vec6 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut out = [0.0; 6];
        for i in 0..6 {
            out[i] = -self.0[i];
        }
        Self(out)
    }
}

impl Mul<f64> for Vec6 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        let mut out = [0.0; 6];
        for i in 0..6 {
            out[i] = self.0[i] * s;
        }
        Self(out)
    }
}

impl Mul<Vec6> for f64 {
    type Output = Vec6;
    #[inline]
    fn mul(self, v: Vec6) -> Vec6 {
        v * self
    }
}

/// 12D beam element vector `[node0 (0..6); node1 (6..12)]`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Vec12(pub [f64; 12]);

impl Vec12 {
    pub const ZERO: Self = Self([0.0; 12]);

    #[inline]
    pub fn from_halves(a: Vec6, b: Vec6) -> Self {
        let mut out = [0.0; 12];
        out[..6].copy_from_slice(&a.0);
        out[6..].copy_from_slice(&b.0);
        Self(out)
    }

    #[inline]
    pub fn head(self) -> Vec6 {
        Vec6([
            self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5],
        ])
    }

    #[inline]
    pub fn tail(self) -> Vec6 {
        Vec6([
            self.0[6], self.0[7], self.0[8], self.0[9], self.0[10], self.0[11],
        ])
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        let mut s = 0.0;
        for i in 0..12 {
            s += self.0[i] * rhs.0[i];
        }
        s
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn max_abs(self) -> f64 {
        self.0.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    }
}

impl Index<usize> for Vec12 {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}

impl IndexMut<usize> for Vec12 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}

impl Add for Vec12 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = [0.0; 12];
        for i in 0..12 {
            out[i] = self.0[i] + rhs.0[i];
        }
        Self(out)
    }
}

impl Sub for Vec12 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [0.0; 12];
        for i in 0..12 {
            out[i] = self.0[i] - rhs.0[i];
        }
        Self(out)
    }
}

impl Neg for Vec12 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut out = [0.0; 12];
        for i in 0..12 {
            out[i] = -self.0[i];
        }
        Self(out)
    }
}

impl Mul<f64> for Vec12 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        let mut out = [0.0; 12];
        for i in 0..12 {
            out[i] = self.0[i] * s;
        }
        Self(out)
    }
}

/// 3x3 row-major FP64 matrix.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    pub const ZERO: Self = Self([[0.0; 3]; 3]);
    pub const IDENTITY: Self = Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    #[inline]
    pub const fn diag(d: [f64; 3]) -> Self {
        Self([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]])
    }

    #[inline]
    pub const fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self([
            [c0.0[0], c1.0[0], c2.0[0]],
            [c0.0[1], c1.0[1], c2.0[1]],
            [c0.0[2], c1.0[2], c2.0[2]],
        ])
    }

    #[inline]
    pub const fn from_rows(r0: Vec3, r1: Vec3, r2: Vec3) -> Self {
        Self([r0.0, r1.0, r2.0])
    }

    #[inline]
    pub const fn col(self, j: usize) -> Vec3 {
        Vec3([self.0[0][j], self.0[1][j], self.0[2][j]])
    }

    #[inline]
    pub const fn row(self, i: usize) -> Vec3 {
        Vec3(self.0[i])
    }

    #[inline]
    pub const fn transpose(self) -> Self {
        let m = self.0;
        Self([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }

    #[inline]
    pub fn symmetrize(self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = 0.5 * (self.0[i][j] + self.0[j][i]);
            }
        }
        Self(out)
    }

    #[inline]
    pub fn trace(self) -> f64 {
        self.0[0][0] + self.0[1][1] + self.0[2][2]
    }

    #[inline]
    pub fn det(self) -> f64 {
        let m = self.0;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    #[inline]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        let m = self.0;
        Vec3([
            m[0][0] * v.0[0] + m[0][1] * v.0[1] + m[0][2] * v.0[2],
            m[1][0] * v.0[0] + m[1][1] * v.0[1] + m[1][2] * v.0[2],
            m[2][0] * v.0[0] + m[2][1] * v.0[1] + m[2][2] * v.0[2],
        ])
    }

    #[inline]
    pub fn mul_mat(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        let mut c = [[0.0; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                let aik = a[i][k];
                for j in 0..3 {
                    c[i][j] += aik * b[k][j];
                }
            }
        }
        Self(c)
    }

    #[inline]
    pub fn inv(self) -> Option<Self> {
        let m = self.0;
        let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
        let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
        let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
        if det.abs() < 1e-30 {
            return None;
        }
        let inv_det = 1.0 / det;
        let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
        let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
        let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
        let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
        let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        Some(Self([
            [c00 * inv_det, c10 * inv_det, c20 * inv_det],
            [c01 * inv_det, c11 * inv_det, c21 * inv_det],
            [c02 * inv_det, c12 * inv_det, c22 * inv_det],
        ]))
    }

    #[inline]
    pub fn max_abs(self) -> f64 {
        let mut m = 0.0_f64;
        for row in &self.0 {
            for &v in row {
                m = m.max(v.abs());
            }
        }
        m
    }

    /// Symmetric cyclic Jacobi eigendecomposition returning `(eigenvalues, eigenvectors_as_cols)`
    /// such that `self_sym == V * diag(evals) * V^T`.
    pub fn sym_eigen(self) -> ([f64; 3], Self) {
        let mut a = self.symmetrize().0;
        let mut v = Self::IDENTITY.0;
        for _ in 0..40 {
            let mut max_off = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let val = a[i][j].abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_off < 1e-15 {
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
            for r in 0..3 {
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
        ([a[0][0], a[1][1], a[2][2]], Self(v))
    }

    /// Project symmetric 3x3 matrix onto the SPD cone by clamping eigenvalues to `>= min_eig`.
    pub fn project_spd(self, min_eig: f64) -> Self {
        let (evals, v) = self.sym_eigen();
        let d = Self::diag([
            evals[0].max(min_eig),
            evals[1].max(min_eig),
            evals[2].max(min_eig),
        ]);
        v.mul_mat(d).mul_mat(v.transpose()).symmetrize()
    }
}

impl Add for Mat3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.0[i][j] + rhs.0[i][j];
            }
        }
        Self(out)
    }
}

impl AddAssign for Mat3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..3 {
            for j in 0..3 {
                self.0[i][j] += rhs.0[i][j];
            }
        }
    }
}

impl Sub for Mat3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.0[i][j] - rhs.0[i][j];
            }
        }
        Self(out)
    }
}

impl Neg for Mat3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = -self.0[i][j];
            }
        }
        Self(out)
    }
}

impl Mul<f64> for Mat3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.0[i][j] * s;
            }
        }
        Self(out)
    }
}

impl Mul<Mat3> for f64 {
    type Output = Mat3;
    #[inline]
    fn mul(self, m: Mat3) -> Mat3 {
        m * self
    }
}

/// 6x6 row-major FP64 spatial matrix.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Mat6(pub [[f64; 6]; 6]);

impl Mat6 {
    pub const ZERO: Self = Self([[0.0; 6]; 6]);
    pub const IDENTITY: Self = Self([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]);

    #[inline]
    pub fn from_blocks(b00: Mat3, b01: Mat3, b10: Mat3, b11: Mat3) -> Self {
        let mut m = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = b00.0[i][j];
                m[i][j + 3] = b01.0[i][j];
                m[i + 3][j] = b10.0[i][j];
                m[i + 3][j + 3] = b11.0[i][j];
            }
        }
        Self(m)
    }

    #[inline]
    pub fn block(&self, bi: usize, bj: usize) -> Mat3 {
        let mut out = [[0.0; 3]; 3];
        let r0 = bi * 3;
        let c0 = bj * 3;
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.0[r0 + i][c0 + j];
            }
        }
        Mat3(out)
    }

    #[inline]
    pub fn set_block(&mut self, bi: usize, bj: usize, b: Mat3) {
        let r0 = bi * 3;
        let c0 = bj * 3;
        for i in 0..3 {
            for j in 0..3 {
                self.0[r0 + i][c0 + j] = b.0[i][j];
            }
        }
    }

    #[inline]
    pub fn transpose(self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.0[j][i];
            }
        }
        Self(out)
    }

    #[inline]
    pub fn symmetrize(self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = 0.5 * (self.0[i][j] + self.0[j][i]);
            }
        }
        Self(out)
    }

    #[inline]
    pub fn mul_vec(self, v: Vec6) -> Vec6 {
        let mut out = [0.0; 6];
        for i in 0..6 {
            let mut s = 0.0;
            for j in 0..6 {
                s += self.0[i][j] * v.0[j];
            }
            out[i] = s;
        }
        Vec6(out)
    }

    #[inline]
    pub fn mul_mat(self, rhs: Self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for k in 0..6 {
                let aik = self.0[i][k];
                if aik != 0.0 {
                    for j in 0..6 {
                        out[i][j] += aik * rhs.0[k][j];
                    }
                }
            }
        }
        Self(out)
    }

    #[inline]
    pub fn max_abs(self) -> f64 {
        let mut m = 0.0_f64;
        for row in &self.0 {
            for &v in row {
                m = m.max(v.abs());
            }
        }
        m
    }

    /// Solve `self * x = b` via Gaussian elimination with partial pivoting.
    pub fn solve_lu(self, b: Vec6) -> Option<Vec6> {
        let mut a = self.0;
        let mut rhs = b.0;
        for k in 0..6 {
            let mut piv = k;
            let mut max_v = a[k][k].abs();
            for (i, row) in a.iter().enumerate().skip(k + 1) {
                let v = row[k].abs();
                if v > max_v {
                    max_v = v;
                    piv = i;
                }
            }
            if max_v < 1e-30 {
                return None;
            }
            if piv != k {
                a.swap(k, piv);
                rhs.swap(k, piv);
            }
            let inv_diag = 1.0 / a[k][k];
            for i in (k + 1)..6 {
                let f = a[i][k] * inv_diag;
                a[i][k] = 0.0;
                for j in (k + 1)..6 {
                    a[i][j] -= f * a[k][j];
                }
                rhs[i] -= f * rhs[k];
            }
        }
        let mut x = [0.0; 6];
        for i in (0..6).rev() {
            let mut s = rhs[i];
            for j in (i + 1)..6 {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        Some(Vec6(x))
    }

    /// Invert 6x6 matrix via Gauss-Jordan with partial pivoting.
    pub fn inv_lu(self) -> Option<Self> {
        let mut a = self.0;
        let mut inv = Self::IDENTITY.0;
        for k in 0..6 {
            let mut piv = k;
            let mut max_v = a[k][k].abs();
            for (i, row) in a.iter().enumerate().skip(k + 1) {
                let v = row[k].abs();
                if v > max_v {
                    max_v = v;
                    piv = i;
                }
            }
            if max_v < 1e-30 {
                return None;
            }
            if piv != k {
                a.swap(k, piv);
                inv.swap(k, piv);
            }
            let inv_diag = 1.0 / a[k][k];
            for j in 0..6 {
                a[k][j] *= inv_diag;
                inv[k][j] *= inv_diag;
            }
            for i in 0..6 {
                if i == k {
                    continue;
                }
                let f = a[i][k];
                if f != 0.0 {
                    for j in 0..6 {
                        a[i][j] -= f * a[k][j];
                        inv[i][j] -= f * inv[k][j];
                    }
                }
            }
        }
        Some(Self(inv))
    }

    /// Cholesky factorization `L * L^T = self` (returns lower-triangular `L`) for SPD 6x6 matrices.
    pub fn cholesky(self) -> Option<Self> {
        let mut l = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let mut sum = self.0[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }
                if i == j {
                    if sum <= 0.0 {
                        return None;
                    }
                    l[i][j] = sum.sqrt();
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
        }
        Some(Self(l))
    }

    /// Solve SPD `self * x = b` via Cholesky factorization (falls back to LU if needed via `solve_lu`).
    pub fn solve_cholesky(self, b: Vec6) -> Option<Vec6> {
        let l = self.cholesky()?;
        let mut y = [0.0; 6];
        for i in 0..6 {
            let mut s = b.0[i];
            for k in 0..i {
                s -= l.0[i][k] * y[k];
            }
            y[i] = s / l.0[i][i];
        }
        let mut x = [0.0; 6];
        for i in (0..6).rev() {
            let mut s = y[i];
            for k in (i + 1)..6 {
                s -= l.0[k][i] * x[k];
            }
            x[i] = s / l.0[i][i];
        }
        Some(Vec6(x))
    }

    /// Symmetric cyclic Jacobi eigendecomposition returning `(eigenvalues, V)`
    /// such that `self_sym == V * diag(evals) * V^T`.
    pub fn sym_eigen(self) -> ([f64; 6], Self) {
        let mut a = self.symmetrize().0;
        let mut v = Self::IDENTITY.0;
        for _ in 0..80 {
            let mut max_off = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..6 {
                for j in (i + 1)..6 {
                    let val = a[i][j].abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_off < 1e-14 {
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
            for r in 0..6 {
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
        let mut evals = [0.0; 6];
        for i in 0..6 {
            evals[i] = a[i][i];
        }
        (evals, Self(v))
    }

    /// Project symmetric 6x6 matrix onto the SPD cone by clamping eigenvalues to `>= min_eig`.
    pub fn project_spd(self, min_eig: f64) -> Self {
        let (evals, v) = self.sym_eigen();
        let mut d = [[0.0; 6]; 6];
        for i in 0..6 {
            d[i][i] = evals[i].max(min_eig);
        }
        v.mul_mat(Self(d)).mul_mat(v.transpose()).symmetrize()
    }
}

impl Add for Mat6 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.0[i][j] + rhs.0[i][j];
            }
        }
        Self(out)
    }
}

impl AddAssign for Mat6 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            for j in 0..6 {
                self.0[i][j] += rhs.0[i][j];
            }
        }
    }
}

impl Sub for Mat6 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.0[i][j] - rhs.0[i][j];
            }
        }
        Self(out)
    }
}

impl SubAssign for Mat6 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            for j in 0..6 {
                self.0[i][j] -= rhs.0[i][j];
            }
        }
    }
}

impl Mul<f64> for Mat6 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        let mut out = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.0[i][j] * s;
            }
        }
        Self(out)
    }
}

impl Mul<Mat6> for f64 {
    type Output = Mat6;
    #[inline]
    fn mul(self, m: Mat6) -> Mat6 {
        m * self
    }
}

/// 12x12 row-major FP64 beam element matrix.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Mat12(pub [[f64; 12]; 12]);

impl Mat12 {
    pub const ZERO: Self = Self([[0.0; 12]; 12]);

    pub fn identity() -> Self {
        let mut m = [[0.0; 12]; 12];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self(m)
    }

    pub fn from_blocks6(b00: Mat6, b01: Mat6, b10: Mat6, b11: Mat6) -> Self {
        let mut m = [[0.0; 12]; 12];
        for i in 0..6 {
            for j in 0..6 {
                m[i][j] = b00.0[i][j];
                m[i][j + 6] = b01.0[i][j];
                m[i + 6][j] = b10.0[i][j];
                m[i + 6][j + 6] = b11.0[i][j];
            }
        }
        Self(m)
    }

    pub fn block6(&self, bi: usize, bj: usize) -> Mat6 {
        let mut out = [[0.0; 6]; 6];
        let r0 = bi * 6;
        let c0 = bj * 6;
        for i in 0..6 {
            for j in 0..6 {
                out[i][j] = self.0[r0 + i][c0 + j];
            }
        }
        Mat6(out)
    }

    pub fn set_block6(&mut self, bi: usize, bj: usize, b: Mat6) {
        let r0 = bi * 6;
        let c0 = bj * 6;
        for i in 0..6 {
            for j in 0..6 {
                self.0[r0 + i][c0 + j] = b.0[i][j];
            }
        }
    }

    pub fn transpose(&self) -> Self {
        let mut out = [[0.0; 12]; 12];
        for i in 0..12 {
            for j in 0..12 {
                out[i][j] = self.0[j][i];
            }
        }
        Self(out)
    }

    pub fn mul_vec(&self, v: Vec12) -> Vec12 {
        let mut out = [0.0; 12];
        for i in 0..12 {
            let mut s = 0.0;
            for j in 0..12 {
                s += self.0[i][j] * v.0[j];
            }
            out[i] = s;
        }
        Vec12(out)
    }

    pub fn mul_mat(&self, rhs: &Self) -> Self {
        let mut out = [[0.0; 12]; 12];
        for i in 0..12 {
            for k in 0..12 {
                let aik = self.0[i][k];
                if aik != 0.0 {
                    for j in 0..12 {
                        out[i][j] += aik * rhs.0[k][j];
                    }
                }
            }
        }
        Self(out)
    }

    /// Compute `T^T * self * T` efficiently.
    pub fn congruent_transform(&self, t: &Self) -> Self {
        let mt = self.mul_mat(t);
        t.transpose().mul_mat(&mt)
    }

    /// Solve `self * x = b` via Gaussian elimination with partial pivoting.
    pub fn solve_lu(&self, b: Vec12) -> Option<Vec12> {
        let mut a = self.0;
        let mut rhs = b.0;
        for k in 0..12 {
            let mut piv = k;
            let mut max_v = a[k][k].abs();
            for (i, row) in a.iter().enumerate().skip(k + 1) {
                let v = row[k].abs();
                if v > max_v {
                    max_v = v;
                    piv = i;
                }
            }
            if max_v < 1e-30 {
                return None;
            }
            if piv != k {
                a.swap(k, piv);
                rhs.swap(k, piv);
            }
            let inv_diag = 1.0 / a[k][k];
            for i in (k + 1)..12 {
                let f = a[i][k] * inv_diag;
                a[i][k] = 0.0;
                for j in (k + 1)..12 {
                    a[i][j] -= f * a[k][j];
                }
                rhs[i] -= f * rhs[k];
            }
        }
        let mut x = [0.0; 12];
        for i in (0..12).rev() {
            let mut s = rhs[i];
            for j in (i + 1)..12 {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        Some(Vec12(x))
    }

    /// Pack symmetric 12x12 local beam matrix into 16 non-redundant scalars matching
    /// `pack_beam_matrix` in `serpworm.physics.gpu_core`.
    pub fn pack_beam_16(&self) -> [f64; 16] {
        let m = self.0;
        [
            m[0][0], m[1][1], m[2][2], m[3][3], m[4][4], m[5][5], m[1][5], m[2][4], m[0][6],
            m[1][7], m[2][8], m[3][9], m[4][10], m[5][11], m[1][11], m[2][10],
        ]
    }

    pub fn max_abs(&self) -> f64 {
        let mut m = 0.0_f64;
        for row in &self.0 {
            for &v in row {
                m = m.max(v.abs());
            }
        }
        m
    }
}

impl Add for Mat12 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut out = [[0.0; 12]; 12];
        for i in 0..12 {
            for j in 0..12 {
                out[i][j] = self.0[i][j] + rhs.0[i][j];
            }
        }
        Self(out)
    }
}

impl Sub for Mat12 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut out = [[0.0; 12]; 12];
        for i in 0..12 {
            for j in 0..12 {
                out[i][j] = self.0[i][j] - rhs.0[i][j];
            }
        }
        Self(out)
    }
}

impl Mul<f64> for Mat12 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let mut out = [[0.0; 12]; 12];
        for i in 0..12 {
            for j in 0..12 {
                out[i][j] = self.0[i][j] * s;
            }
        }
        Self(out)
    }
}
