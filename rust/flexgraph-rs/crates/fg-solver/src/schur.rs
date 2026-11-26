//! Bipartite direct linear solver for rigid-flexible continuum graphs:
//! - Stage 1a: Parallel `6x6` Block-Thomas elimination per corotational beam (clamped-clamped,
//!   Dirichlet fixed base, and free cantilever tip `body_b = None`).
//! - Stage 1b: Exact $O(E)$ symmetric banded elimination (half-bandwidth 10) per Discrete Elastic Rod
//!   (DER) condensing interior position + edge-twist DOFs onto rigid clamp interfaces.
//! - Stage 2: Compile-time Adaptive Rigid Schur Router (`BlockDiagonalPairs`, `TreeChain` $O(N_b)$
//!   zero-fill Block-Tree $LDL^\top$, and `SparseGeneral` symbolic CSC Block-$LDL^\top$ using
//!   `fg_ir::SparseBlockPattern`).
//! - Stage 3: Parallel $O(E)$ back-substitution and multi-RHS `solve_rhs` support (enabling exact
//!   Delassus compliance matrix $C_c = J_c H^{-1} J_c^\top$ evaluation without re-factoring).

use fg_ir::{GraphTopologyAnalysis, SchurRoute, SparseBlockPattern};
#[cfg(feature = "faer-sparse")]
use fg_ir::RigidFlexGraph;
use fg_math::{Mat12, Mat6, Vec12, Vec6};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "faer-sparse")]
use std::sync::Arc;

/// Sparse direct factorization backend for `SchurRoute::SparseGeneral`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseBackend {
    /// Automatic selection: uses `faer` SIMD supernodal sparse $LDL^\top$ when compiled with
    /// the `faer-sparse` feature, falling back cleanly to pure-Rust `SparseBlockLdltFactor`.
    #[default]
    Auto,
    /// Pure-Rust `6x6` Block-CSC $LDL^\top$ (`SparseBlockLdltFactor`, zero external deps).
    #[serde(alias = "custom")]
    CustomBlockCsc,
    /// `faer` SIMD supernodal sparse $LDL^\top$ with AMD fill-reducing ordering.
    #[serde(alias = "faer")]
    FaerSupernodal,
}

/// Numerical failure reason from direct factorization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinearSolveError {
    SingularBeamPivot { beam_id: usize, node_idx: usize },
    SingularRigidPivot { block_idx: usize },
    PatternMissingEntry { col: usize, row: usize },
    InvalidDimensions(&'static str),
}

impl std::fmt::Display for LinearSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingularBeamPivot { beam_id, node_idx } => {
                write!(f, "singular 6x6 pivot in beam {beam_id} at node {node_idx}")
            }
            Self::SingularRigidPivot { block_idx } => {
                write!(f, "singular 6x6 rigid Schur pivot at block {block_idx}")
            }
            Self::PatternMissingEntry { col, row } => {
                write!(
                    f,
                    "symbolic sparse block pattern missing entry (row={row}, col={col})"
                )
            }
            Self::InvalidDimensions(msg) => write!(f, "invalid dimensions: {msg}"),
        }
    }
}

impl std::error::Error for LinearSolveError {}

/// Condensed contribution of a single flexible beam onto its endpoint rigid bodies.
#[derive(Clone, Debug)]
pub struct BeamSchurContribution {
    pub beam_id: usize,
    pub body_a: usize,
    pub body_b: Option<usize>,
    /// `6x6` Schur block `(a, a)`.
    pub s_aa: Mat6,
    /// `6x6` Schur block `(a, b)` when `body_b.is_some()`.
    pub s_ab: Mat6,
    /// `6x6` Schur block `(b, a)` when `body_b.is_some()`.
    pub s_ba: Mat6,
    /// `6x6` Schur block `(b, b)` when `body_b.is_some()`.
    pub s_bb: Mat6,
    /// Condensed `6x1` RHS correction added to `body_a` (`-H_{a,F} H_{FF}^{-1} b_F`).
    pub r_a: Vec6,
    /// Condensed `6x1` RHS correction added to `body_b` (`-H_{b,F} H_{FF}^{-1} b_F`).
    pub r_b: Vec6,
}

/// Cached Block-Thomas factors for a single 6-DOF-per-node corotational beam, supporting
/// both initial back-substitution and multi-RHS solves (`solve_with_rhs`).
#[derive(Clone, Debug)]
pub struct CorotationalBeamFactor {
    pub beam_id: usize,
    pub is_cantilever: bool,
    pub n_free: usize,
    /// `inv_diag[i]` = `(D_i - L_{i-1} D_{i-1}^{-1} U_{i-1})^{-1}` (`6x6`).
    pub inv_diag: Vec<Mat6>,
    /// `lower_factors[i]` = `lower[i] * inv_diag[i]` for `i = 0..n_free-1` (len `n_free.saturating_sub(1)`).
    pub lower_factors: Vec<Mat6>,
    /// `upper[i]` = `A01[i+1]` (len `n_free.saturating_sub(1)`).
    pub upper: Vec<Mat6>,
    /// `left = A01[0]` (`H_{a, F0}`), `right = A10[E-1]` (`H_{b, F_{m-1}}`).
    pub left: Mat6,
    pub right: Mat6,
    /// Solved boundary response matrices `z_a[i]` (`6x6`, response of interior node `i` to unit `v_a`)
    /// and `z_b[i]` (`6x6`, response to unit `v_b` when clamped-clamped).
    pub z_a: Vec<Mat6>,
    pub z_b: Vec<Mat6>,
    /// Initial solved free response `z0[i]` (`H_{FF}^{-1} b_F`).
    pub z0: Vec<Vec6>,
}

impl CorotationalBeamFactor {
    /// Factor a single corotational beam (`E >= 1` element `12x12` matrices `elem_a`)
    /// with optional diagonal `6x6` interior contact/stiffness blocks `interior_k` and
    /// interior RHS `interior_rhs`.
    ///
    /// - Clamped-clamped (`body_b = Some(bb)`): `n_free = E - 1`.
    /// - Free cantilever (`body_b = None`): `n_free = E` (last free node is the free tip).
    pub fn factor_and_condense(
        beam_id: usize,
        body_a: usize,
        body_b: Option<usize>,
        elem_a: &[Mat12],
        interior_k: &[Mat6],
        interior_rhs: &[Vec6],
    ) -> Result<(Self, BeamSchurContribution), LinearSolveError> {
        let n_elem = elem_a.len();
        if n_elem == 0 {
            return Err(LinearSolveError::InvalidDimensions("beam has 0 elements"));
        }
        let is_cantilever = body_b.is_none();
        let n_free = if is_cantilever { n_elem } else { n_elem - 1 };
        if interior_rhs.len() != n_free || (!interior_k.is_empty() && interior_k.len() != n_free) {
            return Err(LinearSolveError::InvalidDimensions(
                "beam interior_rhs/interior_k length mismatch",
            ));
        }

        // Trivial 1-element clamped-clamped beam (zero interior DOFs).
        if n_free == 0 {
            let a0 = &elem_a[0];
            let factor = Self {
                beam_id,
                is_cantilever: false,
                n_free: 0,
                inv_diag: Vec::new(),
                lower_factors: Vec::new(),
                upper: Vec::new(),
                left: Mat6::ZERO,
                right: Mat6::ZERO,
                z_a: Vec::new(),
                z_b: Vec::new(),
                z0: Vec::new(),
            };
            let contrib = BeamSchurContribution {
                beam_id,
                body_a,
                body_b,
                s_aa: a0.block6(0, 0),
                s_ab: a0.block6(0, 1),
                s_ba: a0.block6(1, 0),
                s_bb: a0.block6(1, 1),
                r_a: Vec6::ZERO,
                r_b: Vec6::ZERO,
            };
            return Ok((factor, contrib));
        }

        let mut diag = Vec::with_capacity(n_free);
        let mut lower = Vec::with_capacity(n_free.saturating_sub(1));
        let mut upper = Vec::with_capacity(n_free.saturating_sub(1));

        for i in 0..n_free {
            let mut d = elem_a[i].block6(1, 1);
            if i + 1 < n_elem {
                d += elem_a[i + 1].block6(0, 0);
            }
            if !interior_k.is_empty() {
                d += interior_k[i];
            }
            diag.push(d);
            if i > 0 {
                lower.push(elem_a[i].block6(1, 0));
                upper.push(elem_a[i].block6(0, 1));
            }
        }

        let left = elem_a[0].block6(0, 1);
        let h_f0_a = elem_a[0].block6(1, 0);
        let right = if is_cantilever {
            Mat6::ZERO
        } else {
            elem_a[n_elem - 1].block6(1, 0)
        };
        let h_flast_b = if is_cantilever {
            Mat6::ZERO
        } else {
            elem_a[n_elem - 1].block6(0, 1)
        };

        // Forward Block-Thomas sweep over RHS vector and boundary columns (A and optionally B).
        let mut inv_diag = Vec::with_capacity(n_free);
        let mut lower_factors = Vec::with_capacity(n_free.saturating_sub(1));
        let mut rhs_fwd = Vec::with_capacity(n_free);
        let mut a_fwd = Vec::with_capacity(n_free);
        let mut b_fwd = if is_cantilever {
            Vec::new()
        } else {
            Vec::with_capacity(n_free)
        };

        for i in 0..n_free {
            let mut d = diag[i];
            let mut r = interior_rhs[i];
            let mut ba = if i == 0 { h_f0_a } else { Mat6::ZERO };
            let mut bb = if !is_cantilever && i + 1 == n_free {
                h_flast_b
            } else {
                Mat6::ZERO
            };

            if i > 0 {
                let fac = lower[i - 1].mul_mat(inv_diag[i - 1]);
                d -= fac.mul_mat(upper[i - 1]);
                r -= fac.mul_vec(rhs_fwd[i - 1]);
                ba -= fac.mul_mat(a_fwd[i - 1]);
                if !is_cantilever {
                    bb -= fac.mul_mat(b_fwd[i - 1]);
                }
                lower_factors.push(fac);
            }

            let d_inv = d
                .inv_lu()
                .ok_or(LinearSolveError::SingularBeamPivot { beam_id, node_idx: i })?;
            inv_diag.push(d_inv);
            rhs_fwd.push(r);
            a_fwd.push(ba);
            if !is_cantilever {
                b_fwd.push(bb);
            }
        }

        // Backward Block-Thomas substitution
        let mut z0 = vec![Vec6::ZERO; n_free];
        let mut z_a = vec![Mat6::ZERO; n_free];
        let mut z_b = if is_cantilever {
            Vec::new()
        } else {
            vec![Mat6::ZERO; n_free]
        };

        let last = n_free - 1;
        z0[last] = inv_diag[last].mul_vec(rhs_fwd[last]);
        z_a[last] = inv_diag[last].mul_mat(a_fwd[last]);
        if !is_cantilever {
            z_b[last] = inv_diag[last].mul_mat(b_fwd[last]);
        }

        for i in (0..last).rev() {
            z0[i] = inv_diag[i].mul_vec(rhs_fwd[i] - upper[i].mul_vec(z0[i + 1]));
            z_a[i] = inv_diag[i].mul_mat(a_fwd[i] - upper[i].mul_mat(z_a[i + 1]));
            if !is_cantilever {
                z_b[i] = inv_diag[i].mul_mat(b_fwd[i] - upper[i].mul_mat(z_b[i + 1]));
            }
        }

        // Condensed Schur blocks on endpoints
        let s_aa = elem_a[0].block6(0, 0) - left.mul_mat(z_a[0]);
        let r_a = -(left.mul_vec(z0[0]));
        let (s_ab, s_ba, s_bb, r_b) = if is_cantilever {
            (Mat6::ZERO, Mat6::ZERO, Mat6::ZERO, Vec6::ZERO)
        } else {
            let sab = (-1.0) * left.mul_mat(z_b[0]);
            let sba = (-1.0) * right.mul_mat(z_a[last]);
            let sbb = elem_a[n_elem - 1].block6(1, 1) - right.mul_mat(z_b[last]);
            let rb = -(right.mul_vec(z0[last]));
            (sab, sba, sbb, rb)
        };

        let factor = Self {
            beam_id,
            is_cantilever,
            n_free,
            inv_diag,
            lower_factors,
            upper,
            left,
            right,
            z_a,
            z_b,
            z0,
        };
        let contrib = BeamSchurContribution {
            beam_id,
            body_a,
            body_b,
            s_aa,
            s_ab,
            s_ba,
            s_bb,
            r_a,
            r_b,
        };
        Ok((factor, contrib))
    }

    /// Condense an arbitrary new interior RHS vector `rhs_int` using cached Block-Thomas factors
    /// (zero re-factorization), returning `(z0_new, r_a_corr, r_b_corr)`.
    pub fn condense_rhs(&self, rhs_int: &[Vec6]) -> (Vec<Vec6>, Vec6, Vec6) {
        if self.n_free == 0 {
            return (Vec::new(), Vec6::ZERO, Vec6::ZERO);
        }
        let mut fwd = Vec::with_capacity(self.n_free);
        for i in 0..self.n_free {
            let mut r = rhs_int[i];
            if i > 0 {
                r -= self.lower_factors[i - 1].mul_vec(fwd[i - 1]);
            }
            fwd.push(r);
        }
        let mut z0 = vec![Vec6::ZERO; self.n_free];
        let last = self.n_free - 1;
        z0[last] = self.inv_diag[last].mul_vec(fwd[last]);
        for i in (0..last).rev() {
            z0[i] = self.inv_diag[i].mul_vec(fwd[i] - self.upper[i].mul_vec(z0[i + 1]));
        }
        let r_a = -(self.left.mul_vec(z0[0]));
        let r_b = if self.is_cantilever {
            Vec6::ZERO
        } else {
            -(self.right.mul_vec(z0[last]))
        };
        (z0, r_a, r_b)
    }

    /// Stage 3 back-substitution recovering all free beam node velocities given endpoint twists
    /// `v_a` and `v_b` (use `Vec6::ZERO` for fixed Dirichlet bases or cantilevers).
    pub fn back_substitute(&self, v_a: Vec6, v_b: Vec6) -> Vec<Vec6> {
        self.back_substitute_with_z0(&self.z0, v_a, v_b)
    }

    /// Back-substitute using a custom `z0` returned by `condense_rhs`.
    pub fn back_substitute_with_z0(&self, z0: &[Vec6], v_a: Vec6, v_b: Vec6) -> Vec<Vec6> {
        let mut out = Vec::with_capacity(self.n_free);
        for i in 0..self.n_free {
            let mut v = z0[i] - self.z_a[i].mul_vec(v_a);
            if !self.is_cantilever {
                v -= self.z_b[i].mul_vec(v_b);
            }
            out.push(v);
        }
        out
    }
}

/// Symmetric banded scalar $LDL^\top$ solver (half-bandwidth `bw`, e.g., `bw = 10` for DER rods)
/// operating in $O(m \cdot \text{bw}^2)$ for exact $O(E)$ DER-to-rigid Schur condensation.
#[derive(Clone, Debug)]
pub struct SymmetricBandedLdlt {
    pub n: usize,
    pub bw: usize,
    /// Lower-triangular band storage: entry `(i, j)` with `0 <= i - j <= bw` stored at
    /// `band[i * (bw + 1) + (i - j)]`. After factorization:
    /// - `d = band[i * (bw + 1)]` stores `D_ii`,
    /// - `band[i * (bw + 1) + d]` (`d >= 1`) stores `L_{i, i-d}`.
    band: Vec<f64>,
}

impl SymmetricBandedLdlt {
    /// Allocate zeroed symmetric banded matrix of size `n` with half-bandwidth `bw`.
    pub fn zeros(n: usize, bw: usize) -> Self {
        Self {
            n,
            bw,
            band: vec![0.0; n * (bw + 1)],
        }
    }

    /// Add `val` to symmetric entry `(i, j)` (stores in lower triangle; ignores upper duplicate
    /// if caller passes both — callers should pass only `i >= j` or use `add_sym`).
    #[inline]
    pub fn add_lower(&mut self, i: usize, j: usize, val: f64) {
        debug_assert!(i >= j && i - j <= self.bw);
        self.band[i * (self.bw + 1) + (i - j)] += val;
    }

    /// Add `val` to `(i, j)` for any `(i, j)` with `|i - j| <= bw` (accumulates into lower triangle
    /// when `i >= j` only, so iterating over full symmetric element matrices adds each off-diagonal once).
    #[inline]
    pub fn add_from_full_sym(&mut self, i: usize, j: usize, val: f64) {
        if i >= j {
            debug_assert!(i - j <= self.bw);
            self.band[i * (self.bw + 1) + (i - j)] += val;
        }
    }

    /// In-place banded $LDL^\top$ factorization in $O(n \cdot \text{bw}^2)$.
    pub fn factorize(&mut self, beam_id: usize) -> Result<(), LinearSolveError> {
        let stride = self.bw + 1;
        for j in 0..self.n {
            let d_jj = self.band[j * stride];
            if !d_jj.is_finite() || d_jj.abs() < 1e-30 {
                return Err(LinearSolveError::SingularBeamPivot {
                    beam_id,
                    node_idx: j,
                });
            }
            let inv_d = 1.0 / d_jj;
            let i_max = (j + self.bw).min(self.n - 1);
            // Update column j to store L_{i,j} = A_{i,j} / D_{j,j}, then rank-1 update trailing band
            for i in (j + 1)..=i_max {
                let w_ij = self.band[i * stride + (i - j)];
                if w_ij == 0.0 {
                    continue;
                }
                let l_ij = w_ij * inv_d;
                self.band[i * stride + (i - j)] = l_ij;
                // Update diagonal (i, i)
                self.band[i * stride] -= l_ij * w_ij;
                // Update (r, i) for r = i+1..=i_max
                for r in (i + 1)..=i_max {
                    let w_rj = self.band[r * stride + (r - j)];
                    if w_rj != 0.0 {
                        self.band[r * stride + (r - i)] -= l_ij * w_rj;
                    }
                }
            }
        }
        Ok(())
    }

    /// Solve `A x = b` in-place on `b` in $O(n \cdot \text{bw})$.
    pub fn solve_in_place(&self, b: &mut [f64]) {
        debug_assert_eq!(b.len(), self.n);
        let stride = self.bw + 1;
        // Forward L y = b
        for j in 0..self.n {
            let yj = b[j];
            if yj == 0.0 {
                continue;
            }
            let i_max = (j + self.bw).min(self.n - 1);
            for i in (j + 1)..=i_max {
                b[i] -= self.band[i * stride + (i - j)] * yj;
            }
        }
        // Diagonal D z = y
        for j in 0..self.n {
            b[j] /= self.band[j * stride];
        }
        // Backward L^T x = z
        for j in (0..self.n).rev() {
            let mut s = b[j];
            let i_max = (j + self.bw).min(self.n - 1);
            for i in (j + 1)..=i_max {
                s -= self.band[i * stride + (i - j)] * b[i];
            }
            b[j] = s;
        }
    }
}

/// Assembled rigid-interface Schur matrix over `num_free_bodies` 6x6 blocks.
#[derive(Clone, Debug)]
pub struct RigidSchurMatrix {
    pub n_free: usize,
    /// Diagonal `6x6` blocks `diag[u]` for `u in 0..n_free`.
    pub diag: Vec<Mat6>,
    /// Off-diagonal `6x6` blocks stored per row `u` as sorted `(v, block)` where `v != u`.
    pub off_diag: Vec<Vec<(usize, Mat6)>>,
}

impl RigidSchurMatrix {
    pub fn zeros(n_free: usize) -> Self {
        Self {
            n_free,
            diag: vec![Mat6::ZERO; n_free],
            off_diag: vec![Vec::new(); n_free],
        }
    }

    /// Add `block` to `(u, v)` in free-body slot coordinates.
    pub fn add_block(&mut self, u: usize, v: usize, block: Mat6) {
        if u == v {
            self.diag[u] += block;
        } else {
            let row = &mut self.off_diag[u];
            if let Some((_, existing)) = row.iter_mut().find(|(col, _)| *col == v) {
                *existing += block;
            } else {
                row.push((v, block));
            }
        }
    }

    /// Scatter a condensed beam Schur contribution into the free-body matrix and RHS.
    pub fn scatter_beam(
        &mut self,
        rhs: &mut [Vec6],
        contrib: &BeamSchurContribution,
        body_to_free: &[Option<usize>],
    ) {
        let fa = body_to_free[contrib.body_a];
        let fb = contrib.body_b.and_then(|bb| body_to_free[bb]);
        if let Some(u) = fa {
            self.add_block(u, u, contrib.s_aa);
            rhs[u] += contrib.r_a;
        }
        if let Some(v) = fb {
            self.add_block(v, v, contrib.s_bb);
            rhs[v] += contrib.r_b;
        }
        if let (Some(u), Some(v)) = (fa, fb) {
            self.add_block(u, v, contrib.s_ab);
            self.add_block(v, u, contrib.s_ba);
        }
    }

    /// Fetch block `(u, v)` (zero if absent).
    pub fn get_block(&self, u: usize, v: usize) -> Mat6 {
        if u == v {
            self.diag[u]
        } else {
            self.off_diag[u]
                .iter()
                .find(|(col, _)| *col == v)
                .map_or(Mat6::ZERO, |(_, b)| *b)
        }
    }

    /// Add scalar diagonal shift `lambda * I_6` to every diagonal block (Levenberg-Marquardt damping).
    pub fn add_diagonal_shift(&mut self, lambda: f64) {
        if lambda == 0.0 {
            return;
        }
        let shift = lambda * Mat6::IDENTITY;
        for d in &mut self.diag {
            *d += shift;
        }
    }

    /// Matrix-vector product `y = H * x` over `6x6` blocks (useful for residual verification).
    pub fn mul_vec(&self, x: &[Vec6]) -> Vec<Vec6> {
        let mut y = vec![Vec6::ZERO; self.n_free];
        for u in 0..self.n_free {
            let mut acc = self.diag[u].mul_vec(x[u]);
            for &(v, ref blk) in &self.off_diag[u] {
                acc += blk.mul_vec(x[v]);
            }
            y[u] = acc;
        }
        y
    }
}

/// Cached factorization for `SchurRoute::BlockDiagonalPairs`.
#[derive(Clone, Debug)]
pub struct PairBlocksFactor {
    pub n_free: usize,
    pub singletons: Vec<(usize, Mat6)>, // (slot, inv_H_uu)
    pub pairs: Vec<([usize; 2], Mat12)>, // ([u, v], H_pair_12x12)
}

impl PairBlocksFactor {
    pub fn factorize(
        mat: &RigidSchurMatrix,
        single_blocks: &[usize],
        pair_blocks: &[[usize; 2]],
    ) -> Result<Self, LinearSolveError> {
        let mut singletons = Vec::with_capacity(single_blocks.len());
        for &u in single_blocks {
            let inv = mat.diag[u]
                .inv_lu()
                .ok_or(LinearSolveError::SingularRigidPivot { block_idx: u })?;
            singletons.push((u, inv));
        }
        let mut pairs = Vec::with_capacity(pair_blocks.len());
        for &[u, v] in pair_blocks {
            let h12 = Mat12::from_blocks6(
                mat.diag[u],
                mat.get_block(u, v),
                mat.get_block(v, u),
                mat.diag[v],
            );
            pairs.push(([u, v], h12));
        }
        Ok(Self {
            n_free: mat.n_free,
            singletons,
            pairs,
        })
    }

    pub fn solve(&self, rhs: &[Vec6]) -> Result<Vec<Vec6>, LinearSolveError> {
        let mut out = vec![Vec6::ZERO; self.n_free];
        for &(u, inv) in &self.singletons {
            out[u] = inv.mul_vec(rhs[u]);
        }
        for &([u, v], ref h12) in &self.pairs {
            let b12 = Vec12::from_halves(rhs[u], rhs[v]);
            let x12 = h12
                .solve_lu(b12)
                .ok_or(LinearSolveError::SingularRigidPivot { block_idx: u })?;
            out[u] = x12.head();
            out[v] = x12.tail();
        }
        Ok(out)
    }
}

/// Cached $O(N_b)$ post-order Block-Tree $LDL^\top$ factor for `SchurRoute::TreeChain`.
#[derive(Clone, Debug)]
pub struct TreeBlockLdltFactor {
    pub n_free: usize,
    pub postorder: Vec<usize>,
    pub parent: Vec<Option<usize>>,
    pub d_inv: Vec<Mat6>,
    /// `l_to_parent[u]` = `H_{p, u} * D_u^{-1}`.
    pub l_to_parent: Vec<Mat6>,
    /// `h_up[u]` = `H_{u, p}`.
    pub h_up: Vec<Mat6>,
}

impl TreeBlockLdltFactor {
    pub fn factorize(
        mat: &RigidSchurMatrix,
        postorder: &[usize],
        parent: &[Option<usize>],
    ) -> Result<Self, LinearSolveError> {
        let n = mat.n_free;
        let mut d_work = mat.diag.clone();
        let mut d_inv = vec![Mat6::ZERO; n];
        let mut l_to_parent = vec![Mat6::ZERO; n];
        let mut h_up = vec![Mat6::ZERO; n];

        for &u in postorder {
            let inv = d_work[u]
                .inv_lu()
                .ok_or(LinearSolveError::SingularRigidPivot { block_idx: u })?;
            d_inv[u] = inv;
            if let Some(p) = parent[u] {
                let h_pu = mat.get_block(p, u);
                let h_up_blk = mat.get_block(u, p);
                let l_pu = h_pu.mul_mat(inv);
                l_to_parent[u] = l_pu;
                h_up[u] = h_up_blk;
                d_work[p] -= l_pu.mul_mat(h_up_blk);
            }
        }

        Ok(Self {
            n_free: n,
            postorder: postorder.to_vec(),
            parent: parent.to_vec(),
            d_inv,
            l_to_parent,
            h_up,
        })
    }

    pub fn solve(&self, rhs: &[Vec6]) -> Vec<Vec6> {
        let mut work = rhs.to_vec();
        // Upward elimination (leaves -> roots)
        for &u in &self.postorder {
            if let Some(p) = self.parent[u] {
                let delta = self.l_to_parent[u].mul_vec(work[u]);
                work[p] -= delta;
            }
        }
        // Downward back-substitution (roots -> leaves)
        let mut x = vec![Vec6::ZERO; self.n_free];
        for &u in self.postorder.iter().rev() {
            let mut r = work[u];
            if let Some(p) = self.parent[u] {
                r -= self.h_up[u].mul_vec(x[p]);
            }
            x[u] = self.d_inv[u].mul_vec(r);
        }
        x
    }
}

/// True Sparse Block-$LDL^\top$ factorization over `6x6` block-CSC using the compile-time
/// symbolic elimination tree pattern (`fg_ir::SparseBlockPattern`) for cyclic rigid-flexible
/// graphs (`SchurRoute::SparseGeneral`, e.g., `Tensegrity-6Bar`).
#[derive(Clone, Debug)]
pub struct SparseBlockLdltFactor {
    pub pattern: SparseBlockPattern,
    /// Inverted symmetric diagonal pivots `D_k^{-1}` in permuted order (`k = 0..n_blocks`).
    pub d_inv: Vec<Mat6>,
    /// Strict lower-triangular `6x6` blocks of `L` in CSC order matching `pattern.col_ptr`/`row_idx`.
    pub l_vals: Vec<Mat6>,
}

impl SparseBlockLdltFactor {
    /// Numeric right-looking Sparse Block-$LDL^\top$ factorization of symmetric `mat`
    /// (`P H P^T = L D L^T`) using the precomputed symbolic fill-in pattern `pattern`.
    pub fn factorize(
        mat: &RigidSchurMatrix,
        pattern: &SparseBlockPattern,
    ) -> Result<Self, LinearSolveError> {
        let n = pattern.n_blocks;
        if n != mat.n_free {
            return Err(LinearSolveError::InvalidDimensions(
                "sparse pattern n_blocks != mat.n_free",
            ));
        }
        let mut d_vals = vec![Mat6::ZERO; n];
        for (new_k, &old_u) in pattern.perm.iter().enumerate() {
            d_vals[new_k] = mat.diag[old_u].symmetrize();
        }

        let mut l_vals = vec![Mat6::ZERO; pattern.nnz_offdiag_l];
        // Scatter symmetric strict-lower blocks of P * H * P^T into CSC structure.
        for u in 0..n {
            let pu = pattern.inv_perm[u];
            for &(v, blk) in &mat.off_diag[u] {
                let pv = pattern.inv_perm[v];
                if pu > pv {
                    // Entry (row=pu, col=pv) in strict lower triangle; average with H[v,u]^T for exact symmetry
                    let h_sym = 0.5 * (blk + mat.get_block(v, u).transpose());
                    let start = pattern.col_ptr[pv];
                    let end = pattern.col_ptr[pv + 1];
                    let rel = pattern.row_idx[start..end]
                        .binary_search(&pu)
                        .map_err(|_| LinearSolveError::PatternMissingEntry { col: pv, row: pu })?;
                    l_vals[start + rel] += h_sym;
                }
            }
        }

        let mut d_inv = vec![Mat6::ZERO; n];
        let mut w_col: Vec<Mat6> = Vec::new();

        for k in 0..n {
            let dk = d_vals[k].symmetrize();
            let dk_inv = dk
                .inv_lu()
                .ok_or(LinearSolveError::SingularRigidPivot {
                    block_idx: pattern.perm[k],
                })?;
            d_inv[k] = dk_inv;

            let c_start = pattern.col_ptr[k];
            let c_end = pattern.col_ptr[k + 1];
            let col_len = c_end - c_start;
            if col_len == 0 {
                continue;
            }

            // Copy W_{*, k} = L_{*, k} D_k before overwriting l_vals[c_start..c_end] with L_{*, k}.
            w_col.clear();
            w_col.extend_from_slice(&l_vals[c_start..c_end]);
            for idx in 0..col_len {
                l_vals[c_start + idx] = w_col[idx].mul_mat(dk_inv);
            }

            // Rank-1 block Schur update onto columns i > k:
            // D_i -= L_{i,k} W_{i,k}^T
            // L_{j,i} -= W_{j,k} L_{i,k}^T  for all j > i in column k
            for a in 0..col_len {
                let i = pattern.row_idx[c_start + a];
                let l_ik = l_vals[c_start + a];
                let w_ik = w_col[a];
                d_vals[i] -= l_ik.mul_mat(w_ik.transpose());

                let l_ik_t = l_ik.transpose();
                let col_i_start = pattern.col_ptr[i];
                let col_i_end = pattern.col_ptr[i + 1];
                let col_i_rows = &pattern.row_idx[col_i_start..col_i_end];

                // Both pattern.row_idx[c_start+a+1..c_end] and col_i_rows are strictly sorted,
                // so a linear merge cursor finds every target row in O(1) amortized!
                let mut cursor = 0usize;
                for b in (a + 1)..col_len {
                    let j = pattern.row_idx[c_start + b];
                    while cursor < col_i_rows.len() && col_i_rows[cursor] < j {
                        cursor += 1;
                    }
                    if cursor >= col_i_rows.len() || col_i_rows[cursor] != j {
                        return Err(LinearSolveError::PatternMissingEntry { col: i, row: j });
                    }
                    let update = w_col[b].mul_mat(l_ik_t);
                    l_vals[col_i_start + cursor] -= update;
                    cursor += 1;
                }
            }
        }

        Ok(Self {
            pattern: pattern.clone(),
            d_inv,
            l_vals,
        })
    }

    /// Solve `H x = b` using the cached Sparse Block-$LDL^\top$ factors (`P^T L^{-T} D^{-1} L^{-1} P b`).
    pub fn solve(&self, rhs: &[Vec6]) -> Vec<Vec6> {
        let n = self.pattern.n_blocks;
        // Permute RHS: y[p] = rhs[perm[p]]
        let mut y = vec![Vec6::ZERO; n];
        for (p, &old_u) in self.pattern.perm.iter().enumerate() {
            y[p] = rhs[old_u];
        }

        // Forward solve L z = y (column-oriented)
        for k in 0..n {
            let zk = y[k];
            let c_start = self.pattern.col_ptr[k];
            let c_end = self.pattern.col_ptr[k + 1];
            for idx in c_start..c_end {
                let row = self.pattern.row_idx[idx];
                y[row] -= self.l_vals[idx].mul_vec(zk);
            }
        }

        // Diagonal solve D w = z
        for k in 0..n {
            y[k] = self.d_inv[k].mul_vec(y[k]);
        }

        // Backward solve L^T x_perm = w
        for k in (0..n).rev() {
            let mut s = y[k];
            let c_start = self.pattern.col_ptr[k];
            let c_end = self.pattern.col_ptr[k + 1];
            for idx in c_start..c_end {
                let row = self.pattern.row_idx[idx];
                s -= self.l_vals[idx].transpose().mul_vec(y[row]);
            }
            y[k] = s;
        }

        // Unpermute: x[perm[p]] = y[p]
        let mut x = vec![Vec6::ZERO; n];
        for (p, &old_u) in self.pattern.perm.iter().enumerate() {
            x[old_u] = y[p];
        }
        x
    }
}

/// Cached symbolic AMD + supernodal Cholesky pattern for `faer` sparse $LDL^\top$.
#[cfg(feature = "faer-sparse")]
#[derive(Clone, Debug)]
pub struct FaerSparsePattern {
    pub n_free: usize,
    /// For each free-body block column `v`, sorted list of strict-lower block rows `u > v`.
    block_lower_cols: Vec<Vec<usize>>,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    symbolic: Arc<faer::sparse::linalg::cholesky::SymbolicCholesky<usize>>,
}

#[cfg(feature = "faer-sparse")]
impl FaerSparsePattern {
    /// Build a reusable scalar lower-triangular CSC sparsity pattern + AMD supernodal
    /// symbolic elimination tree from a `RigidFlexGraph` and its `GraphTopologyAnalysis`.
    pub fn from_graph(
        graph: &RigidFlexGraph,
        analysis: &GraphTopologyAnalysis,
    ) -> Result<Self, LinearSolveError> {
        let n_free = analysis.num_free_bodies;
        let mut block_lower_cols: Vec<Vec<usize>> = vec![Vec::new(); n_free];
        let mut add_edge = |ba: usize, bb: usize| {
            if let (Some(u), Some(v)) = (
                analysis.body_to_free_slot[ba],
                analysis.body_to_free_slot[bb],
            ) {
                if u != v {
                    let (row, col) = if u > v { (u, v) } else { (v, u) };
                    block_lower_cols[col].push(row);
                }
            }
        };
        for beam in &graph.beams {
            if let Some(bb) = beam.body_b {
                add_edge(beam.body_a, bb);
            }
        }
        for joint in &graph.joints {
            if joint.implicit_coupling {
                add_edge(joint.body_a, joint.body_b);
            }
        }
        for col in &mut block_lower_cols {
            col.sort_unstable();
            col.dedup();
        }
        Self::from_block_lower_cols(n_free, block_lower_cols)
    }

    /// Build a symbolic pattern directly from a `RigidSchurMatrix`'s active off-diagonal blocks.
    pub fn from_matrix(mat: &RigidSchurMatrix) -> Result<Self, LinearSolveError> {
        let n_free = mat.n_free;
        let mut block_lower_cols: Vec<Vec<usize>> = vec![Vec::new(); n_free];
        for u in 0..n_free {
            for &(v, _) in &mat.off_diag[u] {
                if u > v {
                    block_lower_cols[v].push(u);
                } else if v > u {
                    block_lower_cols[u].push(v);
                }
            }
        }
        for col in &mut block_lower_cols {
            col.sort_unstable();
            col.dedup();
        }
        Self::from_block_lower_cols(n_free, block_lower_cols)
    }

    fn from_block_lower_cols(
        n_free: usize,
        block_lower_cols: Vec<Vec<usize>>,
    ) -> Result<Self, LinearSolveError> {
        use faer::sparse::linalg::cholesky::{
            factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
        };
        use faer::sparse::SymbolicSparseColMatRef;
        use faer::Side;

        let dim = 6 * n_free;
        let mut col_ptr = Vec::with_capacity(dim + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);

        for v in 0..n_free {
            let lower_rows = &block_lower_cols[v];
            for c in 0..6 {
                // Lower triangle of diagonal 6x6 block (v, v)
                for r in c..6 {
                    row_idx.push(6 * v + r);
                }
                // All 6 rows of every strict-lower neighbor block u > v
                for &u in lower_rows {
                    for r in 0..6 {
                        row_idx.push(6 * u + r);
                    }
                }
                col_ptr.push(row_idx.len());
            }
        }

        let sym_ref = SymbolicSparseColMatRef::new_checked(dim, dim, &col_ptr, None, &row_idx);
        let symbolic = factorize_symbolic_cholesky(
            sym_ref,
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .map_err(|_| LinearSolveError::InvalidDimensions("faer symbolic cholesky failed"))?;

        Ok(Self {
            n_free,
            block_lower_cols,
            col_ptr,
            row_idx,
            symbolic: Arc::new(symbolic),
        })
    }

    /// Check whether every off-diagonal block in `mat` is contained in this cached pattern.
    fn covers(&self, mat: &RigidSchurMatrix) -> bool {
        if self.n_free != mat.n_free {
            return false;
        }
        for u in 0..mat.n_free {
            for &(v, _) in &mat.off_diag[u] {
                let (row, col) = if u > v { (u, v) } else { (v, u) };
                if self.block_lower_cols[col].binary_search(&row).is_err() {
                    return false;
                }
            }
        }
        true
    }
}

/// `faer` SIMD supernodal sparse $LDL^\top$ factorization (AMD ordering + zero-copy CSC)
/// accelerating `SchurRoute::SparseGeneral` on large cyclic rigid-flexible graphs.
#[cfg(feature = "faer-sparse")]
#[derive(Clone, Debug)]
pub struct FaerSparseLdltFactor {
    pub n_free: usize,
    symbolic: Arc<faer::sparse::linalg::cholesky::SymbolicCholesky<usize>>,
    l_values: Vec<f64>,
}

#[cfg(feature = "faer-sparse")]
impl FaerSparseLdltFactor {
    /// Factorize `mat` by building a fresh AMD supernodal symbolic structure + numeric $LDL^\top$.
    pub fn factorize(mat: &RigidSchurMatrix) -> Result<Self, LinearSolveError> {
        let pattern = FaerSparsePattern::from_matrix(mat)?;
        Self::factorize_with_pattern(mat, &pattern)
    }

    /// Fast numeric-only `faer` supernodal $LDL^\top$ factorization reusing a precomputed
    /// [`FaerSparsePattern`] (automatically falls back to fresh symbolic analysis if `mat`
    /// contains dynamic edges outside `pattern`).
    pub fn factorize_with_pattern(
        mat: &RigidSchurMatrix,
        pattern: &FaerSparsePattern,
    ) -> Result<Self, LinearSolveError> {
        use faer::dyn_stack::{MemBuffer, MemStack};
        use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
        use faer::{Par, Side};

        if !pattern.covers(mat) {
            return Self::factorize(mat);
        }

        let n_free = mat.n_free;
        let dim = 6 * n_free;
        let mut vals = Vec::with_capacity(pattern.row_idx.len());

        for v in 0..n_free {
            let d_sym = mat.diag[v].symmetrize();
            let lower_rows = &pattern.block_lower_cols[v];
            // Pre-symmetrize off-diagonal coupling blocks for column v once across the 6 scalar cols
            let off_blocks: Vec<Mat6> = lower_rows
                .iter()
                .map(|&u| 0.5 * (mat.get_block(u, v) + mat.get_block(v, u).transpose()))
                .collect();
            for c in 0..6 {
                for r in c..6 {
                    vals.push(d_sym.0[r][c]);
                }
                for blk in &off_blocks {
                    for r in 0..6 {
                        vals.push(blk.0[r][c]);
                    }
                }
            }
        }

        let sym_ref = SymbolicSparseColMatRef::new_checked(
            dim,
            dim,
            &pattern.col_ptr,
            None,
            &pattern.row_idx,
        );
        let mat_ref = SparseColMatRef::new(sym_ref, &vals);
        let mut l_values = vec![0.0_f64; pattern.symbolic.len_val()];
        let scratch_req = pattern
            .symbolic
            .factorize_numeric_ldlt_scratch::<f64>(Par::Seq, Default::default());
        let mut mem = MemBuffer::new(scratch_req);
        pattern
            .symbolic
            .factorize_numeric_ldlt::<f64>(
                &mut l_values,
                mat_ref,
                Side::Lower,
                Default::default(),
                Par::Seq,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .map_err(|_| LinearSolveError::SingularRigidPivot { block_idx: 0 })?;

        Ok(Self {
            n_free,
            symbolic: Arc::clone(&pattern.symbolic),
            l_values,
        })
    }

    /// Solve `H x = rhs` in $O(\mathrm{nnz}(L))$ using the cached `faer` SIMD supernodal factor.
    pub fn solve(&self, rhs: &[Vec6]) -> Vec<Vec6> {
        use faer::dyn_stack::{MemBuffer, MemStack};
        use faer::sparse::linalg::cholesky::LdltRef;
        use faer::{Conj, MatMut, Par};

        let dim = 6 * self.n_free;
        let mut buf = vec![0.0_f64; dim];
        for (u, v) in rhs.iter().take(self.n_free).enumerate() {
            buf[6 * u..6 * u + 6].copy_from_slice(&v.0);
        }
        let ldlt = LdltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);
        let mut x_mat = MatMut::from_column_major_slice_mut(&mut buf, dim, 1);
        let scratch_req = self.symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
        let mut mem = MemBuffer::new(scratch_req);
        ldlt.solve_in_place_with_conj(
            Conj::No,
            x_mat.as_mut(),
            Par::Seq,
            MemStack::new(&mut mem),
        );

        let mut out = Vec::with_capacity(self.n_free);
        for u in 0..self.n_free {
            let mut arr = [0.0; 6];
            arr.copy_from_slice(&buf[6 * u..6 * u + 6]);
            out.push(Vec6(arr));
        }
        out
    }
}

/// Polymorphic Stage-2 factored rigid Schur complement solver selected by `SchurRoute`.
#[derive(Clone, Debug)]
pub enum RigidSchurFactor {
    Empty,
    Pairs(PairBlocksFactor),
    Tree(TreeBlockLdltFactor),
    Sparse(SparseBlockLdltFactor),
    #[cfg(feature = "faer-sparse")]
    FaerSparse(FaerSparseLdltFactor),
}

impl RigidSchurFactor {
    /// Factor `mat` according to the compile-time route in `analysis` using `SparseBackend::Auto`.
    pub fn factorize(
        mat: &RigidSchurMatrix,
        analysis: &GraphTopologyAnalysis,
    ) -> Result<Self, LinearSolveError> {
        #[cfg(feature = "faer-sparse")]
        {
            Self::factorize_with_backend(mat, analysis, SparseBackend::Auto, None)
        }
        #[cfg(not(feature = "faer-sparse"))]
        {
            Self::factorize_with_backend(mat, analysis, SparseBackend::Auto)
        }
    }

    /// Factor `mat` with an explicit [`SparseBackend`] (and optional cached [`FaerSparsePattern`]
    /// when the `faer-sparse` feature is enabled).
    #[cfg(feature = "faer-sparse")]
    pub fn factorize_with_backend(
        mat: &RigidSchurMatrix,
        analysis: &GraphTopologyAnalysis,
        backend: SparseBackend,
        faer_pattern: Option<&FaerSparsePattern>,
    ) -> Result<Self, LinearSolveError> {
        if analysis.num_free_bodies == 0 {
            return Ok(Self::Empty);
        }
        let factor_sparse = || -> Result<Self, LinearSolveError> {
            match backend {
                SparseBackend::CustomBlockCsc => Ok(Self::Sparse(
                    SparseBlockLdltFactor::factorize(mat, &analysis.sparse_pattern)?,
                )),
                SparseBackend::Auto | SparseBackend::FaerSupernodal => {
                    let fac = if let Some(pat) = faer_pattern {
                        FaerSparseLdltFactor::factorize_with_pattern(mat, pat)?
                    } else {
                        FaerSparseLdltFactor::factorize(mat)?
                    };
                    Ok(Self::FaerSparse(fac))
                }
            }
        };
        match analysis.route {
            SchurRoute::BlockDiagonalPairs => {
                let covered = analysis.single_blocks.len() + 2 * analysis.pair_blocks.len();
                if covered == analysis.num_free_bodies {
                    Ok(Self::Pairs(PairBlocksFactor::factorize(
                        mat,
                        &analysis.single_blocks,
                        &analysis.pair_blocks,
                    )?))
                } else {
                    factor_sparse()
                }
            }
            SchurRoute::TreeChain => {
                if analysis.cyclomatic_number == 0 {
                    Ok(Self::Tree(TreeBlockLdltFactor::factorize(
                        mat,
                        &analysis.tree_postorder,
                        &analysis.tree_parent,
                    )?))
                } else {
                    factor_sparse()
                }
            }
            SchurRoute::SparseGeneral => factor_sparse(),
        }
    }

    /// Factor `mat` with an explicit [`SparseBackend`] (zero-dep fallback when `faer-sparse` is off).
    #[cfg(not(feature = "faer-sparse"))]
    pub fn factorize_with_backend(
        mat: &RigidSchurMatrix,
        analysis: &GraphTopologyAnalysis,
        _backend: SparseBackend,
    ) -> Result<Self, LinearSolveError> {
        if analysis.num_free_bodies == 0 {
            return Ok(Self::Empty);
        }
        match analysis.route {
            SchurRoute::BlockDiagonalPairs => {
                let covered = analysis.single_blocks.len() + 2 * analysis.pair_blocks.len();
                if covered == analysis.num_free_bodies {
                    Ok(Self::Pairs(PairBlocksFactor::factorize(
                        mat,
                        &analysis.single_blocks,
                        &analysis.pair_blocks,
                    )?))
                } else {
                    Ok(Self::Sparse(SparseBlockLdltFactor::factorize(
                        mat,
                        &analysis.sparse_pattern,
                    )?))
                }
            }
            SchurRoute::TreeChain => {
                if analysis.cyclomatic_number == 0 {
                    Ok(Self::Tree(TreeBlockLdltFactor::factorize(
                        mat,
                        &analysis.tree_postorder,
                        &analysis.tree_parent,
                    )?))
                } else {
                    Ok(Self::Sparse(SparseBlockLdltFactor::factorize(
                        mat,
                        &analysis.sparse_pattern,
                    )?))
                }
            }
            SchurRoute::SparseGeneral => Ok(Self::Sparse(SparseBlockLdltFactor::factorize(
                mat,
                &analysis.sparse_pattern,
            )?)),
        }
    }

    /// Solve `H_RR_eff * x = rhs` for all free rigid body twists.
    pub fn solve(&self, rhs: &[Vec6]) -> Result<Vec<Vec6>, LinearSolveError> {
        match self {
            Self::Empty => Ok(Vec::new()),
            Self::Pairs(f) => f.solve(rhs),
            Self::Tree(f) => Ok(f.solve(rhs)),
            Self::Sparse(f) => Ok(f.solve(rhs)),
            #[cfg(feature = "faer-sparse")]
            Self::FaerSparse(f) => Ok(f.solve(rhs)),
        }
    }
}

/// Input tuple for parallel corotational beam Block-Thomas condensation.
pub struct CorotationalBeamInput<'a> {
    pub beam_id: usize,
    pub body_a: usize,
    pub body_b: Option<usize>,
    pub elem_a: &'a [Mat12],
    pub interior_k: &'a [Mat6],
    pub interior_rhs: &'a [Vec6],
}

/// Condense multiple corotational beams in parallel via Rayon when the beam count justifies it.
pub fn factor_corotational_beams_parallel(
    beam_inputs: &[CorotationalBeamInput<'_>],
) -> Result<Vec<(CorotationalBeamFactor, BeamSchurContribution)>, LinearSolveError> {
    let run = |inp: &CorotationalBeamInput<'_>| {
        CorotationalBeamFactor::factor_and_condense(
            inp.beam_id,
            inp.body_a,
            inp.body_b,
            inp.elem_a,
            inp.interior_k,
            inp.interior_rhs,
        )
    };
    if beam_inputs.len() >= 4 {
        beam_inputs.par_iter().map(run).collect()
    } else {
        beam_inputs.iter().map(run).collect()
    }
}

/// Direct dense Gaussian elimination with partial pivoting on an `n x n` matrix (used for
/// exact numerical verification of all 3 sparse Schur routes and fallback if dynamic fill occurs).
pub fn solve_dense_reference(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
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
            b.swap(k, piv);
        }
        let inv_diag = 1.0 / a[k][k];
        for i in (k + 1)..n {
            let f = a[i][k] * inv_diag;
            a[i][k] = 0.0;
            for j in (k + 1)..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fg_ir::{
        BeamEdgeSpec, BeamFormulation, JointEdgeSpec, JointType, RigidBodySpec, RigidFlexGraph,
    };
    use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};

    fn make_spd_mat6(seed: u64) -> Mat6 {
        let mut a = [[0.0; 6]; 6];
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (u32::MAX as f64) - 0.5
        };
        for row in &mut a {
            for v in row.iter_mut() {
                *v = next();
            }
        }
        let m = Mat6(a);
        m.transpose().mul_mat(m) + 4.0 * Mat6::IDENTITY
    }

    fn make_coupling_mat6(seed: u64) -> Mat6 {
        let mut a = [[0.0; 6]; 6];
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f64) / (u32::MAX as f64) - 0.5) * 0.25
        };
        for row in &mut a {
            for v in row.iter_mut() {
                *v = next();
            }
        }
        Mat6(a)
    }

    fn make_vec6(seed: u64) -> Vec6 {
        let mut v = [0.0; 6];
        let mut s = seed;
        for val in &mut v {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *val = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
        }
        Vec6(v)
    }

    fn make_body(id: usize, is_fixed: bool) -> RigidBodySpec {
        RigidBodySpec {
            id,
            name: format!("b{id}"),
            is_fixed,
            mass: 0.28,
            inertia: Mat3::diag([1e-3, 1e-3, 1e-3]),
            com_local: Vec3::ZERO,
            rest_pose: Se3::IDENTITY,
            collision_radius: 0.02,
            rayleigh_alpha: 0.05,
            tags: Vec::new(),
        }
    }

    fn make_beam(id: usize, body_a: usize, body_b: Option<usize>, n_el: usize) -> BeamEdgeSpec {
        let poses = (0..=n_el)
            .map(|i| Se3::new(Vec3::new(i as f64 * 0.02, 0.0, 0.0), Quat::IDENTITY))
            .collect();
        BeamEdgeSpec {
            id,
            name: format!("beam_{id}"),
            body_a,
            clamp_a: Se3::IDENTITY,
            body_b,
            clamp_b: body_b.map(|_| Se3::IDENTITY),
            n_elements: n_el,
            formulation: BeamFormulation::CorotationalEulerBernoulli,
            section: BeamSection::Rectangular {
                width: 0.016,
                thickness: 0.0002,
            },
            material: BeamMaterial::SPRING_STEEL,
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: Vec::new(),
            element_transforms: None,
            interior_node_contact_radius: None,
            enable_self_contact: false,
        }
    }

    fn verify_rigid_route_vs_dense(graph: &RigidFlexGraph, expected_route: SchurRoute) {
        let analysis = graph.analyze().expect("valid graph");
        assert_eq!(analysis.route, expected_route);
        let n = analysis.num_free_bodies;
        let mut mat = RigidSchurMatrix::zeros(n);
        let mut rhs = Vec::with_capacity(n);
        for u in 0..n {
            mat.add_block(u, u, make_spd_mat6(100 + u as u64));
            rhs.push(make_vec6(200 + u as u64));
        }
        // Add symmetric coupling blocks along every graph edge
        let mut edge_seed = 500u64;
        for beam in &graph.beams {
            if let Some(bb) = beam.body_b {
                if let (Some(u), Some(v)) = (
                    analysis.body_to_free_slot[beam.body_a],
                    analysis.body_to_free_slot[bb],
                ) {
                    let c = make_coupling_mat6(edge_seed);
                    edge_seed += 1;
                    mat.add_block(u, v, c);
                    mat.add_block(v, u, c.transpose());
                    // Maintain strong diagonal dominance -> guaranteed SPD
                    mat.add_block(u, u, 1.5 * Mat6::IDENTITY);
                    mat.add_block(v, v, 1.5 * Mat6::IDENTITY);
                }
            }
        }
        for joint in &graph.joints {
            if joint.implicit_coupling {
                if let (Some(u), Some(v)) = (
                    analysis.body_to_free_slot[joint.body_a],
                    analysis.body_to_free_slot[joint.body_b],
                ) {
                    let c = make_coupling_mat6(edge_seed);
                    edge_seed += 1;
                    mat.add_block(u, v, c);
                    mat.add_block(v, u, c.transpose());
                    mat.add_block(u, u, 1.5 * Mat6::IDENTITY);
                    mat.add_block(v, v, 1.5 * Mat6::IDENTITY);
                }
            }
        }

        let factor = RigidSchurFactor::factorize(&mat, &analysis).expect("factorize ok");
        let x_fast = factor.solve(&rhs).expect("solve ok");

        // Build exact 6N x 6N dense reference system
        let dim = 6 * n;
        let mut dense_a = vec![vec![0.0; dim]; dim];
        let mut dense_b = vec![0.0; dim];
        for u in 0..n {
            for r in 0..6 {
                dense_b[6 * u + r] = rhs[u][r];
            }
            for v in 0..n {
                let blk = mat.get_block(u, v);
                for r in 0..6 {
                    for c in 0..6 {
                        dense_a[6 * u + r][6 * v + c] = blk.0[r][c];
                    }
                }
            }
        }
        let x_dense = solve_dense_reference(dense_a, dense_b).expect("dense solve ok");
        let mut max_err = 0.0_f64;
        for u in 0..n {
            for r in 0..6 {
                max_err = max_err.max((x_fast[u][r] - x_dense[6 * u + r]).abs());
            }
        }
        assert!(
            max_err < 1e-12,
            "route {expected_route:?} vs dense max_err = {max_err:e}"
        );

        // When testing SparseGeneral (e.g. Tensegrity-6Bar), explicitly cross-check BOTH
        // pure-Rust Block-CSC and `faer` SIMD supernodal (fresh + cached pattern) < 1e-12!
        if expected_route == SchurRoute::SparseGeneral {
            let x_custom = SparseBlockLdltFactor::factorize(&mat, &analysis.sparse_pattern)
                .expect("custom sparse ok")
                .solve(&rhs);
            let mut err_custom = 0.0_f64;
            for u in 0..n {
                for r in 0..6 {
                    err_custom = err_custom.max((x_custom[u][r] - x_dense[6 * u + r]).abs());
                }
            }
            assert!(
                err_custom < 1e-12,
                "SparseGeneral custom Block-CSC vs dense max_err = {err_custom:e}"
            );

            #[cfg(feature = "faer-sparse")]
            {
                let pat = FaerSparsePattern::from_graph(graph, &analysis).expect("faer pattern ok");
                let x_faer = FaerSparseLdltFactor::factorize_with_pattern(&mat, &pat)
                    .expect("faer sparse ok")
                    .solve(&rhs);
                let mut err_faer = 0.0_f64;
                let mut err_cross = 0.0_f64;
                for u in 0..n {
                    for r in 0..6 {
                        err_faer = err_faer.max((x_faer[u][r] - x_dense[6 * u + r]).abs());
                        err_cross = err_cross.max((x_faer[u][r] - x_custom[u][r]).abs());
                    }
                }
                assert!(
                    err_faer < 1e-12,
                    "SparseGeneral faer vs dense max_err = {err_faer:e}"
                );
                assert!(
                    err_cross < 1e-12,
                    "SparseGeneral faer vs custom max_err = {err_cross:e}"
                );
            }
        }
    }

    #[test]
    fn test_all_three_schur_routes_match_dense_to_1e12() {
        // 1. BlockDiagonalPairs (3 disconnected pairs + 1 cantilever singleton)
        let g_pairs = RigidFlexGraph {
            name: "pairs".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: (0..7).map(|i| make_body(i, false)).collect(),
            beams: vec![
                make_beam(0, 0, Some(1), 4),
                make_beam(1, 2, Some(3), 4),
                make_beam(2, 4, Some(5), 4),
                make_beam(3, 6, None, 4),
            ],
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        verify_rigid_route_vs_dense(&g_pairs, SchurRoute::BlockDiagonalPairs);

        // 2. TreeChain (branched tree of 8 bodies + fixed base)
        let mut bodies_tree: Vec<_> = (0..9).map(|i| make_body(i, i == 0)).collect();
        bodies_tree[0].is_fixed = true;
        let g_tree = RigidFlexGraph {
            name: "tree".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: bodies_tree,
            beams: vec![
                make_beam(0, 0, Some(1), 3),
                make_beam(1, 1, Some(2), 3),
                make_beam(2, 2, Some(3), 3),
                make_beam(3, 2, Some(4), 3),
                make_beam(4, 3, Some(5), 3),
                make_beam(5, 4, Some(6), 3),
                make_beam(6, 4, Some(7), 3),
                make_beam(7, 7, Some(8), 3),
            ],
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        verify_rigid_route_vs_dense(&g_tree, SchurRoute::TreeChain);

        // 3. SparseGeneral (icosahedral 6-bar cyclic tensegrity: 6 bodies, 12 cyclic edges, beta_1 = 7)
        let tensegrity_edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 4), (1, 5),
            (2, 3), (2, 5),
            (3, 4), (3, 5),
            (4, 5),
        ];
        let joints: Vec<_> = tensegrity_edges
            .iter()
            .enumerate()
            .map(|(id, &(a, b))| JointEdgeSpec {
                id,
                name: format!("cable_{id}"),
                body_a: a,
                body_b: b,
                joint_type: JointType::CableSpring {
                    anchor_a: Vec3::ZERO,
                    anchor_b: Vec3::ZERO,
                    rest_length: 0.1,
                    stiffness: 500.0,
                    damping: 1.0,
                    compression_allowed: false,
                },
                actuator_id: None,
                kp: 0.0,
                kd: 0.0,
                torque_limit: 0.0,
                speed_limit: 0.0,
                static_friction: 0.0,
                deadband: 0.0,
                limits: None,
                implicit_coupling: true,
                solref_imp: None,
            })
            .collect();
        let g_cyclic = RigidFlexGraph {
            name: "tensegrity_6bar".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: (0..6).map(|i| make_body(i, false)).collect(),
            beams: Vec::new(),
            tendons: Vec::new(),
            joints,
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        verify_rigid_route_vs_dense(&g_cyclic, SchurRoute::SparseGeneral);
    }

    #[test]
    fn test_faer_and_custom_sparse_ldlt_match_dense_on_tensegrity_and_4x4_grid() {
        // Verify BOTH CustomBlockCsc and FaerSupernodal match dense reference to < 1e-12
        // on cyclic Tensegrity-6Bar (6 bodies, 12 edges, beta_1 = 7) as well as 4x4 lattice!
        let tensegrity_edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 4), (1, 5),
            (2, 3), (2, 5),
            (3, 4), (3, 5),
            (4, 5),
        ];
        let t_joints: Vec<_> = tensegrity_edges
            .iter()
            .enumerate()
            .map(|(id, &(a, b))| JointEdgeSpec {
                id,
                name: format!("tc_{id}"),
                body_a: a,
                body_b: b,
                joint_type: JointType::CableSpring {
                    anchor_a: Vec3::ZERO,
                    anchor_b: Vec3::ZERO,
                    rest_length: 0.1,
                    stiffness: 500.0,
                    damping: 1.0,
                    compression_allowed: false,
                },
                actuator_id: None,
                kp: 0.0,
                kd: 0.0,
                torque_limit: 0.0,
                speed_limit: 0.0,
                static_friction: 0.0,
                deadband: 0.0,
                limits: None,
                implicit_coupling: true,
                solref_imp: None,
            })
            .collect();
        let g_tens = RigidFlexGraph {
            name: "tensegrity_6bar_cross".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: (0..6).map(|i| make_body(i, false)).collect(),
            beams: Vec::new(),
            tendons: Vec::new(),
            joints: t_joints,
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let an_tens = g_tens.analyze().expect("valid tensegrity");
        assert_eq!(an_tens.route, SchurRoute::SparseGeneral);
        let mut mat_t = RigidSchurMatrix::zeros(6);
        let mut rhs_t = Vec::with_capacity(6);
        for u in 0..6 {
            mat_t.add_block(u, u, make_spd_mat6(700 + u as u64));
            rhs_t.push(make_vec6(800 + u as u64));
        }
        for (idx, &(a, b)) in tensegrity_edges.iter().enumerate() {
            let c = make_coupling_mat6(900 + idx as u64);
            mat_t.add_block(a, b, c);
            mat_t.add_block(b, a, c.transpose());
            mat_t.add_block(a, a, 2.0 * Mat6::IDENTITY);
            mat_t.add_block(b, b, 2.0 * Mat6::IDENTITY);
        }
        let mut da_t = vec![vec![0.0; 36]; 36];
        let mut db_t = vec![0.0; 36];
        for u in 0..6 {
            for r in 0..6 {
                db_t[6 * u + r] = rhs_t[u][r];
            }
            for v in 0..6 {
                let blk = mat_t.get_block(u, v);
                for r in 0..6 {
                    for c in 0..6 {
                        da_t[6 * u + r][6 * v + c] = blk.0[r][c];
                    }
                }
            }
        }
        let x_dense_t = solve_dense_reference(da_t, db_t).expect("tensegrity dense ok");
        let x_custom_t = SparseBlockLdltFactor::factorize(&mat_t, &an_tens.sparse_pattern)
            .expect("tensegrity custom ok")
            .solve(&rhs_t);
        let mut err_ct = 0.0_f64;
        for u in 0..6 {
            for r in 0..6 {
                err_ct = err_ct.max((x_custom_t[u][r] - x_dense_t[6 * u + r]).abs());
            }
        }
        assert!(err_ct < 1e-12, "custom sparse tensegrity err = {err_ct:e}");
        #[cfg(feature = "faer-sparse")]
        {
            let pat_t = FaerSparsePattern::from_graph(&g_tens, &an_tens).expect("tensegrity pat ok");
            let x_faer_t = FaerSparseLdltFactor::factorize_with_pattern(&mat_t, &pat_t)
                .expect("tensegrity faer ok")
                .solve(&rhs_t);
            let mut err_ft = 0.0_f64;
            let mut err_xt = 0.0_f64;
            for u in 0..6 {
                for r in 0..6 {
                    err_ft = err_ft.max((x_faer_t[u][r] - x_dense_t[6 * u + r]).abs());
                    err_xt = err_xt.max((x_faer_t[u][r] - x_custom_t[u][r]).abs());
                }
            }
            assert!(err_ft < 1e-12, "faer sparse tensegrity vs dense err = {err_ft:e}");
            assert!(err_xt < 1e-12, "faer vs custom tensegrity err = {err_xt:e}");
        }

        // 16-body, 24-edge 4x4 cyclic metamaterial lattice (96 rigid DOFs, beta_1 = 9, SparseGeneral)
        let mut joints = Vec::new();
        let mut jid = 0usize;
        for r in 0..4 {
            for c in 0..4 {
                let u = r * 4 + c;
                if c + 1 < 4 {
                    joints.push(JointEdgeSpec {
                        id: jid,
                        name: format!("h_{r}_{c}"),
                        body_a: u,
                        body_b: u + 1,
                        joint_type: JointType::CableSpring {
                            anchor_a: Vec3::ZERO,
                            anchor_b: Vec3::ZERO,
                            rest_length: 0.1,
                            stiffness: 600.0,
                            damping: 1.5,
                            compression_allowed: true,
                        },
                        actuator_id: None,
                        kp: 0.0,
                        kd: 0.0,
                        torque_limit: 0.0,
                        speed_limit: 0.0,
                        static_friction: 0.0,
                        deadband: 0.0,
                        limits: None,
                        implicit_coupling: true,
                        solref_imp: None,
                    });
                    jid += 1;
                }
                if r + 1 < 4 {
                    joints.push(JointEdgeSpec {
                        id: jid,
                        name: format!("v_{r}_{c}"),
                        body_a: u,
                        body_b: u + 4,
                        joint_type: JointType::CableSpring {
                            anchor_a: Vec3::ZERO,
                            anchor_b: Vec3::ZERO,
                            rest_length: 0.1,
                            stiffness: 600.0,
                            damping: 1.5,
                            compression_allowed: true,
                        },
                        actuator_id: None,
                        kp: 0.0,
                        kd: 0.0,
                        torque_limit: 0.0,
                        speed_limit: 0.0,
                        static_friction: 0.0,
                        deadband: 0.0,
                        limits: None,
                        implicit_coupling: true,
                        solref_imp: None,
                    });
                    jid += 1;
                }
            }
        }
        let g_grid = RigidFlexGraph {
            name: "grid_4x4".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: (0..16).map(|i| make_body(i, false)).collect(),
            beams: Vec::new(),
            tendons: Vec::new(),
            joints,
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let analysis = g_grid.analyze().expect("valid 4x4 grid");
        assert_eq!(analysis.route, SchurRoute::SparseGeneral);
        assert_eq!(analysis.cyclomatic_number, 9);

        let n = analysis.num_free_bodies;
        let mut mat = RigidSchurMatrix::zeros(n);
        let mut rhs = Vec::with_capacity(n);
        for u in 0..n {
            mat.add_block(u, u, make_spd_mat6(1000 + u as u64));
            rhs.push(make_vec6(2000 + u as u64));
        }
        for (idx, j) in g_grid.joints.iter().enumerate() {
            let c = make_coupling_mat6(3000 + idx as u64);
            mat.add_block(j.body_a, j.body_b, c);
            mat.add_block(j.body_b, j.body_a, c.transpose());
            mat.add_block(j.body_a, j.body_a, 2.0 * Mat6::IDENTITY);
            mat.add_block(j.body_b, j.body_b, 2.0 * Mat6::IDENTITY);
        }

        // Dense 96x96 reference
        let dim = 6 * n;
        let mut dense_a = vec![vec![0.0; dim]; dim];
        let mut dense_b = vec![0.0; dim];
        for u in 0..n {
            for r in 0..6 {
                dense_b[6 * u + r] = rhs[u][r];
            }
            for v in 0..n {
                let blk = mat.get_block(u, v);
                for r in 0..6 {
                    for c in 0..6 {
                        dense_a[6 * u + r][6 * v + c] = blk.0[r][c];
                    }
                }
            }
        }
        let x_dense = solve_dense_reference(dense_a, dense_b).expect("dense ok");

        // Custom pure-Rust Block-CSC LDL^T
        let x_custom = SparseBlockLdltFactor::factorize(&mat, &analysis.sparse_pattern)
            .expect("custom sparse ok")
            .solve(&rhs);
        let mut err_custom = 0.0_f64;
        for u in 0..n {
            for r in 0..6 {
                err_custom = err_custom.max((x_custom[u][r] - x_dense[6 * u + r]).abs());
            }
        }
        assert!(err_custom < 1e-12, "custom sparse 4x4 err = {err_custom:e}");

        // `faer` SIMD supernodal sparse LDL^T (both fresh and cached symbolic pattern)
        #[cfg(feature = "faer-sparse")]
        {
            let pat = FaerSparsePattern::from_graph(&g_grid, &analysis).expect("pattern ok");
            let x_faer = FaerSparseLdltFactor::factorize_with_pattern(&mat, &pat)
                .expect("faer sparse ok")
                .solve(&rhs);
            let mut err_faer = 0.0_f64;
            let mut err_cross = 0.0_f64;
            for u in 0..n {
                for r in 0..6 {
                    err_faer = err_faer.max((x_faer[u][r] - x_dense[6 * u + r]).abs());
                    err_cross = err_cross.max((x_faer[u][r] - x_custom[u][r]).abs());
                }
            }
            assert!(err_faer < 1e-12, "faer sparse 4x4 vs dense err = {err_faer:e}");
            assert!(err_cross < 1e-12, "faer vs custom 4x4 err = {err_cross:e}");
        }
    }

    #[test]
    fn test_block_thomas_clamped_and_cantilever_exact_vs_dense() {
        // Test both clamped-clamped (E=6) and free cantilever (E=6) against full 6*(E+1) dense solve
        let (k_loc, m_loc) = fg_math::local_beam_matrices(
            0.02,
            &BeamSection::Rectangular {
                width: 0.016,
                thickness: 0.0002,
            },
            &BeamMaterial::SPRING_STEEL,
            false,
            false,
        );
        let h = 0.0005;
        let elem = m_loc * (1.0 + h * 0.05) + k_loc * (h * 0.002 + h * h);
        let n_el = 6;
        let elems = vec![elem; n_el];
        let plate_a = make_spd_mat6(11);
        let plate_b = make_spd_mat6(22);

        // --- Case 1: Clamped-Clamped ---
        let n_free = n_el - 1;
        let int_k: Vec<Mat6> = (0..n_free).map(|i| make_spd_mat6(30 + i as u64)).collect();
        let int_rhs: Vec<Vec6> = (0..n_free).map(|i| make_vec6(40 + i as u64)).collect();
        let rhs_a = make_vec6(50);
        let rhs_b = make_vec6(51);

        let (fac, contrib) = CorotationalBeamFactor::factor_and_condense(
            0,
            0,
            Some(1),
            &elems,
            &int_k,
            &int_rhs,
        )
        .expect("factor ok");
        let h12 = Mat12::from_blocks6(
            plate_a + contrib.s_aa,
            contrib.s_ab,
            contrib.s_ba,
            plate_b + contrib.s_bb,
        );
        let r12 = Vec12::from_halves(rhs_a + contrib.r_a, rhs_b + contrib.r_b);
        let v_end = h12.solve_lu(r12).expect("12x12 solve");
        let v_int = fac.back_substitute(v_end.head(), v_end.tail());

        // Assemble full (n_el+1)*6 dense matrix
        let total_nodes = n_el + 1;
        let dim = total_nodes * 6;
        let mut da = vec![vec![0.0; dim]; dim];
        let mut db = vec![0.0; dim];
        let mut add6 = |ni: usize, nj: usize, blk: Mat6| {
            for r in 0..6 {
                for c in 0..6 {
                    da[6 * ni + r][6 * nj + c] += blk.0[r][c];
                }
            }
        };
        add6(0, 0, plate_a);
        add6(n_el, n_el, plate_b);
        for e in 0..n_el {
            add6(e, e, elems[e].block6(0, 0));
            add6(e, e + 1, elems[e].block6(0, 1));
            add6(e + 1, e, elems[e].block6(1, 0));
            add6(e + 1, e + 1, elems[e].block6(1, 1));
        }
        for i in 0..n_free {
            add6(i + 1, i + 1, int_k[i]);
            for r in 0..6 {
                db[6 * (i + 1) + r] = int_rhs[i][r];
            }
        }
        for r in 0..6 {
            db[r] = rhs_a[r];
            db[6 * n_el + r] = rhs_b[r];
        }
        let x_ref = solve_dense_reference(da, db).expect("dense ok");
        let mut err = 0.0_f64;
        for r in 0..6 {
            err = err.max((v_end.head()[r] - x_ref[r]).abs());
            err = err.max((v_end.tail()[r] - x_ref[6 * n_el + r]).abs());
        }
        for i in 0..n_free {
            for r in 0..6 {
                err = err.max((v_int[i][r] - x_ref[6 * (i + 1) + r]).abs());
            }
        }
        assert!(err < 1e-12, "clamped block-thomas err = {err:e}");

        // --- Case 2: Cantilever (body_b = None) ---
        let int_k_c: Vec<Mat6> = (0..n_el).map(|i| make_spd_mat6(60 + i as u64)).collect();
        let int_rhs_c: Vec<Vec6> = (0..n_el).map(|i| make_vec6(70 + i as u64)).collect();
        let (fac_c, contrib_c) = CorotationalBeamFactor::factor_and_condense(
            1,
            0,
            None,
            &elems,
            &int_k_c,
            &int_rhs_c,
        )
        .expect("cantilever factor ok");
        let va_c = (plate_a + contrib_c.s_aa)
            .solve_lu(rhs_a + contrib_c.r_a)
            .expect("6x6 solve");
        let v_int_c = fac_c.back_substitute(va_c, Vec6::ZERO);

        let mut da_c = vec![vec![0.0; dim]; dim];
        let mut db_c = vec![0.0; dim];
        let mut add6_c = |ni: usize, nj: usize, blk: Mat6| {
            for r in 0..6 {
                for c in 0..6 {
                    da_c[6 * ni + r][6 * nj + c] += blk.0[r][c];
                }
            }
        };
        add6_c(0, 0, plate_a);
        for e in 0..n_el {
            add6_c(e, e, elems[e].block6(0, 0));
            add6_c(e, e + 1, elems[e].block6(0, 1));
            add6_c(e + 1, e, elems[e].block6(1, 0));
            add6_c(e + 1, e + 1, elems[e].block6(1, 1));
        }
        for i in 0..n_el {
            add6_c(i + 1, i + 1, int_k_c[i]);
            for r in 0..6 {
                db_c[6 * (i + 1) + r] = int_rhs_c[i][r];
            }
        }
        for r in 0..6 {
            db_c[r] = rhs_a[r];
        }
        let x_ref_c = solve_dense_reference(da_c, db_c).expect("dense cantilever ok");
        let mut err_c = 0.0_f64;
        for r in 0..6 {
            err_c = err_c.max((va_c[r] - x_ref_c[r]).abs());
        }
        for i in 0..n_el {
            for r in 0..6 {
                err_c = err_c.max((v_int_c[i][r] - x_ref_c[6 * (i + 1) + r]).abs());
            }
        }
        assert!(err_c < 1e-12, "cantilever block-thomas err = {err_c:e}");
    }

    #[test]
    fn test_symmetric_banded_ldlt_exact_vs_dense() {
        let n = 25;
        let bw = 7;
        let mut banded = SymmetricBandedLdlt::zeros(n, bw);
        let mut dense = vec![vec![0.0; n]; n];
        let mut s = 999u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (u32::MAX as f64) - 0.5
        };
        for i in 0..n {
            let d = 15.0 + next().abs();
            banded.add_lower(i, i, d);
            dense[i][i] = d;
            for j in i.saturating_sub(bw)..i {
                let v = next();
                banded.add_lower(i, j, v);
                dense[i][j] = v;
                dense[j][i] = v;
            }
        }
        let rhs: Vec<f64> = (0..n).map(|_| next()).collect();
        let x_ref = solve_dense_reference(dense, rhs.clone()).expect("dense ok");
        banded.factorize(0).expect("banded ok");
        let mut x = rhs;
        banded.solve_in_place(&mut x);
        let max_err = x
            .iter()
            .zip(x_ref.iter())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(max_err < 1e-13, "banded ldlt err = {max_err:e}");
    }
}
