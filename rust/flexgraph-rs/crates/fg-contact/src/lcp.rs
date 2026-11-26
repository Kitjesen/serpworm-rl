//! Compliance-Based Frictional Contact LCP & Delassus-Schur Solvers
//! (Shen, Ha & Chen, IJRR 2026 `arXiv:2611.18546`, Moreau Contact-Implicit,
//! Goulart & Chen Clarabel SOCP 2024, and MuJoCo `SolRefImp` regularization).
//!
//! Solves the exact 3D maximum-dissipation Signorini-Coulomb complementarity problem:
//! $$u = b_{\mathrm{eff}} + W_{\mathrm{eff}} \lambda, \qquad 0 \le u_{n,i} \perp \lambda_{n,i} \ge 0, \qquad
//! \lambda_{t,i} \in \arg\min_{y \in \mathcal{E}(\mu_{1,i}\lambda_{n,i},\,\mu_{2,i}\lambda_{n,i})} u_{t,i}^\top y$$
//! via four consistent solvers sharing the exact same canonical maximum-dissipation solution:
//! 1. **Block Projected Gauss-Seidel (PGS / Moreau-Jean NSGS)** solving each contact's exact
//!    2D SPD elliptic-cone QP `(W_tt + s A) lambda_t = -b_eff` via Newton secular root-finding;
//! 2. **Alart-Curnier Semismooth Newton** with exact Euclidean ellipse projection
//!    $\Pi_{\mathcal{E}}^{\mathrm{Eucl}}$ and exact $3m\times 3m$ Clarke generalized Jacobian;
//! 3. **Fischer-Burmeister Polyhedral $m$-Facet LCP** (Shen et al., Eq. 33–34) solving each
//!    contact's exact $k$-gon tangential QP with coupled slip multiplier $\gamma_i \ge 0$,
//!    converging monotonically at $O(1/k^2)$ to the continuous cone as $k \to \infty$;
//! 4. **Clarabel Interior-Point 3D Second-Order Cone Program (`ClarabelSocp`)** with De Saxcé /
//!    Acary-Cadoux normal-lift elimination, matching `AlartCurnierNewton` to `< 1e-8` on
//!    well-conditioned problems while converging robustly on extreme mass/stiffness ratios
//!    ($\kappa \ge 10^8$) where Gauss-Seidel stalls.

#[cfg(feature = "clarabel-socp")]
use clarabel::algebra::CscMatrix;
#[cfg(feature = "clarabel-socp")]
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use fg_math::{SolRefImp, Vec3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

fn default_solref_dt() -> f64 {
    0.002
}

/// Single 3D frictional contact constraint in local orthonormal frame `(normal, t1, t2)`.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LcpContactConstraint {
    pub normal: Vec3,
    pub t1: Vec3,
    pub t2: Vec3,
    /// Coulomb friction coefficient along `t1` (`>= 0`).
    pub mu_1: f64,
    /// Coulomb friction coefficient along `t2` (`>= 0`; set equal to `mu_1` for isotropic cone).
    pub mu_2: f64,
    /// Unconstrained (free-motion) contact velocity or displacement increment `[v_n, v_t1, v_t2]`.
    pub free_vel: [f64; 3],
    /// Signed gap error `g_0` (`< 0` if penetrating; Baumgarte/depth correction `b_n = free_vel[0] + gap / dt`).
    pub gap: f64,
    /// Normal Newton restitution coefficient `e in [0, 1]` (`0.0` for inelastic / quasi-static).
    pub restitution: f64,
    /// Optional MuJoCo soft-constraint regularization (`solref`/`solimp`): adds exact diagonal
    /// compliance regularizer `R = ((1 - d(r)) / d(r)) * W_ii` and MuJoCo spring-damper reference bias.
    #[serde(default)]
    pub solref_imp: Option<SolRefImp>,
    /// Optional pre-step normal velocity `v_n^-` (m/s) used by MuJoCo `SolRefImp` dynamic damping
    /// `(dt * B - 1) * v_n^-` when `solref_imp = Some(_)` and `gap <= 0.0`. Defaults to `0.0` when `None`.
    #[serde(default)]
    pub pre_vel_n: Option<f64>,
}

/// Alias for [`LcpContactConstraint`].
pub type ContactConstraint = LcpContactConstraint;

impl LcpContactConstraint {
    /// Build orthonormal tangent basis `(t1, t2)` automatically from a unit normal.
    pub fn from_normal(normal: Vec3, mu: f64, free_vel: [f64; 3]) -> Self {
        let n = normal.normalized_or(Vec3::EZ);
        let ref_axis = if n.z().abs() < 0.9 {
            Vec3::EZ
        } else {
            Vec3::EX
        };
        let t1 = ref_axis.cross(n).normalized();
        let t2 = n.cross(t1).normalized();
        Self {
            normal: n,
            t1,
            t2,
            mu_1: mu.max(0.0),
            mu_2: mu.max(0.0),
            free_vel,
            gap: 0.0,
            restitution: 0.0,
            solref_imp: None,
            pre_vel_n: None,
        }
    }

    /// Attach MuJoCo [`SolRefImp`] soft-constraint regularization parameters.
    #[inline]
    pub fn with_solref_imp(mut self, solref_imp: SolRefImp) -> Self {
        self.solref_imp = Some(solref_imp);
        self
    }

    /// Attach pre-step normal velocity `v_n^-` (m/s) for exact MuJoCo `SolRefImp` dynamic damping.
    #[inline]
    pub fn with_pre_vel_n(mut self, pre_vel_n: f64) -> Self {
        self.pre_vel_n = Some(pre_vel_n);
        self
    }
}

/// Solver method selector for [`DelassusSystem::solve`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LcpSolverMethod {
    /// Block Projected Gauss-Seidel / Moreau-Jean NSGS with exact 2D SPD elliptic-cone QP per contact.
    BlockPgs,
    /// Alart-Curnier semismooth Newton with exact Euclidean ellipse projection & Clarke Jacobian.
    AlartCurnierNewton,
    /// Shen et al. (IJRR 2026 Eq. 33–34) polyhedral $m$-facet LCP with exact $\gamma$-coupled $k$-gon QP.
    FischerBurmeisterPolyhedral,
    /// Exact 3D Second-Order Friction-Cone SOCP via `clarabel` interior-point QDLDL solver
    /// with De Saxcé / Acary-Cadoux normal-lift elimination (falls back to `AlartCurnierNewton`
    /// when compiled with `--no-default-features`).
    ClarabelSocp,
}

/// Alias for [`LcpSolverMethod`].
pub type LcpSolverKind = LcpSolverMethod;

/// Configuration for the frictional complementarity solver.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LcpSolverConfig {
    pub method: LcpSolverMethod,
    pub max_iters: usize,
    pub tol: f64,
    /// Successive over-relaxation factor for PGS (`1.0` = standard Gauss-Seidel).
    pub omega_sor: f64,
    /// Diagonal Tikhonov compliance regularization for near-singular contact sets.
    pub compliance_reg: f64,
    /// Number of friction-cone facets for polyhedral LCP mode (default `16`).
    pub polyhedral_facets: usize,
    /// Timestep `dt` (s) used to map MuJoCo `SolRefImp` reference stiffness into velocity-level
    /// contact bias when `LcpContactConstraint::solref_imp` is `Some(_)`.
    #[serde(default = "default_solref_dt")]
    pub solref_dt: f64,
}

impl Default for LcpSolverConfig {
    fn default() -> Self {
        Self {
            method: LcpSolverMethod::BlockPgs,
            max_iters: 300,
            tol: 1.0e-11,
            omega_sor: 1.0,
            compliance_reg: 1.0e-12,
            polyhedral_facets: 16,
            solref_dt: default_solref_dt(),
        }
    }
}

/// Solution and rigorous complementarity verification diagnostics.
#[derive(Clone, Debug)]
pub struct LcpSolution {
    /// Solved contact impulses/forces `[lambda_n, lambda_t1, lambda_t2]` per contact.
    pub impulses: Vec<[f64; 3]>,
    /// Post-contact local velocities/displacements `u = b + W * lambda` per contact.
    pub post_vel: Vec<[f64; 3]>,
    /// Number of iterations taken.
    pub iterations: usize,
    pub converged: bool,
    /// Max Signorini normal complementarity error `max_i (|u_n * lam_n|, max(0, -u_n), max(0, -lam_n))`.
    pub normal_complementarity_res: f64,
    /// Max Coulomb friction cone & maximum-dissipation proximal residual.
    pub friction_cone_res: f64,
    /// Max Fischer-Burmeister NCP normal residual `max_i |sqrt(u_n^2 + lam_n^2) - u_n - lam_n|`.
    pub fischer_burmeister_res: f64,
}

/// Dense $3m \times 3m$ Delassus / compliance operator system $u = b + W \lambda$.
#[derive(Clone, Debug)]
pub struct DelassusSystem {
    pub contacts: Vec<LcpContactConstraint>,
    /// Row-major `3m x 3m` symmetric compliance/Delassus matrix `W = J_c H^{-1} J_c^T`.
    pub w_matrix: Vec<f64>,
}

impl DelassusSystem {
    /// Construct a Delassus system from `m` contacts and row-major `3m x 3m` matrix `W`.
    pub fn new(contacts: Vec<LcpContactConstraint>, w_matrix: Vec<f64>) -> Self {
        let dim = 3 * contacts.len();
        assert_eq!(
            w_matrix.len(),
            dim * dim,
            "Delassus matrix must have length (3m)^2"
        );
        Self { contacts, w_matrix }
    }

    /// Deduplicate clustered redundant contacts within distance `merge_radius` (Shen et al. IJRR 2026
    /// Sec. Contact Detection), retaining the deepest penetration representative per cluster so `C_c`
    /// remains well-conditioned. Returns the indices of retained contacts.
    pub fn deduplicate_contacts(
        positions: &[Vec3],
        penetrations: &[f64],
        merge_radius: f64,
    ) -> Vec<usize> {
        let mut order: Vec<usize> = (0..positions.len()).collect();
        order.sort_by(|&i, &j| {
            penetrations[j]
                .partial_cmp(&penetrations[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let r2 = merge_radius * merge_radius;
        let mut kept = Vec::new();
        for idx in order {
            if penetrations[idx] <= 0.0 {
                continue;
            }
            let too_close = kept
                .iter()
                .any(|&k: &usize| (positions[idx] - positions[k]).norm_sq() < r2);
            if !too_close {
                kept.push(idx);
            }
        }
        kept.sort_unstable();
        kept
    }

    /// Solve the Signorini-Coulomb complementarity problem using the requested method.
    pub fn solve(&self, cfg: &LcpSolverConfig, warm_start: Option<&[[f64; 3]]>) -> LcpSolution {
        let m = self.contacts.len();
        if m == 0 {
            return LcpSolution {
                impulses: Vec::new(),
                post_vel: Vec::new(),
                iterations: 0,
                converged: true,
                normal_complementarity_res: 0.0,
                friction_cone_res: 0.0,
                fischer_burmeister_res: 0.0,
            };
        }
        match cfg.method {
            LcpSolverMethod::BlockPgs => self.solve_pgs(cfg, warm_start),
            LcpSolverMethod::AlartCurnierNewton => self.solve_alart_curnier_newton(cfg, warm_start),
            LcpSolverMethod::FischerBurmeisterPolyhedral => {
                self.solve_polyhedral_lcp(cfg, warm_start)
            }
            LcpSolverMethod::ClarabelSocp => {
                #[cfg(feature = "clarabel-socp")]
                {
                    self.solve_clarabel_socp(cfg, warm_start)
                }
                #[cfg(not(feature = "clarabel-socp"))]
                {
                    self.solve_alart_curnier_newton(cfg, warm_start)
                }
            }
        }
    }

    /// Build the effective Delassus matrix and free-velocity bias `(W_eff, b_eff)`,
    /// applying MuJoCo `SolRefImp` diagonal compliance regularization `R = ((1 - d(r))/d(r)) * W_ii`
    /// and spring-damper reference bias for every contact with `solref_imp = Some(_)`.
    /// When all contacts have `solref_imp == None`, returns `(w_matrix, free_vec)` unchanged.
    pub fn effective_system(&self, cfg: &LcpSolverConfig) -> (Vec<f64>, Vec<f64>) {
        let m = self.contacts.len();
        let dim = 3 * m;
        let mut w_eff = self.w_matrix.clone();
        let mut b_eff = Vec::with_capacity(dim);
        let dt = cfg.solref_dt.max(1e-9);

        for (i, c) in self.contacts.iter().enumerate() {
            let vn_eff = if c.free_vel[0] < 0.0 {
                (1.0 + c.restitution) * c.free_vel[0]
            } else {
                c.free_vel[0]
            };
            if let Some(sri) = c.solref_imp {
                let r_mag = (-c.gap).max(0.0);
                let [k_kb, b_kb, imp, _] = sri.kbip(r_mag, 2.0 * dt);
                for k in 0..3 {
                    let idx = (3 * i + k) * dim + (3 * i + k);
                    w_eff[idx] += sri.constraint_regularizer(r_mag, self.w_matrix[idx]);
                }
                // Exact MuJoCo velocity-level bias from `(W + R)*(lam/dt) + a0 - aref = 0`
                // with `a0 = (v_free - v_n)/dt` and `aref = -B*v_n - K*d*gap`:
                //   b_eff_n = v_free + (dt*B - 1)*v_n + dt*K*d*gap.
                // Recovers exact closed-loop dynamics `a1 + (d*B)*v_n + (d^2*K)*gap = (1 - d)*a0`
                // when `pre_vel_n = Some(v_n)`, and exact static equilibrium `(-gap) = ((1-d)/(d^2*K))*g`
                // when `pre_vel_n = None` (or `Some(0.0)`).
                // For strictly separated contacts (`gap > 0`), use ballistic `gap / dt` so
                // inactive contacts do not prematurely activate.
                if c.gap <= 0.0 {
                    let vn_cur = c.pre_vel_n.unwrap_or(0.0);
                    b_eff.push(vn_eff + (dt * b_kb - 1.0) * vn_cur + dt * k_kb * imp * c.gap);
                } else {
                    b_eff.push(vn_eff + c.gap / dt);
                }
            } else {
                b_eff.push(vn_eff + c.gap);
            }
            b_eff.push(c.free_vel[1]);
            b_eff.push(c.free_vel[2]);
        }
        (w_eff, b_eff)
    }

    #[inline]
    fn mat_vec_with(w: &[f64], lam: &[[f64; 3]], b: &[f64]) -> Vec<[f64; 3]> {
        let m = lam.len();
        let dim = 3 * m;
        let mut u = vec![[0.0; 3]; m];
        for i in 0..m {
            for r in 0..3 {
                let row = (3 * i + r) * dim;
                let mut sum = b[3 * i + r];
                for j in 0..m {
                    let lj = lam[j];
                    sum += w[row + 3 * j] * lj[0]
                        + w[row + 3 * j + 1] * lj[1]
                        + w[row + 3 * j + 2] * lj[2];
                }
                u[i][r] = sum;
            }
        }
        u
    }

    /// Evaluate exact Signorini, maximum-dissipation Coulomb cone, and Fischer-Burmeister residuals.
    pub fn evaluate_residuals(&self, lam: &[[f64; 3]], u: &[[f64; 3]]) -> (f64, f64, f64) {
        let mut norm_res = 0.0_f64;
        let mut fric_res = 0.0_f64;
        let mut fb_res = 0.0_f64;
        for (i, c) in self.contacts.iter().enumerate() {
            let ln = lam[i][0];
            let un = u[i][0];
            norm_res = norm_res
                .max((-ln).max(0.0))
                .max((-un).max(0.0))
                .max((ln * un).abs());
            fb_res = fb_res.max(fischer_burmeister(un, ln).abs());

            let lt1 = lam[i][1];
            let lt2 = lam[i][2];
            let ut1 = u[i][1];
            let ut2 = u[i][2];
            let c1 = c.mu_1 * ln.max(0.0);
            let c2 = c.mu_2 * ln.max(0.0);

            if c1 <= 1e-15 && c2 <= 1e-15 {
                fric_res = fric_res.max(lt1.hypot(lt2));
            } else {
                let e1 = if c1 > 1e-15 { lt1 / c1 } else { 0.0 };
                let e2 = if c2 > 1e-15 { lt2 / c2 } else { 0.0 };
                let ell = e1.hypot(e2);
                fric_res = fric_res.max((ell - 1.0).max(0.0) * c1.max(c2));
                // Exact Euclidean maximum-dissipation Alart-Curnier proximal residual:
                // zero iff u_t == 0 (stick) or -u_t lies in the outward normal cone of the friction ellipse!
                let proj = project_ellipse_eucl(lt1 - ut1, lt2 - ut2, c1, c2);
                fric_res = fric_res.max((lt1 - proj[0]).hypot(lt2 - proj[1]));
            }
        }
        (norm_res, fric_res, fb_res)
    }

    /// Block Projected Gauss-Seidel (Moreau-Jean NSGS) solving each contact's exact 2D SPD
    /// elliptic-cone QP `(W_tt + s A) lambda_t = -b_eff` (guaranteed exact on arbitrary
    /// non-scalar `W_tt` and anisotropic cones!).
    fn solve_pgs(&self, cfg: &LcpSolverConfig, warm_start: Option<&[[f64; 3]]>) -> LcpSolution {
        let m = self.contacts.len();
        let dim = 3 * m;
        let (w_eff, b) = self.effective_system(cfg);
        let mut lam = match warm_start {
            Some(ws) if ws.len() == m => ws.to_vec(),
            _ => vec![[0.0; 3]; m],
        };
        let omega = cfg.omega_sor.clamp(0.2, 1.8);
        let reg = cfg.compliance_reg.max(0.0);

        let mut iters = 0;
        let mut converged = false;

        for iter in 0..cfg.max_iters {
            iters = iter + 1;
            let mut max_delta = 0.0_f64;

            for i in 0..m {
                let c = &self.contacts[i];
                let r0 = (3 * i) * dim;
                let r1 = (3 * i + 1) * dim;
                let r2 = (3 * i + 2) * dim;

                // Compute off-diagonal residual (excluding contact i's own block W_ii * lam_i)
                let mut rhs0 = b[3 * i];
                let mut rhs1 = b[3 * i + 1];
                let mut rhs2 = b[3 * i + 2];
                for j in 0..m {
                    if j == i {
                        continue;
                    }
                    let lj = lam[j];
                    let col = 3 * j;
                    rhs0 += w_eff[r0 + col] * lj[0]
                        + w_eff[r0 + col + 1] * lj[1]
                        + w_eff[r0 + col + 2] * lj[2];
                    rhs1 += w_eff[r1 + col] * lj[0]
                        + w_eff[r1 + col + 1] * lj[1]
                        + w_eff[r1 + col + 2] * lj[2];
                    rhs2 += w_eff[r2 + col] * lj[0]
                        + w_eff[r2 + col + 1] * lj[1]
                        + w_eff[r2 + col + 2] * lj[2];
                }

                let w00 = w_eff[r0 + 3 * i] + reg;
                let w01 = w_eff[r0 + 3 * i + 1];
                let w02 = w_eff[r0 + 3 * i + 2];
                let w10 = w_eff[r1 + 3 * i];
                let w20 = w_eff[r2 + 3 * i];
                let w11 = w_eff[r1 + 3 * i + 1] + reg;
                let w12 = 0.5 * (w_eff[r1 + 3 * i + 2] + w_eff[r2 + 3 * i + 1]);
                let w22 = w_eff[r2 + 3 * i + 2] + reg;

                let old = lam[i];

                // 1. Exact normal Signorini update with SOR relaxation
                let b_n = rhs0 + w01 * old[1] + w02 * old[2];
                let ln_exact = if w00 > 1e-18 { (-b_n / w00).max(0.0) } else { 0.0 };
                let ln_new = ((1.0 - omega) * old[0] + omega * ln_exact).max(0.0);

                // 2. Exact 2D tangential SPD elliptic-cone QP solve:
                //    min_{y in E(c1, c2)} 0.5 y^T W_tt y + b_t^T y, where b_t = rhs_t + W_tn * ln_new
                let bt1 = rhs1 + w10 * ln_new;
                let bt2 = rhs2 + w20 * ln_new;
                let c1 = c.mu_1 * ln_new;
                let c2 = c.mu_2 * ln_new;
                let lt_exact = solve_2d_ellipse_qp(w11, w12, w22, bt1, bt2, c1, c2);
                let lt_cand = [
                    (1.0 - omega) * old[1] + omega * lt_exact[0],
                    (1.0 - omega) * old[2] + omega * lt_exact[1],
                ];
                let lt_new = if (omega - 1.0).abs() > 1e-12 {
                    project_ellipse_eucl(lt_cand[0], lt_cand[1], c1, c2)
                } else {
                    lt_exact
                };

                lam[i] = [ln_new, lt_new[0], lt_new[1]];
                let delta = (ln_new - old[0])
                    .abs()
                    .max((lt_new[0] - old[1]).abs())
                    .max((lt_new[1] - old[2]).abs());
                max_delta = max_delta.max(delta);
            }

            if max_delta < cfg.tol {
                let u_check = Self::mat_vec_with(&w_eff, &lam, &b);
                let (nr, fr, _) = self.evaluate_residuals(&lam, &u_check);
                if nr.max(fr) < cfg.tol * 10.0 {
                    converged = true;
                    break;
                }
            }
        }

        let u = Self::mat_vec_with(&w_eff, &lam, &b);
        let (nr, fr, fbr) = self.evaluate_residuals(&lam, &u);
        LcpSolution {
            impulses: lam,
            post_vel: u,
            iterations: iters,
            converged: converged || nr.max(fr) < cfg.tol * 10.0,
            normal_complementarity_res: nr,
            friction_cone_res: fr,
            fischer_burmeister_res: fbr,
        }
    }

    /// Alart-Curnier Semismooth Newton solver with exact Euclidean ellipse projection,
    /// exact $3m \times 3m$ Clarke generalized Jacobian, and Armijo merit backtracking.
    fn solve_alart_curnier_newton(
        &self,
        cfg: &LcpSolverConfig,
        warm_start: Option<&[[f64; 3]]>,
    ) -> LcpSolution {
        let (w_eff, b) = self.effective_system(cfg);

        // Warm-start with 10 cheap PGS sweeps to identify initial active contact/stick/slip sets.
        let init_lam = {
            let warm_cfg = LcpSolverConfig {
                method: LcpSolverMethod::BlockPgs,
                max_iters: 10,
                tol: cfg.tol,
                ..*cfg
            };
            self.solve_pgs(&warm_cfg, warm_start).impulses
        };

        let (lam, iters, converged, lu_failed) =
            self.refine_alart_curnier(&w_eff, &b, &init_lam, cfg, cfg.max_iters, true);
        if lu_failed {
            return self.solve_pgs(cfg, Some(&lam));
        }

        let u = Self::mat_vec_with(&w_eff, &lam, &b);
        let (nr, fr, fbr) = self.evaluate_residuals(&lam, &u);
        LcpSolution {
            impulses: lam,
            post_vel: u,
            iterations: iters,
            converged: converged || nr.max(fr) < cfg.tol * 10.0,
            normal_complementarity_res: nr,
            friction_cone_res: fr,
            fischer_burmeister_res: fbr,
        }
    }

    /// Core safeguarded Alart-Curnier semismooth Newton refinement starting from `init_lam`.
    /// Shared by [`solve_alart_curnier_newton`] and IPM-to-Newton crossover polish in `ClarabelSocp`.
    fn refine_alart_curnier(
        &self,
        w_eff: &[f64],
        b: &[f64],
        init_lam: &[[f64; 3]],
        cfg: &LcpSolverConfig,
        max_iters: usize,
        allow_pgs_step: bool,
    ) -> (Vec<[f64; 3]>, usize, bool, bool) {
        let m = self.contacts.len();
        let dim = 3 * m;
        let mut lam = init_lam.to_vec();

        // Per-contact scaling parameters r_n, r_t clamped to [0.1, 100.0] so Alart-Curnier
        // velocity residuals `r * u` stay commensurate with cone-projection residuals even on
        // extreme stiffness/mass ratios (`kappa >= 1e8` where `1 / W_ii ~ 1e-8`).
        let mut rn = vec![1.0; m];
        let mut rt = vec![1.0; m];
        for i in 0..m {
            let w00 = w_eff[(3 * i) * dim + 3 * i].max(1e-9);
            let w11 = w_eff[(3 * i + 1) * dim + 3 * i + 1];
            let w12 = w_eff[(3 * i + 1) * dim + 3 * i + 2];
            let w22 = w_eff[(3 * i + 2) * dim + 3 * i + 2];
            let lmax = (0.5 * (w11 + w22) + (0.5 * (w11 - w22)).hypot(w12)).max(1e-9);
            rn[i] = (1.0 / w00).clamp(0.1, 100.0);
            rt[i] = (1.0 / lmax).clamp(0.1, 100.0);
        }

        let eval_ac = |lam_in: &[[f64; 3]]| -> (Vec<f64>, Vec<[f64; 3]>) {
            let u = Self::mat_vec_with(w_eff, lam_in, b);
            let mut f = vec![0.0; dim];
            for i in 0..m {
                let c = &self.contacts[i];
                let zn = lam_in[i][0] - rn[i] * u[i][0];
                let pn = zn.max(0.0);
                f[3 * i] = lam_in[i][0] - pn;

                let zt1 = lam_in[i][1] - rt[i] * u[i][1];
                let zt2 = lam_in[i][2] - rt[i] * u[i][2];
                let pt = project_ellipse_eucl(zt1, zt2, c.mu_1 * pn, c.mu_2 * pn);
                f[3 * i + 1] = lam_in[i][1] - pt[0];
                f[3 * i + 2] = lam_in[i][2] - pt[1];
            }
            (f, u)
        };

        let mut iters = 0;
        let mut converged = false;

        for iter in 0..max_iters {
            iters = iter + 1;
            let (f, u) = eval_ac(&lam);
            let merit = 0.5 * f.iter().map(|v| v * v).sum::<f64>();
            let f_inf = f.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            let (nr_cur, fr_cur, _) = self.evaluate_residuals(&lam, &u);
            if f_inf < cfg.tol && nr_cur.max(fr_cur) < cfg.tol {
                converged = true;
                break;
            }

            // Assemble exact 3m x 3m Clarke generalized Jacobian J = dF / d(lam).
            let mut jac = vec![0.0; dim * dim];
            for i in 0..m {
                let c = &self.contacts[i];
                let zn = lam[i][0] - rn[i] * u[i][0];
                let zt1 = lam[i][1] - rt[i] * u[i][1];
                let zt2 = lam[i][2] - rt[i] * u[i][2];

                let p_loc = ellipse_proj_jacobian(zn, zt1, zt2, c.mu_1, c.mu_2);

                let r_diag = [rn[i], rt[i], rt[i]];
                for r in 0..3 {
                    let gi = 3 * i + r;
                    jac[gi * dim + gi] += 1.0;
                    for k in 0..3 {
                        let pk = p_loc[r][k];
                        if pk == 0.0 {
                            continue;
                        }
                        let gk = 3 * i + k;
                        jac[gi * dim + gk] -= pk;
                        let pr = pk * r_diag[k];
                        for col in 0..dim {
                            jac[gi * dim + col] += pr * w_eff[gk * dim + col];
                        }
                    }
                }
            }

            for d in 0..dim {
                jac[d * dim + d] += cfg.compliance_reg;
            }
            let rhs: Vec<f64> = f.iter().map(|&v| -v).collect();
            let Some(delta) = solve_dense_lu(&jac, &rhs, dim) else {
                return (lam, iters, false, true);
            };

            let mut alpha = 1.0;
            let mut accepted = false;
            let res_cur = nr_cur.max(fr_cur);
            for _ in 0..10 {
                let mut cand = lam.clone();
                for i in 0..m {
                    cand[i][0] += alpha * delta[3 * i];
                    cand[i][1] += alpha * delta[3 * i + 1];
                    cand[i][2] += alpha * delta[3 * i + 2];
                }
                let (f_cand, u_cand) = eval_ac(&cand);
                let merit_cand = 0.5 * f_cand.iter().map(|v| v * v).sum::<f64>();
                let (nr_c, fr_c, _) = self.evaluate_residuals(&cand, &u_cand);
                if merit_cand < (1.0 - 1e-4 * alpha) * merit
                    || merit_cand == 0.0
                    || nr_c.max(fr_c) < 0.99 * res_cur
                {
                    lam = cand;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                if allow_pgs_step {
                    let step_cfg = LcpSolverConfig {
                        method: LcpSolverMethod::BlockPgs,
                        max_iters: 5,
                        tol: cfg.tol,
                        ..*cfg
                    };
                    lam = self.solve_pgs(&step_cfg, Some(&lam)).impulses;
                } else {
                    break;
                }
            }
        }

        for (i, c) in self.contacts.iter().enumerate() {
            lam[i][0] = lam[i][0].max(0.0);
            let p = project_ellipse_eucl(
                lam[i][1],
                lam[i][2],
                c.mu_1 * lam[i][0],
                c.mu_2 * lam[i][0],
            );
            lam[i][1] = p[0];
            lam[i][2] = p[1];
        }
        (lam, iters, converged, false)
    }

    /// Shen et al. (IJRR 2026, Eq. 33–34) Polyhedral $m$-Facet Frictional LCP formulation:
    /// solves each contact's exact $\gamma$-coupled $k$-gon tangential complementarity problem
    /// `0 <= D^T u_t + gamma * 1 ⊥ beta >= 0`, `0 <= f_n - 1^T beta ⊥ gamma >= 0`
    /// via exact 2D polygon-constrained SPD QP minimization, converging monotonically at
    /// $O(1/k^2)$ to the continuous Signorini-Coulomb cone as `polyhedral_facets` grows!
    fn solve_polyhedral_lcp(
        &self,
        cfg: &LcpSolverConfig,
        warm_start: Option<&[[f64; 3]]>,
    ) -> LcpSolution {
        let m = self.contacts.len();
        let k = cfg.polyhedral_facets.max(4);
        let dim = 3 * m;
        let (w_eff, b) = self.effective_system(cfg);

        // Unit polygon vertices on the unit circle; scaled per contact by (mu_1 * f_n, mu_2 * f_n).
        let unit_verts: Vec<[f64; 2]> = (0..k)
            .map(|f| {
                let th = 2.0 * PI * (f as f64) / (k as f64);
                [th.cos(), th.sin()]
            })
            .collect();

        let mut lam = match warm_start {
            Some(ws) if ws.len() == m => ws.to_vec(),
            _ => vec![[0.0; 3]; m],
        };
        let omega = cfg.omega_sor.clamp(0.2, 1.5);
        let reg = cfg.compliance_reg.max(0.0);

        let mut iters = 0;
        let mut converged = false;

        for iter in 0..cfg.max_iters {
            iters = iter + 1;
            let mut max_delta = 0.0_f64;

            for i in 0..m {
                let c = &self.contacts[i];
                let r0 = (3 * i) * dim;
                let r1 = (3 * i + 1) * dim;
                let r2 = (3 * i + 2) * dim;

                // Off-diagonal free velocity excluding contact i
                let mut rhs0 = b[3 * i];
                let mut rhs1 = b[3 * i + 1];
                let mut rhs2 = b[3 * i + 2];
                for j in 0..m {
                    if j == i {
                        continue;
                    }
                    let lj = lam[j];
                    let col = 3 * j;
                    rhs0 += w_eff[r0 + col] * lj[0]
                        + w_eff[r0 + col + 1] * lj[1]
                        + w_eff[r0 + col + 2] * lj[2];
                    rhs1 += w_eff[r1 + col] * lj[0]
                        + w_eff[r1 + col + 1] * lj[1]
                        + w_eff[r1 + col + 2] * lj[2];
                    rhs2 += w_eff[r2 + col] * lj[0]
                        + w_eff[r2 + col + 1] * lj[1]
                        + w_eff[r2 + col + 2] * lj[2];
                }

                let w00 = w_eff[r0 + 3 * i] + reg;
                let w01 = w_eff[r0 + 3 * i + 1];
                let w02 = w_eff[r0 + 3 * i + 2];
                let w10 = w_eff[r1 + 3 * i];
                let w20 = w_eff[r2 + 3 * i];
                let w11 = w_eff[r1 + 3 * i + 1] + reg;
                let w12 = 0.5 * (w_eff[r1 + 3 * i + 2] + w_eff[r2 + 3 * i + 1]);
                let w22 = w_eff[r2 + 3 * i + 2] + reg;

                let old = lam[i];

                // Normal Signorini update
                let b_n = rhs0 + w01 * old[1] + w02 * old[2];
                let ln_exact = if w00 > 1e-18 { (-b_n / w00).max(0.0) } else { 0.0 };
                let ln_new = ((1.0 - omega) * old[0] + omega * ln_exact).max(0.0);

                // Exact gamma-coupled k-gon tangential LCP solve (Shen et al. Eq. 33-34):
                // min_{y in P_k(mu_1*fn, mu_2*fn)} 0.5 y^T W_tt y + (rhs_t + W_tn*fn)^T y
                let bt1 = rhs1 + w10 * ln_new;
                let bt2 = rhs2 + w20 * ln_new;
                let c1 = c.mu_1 * ln_new;
                let c2 = c.mu_2 * ln_new;
                let lt_new = solve_2d_polygon_qp(w11, w12, w22, bt1, bt2, c1, c2, &unit_verts);

                lam[i] = [ln_new, lt_new[0], lt_new[1]];
                let delta = (ln_new - old[0])
                    .abs()
                    .max((lt_new[0] - old[1]).abs())
                    .max((lt_new[1] - old[2]).abs());
                max_delta = max_delta.max(delta);
            }

            if max_delta < cfg.tol {
                converged = true;
                break;
            }
        }

        let u = Self::mat_vec_with(&w_eff, &lam, &b);
        let (nr, fr, fbr) = self.evaluate_residuals(&lam, &u);
        LcpSolution {
            impulses: lam,
            post_vel: u,
            iterations: iters,
            converged: converged || (nr < cfg.tol * 100.0 && fbr < cfg.tol * 100.0),
            normal_complementarity_res: nr,
            friction_cone_res: fr,
            fischer_burmeister_res: fbr,
        }
    }

    /// Exact 3D Second-Order Friction-Cone SOCP solver backed by `clarabel` interior-point QDLDL
    /// (Goulart & Chen 2024) with De Saxcé / Acary-Cadoux outer shift eliminating Anitescu-Tasora
    /// sliding normal lift so the converged solution satisfies exact Signorini-Coulomb maximum
    /// dissipation (`u_n == 0` on sliding contacts) even on extreme mass/stiffness ratios `kappa >= 1e8`.
    #[cfg(feature = "clarabel-socp")]
    fn solve_clarabel_socp(
        &self,
        cfg: &LcpSolverConfig,
        warm_start: Option<&[[f64; 3]]>,
    ) -> LcpSolution {
        let m = self.contacts.len();
        let dim = 3 * m;
        let (w_eff, b_eff) = self.effective_system(cfg);
        let reg = cfg.compliance_reg.max(0.0);

        // Cone-invariant diagonal variable scaling `y = d * lam` (`d = sqrt(max_diag(W_eff))`):
        // maps `P -> W_eff / d^2` (`||P|| ~ 1`, matching `||A|| ~ 1`), `q -> q / d`, leaves the
        // objective value and all cones (`A y + s' = 0, s' in K`) 100% mathematically invariant,
        // and prevents KKT matrix scaling blow-up when `max(W_ii) >> 1` (e.g. `kappa >= 1e8`).
        let mut max_diag = 1.0_f64;
        for i in 0..dim {
            max_diag = max_diag.max((w_eff[i * dim + i] + reg).abs());
        }
        let scale_d = max_diag.sqrt();
        let inv_d = 1.0 / scale_d;
        let inv_d2 = inv_d * inv_d;

        // Build upper-triangular CSC objective Hessian P_y = (sym(W_eff) + reg * I) / d^2.
        let mut p_colptr = Vec::with_capacity(dim + 1);
        let mut p_rowval = Vec::with_capacity(dim * (dim + 1) / 2);
        let mut p_nzval = Vec::with_capacity(dim * (dim + 1) / 2);
        p_colptr.push(0usize);
        for col in 0..dim {
            for row in 0..=col {
                let mut v = 0.5 * (w_eff[row * dim + col] + w_eff[col * dim + row]);
                if row == col {
                    v += reg;
                }
                v *= inv_d2;
                if row == col || v.abs() > 0.0 {
                    p_rowval.push(row);
                    p_nzval.push(v);
                }
            }
            p_colptr.push(p_rowval.len());
        }
        let p_mat = CscMatrix::new(dim, dim, p_colptr, p_rowval, p_nzval);

        // Build exact 3D elliptic friction-cone constraint A * lam + s = 0, s in K
        // (3 rows per contact, strictly increasing row indices per column).
        let mut a_colptr = Vec::with_capacity(dim + 1);
        let mut a_rowval = Vec::with_capacity(dim * 2);
        let mut a_nzval = Vec::with_capacity(dim * 2);
        let mut cones: Vec<SupportedConeT<f64>> = Vec::with_capacity(m * 2);
        a_colptr.push(0usize);

        for (i, c) in self.contacts.iter().enumerate() {
            let r0 = 3 * i;
            let has1 = c.mu_1 > 1e-15;
            let has2 = c.mu_2 > 1e-15;
            if has1 && has2 {
                // Balanced geometric-mean SOC(3):
                // s0 = sqrt(mu1*mu2)*lam_n >= sqrt((mu2/mu1)*lam_t1^2 + (mu1/mu2)*lam_t2^2)
                let a0 = (c.mu_1 * c.mu_2).sqrt();
                let a1 = (c.mu_2 / c.mu_1).sqrt();
                let a2 = (c.mu_1 / c.mu_2).sqrt();
                // col 3i
                a_rowval.push(r0);
                a_nzval.push(-a0);
                a_colptr.push(a_rowval.len());
                // col 3i+1
                a_rowval.push(r0 + 1);
                a_nzval.push(-a1);
                a_colptr.push(a_rowval.len());
                // col 3i+2
                a_rowval.push(r0 + 2);
                a_nzval.push(-a2);
                a_colptr.push(a_rowval.len());
                cones.push(SupportedConeT::SecondOrderConeT(3));
            } else if !has1 && !has2 {
                // Frictionless: lam_n >= 0, lam_t1 == 0, lam_t2 == 0
                a_rowval.push(r0);
                a_nzval.push(-1.0);
                a_colptr.push(a_rowval.len());
                a_rowval.push(r0 + 1);
                a_nzval.push(1.0);
                a_colptr.push(a_rowval.len());
                a_rowval.push(r0 + 2);
                a_nzval.push(1.0);
                a_colptr.push(a_rowval.len());
                cones.push(SupportedConeT::NonnegativeConeT(1));
                cones.push(SupportedConeT::ZeroConeT(2));
            } else if has1 {
                // 1D friction along t1: mu1*lam_n - lam_t1 >= 0, mu1*lam_n + lam_t1 >= 0, lam_t2 == 0
                a_rowval.push(r0);
                a_nzval.push(-c.mu_1);
                a_rowval.push(r0 + 1);
                a_nzval.push(-c.mu_1);
                a_colptr.push(a_rowval.len());
                a_rowval.push(r0);
                a_nzval.push(1.0);
                a_rowval.push(r0 + 1);
                a_nzval.push(-1.0);
                a_colptr.push(a_rowval.len());
                a_rowval.push(r0 + 2);
                a_nzval.push(1.0);
                a_colptr.push(a_rowval.len());
                cones.push(SupportedConeT::NonnegativeConeT(2));
                cones.push(SupportedConeT::ZeroConeT(1));
            } else {
                // 1D friction along t2: lam_t1 == 0, mu2*lam_n - lam_t2 >= 0, mu2*lam_n + lam_t2 >= 0
                a_rowval.push(r0 + 1);
                a_nzval.push(-c.mu_2);
                a_rowval.push(r0 + 2);
                a_nzval.push(-c.mu_2);
                a_colptr.push(a_rowval.len());
                a_rowval.push(r0);
                a_nzval.push(1.0);
                a_colptr.push(a_rowval.len());
                a_rowval.push(r0 + 1);
                a_nzval.push(1.0);
                a_rowval.push(r0 + 2);
                a_nzval.push(-1.0);
                a_colptr.push(a_rowval.len());
                cones.push(SupportedConeT::ZeroConeT(1));
                cones.push(SupportedConeT::NonnegativeConeT(2));
            }
        }
        let a_mat = CscMatrix::new(dim, dim, a_colptr, a_rowval, a_nzval);
        let b_cone = vec![0.0_f64; dim];

        // Initialize De Saxcé normal-lift shift sigma_i = sqrt((mu1*u_t1)^2 + (mu2*u_t2)^2).
        let mut sigma = vec![0.0_f64; m];
        if let Some(ws) = warm_start {
            if ws.len() == m {
                let u0 = Self::mat_vec_with(&w_eff, ws, &b_eff);
                for (i, c) in self.contacts.iter().enumerate() {
                    sigma[i] = (c.mu_1 * u0[i][1]).hypot(c.mu_2 * u0[i][2]);
                }
            }
        }

        // `p_mat` is already unit-scaled (`max_diag(p_mat) == 1.0`, matching `||a_mat|| ~ 1.0`),
        // so size Clarabel tolerances and QDLDL static regularization in unit-scaled coordinates!
        // Keeping `static_regularization_constant = 1e-12` preserves small eigenvalues (`~1e-8`)
        // on `kappa >= 1e8` problems without perturbing the compliant eigenspace.
        let gap_tol = (cfg.tol * cfg.tol).clamp(1e-15, 1e-12);
        let feas_tol = (cfg.tol * 0.1).clamp(1e-13, 1e-10);
        let settings = DefaultSettings::<f64> {
            verbose: false,
            tol_gap_abs: gap_tol,
            tol_gap_rel: gap_tol,
            tol_feas: feas_tol,
            tol_ktratio: (feas_tol * 10.0).clamp(1e-11, 1e-8),
            reduced_tol_gap_abs: 1e-10,
            reduced_tol_gap_rel: 1e-10,
            reduced_tol_feas: 1e-9,
            equilibrate_enable: true,
            equilibrate_min_scaling: 1e-6,
            equilibrate_max_scaling: 1e6,
            static_regularization_constant: 1e-12,
            iterative_refinement_enable: true,
            iterative_refinement_max_iter: 20,
            iterative_refinement_abstol: 1e-14,
            iterative_refinement_reltol: 1e-14,
            max_iter: 200,
            ..Default::default()
        };

        let max_outer = cfg.max_iters.clamp(16, 64);
        let mut best_lam = vec![[0.0; 3]; m];
        let mut best_res = f64::INFINITY;
        let mut total_ipm_iters = 0usize;
        let mut converged = false;
        let mut relax = 1.0_f64;
        let mut stale_outers = 0usize;
        // Depth-1 Anderson / secant history for rapid De Saxcé fixed-point convergence.
        let mut prev_sigma: Option<Vec<f64>> = None;
        let mut prev_g: Option<Vec<f64>> = None;

        for outer in 0..max_outer {
            let mut q = Vec::with_capacity(dim);
            for i in 0..m {
                q.push((b_eff[3 * i] + sigma[i]) * inv_d);
                q.push(b_eff[3 * i + 1] * inv_d);
                q.push(b_eff[3 * i + 2] * inv_d);
            }

            let Ok(mut solver) =
                DefaultSolver::new(&p_mat, &q, &a_mat, &b_cone, &cones, settings.clone())
            else {
                return self.solve_alart_curnier_newton(cfg, warm_start);
            };
            solver.solve();
            total_ipm_iters += solver.solution.iterations as usize;

            let mut finite_x = solver.solution.x.len() == dim
                && solver.solution.x.iter().all(|v| v.is_finite());
            let mut usable_status = matches!(
                solver.solution.status,
                SolverStatus::Solved
                    | SolverStatus::AlmostSolved
                    | SolverStatus::InsufficientProgress
            ) && finite_x;

            // If QDLDL hit a numerical hiccup, retry once with stronger static regularization.
            if !usable_status {
                let mut safe_settings = settings.clone();
                safe_settings.static_regularization_constant = 1e-8;
                safe_settings.tol_gap_abs = 1e-12;
                safe_settings.tol_gap_rel = 1e-12;
                safe_settings.tol_feas = 1e-11;
                if let Ok(mut retry_solver) =
                    DefaultSolver::new(&p_mat, &q, &a_mat, &b_cone, &cones, safe_settings)
                {
                    retry_solver.solve();
                    total_ipm_iters += retry_solver.solution.iterations as usize;
                    finite_x = retry_solver.solution.x.len() == dim
                        && retry_solver.solution.x.iter().all(|v| v.is_finite());
                    usable_status = matches!(
                        retry_solver.solution.status,
                        SolverStatus::Solved
                            | SolverStatus::AlmostSolved
                            | SolverStatus::InsufficientProgress
                    ) && finite_x;
                    if usable_status {
                        solver = retry_solver;
                    }
                }
            }

            if !usable_status && outer == 0 {
                // Fallback gracefully if Clarabel reports numerical failure on a degenerate input
                return self.solve_alart_curnier_newton(cfg, warm_start);
            }
            if !usable_status {
                break;
            }

            let mut lam_cand = vec![[0.0; 3]; m];
            for (i, c) in self.contacts.iter().enumerate() {
                let ln = (solver.solution.x[3 * i] * inv_d).max(0.0);
                let pt = project_ellipse_eucl(
                    solver.solution.x[3 * i + 1] * inv_d,
                    solver.solution.x[3 * i + 2] * inv_d,
                    c.mu_1 * ln,
                    c.mu_2 * ln,
                );
                lam_cand[i] = [ln, pt[0], pt[1]];
            }

            let u_cand = Self::mat_vec_with(&w_eff, &lam_cand, &b_eff);
            let (nr, fr, _) = self.evaluate_residuals(&lam_cand, &u_cand);
            let res = nr.max(fr);
            let improved = res < best_res;
            if improved {
                if best_res - res > 0.05 * best_res.min(1.0) {
                    stale_outers = 0;
                } else {
                    stale_outers += 1;
                }
                best_res = res;
                best_lam = lam_cand;
                relax = (relax * 1.25).min(1.0);
            } else {
                stale_outers += 1;
                relax = (relax * 0.5).max(0.25);
            }

            let mut g_curr = vec![0.0_f64; m];
            let mut max_dsigma = 0.0_f64;
            for (i, c) in self.contacts.iter().enumerate() {
                let s_target = (c.mu_1 * u_cand[i][1]).hypot(c.mu_2 * u_cand[i][2]);
                g_curr[i] = s_target - sigma[i];
                max_dsigma = max_dsigma.max(g_curr[i].abs());
            }

            let tol_stop = cfg.tol.max(1e-12) * 10.0;
            if res <= tol_stop {
                converged = true;
                break;
            }
            // Exit outer IPM loop once Clarabel reaches its FP64 O(sqrt(s^T z)) SOC boundary floor
            // (fixed-point shift `sigma` converged or plateaued), then run active-set crossover polish.
            if best_res <= 1e-2 && (max_dsigma <= 1e-8 || stale_outers >= 4) {
                break;
            }

            // Safeguarded Anderson(1) acceleration on De Saxcé shift `sigma`:
            // falls back to damped Picard whenever residual fails to decrease.
            let mut next_sigma = vec![0.0_f64; m];
            let used_anderson = if improved {
                if let (Some(ps), Some(pg)) = (&prev_sigma, &prev_g) {
                    let mut dg_dot_g = 0.0_f64;
                    let mut dg_norm2 = 0.0_f64;
                    for i in 0..m {
                        let dg = g_curr[i] - pg[i];
                        dg_dot_g += dg * g_curr[i];
                        dg_norm2 += dg * dg;
                    }
                    if dg_norm2 > 1e-30 {
                        let theta = (dg_dot_g / dg_norm2).clamp(-2.0, 2.0);
                        for i in 0..m {
                            let s_aa = (1.0 - theta) * (sigma[i] + g_curr[i])
                                + theta * (ps[i] + pg[i]);
                            next_sigma[i] = s_aa.max(0.0);
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if !used_anderson {
                for i in 0..m {
                    next_sigma[i] = (sigma[i] + relax * g_curr[i]).max(0.0);
                }
            }
            prev_sigma = Some(sigma);
            prev_g = Some(g_curr);
            sigma = next_sigma;
        }

        // Standard barrier-to-active-set crossover polish: once Clarabel IPM has already solved
        // the global conic problem to `< 1e-1`, 1–8 safeguarded Alart-Curnier Newton steps
        // eliminate the O(sqrt(s^T z)) central-path SOC boundary error down to FP64 precision.
        let tol_stop = cfg.tol.max(1e-12) * 10.0;
        if best_res > tol_stop && best_res < 1e-1 {
            let (polished, polish_iters, _, lu_failed) =
                self.refine_alart_curnier(&w_eff, &b_eff, &best_lam, cfg, 8, false);
            if !lu_failed {
                let u_pol = Self::mat_vec_with(&w_eff, &polished, &b_eff);
                let (nr_p, fr_p, _) = self.evaluate_residuals(&polished, &u_pol);
                if nr_p.max(fr_p) < best_res {
                    best_lam = polished;
                    total_ipm_iters += polish_iters;
                    if nr_p.max(fr_p) <= tol_stop {
                        converged = true;
                    }
                }
            }
        }

        let u = Self::mat_vec_with(&w_eff, &best_lam, &b_eff);
        let (nr, fr, fbr) = self.evaluate_residuals(&best_lam, &u);
        LcpSolution {
            impulses: best_lam,
            post_vel: u,
            iterations: total_ipm_iters,
            converged: converged || nr.max(fr) <= cfg.tol.max(1e-11) * 100.0,
            normal_complementarity_res: nr,
            friction_cone_res: fr,
            fischer_burmeister_res: fbr,
        }
    }
}

/// Exact scalar Fischer-Burmeister NCP function: `phi_FB(a, b) = sqrt(a^2 + b^2) - a - b`.
/// Strictly zero iff `a >= 0, b >= 0, a * b == 0`.
#[inline]
pub fn fischer_burmeister(a: f64, b: f64) -> f64 {
    a.hypot(b) - a - b
}

/// Exact Euclidean orthogonal projection of `(z1, z2)` onto the 2D friction ellipse
/// `(x / a)^2 + (y / b)^2 <= 1` (exact closed-form when `a == b`, 5-step safeguarded Newton
/// on the convex secular equation when `a != b`).
fn project_ellipse_eucl(z1: f64, z2: f64, a: f64, b: f64) -> [f64; 2] {
    if a <= 1e-15 && b <= 1e-15 {
        return [0.0, 0.0];
    }
    if a <= 1e-15 {
        return [0.0, z2.clamp(-b, b)];
    }
    if b <= 1e-15 {
        return [z1.clamp(-a, a), 0.0];
    }
    let u1 = z1 / a;
    let u2 = z2 / b;
    let ell = u1.hypot(u2);
    if ell <= 1.0 {
        return [z1, z2];
    }
    if (a - b).abs() <= 1e-14 * a.max(b) {
        let inv = 1.0 / ell;
        return [z1 * inv, z2 * inv];
    }

    // Solve g(s) = (a * z1 / (a^2 + s))^2 + (b * z2 / (b^2 + s))^2 - 1 = 0 for s > 0.
    let a2 = a * a;
    let b2 = b * b;
    let az1 = a * z1.abs();
    let bz2 = b * z2.abs();
    let mut s = (az1 - a2).max(bz2 - b2).max(0.0);
    for _ in 0..24 {
        let dx = a2 + s;
        let dy = b2 + s;
        let rx = az1 / dx;
        let ry = bz2 / dy;
        let g = rx * rx + ry * ry - 1.0;
        if g.abs() < 1e-15 {
            break;
        }
        let dg = -2.0 * (rx * rx / dx + ry * ry / dy);
        if dg.abs() < 1e-30 {
            break;
        }
        let s_next = (s - g / dg).max(0.0);
        if (s_next - s).abs() <= 1e-15 * s_next.max(1e-30) {
            s = s_next;
            break;
        }
        s = s_next;
    }
    [a2 * z1 / (a2 + s), b2 * z2 / (b2 + s)]
}

/// Exact 3x3 Clarke generalized Jacobian of `(max(0, zn), Pi_Eucl(zt; mu1*pn, mu2*pn))`
/// w.r.t. `(zn, zt1, zt2)`. Uses Euler's homogeneous function theorem for exact `dP/dzn`!
fn ellipse_proj_jacobian(zn: f64, z1: f64, z2: f64, mu1: f64, mu2: f64) -> [[f64; 3]; 3] {
    let mut p_loc = [[0.0; 3]; 3];
    if zn <= 0.0 {
        return p_loc;
    }
    p_loc[0][0] = 1.0;
    let a = mu1 * zn;
    let b = mu2 * zn;
    if a <= 1e-15 && b <= 1e-15 {
        return p_loc;
    }
    if a <= 1e-15 {
        if z2.abs() <= b {
            p_loc[2][2] = 1.0;
        } else {
            p_loc[2][0] = if z2 >= 0.0 { mu2 } else { -mu2 };
        }
        return p_loc;
    }
    if b <= 1e-15 {
        if z1.abs() <= a {
            p_loc[1][1] = 1.0;
        } else {
            p_loc[1][0] = if z1 >= 0.0 { mu1 } else { -mu1 };
        }
        return p_loc;
    }

    let ell = (z1 / a).hypot(z2 / b);
    if ell <= 1.0 {
        // Strictly sticking: Pi_t = z_t
        p_loc[1][1] = 1.0;
        p_loc[2][2] = 1.0;
        return p_loc;
    }

    let a2 = a * a;
    let b2 = b * b;
    let [x, y] = project_ellipse_eucl(z1, z2, a, b);

    // Recover exact Lagrange multiplier s > 0 from x = a^2 z1 / (a^2 + s) or y = b^2 z2 / (b^2 + s)
    let s = if x.abs() > 1e-15 {
        (a2 * (z1 / x - 1.0)).max(0.0)
    } else if y.abs() > 1e-15 {
        (b2 * (z2 / y - 1.0)).max(0.0)
    } else {
        0.0
    };
    let dx = a2 + s;
    let dy = b2 + s;

    // Implicit function theorem on g(s, z1, z2) = (x/a)^2 + (y/b)^2 - 1 = 0:
    // gs = -2 * (x^2 / (a^2 * dx) + y^2 / (b^2 * dy)), gz1 = 2 x / dx, gz2 = 2 y / dy.
    let gs_neg = 2.0 * ((x * x) / (a2 * dx) + (y * y) / (b2 * dy));
    if gs_neg <= 1e-30 {
        return p_loc;
    }
    let ds_dz1 = (2.0 * x / dx) / gs_neg;
    let ds_dz2 = (2.0 * y / dy) / gs_neg;

    let d11 = (a2 - x * ds_dz1) / dx;
    let d12 = (-x * ds_dz2) / dx;
    let d21 = (-y * ds_dz1) / dy;
    let d22 = (b2 - y * ds_dz2) / dy;

    // Exact Euler homogeneity: P(zn, z) is degree-1 homogeneous => dP/dzn = (P - dP/dz * z) / zn
    let inv_zn = 1.0 / zn;
    p_loc[1][0] = (x - (d11 * z1 + d12 * z2)) * inv_zn;
    p_loc[2][0] = (y - (d21 * z1 + d22 * z2)) * inv_zn;
    p_loc[1][1] = d11;
    p_loc[1][2] = d12;
    p_loc[2][1] = d21;
    p_loc[2][2] = d22;
    p_loc
}

/// Solve the exact 2D SPD elliptic-cone QP:
/// `min_{(x/c1)^2 + (y/c2)^2 <= 1} 0.5 [x y] W_tt [x y]^T + [b1 b2] [x y]^T`
/// via exact sticking check + safeguarded Newton on the convex KKT secular equation
/// `(W_tt + s * diag(1/c1^2, 1/c2^2)) y(s) = -b`.
fn solve_2d_ellipse_qp(
    w11: f64,
    w12: f64,
    w22: f64,
    b1: f64,
    b2: f64,
    c1: f64,
    c2: f64,
) -> [f64; 2] {
    if c1 <= 1e-15 && c2 <= 1e-15 {
        return [0.0, 0.0];
    }
    if c1 <= 1e-15 {
        return [0.0, (-b2 / w22.max(1e-15)).clamp(-c2, c2)];
    }
    if c2 <= 1e-15 {
        return [(-b1 / w11.max(1e-15)).clamp(-c1, c1), 0.0];
    }

    // 1. Exact unconstrained sticking minimizer: W_tt y = -b
    let det0 = w11 * w22 - w12 * w12;
    if det0 > 1e-24 {
        let y1 = -(w22 * b1 - w12 * b2) / det0;
        let y2 = -(-w12 * b1 + w11 * b2) / det0;
        if (y1 / c1).hypot(y2 / c2) <= 1.0 {
            return [y1, y2];
        }
    }

    // 2. Sliding boundary: find unique s > 0 where ||y(s)||_E^2 = 1.
    //    Write A = diag(a1, a2) with a1 = 1/c1^2, a2 = 1/c2^2.
    let a1 = 1.0 / (c1 * c1);
    let a2 = 1.0 / (c2 * c2);

    // Bracket [s_lo, s_hi] strictly containing the unique root phi(s) = a1*y1^2 + a2*y2^2 - 1 = 0.
    let eval_s = |s: f64| -> (f64, f64, f64, f64) {
        let m11 = w11 + s * a1;
        let m22 = w22 + s * a2;
        let det = (m11 * m22 - w12 * w12).max(1e-30);
        let inv_det = 1.0 / det;
        let y1 = -(m22 * b1 - w12 * b2) * inv_det;
        let y2 = -(-w12 * b1 + m11 * b2) * inv_det;
        let phi = a1 * y1 * y1 + a2 * y2 * y2 - 1.0;
        // y'(s) = -M^{-1} (A y) => phi'(s) = -2 (A y)^T M^{-1} (A y) < 0
        let ay1 = a1 * y1;
        let ay2 = a2 * y2;
        let dy1 = -(m22 * ay1 - w12 * ay2) * inv_det;
        let dy2 = -(-w12 * ay1 + m11 * ay2) * inv_det;
        let dphi = 2.0 * (ay1 * dy1 + ay2 * dy2);
        (y1, y2, phi, dphi)
    };

    // Exact Cauchy-Schwarz upper bound:
    // y^T W_tt y + s (y^T A y) = -y^T b with y^T A y = 1 => s* <= ||A^{-1/2} b||_2 = hypot(c1*b1, c2*b2).
    let mut s_lo = 0.0;
    let mut s_hi = ((c1 * b1).hypot(c2 * b2) * 1.001).max(1e-30);
    while eval_s(s_hi).2 > 0.0 && s_hi < 1e20 {
        s_lo = s_hi;
        s_hi *= 4.0;
    }

    let mut s = 0.5 * (s_lo + s_hi);
    for _ in 0..50 {
        let (y1, y2, phi, dphi) = eval_s(s);
        if phi.abs() < 1e-14 {
            return [y1, y2];
        }
        if phi > 0.0 {
            s_lo = s;
        } else {
            s_hi = s;
        }
        let s_newton = if dphi.abs() > 1e-30 {
            s - phi / dphi
        } else {
            0.5 * (s_lo + s_hi)
        };
        s = if s_newton > s_lo && s_newton < s_hi {
            s_newton
        } else {
            0.5 * (s_lo + s_hi)
        };
        if (s_hi - s_lo) <= 1e-14 * s_hi.max(1e-30) {
            let (yf1, yf2, _, _) = eval_s(s);
            return [yf1, yf2];
        }
    }
    let (y1, y2, _, _) = eval_s(s);
    [y1, y2]
}

/// Solve the exact 2D SPD polyhedral $k$-gon QP (exact per-contact block solution of
/// Shen et al. IJRR 2026 Eq. 33–34):
/// `min_{y in conv{v_0, ..., v_{k-1}}} 0.5 y^T W_tt y + b^T y`, where `v_f = (c1 cos_f, c2 sin_f)`.
#[allow(clippy::too_many_arguments)]
fn solve_2d_polygon_qp(
    w11: f64,
    w12: f64,
    w22: f64,
    b1: f64,
    b2: f64,
    c1: f64,
    c2: f64,
    unit_verts: &[[f64; 2]],
) -> [f64; 2] {
    if c1 <= 1e-15 && c2 <= 1e-15 {
        return [0.0, 0.0];
    }
    if c1 <= 1e-15 {
        return [0.0, (-b2 / w22.max(1e-15)).clamp(-c2, c2)];
    }
    if c2 <= 1e-15 {
        return [(-b1 / w11.max(1e-15)).clamp(-c1, c1), 0.0];
    }

    let k = unit_verts.len();

    // 1. Check unconstrained sticking minimizer inside the convex k-gon (using normalized unit polygon).
    let det0 = w11 * w22 - w12 * w12;
    if det0 > 1e-24 {
        let y1 = -(w22 * b1 - w12 * b2) / det0;
        let y2 = -(-w12 * b1 + w11 * b2) / det0;
        let u1 = y1 / c1;
        let u2 = y2 / c2;
        // For a regular inscribed k-gon with apothem cos(pi/k), check exact half-planes:
        let apothem = (PI / (k as f64)).cos();
        let mut inside = true;
        for f in 0..k {
            let v0 = unit_verts[f];
            let v1 = unit_verts[(f + 1) % k];
            // Outward unit normal to edge (v0, v1) is (v0 + v1) / ||v0 + v1||
            let mx = v0[0] + v1[0];
            let my = v0[1] + v1[1];
            let mn = mx.hypot(my).max(1e-15);
            if (u1 * mx + u2 * my) / mn > apothem + 1e-15 {
                inside = false;
                break;
            }
        }
        if inside {
            return [y1, y2];
        }
    }

    // 2. Otherwise the unique global minimizer lies on one of the k boundary edges [p0, p1].
    let mut best_q = f64::INFINITY;
    let mut best_y = [0.0, 0.0];
    for f in 0..k {
        let u0 = unit_verts[f];
        let u1 = unit_verts[(f + 1) % k];
        let p0 = [c1 * u0[0], c2 * u0[1]];
        let p1 = [c1 * u1[0], c2 * u1[1]];
        let d0 = p1[0] - p0[0];
        let d1 = p1[1] - p0[1];

        // Q(p0 + alpha * d) = 0.5 * a * alpha^2 + lin * alpha + const
        let wd0 = w11 * d0 + w12 * d1;
        let wd1 = w12 * d0 + w22 * d1;
        let quad = d0 * wd0 + d1 * wd1;
        let wp0 = w11 * p0[0] + w12 * p0[1] + b1;
        let wp1 = w12 * p0[0] + w22 * p0[1] + b2;
        let lin = d0 * wp0 + d1 * wp1;
        let alpha = if quad > 1e-24 {
            (-lin / quad).clamp(0.0, 1.0)
        } else if lin < 0.0 {
            1.0
        } else {
            0.0
        };
        let y0 = p0[0] + alpha * d0;
        let y1 = p0[1] + alpha * d1;
        let q_val = 0.5 * (w11 * y0 * y0 + 2.0 * w12 * y0 * y1 + w22 * y1 * y1)
            + b1 * y0
            + b2 * y1;
        if q_val < best_q {
            best_q = q_val;
            best_y = [y0, y1];
        }
    }
    best_y
}

/// Dense Gaussian elimination with partial pivoting for small `dim x dim` Alart-Curnier Newton steps.
fn solve_dense_lu(a_in: &[f64], b_in: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();
    for k in 0..n {
        let mut piv = k;
        let mut max_v = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_v {
                max_v = v;
                piv = i;
            }
        }
        if max_v < 1e-18 {
            return None;
        }
        if piv != k {
            for j in k..n {
                a.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        let inv_diag = 1.0 / a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k] * inv_diag;
            if f != 0.0 {
                a[i * n + k] = 0.0;
                for j in (k + 1)..n {
                    a[i * n + j] -= f * a[k * n + j];
                }
                b[i] -= f * b[k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    Some(x)
}
