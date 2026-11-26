//! MuJoCo-compatible soft-constraint impedance & reference acceleration parameters
//! (`solref` / `solimp`, Todorov et al., IROS 2012; MuJoCo `engine_core_constraint.c`).
//!
//! Supports both MuJoCo `solref` conventions:
//! - **Positive `(timeconst, dampratio)`**: second-order harmonic oscillator with decay time
//!   constant $\tau > 0$ and damping ratio $\zeta > 0$.
//! - **Direct non-positive `(-stiffness, -damping)`**: explicit stiffness $k = -\text{solref}[0]$
//!   and damping $b = -\text{solref}[1]$.
//!
//! Impedance $d(r) \in [10^{-4}, 0.9999]$ follows MuJoCo's exact $C^1$-smooth piecewise power
//! polynomial parameterized by `(dmin, dmax, width, midpoint, power)`.

use serde::{Deserialize, Serialize};

/// Safe lower/upper impedance bounds matching MuJoCo `mjMINIMP` / `mjMAXIMP`.
pub const SOLIMP_MIN: f64 = 1e-4;
pub const SOLIMP_MAX: f64 = 0.9999;

fn default_timeconst() -> f64 {
    0.02
}
fn default_dampratio() -> f64 {
    1.0
}
fn default_dmin() -> f64 {
    0.9
}
fn default_dmax() -> f64 {
    0.95
}
fn default_width() -> f64 {
    0.001
}
fn default_midpoint() -> f64 {
    0.5
}
fn default_power() -> f64 {
    2.0
}

/// MuJoCo-compatible constraint reference & piecewise-polynomial impedance specification.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolRefImp {
    /// `solref[0]`: positive time constant $\tau$ (s), or negative `-stiffness` in direct mode.
    #[serde(default = "default_timeconst")]
    pub timeconst: f64,
    /// `solref[1]`: positive damping ratio $\zeta$, or negative `-damping` in direct mode.
    #[serde(default = "default_dampratio")]
    pub dampratio: f64,
    /// `solimp[0]`: minimum impedance at zero violation (`[1e-4, 0.9999]`).
    #[serde(default = "default_dmin")]
    pub dmin: f64,
    /// `solimp[1]`: maximum impedance at `|r| >= width` (`[1e-4, 0.9999]`).
    #[serde(default = "default_dmax")]
    pub dmax: f64,
    /// `solimp[2]`: transition band width over constraint violation `|r|` (`>= 0`).
    #[serde(default = "default_width")]
    pub width: f64,
    /// `solimp[3]`: normalized inflection point in `(0, 1)`.
    #[serde(default = "default_midpoint")]
    pub midpoint: f64,
    /// `solimp[4]`: polynomial power `>= 1.0` (`1.0` = linear, `2.0` = quadratic spline).
    #[serde(default = "default_power")]
    pub power: f64,
}

impl Default for SolRefImp {
    fn default() -> Self {
        Self {
            timeconst: default_timeconst(),
            dampratio: default_dampratio(),
            dmin: default_dmin(),
            dmax: default_dmax(),
            width: default_width(),
            midpoint: default_midpoint(),
            power: default_power(),
        }
    }
}

impl SolRefImp {
    /// Construct from MuJoCo XML arrays `solref = [timeconst, dampratio]` and
    /// `solimp = [dmin, dmax, width, midpoint, power]`.
    #[inline]
    pub const fn from_mujoco(solref: [f64; 2], solimp: [f64; 5]) -> Self {
        Self {
            timeconst: solref[0],
            dampratio: solref[1],
            dmin: solimp[0],
            dmax: solimp[1],
            width: solimp[2],
            midpoint: solimp[3],
            power: solimp[4],
        }
    }

    /// Construct in MuJoCo direct (negative) convention `solref = [-stiffness, -damping]`.
    #[inline]
    pub fn direct(stiffness: f64, damping: f64) -> Self {
        Self {
            timeconst: -stiffness.abs(),
            dampratio: -damping.abs(),
            ..Self::default()
        }
    }

    /// Returns `true` when using MuJoCo direct `(-stiffness, -damping)` mode
    /// (both parameters non-positive).
    #[inline]
    pub fn is_direct_mode(&self) -> bool {
        self.timeconst <= 0.0 && self.dampratio <= 0.0
    }

    /// Evaluate MuJoCo's exact $C^1$-smooth piecewise-polynomial impedance $d(r)$
    /// and its derivative $\frac{dd}{d|r|}$ for constraint violation magnitude `r`.
    ///
    /// For normalized depth $x = |r| / \mathrm{width} \in [0, 1]$, midpoint $m \in (0, 1)$,
    /// and power $p \ge 1$:
    /// - $x \le m$: $y(x) = x^p / m^{p-1}$
    /// - $x > m$: $y(x) = 1 - (1 - x)^p / (1 - m)^{p-1}$
    /// - $d(r) = d_{\min} + y(x)\,(d_{\max} - d_{\min})$, clamped to `[1e-4, 0.9999]`.
    pub fn impedance_and_deriv(&self, r: f64) -> (f64, f64) {
        let d0 = self.dmin.clamp(SOLIMP_MIN, SOLIMP_MAX);
        let dw = self.dmax.clamp(SOLIMP_MIN, SOLIMP_MAX);
        let w = self.width;
        if (dw - d0).abs() < 1e-15 || w <= 1e-15 {
            return (0.5 * (d0 + dw), 0.0);
        }

        let r_abs = r.abs();
        let x = r_abs / w;
        if x <= 0.0 {
            return (d0, 0.0);
        }
        if x >= 1.0 {
            return (dw, 0.0);
        }

        let mid = self.midpoint.clamp(SOLIMP_MIN, SOLIMP_MAX);
        let p = self.power.max(1.0);
        let (y, dy_dx) = if (p - 1.0).abs() < 1e-12 {
            (x, 1.0)
        } else if x <= mid {
            let a = mid.powf(1.0 - p);
            (a * x.powf(p), p * a * x.powf(p - 1.0))
        } else {
            let b = (1.0 - mid).powf(1.0 - p);
            let omx = 1.0 - x;
            (1.0 - b * omx.powf(p), p * b * omx.powf(p - 1.0))
        };

        let raw = d0 + y * (dw - d0);
        let imp = raw.clamp(SOLIMP_MIN, SOLIMP_MAX);
        let deriv = if (raw - imp).abs() > 0.0 {
            0.0
        } else {
            (dw - d0) * dy_dx / w
        };
        (imp, deriv)
    }

    /// Evaluate MuJoCo piecewise-polynomial impedance $d(r) \in [10^{-4}, 0.9999]$.
    #[inline]
    pub fn impedance(&self, r: f64) -> f64 {
        self.impedance_and_deriv(r).0
    }

    /// Compute MuJoCo `efc_KBIP = [K, B, d(r), dd/d|r|]` parameters used in
    /// $a_{\mathrm{ref}} = -B v - K d(r) r$, clamping positive `timeconst >= min_timeconst`
    /// (typically `2.0 * dt` per MuJoCo stability rule).
    ///
    /// Closed-loop constraint dynamics satisfy:
    /// $a_1 + (d B) v + (d^2 K) r = (1 - d) a_0$.
    pub fn kbip(&self, r: f64, min_timeconst: f64) -> [f64; 4] {
        let (imp, imp_p) = self.impedance_and_deriv(r);
        let dmax = self.dmax.clamp(SOLIMP_MIN, SOLIMP_MAX);

        let (k, b) = if self.timeconst > 0.0 && self.dampratio > 0.0 {
            let tc = self.timeconst.max(min_timeconst.max(1e-6));
            let dr = self.dampratio.max(1e-6);
            let b_val = 2.0 / (dmax * tc);
            let k_val = 1.0 / (dmax * dmax * tc * tc * dr * dr);
            (k_val, b_val)
        } else {
            // Direct negative convention (element-wise fallback matches MuJoCo mj_makeImpedance)
            let k_in = if self.timeconst <= 0.0 {
                -self.timeconst
            } else {
                let tc = self.timeconst.max(min_timeconst.max(1e-6));
                1.0 / (tc * tc)
            };
            let b_in = if self.dampratio <= 0.0 {
                -self.dampratio
            } else {
                let tc = self.timeconst.abs().max(min_timeconst.max(1e-6));
                2.0 * self.dampratio / tc
            };
            (k_in / (dmax * dmax), b_in / dmax)
        };

        [k, b, imp, imp_p]
    }

    /// Convert to effective acceleration-space `(stiffness, damping) = (d^2 * K, d * B)`
    /// without timestep clamping.
    ///
    /// At full penetration (`d(r) == dmax`):
    /// - Positive `(tau, zeta)` returns `(1 / (tau * zeta)^2, 2 / tau)`.
    /// - Direct negative `(-k, -b)` returns `(k, b)`.
    #[inline]
    pub fn to_stiffness_damping(&self, r: f64) -> (f64, f64) {
        self.to_stiffness_damping_dt(r, 0.0)
    }

    /// Convert to effective acceleration-space `(stiffness, damping) = (d^2 * K, d * B)`
    /// with MuJoCo's `timeconst >= 2 * dt` stability clamp.
    #[inline]
    pub fn to_stiffness_damping_dt(&self, r: f64, dt: f64) -> (f64, f64) {
        let [k, b, imp, _] = self.kbip(r, 2.0 * dt.max(0.0));
        (imp * imp * k, imp * b)
    }

    /// Convert to physical force-space `(k_phys, c_phys)` for penalty or implicit Contact-Schur
    /// assembly given effective constraint inertia/mass `eff_mass > 0`:
    /// - In positive `(timeconst, dampratio)` mode, scales acceleration-space `(k_acc, b_acc)` by
    ///   `eff_mass` so the coupled oscillator has exact time constant $\tau$ and damping ratio $\zeta$.
    /// - In direct negative `(-stiffness, -damping)` mode, returns `((d/dmax)^2 * stiffness, (d/dmax) * damping)`
    ///   treating the negative inputs directly as physical stiffness/damping.
    #[inline]
    pub fn to_force_stiffness_damping(&self, r: f64, eff_mass: f64, dt: f64) -> (f64, f64) {
        let (k_eff, b_eff) = self.to_stiffness_damping_dt(r, dt);
        if self.is_direct_mode() {
            (k_eff, b_eff)
        } else {
            let m = eff_mass.max(0.0);
            (m * k_eff, m * b_eff)
        }
    }

    /// MuJoCo constraint regularization `R = ((1 - d(r)) / d(r)) * diag_approx`
    /// for regularized QP/SOCP Delassus formulations (`A_reg = J H^{-1} J^T + diag(R)`).
    #[inline]
    pub fn constraint_regularizer(&self, r: f64, diag_approx: f64) -> f64 {
        let imp = self.impedance(r);
        ((1.0 - imp) / imp) * diag_approx.max(0.0)
    }

    /// MuJoCo reference acceleration `aref = -B * v - K * d(r) * r`.
    #[inline]
    pub fn reference_accel(&self, r: f64, v: f64, dt: f64) -> f64 {
        let [k, b, imp, _] = self.kbip(r, 2.0 * dt.max(0.0));
        -b * v - k * imp * r
    }
}
