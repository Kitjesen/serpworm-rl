//! Differentiable parameter specifications for Dojo / DiffPD-style implicit function
//! theorem (IFT) reverse-mode adjoints and gradient-based system identification.

use serde::{Deserialize, Serialize};

fn default_one() -> f64 {
    1.0
}

/// Physical parameter families exposed to reverse-mode IFT adjoint differentiation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffParamKind {
    /// Beam Young's modulus $E$ (Pa).
    BeamYoungsModulus,
    /// Beam Rayleigh stiffness-proportional damping $\beta_R$ (s).
    BeamDampingBeta,
    /// Contact forward Coulomb friction coefficient $\mu_{\mathrm{fwd}}$.
    ContactMuFwd,
    /// Contact reverse Coulomb friction coefficient $\mu_{\mathrm{rev}}$ (anisotropic scale/bristle).
    ContactMuRev,
    /// Contact normal stiffness (Kelvin $k_n$, Hunt-Crossley $k_{\mathrm{hc}}$, or IPC $\kappa$).
    ContactStiffness,
    /// Tendon axial spring stiffness $k_{\mathrm{ten}}$ (N/m).
    TendonStiffness,
    /// Rigid-body mass $m$ (kg).
    BodyMass,
}

/// Declarative differentiable parameter selection on a `RigidFlexGraph`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiffParamSpec {
    /// Human-readable parameter identifier (e.g. `"strip_youngs_modulus"`).
    pub name: String,
    pub kind: DiffParamKind,
    /// Target entity indices (beam IDs, contact site IDs, tendon IDs, or body IDs).
    /// An empty vector broadcasts one shared scalar parameter across all entities of `kind`.
    #[serde(default)]
    pub target_ids: Vec<usize>,
    /// Optional physical box constraints `[min, max]` for projected gradient / SysID.
    #[serde(default)]
    pub bounds: Option<[f64; 2]>,
    /// Positive nondimensionalization / preconditioning scale factor (default `1.0`).
    #[serde(default = "default_one")]
    pub scale: f64,
}

impl DiffParamSpec {
    /// Construct a shared scalar differentiable parameter over all entities of `kind`.
    pub fn shared(name: impl Into<String>, kind: DiffParamKind) -> Self {
        Self {
            name: name.into(),
            kind,
            target_ids: Vec::new(),
            bounds: None,
            scale: 1.0,
        }
    }

    /// Construct a targeted differentiable parameter over specific entity IDs.
    pub fn targeted(
        name: impl Into<String>,
        kind: DiffParamKind,
        target_ids: Vec<usize>,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            target_ids,
            bounds: None,
            scale: 1.0,
        }
    }
}
