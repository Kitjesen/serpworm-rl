//! Calibrated surface material library (hard floor, acrylic wall, Bekker soft soil,
//! carpet, gravel) matching `serpworm.config.HardwareConfig` terrain presets.

use serde::{Deserialize, Serialize};

pub const MAT_HARD_FLOOR: usize = 0;
pub const MAT_ACRYLIC_WALL: usize = 1;
pub const MAT_BEKKER_SOFT_SOIL: usize = 2;
pub const MAT_CARPET: usize = 3;
pub const MAT_GRAVEL: usize = 4;

/// Normal contact constitutive model associated with a terrain or obstacle material.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormalContactModel {
    /// Linear Kelvin-Voigt spring-damper `F_n = max(0, kn * pen - cn * vn)`.
    KelvinVoigt { kn: f64, cn: f64 },
    /// Smooth zero-impulse Hunt-Crossley normal force with Bekker elasto-plastic soil rutting:
    /// `F_el = hc_k * pen^p` (loading), `unload_ratio * hc_k * (pen - rut) * pen^(p-1)` (unloading),
    /// `F_n = F_el * max(0, 1 - hc_alpha * vn)`.
    HuntCrossleyBekker {
        hc_k: f64,
        hc_alpha: f64,
        soil_exp: f64,
        unload_ratio: f64,
        rut_relax_m: f64,
    },
}

/// Complete physical surface tribology & terramechanics parameters for an SDF material ID.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMaterial {
    pub id: usize,
    pub name: String,
    pub normal_model: NormalContactModel,
    /// Nominal kinetic Coulomb friction coefficient.
    pub mu: f64,
    /// Stribeck static-to-kinetic friction boost `mu_s / mu_k >= 1.0`.
    pub static_mu_ratio: f64,
    /// Lateral-to-forward friction ratio `mu_lat / mu_fwd`.
    pub mu_lat_ratio: f64,
    /// Rolling resistance coefficient for wheeled contact.
    pub rolling_resistance: f64,
    /// Lateral rut-wall bulldozing stiffness on soft granular soil (N/m).
    pub rut_lat_k: f64,
    /// Effective contact-patch radius for vertical pivoting torque (m).
    pub patch_radius: f64,
    /// Vertical pivoting friction coefficient on circular patch.
    pub pivot_mu: f64,
    /// Surface wall normal stiffness for pipe/obstacle contacts (N/m).
    pub wall_stiffness: f64,
    /// Surface wall normal damping (N*s/m).
    pub wall_damping: f64,
    /// Regularization velocity for smooth tangential wall friction (m/s).
    pub wall_v_reg: f64,
}

impl SurfaceMaterial {
    /// Hard laboratory floor (exact match to `HardwareConfig::with_terrain("hard")`).
    pub fn hard_floor() -> Self {
        Self {
            id: MAT_HARD_FLOOR,
            name: "hard_floor".to_string(),
            normal_model: NormalContactModel::HuntCrossleyBekker {
                hc_k: 1.6e5,
                hc_alpha: 3.5,
                soil_exp: 1.5,
                unload_ratio: 1.0,
                rut_relax_m: 0.025,
            },
            mu: 0.80,
            static_mu_ratio: 1.18,
            mu_lat_ratio: 1.05,
            rolling_resistance: 0.06,
            rut_lat_k: 0.0,
            patch_radius: 0.008,
            pivot_mu: 0.65,
            wall_stiffness: 5000.0,
            wall_damping: 8.0,
            wall_v_reg: 0.015,
        }
    }

    /// Transparent acrylic pipe / duct wall (exact match to `pipe_geometry.py`).
    pub fn acrylic_wall() -> Self {
        Self {
            id: MAT_ACRYLIC_WALL,
            name: "acrylic_wall".to_string(),
            normal_model: NormalContactModel::KelvinVoigt {
                kn: 3200.0,
                cn: 18.0,
            },
            mu: 0.18,
            static_mu_ratio: 1.10,
            mu_lat_ratio: 1.0,
            rolling_resistance: 0.03,
            rut_lat_k: 0.0,
            patch_radius: 0.005,
            pivot_mu: 0.25,
            wall_stiffness: 3200.0,
            wall_damping: 18.0,
            wall_v_reg: 0.015,
        }
    }

    /// Bekker elasto-plastic soft soil / loose sand (exact match to `with_terrain("soft_soil")`).
    pub fn bekker_soft_soil() -> Self {
        Self {
            id: MAT_BEKKER_SOFT_SOIL,
            name: "bekker_soft_soil".to_string(),
            normal_model: NormalContactModel::HuntCrossleyBekker {
                hc_k: 1.2e4,
                hc_alpha: 6.0,
                soil_exp: 1.35,
                unload_ratio: 3.5,
                rut_relax_m: 0.030,
            },
            mu: 0.70,
            static_mu_ratio: 1.15,
            mu_lat_ratio: 1.35,
            rolling_resistance: 0.22,
            rut_lat_k: 650.0,
            patch_radius: 0.013,
            pivot_mu: 0.80,
            wall_stiffness: 1500.0,
            wall_damping: 14.0,
            wall_v_reg: 0.020,
        }
    }

    /// Compliant high-friction carpet (exact match to `with_terrain("carpet")`).
    pub fn carpet() -> Self {
        Self {
            id: MAT_CARPET,
            name: "carpet".to_string(),
            normal_model: NormalContactModel::HuntCrossleyBekker {
                hc_k: 1.8e4,
                hc_alpha: 5.0,
                soil_exp: 1.4,
                unload_ratio: 1.0,
                rut_relax_m: 0.025,
            },
            mu: 0.90,
            static_mu_ratio: 1.22,
            mu_lat_ratio: 1.20,
            rolling_resistance: 0.14,
            rut_lat_k: 0.0,
            patch_radius: 0.011,
            pivot_mu: 0.75,
            wall_stiffness: 2200.0,
            wall_damping: 12.0,
            wall_v_reg: 0.015,
        }
    }

    /// Loose granular gravel with moderate plastic sinkage & high lateral bulldozing.
    pub fn gravel() -> Self {
        Self {
            id: MAT_GRAVEL,
            name: "gravel".to_string(),
            normal_model: NormalContactModel::HuntCrossleyBekker {
                hc_k: 2.8e4,
                hc_alpha: 5.5,
                soil_exp: 1.45,
                unload_ratio: 2.4,
                rut_relax_m: 0.028,
            },
            mu: 0.76,
            static_mu_ratio: 1.25,
            mu_lat_ratio: 1.30,
            rolling_resistance: 0.18,
            rut_lat_k: 480.0,
            patch_radius: 0.012,
            pivot_mu: 0.78,
            wall_stiffness: 2600.0,
            wall_damping: 15.0,
            wall_v_reg: 0.018,
        }
    }

    /// Legacy Kelvin-Voigt floor matching `HardwareConfig::with_terrain("kelvin")`.
    pub fn kelvin_floor() -> Self {
        Self {
            id: MAT_HARD_FLOOR,
            name: "kelvin_floor".to_string(),
            normal_model: NormalContactModel::KelvinVoigt {
                kn: 5000.0,
                cn: 8.0,
            },
            mu: 0.80,
            static_mu_ratio: 1.0,
            mu_lat_ratio: 1.0,
            rolling_resistance: 0.06,
            rut_lat_k: 0.0,
            patch_radius: 0.0,
            pivot_mu: 0.65,
            wall_stiffness: 5000.0,
            wall_damping: 8.0,
            wall_v_reg: 0.015,
        }
    }
}

/// Indexed collection of surface materials.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialLibrary {
    materials: Vec<SurfaceMaterial>,
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        Self::standard()
    }
}

impl MaterialLibrary {
    /// Standard 5-material library:
    /// `[0: hard_floor, 1: acrylic_wall, 2: bekker_soft_soil, 3: carpet, 4: gravel]`.
    pub fn standard() -> Self {
        Self {
            materials: vec![
                SurfaceMaterial::hard_floor(),
                SurfaceMaterial::acrylic_wall(),
                SurfaceMaterial::bekker_soft_soil(),
                SurfaceMaterial::carpet(),
                SurfaceMaterial::gravel(),
            ],
        }
    }

    /// Lookup material by ID, clamping safely to index 0 if out of range.
    #[inline]
    pub fn get(&self, mat_id: usize) -> &SurfaceMaterial {
        self.materials
            .get(mat_id)
            .unwrap_or_else(|| &self.materials[0])
    }

    /// Mutable lookup by ID.
    #[inline]
    pub fn get_mut(&mut self, mat_id: usize) -> Option<&mut SurfaceMaterial> {
        self.materials.get_mut(mat_id)
    }

    /// Register a new custom material and return its assigned material ID.
    pub fn register(&mut self, mut mat: SurfaceMaterial) -> usize {
        let id = self.materials.len();
        mat.id = id;
        self.materials.push(mat);
        id
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }
}
