//! Archetype 3: `TendonEndoscope-3D` — fixed-base superelastic Nitinol continuum catheter
//! with 4 multi-waypoint tendons (`D4` radial dihedral symmetry, order 8), optional free
//! cantilever tip (`body_b = None`), and paired 3D curved lumen geometry.

use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, RigidBodySpec,
    RigidFlexGraph, SolverHints, SymmetryElement, SymmetryGroupKind, SymmetryGroupSpec,
    TendonDriveMode, TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};
use fg_sdf::{
    CrossSection, FunnelConfig, RadiusProfile, SplineKind, SplineTubeSdf, TubeWallMode,
};
use serde::{Deserialize, Serialize};

/// Configuration for the fixed-base 3D 4-tendon Nitinol catheter (`SchurRoute::TreeChain`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EndoscopeConfig {
    pub n_sections: usize,
    pub elements_per_section: usize,
    pub section_length_m: f64,
    pub disc_radius_m: f64,
    pub backbone_radius_m: f64,
    pub tendon_offset_m: f64,
    pub base_height_m: f64,
    pub free_cantilever_tip: bool,
    pub formulation: BeamFormulation,
    pub capstan_mu_theta: f64,
    pub preferred_dt: f64,
    pub material_profile_id: usize,
}

impl Default for EndoscopeConfig {
    fn default() -> Self {
        Self {
            n_sections: 4,
            elements_per_section: 6,
            section_length_m: 0.060,
            disc_radius_m: 0.012,
            backbone_radius_m: 0.0014,
            tendon_offset_m: 0.009,
            base_height_m: 0.12,
            free_cantilever_tip: false,
            formulation: BeamFormulation::CorotationalEulerBernoulli,
            capstan_mu_theta: 0.12,
            preferred_dt: 0.001,
            material_profile_id: 1,
        }
    }
}

/// 3D curved lumen specification paired with `TendonEndoscope-3D` for `SplineTubeSdf`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EndoscopeLumenSpec {
    pub control_points: Vec<Vec3>,
    pub radii_m: Vec<f64>,
    pub mat_id: usize,
}

/// Default 3D curved anatomical/acrylic lumen centerline & variable radius profile
/// sized to enclose `TendonEndoscope-3D` along `+X` with a gentle 3D S-bend in `(Y, Z)`.
pub fn default_endoscope_lumen_waypoints(cfg: &EndoscopeConfig) -> EndoscopeLumenSpec {
    let total_len = (cfg.n_sections as f64) * cfg.section_length_m;
    let z0 = cfg.base_height_m;
    let r0 = cfg.disc_radius_m * 2.1;
    EndoscopeLumenSpec {
        control_points: vec![
            Vec3::new(-0.01, 0.0, z0),
            Vec3::new(0.35 * total_len, 0.0, z0),
            Vec3::new(0.72 * total_len, 0.022, z0 + 0.006),
            Vec3::new(1.12 * total_len, 0.036, z0 - 0.005),
        ],
        radii_m: vec![r0, r0 * 0.96, r0 * 0.90, r0 * 0.85],
        mat_id: cfg.material_profile_id,
    }
}

/// Construct a BVH-accelerated 3D curved variable-radius [`SplineTubeSdf`] lumen
/// enclosing `TendonEndoscope-3D` for constrained catheter navigation.
pub fn build_endoscope_spline_tube_sdf(cfg: &EndoscopeConfig) -> SplineTubeSdf {
    let spec = default_endoscope_lumen_waypoints(cfg);
    let r_start = *spec.radii_m.first().unwrap_or(&0.025);
    let r_end = *spec.radii_m.last().unwrap_or(&0.020);
    SplineTubeSdf::new(
        spec.control_points,
        SplineKind::CubicBSpline,
        RadiusProfile::LinearTaper { r_start, r_end },
        CrossSection::Circular,
        TubeWallMode::InteriorCavity,
        FunnelConfig::NONE,
        spec.mat_id,
        64,
    )
}

/// Build the dihedral symmetry group $D_4$ (`|G| = 8`) for 4 radial tendons `[+Y, +Z, -Y, -Z]`.
pub fn make_d4_radial_symmetry() -> SymmetryGroupSpec {
    // 90 deg CCW rotation around +X: 0(+Y) -> 1(+Z) -> 2(-Y) -> 3(-Z) -> 0
    let rot90 = SymmetryElement {
        name: "r90".to_string(),
        obs_perm: vec![1, 2, 3, 0],
        obs_sign: vec![1.0; 4],
        act_perm: vec![1, 2, 3, 0],
        act_sign: vec![1.0; 4],
        est_perm: Vec::new(),
        est_sign: Vec::new(),
    };
    // Reflection across XZ plane (swaps +Y <-> -Y)
    let refl_y = SymmetryElement {
        name: "sy".to_string(),
        obs_perm: vec![2, 1, 0, 3],
        obs_sign: vec![1.0; 4],
        act_perm: vec![2, 1, 0, 3],
        act_sign: vec![1.0; 4],
        est_perm: Vec::new(),
        est_sign: Vec::new(),
    };
    SymmetryGroupSpec::from_generators(
        SymmetryGroupKind::D4Dihedral,
        4,
        4,
        0,
        &[rot90, refl_y],
        8,
    )
    .expect("D4 symmetry closure must succeed")
}

/// Build the fixed-base 3D 4-tendon Nitinol continuum endoscope (`SchurRoute::TreeChain`).
pub fn build_tendon_endoscope(cfg: &EndoscopeConfig) -> RigidFlexGraph {
    assert!(cfg.n_sections >= 1, "n_sections must be >= 1");
    assert!(cfg.elements_per_section >= 2, "elements_per_section must be >= 2");

    let n_discs = if cfg.free_cantilever_tip {
        cfg.n_sections
    } else {
        cfg.n_sections + 1
    };

    // 1. Fixed base + spacer discs along +X
    let mut bodies = Vec::with_capacity(n_discs);
    for i in 0..n_discs {
        let x_pos = (i as f64) * cfg.section_length_m;
        let mass = if i == 0 { 0.100 } else { 0.012 };
        let i_axial = 0.5 * mass * cfg.disc_radius_m * cfg.disc_radius_m;
        bodies.push(RigidBodySpec {
            id: i,
            name: if i == 0 {
                "base_fixed".to_string()
            } else {
                format!("spacer_disc_{i}")
            },
            is_fixed: i == 0,
            mass,
            inertia: Mat3::diag([i_axial, 0.5 * i_axial, 0.5 * i_axial]),
            com_local: Vec3::ZERO,
            rest_pose: Se3::new(Vec3::new(x_pos, 0.0, cfg.base_height_m), Quat::IDENTITY),
            collision_radius: cfg.disc_radius_m,
            rayleigh_alpha: 0.10,
            tags: vec![if i == 0 { "fixed_base" } else { "spacer_disc" }.to_string()],
        });
    }

    // 2. Superelastic Nitinol backbone sections (optional free cantilever tip on final section)
    let nitinol = BeamMaterial {
        youngs_modulus: 65e9,
        poisson_ratio: 0.33,
        density: 6450.0,
        rayleigh_alpha: 0.10,
        rayleigh_beta: 0.001,
    };
    let section = BeamSection::Circular {
        radius: cfg.backbone_radius_m,
    };
    let elem_len = cfg.section_length_m / (cfg.elements_per_section as f64);

    let mut beams = Vec::with_capacity(cfg.n_sections);
    for s in 0..cfg.n_sections {
        let x0 = (s as f64) * cfg.section_length_m;
        let is_cant = cfg.free_cantilever_tip && (s + 1 == cfg.n_sections);
        let mut poses = Vec::with_capacity(cfg.elements_per_section + 1);
        for j in 0..=cfg.elements_per_section {
            let x = x0 + (j as f64) * elem_len;
            poses.push(Se3::new(
                Vec3::new(x, 0.0, cfg.base_height_m),
                Quat::IDENTITY,
            ));
        }
        beams.push(BeamEdgeSpec {
            id: s,
            name: format!("nitinol_backbone_{s}"),
            body_a: s,
            clamp_a: Se3::IDENTITY,
            body_b: if is_cant { None } else { Some(s + 1) },
            clamp_b: if is_cant { None } else { Some(Se3::IDENTITY) },
            n_elements: cfg.elements_per_section,
            formulation: cfg.formulation,
            section,
            material: nitinol,
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: vec![elem_len; cfg.elements_per_section],
            element_transforms: None,
            interior_node_contact_radius: Some(0.5 * cfg.disc_radius_m),
            enable_self_contact: true,
        });
    }

    // 3. Four full-length multi-waypoint routing tendons (+Y, +Z, -Y, -Z) with D4 symmetry
    let r_t = cfg.tendon_offset_m;
    let tendon_specs = [
        ("tendon_pos_y", Vec3::new(0.0, r_t, 0.0)),
        ("tendon_pos_z", Vec3::new(0.0, 0.0, r_t)),
        ("tendon_neg_y", Vec3::new(0.0, -r_t, 0.0)),
        ("tendon_neg_z", Vec3::new(0.0, 0.0, -r_t)),
    ];
    let mut tendons = Vec::new();
    if n_discs >= 2 {
        let total_len = ((n_discs - 1) as f64) * cfg.section_length_m;
        for (idx, &(tname, off)) in tendon_specs.iter().enumerate() {
            let waypoints: Vec<TendonWaypoint> = (0..n_discs)
                .map(|b_i| TendonWaypoint {
                    body_id: b_i,
                    local_pos: off,
                    wrap_angle_rad: if b_i > 0 && b_i + 1 < n_discs {
                        0.08
                    } else {
                        0.0
                    },
                })
                .collect();
            tendons.push(TendonEdgeSpec {
                id: idx,
                name: tname.to_string(),
                actuator_id: idx,
                waypoints,
                drive_mode: TendonDriveMode::Linear,
                rest_length: total_len,
                slack: 0.0,
                stiffness: 3000.0,
                damping: 1.2,
                capstan_mu_theta: cfg.capstan_mu_theta,
                stick_slip_enabled: cfg.capstan_mu_theta > 0.0,
                servo_torque_limit: 2.0,
                servo_speed_limit: 0.15,
                servo_deadband: 0.0,
                angle_limits: [0.0, 0.025],
                solref_imp: None,
            });
        }
    }

    // 4. Spacer-disc rim isotropic contact pads for 3D curved lumen wall interaction
    let mut contact_sites = Vec::with_capacity(n_discs.saturating_sub(1));
    for i in 1..n_discs {
        contact_sites.push(ContactSiteSpec {
            id: i - 1,
            name: format!("disc_{i}_rim"),
            attachment: ContactAttachment::RigidBody { body_id: i },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: cfg.disc_radius_m,
            patch_radius: 0.004,
            mode: ContactMode::IsotropicPad {
                mu: 0.22,
                pivot_mu: 0.25,
                static_mu_ratio: 1.05,
                sdf_only: true,
            },
            material_profile_id: cfg.material_profile_id,
            solref_imp: None,
        });
    }

    RigidFlexGraph {
        name: if cfg.free_cantilever_tip {
            format!("tendon_endoscope_3d_cantilever_S{}", cfg.n_sections)
        } else {
            format!("tendon_endoscope_3d_S{}", cfg.n_sections)
        },
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams,
        tendons,
        joints: Vec::new(),
        contact_sites,
        symmetry: make_d4_radial_symmetry(),
        solver_hints: SolverHints {
            preferred_dt: cfg.preferred_dt,
            rayleigh_alpha: 0.10,
            rayleigh_beta: 0.001,
            force_route: None,
        },
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    }
}
