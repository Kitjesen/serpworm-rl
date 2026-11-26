//! Archetype 6: `SoftOctopusArm-DER` — tapered 3D Cosserat / Discrete Elastic Rod soft
//! manipulator with 4 tapered multi-waypoint tendons (`D4` radial symmetry, order 8),
//! optional free cantilever tip, and analytical Euler buckling / nonlinear Elastica
//! validation builders.

use crate::endoscope::make_d4_radial_symmetry;
use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, DiffParamKind,
    DiffParamSpec, RigidBodySpec, RigidFlexGraph, SolverHints, SymmetryGroupSpec, TendonDriveMode,
    TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for the tapered 3D `SoftOctopusArm-DER` continuum manipulator (`TreeChain`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OctopusArmConfig {
    pub n_sections: usize,
    pub elements_per_section: usize,
    pub total_length_m: f64,
    /// Proximal (base) cross-section radius (m).
    pub base_radius_m: f64,
    /// Distal (tip) cross-section radius (m).
    pub tip_radius_m: f64,
    /// Silicone / elastomer Young's modulus (Pa; ~4 MPa for reinforced silicone / Ecoflex-ShoreA).
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub density_kg_m3: f64,
    pub base_height_m: f64,
    pub free_cantilever_tip: bool,
    pub capstan_mu_theta: f64,
    pub preferred_dt: f64,
    pub material_profile_id: usize,
}

impl Default for OctopusArmConfig {
    fn default() -> Self {
        Self {
            n_sections: 4,
            elements_per_section: 6,
            total_length_m: 0.24,
            base_radius_m: 0.012,
            tip_radius_m: 0.004,
            youngs_modulus_pa: 6.0e6,
            poisson_ratio: 0.45,
            density_kg_m3: 1120.0,
            base_height_m: 0.20,
            free_cantilever_tip: false,
            capstan_mu_theta: 0.10,
            preferred_dt: 0.001,
            material_profile_id: 0,
        }
    }
}

/// Build the tapered 3D Cosserat/DER `SoftOctopusArm-DER` (`SchurRoute::TreeChain`, `D4`).
pub fn build_soft_octopus_arm(cfg: &OctopusArmConfig) -> RigidFlexGraph {
    assert!(cfg.n_sections >= 1, "n_sections must be >= 1");
    assert!(cfg.elements_per_section >= 2, "elements_per_section must be >= 2");
    assert!(cfg.total_length_m > 0.0, "total_length_m must be > 0");
    assert!(
        cfg.base_radius_m > 0.0 && cfg.tip_radius_m > 0.0,
        "radii must be > 0"
    );

    let sec_len = cfg.total_length_m / (cfg.n_sections as f64);
    let n_discs = if cfg.free_cantilever_tip {
        cfg.n_sections
    } else {
        cfg.n_sections + 1
    };

    let radius_at = |u: f64| -> f64 {
        let uc = u.clamp(0.0, 1.0);
        cfg.base_radius_m + (cfg.tip_radius_m - cfg.base_radius_m) * uc
    };

    // 1. Fixed base + tapered muscular spacer rings along +X
    let mut bodies = Vec::with_capacity(n_discs);
    for i in 0..n_discs {
        let u = (i as f64) / (cfg.n_sections as f64);
        let x_pos = (i as f64) * sec_len;
        let r_ring = 1.25 * radius_at(u);
        let mass = if i == 0 {
            0.12
        } else {
            let r_ratio = r_ring / (1.25 * cfg.base_radius_m);
            (0.014 * r_ratio * r_ratio).max(0.002)
        };
        let i_ax = 0.5 * mass * r_ring * r_ring;
        bodies.push(RigidBodySpec {
            id: i,
            name: if i == 0 {
                "octopus_base".to_string()
            } else {
                format!("muscle_ring_{i}")
            },
            is_fixed: i == 0,
            mass,
            inertia: Mat3::diag([i_ax, 0.5 * i_ax, 0.5 * i_ax]),
            com_local: Vec3::ZERO,
            rest_pose: Se3::new(Vec3::new(x_pos, 0.0, cfg.base_height_m), Quat::IDENTITY),
            collision_radius: r_ring,
            rayleigh_alpha: 0.15,
            tags: vec![
                "soft_octopus_arm".to_string(),
                if i == 0 { "fixed_base" } else { "tapered_ring" }.to_string(),
            ],
        });
    }

    // 2. Tapered DER/Cosserat backbone sections (each section uses mid-section circular radius)
    let elastomer = BeamMaterial {
        youngs_modulus: cfg.youngs_modulus_pa,
        poisson_ratio: cfg.poisson_ratio,
        density: cfg.density_kg_m3,
        rayleigh_alpha: 0.15,
        rayleigh_beta: 0.004,
    };
    let elem_len = sec_len / (cfg.elements_per_section as f64);

    let mut beams = Vec::with_capacity(cfg.n_sections);
    for s in 0..cfg.n_sections {
        let u_mid = ((s as f64) + 0.5) / (cfg.n_sections as f64);
        let r_sec = radius_at(u_mid);
        let x0 = (s as f64) * sec_len;
        let is_cant = cfg.free_cantilever_tip && (s + 1 == cfg.n_sections);

        let mut poses = Vec::with_capacity(cfg.elements_per_section + 1);
        for j in 0..=cfg.elements_per_section {
            poses.push(Se3::new(
                Vec3::new(x0 + (j as f64) * elem_len, 0.0, cfg.base_height_m),
                Quat::IDENTITY,
            ));
        }

        beams.push(BeamEdgeSpec {
            id: s,
            name: format!("der_arm_section_{s}"),
            body_a: s,
            clamp_a: Se3::IDENTITY,
            body_b: if is_cant { None } else { Some(s + 1) },
            clamp_b: if is_cant { None } else { Some(Se3::IDENTITY) },
            n_elements: cfg.elements_per_section,
            formulation: BeamFormulation::DiscreteElasticRod,
            section: BeamSection::Circular { radius: r_sec },
            material: elastomer,
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: vec![elem_len; cfg.elements_per_section],
            element_transforms: None,
            interior_node_contact_radius: Some(r_sec),
            enable_self_contact: true,
        });
    }

    // 3. Four tapered longitudinal muscle tendons (+Y, +Z, -Y, -Z) with D4 symmetry
    let dirs = [
        ("muscle_pos_y", Vec3::EY),
        ("muscle_pos_z", Vec3::EZ),
        ("muscle_neg_y", -Vec3::EY),
        ("muscle_neg_z", -Vec3::EZ),
    ];
    let mut tendons = Vec::new();
    if n_discs >= 2 {
        for (idx, &(tname, dir)) in dirs.iter().enumerate() {
            let mut waypoints = Vec::with_capacity(n_discs);
            let mut rest_len = 0.0_f64;
            let mut prev_world: Option<Vec3> = None;
            for b_i in 0..n_discs {
                let u = (b_i as f64) / (cfg.n_sections as f64);
                let offset = 0.85 * radius_at(u) * dir;
                let w_pt = bodies[b_i].rest_pose.transform_point(offset);
                if let Some(pw) = prev_world {
                    rest_len += (w_pt - pw).norm();
                }
                prev_world = Some(w_pt);
                waypoints.push(TendonWaypoint {
                    body_id: b_i,
                    local_pos: offset,
                    wrap_angle_rad: if b_i > 0 && b_i + 1 < n_discs {
                        0.06
                    } else {
                        0.0
                    },
                });
            }
            tendons.push(TendonEdgeSpec {
                id: idx,
                name: tname.to_string(),
                actuator_id: idx,
                waypoints,
                drive_mode: TendonDriveMode::Linear,
                rest_length: rest_len,
                slack: 0.0,
                stiffness: 600.0,
                damping: 0.8,
                capstan_mu_theta: cfg.capstan_mu_theta,
                stick_slip_enabled: cfg.capstan_mu_theta > 0.0,
                servo_torque_limit: 1.5,
                servo_speed_limit: 0.20,
                servo_deadband: 0.0,
                angle_limits: [0.0, 0.04],
                solref_imp: None,
            });
        }
    }

    // 4. Ventral suction-cup / rim contact pads on spacer rings
    let mut contact_sites = Vec::with_capacity(n_discs.saturating_sub(1));
    for i in 1..n_discs {
        let r_ring = bodies[i].collision_radius;
        contact_sites.push(ContactSiteSpec {
            id: i - 1,
            name: format!("sucker_pad_{i}"),
            attachment: ContactAttachment::RigidBody { body_id: i },
            local_pos: Vec3::ZERO,
            local_frame: Quat::IDENTITY,
            radius: r_ring,
            patch_radius: 0.5 * r_ring,
            mode: ContactMode::IsotropicPad {
                mu: 0.65,
                pivot_mu: 0.40,
                static_mu_ratio: 1.15,
                sdf_only: false,
            },
            material_profile_id: cfg.material_profile_id,
            solref_imp: None,
        });
    }

    // Pre-register differentiable physical parameters for IFT adjoint / SysID
    let diff_params = vec![
        DiffParamSpec::shared("arm_youngs_modulus", DiffParamKind::BeamYoungsModulus),
        DiffParamSpec::shared("arm_rayleigh_beta", DiffParamKind::BeamDampingBeta),
        DiffParamSpec::shared("muscle_tendon_stiffness", DiffParamKind::TendonStiffness),
    ];

    RigidFlexGraph {
        name: if cfg.free_cantilever_tip {
            format!("soft_octopus_arm_der_cantilever_S{}", cfg.n_sections)
        } else {
            format!("soft_octopus_arm_der_S{}", cfg.n_sections)
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
            rayleigh_alpha: 0.15,
            rayleigh_beta: 0.004,
            force_route: None,
        },
        diff_params,
        mesh_colliders: Vec::new(),
    }
}

/// Build a canonical fixed-free circular cantilever rod for analytical Euler buckling
/// and large-deflection nonlinear Elastica validation (PyElastica / Cosserat benchmark).
#[allow(clippy::too_many_arguments)]
pub fn build_euler_elastica_rod(
    length_m: f64,
    radius_m: f64,
    youngs_modulus_pa: f64,
    density_kg_m3: f64,
    n_elements: usize,
    formulation: BeamFormulation,
    gravity: Vec3,
    tip_mass_kg: Option<f64>,
) -> RigidFlexGraph {
    assert!(length_m > 0.0 && radius_m > 0.0 && n_elements >= 2);
    let elem_len = length_m / (n_elements as f64);
    let mut poses = Vec::with_capacity(n_elements + 1);
    for j in 0..=n_elements {
        poses.push(Se3::new(
            Vec3::new((j as f64) * elem_len, 0.0, 0.5),
            Quat::IDENTITY,
        ));
    }

    let mut bodies = vec![RigidBodySpec {
        id: 0,
        name: "clamped_root".to_string(),
        is_fixed: true,
        mass: 1.0,
        inertia: Mat3::diag([1e-2, 1e-2, 1e-2]),
        com_local: Vec3::ZERO,
        rest_pose: poses[0],
        collision_radius: radius_m,
        rayleigh_alpha: 0.05,
        tags: vec!["fixed_base".to_string()],
    }];

    let (body_b, clamp_b) = if let Some(tm) = tip_mass_kg {
        let i_tip = (0.4 * tm * radius_m * radius_m).max(1e-8);
        bodies.push(RigidBodySpec {
            id: 1,
            name: "loaded_tip".to_string(),
            is_fixed: false,
            mass: tm,
            inertia: Mat3::diag([i_tip, i_tip, i_tip]),
            com_local: Vec3::ZERO,
            rest_pose: poses[n_elements],
            collision_radius: radius_m,
            rayleigh_alpha: 0.05,
            tags: vec!["tip_load".to_string()],
        });
        (Some(1), Some(Se3::IDENTITY))
    } else {
        (None, None)
    };

    let beam = BeamEdgeSpec {
        id: 0,
        name: "elastica_rod".to_string(),
        body_a: 0,
        clamp_a: Se3::IDENTITY,
        body_b,
        clamp_b,
        n_elements,
        formulation,
        section: BeamSection::Circular { radius: radius_m },
        material: BeamMaterial {
            youngs_modulus: youngs_modulus_pa,
            poisson_ratio: 0.30,
            density: density_kg_m3,
            rayleigh_alpha: 0.15,
            rayleigh_beta: 0.002,
        },
        thin_shell_compat: false,
        prestress_frac: 0.0,
        centerline_rest_poses: poses,
        element_lengths: vec![elem_len; n_elements],
        element_transforms: None,
        interior_node_contact_radius: None,
        enable_self_contact: false,
    };

    RigidFlexGraph {
        name: "euler_elastica_rod".to_string(),
        gravity,
        bodies,
        beams: vec![beam],
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: Vec::new(),
        symmetry: SymmetryGroupSpec::trivial(),
        solver_hints: SolverHints {
            preferred_dt: 0.001,
            rayleigh_alpha: 0.15,
            rayleigh_beta: 0.002,
            force_route: None,
        },
        diff_params: vec![DiffParamSpec::shared(
            "rod_youngs_modulus",
            DiffParamKind::BeamYoungsModulus,
        )],
        mesh_colliders: Vec::new(),
    }
}

/// Exact analytical Euler critical buckling load for a fixed-free circular cantilever rod:
/// $P_{\mathrm{cr}} = \frac{\pi^2 E I}{4 L^2}$, where $I = \frac{\pi r^4}{4}$.
pub fn euler_buckling_critical_load(
    youngs_modulus_pa: f64,
    radius_m: f64,
    length_m: f64,
) -> f64 {
    let i_area = 0.25 * PI * radius_m.powi(4);
    (PI * PI * youngs_modulus_pa * i_area) / (4.0 * length_m * length_m)
}

/// Linear Euler-Bernoulli small-deflection tip displacement $\delta = \frac{F L^3}{3 E I}$
/// for a circular cantilever subjected to transverse tip force `force_n`.
pub fn cantilever_small_deflection(
    force_n: f64,
    youngs_modulus_pa: f64,
    radius_m: f64,
    length_m: f64,
) -> f64 {
    let i_area = 0.25 * PI * radius_m.powi(4);
    (force_n * length_m.powi(3)) / (3.0 * youngs_modulus_pa * i_area)
}

/// Exact large-deflection nonlinear Euler-Bernoulli Elastica solution (Bisshopp & Drucker 1945)
/// for a clamped-free cantilever rod of length `L`, bending rigidity `EI = E * pi * r^4 / 4`,
/// under a vertical downward tip load `F >= 0` (along `-Z`, rod initially along `+X`).
///
/// Uses the exact nonsingular elliptic-integral parameterization in tip angle $\phi_L \in (0, \pi/2)$:
/// $p = \sqrt{(1 + \sin\phi_L)/2}$, $u_0 = \arcsin(1 / (\sqrt{2}\,p))$,
/// $\sqrt{\alpha}\,L = \int_{u_0}^{\pi/2} \frac{du}{\sqrt{1 - p^2\sin^2 u}}$ ($\alpha = F / EI$),
/// $x_{\text{tip}} = \sqrt{\frac{2\sin\phi_L}{\alpha}}$,
/// $\delta_z = L - \frac{2}{\sqrt{\alpha}} \int_{u_0}^{\pi/2} \sqrt{1 - p^2\sin^2 u}\,du$.
///
/// Strictly monotonic for all $F \in (0, \infty)$ and accurate to $< 10^{-12}$.
/// Returns `(x_tip, delta_z_down, abs_tip_angle_rad)`.
pub fn elastica_cantilever_tip(
    force_n: f64,
    youngs_modulus_pa: f64,
    radius_m: f64,
    length_m: f64,
) -> (f64, f64, f64) {
    if force_n <= 0.0 {
        return (length_m, 0.0, 0.0);
    }
    let ei = youngs_modulus_pa * 0.25 * PI * radius_m.powi(4);
    let alpha = force_n / ei;
    let target = alpha.sqrt() * length_m;

    // Composite Simpson quadrature on [u0, pi/2] (N = 256 even subintervals -> <1e-13 accuracy).
    let eval_elliptic = |phi_l: f64| -> (f64, f64) {
        let sin_phi = phi_l.sin();
        let p2 = 0.5 * (1.0 + sin_phi);
        let u0 = (1.0 / (2.0 * p2).sqrt()).clamp(-1.0, 1.0).asin();
        let u1 = 0.5 * PI;
        let n = 256usize;
        let h = (u1 - u0) / (n as f64);
        let mut sum_f = 0.0;
        let mut sum_e = 0.0;
        for i in 0..=n {
            let u = u0 + (i as f64) * h;
            let s = u.sin();
            let rad = (1.0 - p2 * s * s).max(1e-15).sqrt();
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            sum_f += w / rad;
            sum_e += w * rad;
        }
        (sum_f * h / 3.0, sum_e * h / 3.0)
    };

    // Bisect strictly monotonic F(phi_L) = sqrt(alpha) * L over phi_L in (0, pi/2).
    let mut lo = 0.0_f64;
    let mut hi = 0.5 * PI - 1e-12;
    for _ in 0..90 {
        let mid = 0.5 * (lo + hi);
        let (f_val, _) = eval_elliptic(mid);
        if f_val < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let phi_l = 0.5 * (lo + hi);
    let (_, e_val) = eval_elliptic(phi_l);
    let x_tip = (2.0 * phi_l.sin() / alpha).sqrt();
    let delta_z_down = (length_m - 2.0 * e_val / alpha.sqrt()).max(0.0);
    (x_tip, delta_z_down, phi_l)
}
