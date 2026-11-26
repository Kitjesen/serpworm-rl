//! Archetype 7: `RigidFlexGrid-MxN` (default `4x4`: 16 rigid bodies, 24 flexible beams,
//! cyclomatic number $\beta_1 = 24 - 16 + 1 = 9$, `SchurRoute::SparseGeneral`).
//!
//! Cyclic rigid-flexible metamaterial lattice exercising general Sparse Block-$LDL^\top$
//! and `faer` SIMD supernodal factorization scaling (`4x4`, `6x6`, arbitrary `MxN`).

use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, DiffParamKind,
    DiffParamSpec, RigidBodySpec, RigidFlexGraph, SolverHints, SymmetryElement, SymmetryGroupKind,
    SymmetryGroupSpec, TendonDriveMode, TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};
use serde::{Deserialize, Serialize};

/// Configuration for the cyclic `RigidFlexGrid-MxN` metamaterial lattice (`SparseGeneral`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlexGridConfig {
    /// Grid rows `M >= 2` (default `4`).
    pub rows: usize,
    /// Grid columns `N >= 2` (default `4`).
    pub cols: usize,
    /// Center-to-center spacing between adjacent rigid hubs (m).
    pub pitch_m: f64,
    /// Number of corotational beam elements per flexible strut (`>= 2`).
    pub elements_per_beam: usize,
    /// Resting height of hub centers above `z = 0` (m).
    pub height_m: f64,
    /// Mass of each rigid lattice hub (kg).
    pub hub_mass_kg: f64,
    /// Optional Dirichlet pinning of the 4 outer corners (`false` = free floating/grounded grid).
    pub pin_corners: bool,
    /// Beam Young's modulus (Pa).
    pub youngs_modulus_pa: f64,
    /// Beam rectangular width (m) & thickness (m).
    pub beam_width_m: f64,
    pub beam_thickness_m: f64,
    /// Preferred timestep (s).
    pub preferred_dt: f64,
    /// Material profile ID for hub ground pads.
    pub material_profile_id: usize,
}

impl Default for FlexGridConfig {
    fn default() -> Self {
        Self {
            rows: 4,
            cols: 4,
            pitch_m: 0.12,
            elements_per_beam: 2,
            height_m: 0.045,
            hub_mass_kg: 0.08,
            pin_corners: false,
            youngs_modulus_pa: 65e9,
            beam_width_m: 0.010,
            beam_thickness_m: 0.0005,
            preferred_dt: 0.001,
            material_profile_id: 0,
        }
    }
}

/// Build an orthonormal rotation quaternion whose local `+X` axis points along `dir`.
fn frame_from_x(dir: Vec3) -> Quat {
    let tx = dir.normalized();
    let up = if tx.z().abs() < 0.9 { Vec3::EZ } else { Vec3::EY };
    let ty = up.cross(tx).normalized();
    let tz = tx.cross(ty);
    Quat::from_mat3(Mat3([
        [tx.x(), ty.x(), tz.x()],
        [tx.y(), ty.y(), tz.y()],
        [tx.z(), ty.z(), tz.z()],
    ]))
}

/// Build `Z2` bilateral reflection symmetry for the 4 perimeter actuation tendons.
fn make_grid_z2_symmetry() -> SymmetryGroupSpec {
    // 4 perimeter edge tendons: [0: south, 1: north, 2: west, 3: east]
    // Reflecting across X swaps south <-> north (0 <-> 1) and keeps west/east unchanged
    let refl = SymmetryElement {
        name: "sn_mirror".to_string(),
        obs_perm: vec![1, 0, 2, 3],
        obs_sign: vec![1.0; 4],
        act_perm: vec![1, 0, 2, 3],
        act_sign: vec![1.0; 4],
        est_perm: Vec::new(),
        est_sign: Vec::new(),
    };
    SymmetryGroupSpec::from_generators(SymmetryGroupKind::Z2Bilateral, 4, 4, 0, &[refl], 2)
        .expect("Grid Z2 symmetry closure must succeed")
}

/// Build a cyclic `M x N` rigid-flexible metamaterial lattice (`SchurRoute::SparseGeneral`).
///
/// For `4 x 4` (unpinned): 16 free rigid hubs, 24 flexible beams, cyclomatic number $\beta_1 = 9$.
/// For `6 x 6` (unpinned): 36 free rigid hubs, 60 flexible beams, cyclomatic number $\beta_1 = 25$.
pub fn build_rigid_flex_grid(cfg: &FlexGridConfig) -> RigidFlexGraph {
    assert!(cfg.rows >= 2 && cfg.cols >= 2, "grid must be at least 2x2");
    assert!(cfg.elements_per_beam >= 2, "elements_per_beam must be >= 2");

    let n_bodies = cfg.rows * cfg.cols;
    let hub_r = 0.018_f64;
    let i_hub = 0.4 * cfg.hub_mass_kg * hub_r * hub_r;
    let hub_inertia = Mat3::diag([i_hub, i_hub, i_hub]);

    let x_center = 0.5 * ((cfg.cols - 1) as f64) * cfg.pitch_m;
    let y_center = 0.5 * ((cfg.rows - 1) as f64) * cfg.pitch_m;

    let mut bodies = Vec::with_capacity(n_bodies);
    let mut contact_sites = Vec::with_capacity(n_bodies);

    for r in 0..cfg.rows {
        for c in 0..cfg.cols {
            let id = r * cfg.cols + c;
            let is_corner =
                (r == 0 || r + 1 == cfg.rows) && (c == 0 || c + 1 == cfg.cols);
            let is_fixed = cfg.pin_corners && is_corner;
            let pos = Vec3::new(
                (c as f64) * cfg.pitch_m - x_center,
                (r as f64) * cfg.pitch_m - y_center,
                cfg.height_m,
            );
            bodies.push(RigidBodySpec {
                id,
                name: format!("hub_{r}_{c}"),
                is_fixed,
                mass: cfg.hub_mass_kg,
                inertia: hub_inertia,
                com_local: Vec3::ZERO,
                rest_pose: Se3::new(pos, Quat::IDENTITY),
                collision_radius: hub_r,
                rayleigh_alpha: 0.10,
                tags: vec![format!("row_{r}"), format!("col_{c}")],
            });

            if !is_fixed {
                let cid = contact_sites.len();
                contact_sites.push(ContactSiteSpec {
                    id: cid,
                    name: format!("hub_{r}_{c}_pad"),
                    attachment: ContactAttachment::RigidBody { body_id: id },
                    local_pos: Vec3::new(0.0, 0.0, -(cfg.height_m - 0.013)),
                    local_frame: Quat::IDENTITY,
                    radius: 0.012,
                    patch_radius: 0.008,
                    mode: ContactMode::IsotropicPad {
                        mu: 0.75,
                        pivot_mu: 0.45,
                        static_mu_ratio: 1.08,
                        sdf_only: false,
                    },
                    material_profile_id: cfg.material_profile_id,
                    solref_imp: None,
                });
            }
        }
    }

    // Flexible beams along all horizontal & vertical lattice edges
    let mat = BeamMaterial {
        youngs_modulus: cfg.youngs_modulus_pa,
        poisson_ratio: 0.30,
        density: 6450.0,
        rayleigh_alpha: 0.10,
        rayleigh_beta: 0.002,
    };
    let sec = BeamSection::Rectangular {
        width: cfg.beam_width_m,
        thickness: cfg.beam_thickness_m,
    };

    let n_beams = (cfg.cols - 1) * cfg.rows + cfg.cols * (cfg.rows - 1);
    let mut beams = Vec::with_capacity(n_beams);

    let mut push_beam = |name: String, ba: usize, bb: usize| {
        let pa = bodies[ba].rest_pose.pos;
        let pb = bodies[bb].rest_pose.pos;
        let dir = (pb - pa).normalized();
        let q = frame_from_x(dir);
        let local_a = hub_r * dir;
        let local_b = -hub_r * dir;
        let world_a = pa + local_a;
        let world_b = pb + local_b;
        let span = world_b - world_a;
        let elem_len = span.norm() / (cfg.elements_per_beam as f64);

        let mut poses = Vec::with_capacity(cfg.elements_per_beam + 1);
        for j in 0..=cfg.elements_per_beam {
            let u = (j as f64) / (cfg.elements_per_beam as f64);
            poses.push(Se3::new(world_a + u * span, q));
        }

        let clamp_a = Se3::new(local_a, q);
        let clamp_b = Se3::new(local_b, q);
        let mut elem_tfs = vec![[Se3::IDENTITY, Se3::IDENTITY]; cfg.elements_per_beam];
        elem_tfs[0][0] = clamp_a;
        elem_tfs[cfg.elements_per_beam - 1][1] = clamp_b;

        let bid = beams.len();
        beams.push(BeamEdgeSpec {
            id: bid,
            name,
            body_a: ba,
            clamp_a,
            body_b: Some(bb),
            clamp_b: Some(clamp_b),
            n_elements: cfg.elements_per_beam,
            formulation: BeamFormulation::CorotationalEulerBernoulli,
            section: sec,
            material: mat,
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: vec![elem_len; cfg.elements_per_beam],
            element_transforms: Some(elem_tfs),
            interior_node_contact_radius: None,
            enable_self_contact: false,
        });
    };

    for r in 0..cfg.rows {
        for c in 0..(cfg.cols - 1) {
            push_beam(
                format!("hbeam_{r}_{c}"),
                r * cfg.cols + c,
                r * cfg.cols + c + 1,
            );
        }
    }
    for r in 0..(cfg.rows - 1) {
        for c in 0..cfg.cols {
            push_beam(
                format!("vbeam_{r}_{c}"),
                r * cfg.cols + c,
                (r + 1) * cfg.cols + c,
            );
        }
    }

    // 4 multi-waypoint perimeter morphing tendons (south, north, west, east edges)
    let make_edge_tendon = |id: usize, name: &str, node_ids: Vec<usize>| -> TendonEdgeSpec {
        let z_off = Vec3::new(0.0, 0.0, 0.012);
        let mut rest_len = 0.0;
        for w in node_ids.windows(2) {
            let p0 = bodies[w[0]].rest_pose.transform_point(z_off);
            let p1 = bodies[w[1]].rest_pose.transform_point(z_off);
            rest_len += (p1 - p0).norm();
        }
        TendonEdgeSpec {
            id,
            name: name.to_string(),
            actuator_id: id,
            waypoints: node_ids
                .into_iter()
                .map(|bid| TendonWaypoint {
                    body_id: bid,
                    local_pos: z_off,
                    wrap_angle_rad: 0.0,
                })
                .collect(),
            drive_mode: TendonDriveMode::Linear,
            rest_length: rest_len,
            slack: 0.0,
            stiffness: 1500.0,
            damping: 1.5,
            capstan_mu_theta: 0.0,
            stick_slip_enabled: false,
            servo_torque_limit: 3.0,
            servo_speed_limit: 0.25,
            servo_deadband: 0.0,
            angle_limits: [0.0, 0.05],
            solref_imp: None,
        }
    };

    let south: Vec<usize> = (0..cfg.cols).collect();
    let north: Vec<usize> = (0..cfg.cols).map(|c| (cfg.rows - 1) * cfg.cols + c).collect();
    let west: Vec<usize> = (0..cfg.rows).map(|r| r * cfg.cols).collect();
    let east: Vec<usize> = (0..cfg.rows)
        .map(|r| r * cfg.cols + (cfg.cols - 1))
        .collect();

    let tendons = vec![
        make_edge_tendon(0, "perimeter_south", south),
        make_edge_tendon(1, "perimeter_north", north),
        make_edge_tendon(2, "perimeter_west", west),
        make_edge_tendon(3, "perimeter_east", east),
    ];

    RigidFlexGraph {
        name: format!("rigid_flex_grid_{}x{}", cfg.rows, cfg.cols),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams,
        tendons,
        joints: Vec::new(),
        contact_sites,
        symmetry: make_grid_z2_symmetry(),
        solver_hints: SolverHints {
            preferred_dt: cfg.preferred_dt,
            rayleigh_alpha: 0.10,
            rayleigh_beta: 0.002,
            force_route: None,
        },
        diff_params: vec![
            DiffParamSpec::shared("grid_beam_youngs", DiffParamKind::BeamYoungsModulus),
            DiffParamSpec::shared("grid_hub_mass", DiffParamKind::BodyMass),
        ],
        mesh_colliders: Vec::new(),
    }
}

/// Convenience builder for the canonical `4x4` cyclic metamaterial lattice
/// (16 rigid bodies, 24 beams, $\beta_1 = 9$, `SchurRoute::SparseGeneral`).
pub fn build_rigid_flex_grid_4x4() -> RigidFlexGraph {
    build_rigid_flex_grid(&FlexGridConfig::default())
}

/// Convenience builder for the `6x6` cyclic metamaterial lattice
/// (36 rigid bodies, 60 beams, $\beta_1 = 25$, `SchurRoute::SparseGeneral`) for `faer` vs Block-CSC benchmarks.
pub fn build_rigid_flex_grid_6x6() -> RigidFlexGraph {
    build_rigid_flex_grid(&FlexGridConfig {
        rows: 6,
        cols: 6,
        ..Default::default()
    })
}
