//! Exact `SerpWorm-N` (`N >= 1`) parametric robot builder matching `assets/inputs/parameters.json`,
//! `src/serpworm/physics/topology.py`, and `src/serpworm/physics/gpu_core.py` to FP64 machine precision.

use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, JointEdgeSpec,
    JointType, RigidBodySpec, RigidFlexGraph, SolverHints, SymmetryElement, SymmetryGroupKind,
    SymmetryGroupSpec, TendonDriveMode, TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Passive ratchet wheel radius (m) matching `serpworm/physics/topology.py`.
pub const WHEEL_RADIUS_M: f64 = 0.018;

/// Vertical wheel mount drop vector (m) matching `serpworm/physics/topology.py`.
pub const WHEEL_DROP_M: [f64; 3] = [0.0, 0.0, -0.045];

/// Plate wheel-mount hole coordinates (m) indexed by `[plate_parity (0=front,1=back)][side (0=L,1=R)]`.
pub const WHEEL_HOLES_M: [[[f64; 3]; 2]; 2] = [
    [
        [-0.014, 0.0617601286221, -0.0689776402648],
        [-0.014, -0.0617601286221, -0.0689776402648],
    ],
    [
        [-0.0005, 0.0617601286221, -0.0700229145182],
        [-0.0005, -0.0617601286221, -0.0700229145182],
    ],
];

/// Inter-segment yaw connector arm on back plate (m) matching `gpu_core.py:367`.
pub const CONNECTOR_ARM_M: [f64; 3] = [-0.058, 0.0, -0.0010452742534];

/// Nominal 0.28 kg plate diagonal rotational inertia (kg*m^2) matching `gpu_core.py:368`.
pub const BASE_PLATE_MASS_KG: f64 = 0.28;
pub const BASE_PLATE_INERTIA_DIAG: [f64; 3] = [
    0.0016 * BASE_PLATE_MASS_KG,
    0.0009 * BASE_PLATE_MASS_KG,
    0.0009 * BASE_PLATE_MASS_KG,
];

/// Exact subset of `assets/inputs/parameters.json` governing strip geometry and material properties.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParametersJson {
    pub strip_count: usize,
    pub elements_per_strip: usize,
    pub strip_width_m: f64,
    pub strip_thickness_m: f64,
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub steel_density_kg_m3: f64,
    pub stress_free_bow_m: f64,
    #[serde(default = "default_rayleigh_beta")]
    pub stiffness_proportional_damping_s: f64,
    pub front_clamp_hole_pairs_m: [[[f64; 3]; 2]; 8],
    pub back_clamp_hole_pairs_m: [[[f64; 3]; 2]; 8],
}

fn default_rayleigh_beta() -> f64 {
    0.002
}

impl ParametersJson {
    /// Exact geometry & material values from `assets/inputs/parameters.json`.
    pub fn default_embedded() -> Self {
        Self {
            strip_count: 8,
            elements_per_strip: 12,
            strip_width_m: 0.016,
            strip_thickness_m: 0.00015,
            youngs_modulus_pa: 200_000_000_000.0,
            poisson_ratio: 0.3,
            steel_density_kg_m3: 7850.0,
            stress_free_bow_m: 0.05071737725588745,
            stiffness_proportional_damping_s: 0.002,
            front_clamp_hole_pairs_m: [
                [[-0.0165, 0.04810739, 0.01709522], [-0.0165, 0.04428056, 0.02633402]],
                [[-0.0165, 0.02375357, 0.046861], [-0.0165, 0.01451477, 0.05068784]],
                [[-0.0165, -0.01451477, 0.05068784], [-0.0165, -0.02375357, 0.046861]],
                [[-0.0165, -0.04428056, 0.02633402], [-0.0165, -0.04810739, 0.01709522]],
                [[-0.0165, -0.04810739, -0.01193433], [-0.0165, -0.04428056, -0.02117312]],
                [[-0.0165, -0.02375357, -0.04170011], [-0.0165, -0.01451477, -0.04552695]],
                [[-0.0165, 0.01451477, -0.04552695], [-0.0165, 0.02375357, -0.04170011]],
                [[-0.0165, 0.04428056, -0.02117312], [-0.0165, 0.04810739, -0.01193433]],
            ],
            back_clamp_hole_pairs_m: [
                [[0.0, 0.04810739, 0.01604995], [0.0, 0.04428056, 0.02528874]],
                [[0.0, 0.02375357, 0.04581573], [0.0, 0.01451477, 0.04964257]],
                [[0.0, -0.01451477, 0.04964257], [0.0, -0.02375357, 0.04581573]],
                [[0.0, -0.04428056, 0.02528874], [0.0, -0.04810739, 0.01604995]],
                [[0.0, -0.04810739, -0.0129796], [0.0, -0.04428056, -0.0222184]],
                [[0.0, -0.02375357, -0.04274539], [0.0, -0.01451477, -0.04657222]],
                [[0.0, 0.01451477, -0.04657222], [0.0, 0.02375357, -0.04274539]],
                [[0.0, 0.04428056, -0.0222184], [0.0, 0.04810739, -0.0129796]],
            ],
        }
    }

    /// Parse from a JSON string (ignores extra metadata fields in `parameters.json`).
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Load directly from `assets/inputs/parameters.json` (or any compatible JSON path).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, crate::RobotLoadError> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Ok(Self::from_json_str(&content)?)
    }

    /// Attempt to load `assets/inputs/parameters.json` relative to common repo roots,
    /// falling back to the exact embedded copy.
    pub fn load_repo_or_embedded() -> Self {
        let candidates = [
            "assets/inputs/parameters.json",
            "../../assets/inputs/parameters.json",
            "../../../assets/inputs/parameters.json",
            "../../../../assets/inputs/parameters.json",
        ];
        for cand in candidates {
            if let Ok(p) = Self::from_file(cand) {
                return p;
            }
        }
        Self::default_embedded()
    }
}

/// Hardware configuration preset matching `serpworm.config.HardwareConfig`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerpWormPreset {
    /// Exact match to `GPUCore(..., hardware=None)` (legacy DS3218 defaults).
    #[default]
    GpuCoreDefault,
    /// Exact match to `HardwareConfig.ds3218_default()`.
    Ds3218Default,
    /// Exact match to `HardwareConfig.bench_exp128_calibrated()`.
    BenchExp128Calibrated,
    /// Exact match to `HardwareConfig.v5_realistic(terrain='hard')`.
    V5RealisticHard,
}

/// Complete configuration for building `SerpWorm-N`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SerpWormConfig {
    pub n_segments: usize,
    pub elements_per_strip: usize,
    pub preferred_dt: f64,
    pub beam_formulation: BeamFormulation,
    pub strip_width_m: Option<f64>,
    pub strip_thickness_m: Option<f64>,
    pub stress_free_bow_m: Option<f64>,
    pub strip_thin_shell_compat: bool,
    pub strip_prestress_frac: f64,
    pub strip_node_contact: bool,
    pub rayleigh_alpha: f64,
    pub rayleigh_beta: f64,
    pub plate_masses_kg: Option<Vec<f64>>,
    pub tendon_crank_mode: bool,
    pub tendon_horn_radius_m: f64,
    pub tendon_guide_dist_m: f64,
    pub tendon_slack_m: f64,
    pub tendon_stiffness_n_m: f64,
    pub tendon_damping_ns_m: f64,
    pub tendon_y_m: f64,
    pub tendon_capstan_mu_theta: f64,
    pub servo_torque_limit_nm: f64,
    pub servo_speed_limit_rad_s: f64,
    pub servo_deadband_rad: f64,
    pub plate_stop_gap_m: f64,
    pub yaw_kp_nm_rad: f64,
    pub yaw_kd_nms_rad: f64,
    pub yaw_torque_limit_nm: f64,
    pub yaw_speed_limit_rad_s: f64,
    pub yaw_static_friction_nm: f64,
    pub ground_contact: bool,
    pub wheel_mu: f64,
    pub wheel_mu_lat_ratio: f64,
    pub wheel_rolling: f64,
    pub wheel_mass_kg: f64,
    pub wheel_patch_radius_m: f64,
    pub wheel_pivot_mu: f64,
    pub ratchet_backlash_rad: f64,
    pub wheel_static_mu_ratio: f64,
    pub material_profile_id: usize,
}

impl SerpWormConfig {
    /// Create configuration from a named hardware preset matching `serpworm.config.HardwareConfig`.
    pub fn from_preset(preset: SerpWormPreset, n_segments: usize, elements_per_strip: usize) -> Self {
        match preset {
            SerpWormPreset::GpuCoreDefault => Self {
                n_segments,
                elements_per_strip,
                preferred_dt: 0.00025,
                beam_formulation: BeamFormulation::CorotationalEulerBernoulli,
                strip_width_m: None,
                strip_thickness_m: None,
                stress_free_bow_m: None,
                strip_thin_shell_compat: false,
                strip_prestress_frac: 0.0,
                strip_node_contact: false,
                rayleigh_alpha: 0.05,
                rayleigh_beta: 0.002,
                plate_masses_kg: None,
                tendon_crank_mode: false,
                tendon_horn_radius_m: 0.0235,
                tendon_guide_dist_m: 0.0350,
                tendon_slack_m: 0.0,
                tendon_stiffness_n_m: 2000.0,
                tendon_damping_ns_m: 0.5,
                tendon_y_m: 0.0285,
                tendon_capstan_mu_theta: 0.0,
                servo_torque_limit_nm: 0.5,
                servo_speed_limit_rad_s: 2.0,
                servo_deadband_rad: 0.0,
                plate_stop_gap_m: 0.075,
                yaw_kp_nm_rad: 2.0,
                yaw_kd_nms_rad: 0.03,
                yaw_torque_limit_nm: 0.5,
                yaw_speed_limit_rad_s: -1.0,
                yaw_static_friction_nm: 0.0,
                ground_contact: true,
                wheel_mu: 0.80,
                wheel_mu_lat_ratio: 1.0,
                wheel_rolling: 0.06,
                wheel_mass_kg: 0.009,
                wheel_patch_radius_m: 0.0,
                wheel_pivot_mu: 0.65,
                ratchet_backlash_rad: 0.0,
                wheel_static_mu_ratio: 1.0,
                material_profile_id: 0,
            },
            SerpWormPreset::Ds3218Default => Self {
                preferred_dt: 0.0005,
                yaw_kp_nm_rad: 10.0,
                yaw_kd_nms_rad: 0.12,
                yaw_torque_limit_nm: 2.45,
                yaw_speed_limit_rad_s: 8.2,
                servo_torque_limit_nm: 2.45,
                servo_speed_limit_rad_s: 8.2,
                ..Self::from_preset(SerpWormPreset::GpuCoreDefault, n_segments, elements_per_strip)
            },
            SerpWormPreset::BenchExp128Calibrated => Self {
                preferred_dt: 0.0005,
                strip_thickness_m: Some(0.000220),
                stress_free_bow_m: Some(0.0300),
                tendon_crank_mode: true,
                tendon_horn_radius_m: 0.0250,
                tendon_guide_dist_m: 0.0360,
                tendon_slack_m: 0.0114,
                tendon_stiffness_n_m: 8000.0,
                tendon_damping_ns_m: 2.5,
                plate_stop_gap_m: 0.068,
                ..Self::from_preset(SerpWormPreset::Ds3218Default, n_segments, elements_per_strip)
            },
            SerpWormPreset::V5RealisticHard => {
                let mut masses = Vec::with_capacity(2 * n_segments);
                for p in 0..(2 * n_segments) {
                    masses.push(if p == 0 {
                        0.1460
                    } else if p % 2 == 0 {
                        0.2530
                    } else {
                        0.2893
                    });
                }
                Self {
                    rayleigh_alpha: 0.05,
                    rayleigh_beta: 0.0005,
                    plate_masses_kg: Some(masses),
                    wheel_mass_kg: 0.0179,
                    tendon_capstan_mu_theta: 0.18,
                    yaw_static_friction_nm: 0.06,
                    servo_deadband_rad: 0.5_f64.to_radians(),
                    wheel_static_mu_ratio: 1.18,
                    wheel_mu_lat_ratio: 1.05,
                    wheel_patch_radius_m: 0.008,
                    ratchet_backlash_rad: 10.0_f64.to_radians(),
                    ..Self::from_preset(
                        SerpWormPreset::BenchExp128Calibrated,
                        n_segments,
                        elements_per_strip,
                    )
                }
            }
        }
    }
}

impl Default for SerpWormConfig {
    fn default() -> Self {
        Self::from_preset(SerpWormPreset::GpuCoreDefault, 5, 16)
    }
}

/// Build the $\mathbb{Z}_2$ bilateral symmetry group for an `n_segments` SerpWorm
/// (`2*n_segments` tendon actuators + `n_segments - 1` yaw actuators).
pub fn make_z2_bilateral_symmetry(n_segments: usize) -> SymmetryGroupSpec {
    let n_t = 2 * n_segments;
    let n_y = n_segments.saturating_sub(1);
    let n_a = n_t + n_y;
    let mut act_perm = Vec::with_capacity(n_a);
    let mut act_sign = Vec::with_capacity(n_a);
    for seg in 0..n_segments {
        act_perm.push(2 * seg + 1);
        act_perm.push(2 * seg);
        act_sign.push(1.0);
        act_sign.push(1.0);
    }
    for j in 0..n_y {
        act_perm.push(n_t + j);
        act_sign.push(-1.0);
    }
    // 7D EstNet [vx, vy, wz, wall_conf, wall_side, drag, slip]
    let est_perm = (0..7).collect();
    let est_sign = vec![1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0];
    let mirror = SymmetryElement {
        name: "lr_mirror".to_string(),
        obs_perm: act_perm.clone(),
        obs_sign: act_sign.clone(),
        act_perm,
        act_sign,
        est_perm,
        est_sign,
    };
    SymmetryGroupSpec::from_generators(
        SymmetryGroupKind::Z2Bilateral,
        n_a,
        n_a,
        7,
        &[mirror],
        2,
    )
    .expect("Z2 bilateral symmetry closure must succeed")
}

/// Build an exact `RigidFlexGraph` for `SerpWorm-N` matching `serpworm.physics.topology` and `GPUCore`.
pub fn build_serpworm(cfg: &SerpWormConfig, params_opt: Option<&ParametersJson>) -> RigidFlexGraph {
    assert!(cfg.n_segments >= 1, "n_segments must be >= 1");
    assert!(cfg.elements_per_strip >= 2, "elements_per_strip must be >= 2");

    let owned_params;
    let params = match params_opt {
        Some(p) => p,
        None => {
            owned_params = ParametersJson::load_repo_or_embedded();
            &owned_params
        }
    };

    let n_seg = cfg.n_segments;
    let n_elem = cfg.elements_per_strip;
    let n_plates = 2 * n_seg;

    let strip_width = cfg.strip_width_m.unwrap_or(params.strip_width_m);
    let strip_thickness = cfg.strip_thickness_m.unwrap_or(params.strip_thickness_m);
    let bow = cfg.stress_free_bow_m.unwrap_or(params.stress_free_bow_m);

    // Exact wheel mounts = WHEEL_HOLES_M + WHEEL_DROP_M
    let mut wheel_mounts = [[[0.0; 3]; 2]; 2];
    for parity in 0..2 {
        for side in 0..2 {
            for c in 0..3 {
                wheel_mounts[parity][side][c] = WHEEL_HOLES_M[parity][side][c] + WHEEL_DROP_M[c];
            }
        }
    }
    let z0 = -wheel_mounts[0][0][2] + WHEEL_RADIUS_M + 0.001;

    // 1. Rigid plates (2 * n_seg)
    let mut bodies = Vec::with_capacity(n_plates);
    for seg in 0..n_seg {
        for j in 0..2 {
            let p = 2 * seg + j;
            let mass = cfg
                .plate_masses_kg
                .as_ref()
                .and_then(|v| v.get(p).copied())
                .unwrap_or(BASE_PLATE_MASS_KG);
            let scale = mass / BASE_PLATE_MASS_KG;
            let inertia = Mat3::diag([
                BASE_PLATE_INERTIA_DIAG[0] * scale,
                BASE_PLATE_INERTIA_DIAG[1] * scale,
                BASE_PLATE_INERTIA_DIAG[2] * scale,
            ]);
            let pos = Vec3::new(
                -(seg as f64) * 0.1755 - (j as f64) * 0.1175,
                0.0,
                z0 + (j as f64) * 0.0010452742534,
            );
            bodies.push(RigidBodySpec {
                id: p,
                name: format!("plate_{p}"),
                is_fixed: false,
                mass,
                inertia,
                com_local: Vec3::ZERO,
                rest_pose: Se3::new(pos, Quat::IDENTITY),
                collision_radius: 0.055,
                rayleigh_alpha: cfg.rayleigh_alpha,
                tags: vec![
                    format!("segment_{seg}"),
                    if j == 0 { "front_plate" } else { "back_plate" }.to_string(),
                ],
            });
        }
    }

    // 2. Pre-curved spring-steel strips (8 * n_seg)
    let beam_mat = BeamMaterial {
        youngs_modulus: params.youngs_modulus_pa,
        poisson_ratio: params.poisson_ratio,
        density: params.steel_density_kg_m3,
        rayleigh_alpha: cfg.rayleigh_alpha,
        rayleigh_beta: cfg.rayleigh_beta,
    };
    let beam_sec = BeamSection::Rectangular {
        width: strip_width,
        thickness: strip_thickness,
    };

    let mut beams = Vec::with_capacity(8 * n_seg);
    for seg in 0..n_seg {
        let pa_pos = bodies[2 * seg].rest_pose.pos;
        let pb_pos = bodies[2 * seg + 1].rest_pose.pos;
        for strip in 0..8 {
            let sid = seg * 8 + strip;
            let f0 = Vec3::from(params.front_clamp_hole_pairs_m[strip][0]);
            let f1 = Vec3::from(params.front_clamp_hole_pairs_m[strip][1]);
            let b0 = Vec3::from(params.back_clamp_hole_pairs_m[strip][0]);
            let b1 = Vec3::from(params.back_clamp_hole_pairs_m[strip][1]);

            let front = 0.5 * (f0 + f1);
            let back = 0.5 * (b0 + b1);
            let a = pa_pos + front;
            let b = pb_pos + back;

            let radial = Vec3::new(0.0, front.y(), front.z()).normalized();
            let side = (f0 - f1).normalized();

            let mut frames = Vec::with_capacity(n_elem + 1);
            for j in 0..=n_elem {
                let u = (j as f64) / (n_elem as f64);
                let point = a + u * (b - a) + (4.0 * bow * u * (1.0 - u)) * radial;
                let tangent = (b - a + (4.0 * bow * (1.0 - 2.0 * u)) * radial).normalized();
                let width_axis = (side - side.dot(tangent) * tangent).normalized();
                let normal_axis = tangent.cross(width_axis);
                // Rotation matrix with columns (tangent, width_axis, normal_axis)
                let r_mat = Mat3([
                    [tangent.x(), width_axis.x(), normal_axis.x()],
                    [tangent.y(), width_axis.y(), normal_axis.y()],
                    [tangent.z(), width_axis.z(), normal_axis.z()],
                ]);
                let q = Quat::from_mat3(r_mat);
                frames.push(Se3::new(point, q));
            }

            let mut element_lengths = Vec::with_capacity(n_elem);
            for j in 0..n_elem {
                element_lengths.push((frames[j + 1].pos - frames[j].pos).norm());
            }

            let clamp_a = Se3::new(front, frames[0].rot);
            let clamp_b = Se3::new(back, frames[n_elem].rot);
            let mut elem_tfs = vec![[Se3::IDENTITY, Se3::IDENTITY]; n_elem];
            elem_tfs[0][0] = clamp_a;
            elem_tfs[n_elem - 1][1] = clamp_b;

            beams.push(BeamEdgeSpec {
                id: sid,
                name: format!("seg{seg}_strip{strip}"),
                body_a: 2 * seg,
                clamp_a,
                body_b: Some(2 * seg + 1),
                clamp_b: Some(clamp_b),
                n_elements: n_elem,
                formulation: cfg.beam_formulation,
                section: beam_sec,
                material: beam_mat,
                thin_shell_compat: cfg.strip_thin_shell_compat,
                prestress_frac: cfg.strip_prestress_frac,
                centerline_rest_poses: frames,
                element_lengths,
                element_transforms: Some(elem_tfs),
                interior_node_contact_radius: if cfg.strip_node_contact {
                    Some(0.002)
                } else {
                    None
                },
                enable_self_contact: false,
            });
        }
    }

    // 3. Axial tendons (4 per segment = 2 per left/right servo)
    let ty = cfg.tendon_y_m;
    let anchors = [
        Vec3::new(-0.0165, ty, 0.039),
        Vec3::new(-0.0165, ty, -0.039),
        Vec3::new(-0.0165, -ty, 0.039),
        Vec3::new(-0.0165, -ty, -0.039),
    ];
    let guides = [
        Vec3::new(0.0, ty, 0.039),
        Vec3::new(0.0, ty, -0.039),
        Vec3::new(0.0, -ty, 0.039),
        Vec3::new(0.0, -ty, -0.039),
    ];
    let drive_mode = if cfg.tendon_crank_mode {
        TendonDriveMode::EccentricCrank {
            horn_radius: cfg.tendon_horn_radius_m,
            guide_dist: cfg.tendon_guide_dist_m,
        }
    } else {
        TendonDriveMode::LegacyWrap {
            horn_radius: cfg.tendon_horn_radius_m,
            guide_dist: cfg.tendon_guide_dist_m,
        }
    };

    let mut tendons = Vec::with_capacity(4 * n_seg);
    for seg in 0..n_seg {
        for side_i in 0..2 {
            let side_lbl = if side_i == 0 { "left" } else { "right" };
            let actuator_id = 2 * seg + side_i;
            for k_pair in 0..2 {
                let k = 2 * side_i + k_pair;
                let tid = tendons.len();
                tendons.push(TendonEdgeSpec {
                    id: tid,
                    name: format!("seg{seg}_tendon_{side_lbl}_{k_pair}"),
                    actuator_id,
                    waypoints: vec![
                        TendonWaypoint {
                            body_id: 2 * seg,
                            local_pos: anchors[k],
                            wrap_angle_rad: 0.0,
                        },
                        TendonWaypoint {
                            body_id: 2 * seg + 1,
                            local_pos: guides[k],
                            wrap_angle_rad: 0.0,
                        },
                    ],
                    drive_mode,
                    rest_length: 0.101,
                    slack: cfg.tendon_slack_m,
                    stiffness: cfg.tendon_stiffness_n_m,
                    damping: cfg.tendon_damping_ns_m,
                    capstan_mu_theta: cfg.tendon_capstan_mu_theta,
                    stick_slip_enabled: cfg.tendon_capstan_mu_theta > 0.0,
                    servo_torque_limit: cfg.servo_torque_limit_nm,
                    servo_speed_limit: cfg.servo_speed_limit_rad_s,
                    servo_deadband: cfg.servo_deadband_rad,
                    angle_limits: [0.0, std::f64::consts::PI],
                    solref_imp: None,
                });
            }
        }
    }

    // 4. Stop-gap axial guards + inter-segment yaw revolute connectors
    let mut joints = Vec::with_capacity(n_seg + n_seg.saturating_sub(1));
    for seg in 0..n_seg {
        let jid = joints.len();
        joints.push(JointEdgeSpec {
            id: jid,
            name: format!("seg{seg}_stop_gap"),
            body_a: 2 * seg,
            body_b: 2 * seg + 1,
            joint_type: JointType::AxialStopGap {
                axis_local_a: Vec3::new(-1.0, 0.0, 0.0),
                min_gap: cfg.plate_stop_gap_m,
                stiffness: 50_000.0,
            },
            actuator_id: None,
            kp: 0.0,
            kd: 0.0,
            torque_limit: 0.0,
            speed_limit: 0.0,
            static_friction: 0.0,
            deadband: 0.0,
            limits: None,
            implicit_coupling: false,
            solref_imp: None,
        });
    }
    for j in 0..n_seg.saturating_sub(1) {
        let jid = joints.len();
        joints.push(JointEdgeSpec {
            id: jid,
            name: format!("inter_seg_yaw_{j}"),
            body_a: 2 * j + 1,
            body_b: 2 * j + 2,
            joint_type: JointType::Revolute {
                axis_local_a: Vec3::EZ,
                anchor_a: Vec3::from(CONNECTOR_ARM_M),
                anchor_b: Vec3::ZERO,
                pos_stiffness: 10_000.0,
                pos_damping: 10.0,
                align_stiffness: 20.0,
                align_damping: 0.1,
            },
            actuator_id: Some(2 * n_seg + j),
            kp: cfg.yaw_kp_nm_rad,
            kd: cfg.yaw_kd_nms_rad,
            torque_limit: cfg.yaw_torque_limit_nm,
            speed_limit: cfg.yaw_speed_limit_rad_s,
            static_friction: cfg.yaw_static_friction_nm,
            deadband: 0.0,
            limits: None,
            implicit_coupling: false,
            solref_imp: None,
        });
    }

    // 5. Passive one-way ratchet wheels (2 per plate)
    let mut contact_sites = Vec::new();
    if cfg.ground_contact {
        contact_sites.reserve(2 * n_plates);
        for p in 0..n_plates {
            for s in 0..2 {
                let cid = contact_sites.len();
                contact_sites.push(ContactSiteSpec {
                    id: cid,
                    name: format!("plate_{p}_wheel_{s}"),
                    attachment: ContactAttachment::RigidBody { body_id: p },
                    local_pos: Vec3::from(wheel_mounts[p % 2][s]),
                    local_frame: Quat::IDENTITY,
                    radius: WHEEL_RADIUS_M,
                    patch_radius: cfg.wheel_patch_radius_m,
                    mode: ContactMode::RatchetWheel {
                        wheel_mass: cfg.wheel_mass_kg,
                        rolling_resistance: cfg.wheel_rolling,
                        backlash_rad: cfg.ratchet_backlash_rad,
                        pivot_mu: cfg.wheel_pivot_mu,
                        mu_fwd: cfg.wheel_mu,
                        mu_lat_ratio: cfg.wheel_mu_lat_ratio,
                        static_mu_ratio: cfg.wheel_static_mu_ratio,
                        axle_axis_local: Vec3::EY,
                    },
                    material_profile_id: cfg.material_profile_id,
                    solref_imp: None,
                });
            }
        }
    }

    RigidFlexGraph {
        name: format!("serpworm_N{n_seg}"),
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams,
        tendons,
        joints,
        contact_sites,
        symmetry: make_z2_bilateral_symmetry(n_seg),
        solver_hints: SolverHints {
            preferred_dt: cfg.preferred_dt,
            rayleigh_alpha: cfg.rayleigh_alpha,
            rayleigh_beta: cfg.rayleigh_beta,
            force_route: None,
        },
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    }
}
