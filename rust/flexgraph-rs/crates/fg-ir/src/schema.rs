//! Declarative `RigidFlexGraph` schema with serde JSON/YAML serialization and validation.

use crate::analysis::{analyze_graph, GraphTopologyAnalysis};
use crate::collider::{ColliderGeometry, MeshColliderSpec};
use crate::diff::{DiffParamKind, DiffParamSpec};
use crate::symmetry::SymmetryGroupSpec;
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, SolRefImp, SpatialInertia, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Validation / compilation error for `RigidFlexGraph`.
#[derive(Debug, Error)]
pub enum IrError {
    #[error("body index {0} out of bounds (num_bodies={1})")]
    BodyOutOfBounds(usize, usize),
    #[error("beam {beam_id} has invalid element count {n_elements} (must be >= 1)")]
    InvalidElementCount { beam_id: usize, n_elements: usize },
    #[error("beam {beam_id} centerline_rest_poses length {actual} does not match n_elements+1 ({expected})")]
    CenterlineLengthMismatch {
        beam_id: usize,
        actual: usize,
        expected: usize,
    },
    #[error("beam {beam_id} element {elem_idx} has non-positive length {length}")]
    NonPositiveElementLength {
        beam_id: usize,
        elem_idx: usize,
        length: f64,
    },
    #[error("rigid body {body_id} ({name}) has non-positive mass {mass}")]
    InvalidBodyMass {
        body_id: usize,
        name: String,
        mass: f64,
    },
    #[error("tendon {tendon_id} must have at least 2 waypoints (got {actual})")]
    TooFewTendonWaypoints { tendon_id: usize, actual: usize },
    #[error("contact site {site_id} references invalid beam {beam_id} or interior node {node_idx}")]
    InvalidBeamContactSite {
        site_id: usize,
        beam_id: usize,
        node_idx: usize,
    },
    #[error("joint {joint_id} connects body {body_id} to itself")]
    SelfJoint { joint_id: usize, body_id: usize },
    #[error("symmetry group error: {0}")]
    InvalidSymmetry(String),
    #[error("invalid differentiable parameter spec: {0}")]
    InvalidDiffParam(String),
    #[error("invalid mesh collider spec: {0}")]
    InvalidCollider(String),
    #[error("invalid SolRefImp parameter: {0}")]
    InvalidSolRefImp(String),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("YAML serialization error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

fn default_gravity() -> Vec3 {
    Vec3::new(0.0, 0.0, -9.81)
}

fn default_rayleigh_alpha() -> f64 {
    0.05
}

fn default_rayleigh_beta() -> f64 {
    0.002
}

fn default_dt() -> f64 {
    0.00025
}

fn default_rest_pose() -> Se3 {
    Se3::IDENTITY
}

/// Rigid body vertex in the rigid-flexible bipartite graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidBodySpec {
    pub id: usize,
    pub name: String,
    /// When `true`, this body is a Dirichlet fixed base (velocity strictly zero).
    #[serde(default)]
    pub is_fixed: bool,
    pub mass: f64,
    /// 3x3 rotational inertia tensor in body frame about COM.
    pub inertia: Mat3,
    #[serde(default)]
    pub com_local: Vec3,
    #[serde(default = "default_rest_pose")]
    pub rest_pose: Se3,
    #[serde(default)]
    pub collision_radius: f64,
    #[serde(default = "default_rayleigh_alpha")]
    pub rayleigh_alpha: f64,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl RigidBodySpec {
    pub fn spatial_inertia(&self) -> SpatialInertia {
        SpatialInertia {
            mass: self.mass,
            com: self.com_local,
            rot_inertia_com: self.inertia,
        }
    }
}

/// Continuum beam constitutive/kinematic formulation selector.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeamFormulation {
    /// Cubic-Hermite corotational 3D Euler-Bernoulli beam (exact `serpworm.GPUCore` parity).
    #[default]
    CorotationalEulerBernoulli,
    /// Cubic-Hermite corotational 3D Timoshenko beam with shear-deformation factor.
    CorotationalTimoshenko,
    /// Discrete Elastic Rod (DER) / Cosserat rod with parallel-transport Bishop frame.
    DiscreteElasticRod,
}

/// Flexible beam edge connecting `body_a` to `body_b` (or `None` for a free cantilever tip).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BeamEdgeSpec {
    pub id: usize,
    pub name: String,
    pub body_a: usize,
    /// Local clamp frame on `body_a` (exact match to `transforms[s, 0, 0]`).
    #[serde(default = "default_rest_pose")]
    pub clamp_a: Se3,
    /// Optional second endpoint rigid body; `None` represents a free cantilever beam tip.
    #[serde(default)]
    pub body_b: Option<usize>,
    /// Local clamp frame on `body_b` when `body_b.is_some()` (exact match to `transforms[s, E-1, 1]`).
    #[serde(default)]
    pub clamp_b: Option<Se3>,
    pub n_elements: usize,
    #[serde(default)]
    pub formulation: BeamFormulation,
    pub section: BeamSection,
    pub material: BeamMaterial,
    #[serde(default)]
    pub thin_shell_compat: bool,
    /// Assembly prestress fraction in `[0, 1]` (`0` = stress-free curved rest pose, `1` = straight ribbon buckled into clamps).
    #[serde(default)]
    pub prestress_frac: f64,
    /// World-space rest poses for the `n_elements + 1` beam nodes along the centerline.
    pub centerline_rest_poses: Vec<Se3>,
    /// Per-element rest lengths (length `n_elements`). If empty, inferred from `centerline_rest_poses`.
    #[serde(default)]
    pub element_lengths: Vec<f64>,
    /// Optional explicit per-element endpoint transforms `[start, end]` for exact legacy parity.
    #[serde(default)]
    pub element_transforms: Option<Vec<[Se3; 2]>>,
    /// If `Some(r)`, interior nodes participate in SDF/ground/pipe contact with collision radius `r`.
    #[serde(default)]
    pub interior_node_contact_radius: Option<f64>,
    /// Enable beam-beam self-contact broadphase/narrowphase (IPC / Py-DiSMech $C^2$).
    #[serde(default)]
    pub enable_self_contact: bool,
}

impl BeamEdgeSpec {
    /// Resolve element lengths, computing Euclidean chord lengths from `centerline_rest_poses`
    /// if `element_lengths` was left empty.
    pub fn resolved_lengths(&self) -> Vec<f64> {
        if !self.element_lengths.is_empty() {
            return self.element_lengths.clone();
        }
        (0..self.n_elements)
            .map(|e| {
                let p0 = self.centerline_rest_poses[e].pos;
                let p1 = self.centerline_rest_poses[e + 1].pos;
                (p1 - p0).norm()
            })
            .collect()
    }

    /// Number of interior (non-rigid-endpoint) nodes on this beam.
    /// - Clamped-clamped (`body_b.is_some()`): `n_elements - 1` interior nodes.
    /// - Free cantilever (`body_b.is_none()`): `n_elements` free nodes (including free tip).
    pub fn num_free_beam_nodes(&self) -> usize {
        if self.body_b.is_some() {
            self.n_elements.saturating_sub(1)
        } else {
            self.n_elements
        }
    }
}

/// Single routing waypoint along a multi-waypoint tendon.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TendonWaypoint {
    pub body_id: usize,
    pub local_pos: Vec3,
    /// Optional local guide wrap angle (rad) for distributed Capstan friction.
    #[serde(default)]
    pub wrap_angle_rad: f64,
}

/// Actuator-to-cable retraction kinematics mode.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum TendonDriveMode {
    /// Eccentric crank arm calibrated to bench data (`tendon_crank_mode = true`).
    EccentricCrank { horn_radius: f64, guide_dist: f64 },
    /// Legacy wrap-pulley formula (`tendon_crank_mode = false`).
    LegacyWrap { horn_radius: f64, guide_dist: f64 },
    /// Constant-radius Capstan winding drum.
    Capstan { drum_radius: f64 },
    /// Direct linear spool/piston retraction (takeup = actuator command in meters).
    Linear,
}

/// Multi-waypoint tendon cable connecting 2 or more rigid bodies (anchors, guide discs, tip).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TendonEdgeSpec {
    pub id: usize,
    pub name: String,
    /// Actuator channel index (multiple tendons may share one servo, e.g. 2 cords/servo on SerpWorm).
    pub actuator_id: usize,
    pub waypoints: Vec<TendonWaypoint>,
    pub drive_mode: TendonDriveMode,
    /// Nominal unstretched tendon path length (m). If `<= 0.0`, computed from rest waypoint positions.
    #[serde(default)]
    pub rest_length: f64,
    #[serde(default)]
    pub slack: f64,
    pub stiffness: f64,
    pub damping: f64,
    #[serde(default)]
    pub capstan_mu_theta: f64,
    #[serde(default)]
    pub stick_slip_enabled: bool,
    pub servo_torque_limit: f64,
    pub servo_speed_limit: f64,
    #[serde(default)]
    pub servo_deadband: f64,
    #[serde(default = "default_tendon_angle_limits")]
    pub angle_limits: [f64; 2],
    /// Optional MuJoCo-compatible soft-constraint impedance & stiffness parameters.
    #[serde(default)]
    pub solref_imp: Option<SolRefImp>,
}

fn default_tendon_angle_limits() -> [f64; 2] {
    [0.0, std::f64::consts::PI]
}

/// Articulated or compliant joint type connecting two rigid bodies.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JointType {
    /// Revolute hinge with compliant position & axis alignment penalty (covers SerpWorm yaw connector & legs).
    Revolute {
        axis_local_a: Vec3,
        anchor_a: Vec3,
        anchor_b: Vec3,
        pos_stiffness: f64,
        pos_damping: f64,
        align_stiffness: f64,
        align_damping: f64,
    },
    /// Prismatic sliding joint along `axis_local_a`.
    Prismatic {
        axis_local_a: Vec3,
        anchor_a: Vec3,
        anchor_b: Vec3,
        perp_stiffness: f64,
        perp_damping: f64,
    },
    /// 3-DoF spherical ball joint.
    Spherical {
        anchor_a: Vec3,
        anchor_b: Vec3,
        pos_stiffness: f64,
        pos_damping: f64,
    },
    /// Rigid weld constraint with finite penalty stiffness.
    Weld {
        rel_pose: Se3,
        pos_stiffness: f64,
        rot_stiffness: f64,
    },
    /// One-sided axial anti-collision stop gap (`k * max(min_gap - gap, 0)` along `axis_local_a`).
    AxialStopGap {
        axis_local_a: Vec3,
        min_gap: f64,
        stiffness: f64,
    },
    /// Two-point elastic cable or strut (for tensegrity structures and parallel elastic bands).
    CableSpring {
        anchor_a: Vec3,
        anchor_b: Vec3,
        rest_length: f64,
        stiffness: f64,
        damping: f64,
        /// `false` = tension-only cable; `true` = bilateral spring/strut.
        compression_allowed: bool,
    },
}

/// Joint or coupling edge between `body_a` and `body_b`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointEdgeSpec {
    pub id: usize,
    pub name: String,
    pub body_a: usize,
    pub body_b: usize,
    pub joint_type: JointType,
    #[serde(default)]
    pub actuator_id: Option<usize>,
    #[serde(default)]
    pub kp: f64,
    #[serde(default)]
    pub kd: f64,
    #[serde(default)]
    pub torque_limit: f64,
    /// `<= 0.0` disables the DC motor speed-torque envelope.
    #[serde(default)]
    pub speed_limit: f64,
    #[serde(default)]
    pub static_friction: f64,
    #[serde(default)]
    pub deadband: f64,
    #[serde(default)]
    pub limits: Option<[f64; 2]>,
    /// When `true`, adds `(body_a, body_b)` to the implicit rigid Schur graph $\mathcal{G}_R$.
    #[serde(default)]
    pub implicit_coupling: bool,
    /// Optional MuJoCo-compatible soft-constraint impedance & stiffness parameters.
    #[serde(default)]
    pub solref_imp: Option<SolRefImp>,
}

/// Attachment point of a contact site.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "attach_type", rename_all = "snake_case")]
pub enum ContactAttachment {
    RigidBody { body_id: usize },
    BeamInteriorNode { beam_id: usize, node_local_idx: usize },
}

/// Unified contact tribology & patch mechanics mode (ARPC).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ContactMode {
    /// Passive one-way ratchet wheel with spin inertia, rolling resistance, pawl backlash & pivoting patch.
    RatchetWheel {
        wheel_mass: f64,
        rolling_resistance: f64,
        backlash_rad: f64,
        pivot_mu: f64,
        mu_fwd: f64,
        mu_lat_ratio: f64,
        static_mu_ratio: f64,
        axle_axis_local: Vec3,
    },
    /// Wheelless anisotropic ventral scale / bristle (`mu_fwd << mu_rev, mu_lat`).
    AnisotropicScale {
        mu_fwd: f64,
        mu_rev: f64,
        mu_lat: f64,
        pivot_mu: f64,
        static_mu_ratio: f64,
        fwd_axis_local: Vec3,
    },
    /// Isotropic rubber/metal footpad or continuum wall-contact site.
    IsotropicPad {
        mu: f64,
        pivot_mu: f64,
        static_mu_ratio: f64,
        sdf_only: bool,
    },
}

/// Contact site attached to a rigid body or a flexible beam interior node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactSiteSpec {
    pub id: usize,
    pub name: String,
    pub attachment: ContactAttachment,
    #[serde(default)]
    pub local_pos: Vec3,
    #[serde(default)]
    pub local_frame: Quat,
    pub radius: f64,
    #[serde(default)]
    pub patch_radius: f64,
    pub mode: ContactMode,
    #[serde(default)]
    pub material_profile_id: usize,
    /// Optional MuJoCo-compatible soft-constraint impedance & stiffness parameters.
    #[serde(default)]
    pub solref_imp: Option<SolRefImp>,
}

/// Solver hints embedded in a robot description.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolverHints {
    #[serde(default = "default_dt")]
    pub preferred_dt: f64,
    #[serde(default = "default_rayleigh_alpha")]
    pub rayleigh_alpha: f64,
    #[serde(default = "default_rayleigh_beta")]
    pub rayleigh_beta: f64,
    /// Optional forced Schur route override for testing or benchmarking.
    #[serde(default)]
    pub force_route: Option<crate::analysis::SchurRoute>,
}

impl Default for SolverHints {
    fn default() -> Self {
        Self {
            preferred_dt: default_dt(),
            rayleigh_alpha: default_rayleigh_alpha(),
            rayleigh_beta: default_rayleigh_beta(),
            force_route: None,
        }
    }
}

/// Complete declarative rigid-flexible continuum robot graph IR:
/// $\mathcal{G} = (\mathcal{V}_{\mathrm{rigid}}, \mathcal{E}_{\mathrm{beam}}, \mathcal{E}_{\mathrm{tendon}}, \mathcal{E}_{\mathrm{joint}}, \mathcal{C}_{\mathrm{site}}, \mathcal{G}_{\mathrm{sym}})$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidFlexGraph {
    pub name: String,
    #[serde(default = "default_gravity")]
    pub gravity: Vec3,
    pub bodies: Vec<RigidBodySpec>,
    #[serde(default)]
    pub beams: Vec<BeamEdgeSpec>,
    #[serde(default)]
    pub tendons: Vec<TendonEdgeSpec>,
    #[serde(default)]
    pub joints: Vec<JointEdgeSpec>,
    #[serde(default)]
    pub contact_sites: Vec<ContactSiteSpec>,
    #[serde(default)]
    pub symmetry: SymmetryGroupSpec,
    #[serde(default)]
    pub solver_hints: SolverHints,
    /// Optional differentiable parameter selections for Dojo/DiffPD-style IFT adjoints.
    #[serde(default)]
    pub diff_params: Vec<DiffParamSpec>,
    /// Optional mesh/primitive collider descriptors for ROS URDF, MJCF, and Parry3D.
    #[serde(default)]
    pub mesh_colliders: Vec<MeshColliderSpec>,
}

impl Default for RigidFlexGraph {
    fn default() -> Self {
        Self {
            name: String::new(),
            gravity: default_gravity(),
            bodies: Vec::new(),
            beams: Vec::new(),
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: SymmetryGroupSpec::default(),
            solver_hints: SolverHints::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        }
    }
}

fn validate_solref_imp(s: &SolRefImp, ctx: &str) -> Result<(), IrError> {
    let vals = [
        s.timeconst,
        s.dampratio,
        s.dmin,
        s.dmax,
        s.width,
        s.midpoint,
        s.power,
    ];
    if !vals.iter().all(|v| v.is_finite()) {
        return Err(IrError::InvalidSolRefImp(format!(
            "{ctx}: non-finite solref/solimp parameter"
        )));
    }
    if s.width < 0.0 {
        return Err(IrError::InvalidSolRefImp(format!(
            "{ctx}: width {} must be >= 0",
            s.width
        )));
    }
    if s.power < 1.0 {
        return Err(IrError::InvalidSolRefImp(format!(
            "{ctx}: power {} must be >= 1.0",
            s.power
        )));
    }
    if !(0.0 < s.midpoint && s.midpoint < 1.0) {
        return Err(IrError::InvalidSolRefImp(format!(
            "{ctx}: midpoint {} must be in (0, 1)",
            s.midpoint
        )));
    }
    Ok(())
}

impl RigidFlexGraph {
    /// Validate topological, physical, and group-theoretic consistency.
    pub fn validate(&self) -> Result<(), IrError> {
        let nb = self.bodies.len();
        for body in &self.bodies {
            if !body.is_fixed && (!body.mass.is_finite() || body.mass <= 0.0) {
                return Err(IrError::InvalidBodyMass {
                    body_id: body.id,
                    name: body.name.clone(),
                    mass: body.mass,
                });
            }
        }

        for beam in &self.beams {
            if beam.body_a >= nb {
                return Err(IrError::BodyOutOfBounds(beam.body_a, nb));
            }
            if let Some(bb) = beam.body_b {
                if bb >= nb {
                    return Err(IrError::BodyOutOfBounds(bb, nb));
                }
            }
            if beam.n_elements < 1 {
                return Err(IrError::InvalidElementCount {
                    beam_id: beam.id,
                    n_elements: beam.n_elements,
                });
            }
            if beam.centerline_rest_poses.len() != beam.n_elements + 1 {
                return Err(IrError::CenterlineLengthMismatch {
                    beam_id: beam.id,
                    actual: beam.centerline_rest_poses.len(),
                    expected: beam.n_elements + 1,
                });
            }
            let lengths = beam.resolved_lengths();
            if lengths.len() != beam.n_elements {
                return Err(IrError::CenterlineLengthMismatch {
                    beam_id: beam.id,
                    actual: lengths.len(),
                    expected: beam.n_elements,
                });
            }
            for (ei, &len) in lengths.iter().enumerate() {
                if !len.is_finite() || len <= 0.0 {
                    return Err(IrError::NonPositiveElementLength {
                        beam_id: beam.id,
                        elem_idx: ei,
                        length: len,
                    });
                }
            }
        }

        for tendon in &self.tendons {
            if tendon.waypoints.len() < 2 {
                return Err(IrError::TooFewTendonWaypoints {
                    tendon_id: tendon.id,
                    actual: tendon.waypoints.len(),
                });
            }
            for wp in &tendon.waypoints {
                if wp.body_id >= nb {
                    return Err(IrError::BodyOutOfBounds(wp.body_id, nb));
                }
            }
            if let Some(ref sri) = tendon.solref_imp {
                validate_solref_imp(sri, &format!("tendon {}", tendon.id))?;
            }
        }

        for joint in &self.joints {
            if joint.body_a >= nb {
                return Err(IrError::BodyOutOfBounds(joint.body_a, nb));
            }
            if joint.body_b >= nb {
                return Err(IrError::BodyOutOfBounds(joint.body_b, nb));
            }
            if joint.body_a == joint.body_b {
                return Err(IrError::SelfJoint {
                    joint_id: joint.id,
                    body_id: joint.body_a,
                });
            }
            if let Some(ref sri) = joint.solref_imp {
                validate_solref_imp(sri, &format!("joint {}", joint.id))?;
            }
        }

        for site in &self.contact_sites {
            match site.attachment {
                ContactAttachment::RigidBody { body_id } => {
                    if body_id >= nb {
                        return Err(IrError::BodyOutOfBounds(body_id, nb));
                    }
                }
                ContactAttachment::BeamInteriorNode {
                    beam_id,
                    node_local_idx,
                } => {
                    if beam_id >= self.beams.len() {
                        return Err(IrError::InvalidBeamContactSite {
                            site_id: site.id,
                            beam_id,
                            node_idx: node_local_idx,
                        });
                    }
                    let beam = &self.beams[beam_id];
                    let max_nodes = beam.num_free_beam_nodes();
                    if node_local_idx >= max_nodes {
                        return Err(IrError::InvalidBeamContactSite {
                            site_id: site.id,
                            beam_id,
                            node_idx: node_local_idx,
                        });
                    }
                }
            }
            if let Some(ref sri) = site.solref_imp {
                validate_solref_imp(sri, &format!("contact_site {}", site.id))?;
            }
        }

        for dp in &self.diff_params {
            if !dp.scale.is_finite() || dp.scale <= 0.0 {
                return Err(IrError::InvalidDiffParam(format!(
                    "{}: scale {} must be finite and > 0",
                    dp.name, dp.scale
                )));
            }
            if let Some([lo, hi]) = dp.bounds {
                if !lo.is_finite() || !hi.is_finite() || lo > hi {
                    return Err(IrError::InvalidDiffParam(format!(
                        "{}: invalid bounds [{lo}, {hi}]",
                        dp.name
                    )));
                }
            }
            let max_id = match dp.kind {
                DiffParamKind::BeamYoungsModulus | DiffParamKind::BeamDampingBeta => {
                    self.beams.len()
                }
                DiffParamKind::ContactMuFwd
                | DiffParamKind::ContactMuRev
                | DiffParamKind::ContactStiffness => self.contact_sites.len(),
                DiffParamKind::TendonStiffness => self.tendons.len(),
                DiffParamKind::BodyMass => nb,
            };
            for &tid in &dp.target_ids {
                if tid >= max_id {
                    return Err(IrError::InvalidDiffParam(format!(
                        "{}: target_id {tid} out of bounds (len={max_id})",
                        dp.name
                    )));
                }
            }
        }

        for col in &self.mesh_colliders {
            if let Some(bid) = col.body_id {
                if bid >= nb {
                    return Err(IrError::BodyOutOfBounds(bid, nb));
                }
            }
            if !col.margin.is_finite() || col.margin < 0.0 {
                return Err(IrError::InvalidCollider(format!(
                    "{}: margin {} must be >= 0",
                    col.name, col.margin
                )));
            }
            if let Some(ref sri) = col.solref_imp {
                validate_solref_imp(sri, &format!("mesh_collider {}", col.id))?;
            }
            match &col.geometry {
                ColliderGeometry::TriMesh { vertices, indices } => {
                    if vertices.is_empty() || indices.is_empty() {
                        return Err(IrError::InvalidCollider(format!(
                            "{}: TriMesh vertices/indices cannot be empty",
                            col.name
                        )));
                    }
                    let nv = vertices.len();
                    for (fi, tri) in indices.iter().enumerate() {
                        if tri.iter().any(|&idx| (idx as usize) >= nv) {
                            return Err(IrError::InvalidCollider(format!(
                                "{}: triangle {fi} index {tri:?} out of bounds (vertices={nv})",
                                col.name
                            )));
                        }
                    }
                }
                ColliderGeometry::ConvexHull { points } => {
                    if points.len() < 4 {
                        return Err(IrError::InvalidCollider(format!(
                            "{}: ConvexHull requires at least 4 points (got {})",
                            col.name,
                            points.len()
                        )));
                    }
                }
                ColliderGeometry::Sphere { radius } => {
                    if !radius.is_finite() || *radius <= 0.0 {
                        return Err(IrError::InvalidCollider(format!(
                            "{}: Sphere radius {radius} must be > 0",
                            col.name
                        )));
                    }
                }
                ColliderGeometry::Box { half_extents } => {
                    if !half_extents.0.iter().all(|v| v.is_finite() && *v > 0.0) {
                        return Err(IrError::InvalidCollider(format!(
                            "{}: Box half_extents {:?} must be > 0",
                            col.name, half_extents
                        )));
                    }
                }
                ColliderGeometry::Cylinder {
                    radius,
                    half_height,
                }
                | ColliderGeometry::Capsule {
                    radius,
                    half_height,
                } => {
                    if !radius.is_finite()
                        || *radius <= 0.0
                        || !half_height.is_finite()
                        || *half_height <= 0.0
                    {
                        return Err(IrError::InvalidCollider(format!(
                            "{}: Cylinder/Capsule radius/half_height must be > 0",
                            col.name
                        )));
                    }
                }
                ColliderGeometry::ExternalMesh { uri, scale } => {
                    if uri.trim().is_empty()
                        || !scale.0.iter().all(|v| v.is_finite() && *v > 0.0)
                    {
                        return Err(IrError::InvalidCollider(format!(
                            "{}: ExternalMesh requires non-empty uri and positive scale",
                            col.name
                        )));
                    }
                }
            }
        }

        self.symmetry.validate()?;
        Ok(())
    }

    /// Analyze graph connectivity, classify optimal Schur route, and build elimination tree
    /// + symbolic sparse block-LDL^T pattern.
    pub fn analyze(&self) -> Result<GraphTopologyAnalysis, IrError> {
        analyze_graph(self)
    }

    /// Serialize to pretty JSON string.
    pub fn to_json_string(&self) -> Result<String, IrError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize from JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, IrError> {
        Ok(serde_json::from_str(s)?)
    }

    /// Serialize to YAML string.
    pub fn to_yaml_string(&self) -> Result<String, IrError> {
        Ok(serde_yaml::to_string(self)?)
    }

    /// Deserialize from YAML string.
    pub fn from_yaml_str(s: &str) -> Result<Self, IrError> {
        Ok(serde_yaml::from_str(s)?)
    }
}
