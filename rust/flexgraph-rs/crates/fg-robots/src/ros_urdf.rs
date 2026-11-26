//! Standard ROS URDF importer backed by `urdf-rs` (`0.10`, Apache-2.0).
//!
//! Converts unmodified standard ROS URDF files (`<link>`, `<inertial>`, `<joint>` across
//! revolute/continuous/prismatic/fixed/floating/planar/spherical, `<limit>`, `<dynamics>`,
//! `<mimic>`, and `<collision>` box/cylinder/capsule/sphere/mesh) into a validated
//! [`RigidFlexGraph`] with `mesh_colliders` populated for Parry3D narrowphase.

use crate::RobotLoadError;
use fg_ir::{
    ColliderGeometry, ContactAttachment, ContactMode, ContactSiteSpec, JointEdgeSpec, JointType,
    MeshColliderSpec, RigidBodySpec, RigidFlexGraph, SolverHints, SymmetryGroupSpec,
};
use fg_math::{Mat3, Quat, Se3, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

fn quat_from_rpy(rpy: [f64; 3]) -> Quat {
    let qx = Quat::from_axis_angle(Vec3::EX, rpy[0]);
    let qy = Quat::from_axis_angle(Vec3::EY, rpy[1]);
    let qz = Quat::from_axis_angle(Vec3::EZ, rpy[2]);
    qz.mul(qy).mul(qx).normalized()
}

fn pose_to_se3(p: &urdf_rs::Pose) -> Se3 {
    Se3::new(Vec3::from(p.xyz.0), quat_from_rpy(p.rpy.0))
}

fn convert_urdf_geometry(geom: &urdf_rs::Geometry) -> (ColliderGeometry, f64) {
    match geom {
        urdf_rs::Geometry::Box { size } => {
            let hx = (0.5 * size.0[0]).max(1e-4);
            let hy = (0.5 * size.0[1]).max(1e-4);
            let hz = (0.5 * size.0[2]).max(1e-4);
            let r_eq = hx.min(hy).min(hz);
            (
                ColliderGeometry::Box {
                    half_extents: Vec3::new(hx, hy, hz),
                },
                r_eq,
            )
        }
        urdf_rs::Geometry::Cylinder { radius, length } => {
            let r = radius.max(1e-4);
            let hh = (0.5 * length).max(1e-4);
            (
                ColliderGeometry::Cylinder {
                    radius: r,
                    half_height: hh,
                },
                r,
            )
        }
        urdf_rs::Geometry::Capsule { radius, length } => {
            let r = radius.max(1e-4);
            let hh = (0.5 * length).max(1e-4);
            (
                ColliderGeometry::Capsule {
                    radius: r,
                    half_height: hh,
                },
                r,
            )
        }
        urdf_rs::Geometry::Sphere { radius } => {
            let r = radius.max(1e-4);
            (ColliderGeometry::Sphere { radius: r }, r)
        }
        urdf_rs::Geometry::Mesh { filename, scale } => {
            let s = scale.map(|v| Vec3::from(v.0)).unwrap_or(Vec3::new(1.0, 1.0, 1.0));
            let safe_scale = Vec3::new(
                if s.x() > 0.0 { s.x() } else { 1.0 },
                if s.y() > 0.0 { s.y() } else { 1.0 },
                if s.z() > 0.0 { s.z() } else { 1.0 },
            );
            (
                ColliderGeometry::ExternalMesh {
                    uri: filename.clone(),
                    scale: safe_scale,
                },
                0.02,
            )
        }
    }
}

/// Convert a parsed [`urdf_rs::Robot`] into a validated [`RigidFlexGraph`].
pub fn graph_from_urdf_robot(robot: &urdf_rs::Robot) -> Result<RigidFlexGraph, RobotLoadError> {
    if robot.links.is_empty() {
        return Err(RobotLoadError::RosUrdf(
            "ROS URDF contains no <link> elements".into(),
        ));
    }

    let mut link_to_id: HashMap<&str, usize> = HashMap::with_capacity(robot.links.len());
    for (i, link) in robot.links.iter().enumerate() {
        link_to_id.insert(link.name.as_str(), i);
    }

    // Identify child links and build parent->child adjacency for kinematic rest-pose propagation.
    let mut child_links: HashSet<usize> = HashSet::new();
    let mut children_of: Vec<Vec<(usize, Se3)>> = vec![Vec::new(); robot.links.len()];
    let mut floating_children: HashSet<usize> = HashSet::new();

    for j in &robot.joints {
        let &pid = link_to_id.get(j.parent.link.as_str()).ok_or_else(|| {
            RobotLoadError::RosUrdf(format!(
                "joint '{}' references unknown parent link '{}'",
                j.name, j.parent.link
            ))
        })?;
        let &cid = link_to_id.get(j.child.link.as_str()).ok_or_else(|| {
            RobotLoadError::RosUrdf(format!(
                "joint '{}' references unknown child link '{}'",
                j.name, j.child.link
            ))
        })?;
        child_links.insert(cid);
        if j.joint_type == urdf_rs::JointType::Floating {
            floating_children.insert(cid);
        }
        children_of[pid].push((cid, pose_to_se3(&j.origin)));
    }

    // Propagate world rest poses topologically from root link(s).
    let mut world_poses = vec![Se3::IDENTITY; robot.links.len()];
    let mut visited = vec![false; robot.links.len()];
    let mut queue = VecDeque::new();
    for i in 0..robot.links.len() {
        if !child_links.contains(&i) {
            visited[i] = true;
            queue.push_back(i);
        }
    }
    while let Some(u) = queue.pop_front() {
        let p_u = world_poses[u];
        for &(v, rel) in &children_of[u] {
            if !visited[v] {
                visited[v] = true;
                world_poses[v] = p_u.compose(rel);
                queue.push_back(v);
            }
        }
    }

    // 1. Build RigidBodySpecs, MeshColliderSpecs, and rigid ContactSiteSpecs.
    let mut bodies = Vec::with_capacity(robot.links.len());
    let mut mesh_colliders = Vec::new();
    let mut contact_sites = Vec::new();

    for (id, link) in robot.links.iter().enumerate() {
        let is_root = !child_links.contains(&id);
        let raw_mass = link.inertial.mass.value;
        let is_fixed = is_root
            && (link.name.eq_ignore_ascii_case("world")
                || raw_mass <= 0.0
                || floating_children.is_empty());

        let mass = if is_fixed {
            if raw_mass > 0.0 {
                raw_mass
            } else {
                1.0
            }
        } else if raw_mass > 0.0 {
            raw_mass
        } else {
            0.05
        };

        let iner = &link.inertial.inertia;
        let mut inertia = Mat3([
            [iner.ixx, iner.ixy, iner.ixz],
            [iner.ixy, iner.iyy, iner.iyz],
            [iner.ixz, iner.iyz, iner.izz],
        ]);
        let diag_ok =
            iner.ixx > 1e-12 && iner.iyy > 1e-12 && iner.izz > 1e-12;

        let mut max_radius = 0.02_f64;
        for (ci, col) in link.collision.iter().enumerate() {
            let (geometry, r_eq) = convert_urdf_geometry(&col.geometry);
            max_radius = max_radius.max(r_eq);
            let local_pose = pose_to_se3(&col.origin);
            let col_name = col
                .name
                .clone()
                .unwrap_or_else(|| format!("{}_col_{ci}", link.name));
            let col_id = mesh_colliders.len();
            mesh_colliders.push(MeshColliderSpec {
                id: col_id,
                name: col_name.clone(),
                body_id: Some(id),
                local_pose,
                geometry: geometry.clone(),
                material_profile_id: 0,
                solref_imp: None,
                margin: 0.0,
                collision_group: u32::MAX,
                collision_mask: u32::MAX,
            });

            if !is_fixed {
                let site_radius = match geometry {
                    ColliderGeometry::Sphere { radius } => radius,
                    ColliderGeometry::Capsule { radius, .. }
                    | ColliderGeometry::Cylinder { radius, .. } => radius,
                    ColliderGeometry::Box { half_extents } => {
                        half_extents.x().min(half_extents.y()).min(half_extents.z())
                    }
                    _ => r_eq,
                };
                let site_id = contact_sites.len();
                contact_sites.push(ContactSiteSpec {
                    id: site_id,
                    name: format!("{col_name}_site"),
                    attachment: ContactAttachment::RigidBody { body_id: id },
                    local_pos: local_pose.pos,
                    local_frame: local_pose.rot,
                    radius: site_radius,
                    patch_radius: 0.25 * site_radius,
                    mode: ContactMode::IsotropicPad {
                        mu: 0.80,
                        pivot_mu: 0.45,
                        static_mu_ratio: 1.05,
                        sdf_only: false,
                    },
                    material_profile_id: 0,
                    solref_imp: None,
                });
            }
        }

        if !diag_ok {
            let i_iso = (0.4 * mass * max_radius * max_radius).max(1e-5);
            inertia = Mat3::diag([i_iso, i_iso, i_iso]);
        }

        let mut tags = vec!["ros_urdf".to_string()];
        if is_fixed {
            tags.push("fixed_base".to_string());
        }

        bodies.push(RigidBodySpec {
            id,
            name: link.name.clone(),
            is_fixed,
            mass,
            inertia,
            com_local: Vec3::from(link.inertial.origin.xyz.0),
            rest_pose: world_poses[id],
            collision_radius: max_radius,
            rayleigh_alpha: 0.05,
            tags,
        });
    }

    // 2. First assign actuator IDs to non-mimic actuated joints so <mimic> can reference masters.
    let mut joint_name_to_act: HashMap<&str, usize> = HashMap::new();
    let mut next_act = 0usize;
    for j in &robot.joints {
        let is_actuated = matches!(
            j.joint_type,
            urdf_rs::JointType::Revolute
                | urdf_rs::JointType::Continuous
                | urdf_rs::JointType::Prismatic
                | urdf_rs::JointType::Planar
        );
        if is_actuated && j.mimic.is_none() {
            joint_name_to_act.insert(j.name.as_str(), next_act);
            next_act += 1;
        }
    }

    // 3. Convert URDF joints to JointEdgeSpec.
    let mut joints = Vec::new();
    for j in &robot.joints {
        if j.joint_type == urdf_rs::JointType::Floating {
            // 6-DoF floating joint leaves child completely unconstrained relative to parent.
            continue;
        }
        let body_a = link_to_id[j.parent.link.as_str()];
        let body_b = link_to_id[j.child.link.as_str()];
        let origin = pose_to_se3(&j.origin);
        let axis_raw = Vec3::from(j.axis.xyz.0);
        // URDF <axis> is defined in the joint frame (rotated by `origin.rpy` relative to parent link).
        let axis_local_a = origin.rot.rotate(axis_raw).normalized_or(Vec3::EX);

        let (damping, static_friction) = j
            .dynamics
            .as_ref()
            .map(|d| (d.damping.max(0.0), d.friction.max(0.0)))
            .unwrap_or((0.15, 0.0));

        let torque_limit = if j.limit.effort.is_finite() && j.limit.effort > 0.0 {
            j.limit.effort
        } else {
            5.0
        };
        let speed_limit = if j.limit.velocity.is_finite() && j.limit.velocity > 0.0 {
            j.limit.velocity
        } else {
            6.0
        };
        let has_bounds = j.limit.lower.is_finite()
            && j.limit.upper.is_finite()
            && j.limit.upper > j.limit.lower;

        let (joint_type, limits, default_kp) = match j.joint_type {
            urdf_rs::JointType::Revolute => (
                JointType::Revolute {
                    axis_local_a,
                    anchor_a: origin.pos,
                    anchor_b: Vec3::ZERO,
                    pos_stiffness: 12_000.0,
                    pos_damping: 15.0,
                    align_stiffness: 30.0,
                    align_damping: 0.2,
                },
                has_bounds.then_some([j.limit.lower, j.limit.upper]),
                10.0,
            ),
            urdf_rs::JointType::Continuous => (
                JointType::Revolute {
                    axis_local_a,
                    anchor_a: origin.pos,
                    anchor_b: Vec3::ZERO,
                    pos_stiffness: 12_000.0,
                    pos_damping: 15.0,
                    align_stiffness: 30.0,
                    align_damping: 0.2,
                },
                None,
                10.0,
            ),
            urdf_rs::JointType::Prismatic => (
                JointType::Prismatic {
                    axis_local_a,
                    anchor_a: origin.pos,
                    anchor_b: Vec3::ZERO,
                    perp_stiffness: 12_000.0,
                    perp_damping: 15.0,
                },
                has_bounds.then_some([j.limit.lower, j.limit.upper]),
                80.0,
            ),
            urdf_rs::JointType::Planar => {
                // Planar slide orthogonal to normal `axis_local_a`
                let helper = if axis_local_a.z().abs() < 0.9 {
                    Vec3::EZ
                } else {
                    Vec3::EX
                };
                let slide_axis = axis_local_a.cross(helper).normalized();
                (
                    JointType::Prismatic {
                        axis_local_a: slide_axis,
                        anchor_a: origin.pos,
                        anchor_b: Vec3::ZERO,
                        perp_stiffness: 12_000.0,
                        perp_damping: 15.0,
                    },
                    has_bounds.then_some([j.limit.lower, j.limit.upper]),
                    60.0,
                )
            }
            urdf_rs::JointType::Spherical => (
                JointType::Spherical {
                    anchor_a: origin.pos,
                    anchor_b: Vec3::ZERO,
                    pos_stiffness: 12_000.0,
                    pos_damping: 15.0,
                },
                None,
                0.0,
            ),
            urdf_rs::JointType::Fixed => (
                JointType::Weld {
                    rel_pose: origin,
                    pos_stiffness: 20_000.0,
                    rot_stiffness: 200.0,
                },
                None,
                0.0,
            ),
            urdf_rs::JointType::Floating => unreachable!(),
        };

        let actuator_id = if let Some(ref mimic) = j.mimic {
            let mult = mimic.multiplier.unwrap_or(1.0);
            let off = mimic.offset.unwrap_or(0.0);
            bodies[body_b]
                .tags
                .push(format!("mimic:{}:{mult}:{off}", mimic.joint));
            joint_name_to_act.get(mimic.joint.as_str()).copied()
        } else {
            joint_name_to_act.get(j.name.as_str()).copied()
        };

        let jid = joints.len();
        joints.push(JointEdgeSpec {
            id: jid,
            name: j.name.clone(),
            body_a,
            body_b,
            joint_type,
            actuator_id,
            kp: if actuator_id.is_some() { default_kp } else { 0.0 },
            kd: damping,
            torque_limit: if actuator_id.is_some() {
                torque_limit
            } else {
                0.0
            },
            speed_limit: if actuator_id.is_some() {
                speed_limit
            } else {
                0.0
            },
            static_friction,
            deadband: 0.0,
            limits,
            implicit_coupling: true,
            solref_imp: None,
        });
    }

    let graph = RigidFlexGraph {
        name: if robot.name.is_empty() {
            "ros_urdf_robot".to_string()
        } else {
            robot.name.clone()
        },
        gravity: Vec3::new(0.0, 0.0, -9.81),
        bodies,
        beams: Vec::new(),
        tendons: Vec::new(),
        joints,
        contact_sites,
        symmetry: SymmetryGroupSpec::trivial(),
        solver_hints: SolverHints {
            preferred_dt: 0.001,
            rayleigh_alpha: 0.05,
            rayleigh_beta: 0.002,
            force_route: None,
        },
        diff_params: Vec::new(),
        mesh_colliders,
    };
    graph.validate()?;
    Ok(graph)
}

/// Parse an unmodified standard ROS URDF XML string via `urdf-rs` into a validated [`RigidFlexGraph`].
pub fn load_graph_from_ros_urdf_str(xml: &str) -> Result<RigidFlexGraph, RobotLoadError> {
    let robot = urdf_rs::read_from_string(xml)
        .map_err(|e| RobotLoadError::RosUrdf(e.to_string()))?;
    graph_from_urdf_robot(&robot)
}

/// Load an unmodified standard ROS URDF file via `urdf-rs` into a validated [`RigidFlexGraph`].
pub fn load_graph_from_ros_urdf_file(
    path: impl AsRef<Path>,
) -> Result<RigidFlexGraph, RobotLoadError> {
    let xml = std::fs::read_to_string(path.as_ref())?;
    load_graph_from_ros_urdf_str(&xml)
}
