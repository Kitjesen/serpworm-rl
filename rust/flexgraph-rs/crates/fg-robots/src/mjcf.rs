//! MuJoCo MJCF XML subset importer (`<mujoco>`, `<option>`, `<compiler>`, `<default>`,
//! `<worldbody>` nested `<body>`, `<inertial>`, `<joint>`, `<geom>` incl. `fromto`, `<site>`,
//! `<tendon><spatial>`, `<actuator><motor|position|general>`, `<flexcomp>`, and
//! `solref`/`solimp` mapped directly to [`fg_math::SolRefImp`]).

use crate::loader::{parse_f64_ctx, parse_xml_str, XmlNode};
use crate::RobotLoadError;
use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ColliderGeometry, ContactAttachment, ContactMode,
    ContactSiteSpec, JointEdgeSpec, JointType, MeshColliderSpec, RigidBodySpec, RigidFlexGraph,
    SolRefImp, SolverHints, SymmetryGroupSpec, TendonDriveMode, TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

fn parse_f64_list(s: &str, ctx: &str) -> Result<Vec<f64>, RobotLoadError> {
    let mut out = Vec::new();
    for tok in s
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
    {
        out.push(parse_f64_ctx(tok, ctx)?);
    }
    Ok(out)
}

fn parse_vec3_mjcf(s: &str, ctx: &str) -> Result<Vec3, RobotLoadError> {
    let v = parse_f64_list(s, ctx)?;
    if v.len() != 3 {
        return Err(RobotLoadError::Mjcf(format!(
            "{ctx}: expected 3 floats, got {} ('{s}')",
            v.len()
        )));
    }
    Ok(Vec3::new(v[0], v[1], v[2]))
}

/// Parse a MuJoCo quaternion `[w, x, y, z]` into `fg_math::Quat([x, y, z, w])`.
fn parse_mjcf_quat(s: &str, ctx: &str) -> Result<Quat, RobotLoadError> {
    let v = parse_f64_list(s, ctx)?;
    if v.len() != 4 {
        return Err(RobotLoadError::Mjcf(format!(
            "{ctx}: expected 4 floats [w,x,y,z], got {}",
            v.len()
        )));
    }
    Ok(Quat::new(v[1], v[2], v[3], v[0]).normalized())
}

fn quat_from_euler_xyz(e: Vec3, angle_scale: f64) -> Quat {
    let qx = Quat::from_axis_angle(Vec3::EX, e.x() * angle_scale);
    let qy = Quat::from_axis_angle(Vec3::EY, e.y() * angle_scale);
    let qz = Quat::from_axis_angle(Vec3::EZ, e.z() * angle_scale);
    qx.mul(qy).mul(qz).normalized()
}

/// Orthonormal quaternion whose local `+Z` axis points along `dir`.
fn quat_from_z_axis(dir: Vec3) -> Quat {
    let tz = dir.normalized_or(Vec3::EZ);
    let helper = if tz.x().abs() < 0.9 { Vec3::EX } else { Vec3::EY };
    let ty = tz.cross(helper).normalized();
    let tx = ty.cross(tz);
    Quat::from_mat3(Mat3([
        [tx.x(), ty.x(), tz.x()],
        [tx.y(), ty.y(), tz.y()],
        [tx.z(), ty.z(), tz.z()],
    ]))
}

/// Orthonormal quaternion whose local `+X` axis points along `dir`.
fn quat_from_x_axis(dir: Vec3) -> Quat {
    let tx = dir.normalized_or(Vec3::EX);
    let helper = if tx.z().abs() < 0.9 { Vec3::EZ } else { Vec3::EY };
    let ty = helper.cross(tx).normalized();
    let tz = tx.cross(ty);
    Quat::from_mat3(Mat3([
        [tx.x(), ty.x(), tz.x()],
        [tx.y(), ty.y(), tz.y()],
        [tx.z(), ty.z(), tz.z()],
    ]))
}

fn parse_solref_imp_attrs(
    solref_str: Option<&str>,
    solimp_str: Option<&str>,
    fallback: Option<SolRefImp>,
) -> Result<Option<SolRefImp>, RobotLoadError> {
    if solref_str.is_none() && solimp_str.is_none() {
        return Ok(fallback);
    }
    let mut sri = fallback.unwrap_or_default();
    if let Some(sr) = solref_str {
        let v = parse_f64_list(sr, "solref")?;
        if v.len() >= 2 {
            sri.timeconst = v[0];
            sri.dampratio = v[1];
        }
    }
    if let Some(si) = solimp_str {
        let v = parse_f64_list(si, "solimp")?;
        if !v.is_empty() {
            sri.dmin = v[0];
        }
        if v.len() >= 2 {
            sri.dmax = v[1];
        }
        if v.len() >= 3 {
            sri.width = v[2];
        }
        if v.len() >= 4 {
            sri.midpoint = v[3];
        }
        if v.len() >= 5 {
            sri.power = v[4];
        }
    }
    Ok(Some(sri))
}

#[derive(Clone, Debug, Default)]
struct MjcfDefaults {
    geom_density: f64,
    geom_friction: [f64; 3],
    geom_solref_imp: Option<SolRefImp>,
    joint_damping: f64,
    joint_stiffness: f64,
    joint_solref_imp: Option<SolRefImp>,
    joint_limited: bool,
    tendon_stiffness: f64,
    tendon_damping: f64,
    tendon_solref_imp: Option<SolRefImp>,
    motor_gear: f64,
    motor_kp: f64,
}

impl MjcfDefaults {
    fn base() -> Self {
        Self {
            geom_density: 1000.0,
            geom_friction: [1.0, 0.005, 0.0001],
            geom_solref_imp: None,
            joint_damping: 0.1,
            joint_stiffness: 0.0,
            joint_solref_imp: None,
            joint_limited: false,
            tendon_stiffness: 1500.0,
            tendon_damping: 1.0,
            tendon_solref_imp: None,
            motor_gear: 1.0,
            motor_kp: 12.0,
        }
    }

    fn apply_node(&mut self, def_node: &XmlNode) -> Result<(), RobotLoadError> {
        if let Some(g) = def_node.child("geom") {
            if let Some(d) = g.attr("density") {
                self.geom_density = parse_f64_ctx(d, "default.geom.density")?;
            }
            if let Some(f) = g.attr("friction") {
                let v = parse_f64_list(f, "default.geom.friction")?;
                let n = v.len().min(3);
                self.geom_friction[..n].copy_from_slice(&v[..n]);
            }
            self.geom_solref_imp =
                parse_solref_imp_attrs(g.attr("solref"), g.attr("solimp"), self.geom_solref_imp)?;
        }
        if let Some(j) = def_node.child("joint") {
            if let Some(d) = j.attr("damping") {
                self.joint_damping = parse_f64_ctx(d, "default.joint.damping")?;
            }
            if let Some(k) = j.attr("stiffness") {
                self.joint_stiffness = parse_f64_ctx(k, "default.joint.stiffness")?;
            }
            if let Some(lim) = j.attr("limited") {
                self.joint_limited = lim == "true" || lim == "1";
            }
            self.joint_solref_imp = parse_solref_imp_attrs(
                j.attr("solreflimit").or_else(|| j.attr("solref")),
                j.attr("solimplimit").or_else(|| j.attr("solimp")),
                self.joint_solref_imp,
            )?;
        }
        if let Some(t) = def_node.child("tendon") {
            if let Some(k) = t.attr("stiffness") {
                self.tendon_stiffness = parse_f64_ctx(k, "default.tendon.stiffness")?;
            }
            if let Some(d) = t.attr("damping") {
                self.tendon_damping = parse_f64_ctx(d, "default.tendon.damping")?;
            }
            self.tendon_solref_imp = parse_solref_imp_attrs(
                t.attr("solreflimit").or_else(|| t.attr("solref")),
                t.attr("solimplimit").or_else(|| t.attr("solimp")),
                self.tendon_solref_imp,
            )?;
        }
        if let Some(m) = def_node.child("motor").or_else(|| def_node.child("general")) {
            if let Some(gear) = m.attr("gear") {
                let v = parse_f64_list(gear, "default.motor.gear")?;
                if !v.is_empty() {
                    self.motor_gear = v[0].abs().max(1e-3);
                }
            }
        }
        if let Some(p) = def_node.child("position") {
            if let Some(kp) = p.attr("kp") {
                self.motor_kp = parse_f64_ctx(kp, "default.position.kp")?;
            }
        }
        Ok(())
    }
}

/// Approximate `(mass, diag_inertia, radius_equiv)` of a MuJoCo geom primitive for automatic
/// inertia inference when `<body>` omits `<inertial>`.
fn geom_mass_inertia(geom: &ColliderGeometry, density: f64, explicit_mass: Option<f64>) -> (f64, Vec3, f64) {
    match *geom {
        ColliderGeometry::Sphere { radius } => {
            let vol = (4.0 / 3.0) * PI * radius.powi(3);
            let m = explicit_mass.unwrap_or_else(|| (density * vol).max(1e-3));
            let i = (0.4 * m * radius * radius).max(1e-6);
            (m, Vec3::new(i, i, i), radius)
        }
        ColliderGeometry::Box { half_extents } => {
            let sx = 2.0 * half_extents.x();
            let sy = 2.0 * half_extents.y();
            let sz = 2.0 * half_extents.z();
            let m = explicit_mass.unwrap_or_else(|| (density * sx * sy * sz).max(1e-3));
            let ix = (m / 12.0 * (sy * sy + sz * sz)).max(1e-6);
            let iy = (m / 12.0 * (sx * sx + sz * sz)).max(1e-6);
            let iz = (m / 12.0 * (sx * sx + sy * sy)).max(1e-6);
            let r = half_extents.x().min(half_extents.y()).min(half_extents.z());
            (m, Vec3::new(ix, iy, iz), r)
        }
        ColliderGeometry::Cylinder {
            radius,
            half_height,
        } => {
            let h = 2.0 * half_height;
            let vol = PI * radius * radius * h;
            let m = explicit_mass.unwrap_or_else(|| (density * vol).max(1e-3));
            let iz = (0.5 * m * radius * radius).max(1e-6);
            let ixy = (m / 12.0 * (3.0 * radius * radius + h * h)).max(1e-6);
            (m, Vec3::new(ixy, ixy, iz), radius)
        }
        ColliderGeometry::Capsule {
            radius,
            half_height,
        } => {
            let h = 2.0 * half_height;
            let vol_cyl = PI * radius * radius * h;
            let vol_caps = (4.0 / 3.0) * PI * radius.powi(3);
            let vol_tot = vol_cyl + vol_caps;
            let m = explicit_mass.unwrap_or_else(|| (density * vol_tot).max(1e-3));
            let m_cyl = m * (vol_cyl / vol_tot);
            let m_caps = m * (vol_caps / vol_tot);
            let iz = (0.5 * m_cyl * radius * radius + 0.4 * m_caps * radius * radius).max(1e-6);
            let ixy = (m_cyl * (0.25 * radius * radius + h * h / 12.0)
                + m_caps * (0.4 * radius * radius + 0.375 * radius * h + 0.25 * h * h))
                .max(1e-6);
            (m, Vec3::new(ixy, ixy, iz), radius)
        }
        _ => {
            let m = explicit_mass.unwrap_or(0.05);
            (m, Vec3::new(1e-4, 1e-4, 1e-4), 0.02)
        }
    }
}

struct ParsedMjcfGeom {
    geom: ColliderGeometry,
    local_pose: Se3,
    mass: f64,
    diag_inertia: Vec3,
    equiv_radius: f64,
    mu_slide: f64,
    solref_imp: Option<SolRefImp>,
}

struct MjcfBuilder {
    angle_scale: f64,
    defaults: MjcfDefaults,
    bodies: Vec<RigidBodySpec>,
    beams: Vec<BeamEdgeSpec>,
    tendons: Vec<TendonEdgeSpec>,
    joints: Vec<JointEdgeSpec>,
    contact_sites: Vec<ContactSiteSpec>,
    mesh_colliders: Vec<MeshColliderSpec>,
    body_name_to_id: HashMap<String, usize>,
    joint_name_to_idx: HashMap<String, usize>,
    tendon_name_to_idx: HashMap<String, usize>,
    site_map: HashMap<String, (usize, Vec3)>,
    world_anchor_id: Option<usize>,
}

impl MjcfBuilder {
    fn ensure_world_anchor(&mut self) -> usize {
        if let Some(id) = self.world_anchor_id {
            return id;
        }
        let id = self.bodies.len();
        self.bodies.push(RigidBodySpec {
            id,
            name: "worldbody".to_string(),
            is_fixed: true,
            mass: 1.0,
            inertia: Mat3::IDENTITY,
            com_local: Vec3::ZERO,
            rest_pose: Se3::IDENTITY,
            collision_radius: 0.0,
            rayleigh_alpha: 0.05,
            tags: vec!["mjcf_world".to_string()],
        });
        self.body_name_to_id.insert("worldbody".to_string(), id);
        self.world_anchor_id = Some(id);
        id
    }

    fn parse_geom_node(
        &mut self,
        gnode: &XmlNode,
        owner_body: Option<usize>,
    ) -> Result<Option<ParsedMjcfGeom>, RobotLoadError> {
        let gtype = gnode.attr("type").unwrap_or("sphere");
        if gtype == "plane" {
            // Infinite ground planes belong in WorldSdf rather than rigid MeshColliders.
            return Ok(None);
        }

        let size_vals = match gnode.attr("size") {
            Some(s) => parse_f64_list(s, "geom.size")?,
            None => vec![0.02],
        };

        let (local_pose, geom) = if let Some(ft_str) = gnode.attr("fromto") {
            let ft = parse_f64_list(ft_str, "geom.fromto")?;
            if ft.len() != 6 {
                return Err(RobotLoadError::Mjcf(format!(
                    "geom.fromto requires 6 floats, got {}",
                    ft.len()
                )));
            }
            let p1 = Vec3::new(ft[0], ft[1], ft[2]);
            let p2 = Vec3::new(ft[3], ft[4], ft[5]);
            let center = 0.5 * (p1 + p2);
            let diff = p2 - p1;
            let half_h = (0.5 * diff.norm()).max(1e-4);
            let rot = quat_from_z_axis(diff);
            let r = size_vals.first().copied().unwrap_or(0.02).max(1e-4);
            let g = match gtype {
                "cylinder" => ColliderGeometry::Cylinder {
                    radius: r,
                    half_height: half_h,
                },
                "box" => ColliderGeometry::Box {
                    half_extents: Vec3::new(r, r, half_h),
                },
                _ => ColliderGeometry::Capsule {
                    radius: r,
                    half_height: half_h,
                },
            };
            (Se3::new(center, rot), g)
        } else {
            let pos = match gnode.attr("pos") {
                Some(s) => parse_vec3_mjcf(s, "geom.pos")?,
                None => Vec3::ZERO,
            };
            let rot = if let Some(q) = gnode.attr("quat") {
                parse_mjcf_quat(q, "geom.quat")?
            } else if let Some(e) = gnode.attr("euler") {
                quat_from_euler_xyz(parse_vec3_mjcf(e, "geom.euler")?, self.angle_scale)
            } else {
                Quat::IDENTITY
            };
            let g = match gtype {
                "sphere" => ColliderGeometry::Sphere {
                    radius: size_vals.first().copied().unwrap_or(0.02).max(1e-4),
                },
                "box" => {
                    let hx = size_vals.first().copied().unwrap_or(0.02).max(1e-4);
                    let hy = size_vals.get(1).copied().unwrap_or(hx).max(1e-4);
                    let hz = size_vals.get(2).copied().unwrap_or(hy).max(1e-4);
                    ColliderGeometry::Box {
                        half_extents: Vec3::new(hx, hy, hz),
                    }
                }
                "cylinder" => {
                    let r = size_vals.first().copied().unwrap_or(0.02).max(1e-4);
                    let hh = size_vals.get(1).copied().unwrap_or(r).max(1e-4);
                    ColliderGeometry::Cylinder {
                        radius: r,
                        half_height: hh,
                    }
                }
                "capsule" => {
                    let r = size_vals.first().copied().unwrap_or(0.02).max(1e-4);
                    let hh = size_vals.get(1).copied().unwrap_or(r).max(1e-4);
                    ColliderGeometry::Capsule {
                        radius: r,
                        half_height: hh,
                    }
                }
                "ellipsoid" => {
                    let rx = size_vals.first().copied().unwrap_or(0.02).max(1e-4);
                    let ry = size_vals.get(1).copied().unwrap_or(rx).max(1e-4);
                    let rz = size_vals.get(2).copied().unwrap_or(ry).max(1e-4);
                    ColliderGeometry::Sphere {
                        radius: (rx * ry * rz).cbrt(),
                    }
                }
                "mesh" => ColliderGeometry::ExternalMesh {
                    uri: gnode.attr("mesh").unwrap_or("mesh").to_string(),
                    scale: Vec3::new(1.0, 1.0, 1.0),
                },
                _ => ColliderGeometry::Sphere {
                    radius: size_vals.first().copied().unwrap_or(0.02).max(1e-4),
                },
            };
            (Se3::new(pos, rot), g)
        };

        let density = gnode
            .attr("density")
            .map(|s| parse_f64_ctx(s, "geom.density"))
            .transpose()?
            .unwrap_or(self.defaults.geom_density);
        let explicit_mass = gnode
            .attr("mass")
            .map(|s| parse_f64_ctx(s, "geom.mass"))
            .transpose()?;
        let (m, diag_i, r_eq) = geom_mass_inertia(&geom, density, explicit_mass);

        let mut friction = self.defaults.geom_friction;
        if let Some(f_str) = gnode.attr("friction") {
            let fv = parse_f64_list(f_str, "geom.friction")?;
            let n = fv.len().min(3);
            friction[..n].copy_from_slice(&fv[..n]);
        }
        let sri = parse_solref_imp_attrs(
            gnode.attr("solref"),
            gnode.attr("solimp"),
            self.defaults.geom_solref_imp,
        )?;
        let margin = gnode
            .attr("margin")
            .map(|s| parse_f64_ctx(s, "geom.margin"))
            .transpose()?
            .unwrap_or(0.0)
            .max(0.0);

        let col_id = self.mesh_colliders.len();
        let cname = gnode
            .attr("name")
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("mjcf_geom_{col_id}"));
        self.mesh_colliders.push(MeshColliderSpec {
            id: col_id,
            name: cname,
            body_id: owner_body,
            local_pose,
            geometry: geom.clone(),
            material_profile_id: 0,
            solref_imp: sri,
            margin,
            collision_group: u32::MAX,
            collision_mask: u32::MAX,
        });

        Ok(Some(ParsedMjcfGeom {
            geom,
            local_pose,
            mass: m,
            diag_inertia: diag_i,
            equiv_radius: r_eq,
            mu_slide: friction[0],
            solref_imp: sri,
        }))
    }

    fn parse_body_recursive(
        &mut self,
        bnode: &XmlNode,
        parent_body: Option<usize>,
        parent_pose: Se3,
    ) -> Result<(), RobotLoadError> {
        let local_pos = match bnode.attr("pos") {
            Some(s) => parse_vec3_mjcf(s, "body.pos")?,
            None => Vec3::ZERO,
        };
        let local_rot = if let Some(q) = bnode.attr("quat") {
            parse_mjcf_quat(q, "body.quat")?
        } else if let Some(e) = bnode.attr("euler") {
            quat_from_euler_xyz(parse_vec3_mjcf(e, "body.euler")?, self.angle_scale)
        } else {
            Quat::IDENTITY
        };
        let local_pose = Se3::new(local_pos, local_rot);
        let world_pose = parent_pose.compose(local_pose);

        let body_id = self.bodies.len();
        let name = bnode
            .attr("name")
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("body_{body_id}"));

        // Inspect joints declared directly on this body
        let joint_nodes: Vec<&XmlNode> = bnode
            .children_named("joint")
            .chain(bnode.children_named("freejoint"))
            .collect();
        let has_free_joint = joint_nodes
            .iter()
            .any(|j| j.tag == "freejoint" || j.attr("type") == Some("free"));
        let is_fixed = parent_body.is_none() && joint_nodes.is_empty();

        // Push placeholder RigidBodySpec first so geoms/sites/children can reference `body_id`
        self.bodies.push(RigidBodySpec {
            id: body_id,
            name: name.clone(),
            is_fixed,
            mass: 0.1,
            inertia: Mat3::diag([1e-4, 1e-4, 1e-4]),
            com_local: Vec3::ZERO,
            rest_pose: world_pose,
            collision_radius: 0.02,
            rayleigh_alpha: 0.05,
            tags: vec!["mjcf".to_string()],
        });
        self.body_name_to_id.insert(name.clone(), body_id);

        // Parse geoms attached to this body and collect exact `(mass, pose, diag_inertia)` for fallback inertia
        let mut geom_mass_sum = 0.0_f64;
        let mut geom_com_acc = Vec3::ZERO;
        let mut geom_inertias: Vec<(f64, Se3, Vec3)> = Vec::new();
        let mut max_radius = 0.02_f64;

        for gnode in bnode.children_named("geom") {
            if let Some(ParsedMjcfGeom {
                geom,
                local_pose: gpose,
                mass: gm,
                diag_inertia: gi,
                equiv_radius: r_eq,
                mu_slide,
                solref_imp: sri,
            }) = self.parse_geom_node(gnode, Some(body_id))?
            {
                geom_mass_sum += gm;
                geom_com_acc += gm * gpose.pos;
                geom_inertias.push((gm, gpose, gi));
                max_radius = max_radius.max(r_eq);

                if !is_fixed {
                    // Add contact site(s): for capsules add distal endpoint pad too so legged MJCF (Ant) contacts cleanly
                    let add_site = |sites: &mut Vec<ContactSiteSpec>, lp: Vec3, rad: f64, suffix: &str| {
                        let cid = sites.len();
                        sites.push(ContactSiteSpec {
                            id: cid,
                            name: format!("{name}_{suffix}_{cid}"),
                            attachment: ContactAttachment::RigidBody { body_id },
                            local_pos: lp,
                            local_frame: Quat::IDENTITY,
                            radius: rad,
                            patch_radius: 0.25 * rad,
                            mode: ContactMode::IsotropicPad {
                                mu: mu_slide.max(0.05),
                                pivot_mu: 0.45,
                                static_mu_ratio: 1.05,
                                sdf_only: false,
                            },
                            material_profile_id: 0,
                            solref_imp: sri,
                        });
                    };
                    match geom {
                        ColliderGeometry::Capsule {
                            radius,
                            half_height,
                        } => {
                            let end_b = gpose.transform_point(Vec3::new(0.0, 0.0, half_height));
                            add_site(&mut self.contact_sites, end_b, radius, "tip");
                        }
                        _ => {
                            add_site(&mut self.contact_sites, gpose.pos, r_eq, "pad");
                        }
                    }
                }
            }
        }

        // Resolve inertial properties (explicit <inertial> takes priority over geom-derived inertia)
        let (mass, com_local, inertia) = if let Some(inode) = bnode.child("inertial") {
            let m = parse_f64_ctx(inode.attr("mass").unwrap_or("0.1"), "inertial.mass")?;
            let com = match inode.attr("pos") {
                Some(s) => parse_vec3_mjcf(s, "inertial.pos")?,
                None => Vec3::ZERO,
            };
            let mat = if let Some(fi) = inode.attr("fullinertia") {
                let v = parse_f64_list(fi, "inertial.fullinertia")?;
                if v.len() == 6 {
                    Mat3([
                        [v[0], v[3], v[4]],
                        [v[3], v[1], v[5]],
                        [v[4], v[5], v[2]],
                    ])
                } else {
                    Mat3::diag([1e-4, 1e-4, 1e-4])
                }
            } else if let Some(di) = inode.attr("diaginertia") {
                let v = parse_vec3_mjcf(di, "inertial.diaginertia")?;
                Mat3::diag(v.0)
            } else {
                let i0 = (0.4 * m * max_radius * max_radius).max(1e-5);
                Mat3::diag([i0, i0, i0])
            };
            (if m > 0.0 { m } else { 0.1 }, com, mat)
        } else if geom_mass_sum > 0.0 {
            let com = geom_com_acc / geom_mass_sum;
            let mut mat = Mat3::ZERO;
            for (gm, gpose, gi) in &geom_inertias {
                let r = gpose.rot.to_mat3();
                let i_rot = r.mul_mat(Mat3::diag(gi.0)).mul_mat(r.transpose());
                let d = gpose.pos - com;
                let i_shift = (*gm) * (d.norm_sq() * Mat3::IDENTITY - d.outer(d));
                mat = mat + i_rot + i_shift;
            }
            for k in 0..3 {
                mat.0[k][k] = mat.0[k][k].max(1e-5);
            }
            (geom_mass_sum, com, mat.symmetrize())
        } else {
            let m = if is_fixed { 1.0 } else { 0.1 };
            let i0 = (0.4 * m * max_radius * max_radius).max(1e-5);
            (m, Vec3::ZERO, Mat3::diag([i0, i0, i0]))
        };

        self.bodies[body_id].mass = mass;
        self.bodies[body_id].com_local = com_local;
        self.bodies[body_id].inertia = inertia;
        self.bodies[body_id].collision_radius = max_radius;

        // Parse <site> waypoints attached to this body
        for snode in bnode.children_named("site") {
            if let Some(sname) = snode.attr("name") {
                let spos = match snode.attr("pos") {
                    Some(s) => parse_vec3_mjcf(s, "site.pos")?,
                    None => Vec3::ZERO,
                };
                self.site_map.insert(sname.to_string(), (body_id, spos));
            }
        }

        // Parse <joint> elements connecting parent_body -> body_id
        for jnode in joint_nodes {
            let jtype = if jnode.tag == "freejoint" {
                "free"
            } else {
                jnode.attr("type").unwrap_or("hinge")
            };
            if jtype == "free" {
                continue;
            }
            let parent_id = match parent_body {
                Some(pid) => pid,
                None => self.ensure_world_anchor(),
            };
            let j_pos_child = match jnode.attr("pos") {
                Some(s) => parse_vec3_mjcf(s, "joint.pos")?,
                None => Vec3::ZERO,
            };
            let j_axis_child = match jnode.attr("axis") {
                Some(s) => parse_vec3_mjcf(s, "joint.axis")?.normalized_or(Vec3::EZ),
                None => Vec3::EZ,
            };
            // Express anchor and axis in parent body frame
            let p_parent = self.bodies[parent_id].rest_pose;
            let j_world = world_pose.transform_point(j_pos_child);
            let anchor_a = p_parent.rot.inv_rotate(j_world - p_parent.pos);
            let axis_world = world_pose.rot.rotate(j_axis_child);
            let axis_local_a = p_parent.rot.inv_rotate(axis_world).normalized_or(Vec3::EZ);

            let damping = jnode
                .attr("damping")
                .map(|s| parse_f64_ctx(s, "joint.damping"))
                .transpose()?
                .unwrap_or(self.defaults.joint_damping);
            let stiffness = jnode
                .attr("stiffness")
                .map(|s| parse_f64_ctx(s, "joint.stiffness"))
                .transpose()?
                .unwrap_or(self.defaults.joint_stiffness);
            let limited = jnode
                .attr("limited")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(self.defaults.joint_limited || jnode.attr("range").is_some());
            let limits = if limited {
                if let Some(r_str) = jnode.attr("range") {
                    let rv = parse_f64_list(r_str, "joint.range")?;
                    if rv.len() == 2 {
                        let scale = if jtype == "slide" { 1.0 } else { self.angle_scale };
                        Some([rv[0] * scale, rv[1] * scale])
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let sri = parse_solref_imp_attrs(
                jnode.attr("solreflimit").or_else(|| jnode.attr("solref")),
                jnode.attr("solimplimit").or_else(|| jnode.attr("solimp")),
                self.defaults.joint_solref_imp,
            )?;

            let jt = match jtype {
                "hinge" => JointType::Revolute {
                    axis_local_a,
                    anchor_a,
                    anchor_b: j_pos_child,
                    pos_stiffness: 12_000.0,
                    pos_damping: 15.0,
                    align_stiffness: 30.0,
                    align_damping: 0.2,
                },
                "slide" => JointType::Prismatic {
                    axis_local_a,
                    anchor_a,
                    anchor_b: j_pos_child,
                    perp_stiffness: 12_000.0,
                    perp_damping: 15.0,
                },
                "ball" => JointType::Spherical {
                    anchor_a,
                    anchor_b: j_pos_child,
                    pos_stiffness: 12_000.0,
                    pos_damping: 15.0,
                },
                other => {
                    return Err(RobotLoadError::Mjcf(format!(
                        "unsupported MJCF joint type '{other}'"
                    )));
                }
            };

            let jid = self.joints.len();
            let jname = jnode
                .attr("name")
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("{name}_joint_{jid}"));
            self.joint_name_to_idx.insert(jname.clone(), jid);
            self.joints.push(JointEdgeSpec {
                id: jid,
                name: jname,
                body_a: parent_id,
                body_b: body_id,
                joint_type: jt,
                actuator_id: None,
                kp: stiffness,
                kd: damping,
                torque_limit: 0.0,
                speed_limit: 0.0,
                static_friction: 0.0,
                deadband: 0.0,
                limits,
                implicit_coupling: true,
                solref_imp: sri,
            });
        }

        // If a child body has a parent and NO joints at all (rigidly welded sub-body), add a Weld joint
        if let Some(pid) = parent_body {
            if !has_free_joint && bnode.children_named("joint").next().is_none() {
                let jid = self.joints.len();
                self.joints.push(JointEdgeSpec {
                    id: jid,
                    name: format!("{name}_weld"),
                    body_a: pid,
                    body_b: body_id,
                    joint_type: JointType::Weld {
                        rel_pose: local_pose,
                        pos_stiffness: 20_000.0,
                        rot_stiffness: 200.0,
                    },
                    actuator_id: None,
                    kp: 0.0,
                    kd: 0.0,
                    torque_limit: 0.0,
                    speed_limit: 0.0,
                    static_friction: 0.0,
                    deadband: 0.0,
                    limits: None,
                    implicit_coupling: true,
                    solref_imp: None,
                });
            }
        }

        // Recurse into child <body> elements
        for child in bnode.children_named("body") {
            self.parse_body_recursive(child, Some(body_id), world_pose)?;
        }
        Ok(())
    }

    fn parse_flexcomp(&mut self, fc: &XmlNode) -> Result<(), RobotLoadError> {
        let name = fc.attr("name").unwrap_or("flexcomp").to_string();
        let pos = match fc.attr("pos") {
            Some(s) => parse_vec3_mjcf(s, "flexcomp.pos")?,
            None => Vec3::new(0.0, 0.0, 0.10),
        };
        let count_v = match fc.attr("count") {
            Some(s) => parse_f64_list(s, "flexcomp.count")?,
            None => vec![6.0, 1.0, 1.0],
        };
        let n_nodes = (count_v.first().copied().unwrap_or(6.0).round() as usize).max(3);
        let spacing_v = match fc.attr("spacing") {
            Some(s) => parse_f64_list(s, "flexcomp.spacing")?,
            None => vec![0.025, 0.025, 0.025],
        };
        let dx = spacing_v.first().copied().unwrap_or(0.025).max(1e-4);
        let radius = fc
            .attr("radius")
            .map(|s| parse_f64_ctx(s, "flexcomp.radius"))
            .transpose()?
            .unwrap_or(0.003);

        let elast = fc.child("elasticity");
        let young = elast
            .and_then(|e| e.attr("young"))
            .or_else(|| fc.attr("young"))
            .map(|s| parse_f64_ctx(s, "flexcomp.young"))
            .transpose()?
            .unwrap_or(5.0e7);
        let poisson = elast
            .and_then(|e| e.attr("poisson"))
            .or_else(|| fc.attr("poisson"))
            .map(|s| parse_f64_ctx(s, "flexcomp.poisson"))
            .transpose()?
            .unwrap_or(0.35);
        let damping = elast
            .and_then(|e| e.attr("damping"))
            .map(|s| parse_f64_ctx(s, "flexcomp.damping"))
            .transpose()?
            .unwrap_or(0.002);

        // Attach to explicit body_a / body_b if provided, else create a fixed anchor body at `pos`
        let body_a = if let Some(ba_name) = fc.attr("body_a") {
            *self.body_name_to_id.get(ba_name).ok_or_else(|| {
                RobotLoadError::Mjcf(format!("flexcomp '{name}' unknown body_a '{ba_name}'"))
            })?
        } else {
            let id = self.bodies.len();
            self.bodies.push(RigidBodySpec {
                id,
                name: format!("{name}_root"),
                is_fixed: fc.attr("pinned").map(|v| v != "false").unwrap_or(true),
                mass: 0.1,
                inertia: Mat3::diag([1e-4, 1e-4, 1e-4]),
                com_local: Vec3::ZERO,
                rest_pose: Se3::new(pos, Quat::IDENTITY),
                collision_radius: radius,
                rayleigh_alpha: 0.08,
                tags: vec!["mjcf_flexcomp".to_string()],
            });
            self.body_name_to_id.insert(format!("{name}_root"), id);
            id
        };

        let body_b = match fc.attr("body_b") {
            Some(bb_name) if !bb_name.is_empty() && bb_name != "none" => {
                Some(*self.body_name_to_id.get(bb_name).ok_or_else(|| {
                    RobotLoadError::Mjcf(format!("flexcomp '{name}' unknown body_b '{bb_name}'"))
                })?)
            }
            _ => None,
        };

        let n_elem = n_nodes - 1;
        let world_a = self.bodies[body_a].rest_pose.pos;
        let world_b = match body_b {
            Some(bb) => self.bodies[bb].rest_pose.pos,
            None => world_a + Vec3::new((n_elem as f64) * dx, 0.0, 0.0),
        };
        let span = world_b - world_a;
        let q = quat_from_x_axis(span);
        let elem_len = (span.norm() / (n_elem as f64)).max(1e-4);

        let mut poses = Vec::with_capacity(n_elem + 1);
        for j in 0..=n_elem {
            let u = (j as f64) / (n_elem as f64);
            poses.push(Se3::new(world_a + u * span, q));
        }

        let q_ca = self.bodies[body_a].rest_pose.rot.conj().mul(q).normalized();
        let clamp_b = body_b.map(|bb| {
            let q_cb = self.bodies[bb].rest_pose.rot.conj().mul(q).normalized();
            Se3::new(Vec3::ZERO, q_cb)
        });

        let bid = self.beams.len();
        self.beams.push(BeamEdgeSpec {
            id: bid,
            name,
            body_a,
            clamp_a: Se3::new(Vec3::ZERO, q_ca),
            body_b,
            clamp_b,
            n_elements: n_elem,
            formulation: BeamFormulation::DiscreteElasticRod,
            section: BeamSection::Circular { radius },
            material: BeamMaterial {
                youngs_modulus: young,
                poisson_ratio: poisson,
                density: 1200.0,
                rayleigh_alpha: 0.10,
                rayleigh_beta: damping,
            },
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: vec![elem_len; n_elem],
            element_transforms: None,
            interior_node_contact_radius: Some(radius),
            enable_self_contact: false,
        });
        Ok(())
    }
}

/// Parse a MuJoCo MJCF XML string (`<mujoco>`) into a validated [`RigidFlexGraph`].
///
/// Supports `<option>`, `<compiler>`, `<default>`, `<worldbody>` nested `<body>`/`<inertial>`/
/// `<joint>`/`<geom>`/`<site>`, `<tendon><spatial>`, `<actuator><motor|position|general>`,
/// `<flexcomp>`, and `solref`/`solimp` impedance parameters.
pub fn load_graph_from_mjcf_str(xml: &str) -> Result<RigidFlexGraph, RobotLoadError> {
    let root = parse_xml_str(xml)?;
    if root.tag != "mujoco" {
        return Err(RobotLoadError::Mjcf(format!(
            "expected root <mujoco>, found <{}>",
            root.tag
        )));
    }
    let model_name = root.attr("model").unwrap_or("mjcf_robot").to_string();

    // Compiler angle convention: MuJoCo defaults to degrees unless angle="radian"
    let angle_scale = match root.child("compiler").and_then(|c| c.attr("angle")) {
        Some(a) if a.eq_ignore_ascii_case("radian") => 1.0,
        _ => PI / 180.0,
    };

    let mut gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut preferred_dt = 0.001;
    if let Some(opt) = root.child("option") {
        if let Some(g) = opt.attr("gravity") {
            gravity = parse_vec3_mjcf(g, "option.gravity")?;
        }
        if let Some(dt) = opt.attr("timestep") {
            preferred_dt = parse_f64_ctx(dt, "option.timestep")?;
        }
    }

    let mut defaults = MjcfDefaults::base();
    if let Some(def) = root.child("default") {
        defaults.apply_node(def)?;
    }

    let mut builder = MjcfBuilder {
        angle_scale,
        defaults,
        bodies: Vec::new(),
        beams: Vec::new(),
        tendons: Vec::new(),
        joints: Vec::new(),
        contact_sites: Vec::new(),
        mesh_colliders: Vec::new(),
        body_name_to_id: HashMap::new(),
        joint_name_to_idx: HashMap::new(),
        tendon_name_to_idx: HashMap::new(),
        site_map: HashMap::new(),
        world_anchor_id: None,
    };

    // 1. Parse <worldbody>
    let worldbody = root
        .child("worldbody")
        .ok_or_else(|| RobotLoadError::Mjcf("<mujoco> missing <worldbody>".into()))?;
    for gnode in worldbody.children_named("geom") {
        let _ = builder.parse_geom_node(gnode, None)?;
    }
    for snode in worldbody.children_named("site") {
        if let Some(sname) = snode.attr("name") {
            let wid = builder.ensure_world_anchor();
            let spos = match snode.attr("pos") {
                Some(s) => parse_vec3_mjcf(s, "worldbody.site.pos")?,
                None => Vec3::ZERO,
            };
            builder.site_map.insert(sname.to_string(), (wid, spos));
        }
    }
    for bnode in worldbody.children_named("body") {
        builder.parse_body_recursive(bnode, None, Se3::IDENTITY)?;
    }
    for fc in worldbody.children_named("flexcomp") {
        builder.parse_flexcomp(fc)?;
    }
    for fc in root.children_named("flexcomp") {
        builder.parse_flexcomp(fc)?;
    }

    // 2. Parse <tendon><spatial>
    let mut next_act_id = 0usize;
    if let Some(tnode) = root.child("tendon") {
        for spatial in tnode.children_named("spatial") {
            let tid = builder.tendons.len();
            let tname = spatial
                .attr("name")
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("spatial_tendon_{tid}"));
            let mut waypoints = Vec::new();
            for sref in spatial.children_named("site") {
                let sname = sref.attr("site").ok_or_else(|| {
                    RobotLoadError::Mjcf(format!("tendon '{tname}' <site> missing 'site' attr"))
                })?;
                let &(body_id, local_pos) = builder.site_map.get(sname).ok_or_else(|| {
                    RobotLoadError::Mjcf(format!("tendon '{tname}' references unknown site '{sname}'"))
                })?;
                waypoints.push(TendonWaypoint {
                    body_id,
                    local_pos,
                    wrap_angle_rad: 0.0,
                });
            }
            if waypoints.len() < 2 {
                return Err(RobotLoadError::Mjcf(format!(
                    "spatial tendon '{tname}' requires >= 2 <site> waypoints"
                )));
            }
            let mut geom_len = 0.0;
            for w in waypoints.windows(2) {
                let p0 = builder.bodies[w[0].body_id]
                    .rest_pose
                    .transform_point(w[0].local_pos);
                let p1 = builder.bodies[w[1].body_id]
                    .rest_pose
                    .transform_point(w[1].local_pos);
                geom_len += (p1 - p0).norm();
            }
            let rest_length = spatial
                .attr("springlength")
                .map(|s| parse_f64_list(s, "tendon.springlength"))
                .transpose()?
                .and_then(|v| v.first().copied())
                .filter(|&l| l > 0.0)
                .unwrap_or(geom_len);
            let stiffness = spatial
                .attr("stiffness")
                .map(|s| parse_f64_ctx(s, "tendon.stiffness"))
                .transpose()?
                .unwrap_or(builder.defaults.tendon_stiffness);
            let damping = spatial
                .attr("damping")
                .map(|s| parse_f64_ctx(s, "tendon.damping"))
                .transpose()?
                .unwrap_or(builder.defaults.tendon_damping);
            let sri = parse_solref_imp_attrs(
                spatial.attr("solreflimit").or_else(|| spatial.attr("solref")),
                spatial.attr("solimplimit").or_else(|| spatial.attr("solimp")),
                builder.defaults.tendon_solref_imp,
            )?;

            let act_id = next_act_id;
            next_act_id += 1;
            builder.tendon_name_to_idx.insert(tname.clone(), tid);
            builder.tendons.push(TendonEdgeSpec {
                id: tid,
                name: tname,
                actuator_id: act_id,
                waypoints,
                drive_mode: TendonDriveMode::Linear,
                rest_length,
                slack: 0.0,
                stiffness: stiffness.max(100.0),
                damping: damping.max(0.1),
                capstan_mu_theta: 0.0,
                stick_slip_enabled: false,
                servo_torque_limit: 5.0,
                servo_speed_limit: 0.5,
                servo_deadband: 0.0,
                angle_limits: [0.0, 0.25],
                solref_imp: sri,
            });
        }
    }

    // 3. Parse <actuator> (<motor>, <position>, <general>)
    if let Some(anode) = root.child("actuator") {
        for act in &anode.children {
            if !matches!(act.tag.as_str(), "motor" | "position" | "velocity" | "general") {
                continue;
            }
            let gear = act
                .attr("gear")
                .map(|s| parse_f64_list(s, "actuator.gear"))
                .transpose()?
                .and_then(|v| v.first().copied())
                .map(|g| g.abs().max(1e-3))
                .unwrap_or(builder.defaults.motor_gear);
            let explicit_kp = act
                .attr("kp")
                .map(|s| parse_f64_ctx(s, "actuator.kp"))
                .transpose()?;
            let gainprm_0 = act
                .attr("gainprm")
                .map(|s| parse_f64_list(s, "actuator.gainprm"))
                .transpose()?
                .and_then(|v| v.first().copied())
                .filter(|&g| g.abs() > 0.0);
            let biasprm_kd = act
                .attr("biasprm")
                .map(|s| parse_f64_list(s, "actuator.biasprm"))
                .transpose()?
                .and_then(|v| (v.len() >= 3).then(|| (-v[2]).max(0.0)));
            let kp = explicit_kp
                .or_else(|| gainprm_0.map(f64::abs))
                .unwrap_or_else(|| {
                    if act.tag == "position" {
                        builder.defaults.motor_kp
                    } else {
                        (0.2 * gear).max(6.0)
                    }
                });
            let kv = act
                .attr("kv")
                .map(|s| parse_f64_ctx(s, "actuator.kv"))
                .transpose()?
                .or(biasprm_kd);
            let torque_limit = act
                .attr("forcerange")
                .map(|s| parse_f64_list(s, "actuator.forcerange"))
                .transpose()?
                .and_then(|v| (v.len() >= 2).then(|| v[0].abs().max(v[1].abs())))
                .unwrap_or_else(|| (gear * 0.5).max(5.0));

            if let Some(jname) = act.attr("joint") {
                if let Some(&jidx) = builder.joint_name_to_idx.get(jname) {
                    let j = &mut builder.joints[jidx];
                    if j.actuator_id.is_none() {
                        j.actuator_id = Some(next_act_id);
                        next_act_id += 1;
                    }
                    j.kp = j.kp.max(kp);
                    if let Some(kd) = kv {
                        j.kd = kd.max(0.0);
                    }
                    j.torque_limit = torque_limit;
                    j.speed_limit = 8.0;
                }
            } else if let Some(tname) = act.attr("tendon") {
                if let Some(&tidx) = builder.tendon_name_to_idx.get(tname) {
                    let t = &mut builder.tendons[tidx];
                    t.servo_torque_limit = torque_limit;
                    if let Some(k_act) = explicit_kp {
                        if k_act > 0.0 {
                            t.stiffness = k_act;
                        }
                    }
                    if let Some(d_act) = kv {
                        if d_act >= 0.0 {
                            t.damping = d_act;
                        }
                    }
                    if let Some(cr) = act.attr("ctrlrange") {
                        let rv = parse_f64_list(cr, "actuator.ctrlrange")?;
                        if rv.len() == 2 {
                            t.angle_limits = [rv[0], rv[1]];
                        }
                    }
                }
            }
        }
    }

    if builder.bodies.is_empty() {
        return Err(RobotLoadError::Mjcf(
            "MJCF <worldbody> contains no <body> or <flexcomp> elements".into(),
        ));
    }

    let graph = RigidFlexGraph {
        name: model_name,
        gravity,
        bodies: builder.bodies,
        beams: builder.beams,
        tendons: builder.tendons,
        joints: builder.joints,
        contact_sites: builder.contact_sites,
        symmetry: SymmetryGroupSpec::trivial(),
        solver_hints: SolverHints {
            preferred_dt,
            rayleigh_alpha: 0.05,
            rayleigh_beta: 0.002,
            force_route: None,
        },
        diff_params: Vec::new(),
        mesh_colliders: builder.mesh_colliders,
    };
    graph.validate()?;
    Ok(graph)
}

/// Load a MuJoCo MJCF XML file into a validated [`RigidFlexGraph`].
pub fn load_graph_from_mjcf_file(path: impl AsRef<Path>) -> Result<RigidFlexGraph, RobotLoadError> {
    let xml = std::fs::read_to_string(path.as_ref())?;
    load_graph_from_mjcf_str(&xml)
}
