//! Declarative JSON, YAML, and URDF-Flex XML loader producing validated `RigidFlexGraph` instances.

use crate::{
    build_bristle_worm, build_endoscope_spline_tube_sdf, build_flexspine_centipede,
    build_rigid_flex_grid, build_serpworm, build_soft_octopus_arm, build_tendon_endoscope,
    build_tensegrity_6bar, load_graph_from_mjcf_file, BristleWormConfig, CentipedeConfig,
    EndoscopeConfig, FlexGridConfig, OctopusArmConfig, SerpWormConfig, SerpWormPreset,
    TensegrityConfig,
};
use fg_ir::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, IrError,
    JointEdgeSpec, JointType, RigidBodySpec, RigidFlexGraph, SolverHints, SymmetryGroupSpec,
    TendonDriveMode, TendonEdgeSpec, TendonWaypoint,
};
use fg_math::{BeamMaterial, BeamSection, Mat3, Quat, Se3, Vec3};
use fg_sdf::{
    HeightFieldSdf, MaterialLibrary, PlaneSdf, WorldSdf, MAT_BEKKER_SOFT_SOIL, MAT_HARD_FLOOR,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Error returned while loading or parsing a declarative robot description.
#[derive(Debug, Error)]
pub enum RobotLoadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("IR validation error: {0}")]
    Ir(#[from] IrError),
    #[error("URDF-Flex XML parse error: {0}")]
    UrdfFlex(String),
    #[error("ROS URDF parse error: {0}")]
    RosUrdf(String),
    #[error("MuJoCo MJCF parse error: {0}")]
    Mjcf(String),
    #[error("unknown archetype or unsupported file extension: {0}")]
    Unsupported(String),
}

/// Optional compact archetype envelope accepted by JSON and YAML loaders in addition to raw `RigidFlexGraph`.
#[derive(Debug, Deserialize)]
#[serde(tag = "archetype", rename_all = "snake_case")]
enum ArchetypeEnvelope {
    Serpworm {
        #[serde(default)]
        config: Box<SerpWormConfig>,
    },
    BristleWorm {
        #[serde(default)]
        config: BristleWormConfig,
    },
    TendonEndoscope {
        #[serde(default)]
        config: EndoscopeConfig,
    },
    FlexspineCentipede {
        #[serde(default)]
        config: CentipedeConfig,
    },
    Tensegrity6bar {
        #[serde(default)]
        config: TensegrityConfig,
    },
    SoftOctopusArm {
        #[serde(default)]
        config: OctopusArmConfig,
    },
    RigidFlexGrid {
        #[serde(default)]
        config: FlexGridConfig,
    },
}

/// Build one of the 7 calibrated archetypes by canonical name.
pub fn build_archetype_by_name(name: &str) -> Result<RigidFlexGraph, RobotLoadError> {
    let key = name.trim().to_ascii_lowercase();
    let g = match key.as_str() {
        "serpworm" | "serpworm_5" | "serpworm-5" => {
            build_serpworm(&SerpWormConfig::from_preset(SerpWormPreset::GpuCoreDefault, 5, 16), None)
        }
        "serpworm_1" | "serpworm-1" => {
            build_serpworm(&SerpWormConfig::from_preset(SerpWormPreset::GpuCoreDefault, 1, 16), None)
        }
        "serpworm_3" | "serpworm-3" => {
            build_serpworm(&SerpWormConfig::from_preset(SerpWormPreset::GpuCoreDefault, 3, 16), None)
        }
        "serpworm_7" | "serpworm-7" => {
            build_serpworm(&SerpWormConfig::from_preset(SerpWormPreset::GpuCoreDefault, 7, 16), None)
        }
        "bristle_worm" | "bristleworm" => build_bristle_worm(&BristleWormConfig::default(), None),
        "tendon_endoscope" | "tendon_endoscope_3d" | "endoscope" => {
            build_tendon_endoscope(&EndoscopeConfig::default())
        }
        "flexspine_centipede" | "centipede" => {
            build_flexspine_centipede(&CentipedeConfig::default())
        }
        "tensegrity_6bar" | "tensegrity" | "tensegrity-6bar" => {
            build_tensegrity_6bar(&TensegrityConfig::default())
        }
        "soft_octopus_arm" | "soft_octopus_arm_der" | "octopus_arm" => {
            build_soft_octopus_arm(&OctopusArmConfig::default())
        }
        "rigid_flex_grid_4x4" | "rigid_flex_grid" | "flex_grid_4x4" => {
            build_rigid_flex_grid(&FlexGridConfig::default())
        }
        "rigid_flex_grid_6x6" | "flex_grid_6x6" => build_rigid_flex_grid(&FlexGridConfig {
            rows: 6,
            cols: 6,
            ..Default::default()
        }),
        _ => return Err(RobotLoadError::Unsupported(name.to_string())),
    };
    g.validate()?;
    Ok(g)
}

/// Build the default paired [`WorldSdf`] environment for one of the 7 archetypes:
/// - `SerpWorm-N`, `BristleWorm`, `Tensegrity-6Bar`, `RigidFlexGrid-*`, `SoftOctopusArm-DER`: flat hard floor (`z = 0`, `MAT_HARD_FLOOR`)
/// - `TendonEndoscope-3D`: 3D curved variable-radius `SplineTubeSdf` lumen (`MAT_ACRYLIC_WALL`)
/// - `FlexSpine-Centipede`: flat Bekker elasto-plastic soft-soil `HeightFieldSdf` (`MAT_BEKKER_SOFT_SOIL`)
pub fn build_archetype_default_world(name: &str) -> Result<WorldSdf, RobotLoadError> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "serpworm" | "serpworm_1" | "serpworm-1" | "serpworm_3" | "serpworm-3" | "serpworm_5"
        | "serpworm-5" | "serpworm_7" | "serpworm-7" | "bristle_worm" | "bristleworm"
        | "tensegrity_6bar" | "tensegrity" | "tensegrity-6bar" | "soft_octopus_arm"
        | "soft_octopus_arm_der" | "octopus_arm" | "rigid_flex_grid_4x4" | "rigid_flex_grid"
        | "flex_grid_4x4" | "rigid_flex_grid_6x6" | "flex_grid_6x6" => {
            let mut world = WorldSdf::new(MaterialLibrary::standard());
            world.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_HARD_FLOOR)));
            Ok(world)
        }
        "tendon_endoscope" | "tendon_endoscope_3d" | "endoscope" => {
            let mut world = WorldSdf::new(MaterialLibrary::standard());
            let tube = build_endoscope_spline_tube_sdf(&EndoscopeConfig::default());
            world.add_unbounded(Arc::new(tube));
            Ok(world)
        }
        "flexspine_centipede" | "centipede" => {
            let mut world = WorldSdf::new(MaterialLibrary::standard());
            let soil = HeightFieldSdf::analytic(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, MAT_BEKKER_SOFT_SOIL);
            world.add_unbounded(Arc::new(soil));
            Ok(world)
        }
        _ => Err(RobotLoadError::Unsupported(name.to_string())),
    }
}

/// Parse a `RigidFlexGraph` (or compact archetype descriptor) from a JSON string and validate it.
pub fn load_graph_from_json_str(s: &str) -> Result<RigidFlexGraph, RobotLoadError> {
    if let Ok(env) = serde_json::from_str::<ArchetypeEnvelope>(s) {
        let g = match env {
            ArchetypeEnvelope::Serpworm { config } => build_serpworm(&config, None),
            ArchetypeEnvelope::BristleWorm { config } => build_bristle_worm(&config, None),
            ArchetypeEnvelope::TendonEndoscope { config } => build_tendon_endoscope(&config),
            ArchetypeEnvelope::FlexspineCentipede { config } => build_flexspine_centipede(&config),
            ArchetypeEnvelope::Tensegrity6bar { config } => build_tensegrity_6bar(&config),
            ArchetypeEnvelope::SoftOctopusArm { config } => build_soft_octopus_arm(&config),
            ArchetypeEnvelope::RigidFlexGrid { config } => build_rigid_flex_grid(&config),
        };
        g.validate()?;
        return Ok(g);
    }
    let g = RigidFlexGraph::from_json_str(s)?;
    g.validate()?;
    Ok(g)
}

/// Load a `RigidFlexGraph` from a JSON file.
pub fn load_graph_from_json_file(path: impl AsRef<Path>) -> Result<RigidFlexGraph, RobotLoadError> {
    let s = std::fs::read_to_string(path.as_ref())?;
    load_graph_from_json_str(&s)
}

/// Parse a `RigidFlexGraph` (or compact archetype descriptor) from a YAML string and validate it.
pub fn load_graph_from_yaml_str(s: &str) -> Result<RigidFlexGraph, RobotLoadError> {
    if let Ok(env) = serde_yaml::from_str::<ArchetypeEnvelope>(s) {
        let g = match env {
            ArchetypeEnvelope::Serpworm { config } => build_serpworm(&config, None),
            ArchetypeEnvelope::BristleWorm { config } => build_bristle_worm(&config, None),
            ArchetypeEnvelope::TendonEndoscope { config } => build_tendon_endoscope(&config),
            ArchetypeEnvelope::FlexspineCentipede { config } => build_flexspine_centipede(&config),
            ArchetypeEnvelope::Tensegrity6bar { config } => build_tensegrity_6bar(&config),
            ArchetypeEnvelope::SoftOctopusArm { config } => build_soft_octopus_arm(&config),
            ArchetypeEnvelope::RigidFlexGrid { config } => build_rigid_flex_grid(&config),
        };
        g.validate()?;
        return Ok(g);
    }
    let g = RigidFlexGraph::from_yaml_str(s)?;
    g.validate()?;
    Ok(g)
}

/// Load a `RigidFlexGraph` from a YAML file.
pub fn load_graph_from_yaml_file(path: impl AsRef<Path>) -> Result<RigidFlexGraph, RobotLoadError> {
    let s = std::fs::read_to_string(path.as_ref())?;
    load_graph_from_yaml_str(&s)
}

/// Auto-detect format by extension (`.json`, `.yaml`/`.yml`, `.mjcf`, `.urdf`/`.xml`) and load a `RigidFlexGraph`.
/// For `.xml`, inspects the root tag (`<mujoco>` -> MJCF, `<robot>` with `<beam>`/`<tendon>` -> URDF-Flex,
/// standard ROS `<robot>` -> `urdf-rs` when `ros-urdf` feature is enabled).
pub fn load_graph_from_path(path: impl AsRef<Path>) -> Result<RigidFlexGraph, RobotLoadError> {
    let p = path.as_ref();
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "json" => load_graph_from_json_file(p),
        "yaml" | "yml" => load_graph_from_yaml_file(p),
        "mjcf" => load_graph_from_mjcf_file(p),
        "urdf" | "xml" => {
            let content = std::fs::read_to_string(p)?;
            if content.contains("<mujoco") {
                return crate::load_graph_from_mjcf_str(&content);
            }
            let has_flex_ext = content.contains("<beam")
                || content.contains("<tendon")
                || content.contains("<contact_site")
                || content.contains("<fixed_base");
            #[cfg(feature = "ros-urdf")]
            if !has_flex_ext {
                if let Ok(g) = crate::load_graph_from_ros_urdf_str(&content) {
                    return Ok(g);
                }
            }
            let _ = has_flex_ext;
            load_graph_from_urdf_flex_str(&content)
        }
        _ => Err(RobotLoadError::Unsupported(p.display().to_string())),
    }
}

// ============================================================================
// Pure-Rust Lightweight XML & URDF-Flex Parser (zero extra crate dependencies)
// ============================================================================

#[derive(Debug, Clone)]
pub(crate) struct XmlNode {
    pub(crate) tag: String,
    pub(crate) attrs: HashMap<String, String>,
    pub(crate) children: Vec<XmlNode>,
}

impl XmlNode {
    pub(crate) fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.get(k).map(String::as_str)
    }

    pub(crate) fn child(&self, tag: &str) -> Option<&XmlNode> {
        self.children.iter().find(|c| c.tag == tag)
    }

    pub(crate) fn children_named<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a XmlNode> {
        self.children.iter().filter(move |c| c.tag == tag)
    }
}

pub(crate) fn parse_xml_str(src: &str) -> Result<XmlNode, RobotLoadError> {
    let mut parser = XmlParser::new(src);
    parser.parse_element()
}

pub(crate) fn parse_f64_ctx(s: &str, ctx: &str) -> Result<f64, RobotLoadError> {
    parse_f64(s, ctx)
}

struct XmlParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> XmlParser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws_and_misc(&mut self) -> Result<(), RobotLoadError> {
        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.starts_with("<?") {
                let end = self.find_str("?>").ok_or_else(|| {
                    RobotLoadError::UrdfFlex("unterminated XML processing instruction".into())
                })?;
                self.pos = end + 2;
            } else if self.starts_with("<!--") {
                let end = self.find_str("-->").ok_or_else(|| {
                    RobotLoadError::UrdfFlex("unterminated XML comment".into())
                })?;
                self.pos = end + 3;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn starts_with(&self, pat: &str) -> bool {
        self.bytes[self.pos..].starts_with(pat.as_bytes())
    }

    fn find_str(&self, pat: &str) -> Option<usize> {
        let p = pat.as_bytes();
        self.bytes[self.pos..]
            .windows(p.len())
            .position(|w| w == p)
            .map(|rel| self.pos + rel)
    }

    fn parse_ident(&mut self) -> Result<String, RobotLoadError> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b':' | b'.') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(RobotLoadError::UrdfFlex(format!(
                "expected XML identifier at byte {}",
                self.pos
            )));
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }

    fn parse_attr_value(&mut self) -> Result<String, RobotLoadError> {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return Err(RobotLoadError::UrdfFlex("unexpected EOF in attribute".into()));
        }
        let quote = self.bytes[self.pos];
        if quote != b'"' && quote != b'\'' {
            return Err(RobotLoadError::UrdfFlex(format!(
                "expected quote in XML attribute at byte {}",
                self.pos
            )));
        }
        self.pos += 1;
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != quote {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return Err(RobotLoadError::UrdfFlex("unterminated attribute quote".into()));
        }
        let val = String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned();
        self.pos += 1;
        Ok(val)
    }

    fn parse_element(&mut self) -> Result<XmlNode, RobotLoadError> {
        self.skip_ws_and_misc()?;
        if !self.starts_with("<") || self.starts_with("</") {
            return Err(RobotLoadError::UrdfFlex(format!(
                "expected '<' at byte {}",
                self.pos
            )));
        }
        self.pos += 1;
        let tag = self.parse_ident()?;
        let mut attrs = HashMap::new();

        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.starts_with("/>") {
                self.pos += 2;
                return Ok(XmlNode {
                    tag,
                    attrs,
                    children: Vec::new(),
                });
            }
            if self.starts_with(">") {
                self.pos += 1;
                break;
            }
            let key = self.parse_ident()?;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if !self.starts_with("=") {
                return Err(RobotLoadError::UrdfFlex(format!(
                    "expected '=' after attribute '{key}'"
                )));
            }
            self.pos += 1;
            let val = self.parse_attr_value()?;
            attrs.insert(key, val);
        }

        let mut children = Vec::new();
        loop {
            self.skip_ws_and_misc()?;
            if self.pos >= self.bytes.len() {
                return Err(RobotLoadError::UrdfFlex(format!(
                    "unclosed XML element <{tag}>"
                )));
            }
            if self.starts_with("</") {
                self.pos += 2;
                let close_tag = self.parse_ident()?;
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                    self.pos += 1;
                }
                if !self.starts_with(">") {
                    return Err(RobotLoadError::UrdfFlex(format!(
                        "expected '>' closing </{close_tag}>"
                    )));
                }
                self.pos += 1;
                if close_tag != tag {
                    return Err(RobotLoadError::UrdfFlex(format!(
                        "mismatched closing tag </{close_tag}> for <{tag}>"
                    )));
                }
                break;
            }
            if self.starts_with("<") {
                children.push(self.parse_element()?);
            } else {
                // Skip text node content until next tag
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'<' {
                    self.pos += 1;
                }
            }
        }

        Ok(XmlNode {
            tag,
            attrs,
            children,
        })
    }
}

fn parse_f64(s: &str, ctx: &str) -> Result<f64, RobotLoadError> {
    s.trim()
        .parse::<f64>()
        .map_err(|e| RobotLoadError::UrdfFlex(format!("invalid f64 for {ctx} ('{s}'): {e}")))
}

fn parse_vec3(s: &str, ctx: &str) -> Result<Vec3, RobotLoadError> {
    let parts: Vec<&str> = s.split(|c: char| c.is_ascii_whitespace() || c == ',').filter(|p| !p.is_empty()).collect();
    if parts.len() != 3 {
        return Err(RobotLoadError::UrdfFlex(format!(
            "expected 3 floats for {ctx}, got '{}'",
            s
        )));
    }
    Ok(Vec3::new(
        parse_f64(parts[0], ctx)?,
        parse_f64(parts[1], ctx)?,
        parse_f64(parts[2], ctx)?,
    ))
}

/// Convert URDF roll-pitch-yaw (`Rz(yaw) * Ry(pitch) * Rx(roll)`) to a unit quaternion.
fn quat_from_rpy(rpy: Vec3) -> Quat {
    let qx = Quat::from_axis_angle(Vec3::EX, rpy.x());
    let qy = Quat::from_axis_angle(Vec3::EY, rpy.y());
    let qz = Quat::from_axis_angle(Vec3::EZ, rpy.z());
    qz.mul(qy).mul(qx).normalized()
}

fn parse_origin(node: Option<&XmlNode>) -> Result<Se3, RobotLoadError> {
    let Some(n) = node else {
        return Ok(Se3::IDENTITY);
    };
    let pos = match n.attr("xyz") {
        Some(v) => parse_vec3(v, "origin.xyz")?,
        None => Vec3::ZERO,
    };
    let rot = match n.attr("rpy") {
        Some(v) => quat_from_rpy(parse_vec3(v, "origin.rpy")?),
        None => Quat::IDENTITY,
    };
    Ok(Se3::new(pos, rot))
}

fn frame_along_x(dir: Vec3) -> Quat {
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

/// Parse a URDF-Flex XML string into a validated `RigidFlexGraph`.
///
/// Supports standard URDF `<robot>`, `<link>` (with `<inertial>`, `<origin>`), `<joint>`
/// (`revolute`, `continuous`, `prismatic`, `spherical`, `fixed`, `stop_gap`, `cable_spring`),
/// plus FlexGraph continuum extensions:
/// - `<fixed_base link="..."/>`
/// - `<beam name="..." body_a="..." [body_b="..."] n_elem="..." section="rect|circle" .../>`
/// - `<tendon name="..." actuator="..." mode="linear|capstan|crank|legacy_wrap" waypoints="link:x,y,z;..." .../>`
/// - `<contact_site name="..." body="..." pos="..." mode="ratchet_wheel|anisotropic_scale|isotropic_pad" .../>`
pub fn load_graph_from_urdf_flex_str(xml: &str) -> Result<RigidFlexGraph, RobotLoadError> {
    let mut parser = XmlParser::new(xml);
    let root = parser.parse_element()?;
    if root.tag != "robot" {
        return Err(RobotLoadError::UrdfFlex(format!(
            "expected root <robot>, found <{}>",
            root.tag
        )));
    }
    let robot_name = root.attr("name").unwrap_or("urdf_flex_robot").to_string();

    let gravity = match root.child("gravity").and_then(|g| g.attr("xyz")) {
        Some(s) => parse_vec3(s, "gravity.xyz")?,
        None => Vec3::new(0.0, 0.0, -9.81),
    };

    let mut solver_hints = SolverHints::default();
    if let Some(sh) = root.child("solver_hints") {
        if let Some(dt) = sh.attr("dt") {
            solver_hints.preferred_dt = parse_f64(dt, "solver_hints.dt")?;
        }
        if let Some(a) = sh.attr("rayleigh_alpha") {
            solver_hints.rayleigh_alpha = parse_f64(a, "solver_hints.rayleigh_alpha")?;
        }
        if let Some(b) = sh.attr("rayleigh_beta") {
            solver_hints.rayleigh_beta = parse_f64(b, "solver_hints.rayleigh_beta")?;
        }
    }

    let mut fixed_links = std::collections::HashSet::new();
    for fb in root.children_named("fixed_base") {
        if let Some(lname) = fb.attr("link") {
            fixed_links.insert(lname.to_string());
        }
    }

    // 1. Parse <link> elements into RigidBodySpec
    let mut bodies = Vec::new();
    let mut link_index: HashMap<String, usize> = HashMap::new();
    let mut explicit_pose = Vec::new();

    for link in root.children_named("link") {
        let name = link
            .attr("name")
            .ok_or_else(|| RobotLoadError::UrdfFlex("<link> missing 'name'".into()))?
            .to_string();
        let is_fixed = fixed_links.contains(&name)
            || link.attr("fixed").map(|v| v == "true" || v == "1").unwrap_or(false);

        let mut mass = 0.25;
        let mut com_local = Vec3::ZERO;
        let mut inertia = Mat3::diag([1e-4, 1e-4, 1e-4]);
        if let Some(iner) = link.child("inertial") {
            if let Some(m_node) = iner.child("mass") {
                if let Some(v) = m_node.attr("value") {
                    mass = parse_f64(v, "inertial.mass")?;
                }
            }
            if let Some(orig) = iner.child("origin") {
                com_local = parse_origin(Some(orig))?.pos;
            }
            if let Some(i_node) = iner.child("inertia") {
                let ixx = parse_f64(i_node.attr("ixx").unwrap_or("0.0001"), "ixx")?;
                let ixy = parse_f64(i_node.attr("ixy").unwrap_or("0.0"), "ixy")?;
                let ixz = parse_f64(i_node.attr("ixz").unwrap_or("0.0"), "ixz")?;
                let iyy = parse_f64(i_node.attr("iyy").unwrap_or("0.0001"), "iyy")?;
                let iyz = parse_f64(i_node.attr("iyz").unwrap_or("0.0"), "iyz")?;
                let izz = parse_f64(i_node.attr("izz").unwrap_or("0.0001"), "izz")?;
                inertia = Mat3([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]);
            }
        }

        let has_origin = link.child("origin").is_some() || link.attr("xyz").is_some();
        let rest_pose = if let Some(orig) = link.child("origin") {
            parse_origin(Some(orig))?
        } else if let Some(xyz) = link.attr("xyz") {
            let pos = parse_vec3(xyz, "link.xyz")?;
            let rot = match link.attr("rpy") {
                Some(r) => quat_from_rpy(parse_vec3(r, "link.rpy")?),
                None => Quat::IDENTITY,
            };
            Se3::new(pos, rot)
        } else {
            Se3::IDENTITY
        };

        let collision_radius = link
            .attr("collision_radius")
            .map(|v| parse_f64(v, "link.collision_radius"))
            .transpose()?
            .unwrap_or(0.02);

        let id = bodies.len();
        link_index.insert(name.clone(), id);
        explicit_pose.push(has_origin);
        bodies.push(RigidBodySpec {
            id,
            name,
            is_fixed,
            mass,
            inertia,
            com_local,
            rest_pose,
            collision_radius,
            rayleigh_alpha: solver_hints.rayleigh_alpha,
            tags: vec!["urdf_flex".to_string()],
        });
    }

    let resolve_body = |name: &str| -> Result<usize, RobotLoadError> {
        link_index
            .get(name)
            .copied()
            .ok_or_else(|| RobotLoadError::UrdfFlex(format!("unknown link '{name}'")))
    };

    // 2. Parse <joint> elements and propagate kinematic rest poses to child links lacking explicit pose
    let mut joints = Vec::new();
    let mut next_actuator_id = 0usize;

    for jnode in root.children_named("joint") {
        let jname = jnode
            .attr("name")
            .ok_or_else(|| RobotLoadError::UrdfFlex("<joint> missing 'name'".into()))?
            .to_string();
        let jtype_str = jnode.attr("type").unwrap_or("revolute");
        let parent_name = jnode
            .child("parent")
            .and_then(|p| p.attr("link"))
            .ok_or_else(|| RobotLoadError::UrdfFlex(format!("joint '{jname}' missing <parent>")))?;
        let child_name = jnode
            .child("child")
            .and_then(|c| c.attr("link"))
            .ok_or_else(|| RobotLoadError::UrdfFlex(format!("joint '{jname}' missing <child>")))?;
        let body_a = resolve_body(parent_name)?;
        let body_b = resolve_body(child_name)?;
        let origin = parse_origin(jnode.child("origin"))?;

        if !explicit_pose[body_b] {
            bodies[body_b].rest_pose = bodies[body_a].rest_pose.compose(origin);
            explicit_pose[body_b] = true;
        }

        let axis = match jnode.child("axis").and_then(|a| a.attr("xyz")) {
            Some(s) => parse_vec3(s, "joint.axis")?.normalized(),
            None => Vec3::EZ,
        };

        let mut limits = None;
        let mut torque_limit = 2.0;
        let mut speed_limit = 6.0;
        if let Some(lim) = jnode.child("limit") {
            if let (Some(lo), Some(hi)) = (lim.attr("lower"), lim.attr("upper")) {
                limits = Some([parse_f64(lo, "limit.lower")?, parse_f64(hi, "limit.upper")?]);
            }
            if let Some(eff) = lim.attr("effort") {
                torque_limit = parse_f64(eff, "limit.effort")?;
            }
            if let Some(vel) = lim.attr("velocity") {
                speed_limit = parse_f64(vel, "limit.velocity")?;
            }
        }

        let kp = jnode
            .attr("kp")
            .map(|v| parse_f64(v, "joint.kp"))
            .transpose()?
            .unwrap_or(8.0);
        let kd = jnode
            .attr("kd")
            .map(|v| parse_f64(v, "joint.kd"))
            .transpose()?
            .unwrap_or(0.15);
        let implicit_coupling = jnode
            .attr("implicit_coupling")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let (joint_type, actuated) = match jtype_str {
            "revolute" | "continuous" => (
                JointType::Revolute {
                    axis_local_a: axis,
                    anchor_a: origin.pos,
                    anchor_b: Vec3::ZERO,
                    pos_stiffness: 10_000.0,
                    pos_damping: 10.0,
                    align_stiffness: 20.0,
                    align_damping: 0.1,
                },
                true,
            ),
            "prismatic" => (
                JointType::Prismatic {
                    axis_local_a: axis,
                    anchor_a: origin.pos,
                    anchor_b: Vec3::ZERO,
                    perp_stiffness: 10_000.0,
                    perp_damping: 10.0,
                },
                true,
            ),
            "spherical" => (
                JointType::Spherical {
                    anchor_a: origin.pos,
                    anchor_b: Vec3::ZERO,
                    pos_stiffness: 10_000.0,
                    pos_damping: 10.0,
                },
                false,
            ),
            "fixed" | "weld" => (
                JointType::Weld {
                    rel_pose: origin,
                    pos_stiffness: 20_000.0,
                    rot_stiffness: 200.0,
                },
                false,
            ),
            "stop_gap" => (
                JointType::AxialStopGap {
                    axis_local_a: axis,
                    min_gap: parse_f64(jnode.attr("min_gap").unwrap_or("0.05"), "joint.min_gap")?,
                    stiffness: parse_f64(
                        jnode.attr("stiffness").unwrap_or("50000.0"),
                        "joint.stiffness",
                    )?,
                },
                false,
            ),
            "cable_spring" => {
                let anchor_a = match jnode.attr("anchor_a") {
                    Some(s) => parse_vec3(s, "joint.anchor_a")?,
                    None => origin.pos,
                };
                let anchor_b = match jnode.attr("anchor_b") {
                    Some(s) => parse_vec3(s, "joint.anchor_b")?,
                    None => Vec3::ZERO,
                };
                let rest_in = jnode
                    .attr("rest_length")
                    .map(|v| parse_f64(v, "joint.rest_length"))
                    .transpose()?
                    .unwrap_or(0.0);
                let rest_length = if rest_in > 0.0 {
                    rest_in
                } else {
                    let pa = bodies[body_a].rest_pose.transform_point(anchor_a);
                    let pb = bodies[body_b].rest_pose.transform_point(anchor_b);
                    (pb - pa).norm()
                };
                let stiffness = parse_f64(
                    jnode.attr("stiffness").unwrap_or("500.0"),
                    "joint.stiffness",
                )?;
                let damping = parse_f64(jnode.attr("damping").unwrap_or("1.0"), "joint.damping")?;
                let compression_allowed = jnode
                    .attr("compression_allowed")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);
                (
                    JointType::CableSpring {
                        anchor_a,
                        anchor_b,
                        rest_length,
                        stiffness,
                        damping,
                        compression_allowed,
                    },
                    false,
                )
            }
            other => {
                return Err(RobotLoadError::UrdfFlex(format!(
                    "unsupported joint type '{other}' on '{jname}'"
                )));
            }
        };

        let actuator_id = if let Some(aid) = jnode.attr("actuator") {
            let parsed = aid.parse::<usize>().map_err(|e| {
                RobotLoadError::UrdfFlex(format!("invalid actuator index '{aid}': {e}"))
            })?;
            next_actuator_id = next_actuator_id.max(parsed + 1);
            Some(parsed)
        } else if actuated {
            let id = next_actuator_id;
            next_actuator_id += 1;
            Some(id)
        } else {
            None
        };

        let jid = joints.len();
        joints.push(JointEdgeSpec {
            id: jid,
            name: jname,
            body_a,
            body_b,
            joint_type,
            actuator_id,
            kp,
            kd,
            torque_limit,
            speed_limit,
            static_friction: 0.0,
            deadband: 0.0,
            limits,
            implicit_coupling,
            solref_imp: None,
        });
    }

    // 3. Parse FlexGraph <beam> extensions
    let mut beams = Vec::new();
    for bnode in root.children_named("beam") {
        let bname = bnode.attr("name").unwrap_or("beam").to_string();
        let body_a = resolve_body(
            bnode
                .attr("body_a")
                .ok_or_else(|| RobotLoadError::UrdfFlex(format!("beam '{bname}' missing body_a")))?,
        )?;
        let body_b = match bnode.attr("body_b") {
            Some(name) if !name.is_empty() && name != "none" => Some(resolve_body(name)?),
            _ => None,
        };
        let n_elements = bnode
            .attr("n_elem")
            .unwrap_or("4")
            .parse::<usize>()
            .map_err(|e| RobotLoadError::UrdfFlex(format!("invalid beam n_elem: {e}")))?;

        let section = match bnode.attr("section").unwrap_or("circle") {
            "rect" | "rectangular" => BeamSection::Rectangular {
                width: parse_f64(bnode.attr("width").unwrap_or("0.016"), "beam.width")?,
                thickness: parse_f64(bnode.attr("thickness").unwrap_or("0.0002"), "beam.thickness")?,
            },
            "circle" | "circular" => BeamSection::Circular {
                radius: parse_f64(bnode.attr("radius").unwrap_or("0.0015"), "beam.radius")?,
            },
            "hollow" | "hollow_circular" => BeamSection::HollowCircular {
                outer_radius: parse_f64(
                    bnode.attr("outer_radius").unwrap_or("0.002"),
                    "beam.outer_radius",
                )?,
                inner_radius: parse_f64(
                    bnode.attr("inner_radius").unwrap_or("0.001"),
                    "beam.inner_radius",
                )?,
            },
            other => {
                return Err(RobotLoadError::UrdfFlex(format!(
                    "unsupported beam section '{other}'"
                )));
            }
        };

        let formulation = match bnode.attr("formulation").unwrap_or("corotational") {
            "corotational" | "euler_bernoulli" => BeamFormulation::CorotationalEulerBernoulli,
            "timoshenko" => BeamFormulation::CorotationalTimoshenko,
            "der" | "cosserat" => BeamFormulation::DiscreteElasticRod,
            other => {
                return Err(RobotLoadError::UrdfFlex(format!(
                    "unsupported beam formulation '{other}'"
                )));
            }
        };

        let material = BeamMaterial {
            youngs_modulus: parse_f64(bnode.attr("E").unwrap_or("200e9"), "beam.E")?,
            poisson_ratio: parse_f64(bnode.attr("nu").unwrap_or("0.3"), "beam.nu")?,
            density: parse_f64(bnode.attr("rho").unwrap_or("7850.0"), "beam.rho")?,
            rayleigh_alpha: solver_hints.rayleigh_alpha,
            rayleigh_beta: solver_hints.rayleigh_beta,
        };

        let local_a = match bnode.attr("clamp_a") {
            Some(s) => parse_vec3(s, "beam.clamp_a")?,
            None => Vec3::ZERO,
        };
        let world_a = bodies[body_a].rest_pose.transform_point(local_a);

        let (world_b, local_b_opt) = match body_b {
            Some(bb) => {
                let lb = match bnode.attr("clamp_b") {
                    Some(s) => parse_vec3(s, "beam.clamp_b")?,
                    None => Vec3::ZERO,
                };
                (bodies[bb].rest_pose.transform_point(lb), Some(lb))
            }
            None => {
                let len = parse_f64(bnode.attr("length").unwrap_or("0.06"), "beam.length")?;
                let dir = match bnode.attr("dir") {
                    Some(s) => parse_vec3(s, "beam.dir")?.normalized(),
                    None => Vec3::EX,
                };
                (world_a + len * bodies[body_a].rest_pose.transform_vector(dir), None)
            }
        };

        let bow = parse_f64(bnode.attr("bow").unwrap_or("0.0"), "beam.bow")?;
        let bow_dir = match bnode.attr("bow_dir") {
            Some(s) => parse_vec3(s, "beam.bow_dir")?.normalized(),
            None => Vec3::EZ,
        };

        let mut poses = Vec::with_capacity(n_elements + 1);
        for j in 0..=n_elements {
            let u = (j as f64) / (n_elements as f64);
            let p = world_a + u * (world_b - world_a) + (4.0 * bow * u * (1.0 - u)) * bow_dir;
            let tang = (world_b - world_a + (4.0 * bow * (1.0 - 2.0 * u)) * bow_dir).normalized();
            poses.push(Se3::new(p, frame_along_x(tang)));
        }
        let lengths: Vec<f64> = (0..n_elements)
            .map(|j| (poses[j + 1].pos - poses[j].pos).norm())
            .collect();

        let q_ca = bodies[body_a]
            .rest_pose
            .rot
            .conj()
            .mul(poses[0].rot)
            .normalized();
        let clamp_a = Se3::new(local_a, q_ca);
        let clamp_b = match (body_b, local_b_opt) {
            (Some(bb), Some(lb)) => {
                let q_cb = bodies[bb]
                    .rest_pose
                    .rot
                    .conj()
                    .mul(poses[n_elements].rot)
                    .normalized();
                Some(Se3::new(lb, q_cb))
            }
            _ => None,
        };

        let bid = beams.len();
        beams.push(BeamEdgeSpec {
            id: bid,
            name: bname,
            body_a,
            clamp_a,
            body_b,
            clamp_b,
            n_elements,
            formulation,
            section,
            material,
            thin_shell_compat: bnode
                .attr("thin_shell")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: lengths,
            element_transforms: None,
            interior_node_contact_radius: bnode
                .attr("contact_radius")
                .map(|v| parse_f64(v, "beam.contact_radius"))
                .transpose()?,
            enable_self_contact: bnode
                .attr("self_contact")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
        });
    }

    // 4. Parse FlexGraph <tendon> extensions
    let mut tendons = Vec::new();
    for tnode in root.children_named("tendon") {
        let tname = tnode.attr("name").unwrap_or("tendon").to_string();
        let actuator_id = match tnode.attr("actuator") {
            Some(a) => {
                let idx = a.parse::<usize>().map_err(|e| {
                    RobotLoadError::UrdfFlex(format!("invalid tendon actuator '{a}': {e}"))
                })?;
                next_actuator_id = next_actuator_id.max(idx + 1);
                idx
            }
            None => {
                let idx = next_actuator_id;
                next_actuator_id += 1;
                idx
            }
        };
        let wp_str = tnode
            .attr("waypoints")
            .ok_or_else(|| RobotLoadError::UrdfFlex(format!("tendon '{tname}' missing waypoints")))?;
        let mut waypoints = Vec::new();
        for seg in wp_str.split(';').filter(|s| !s.trim().is_empty()) {
            let (lname, coords) = seg.split_once(':').ok_or_else(|| {
                RobotLoadError::UrdfFlex(format!(
                    "tendon waypoint '{seg}' must use 'link_name:x,y,z' format"
                ))
            })?;
            let bid = resolve_body(lname.trim())?;
            let local_pos = parse_vec3(coords, "tendon.waypoint")?;
            waypoints.push(TendonWaypoint {
                body_id: bid,
                local_pos,
                wrap_angle_rad: 0.0,
            });
        }
        let drive_mode = match tnode.attr("mode").unwrap_or("linear") {
            "linear" => TendonDriveMode::Linear,
            "capstan" => TendonDriveMode::Capstan {
                drum_radius: parse_f64(
                    tnode.attr("drum_radius").unwrap_or("0.015"),
                    "tendon.drum_radius",
                )?,
            },
            "crank" => TendonDriveMode::EccentricCrank {
                horn_radius: parse_f64(
                    tnode.attr("horn_radius").unwrap_or("0.025"),
                    "tendon.horn_radius",
                )?,
                guide_dist: parse_f64(
                    tnode.attr("guide_dist").unwrap_or("0.036"),
                    "tendon.guide_dist",
                )?,
            },
            "legacy_wrap" => TendonDriveMode::LegacyWrap {
                horn_radius: parse_f64(
                    tnode.attr("horn_radius").unwrap_or("0.0235"),
                    "tendon.horn_radius",
                )?,
                guide_dist: parse_f64(
                    tnode.attr("guide_dist").unwrap_or("0.035"),
                    "tendon.guide_dist",
                )?,
            },
            other => {
                return Err(RobotLoadError::UrdfFlex(format!(
                    "unsupported tendon mode '{other}'"
                )));
            }
        };

        let capstan_mu_theta = parse_f64(
            tnode.attr("capstan_mu_theta").unwrap_or("0.0"),
            "tendon.capstan_mu_theta",
        )?;
        let stick_slip_enabled = match tnode.attr("stick_slip") {
            Some(v) => v == "true" || v == "1",
            None => capstan_mu_theta > 0.0,
        };
        let tid = tendons.len();
        tendons.push(TendonEdgeSpec {
            id: tid,
            name: tname,
            actuator_id,
            waypoints,
            drive_mode,
            rest_length: parse_f64(
                tnode.attr("rest_length").unwrap_or("0.0"),
                "tendon.rest_length",
            )?,
            slack: parse_f64(tnode.attr("slack").unwrap_or("0.0"), "tendon.slack")?,
            stiffness: parse_f64(
                tnode.attr("stiffness").unwrap_or("2000.0"),
                "tendon.stiffness",
            )?,
            damping: parse_f64(tnode.attr("damping").unwrap_or("1.0"), "tendon.damping")?,
            capstan_mu_theta,
            stick_slip_enabled,
            servo_torque_limit: parse_f64(
                tnode.attr("torque_limit").unwrap_or("2.0"),
                "tendon.torque_limit",
            )?,
            servo_speed_limit: parse_f64(
                tnode.attr("speed_limit").unwrap_or("2.0"),
                "tendon.speed_limit",
            )?,
            servo_deadband: 0.0,
            angle_limits: [0.0, std::f64::consts::PI],
            solref_imp: None,
        });
    }

    // 5. Parse FlexGraph <contact_site> extensions
    let mut contact_sites = Vec::new();
    for cnode in root.children_named("contact_site") {
        let cname = cnode.attr("name").unwrap_or("site").to_string();
        let body_id = resolve_body(
            cnode
                .attr("body")
                .ok_or_else(|| RobotLoadError::UrdfFlex(format!("contact_site '{cname}' missing body")))?,
        )?;
        let local_pos = match cnode.attr("pos") {
            Some(s) => parse_vec3(s, "contact_site.pos")?,
            None => Vec3::ZERO,
        };
        let radius = parse_f64(cnode.attr("radius").unwrap_or("0.012"), "contact_site.radius")?;
        let mode = match cnode.attr("mode").unwrap_or("isotropic_pad") {
            "isotropic_pad" => ContactMode::IsotropicPad {
                mu: parse_f64(cnode.attr("mu").unwrap_or("0.8"), "contact_site.mu")?,
                pivot_mu: 0.5,
                static_mu_ratio: 1.0,
                sdf_only: false,
            },
            "anisotropic_scale" => ContactMode::AnisotropicScale {
                mu_fwd: parse_f64(cnode.attr("mu_fwd").unwrap_or("0.15"), "mu_fwd")?,
                mu_rev: parse_f64(cnode.attr("mu_rev").unwrap_or("0.9"), "mu_rev")?,
                mu_lat: parse_f64(cnode.attr("mu_lat").unwrap_or("0.9"), "mu_lat")?,
                pivot_mu: 0.5,
                static_mu_ratio: 1.1,
                fwd_axis_local: Vec3::new(-1.0, 0.0, 0.0),
            },
            "ratchet_wheel" => ContactMode::RatchetWheel {
                wheel_mass: 0.009,
                rolling_resistance: 0.06,
                backlash_rad: 0.0,
                pivot_mu: 0.65,
                mu_fwd: parse_f64(cnode.attr("mu_fwd").unwrap_or("0.8"), "mu_fwd")?,
                mu_lat_ratio: 1.0,
                static_mu_ratio: 1.0,
                axle_axis_local: Vec3::EY,
            },
            other => {
                return Err(RobotLoadError::UrdfFlex(format!(
                    "unsupported contact_site mode '{other}'"
                )));
            }
        };
        let cid = contact_sites.len();
        contact_sites.push(ContactSiteSpec {
            id: cid,
            name: cname,
            attachment: ContactAttachment::RigidBody { body_id },
            local_pos,
            local_frame: Quat::IDENTITY,
            radius,
            patch_radius: 0.006,
            mode,
            material_profile_id: cnode
                .attr("mat_id")
                .unwrap_or("0")
                .parse::<usize>()
                .unwrap_or(0),
            solref_imp: None,
        });
    }

    let graph = RigidFlexGraph {
        name: robot_name,
        gravity,
        bodies,
        beams,
        tendons,
        joints,
        contact_sites,
        symmetry: SymmetryGroupSpec::trivial(),
        solver_hints,
        diff_params: Vec::new(),
        mesh_colliders: Vec::new(),
    };
    graph.validate()?;
    Ok(graph)
}

/// Load a `RigidFlexGraph` from a URDF-Flex XML file.
pub fn load_graph_from_urdf_flex_file(
    path: impl AsRef<Path>,
) -> Result<RigidFlexGraph, RobotLoadError> {
    let xml = std::fs::read_to_string(path.as_ref())?;
    load_graph_from_urdf_flex_str(&xml)
}
