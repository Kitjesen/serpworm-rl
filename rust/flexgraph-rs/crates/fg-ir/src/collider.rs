//! Mesh and primitive collider geometry descriptors (`MeshColliderSpec`) for ROS URDF,
//! MuJoCo MJCF, and Parry3D narrowphase/BVH integration.

use fg_math::{Se3, SolRefImp, Vec3};
use serde::{Deserialize, Serialize};

fn default_unit_scale() -> Vec3 {
    Vec3::new(1.0, 1.0, 1.0)
}

fn default_identity_se3() -> Se3 {
    Se3::IDENTITY
}

fn default_collision_mask() -> u32 {
    u32::MAX
}

/// Collision shape geometry compatible with URDF, MJCF, and `parry3d-f64`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum ColliderGeometry {
    /// Indexed triangle mesh (`parry3d_f64::shape::TriMesh`).
    TriMesh {
        vertices: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
    },
    /// Convex hull point cloud (`parry3d_f64::shape::ConvexPolyhedron`).
    ConvexHull { points: Vec<Vec3> },
    /// Sphere of given radius.
    Sphere { radius: f64 },
    /// Box with half-extents `[hx, hy, hz]`.
    Box { half_extents: Vec3 },
    /// Cylinder aligned along local Z axis (URDF / MJCF convention).
    Cylinder { radius: f64, half_height: f64 },
    /// Capsule aligned along local Z axis (`radius` + cylindrical `half_height`).
    Capsule { radius: f64, half_height: f64 },
    /// External mesh file URI (STL / OBJ / DAE from URDF/MJCF) with optional scale.
    ExternalMesh {
        uri: String,
        #[serde(default = "default_unit_scale")]
        scale: Vec3,
    },
}

/// Rigid-body or static-environment mesh/primitive collider descriptor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshColliderSpec {
    pub id: usize,
    pub name: String,
    /// Owning rigid body index (`None` => static environment collider in world coordinates).
    #[serde(default)]
    pub body_id: Option<usize>,
    /// Pose relative to owning body (or world pose when `body_id == None`).
    #[serde(default = "default_identity_se3")]
    pub local_pose: Se3,
    pub geometry: ColliderGeometry,
    /// Material profile ID into `fg-sdf` `MaterialLibrary`.
    #[serde(default)]
    pub material_profile_id: usize,
    /// Optional MuJoCo-compatible constraint impedance & stiffness parameters.
    #[serde(default)]
    pub solref_imp: Option<SolRefImp>,
    /// Contact margin / inflation thickness (m) for IPC / Parry3D.
    #[serde(default)]
    pub margin: f64,
    /// Collision filter membership bitmask.
    #[serde(default = "default_collision_mask")]
    pub collision_group: u32,
    /// Collision filter interaction bitmask.
    #[serde(default = "default_collision_mask")]
    pub collision_mask: u32,
}

impl Default for MeshColliderSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            body_id: None,
            local_pose: Se3::IDENTITY,
            geometry: ColliderGeometry::Sphere { radius: 0.01 },
            material_profile_id: 0,
            solref_imp: None,
            margin: 0.0,
            collision_group: u32::MAX,
            collision_mask: u32::MAX,
        }
    }
}
