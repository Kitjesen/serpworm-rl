//! `fg-ir`: Declarative `RigidFlexGraph` intermediate representation, serde JSON/YAML I/O,
//! finite symmetry group specifications ($\mathbb{Z}_2, \mathbb{Z}_2\times\mathbb{Z}_2, C_4, D_4$),
//! and compile-time graph sparsity / elimination-tree analyzer for `flexgraph-rs`.

pub mod analysis;
pub mod collider;
pub mod diff;
pub mod schema;
pub mod symmetry;

pub use analysis::{
    analyze_graph, BeamDofMap, GraphTopologyAnalysis, NodeIndexing, SchurRoute, SparseBlockPattern,
};
pub use collider::{ColliderGeometry, MeshColliderSpec};
pub use diff::{DiffParamKind, DiffParamSpec};
pub use fg_math::SolRefImp;
pub use schema::{
    BeamEdgeSpec, BeamFormulation, ContactAttachment, ContactMode, ContactSiteSpec, IrError,
    JointEdgeSpec, JointType, RigidBodySpec, RigidFlexGraph, SolverHints, TendonDriveMode,
    TendonEdgeSpec, TendonWaypoint,
};
pub use symmetry::{SymmetryElement, SymmetryGroupKind, SymmetryGroupSpec};
