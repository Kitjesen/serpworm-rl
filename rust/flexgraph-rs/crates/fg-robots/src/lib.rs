//! `fg-robots`: Declarative JSON / YAML / URDF-Flex / Standard ROS URDF (`urdf-rs`) /
//! MuJoCo MJCF robot loaders and 7 deeply calibrated rigid-flexible continuum robot
//! archetypes exercising all three `fg-ir` Schur routes:
//!
//! 1. `SerpWorm-N` (`N = 1, 2, 3, 5, 7`, `BlockDiagonalPairs`, `Z2`, `<1e-10` FP64 parity with `serpworm.GPUCore`)
//! 2. `BristleWorm` (wheelless anisotropic ventral-scale crawler, `BlockDiagonalPairs`, `Z2xZ2`)
//! 3. `TendonEndoscope-3D` (fixed-base nitinol catheter + 4 multi-waypoint tendons + optional cantilever tip, `TreeChain`, `D4`)
//! 4. `FlexSpine-Centipede` (parallel leaf-spring compliant spine + articulated revolute legs + Bekker feet, `TreeChain`, `Z2`)
//! 5. `Tensegrity-6Bar` (classic 6-strut icosahedral tensegrity ball with cyclic closed-loop graph $\beta_1 = 7$, `SparseGeneral`)
//! 6. `SoftOctopusArm-DER` (tapered 3D Cosserat/DER soft manipulator + 4 tapered tendons + analytical Euler/Elastica validation, `TreeChain`, `D4`)
//! 7. `RigidFlexGrid-4x4` / `6x6` (16-body / 24-beam $\beta_1 = 9$ and 36-body / 60-beam $\beta_1 = 25$ cyclic metamaterial lattice, `SparseGeneral`)

#![allow(clippy::needless_range_loop)]

pub mod bristle_worm;
pub mod centipede;
pub mod endoscope;
pub mod flex_grid;
pub mod loader;
pub mod mjcf;
pub mod octopus_arm;
#[cfg(feature = "ros-urdf")]
pub mod ros_urdf;
pub mod serpworm;
pub mod tensegrity;

pub use bristle_worm::{build_bristle_worm, BristleWormConfig};
pub use centipede::{build_flexspine_centipede, CentipedeConfig};
pub use endoscope::{
    build_endoscope_spline_tube_sdf, build_tendon_endoscope, default_endoscope_lumen_waypoints,
    EndoscopeConfig, EndoscopeLumenSpec,
};
pub use flex_grid::{
    build_rigid_flex_grid, build_rigid_flex_grid_4x4, build_rigid_flex_grid_6x6, FlexGridConfig,
};
pub use loader::{
    build_archetype_by_name, build_archetype_default_world, load_graph_from_json_file,
    load_graph_from_json_str, load_graph_from_path, load_graph_from_urdf_flex_file,
    load_graph_from_urdf_flex_str, load_graph_from_yaml_file, load_graph_from_yaml_str,
    RobotLoadError,
};
pub use mjcf::{load_graph_from_mjcf_file, load_graph_from_mjcf_str};
pub use octopus_arm::{
    build_euler_elastica_rod, build_soft_octopus_arm, cantilever_small_deflection,
    elastica_cantilever_tip, euler_buckling_critical_load, OctopusArmConfig,
};
#[cfg(feature = "ros-urdf")]
pub use ros_urdf::{
    graph_from_urdf_robot, load_graph_from_ros_urdf_file, load_graph_from_ros_urdf_str,
};
pub use serpworm::{
    build_serpworm, ParametersJson, SerpWormConfig, SerpWormPreset, WHEEL_DROP_M, WHEEL_HOLES_M,
    WHEEL_RADIUS_M,
};
pub use tensegrity::{
    build_tensegrity_6bar, tensegrity_6bar_cable_pairs, tensegrity_6bar_equilibrium_residual,
    TensegrityConfig,
};

#[cfg(test)]
mod tests;
