//! Core [`FlexGraphWorld`] simulation engine unifying:
//! - Dual continuum formulations (Corotational Euler-Bernoulli/Timoshenko + DER/Cosserat Rod)
//! - Bipartite Direct Solver (`BlockDiagonalPairs` / `TreeChain` / `SparseGeneral` Symbolic Block-$LDL^\top$)
//! - Tri-Engine Contact Mechanics (ARPC / GPUCore Parity, Delassus-Schur LCP, IPC/Py-DiSMech Barrier + ACCD)
//! - Implicit Lie-Group Time Integration (`LinearlyImplicitEuler`, `DampedNewtonEuler`, `NewmarkBeta`, `Bdf2`)
//! - Multi-environment Rayon parallel stepping.

use crate::actuation::{
    advance_tendon_servos, evaluate_actuation_with_masses, ActuationState,
};
use crate::continuum::{CorotationalBeamData, DerBeamData, DerBeamState};
use crate::integrator::{
    BeamFactorKind, ContactBackend, EnergyMomentumReport, FactoredBipartiteSystem, IntegratorKind,
    SolverConfig, SolverError, StepDiagnostics, WorldState,
};
#[cfg(feature = "faer-sparse")]
use crate::schur::FaerSparsePattern;
use crate::schur::{
    BeamSchurContribution, CorotationalBeamFactor, LinearSolveError, RigidSchurFactor,
    RigidSchurMatrix,
};
use fg_contact::{
    accd_point_sdf, compute_safe_step_segments, compute_safe_step_segments_tight_inclusion,
    eval_arpc_site, eval_arpc_site_with_mass, eval_sdf_surface_contact, implicit_wheel_contact,
    ipc_mollifier_friction_eval, ipc_point_sdf_barrier, ArpcSiteState, DelassusSystem,
    LcpContactConstraint, LcpSolverConfig, TightInclusionConfig,
};
#[cfg(feature = "parry-mesh")]
use fg_contact::tight_inclusion_point_triangle_ccd;
use fg_ir::{
    BeamFormulation, ColliderGeometry, ContactAttachment, ContactMode, GraphTopologyAnalysis,
    RigidFlexGraph,
};
use fg_math::{so3_dexp_inv, Mat12, Mat3, Mat6, Quat, Se3, SolRefImp, Vec12, Vec3, Vec6};
#[cfg(feature = "parry-mesh")]
use fg_sdf::{Aabb, ConvexHullSdf, MeshSignMode, TriMeshSdf};
use fg_sdf::{BoxSdf, CapsuleSdf, CylinderSdf, SdfField, SphereSdf, WorldSdf};
use rayon::prelude::*;
use std::sync::Arc;

/// Precomputed per-beam continuum data.
#[derive(Clone, Debug)]
pub(crate) enum BeamRuntime {
    Corotational(Box<CorotationalBeamData>),
    Der(Box<DerBeamData>),
}

/// Output of one linearized Newton tangent solve: `(delta_xi_nodes, delta_der_thetas)`.
type NewtonLinearStep = (Vec<Vec6>, Vec<Option<Vec<f64>>>);

/// Per-beam DER twist kinematics tuple `(dtheta, theta_vel, theta_accel)`.
type DerKinematics = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Evaluated contact forces & tangent matrices for one step or Newton residual evaluation.
pub(crate) struct ContactEvalBundle {
    pub(crate) body_wrenches: Vec<Vec6>,
    pub(crate) body_k_tan: Vec<Mat6>,
    /// Per-beam interior node forces (indexed `[beam_idx][free_node_local_idx]`).
    pub(crate) beam_int_forces: Vec<Vec<Vec3>>,
    /// Per-beam interior node 3x3 SPD contact stiffness blocks.
    pub(crate) beam_int_k: Vec<Vec<Mat3>>,
    /// Proposed updated per-site ARPC states (committed only after step/Newton convergence).
    pub(crate) next_site_states: Vec<ArpcSiteState>,
    pub(crate) site_normal_forces: Vec<f64>,
    pub(crate) site_slips: Vec<[f64; 2]>,
    pub(crate) active_site_contacts: usize,
    pub(crate) active_beam_contacts: usize,
    pub(crate) active_self_contacts: usize,
    pub(crate) _potential_energy: f64,
}

/// Complete rigid-flexible continuum simulation world.
pub struct FlexGraphWorld {
    pub graph: RigidFlexGraph,
    pub analysis: GraphTopologyAnalysis,
    pub sdf: WorldSdf,
    pub config: SolverConfig,
    pub state: WorldState,
    pub(crate) rest_poses: Vec<Se3>,
    pub(crate) beams: Vec<BeamRuntime>,
    /// Precomputed segment pairs `(node_u, node_v)` across all beams with `enable_self_contact == true`.
    self_contact_segments: Vec<(usize, usize)>,
    /// Sorted unique tendon actuator IDs present in `graph.tendons`.
    pub(crate) tendon_actuator_ids: Vec<usize>,
    /// Sorted unique joint actuator IDs present in `graph.joints`.
    pub(crate) joint_actuator_ids: Vec<usize>,
    /// Cached AMD + supernodal symbolic Cholesky pattern for `faer` sparse $LDL^\top$.
    #[cfg(feature = "faer-sparse")]
    faer_pattern: Option<FaerSparsePattern>,
    /// Static triangle meshes registered for Tight Inclusion Point-Triangle CCD.
    #[cfg(feature = "parry-mesh")]
    static_trimeshes: Vec<Arc<TriMeshSdf>>,
}

impl FlexGraphWorld {
    /// Validate and compile a [`RigidFlexGraph`] with a 3D [`WorldSdf`] and [`SolverConfig`].
    ///
    /// Automatically mounts any static environment colliders (`mesh_colliders` with `body_id == None`)
    /// into `sdf` (including `TriMeshSdf` and `ConvexHullSdf` when compiled with `parry-mesh`).
    pub fn new(
        graph: RigidFlexGraph,
        mut sdf: WorldSdf,
        config: SolverConfig,
    ) -> Result<Self, SolverError> {
        let analysis = graph
            .analyze()
            .map_err(|e| SolverError::Ir(e.to_string()))?;

        // Mount static environment colliders from `graph.mesh_colliders` (`body_id == None`) into `WorldSdf`
        let mut mounted_static = false;
        #[cfg(feature = "parry-mesh")]
        let mut static_trimeshes: Vec<Arc<TriMeshSdf>> = Vec::new();
        for col in &graph.mesh_colliders {
            if col.body_id.is_some() {
                continue;
            }
            match &col.geometry {
                ColliderGeometry::Sphere { radius } => {
                    sdf.add_bounded(Arc::new(SphereSdf::new(
                        col.local_pose.pos,
                        *radius + col.margin,
                        col.material_profile_id,
                    )));
                    mounted_static = true;
                }
                ColliderGeometry::Box { half_extents } => {
                    sdf.add_bounded(Arc::new(BoxSdf::new(
                        col.local_pose,
                        *half_extents,
                        col.margin,
                        col.material_profile_id,
                    )));
                    mounted_static = true;
                }
                ColliderGeometry::Cylinder {
                    radius,
                    half_height,
                } => {
                    sdf.add_bounded(Arc::new(CylinderSdf::new(
                        col.local_pose,
                        *radius + col.margin,
                        Some(*half_height),
                        col.material_profile_id,
                        false,
                    )));
                    mounted_static = true;
                }
                ColliderGeometry::Capsule {
                    radius,
                    half_height,
                } => {
                    let a = col
                        .local_pose
                        .transform_point(Vec3::new(0.0, 0.0, -half_height));
                    let b = col
                        .local_pose
                        .transform_point(Vec3::new(0.0, 0.0, *half_height));
                    sdf.add_bounded(Arc::new(CapsuleSdf::new(
                        a,
                        b,
                        *radius + col.margin,
                        col.material_profile_id,
                    )));
                    mounted_static = true;
                }
                #[cfg(feature = "parry-mesh")]
                ColliderGeometry::TriMesh { vertices, indices } => {
                    let mesh_sdf = Arc::new(
                        TriMeshSdf::new(
                            vertices.clone(),
                            indices.clone(),
                            col.local_pose,
                            MeshSignMode::ClosedSolid,
                            col.material_profile_id,
                        )
                        .map_err(SolverError::Ir)?,
                    );
                    sdf.add_bounded(Arc::clone(&mesh_sdf) as Arc<dyn SdfField>);
                    static_trimeshes.push(mesh_sdf);
                    mounted_static = true;
                }
                #[cfg(feature = "parry-mesh")]
                ColliderGeometry::ConvexHull { points } => {
                    let hull_sdf = ConvexHullSdf::from_points(
                        points,
                        col.local_pose,
                        col.material_profile_id,
                        false,
                    )
                    .map_err(SolverError::Ir)?;
                    sdf.add_bounded(Arc::new(hull_sdf));
                    mounted_static = true;
                }
                #[cfg(feature = "parry-mesh")]
                ColliderGeometry::ExternalMesh { uri, scale } => {
                    if uri.to_ascii_lowercase().ends_with(".obj") {
                        if let Ok(obj_text) = std::fs::read_to_string(uri) {
                            let mesh_sdf = Arc::new(
                                TriMeshSdf::from_obj_str(
                                    &obj_text,
                                    *scale,
                                    col.local_pose,
                                    MeshSignMode::ClosedSolid,
                                    col.material_profile_id,
                                )
                                .map_err(SolverError::Ir)?,
                            );
                            sdf.add_bounded(Arc::clone(&mesh_sdf) as Arc<dyn SdfField>);
                            static_trimeshes.push(mesh_sdf);
                            mounted_static = true;
                        }
                    }
                }
                #[cfg(not(feature = "parry-mesh"))]
                ColliderGeometry::TriMesh { .. }
                | ColliderGeometry::ConvexHull { .. }
                | ColliderGeometry::ExternalMesh { .. } => {}
            }
        }
        if mounted_static {
            sdf.rebuild_bvh();
        }
        let total_nodes = analysis.indexing.total_nodes;

        let mut rest_poses = vec![Se3::IDENTITY; total_nodes];
        for (bi, body) in graph.bodies.iter().enumerate() {
            rest_poses[bi] = body.rest_pose;
        }
        for (s, beam) in graph.beams.iter().enumerate() {
            let map = &analysis.indexing.beam_maps[s];
            for (local_i, &nid) in map.free_node_ids.iter().enumerate() {
                rest_poses[nid] = beam.centerline_rest_poses[local_i + 1];
            }
        }

        let stable_log_rest = config.use_strain_jacobian;
        let mut beams = Vec::with_capacity(graph.beams.len());
        let mut der_states = Vec::with_capacity(graph.beams.len());
        let mut der_theta_accels = Vec::with_capacity(graph.beams.len());
        let mut prev_der_dtheta = Vec::with_capacity(graph.beams.len());
        let mut prev_der_theta_vel = Vec::with_capacity(graph.beams.len());
        let mut self_contact_segments = Vec::new();

        for (s, spec) in graph.beams.iter().enumerate() {
            let map = &analysis.indexing.beam_maps[s];
            if spec.enable_self_contact {
                for e in 0..spec.n_elements {
                    self_contact_segments.push((map.chain_node_ids[e], map.chain_node_ids[e + 1]));
                }
            }
            match spec.formulation {
                BeamFormulation::CorotationalEulerBernoulli
                | BeamFormulation::CorotationalTimoshenko => {
                    let data = CorotationalBeamData::from_spec(
                        spec,
                        map,
                        &rest_poses,
                        stable_log_rest,
                    );
                    beams.push(BeamRuntime::Corotational(Box::new(data)));
                    der_states.push(None);
                    der_theta_accels.push(Vec::new());
                    prev_der_dtheta.push(Vec::new());
                    prev_der_theta_vel.push(Vec::new());
                }
                BeamFormulation::DiscreteElasticRod => {
                    let (data, st) = DerBeamData::from_spec(spec);
                    der_theta_accels.push(vec![0.0; spec.n_elements]);
                    prev_der_dtheta.push(vec![0.0; spec.n_elements]);
                    prev_der_theta_vel.push(vec![0.0; spec.n_elements]);
                    beams.push(BeamRuntime::Der(Box::new(data)));
                    der_states.push(Some(st));
                }
            }
        }

        let actuation = ActuationState::from_specs(&graph.tendons, &graph.joints, &rest_poses);
        let n_sites = graph.contact_sites.len();
        let prev_positions: Vec<Vec3> = rest_poses.iter().map(|p| p.pos).collect();

        let state = WorldState {
            time: 0.0,
            step_count: 0,
            poses: rest_poses.clone(),
            vels: vec![Vec6::ZERO; total_nodes],
            accels: vec![Vec6::ZERO; total_nodes],
            prev_dx: vec![Vec6::ZERO; total_nodes],
            prev_vels: vec![Vec6::ZERO; total_nodes],
            actuation,
            contact_states: vec![ArpcSiteState::default(); n_sites],
            contact_normal_forces: vec![0.0; n_sites],
            contact_slips: vec![[0.0; 2]; n_sites],
            der_states,
            der_theta_accels,
            prev_der_dtheta,
            prev_der_theta_vel,
            prev_positions,
            lcp_warm_start: Vec::new(),
            failed: false,
        };

        let mut tendon_actuator_ids: Vec<usize> =
            graph.tendons.iter().map(|t| t.actuator_id).collect();
        tendon_actuator_ids.sort_unstable();
        tendon_actuator_ids.dedup();
        let mut joint_actuator_ids: Vec<usize> =
            graph.joints.iter().filter_map(|j| j.actuator_id).collect();
        joint_actuator_ids.sort_unstable();
        joint_actuator_ids.dedup();

        #[cfg(feature = "faer-sparse")]
        let faer_pattern = if analysis.route == fg_ir::SchurRoute::SparseGeneral
            && analysis.num_free_bodies > 0
        {
            FaerSparsePattern::from_graph(&graph, &analysis).ok()
        } else {
            None
        };

        Ok(Self {
            graph,
            analysis,
            sdf,
            config,
            state,
            rest_poses,
            beams,
            self_contact_segments,
            tendon_actuator_ids,
            joint_actuator_ids,
            #[cfg(feature = "faer-sparse")]
            faer_pattern,
            #[cfg(feature = "parry-mesh")]
            static_trimeshes,
        })
    }

    /// Mount an external static [`TriMeshSdf`] (e.g. `corrugated_pipe_mesh` or `rubble_terrain_mesh`)
    /// into both `self.sdf` and the Tight Inclusion Point-Triangle CCD mesh registry.
    #[cfg(feature = "parry-mesh")]
    pub fn register_static_trimesh(&mut self, mesh: Arc<TriMeshSdf>) {
        if mesh.bounds().is_some() {
            self.sdf.add_bounded(Arc::clone(&mesh) as Arc<dyn SdfField>);
            self.sdf.rebuild_bvh();
        } else {
            self.sdf
                .add_unbounded(Arc::clone(&mesh) as Arc<dyn SdfField>);
        }
        self.static_trimeshes.push(mesh);
    }

    /// Rebuild cached beam constitutive matrices (`CorotationalBeamData` / `DerBeamData`)
    /// after mutating `graph.beams[s].material` (used by differentiable SysID / adjoints).
    pub fn rebuild_beam_data(&mut self) {
        let stable_log_rest = self.config.use_strain_jacobian;
        for (s, spec) in self.graph.beams.iter().enumerate() {
            let map = &self.analysis.indexing.beam_maps[s];
            match spec.formulation {
                BeamFormulation::CorotationalEulerBernoulli
                | BeamFormulation::CorotationalTimoshenko => {
                    let data = CorotationalBeamData::from_spec(
                        spec,
                        map,
                        &self.rest_poses,
                        stable_log_rest,
                    );
                    self.beams[s] = BeamRuntime::Corotational(Box::new(data));
                }
                BeamFormulation::DiscreteElasticRod => {
                    let (data, _) = DerBeamData::from_spec(spec);
                    self.beams[s] = BeamRuntime::Der(Box::new(data));
                }
            }
        }
    }

    /// Factor `rigid_mat` using the configured sparse backend (`faer` supernodal or custom Block-CSC).
    #[inline]
    pub(crate) fn factorize_rigid_schur(
        &self,
        rigid_mat: &RigidSchurMatrix,
    ) -> Result<RigidSchurFactor, LinearSolveError> {
        #[cfg(feature = "faer-sparse")]
        {
            RigidSchurFactor::factorize_with_backend(
                rigid_mat,
                &self.analysis,
                self.config.sparse_backend,
                self.faer_pattern.as_ref(),
            )
        }
        #[cfg(not(feature = "faer-sparse"))]
        {
            RigidSchurFactor::factorize_with_backend(
                rigid_mat,
                &self.analysis,
                self.config.sparse_backend,
            )
        }
    }

    /// Reset world state cleanly to its initial rest configuration and zero velocities.
    pub fn reset(&mut self) {
        let total = self.analysis.indexing.total_nodes;
        self.state.time = 0.0;
        self.state.step_count = 0;
        self.state.poses.clone_from(&self.rest_poses);
        self.state.vels.fill(Vec6::ZERO);
        self.state.accels.fill(Vec6::ZERO);
        self.state.prev_dx.fill(Vec6::ZERO);
        self.state.prev_vels.fill(Vec6::ZERO);
        self.state.actuation = ActuationState::from_specs(
            &self.graph.tendons,
            &self.graph.joints,
            &self.rest_poses,
        );
        for cs in &mut self.state.contact_states {
            *cs = ArpcSiteState::default();
        }
        self.state.contact_normal_forces.fill(0.0);
        self.state.contact_slips.fill([0.0; 2]);
        for (s, spec) in self.graph.beams.iter().enumerate() {
            if matches!(spec.formulation, BeamFormulation::DiscreteElasticRod) {
                let (_, st) = DerBeamData::from_spec(spec);
                self.state.der_states[s] = Some(st);
                self.state.der_theta_accels[s].fill(0.0);
                self.state.prev_der_dtheta[s].fill(0.0);
                self.state.prev_der_theta_vel[s].fill(0.0);
            }
        }
        self.state.prev_positions = self.rest_poses.iter().map(|p| p.pos).collect();
        self.state.lcp_warm_start.clear();
        self.state.failed = false;
        debug_assert_eq!(self.state.poses.len(), total);
    }

    /// Set tendon servo actuator targets (maps packed slice `[t_0, ...]` across distinct tendon
    /// actuator IDs in ascending order, or direct index if matching `tendon_servo_target.len()`).
    pub fn set_tendon_targets(&mut self, targets: &[f64]) {
        if targets.len() == self.tendon_actuator_ids.len() {
            for (&aid, &src) in self.tendon_actuator_ids.iter().zip(targets.iter()) {
                self.state.actuation.tendon_servo_target[aid] = src;
            }
        } else {
            for (dst, &src) in self
                .state
                .actuation
                .tendon_servo_target
                .iter_mut()
                .zip(targets.iter())
            {
                *dst = src;
            }
        }
    }

    /// Set articulated joint actuator targets. Supports both packed slices matching the count of
    /// distinct actuated joints (e.g. 4 yaw targets for `SerpWorm-5` whose `actuator_id`s are
    /// `10..14`) and full-length indexed slices.
    pub fn set_joint_targets(&mut self, targets: &[f64]) {
        if targets.len() == self.joint_actuator_ids.len() {
            for (&aid, &src) in self.joint_actuator_ids.iter().zip(targets.iter()) {
                self.state.actuation.joint_targets[aid] = src;
            }
        } else {
            for (dst, &src) in self
                .state
                .actuation
                .joint_targets
                .iter_mut()
                .zip(targets.iter())
            {
                *dst = src;
            }
        }
    }

    /// Set unified targets `[tendon_targets..., joint_targets...]` (convenience for FFI/RL).
    /// Seamlessly handles both global actuator IDs (e.g. `SerpWorm`: tendons `0..2N`, yaw `2N..3N-1`)
    /// and separate 0-based tendon/joint IDs.
    pub fn set_all_targets(&mut self, targets: &[f64]) {
        let nt = self.tendon_actuator_ids.len();
        let nj = self.joint_actuator_ids.len();
        let split = nt.min(targets.len());
        for (&aid, &src) in self.tendon_actuator_ids.iter().zip(targets[..split].iter()) {
            self.state.actuation.tendon_servo_target[aid] = src;
        }
        if targets.len() > nt {
            let rem = &targets[nt..];
            let j_cnt = nj.min(rem.len());
            for (&aid, &src) in self.joint_actuator_ids.iter().zip(rem[..j_cnt].iter()) {
                self.state.actuation.joint_targets[aid] = src;
            }
        }
    }

    /// Number of distinct tendon servo channels.
    pub fn num_tendon_actuators(&self) -> usize {
        self.tendon_actuator_ids.len()
    }

    /// Number of distinct articulated joint actuator channels.
    pub fn num_joint_actuators(&self) -> usize {
        self.joint_actuator_ids.len()
    }

    /// Total number of distinct actuator channels (`num_tendon_actuators + num_joint_actuators`).
    pub fn num_total_actuators(&self) -> usize {
        self.tendon_actuator_ids.len() + self.joint_actuator_ids.len()
    }

    /// Total number of nodes (`num_bodies + num_beam_nodes`).
    pub fn total_nodes(&self) -> usize {
        self.analysis.indexing.total_nodes
    }

    /// Export all node poses as flat `[px, py, pz, qx, qy, qz, qw]` (`len = 7 * total_nodes`).
    pub fn poses_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.state.poses.len() * 7);
        for p in &self.state.poses {
            out.extend_from_slice(&p.to_array7());
        }
        out
    }

    /// Export all node velocities as flat `[vx, vy, vz, wx, wy, wz]` (`len = 6 * total_nodes`).
    pub fn vels_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.state.vels.len() * 6);
        for v in &self.state.vels {
            out.extend_from_slice(&v.0);
        }
        out
    }

    /// Advance the world by one time step `dt`.
    pub fn step(&mut self, dt: f64) -> Result<StepDiagnostics, SolverError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SolverError::InvalidDt(dt));
        }
        match self.config.integrator {
            IntegratorKind::LinearlyImplicitEuler => self.step_linearly_implicit_euler(dt),
            IntegratorKind::DampedNewtonEuler
            | IntegratorKind::NewmarkBeta { .. }
            | IntegratorKind::Bdf2 => self.step_damped_newton(dt),
        }
    }

    /// Advance the world by `n_steps` consecutive time steps, returning the final step diagnostics.
    pub fn step_n(&mut self, n_steps: usize, dt: f64) -> Result<StepDiagnostics, SolverError> {
        let mut last = StepDiagnostics::default();
        for _ in 0..n_steps {
            last = self.step(dt)?;
        }
        Ok(last)
    }

    /// Step a batch of independent `FlexGraphWorld` environments in parallel across CPU cores via Rayon.
    pub fn step_batch_rayon(
        worlds: &mut [FlexGraphWorld],
        dt: f64,
    ) -> Result<Vec<StepDiagnostics>, SolverError> {
        worlds.par_iter_mut().map(|w| w.step(dt)).collect()
    }

    /// Step a batch of `FlexGraphWorld` environments for `n_steps` each in parallel via Rayon.
    pub fn step_batch_n_rayon(
        worlds: &mut [FlexGraphWorld],
        n_steps: usize,
        dt: f64,
    ) -> Result<Vec<StepDiagnostics>, SolverError> {
        worlds
            .par_iter_mut()
            .map(|w| w.step_n(n_steps, dt))
            .collect()
    }

    /// Rigid body `6x6` inertia matrix at pose `q` (world-fixed or rotating per `config.world_fixed_inertia`).
    #[inline]
    pub(crate) fn body_inertia_mat6(&self, body_id: usize, q: Quat) -> Mat6 {
        let si = self.graph.bodies[body_id].spatial_inertia();
        if self.config.world_fixed_inertia {
            si.to_mat6()
        } else {
            si.world_mat6(q)
        }
    }

    /// Packed `(mass, is_fixed)` per rigid body for MuJoCo `SolRefImp` reduced-mass scaling.
    #[inline]
    pub(crate) fn body_masses(&self) -> Vec<(f64, bool)> {
        self.graph
            .bodies
            .iter()
            .map(|b| (b.mass, b.is_fixed))
            .collect()
    }

    /// Rigid-body gyroscopic / Coriolis torque wrench `omega x (I_w omega)` in world coordinates.
    #[inline]
    pub(crate) fn body_gyro_wrench(&self, body_id: usize, q: Quat, vel: Vec6) -> Vec6 {
        if self.config.world_fixed_inertia {
            return Vec6::ZERO;
        }
        let m6 = self.graph.bodies[body_id].spatial_inertia().world_mat6(q);
        let iw = m6.block(1, 1);
        let w = vel.ang();
        let tau_gyro = w.cross(iw.mul_vec(w));
        Vec6::from_parts(Vec3::ZERO, tau_gyro)
    }

    /// Evaluate all active contacts (sites + beam interior SDF + optional self-contact).
    pub(crate) fn evaluate_contacts(
        &self,
        poses: &[Se3],
        vels: &[Vec6],
        dt: f64,
        k_scale_barrier: f64,
        c_scale_fric: f64,
    ) -> ContactEvalBundle {
        let nb = self.graph.bodies.len();
        let n_sites = self.graph.contact_sites.len();
        let mut body_wrenches = vec![Vec6::ZERO; nb];
        let mut body_k_tan = vec![Mat6::ZERO; nb];
        let mut beam_int_forces: Vec<Vec<Vec3>> = self
            .analysis
            .indexing
            .beam_maps
            .iter()
            .map(|m| vec![Vec3::ZERO; m.free_node_ids.len()])
            .collect();
        let mut beam_int_k: Vec<Vec<Mat3>> = self
            .analysis
            .indexing
            .beam_maps
            .iter()
            .map(|m| vec![Mat3::ZERO; m.free_node_ids.len()])
            .collect();

        let mut next_site_states = self.state.contact_states.clone();
        let mut site_normal_forces = vec![0.0; n_sites];
        let mut site_slips = vec![[0.0; 2]; n_sites];
        let mut active_site_contacts = 0usize;
        let mut active_beam_contacts = 0usize;
        let mut active_self_contacts = 0usize;
        let mut potential_energy = 0.0;

        // --- 1. Contact Sites (RigidBody or BeamInteriorNode) ---
        match &self.config.contact {
            ContactBackend::None | ContactBackend::DelassusLcp { .. } => {
                // Handled separately (none or post-solve Delassus LCP).
            }
            ContactBackend::Arpc { implicit } => {
                for (si, site) in self.graph.contact_sites.iter().enumerate() {
                    match site.attachment {
                        ContactAttachment::RigidBody { body_id } => {
                            let ev = if site.solref_imp.is_some() {
                                eval_arpc_site_with_mass(
                                    site,
                                    poses[body_id],
                                    vels[body_id],
                                    &self.sdf,
                                    &self.sdf.materials,
                                    &self.state.contact_states[si],
                                    dt,
                                    *implicit,
                                    self.graph.bodies[body_id].mass,
                                )
                            } else {
                                eval_arpc_site(
                                    site,
                                    poses[body_id],
                                    vels[body_id],
                                    &self.sdf,
                                    &self.sdf.materials,
                                    &self.state.contact_states[si],
                                    dt,
                                    *implicit,
                                )
                            };
                            next_site_states[si] = ev.next_state;
                            site_normal_forces[si] = ev.normal_force;
                            site_slips[si] = ev.slip_2d;
                            if ev.active {
                                active_site_contacts += 1;
                                body_wrenches[body_id] += ev.wrench;
                                if *implicit {
                                    body_k_tan[body_id] += ev.k_tan;
                                }
                            }
                        }
                        ContactAttachment::BeamInteriorNode {
                            beam_id,
                            node_local_idx,
                        } => {
                            let nid = self.analysis.indexing.beam_maps[beam_id].free_node_ids
                                [node_local_idx];
                            let pos = poses[nid].transform_point(site.local_pos);
                            let vel = vels[nid].lin();
                            let ev = eval_sdf_surface_contact(
                                pos,
                                vel,
                                site.radius,
                                &self.sdf,
                                &self.sdf.materials,
                                dt,
                                *implicit,
                                1.0,
                            );
                            site_normal_forces[si] = ev.normal_force;
                            if ev.active {
                                active_site_contacts += 1;
                                beam_int_forces[beam_id][node_local_idx] += ev.force_world;
                                if *implicit {
                                    beam_int_k[beam_id][node_local_idx] += ev.k_tan_3x3;
                                }
                            }
                        }
                    }
                }
            }
            ContactBackend::GpuCoreKelvinParity { kn, cn } => {
                for (si, site) in self.graph.contact_sites.iter().enumerate() {
                    if let ContactAttachment::RigidBody { body_id } = site.attachment {
                        if let ContactMode::RatchetWheel {
                            wheel_mass,
                            rolling_resistance,
                            mu_fwd,
                            axle_axis_local,
                            ..
                        } = site.mode
                        {
                            let pose = poses[body_id];
                            let vel = vels[body_id];
                            let axle = pose.rot.rotate(axle_axis_local);
                            let up0 = Vec3::EZ;
                            let rim = -site.radius * (up0 - axle * axle.z()).normalized_or(Vec3::EZ);
                            let offset = pose.rot.rotate(site.local_pos);
                            let arm = offset + rim;
                            let contact_pos = pose.pos + arm;
                            let velocity = vel.lin() + vel.ang().cross(arm);

                            let q_sdf = self.sdf.query(contact_pos);
                            let up_g = q_sdf.normal.normalized_or(Vec3::EZ);
                            let penetration = -q_sdf.phi;
                            let forward = up_g.cross(axle).normalized_or(Vec3::EX);
                            let lateral = up_g.cross(forward);
                            let vn = velocity.dot(up_g);
                            let tv = [velocity.dot(forward), velocity.dot(lateral)];
                            let prev_st = self.state.contact_states[si];
                            let mem_2d = [
                                prev_st.memory_world.dot(forward),
                                prev_st.memory_world.dot(lateral),
                            ];

                            let active = penetration > 0.0;
                            let normal = if active {
                                (kn * penetration - cn * vn).max(0.0)
                            } else {
                                0.0
                            };
                            let eval = implicit_wheel_contact(
                                tv,
                                prev_st.spin,
                                mem_2d,
                                normal,
                                dt,
                                site.radius,
                                mu_fwd,
                                rolling_resistance,
                                wheel_mass,
                            );
                            let inertia = 0.5 * wheel_mass * site.radius * site.radius;
                            let next_spin = if active {
                                eval.omega
                            } else {
                                prev_st.spin * (1.0 - dt * 2.0e-6 / inertia).max(0.0)
                            };
                            next_site_states[si] = ArpcSiteState {
                                spin: next_spin,
                                wheel_angle: prev_st.wheel_angle + dt * next_spin,
                                memory_world: if active {
                                    forward * eval.memory[0] + lateral * eval.memory[1]
                                } else {
                                    Vec3::ZERO
                                },
                                ratchet_play: 0.0,
                                soil_sinkage: 0.0,
                            };
                            site_normal_forces[si] = normal;
                            site_slips[si] = eval.slip;

                            if active {
                                active_site_contacts += 1;
                                let force =
                                    forward * eval.force[0] + lateral * eval.force[1] + up_g * normal;
                                let rt = rim.cross(force);
                                let torque = offset.cross(force) + rt
                                    - axle * rt.dot(axle)
                                    + axle * eval.reaction;
                                body_wrenches[body_id] += Vec6::from_parts(force, torque);
                            }
                            continue;
                        }
                    }
                    // Non-wheel sites fall back to explicit ARPC
                    if let ContactAttachment::RigidBody { body_id } = site.attachment {
                        let ev = if site.solref_imp.is_some() {
                            eval_arpc_site_with_mass(
                                site,
                                poses[body_id],
                                vels[body_id],
                                &self.sdf,
                                &self.sdf.materials,
                                &self.state.contact_states[si],
                                dt,
                                false,
                                self.graph.bodies[body_id].mass,
                            )
                        } else {
                            eval_arpc_site(
                                site,
                                poses[body_id],
                                vels[body_id],
                                &self.sdf,
                                &self.sdf.materials,
                                &self.state.contact_states[si],
                                dt,
                                false,
                            )
                        };
                        next_site_states[si] = ev.next_state;
                        site_normal_forces[si] = ev.normal_force;
                        site_slips[si] = ev.slip_2d;
                        if ev.active {
                            active_site_contacts += 1;
                            body_wrenches[body_id] += ev.wrench;
                        }
                    }
                }
            }
            ContactBackend::SdfBarrier {
                barrier,
                mu,
                friction_eps_v,
                ..
            } => {
                let eps_u = (friction_eps_v * dt).max(1e-9);
                for (si, site) in self.graph.contact_sites.iter().enumerate() {
                    if let ContactAttachment::RigidBody { body_id } = site.attachment {
                        let pose = poses[body_id];
                        let offset = pose.rot.rotate(site.local_pos);
                        let p = pose.pos + offset;
                        let mut cfg_site = *barrier;
                        cfg_site.thickness = cfg_site.thickness.max(site.radius);
                        let ev = ipc_point_sdf_barrier(&self.sdf, p, &cfg_site);
                        if ev.active {
                            active_site_contacts += 1;
                            potential_energy += ev.energy;
                            let mut f = -ev.grad;
                            let mut k3 = ev.hess_spd * k_scale_barrier;
                            let lam_n = f.dot(ev.normal).max(0.0);
                            site_normal_forces[si] = lam_n;
                            if *mu > 0.0 && lam_n > 0.0 {
                                let v_pt = vels[body_id].lin() + vels[body_id].ang().cross(offset);
                                let fr = ipc_mollifier_friction_eval(
                                    v_pt * dt,
                                    ev.normal,
                                    lam_n,
                                    *mu,
                                    eps_u,
                                );
                                f -= fr.grad;
                                k3 += fr.hess * (dt * c_scale_fric);
                            }
                            body_wrenches[body_id] += Vec6::from_parts(f, offset.cross(f));
                            let j = Mat6::from_blocks(
                                Mat3::IDENTITY,
                                -offset.skew(),
                                Mat3::ZERO,
                                Mat3::ZERO,
                            );
                            let k6 = Mat6::from_blocks(k3, Mat3::ZERO, Mat3::ZERO, Mat3::ZERO);
                            body_k_tan[body_id] += j.transpose().mul_mat(k6).mul_mat(j);
                        }
                    }
                }
            }
        }

        // --- 2. Beam Interior Node SDF Contact (pipe wall / ground protection) ---
        if !matches!(
            self.config.contact,
            ContactBackend::None | ContactBackend::DelassusLcp { .. }
        ) {
            let implicit_beam = match &self.config.contact {
                ContactBackend::Arpc { implicit } => *implicit,
                ContactBackend::SdfBarrier { .. } => true,
                _ => false,
            };
            for (s, spec) in self.graph.beams.iter().enumerate() {
                let Some(radius) = spec.interior_node_contact_radius else {
                    continue;
                };
                let map = &self.analysis.indexing.beam_maps[s];
                for (li, &nid) in map.free_node_ids.iter().enumerate() {
                    let pos = poses[nid].pos;
                    let vel = vels[nid].lin();
                    match &self.config.contact {
                        ContactBackend::SdfBarrier {
                            barrier,
                            mu,
                            friction_eps_v,
                            ..
                        } => {
                            let mut cfg_b = *barrier;
                            cfg_b.thickness = cfg_b.thickness.max(radius);
                            let ev = ipc_point_sdf_barrier(&self.sdf, pos, &cfg_b);
                            if ev.active {
                                active_beam_contacts += 1;
                                potential_energy += ev.energy;
                                let mut f = -ev.grad;
                                let mut k3 = ev.hess_spd * k_scale_barrier;
                                let lam_n = f.dot(ev.normal).max(0.0);
                                if *mu > 0.0 && lam_n > 0.0 {
                                    let fr = ipc_mollifier_friction_eval(
                                        vel * dt,
                                        ev.normal,
                                        lam_n,
                                        *mu,
                                        (friction_eps_v * dt).max(1e-9),
                                    );
                                    f -= fr.grad;
                                    k3 += fr.hess * (dt * c_scale_fric);
                                }
                                beam_int_forces[s][li] += f;
                                beam_int_k[s][li] += k3;
                            }
                        }
                        _ => {
                            let ev = eval_sdf_surface_contact(
                                pos,
                                vel,
                                radius,
                                &self.sdf,
                                &self.sdf.materials,
                                dt,
                                implicit_beam,
                                0.25,
                            );
                            if ev.active {
                                active_beam_contacts += 1;
                                beam_int_forces[s][li] += ev.force_world;
                                if implicit_beam {
                                    beam_int_k[s][li] += ev.k_tan_3x3;
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- 3. Engine A Codimensional Beam-Beam Self-Contact (IPC / Py-DiSMech C^2) ---
        if self.config.enable_self_contact && !self.self_contact_segments.is_empty() {
            let engine = self.config.self_contact_engine();
            let points: Vec<Vec3> = poses.iter().map(|p| p.pos).collect();
            let asm = engine.assemble_segments(
                &points,
                Some(&self.state.prev_positions),
                &self.self_contact_segments,
                dt,
            );
            active_self_contacts = asm.active_pairs;
            potential_energy += asm.total_energy;

            if asm.active_pairs > 0 {
                // Scatter forces and diagonal SPD 3x3 Hessians onto rigid bodies and beam interior nodes
                for bi in 0..nb {
                    let f = asm.forces[bi];
                    if f.norm_sq() > 0.0 {
                        body_wrenches[bi] += Vec6::from_parts(f, Vec3::ZERO);
                    }
                    let h = asm.diag_hess_spd[bi] * k_scale_barrier;
                    if h.max_abs() > 0.0 {
                        body_k_tan[bi] += Mat6::from_blocks(h, Mat3::ZERO, Mat3::ZERO, Mat3::ZERO);
                    }
                }
                for (s, map) in self.analysis.indexing.beam_maps.iter().enumerate() {
                    for (li, &nid) in map.free_node_ids.iter().enumerate() {
                        beam_int_forces[s][li] += asm.forces[nid];
                        beam_int_k[s][li] += asm.diag_hess_spd[nid] * k_scale_barrier;
                    }
                }
            }
        }

        ContactEvalBundle {
            body_wrenches,
            body_k_tan,
            beam_int_forces,
            beam_int_k,
            next_site_states,
            site_normal_forces,
            site_slips,
            active_site_contacts,
            active_beam_contacts,
            active_self_contacts,
            _potential_energy: potential_energy,
        }
    }

    /// 1-step Linearly Implicit Euler matching `serpworm.physics.gpu_core.GPUCore` to `< 1e-10` FP64,
    /// extended seamlessly to all 3 Schur routes, cantilevers, DER rods, ARPC, IPC self-contact,
    /// and Delassus-Schur Frictional Contact LCP!
    fn step_linearly_implicit_euler(&mut self, dt: f64) -> Result<StepDiagnostics, SolverError> {
        let (diag, _) = self.step_linearly_implicit_euler_inner(dt, false)?;
        Ok(diag)
    }

    /// Run one Linearly Implicit Euler step and also return the cached [`FactoredBipartiteSystem`]
    /// for $O(N)$ reverse-mode IFT adjoint back-solves (zero re-factorization!).
    pub(crate) fn step_linearly_implicit_euler_with_factor(
        &mut self,
        dt: f64,
    ) -> Result<(StepDiagnostics, FactoredBipartiteSystem), SolverError> {
        let (diag, fac_opt) = self.step_linearly_implicit_euler_inner(dt, true)?;
        Ok((
            diag,
            fac_opt.expect("return_factor=true always returns FactoredBipartiteSystem"),
        ))
    }

    fn step_linearly_implicit_euler_inner(
        &mut self,
        dt: f64,
        return_factor: bool,
    ) -> Result<(StepDiagnostics, Option<FactoredBipartiteSystem>), SolverError> {
        let nb = self.graph.bodies.len();
        let total = self.analysis.indexing.total_nodes;
        let g6 = Vec6::from_parts(self.graph.gravity, Vec3::ZERO);

        // 1. Advance tendon servos once per step (matching gpu_core.py:500-516)
        advance_tendon_servos(&self.graph.tendons, &mut self.state.actuation, dt);

        // 2. Evaluate actuation (tendons + joints)
        let body_masses = self.body_masses();
        let act_eval = evaluate_actuation_with_masses(
            nb,
            &self.graph.tendons,
            &self.graph.joints,
            &self.rest_poses,
            &self.state.poses,
            &self.state.vels,
            &mut self.state.actuation,
            dt * dt,
            dt,
            false,
            true,
            self.config.joint_stribeck_friction,
            Some(&body_masses),
            dt,
        );

        // 3. Evaluate contacts (ARPC / GpuCoreKelvinParity / SdfBarrier + SelfContact)
        let contacts = self.evaluate_contacts(
            &self.state.poses,
            &self.state.vels,
            dt,
            dt * dt,
            dt,
        );

        // 4. Assemble global RHS and per-beam condensation (Stage 1)
        let mut rhs_bodies = vec![Vec6::ZERO; nb];
        let mut rigid_mat = RigidSchurMatrix::zeros(self.analysis.num_free_bodies);

        for bi in 0..nb {
            let pm = self.body_inertia_mat6(bi, self.state.poses[bi].rot);
            let alpha = self.graph.bodies[bi].rayleigh_alpha;
            let gyro = self.body_gyro_wrench(bi, self.state.poses[bi].rot, self.state.vels[bi]);
            rhs_bodies[bi] += pm.mul_vec(self.state.vels[bi] + dt * g6)
                + dt * (act_eval.body_wrenches[bi] + contacts.body_wrenches[bi] - gyro);
            if let Some(slot) = self.analysis.body_to_free_slot[bi] {
                let lhs = (1.0 + dt * alpha) * pm + contacts.body_k_tan[bi];
                rigid_mat.add_block(slot, slot, lhs);
            }
        }

        // Scatter implicit joint blocks into rigid Schur matrix
        for &(ba, bb, blk) in &act_eval.implicit_blocks {
            if let (Some(u), Some(v)) = (
                self.analysis.body_to_free_slot[ba],
                self.analysis.body_to_free_slot[bb],
            ) {
                rigid_mat.add_block(u, v, blk);
            }
        }

        let mut beam_factors: Vec<BeamFactorKind> = Vec::with_capacity(self.beams.len());
        let mut beam_contribs: Vec<BeamSchurContribution> = Vec::with_capacity(self.beams.len());

        // Evaluate and condense each beam (parallelized across beams when >= 4 corotational beams)
        let all_corot = self
            .beams
            .iter()
            .all(|b| matches!(b, BeamRuntime::Corotational(_)));

        if all_corot && self.beams.len() >= 4 {
            let use_jac = self.config.use_strain_jacobian;
            let results: Result<Vec<(CorotationalBeamFactor, BeamSchurContribution, Vec6, Vec6)>, LinearSolveError> =
                self.beams
                    .par_iter()
                    .enumerate()
                    .map(|(s, br)| {
                        let BeamRuntime::Corotational(data) = br else {
                            unreachable!();
                        };
                        let map = &self.analysis.indexing.beam_maps[s];
                        self.condense_corotational_vel_step(
                            data,
                            map,
                            &self.state.poses,
                            &self.state.vels,
                            g6,
                            &contacts.beam_int_forces[s],
                            &contacts.beam_int_k[s],
                            dt,
                            use_jac,
                        )
                    })
                    .collect();
            for (fac, contrib, ra_el, rb_el) in results? {
                rhs_bodies[contrib.body_a] += ra_el;
                if let Some(bb) = contrib.body_b {
                    rhs_bodies[bb] += rb_el;
                }
                beam_factors.push(BeamFactorKind::Corotational(Box::new(fac)));
                beam_contribs.push(contrib);
            }
        } else {
            for (s, br) in self.beams.iter().enumerate() {
                let map = &self.analysis.indexing.beam_maps[s];
                match br {
                    BeamRuntime::Corotational(data) => {
                        let (fac, contrib, ra_el, rb_el) = self.condense_corotational_vel_step(
                            data,
                            map,
                            &self.state.poses,
                            &self.state.vels,
                            g6,
                            &contacts.beam_int_forces[s],
                            &contacts.beam_int_k[s],
                            dt,
                            self.config.use_strain_jacobian,
                        )?;
                        rhs_bodies[contrib.body_a] += ra_el;
                        if let Some(bb) = contrib.body_b {
                            rhs_bodies[bb] += rb_el;
                        }
                        beam_factors.push(BeamFactorKind::Corotational(Box::new(fac)));
                        beam_contribs.push(contrib);
                    }
                    BeamRuntime::Der(data) => {
                        let der_st = self.state.der_states[s]
                            .as_ref()
                            .expect("DER beam must have DerBeamState");
                        let c_m = 1.0 + dt * data.rayleigh_alpha;
                        let c_k = dt * data.rayleigh_beta + dt * dt;
                        let out = data.assemble_and_condense(
                            map,
                            &self.state.poses,
                            &self.state.vels,
                            None,
                            None,
                            der_st,
                            self.graph.gravity,
                            &contacts.beam_int_forces[s],
                            &contacts.beam_int_k[s],
                            dt,
                            c_m,
                            c_k,
                            true,
                            self.config.project_der_hessian_spd,
                        )?;
                        rhs_bodies[out.contrib.body_a] += out.end_a_rhs;
                        if let Some(bb) = out.contrib.body_b {
                            rhs_bodies[bb] += out.end_b_rhs;
                        }
                        beam_factors.push(BeamFactorKind::Der(Box::new(out.factor)));
                        beam_contribs.push(out.contrib);
                    }
                }
            }
        }

        // Scatter all beam Schur blocks & condensed RHS into Stage 2 rigid system
        let mut rhs_free = vec![Vec6::ZERO; self.analysis.num_free_bodies];
        for (slot, &body_id) in self.analysis.free_body_ids.iter().enumerate() {
            rhs_free[slot] = rhs_bodies[body_id];
        }
        for contrib in &beam_contribs {
            rigid_mat.scatter_beam(&mut rhs_free, contrib, &self.analysis.body_to_free_slot);
        }
        if self.config.lm_initial > 0.0 {
            rigid_mat.add_diagonal_shift(self.config.lm_initial);
        }

        // Stage 2: Adaptive Rigid Schur Factorization & Solve
        let rigid_factor = self.factorize_rigid_schur(&rigid_mat)?;
        let v_free = rigid_factor.solve(&rhs_free)?;

        let mut v_new = vec![Vec6::ZERO; total];
        for (slot, &body_id) in self.analysis.free_body_ids.iter().enumerate() {
            v_new[body_id] = v_free[slot];
        }

        // Stage 3: Back-substitute all beam interior velocities & DER twist velocities
        let mut der_theta_vels: Vec<Option<Vec<f64>>> = vec![None; self.beams.len()];
        for (s, fac) in beam_factors.iter().enumerate() {
            let map = &self.analysis.indexing.beam_maps[s];
            let va = v_new[map.node_a];
            let vb = map.node_b.map_or(Vec6::ZERO, |bb| v_new[bb]);
            match fac {
                BeamFactorKind::Corotational(cf) => {
                    let vi = cf.back_substitute(va, vb);
                    for (li, &nid) in map.free_node_ids.iter().enumerate() {
                        v_new[nid] = vi[li];
                    }
                }
                BeamFactorKind::Der(df) => {
                    let (v_verts, v_thetas) = df.back_substitute(va, vb);
                    for (li, &nid) in map.free_node_ids.iter().enumerate() {
                        v_new[nid] = Vec6::from_parts(v_verts[li], Vec3::ZERO);
                    }
                    der_theta_vels[s] = Some(v_thetas);
                }
            }
        }

        // Optional Stage 3b: Compliance-Based Frictional Contact LCP (Engine B, Shen et al. IJRR 2026)
        let mut lcp_iters = 0usize;
        let mut lcp_residual = 0.0_f64;
        let mut active_site_contacts = contacts.active_site_contacts;
        let saved_factor = if let ContactBackend::DelassusLcp {
            lcp,
            dedup_radius,
            baumgarte_beta,
        } = self.config.contact
        {
            let factored = FactoredBipartiteSystem {
                beam_factors,
                rigid_factor,
            };
            let (dv_lcp, n_con, iters, res, normal_forces) = self.solve_delassus_lcp_correction(
                &factored,
                &v_new,
                dt,
                &lcp,
                dedup_radius,
                baumgarte_beta,
            )?;
            for i in 0..total {
                v_new[i] += dv_lcp[i];
            }
            active_site_contacts = n_con;
            lcp_iters = iters;
            lcp_residual = res;
            for (si, nf) in normal_forces {
                if si < self.state.contact_normal_forces.len() {
                    self.state.contact_normal_forces[si] = nf;
                }
            }
            if return_factor {
                Some(factored)
            } else {
                None
            }
        } else {
            self.state.contact_states = contacts.next_site_states;
            self.state.contact_normal_forces = contacts.site_normal_forces;
            self.state.contact_slips = contacts.site_slips;
            if return_factor {
                Some(FactoredBipartiteSystem {
                    beam_factors,
                    rigid_factor,
                })
            } else {
                None
            }
        };

        // Verify finiteness
        if v_new.iter().any(|v| v.0.iter().any(|x| !x.is_finite())) {
            self.state.failed = true;
            return Err(SolverError::NonFiniteState);
        }

        // Exact SE(3) exponential pose integration (exact parity with gpu_core.py:985-988)
        self.state.prev_positions = self.state.poses.iter().map(|p| p.pos).collect();
        self.state.prev_vels.clone_from(&self.state.vels);
        let mut max_q_err = 0.0_f64;
        for i in 0..total {
            self.state.accels[i] = (v_new[i] - self.state.vels[i]) * (1.0 / dt);
            self.state.prev_dx[i] = v_new[i] * dt;
            self.state.vels[i] = v_new[i];
            if i < nb && self.graph.bodies[i].is_fixed {
                self.state.vels[i] = Vec6::ZERO;
                continue;
            }
            self.state.poses[i] = self.state.poses[i].integrate_decoupled(v_new[i], dt);
            max_q_err = max_q_err.max((self.state.poses[i].rot.norm() - 1.0).abs());
        }

        // Advance DER twist angles & parallel-transport Bishop frames
        for (s, br) in self.beams.iter().enumerate() {
            if let (BeamRuntime::Der(data), Some(ref mut dst), Some(ref v_th)) =
                (br, &mut self.state.der_states[s], &der_theta_vels[s])
            {
                let map = &self.analysis.indexing.beam_maps[s];
                for e in 0..data.n_elements {
                    self.state.prev_der_theta_vel[s][e] = dst.theta_vel[e];
                    self.state.prev_der_dtheta[s][e] = dt * v_th[e];
                    self.state.der_theta_accels[s][e] = (v_th[e] - dst.theta_vel[e]) / dt;
                    dst.theta_vel[e] = v_th[e];
                    dst.theta[e] += dt * v_th[e];
                }
                data.update_bishop_frames_and_sync_poses(map, &mut self.state.poses, dst);
            }
        }

        self.state.time += dt;
        self.state.step_count += 1;

        Ok((
            StepDiagnostics {
                step_index: self.state.step_count,
                sim_time: self.state.time,
                newton_iters: 1,
                residual_inf: 0.0,
                residual_l2: 0.0,
                line_search_alpha: 1.0,
                ccd_toi: 1.0,
                active_site_contacts,
                active_beam_contacts: contacts.active_beam_contacts,
                active_self_contacts: contacts.active_self_contacts,
                lcp_iters,
                lcp_residual,
                max_quat_norm_err: max_q_err,
                failed: self.state.failed,
            },
            saved_factor,
        ))
    }

    /// Helper: assemble & Block-Thomas-condense one corotational beam in linearly-implicit velocity form.
    #[allow(clippy::too_many_arguments)]
    fn condense_corotational_vel_step(
        &self,
        data: &CorotationalBeamData,
        map: &fg_ir::BeamDofMap,
        poses: &[Se3],
        vels: &[Vec6],
        g6: Vec6,
        int_forces: &[Vec3],
        int_k3: &[Mat3],
        dt: f64,
        use_strain_jacobian: bool,
    ) -> Result<(CorotationalBeamFactor, BeamSchurContribution, Vec6, Vec6), LinearSolveError> {
        let eval = data.evaluate(map, poses, use_strain_jacobian);
        let n_el = data.n_elements;
        let n_free = map.free_node_ids.len();

        let mut elem_a = Vec::with_capacity(n_el);
        let mut int_rhs = vec![Vec6::ZERO; n_free];
        let mut end_a_rhs = Vec6::ZERO;
        let mut end_b_rhs = Vec6::ZERO;

        let cm = 1.0 + dt * data.rayleigh_alpha;
        let ck = dt * data.rayleigh_beta + dt * dt;

        for e in 0..n_el {
            let n0 = map.chain_node_ids[e];
            let n1 = map.chain_node_ids[e + 1];
            let vg = Vec12::from_halves(vels[n0] + dt * g6, vels[n1] + dt * g6);
            let m_vg = eval.elem_m[e].mul_vec(vg);
            let rhs_e = m_vg + eval.elem_elastic[e] * dt;

            if e == 0 {
                end_a_rhs += rhs_e.head();
            } else {
                int_rhs[e - 1] += rhs_e.head();
            }
            if e + 1 == n_el && data.body_b.is_some() {
                end_b_rhs += rhs_e.tail();
            } else {
                int_rhs[e] += rhs_e.tail();
            }
            elem_a.push(eval.elem_m[e] * cm + eval.elem_k[e] * ck);
        }

        let mut int_k6 = vec![Mat6::ZERO; n_free];
        for i in 0..n_free {
            int_rhs[i] += Vec6::from_parts(int_forces[i] * dt, Vec3::ZERO);
            if int_k3[i].max_abs() > 0.0 {
                int_k6[i] = Mat6::from_blocks(int_k3[i], Mat3::ZERO, Mat3::ZERO, Mat3::ZERO);
            }
        }

        let (fac, contrib) = CorotationalBeamFactor::factor_and_condense(
            data.beam_id,
            data.body_a,
            data.body_b,
            &elem_a,
            &int_k6,
            &int_rhs,
        )?;
        Ok((fac, contrib, end_a_rhs, end_b_rhs))
    }

    /// Full Nonlinear Damped Newton-Raphson with Armijo Merit Line Search, Conservative ACCD
    /// Clamping, and Adaptive Levenberg-Marquardt Regularization on $SE(3)$ supporting:
    /// - `IntegratorKind::DampedNewtonEuler` (exact angular-momentum conserving Backward Euler),
    /// - `IntegratorKind::NewmarkBeta { beta, gamma }` (2nd-order symplectic/trapezoidal),
    /// - `IntegratorKind::Bdf2` (2nd-order L-stable).
    fn step_damped_newton(&mut self, dt: f64) -> Result<StepDiagnostics, SolverError> {
        let nb = self.graph.bodies.len();
        let total = self.analysis.indexing.total_nodes;
        let g6 = Vec6::from_parts(self.graph.gravity, Vec3::ZERO);

        // Advance tendon servos once at start of step
        advance_tendon_servos(&self.graph.tendons, &mut self.state.actuation, dt);

        // Save step-n base state
        let x_n = self.state.poses.clone();
        let v_n = self.state.vels.clone();
        let a_n = self.state.accels.clone();
        let prev_dx = self.state.prev_dx.clone();
        let prev_v = self.state.prev_vels.clone();
        let der_n = self.state.der_states.clone();
        let der_a_n = self.state.der_theta_accels.clone();
        let prev_der_dth = self.state.prev_der_dtheta.clone();
        let prev_der_v = self.state.prev_der_theta_vel.clone();

        // Precompute step-n world angular momentum L_n = I_w(q_n) * w_n for exact Euler conservation
        let ang_mom_n: Vec<Vec3> = (0..nb)
            .map(|bi| {
                let m6 = self.body_inertia_mat6(bi, x_n[bi].rot);
                m6.block(1, 1).mul_vec(v_n[bi].ang())
            })
            .collect();

        // Kinematic coefficients (c_v = dv / d(dx), c_a = da / d(dx))
        let is_bdf2_active =
            matches!(self.config.integrator, IntegratorKind::Bdf2) && self.state.step_count > 0;
        let (c_v, c_a) = match self.config.integrator {
            IntegratorKind::DampedNewtonEuler | IntegratorKind::LinearlyImplicitEuler => {
                (1.0 / dt, 1.0 / (dt * dt))
            }
            IntegratorKind::NewmarkBeta { beta, gamma } => {
                let b = beta.clamp(0.05, 0.5);
                let g = gamma.clamp(0.5, 1.0);
                (g / (b * dt), 1.0 / (b * dt * dt))
            }
            IntegratorKind::Bdf2 => {
                if is_bdf2_active {
                    (1.5 / dt, 2.25 / (dt * dt))
                } else {
                    (1.0 / dt, 1.0 / (dt * dt))
                }
            }
        };

        // Kinematic evaluator: given trial poses `x_k` -> `(v_k, a_k)` on SE(3)
        let compute_kinematics = |x_k: &[Se3]| -> (Vec<Vec6>, Vec<Vec6>, Vec<Vec6>) {
            let mut dx = vec![Vec6::ZERO; total];
            let mut v_k = vec![Vec6::ZERO; total];
            let mut a_k = vec![Vec6::ZERO; total];
            for i in 0..total {
                if i < nb && self.graph.bodies[i].is_fixed {
                    continue;
                }
                let dp = x_k[i].pos - x_n[i].pos;
                let dth = x_k[i].rot.mul(x_n[i].rot.conj()).log();
                dx[i] = Vec6::from_parts(dp, dth);
                match self.config.integrator {
                    IntegratorKind::DampedNewtonEuler | IntegratorKind::LinearlyImplicitEuler => {
                        v_k[i] = dx[i] * (1.0 / dt);
                        a_k[i] = (v_k[i] - v_n[i]) * (1.0 / dt);
                    }
                    IntegratorKind::NewmarkBeta { beta, gamma } => {
                        let b = beta.clamp(0.05, 0.5);
                        let g = gamma.clamp(0.5, 1.0);
                        a_k[i] = dx[i] * (1.0 / (b * dt * dt))
                            - v_n[i] * (1.0 / (b * dt))
                            - a_n[i] * (0.5 / b - 1.0);
                        v_k[i] = v_n[i] + a_n[i] * ((1.0 - g) * dt) + a_k[i] * (g * dt);
                    }
                    IntegratorKind::Bdf2 => {
                        if is_bdf2_active {
                            v_k[i] = dx[i] * (1.5 / dt) - prev_dx[i] * (0.5 / dt);
                            a_k[i] =
                                v_k[i] * (1.5 / dt) - v_n[i] * (2.0 / dt) + prev_v[i] * (0.5 / dt);
                        } else {
                            v_k[i] = dx[i] * (1.0 / dt);
                            a_k[i] = (v_k[i] - v_n[i]) * (1.0 / dt);
                        }
                    }
                }
            }
            (dx, v_k, a_k)
        };

        // DER twist kinematics evaluator matching the active integrator (Euler / NewmarkBeta / Bdf2)
        let compute_der_kinematics = |der_k: &[Option<DerBeamState>]| -> DerKinematics {
                let mut dth_all = vec![Vec::new(); self.beams.len()];
                let mut vth_all = vec![Vec::new(); self.beams.len()];
                let mut ath_all = vec![Vec::new(); self.beams.len()];
                for (s, br) in self.beams.iter().enumerate() {
                    if let (BeamRuntime::Der(data), Some(dst), Some(dn)) = (br, &der_k[s], &der_n[s])
                    {
                        let mut dth_v = vec![0.0; data.n_elements];
                        let mut vth_v = vec![0.0; data.n_elements];
                        let mut ath_v = vec![0.0; data.n_elements];
                        for e in 0..data.n_elements {
                            let dth = dst.theta[e] - dn.theta[e];
                            dth_v[e] = dth;
                            match self.config.integrator {
                                IntegratorKind::DampedNewtonEuler
                                | IntegratorKind::LinearlyImplicitEuler => {
                                    vth_v[e] = dth / dt;
                                    ath_v[e] = (vth_v[e] - dn.theta_vel[e]) / dt;
                                }
                                IntegratorKind::NewmarkBeta { beta, gamma } => {
                                    let b = beta.clamp(0.05, 0.5);
                                    let g = gamma.clamp(0.5, 1.0);
                                    let a_th = dth / (b * dt * dt)
                                        - dn.theta_vel[e] / (b * dt)
                                        - der_a_n[s][e] * (0.5 / b - 1.0);
                                    ath_v[e] = a_th;
                                    vth_v[e] = dn.theta_vel[e]
                                        + (1.0 - g) * dt * der_a_n[s][e]
                                        + g * dt * a_th;
                                }
                                IntegratorKind::Bdf2 => {
                                    if is_bdf2_active {
                                        let v_th = 1.5 * dth / dt - 0.5 * prev_der_dth[s][e] / dt;
                                        vth_v[e] = v_th;
                                        ath_v[e] = 1.5 * v_th / dt - 2.0 * dn.theta_vel[e] / dt
                                            + 0.5 * prev_der_v[s][e] / dt;
                                    } else {
                                        vth_v[e] = dth / dt;
                                        ath_v[e] = (vth_v[e] - dn.theta_vel[e]) / dt;
                                    }
                                }
                            }
                        }
                        dth_all[s] = dth_v;
                        vth_all[s] = vth_v;
                        ath_all[s] = ath_v;
                    }
                }
                (dth_all, vth_all, ath_all)
            };

        let residual_norms = |r: &[Vec6], dth_r: &[Vec<f64>]| -> (f64, f64) {
            let mut inf = r.iter().fold(0.0_f64, |acc, v| acc.max(v.max_abs()));
            let mut l2_sq: f64 = r.iter().map(|v| v.dot(*v)).sum();
            for v in dth_r {
                for &val in v {
                    inf = inf.max(val.abs());
                    l2_sq += val * val;
                }
            }
            (inf, l2_sq.sqrt())
        };

        // Initial predictor x^{(0)} on SE(3)
        let mut x_trial = x_n.clone();
        for i in 0..total {
            if i < nb && self.graph.bodies[i].is_fixed {
                continue;
            }
            let dx0 = match self.config.integrator {
                IntegratorKind::NewmarkBeta { .. } => v_n[i] * dt + a_n[i] * (0.5 * dt * dt),
                _ => v_n[i] * dt,
            };
            x_trial[i] = x_n[i].integrate_decoupled(dx0, 1.0);
        }

        // Initial DER trial twists
        let mut der_trial = der_n.clone();
        for (s, br) in self.beams.iter().enumerate() {
            if let (BeamRuntime::Der(data), Some(ref mut dst), Some(ref dn)) =
                (br, &mut der_trial[s], &der_n[s])
            {
                for e in 0..data.n_elements {
                    let dth0 = match self.config.integrator {
                        IntegratorKind::NewmarkBeta { .. } => {
                            dt * dn.theta_vel[e] + 0.5 * dt * dt * der_a_n[s][e]
                        }
                        _ => dt * dn.theta_vel[e],
                    };
                    dst.theta[e] = dn.theta[e] + dth0;
                }
            }
        }

        // Optional conservative CCD clamp on initial predictor
        let mut ccd_toi_min = 1.0_f64;
        if self.config.enable_ccd_clamping {
            let t_pred = self.compute_global_ccd_toi(&x_n, &x_trial);
            if t_pred < 1.0 {
                ccd_toi_min = ccd_toi_min.min(t_pred);
                for i in 0..total {
                    if i < nb && self.graph.bodies[i].is_fixed {
                        continue;
                    }
                    let dp = (x_trial[i].pos - x_n[i].pos) * t_pred;
                    let dth = x_trial[i].rot.mul(x_n[i].rot.conj()).log() * t_pred;
                    x_trial[i] = x_n[i].integrate_decoupled(Vec6::from_parts(dp, dth), 1.0);
                }
            }
        }

        let mut lm = self.config.lm_initial;
        let mut newton_iters = 0usize;
        let mut last_alpha = 1.0_f64;
        let mut last_res_inf = 0.0_f64;
        let mut last_res_l2 = 0.0_f64;

        let eval_residual = |world: &FlexGraphWorld,
                             x_k: &[Se3],
                             v_k: &[Vec6],
                             a_k: &[Vec6],
                             der_k: &[Option<DerBeamState>],
                             der_th_a: &[Vec<f64>],
                             act_scratch: &mut ActuationState|
         -> (Vec<Vec6>, ContactEvalBundle, Vec<Vec<f64>>) {
            let body_masses = world.body_masses();
            let act_ev = evaluate_actuation_with_masses(
                nb,
                &world.graph.tendons,
                &world.graph.joints,
                &world.rest_poses,
                x_k,
                v_k,
                act_scratch,
                1.0,
                c_v,
                true,
                false,
                world.config.joint_stribeck_friction,
                Some(&body_masses),
                dt,
            );
            let con = world.evaluate_contacts(x_k, v_k, dt, 1.0, c_v);
            let mut r = vec![Vec6::ZERO; total];

            // Rigid-body dynamic residual r_b = f_inert + alpha*M*v - f_act - f_con
            for bi in 0..nb {
                if world.graph.bodies[bi].is_fixed {
                    continue;
                }
                let m6 = world.body_inertia_mat6(bi, x_k[bi].rot);
                let alpha = world.graph.bodies[bi].rayleigh_alpha;
                let f_inert = if matches!(world.config.integrator, IntegratorKind::DampedNewtonEuler)
                    && !world.config.world_fixed_inertia
                {
                    // Exact angular-momentum conserving Backward Euler residual:
                    // f_lin = m*(a_lin - g), tau = (I_w(q_k) w_k - I_w(q_n) w_n) / dt
                    let f_lin = m6.block(0, 0).mul_vec(a_k[bi].lin() - world.graph.gravity);
                    let ang_mom_k = m6.block(1, 1).mul_vec(v_k[bi].ang());
                    let tau_inert = (ang_mom_k - ang_mom_n[bi]) * (1.0 / dt);
                    Vec6::from_parts(f_lin, tau_inert)
                } else {
                    m6.mul_vec(a_k[bi] - g6) + world.body_gyro_wrench(bi, x_k[bi].rot, v_k[bi])
                };
                r[bi] = f_inert + (alpha * m6.mul_vec(v_k[bi]))
                    - act_ev.body_wrenches[bi]
                    - con.body_wrenches[bi];
            }

            // Corotational and DER beam residuals
            let mut der_th_res: Vec<Vec<f64>> = vec![Vec::new(); world.beams.len()];
            for (s, br) in world.beams.iter().enumerate() {
                let map = &world.analysis.indexing.beam_maps[s];
                match br {
                    BeamRuntime::Corotational(data) => {
                        let ev = data.evaluate(map, x_k, world.config.use_strain_jacobian);
                        for e in 0..data.n_elements {
                            let n0 = map.chain_node_ids[e];
                            let n1 = map.chain_node_ids[e + 1];
                            let a12 = Vec12::from_halves(a_k[n0] - g6, a_k[n1] - g6);
                            let v12 = Vec12::from_halves(v_k[n0], v_k[n1]);
                            let r_e = ev.elem_m[e].mul_vec(a12 + v12 * data.rayleigh_alpha)
                                + ev.elem_k[e].mul_vec(v12 * data.rayleigh_beta)
                                - ev.elem_elastic[e];
                            if !(n0 < nb && world.graph.bodies[n0].is_fixed) {
                                r[n0] += r_e.head();
                            }
                            if !(n1 < nb && world.graph.bodies[n1].is_fixed) {
                                r[n1] += r_e.tail();
                            }
                        }
                        for (li, &nid) in map.free_node_ids.iter().enumerate() {
                            r[nid] -= Vec6::from_parts(con.beam_int_forces[s][li], Vec3::ZERO);
                        }
                    }
                    BeamRuntime::Der(data) => {
                        let dst = der_k[s].as_ref().unwrap();
                        if let Ok(out) = data.assemble_and_condense(
                            map,
                            x_k,
                            v_k,
                            Some(a_k),
                            Some(&der_th_a[s]),
                            dst,
                            world.graph.gravity,
                            &con.beam_int_forces[s],
                            &con.beam_int_k[s],
                            dt,
                            c_a + data.rayleigh_alpha * c_v,
                            1.0 + data.rayleigh_beta * c_v,
                            false,
                            world.config.project_der_hessian_spd,
                        ) {
                            if !world.graph.bodies[data.body_a].is_fixed {
                                r[data.body_a] -= out.end_a_rhs;
                            }
                            if let Some(bb) = data.body_b {
                                if !world.graph.bodies[bb].is_fixed {
                                    r[bb] -= out.end_b_rhs;
                                }
                            }
                            for (li, &nid) in map.free_node_ids.iter().enumerate() {
                                r[nid] = Vec6::from_parts(out.res_verts[li], Vec3::ZERO);
                            }
                            der_th_res[s] = out.res_thetas;
                        }
                    }
                }
            }
            (r, con, der_th_res)
        };

        // Note: inside Newton iterations, evaluate ARPC with `implicit = false` for exact uncompensated
        // residual forces, while assembling SPD tangent stiffness via `dt`-scaled `Contact-Schur`!
        let saved_contact = self.config.contact.clone();
        let newton_contact_implicit = match saved_contact {
            ContactBackend::Arpc { implicit } => implicit,
            _ => false,
        };
        if let ContactBackend::Arpc { .. } = self.config.contact {
            self.config.contact = ContactBackend::Arpc { implicit: false };
        }

        let mut act_scratch = self.state.actuation.clone();

        for iter in 0..self.config.newton_max_iters {
            newton_iters = iter + 1;
            let (_dx_k, v_k, a_k) = compute_kinematics(&x_trial);
            let (_dth_k, th_v_k, th_a_k) = compute_der_kinematics(&der_trial);

            // Sync DER trial twist velocities from trial twist angles
            for (s, dst_opt) in der_trial.iter_mut().enumerate() {
                if let Some(ref mut dst) = dst_opt {
                    dst.theta_vel.clone_from(&th_v_k[s]);
                }
            }

            let (r_k, con_k, der_th_res_k) = eval_residual(
                self,
                &x_trial,
                &v_k,
                &a_k,
                &der_trial,
                &th_a_k,
                &mut act_scratch,
            );
            let (res_inf, res_l2) = residual_norms(&r_k, &der_th_res_k);
            last_res_inf = res_inf;
            last_res_l2 = res_l2;

            if res_inf < self.config.newton_tol {
                break;
            }

            // Assemble tangent system H * delta_xi = -r_k
            // For ARPC contact stiffness in Newton tangent, query SPD Contact-Schur scaled to position space (div by dt^2)
            let mut arpc_tan_k = vec![Mat6::ZERO; nb];
            if newton_contact_implicit {
                for (si, site) in self.graph.contact_sites.iter().enumerate() {
                    if let ContactAttachment::RigidBody { body_id } = site.attachment {
                        let ev = if site.solref_imp.is_some() {
                            eval_arpc_site_with_mass(
                                site,
                                x_trial[body_id],
                                v_k[body_id],
                                &self.sdf,
                                &self.sdf.materials,
                                &self.state.contact_states[si],
                                dt,
                                true,
                                self.graph.bodies[body_id].mass,
                            )
                        } else {
                            eval_arpc_site(
                                site,
                                x_trial[body_id],
                                v_k[body_id],
                                &self.sdf,
                                &self.sdf.materials,
                                &self.state.contact_states[si],
                                dt,
                                true,
                            )
                        };
                        if ev.active {
                            // `ev.k_tan` is `dt*C + dt^2*K` (velocity space); divide by `dt^2` -> `K + (1/dt)*C`!
                            arpc_tan_k[body_id] += ev.k_tan * (1.0 / (dt * dt));
                        }
                    }
                }
            }

            let body_masses = self.body_masses();
            let act_tan = evaluate_actuation_with_masses(
                nb,
                &self.graph.tendons,
                &self.graph.joints,
                &self.rest_poses,
                &x_trial,
                &v_k,
                &mut act_scratch,
                1.0,
                c_v,
                true,
                false,
                self.config.joint_stribeck_friction,
                Some(&body_masses),
                dt,
            );

            // Solve H * delta_xi = -r_k via Bipartite Direct Solver (with adaptive LM retry on singularity)
            let mut solve_ok = None;
            for lm_try in 0..6 {
                let cur_lm = if lm_try == 0 {
                    lm
                } else {
                    (lm.max(1e-6)) * 10.0_f64.powi(lm_try)
                };
                if let Ok((dxi, d_der_th)) = self.solve_newton_linear_step(
                    &x_n,
                    &x_trial,
                    &v_k,
                    &a_k,
                    &der_trial,
                    &th_a_k,
                    &r_k,
                    &con_k,
                    &arpc_tan_k,
                    &act_tan.implicit_blocks,
                    dt,
                    c_v,
                    c_a,
                    cur_lm,
                ) {
                    solve_ok = Some((dxi, d_der_th, cur_lm));
                    break;
                }
            }

            let Some((delta_xi, delta_der_th, used_lm)) = solve_ok else {
                self.config.contact = saved_contact;
                self.state.failed = true;
                return Err(SolverError::LinearSolve(
                    LinearSolveError::SingularRigidPivot { block_idx: 0 },
                ));
            };
            lm = used_lm;

            let mut step_inf = delta_xi.iter().fold(0.0_f64, |acc, v| acc.max(v.max_abs()));
            for dth in delta_der_th.iter().flatten() {
                for &d in dth {
                    step_inf = step_inf.max(d.abs());
                }
            }
            let mut energy_err: f64 = r_k
                .iter()
                .zip(delta_xi.iter())
                .map(|(r, d)| r.dot(*d).abs())
                .sum();
            for (r_th, dth_opt) in der_th_res_k.iter().zip(delta_der_th.iter()) {
                if let Some(dth) = dth_opt {
                    for (&rv, &dv) in r_th.iter().zip(dth.iter()) {
                        energy_err += (rv * dv).abs();
                    }
                }
            }
            if step_inf < 1e-16 {
                break;
            }

            // Conservative ACCD step clamping BEFORE Armijo backtracking
            let mut alpha = 1.0_f64;
            let mut iter_ccd_toi = 1.0_f64;
            if self.config.enable_ccd_clamping {
                let x_cand: Vec<Se3> = (0..total)
                    .map(|i| {
                        if i < nb && self.graph.bodies[i].is_fixed {
                            x_trial[i]
                        } else {
                            x_trial[i].integrate_decoupled(delta_xi[i], 1.0)
                        }
                    })
                    .collect();
                let toi = self.compute_global_ccd_toi(&x_trial, &x_cand);
                iter_ccd_toi = toi;
                ccd_toi_min = ccd_toi_min.min(toi);
                alpha = alpha.min(toi);
            }

            // Armijo merit backtracking on ||r||_2
            let mut accepted = false;
            let max_ls = if step_inf < 1e-13 && energy_err < 1e-20 {
                3
            } else {
                self.config.line_search_max_iters
            };
            for _ls in 0..max_ls {
                let mut x_cand = x_trial.clone();
                for i in 0..total {
                    if i < nb && self.graph.bodies[i].is_fixed {
                        continue;
                    }
                    x_cand[i] = x_trial[i].integrate_decoupled(delta_xi[i], alpha);
                }
                let mut der_cand = der_trial.clone();
                for (s, dth_opt) in delta_der_th.iter().enumerate() {
                    if let (Some(ref mut dst), Some(ref dth)) = (&mut der_cand[s], dth_opt) {
                        for (t, &dt_i) in dst.theta.iter_mut().zip(dth.iter()) {
                            *t += alpha * dt_i;
                        }
                    }
                }

                let (_dx_c, v_c, a_c) = compute_kinematics(&x_cand);
                let (_dth_c, th_v_c, th_a_c) = compute_der_kinematics(&der_cand);
                for (s, dst_opt) in der_cand.iter_mut().enumerate() {
                    if let Some(ref mut dst) = dst_opt {
                        dst.theta_vel.clone_from(&th_v_c[s]);
                    }
                }
                let (r_c, _, der_th_res_c) = eval_residual(
                    self,
                    &x_cand,
                    &v_c,
                    &a_c,
                    &der_cand,
                    &th_a_c,
                    &mut act_scratch,
                );
                let (res_c_inf, res_c_l2) = residual_norms(&r_c, &der_th_res_c);

                if res_c_l2.is_finite() && res_c_l2 <= (1.0 - 1.0e-4 * alpha) * res_l2 + 1.0e-14 {
                    x_trial = x_cand;
                    der_trial = der_cand;
                    last_alpha = alpha;
                    last_res_inf = res_c_inf;
                    last_res_l2 = res_c_l2;
                    lm *= 0.25;
                    if lm < 1e-12 {
                        lm = 0.0;
                    }
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }

            if !accepted {
                // If already at machine-precision energy equilibrium (FP64 ULP floor on stiff steel modes),
                // accept convergence cleanly without futile LM spinning.
                if step_inf < 1e-13 && energy_err < 1e-20 {
                    break;
                }
                // Take damped safeguard step strictly clamped by ACCD TOI, and increase LM damping
                let tiny = alpha.max(1.0e-4).min(iter_ccd_toi);
                for i in 0..total {
                    if i < nb && self.graph.bodies[i].is_fixed {
                        continue;
                    }
                    x_trial[i] = x_trial[i].integrate_decoupled(delta_xi[i], tiny);
                }
                for (s, dth_opt) in delta_der_th.iter().enumerate() {
                    if let (Some(ref mut dst), Some(ref dth)) = (&mut der_trial[s], dth_opt) {
                        for (t, &dt_i) in dst.theta.iter_mut().zip(dth.iter()) {
                            *t += tiny * dt_i;
                        }
                    }
                }
                lm = (lm * 10.0).max(1.0e-4);
                last_alpha = tiny;
            }
        }

        // Restore original contact backend & commit final converged state!
        self.config.contact = saved_contact;
        let (mut dx_fin, mut v_fin, a_fin) = compute_kinematics(&x_trial);
        let (dth_fin, th_v_fin, th_a_fin) = compute_der_kinematics(&der_trial);
        if x_trial
            .iter()
            .any(|p| p.pos.0.iter().any(|v| !v.is_finite()))
            || v_fin.iter().any(|v| v.0.iter().any(|x| !x.is_finite()))
        {
            self.state.failed = true;
            return Err(SolverError::NonFiniteState);
        }

        // Commit actuation persistent memories at converged state
        let body_masses_fin = self.body_masses();
        let _ = evaluate_actuation_with_masses(
            nb,
            &self.graph.tendons,
            &self.graph.joints,
            &self.rest_poses,
            &x_trial,
            &v_fin,
            &mut self.state.actuation,
            0.0,
            0.0,
            false,
            true,
            self.config.joint_stribeck_friction,
            Some(&body_masses_fin),
            dt,
        );

        // Apply Engine B (Delassus-Schur Frictional Contact LCP) if selected, or commit ARPC/Barrier states
        let (
            active_site_contacts,
            active_beam_contacts,
            active_self_contacts,
            lcp_iters,
            lcp_residual,
        ) = if let ContactBackend::DelassusLcp {
            lcp,
            dedup_radius,
            baumgarte_beta,
        } = self.config.contact
        {
            let factored = self.factor_velocity_bipartite_operator(
                &x_trial,
                &v_fin,
                &der_trial,
                dt,
                c_v / c_a,
                1.0 / c_a,
            )?;
            let (dv_lcp, n_con, iters, res, normal_forces) = self.solve_delassus_lcp_correction(
                &factored,
                &v_fin,
                dt,
                &lcp,
                dedup_radius,
                baumgarte_beta,
            )?;
            self.state.contact_normal_forces.fill(0.0);
            for (si, nf) in normal_forces {
                if si < self.state.contact_normal_forces.len() {
                    self.state.contact_normal_forces[si] = nf;
                }
            }
            if n_con > 0 {
                for i in 0..total {
                    if i < nb && self.graph.bodies[i].is_fixed {
                        continue;
                    }
                    if dv_lcp[i].max_abs() > 0.0 {
                        v_fin[i] += dv_lcp[i];
                        x_trial[i] = x_n[i].integrate_decoupled(v_fin[i], dt);
                        dx_fin[i] = Vec6::from_parts(
                            x_trial[i].pos - x_n[i].pos,
                            x_trial[i].rot.mul(x_n[i].rot.conj()).log(),
                        );
                        // Keep smooth pre-impulse `a_fin[i]` (do not inject Dirac spike `dv_lcp/dt`
                        // into Newmark's next-step trapezoidal predictor).
                    }
                }
            }
            (n_con, 0, 0, iters, res)
        } else {
            let con_final = self.evaluate_contacts(&x_trial, &v_fin, dt, 1.0, c_v);
            self.state.contact_states = con_final.next_site_states;
            self.state.contact_normal_forces = con_final.site_normal_forces;
            self.state.contact_slips = con_final.site_slips;
            (
                con_final.active_site_contacts,
                con_final.active_beam_contacts,
                con_final.active_self_contacts,
                0,
                0.0,
            )
        };

        self.state.prev_positions = x_n.iter().map(|p| p.pos).collect();
        self.state.prev_vels = v_n;
        self.state.prev_dx = dx_fin;
        self.state.poses = x_trial;
        self.state.vels = v_fin;
        self.state.accels = a_fin;

        // Commit DER states & update Bishop frames
        for (s, br) in self.beams.iter().enumerate() {
            if let (BeamRuntime::Der(data), Some(mut dst), Some(ref dn)) =
                (br, der_trial[s].take(), &der_n[s])
            {
                let map = &self.analysis.indexing.beam_maps[s];
                self.state.prev_der_theta_vel[s].clone_from(&dn.theta_vel);
                self.state.prev_der_dtheta[s].clone_from(&dth_fin[s]);
                self.state.der_theta_accels[s].clone_from(&th_a_fin[s]);
                dst.theta_vel.clone_from(&th_v_fin[s]);
                data.update_bishop_frames_and_sync_poses(map, &mut self.state.poses, &mut dst);
                self.state.der_states[s] = Some(dst);
            }
        }

        let mut max_q_err = 0.0_f64;
        for p in &mut self.state.poses {
            p.rot = p.rot.normalized();
            max_q_err = max_q_err.max((p.rot.norm() - 1.0).abs());
        }

        self.state.time += dt;
        self.state.step_count += 1;

        Ok(StepDiagnostics {
            step_index: self.state.step_count,
            sim_time: self.state.time,
            newton_iters,
            residual_inf: last_res_inf,
            residual_l2: last_res_l2,
            line_search_alpha: last_alpha,
            ccd_toi: ccd_toi_min,
            active_site_contacts,
            active_beam_contacts,
            active_self_contacts,
            lcp_iters,
            lcp_residual,
            max_quat_norm_err: max_q_err,
            failed: self.state.failed,
        })
    }

    /// Solve one linearized Newton tangent system `H * delta_xi = -r_k` using the exact
    /// Bipartite Direct Solver across all corotational + DER beams and all 3 Schur routes.
    #[allow(clippy::too_many_arguments)]
    fn solve_newton_linear_step(
        &self,
        x_n: &[Se3],
        x_k: &[Se3],
        v_k: &[Vec6],
        a_k: &[Vec6],
        der_k: &[Option<DerBeamState>],
        der_th_a: &[Vec<f64>],
        r_k: &[Vec6],
        con_k: &ContactEvalBundle,
        arpc_tan_k: &[Mat6],
        joint_blocks: &[(usize, usize, Mat6)],
        dt: f64,
        c_v: f64,
        c_a: f64,
        lm_shift: f64,
    ) -> Result<NewtonLinearStep, LinearSolveError> {
        let nb = self.graph.bodies.len();
        let total = self.analysis.indexing.total_nodes;

        let mut rigid_mat = RigidSchurMatrix::zeros(self.analysis.num_free_bodies);
        let mut rhs_free = vec![Vec6::ZERO; self.analysis.num_free_bodies];

        for (slot, &bi) in self.analysis.free_body_ids.iter().enumerate() {
            let m6 = self.body_inertia_mat6(bi, x_k[bi].rot);
            let alpha = self.graph.bodies[bi].rayleigh_alpha;
            let dth = x_k[bi].rot.mul(x_n[bi].rot.conj()).log();
            let dexp_inv = so3_dexp_inv(dth);
            let mut lhs = con_k.body_k_tan[bi] + arpc_tan_k[bi];
            if self.config.world_fixed_inertia {
                lhs += (c_a + alpha * c_v) * m6;
            } else {
                let iw = m6.block(1, 1);
                let w = v_k[bi].ang();
                let l_vec = iw.mul_vec(w);
                let lin_blk = (c_a + alpha * c_v) * m6.block(0, 0);
                // Exact Lie-group tangent of world angular inertia/momentum wrt left perturbation delta_theta:
                // d(Delta theta)/d(delta_theta) = dexp_inv(Delta theta)
                let mut rot_blk = (c_a + alpha * c_v) * iw.mul_mat(dexp_inv);
                let d_l_dtheta = -(l_vec.skew()) + iw.mul_mat(w.skew());
                if matches!(self.config.integrator, IntegratorKind::DampedNewtonEuler) {
                    rot_blk += d_l_dtheta * (1.0 / dt);
                } else {
                    // Exact derivative of I_w(q) * a + w x (I_w(q) * w)
                    let a_ang = a_k[bi].ang();
                    let ia = iw.mul_vec(a_ang);
                    let d_ia_dq = -(ia.skew()) + iw.mul_mat(a_ang.skew());
                    let d_gyro_dw = -(l_vec.skew()) + w.skew().mul_mat(iw);
                    rot_blk += d_ia_dq + d_gyro_dw.mul_mat(dexp_inv) * c_v;
                }
                lhs += Mat6::from_blocks(lin_blk, Mat3::ZERO, Mat3::ZERO, rot_blk);
            }
            rigid_mat.add_block(slot, slot, lhs);
            rhs_free[slot] = -r_k[bi];
        }

        // Only scatter joint blocks that lie within the symbolic rigid sparsity graph
        for &(ba, bb, blk) in joint_blocks {
            if let (Some(u), Some(v)) = (
                self.analysis.body_to_free_slot[ba],
                self.analysis.body_to_free_slot[bb],
            ) {
                if u == v || self.joint_edge_in_rigid_graph(ba, bb) {
                    rigid_mat.add_block(u, v, blk);
                }
            }
        }

        let mut beam_factors: Vec<BeamFactorKind> = Vec::with_capacity(self.beams.len());
        let mut beam_contribs: Vec<BeamSchurContribution> = Vec::with_capacity(self.beams.len());

        for (s, br) in self.beams.iter().enumerate() {
            let map = &self.analysis.indexing.beam_maps[s];
            match br {
                BeamRuntime::Corotational(data) => {
                    let ev = data.evaluate(map, x_k, self.config.use_strain_jacobian);
                    let cm = c_a + data.rayleigh_alpha * c_v;
                    let ck = 1.0 + data.rayleigh_beta * c_v;
                    let elem_a: Vec<Mat12> = (0..data.n_elements)
                        .map(|e| ev.elem_m[e] * cm + ev.elem_k[e] * ck)
                        .collect();
                    let int_rhs: Vec<Vec6> = map.free_node_ids.iter().map(|&nid| -r_k[nid]).collect();
                    let int_k6: Vec<Mat6> = con_k.beam_int_k[s]
                        .iter()
                        .map(|&k3| {
                            let mut blk =
                                Mat6::from_blocks(k3, Mat3::ZERO, Mat3::ZERO, Mat3::ZERO);
                            if lm_shift > 0.0 {
                                blk += lm_shift * Mat6::IDENTITY;
                            }
                            blk
                        })
                        .collect();
                    let (fac, contrib) = CorotationalBeamFactor::factor_and_condense(
                        data.beam_id,
                        data.body_a,
                        data.body_b,
                        &elem_a,
                        &int_k6,
                        &int_rhs,
                    )?;
                    beam_factors.push(BeamFactorKind::Corotational(Box::new(fac)));
                    beam_contribs.push(contrib);
                }
                BeamRuntime::Der(data) => {
                    let dst = der_k[s].as_ref().unwrap();
                    let cm = c_a + data.rayleigh_alpha * c_v;
                    let ck = 1.0 + data.rayleigh_beta * c_v;
                    let k_int_lm: Vec<Mat3> = if lm_shift > 0.0 {
                        con_k.beam_int_k[s]
                            .iter()
                            .map(|&k| k + lm_shift * Mat3::IDENTITY)
                            .collect()
                    } else {
                        con_k.beam_int_k[s].clone()
                    };
                    let out = data.assemble_and_condense(
                        map,
                        x_k,
                        v_k,
                        Some(a_k),
                        Some(&der_th_a[s]),
                        dst,
                        self.graph.gravity,
                        &con_k.beam_int_forces[s],
                        &k_int_lm,
                        dt,
                        cm,
                        ck,
                        false,
                        self.config.project_der_hessian_spd,
                    )?;
                    beam_factors.push(BeamFactorKind::Der(Box::new(out.factor)));
                    beam_contribs.push(out.contrib);
                }
            }
        }

        for contrib in &beam_contribs {
            rigid_mat.scatter_beam(&mut rhs_free, contrib, &self.analysis.body_to_free_slot);
        }
        if lm_shift > 0.0 {
            rigid_mat.add_diagonal_shift(lm_shift);
        }

        let rigid_factor = self.factorize_rigid_schur(&rigid_mat)?;
        let d_free = rigid_factor.solve(&rhs_free)?;

        let mut delta_xi = vec![Vec6::ZERO; total];
        for (slot, &bi) in self.analysis.free_body_ids.iter().enumerate() {
            delta_xi[bi] = d_free[slot];
        }

        let mut delta_der_th: Vec<Option<Vec<f64>>> = vec![None; self.beams.len()];
        for (s, fac) in beam_factors.iter().enumerate() {
            let map = &self.analysis.indexing.beam_maps[s];
            let va = delta_xi[map.node_a];
            let vb = map.node_b.map_or(Vec6::ZERO, |bb| delta_xi[bb]);
            match fac {
                BeamFactorKind::Corotational(cf) => {
                    let di = cf.back_substitute(va, vb);
                    for (li, &nid) in map.free_node_ids.iter().enumerate() {
                        delta_xi[nid] = di[li];
                    }
                }
                BeamFactorKind::Der(df) => {
                    let (dv, dth) = df.back_substitute(va, vb);
                    for (li, &nid) in map.free_node_ids.iter().enumerate() {
                        delta_xi[nid] = Vec6::from_parts(dv[li], Vec3::ZERO);
                    }
                    delta_der_th[s] = Some(dth);
                }
            }
        }
        debug_assert_eq!(nb, self.graph.bodies.len());
        Ok((delta_xi, delta_der_th))
    }

    /// Check whether `(ba, bb)` is already present in the compile-time rigid Schur sparsity graph.
    fn joint_edge_in_rigid_graph(&self, ba: usize, bb: usize) -> bool {
        if ba == bb {
            return true;
        }
        let in_beam = self
            .graph
            .beams
            .iter()
            .any(|b| (b.body_a == ba && b.body_b == Some(bb)) || (b.body_a == bb && b.body_b == Some(ba)));
        if in_beam {
            return true;
        }
        self.graph.joints.iter().any(|j| {
            j.implicit_coupling
                && ((j.body_a == ba && j.body_b == bb) || (j.body_a == bb && j.body_b == ba))
        })
    }

    /// Factor the velocity-space bipartite operator `H_vel = M + scale_c * C + scale_k * K`
    /// for exact Delassus compliance solves `W = J_c H_vel^{-1} J_c^T` after a nonlinear Newton step.
    fn factor_velocity_bipartite_operator(
        &self,
        poses: &[Se3],
        vels: &[Vec6],
        der_states: &[Option<DerBeamState>],
        dt: f64,
        scale_c: f64,
        scale_k: f64,
    ) -> Result<FactoredBipartiteSystem, LinearSolveError> {
        let nb = self.graph.bodies.len();
        let mut rigid_mat = RigidSchurMatrix::zeros(self.analysis.num_free_bodies);
        let mut rhs_dummy = vec![Vec6::ZERO; self.analysis.num_free_bodies];

        for (slot, &bi) in self.analysis.free_body_ids.iter().enumerate() {
            let m6 = self.body_inertia_mat6(bi, poses[bi].rot);
            let alpha = self.graph.bodies[bi].rayleigh_alpha;
            rigid_mat.add_block(slot, slot, (1.0 + scale_c * alpha) * m6);
        }

        let mut act_scratch = self.state.actuation.clone();
        let body_masses = self.body_masses();
        let act_ev = evaluate_actuation_with_masses(
            nb,
            &self.graph.tendons,
            &self.graph.joints,
            &self.rest_poses,
            poses,
            vels,
            &mut act_scratch,
            scale_k,
            scale_c,
            false,
            false,
            self.config.joint_stribeck_friction,
            Some(&body_masses),
            dt,
        );
        for &(ba, bb, blk) in &act_ev.implicit_blocks {
            if let (Some(u), Some(v)) = (
                self.analysis.body_to_free_slot[ba],
                self.analysis.body_to_free_slot[bb],
            ) {
                if u == v || self.joint_edge_in_rigid_graph(ba, bb) {
                    rigid_mat.add_block(u, v, blk);
                }
            }
        }

        let mut beam_factors = Vec::with_capacity(self.beams.len());
        let mut beam_contribs = Vec::with_capacity(self.beams.len());
        for (s, br) in self.beams.iter().enumerate() {
            let map = &self.analysis.indexing.beam_maps[s];
            let n_free = map.free_node_ids.len();
            match br {
                BeamRuntime::Corotational(data) => {
                    let ev = data.evaluate(map, poses, self.config.use_strain_jacobian);
                    let cm = 1.0 + scale_c * data.rayleigh_alpha;
                    let ck = scale_k + scale_c * data.rayleigh_beta;
                    let elem_a: Vec<Mat12> = (0..data.n_elements)
                        .map(|e| ev.elem_m[e] * cm + ev.elem_k[e] * ck)
                        .collect();
                    let zero_k = vec![Mat6::ZERO; n_free];
                    let zero_rhs = vec![Vec6::ZERO; n_free];
                    let (fac, contrib) = CorotationalBeamFactor::factor_and_condense(
                        data.beam_id,
                        data.body_a,
                        data.body_b,
                        &elem_a,
                        &zero_k,
                        &zero_rhs,
                    )?;
                    beam_factors.push(BeamFactorKind::Corotational(Box::new(fac)));
                    beam_contribs.push(contrib);
                }
                BeamRuntime::Der(data) => {
                    let dst = der_states[s].as_ref().expect("DER beam must have state");
                    let cm = 1.0 + scale_c * data.rayleigh_alpha;
                    let ck = scale_k + scale_c * data.rayleigh_beta;
                    let zero_f = vec![Vec3::ZERO; n_free];
                    let zero_k = vec![Mat3::ZERO; n_free];
                    let out = data.assemble_and_condense(
                        map,
                        poses,
                        vels,
                        None,
                        None,
                        dst,
                        self.graph.gravity,
                        &zero_f,
                        &zero_k,
                        dt,
                        cm,
                        ck,
                        true,
                        self.config.project_der_hessian_spd,
                    )?;
                    beam_factors.push(BeamFactorKind::Der(Box::new(out.factor)));
                    beam_contribs.push(out.contrib);
                }
            }
        }

        for contrib in &beam_contribs {
            rigid_mat.scatter_beam(&mut rhs_dummy, contrib, &self.analysis.body_to_free_slot);
        }
        if self.config.lm_initial > 0.0 {
            rigid_mat.add_diagonal_shift(self.config.lm_initial);
        }
        let rigid_factor = self.factorize_rigid_schur(&rigid_mat)?;
        Ok(FactoredBipartiteSystem {
            beam_factors,
            rigid_factor,
        })
    }

    /// Compute global conservative CCD safe step fraction `t_safe in (0, 1]` across
    /// beam self-contact segments and SDF-barrier contact sites/interior nodes (each point
    /// evaluated with its own individual radius `barrier.thickness.max(r_i)`).
    ///
    /// When `config.use_tight_inclusion_ccd == true`, uses provably conservative
    /// Tight Inclusion Edge-Edge CCD (Wang et al. TOG 2021) for self-contact segments
    /// and Tight Inclusion Point-Triangle CCD against all registered static `TriMeshSdf`s.
    fn compute_global_ccd_toi(&self, x0: &[Se3], x1: &[Se3]) -> f64 {
        let mut t_safe = 1.0_f64;
        if self.config.enable_self_contact && !self.self_contact_segments.is_empty() {
            let p0: Vec<Vec3> = x0.iter().map(|s| s.pos).collect();
            let p1: Vec<Vec3> = x1.iter().map(|s| s.pos).collect();
            let t_seg = if self.config.use_tight_inclusion_ccd {
                let ti_cfg = TightInclusionConfig {
                    min_dist: self.config.self_contact_barrier.thickness,
                    ..TightInclusionConfig::default()
                };
                let raw_toi = compute_safe_step_segments_tight_inclusion(
                    &p0,
                    &p1,
                    &self.self_contact_segments,
                    &ti_cfg,
                );
                if raw_toi < 1.0 {
                    // Conservative 0.9 margin matching ACCD so IPC log-barrier stays strictly finite
                    (0.9 * raw_toi).max(0.0)
                } else {
                    1.0
                }
            } else {
                compute_safe_step_segments(
                    &p0,
                    &p1,
                    &self.self_contact_segments,
                    self.config.self_contact_barrier.thickness,
                    0.1,
                )
            };
            t_safe = t_safe.min(t_seg);
            if t_safe <= 0.0 {
                return 0.0;
            }
        }
        if let ContactBackend::SdfBarrier {
            barrier,
            use_ccd: true,
            ..
        } = &self.config.contact
        {
            let check_point = |p0: Vec3, p1: Vec3, xi: f64, mut t_cur: f64| -> f64 {
                #[cfg(feature = "parry-mesh")]
                if self.config.use_tight_inclusion_ccd && !self.static_trimeshes.is_empty() {
                    let ti_cfg = TightInclusionConfig {
                        min_dist: xi,
                        ..TightInclusionConfig::default()
                    };
                    let swept = Aabb::from_point(p0)
                        .union(Aabb::from_point(p1))
                        .expand(xi + 1e-4);
                    for mesh in &self.static_trimeshes {
                        for tri_idx in mesh.triangles_near_aabb(swept) {
                            let [t0, t1, t2] = mesh.world_triangle(tri_idx);
                            if let Some(toi) = tight_inclusion_point_triangle_ccd(
                                p0, t0, t1, t2, p1, t0, t1, t2, t_cur, &ti_cfg,
                            ) {
                                // Conservative 0.9 margin matching ACCD so barrier remains finite
                                let safe_toi = (0.9 * toi).max(0.0);
                                if safe_toi < t_cur {
                                    t_cur = safe_toi;
                                    if t_cur <= 0.0 {
                                        return 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(toi) = accd_point_sdf(&self.sdf, p0, p1 - p0, xi, 0.1, t_cur) {
                    if toi < t_cur {
                        t_cur = toi;
                    }
                }
                t_cur
            };

            for site in &self.graph.contact_sites {
                let (p0, p1) = match site.attachment {
                    ContactAttachment::RigidBody { body_id } => (
                        x0[body_id].transform_point(site.local_pos),
                        x1[body_id].transform_point(site.local_pos),
                    ),
                    ContactAttachment::BeamInteriorNode {
                        beam_id,
                        node_local_idx,
                    } => {
                        let nid =
                            self.analysis.indexing.beam_maps[beam_id].free_node_ids[node_local_idx];
                        (
                            x0[nid].transform_point(site.local_pos),
                            x1[nid].transform_point(site.local_pos),
                        )
                    }
                };
                let xi = barrier.thickness.max(site.radius);
                t_safe = check_point(p0, p1, xi, t_safe);
                if t_safe <= 0.0 {
                    return 0.0;
                }
            }
            for (s, spec) in self.graph.beams.iter().enumerate() {
                if let Some(r) = spec.interior_node_contact_radius {
                    let xi = barrier.thickness.max(r);
                    for &nid in &self.analysis.indexing.beam_maps[s].free_node_ids {
                        t_safe = check_point(x0[nid].pos, x1[nid].pos, xi, t_safe);
                        if t_safe <= 0.0 {
                            return 0.0;
                        }
                    }
                }
            }
        }
        t_safe
    }

    /// Engine B: Exact Compliance-Based Frictional Contact LCP / Delassus-Schur correction
    /// (`W = J_c H^{-1} J_c^T`, Shen et al. IJRR 2026 & Moreau Contact-Implicit).
    #[allow(clippy::type_complexity)]
    fn solve_delassus_lcp_correction(
        &mut self,
        factored: &FactoredBipartiteSystem,
        v_free: &[Vec6],
        dt: f64,
        lcp_cfg: &LcpSolverConfig,
        dedup_radius: f64,
        baumgarte_beta: f64,
    ) -> Result<(Vec<Vec6>, usize, usize, f64, Vec<(usize, f64)>), LinearSolveError> {
        let total = self.analysis.indexing.total_nodes;

        struct Candidate {
            node_id: usize,
            arm: Vec3,
            pos: Vec3,
            pen: f64,
            normal: Vec3,
            mu_1: f64,
            mu_2: f64,
            site_idx: Option<usize>,
            solref_imp: Option<SolRefImp>,
        }

        let mut cands = Vec::new();
        for (si, site) in self.graph.contact_sites.iter().enumerate() {
            let (nid, arm, pos) = match site.attachment {
                ContactAttachment::RigidBody { body_id } => {
                    let off = self.state.poses[body_id].rot.rotate(site.local_pos);
                    let c_pos = self.state.poses[body_id].pos + off;
                    let q0 = self.sdf.query(c_pos);
                    let rim = -site.radius * q0.normal.normalized_or(Vec3::EZ);
                    let arm = off + rim;
                    (body_id, arm, self.state.poses[body_id].pos + arm)
                }
                ContactAttachment::BeamInteriorNode {
                    beam_id,
                    node_local_idx,
                } => {
                    let nid = self.analysis.indexing.beam_maps[beam_id].free_node_ids[node_local_idx];
                    let pos = self.state.poses[nid].pos;
                    (nid, Vec3::ZERO, pos)
                }
            };
            let q = self.sdf.query(pos);
            let pen = match site.attachment {
                ContactAttachment::RigidBody { .. } => -q.phi,
                ContactAttachment::BeamInteriorNode { .. } => site.radius - q.phi,
            };
            if pen > 0.0 {
                let mat = self.sdf.materials.get(q.mat_id);
                let (mu1, mu2) = match site.mode {
                    ContactMode::RatchetWheel {
                        mu_fwd,
                        mu_lat_ratio,
                        ..
                    } => (mu_fwd, mu_fwd * mu_lat_ratio),
                    ContactMode::AnisotropicScale { mu_fwd, mu_lat, .. } => (mu_fwd, mu_lat),
                    ContactMode::IsotropicPad { mu, .. } => {
                        let m = if mu > 0.0 { mu } else { mat.mu };
                        (m, m)
                    }
                };
                cands.push(Candidate {
                    node_id: nid,
                    arm,
                    pos,
                    pen,
                    normal: q.normal.normalized_or(Vec3::EZ),
                    mu_1: mu1,
                    mu_2: mu2,
                    site_idx: Some(si),
                    solref_imp: site.solref_imp,
                });
            }
        }
        // Beam interior contact nodes
        for (s, spec) in self.graph.beams.iter().enumerate() {
            let Some(radius) = spec.interior_node_contact_radius else {
                continue;
            };
            for &nid in &self.analysis.indexing.beam_maps[s].free_node_ids {
                let pos = self.state.poses[nid].pos;
                let q = self.sdf.query(pos);
                let pen = radius - q.phi;
                if pen > 0.0 {
                    let mat = self.sdf.materials.get(q.mat_id);
                    cands.push(Candidate {
                        node_id: nid,
                        arm: Vec3::ZERO,
                        pos,
                        pen,
                        normal: q.normal.normalized_or(Vec3::EZ),
                        mu_1: mat.mu,
                        mu_2: mat.mu,
                        site_idx: None,
                        solref_imp: None,
                    });
                }
            }
        }

        if cands.is_empty() {
            self.state.lcp_warm_start.clear();
            return Ok((vec![Vec6::ZERO; total], 0, 0, 0.0, Vec::new()));
        }

        // Deduplicate clustered redundant contacts (Shen et al. IJRR 2026)
        let kept = if dedup_radius > 0.0 {
            let pts: Vec<Vec3> = cands.iter().map(|c| c.pos).collect();
            let pens: Vec<f64> = cands.iter().map(|c| c.pen).collect();
            DelassusSystem::deduplicate_contacts(&pts, &pens, dedup_radius)
        } else {
            (0..cands.len()).collect()
        };

        let m = kept.len();
        let mut constraints = Vec::with_capacity(m);
        let mut jac_rows: Vec<(usize, [Vec6; 3])> = Vec::with_capacity(m);

        for &idx in &kept {
            let c = &cands[idx];
            let mut lc = LcpContactConstraint::from_normal(c.normal, c.mu_1, [0.0; 3]);
            lc.mu_2 = c.mu_2;
            let j_n = Vec6::from_parts(lc.normal, c.arm.cross(lc.normal));
            let j_t1 = Vec6::from_parts(lc.t1, c.arm.cross(lc.t1));
            let j_t2 = Vec6::from_parts(lc.t2, c.arm.cross(lc.t2));
            let vf = v_free[c.node_id];
            lc.free_vel = [j_n.dot(vf), j_t1.dot(vf), j_t2.dot(vf)];
            if let Some(sri) = c.solref_imp {
                // MuJoCo SolRefImp convention: `gap` is signed separation in meters (`-pen <= 0`)
                // and `pre_vel_n` is pre-step normal velocity `v_n^-` for exact closed-loop damping.
                let vn_pre = j_n.dot(self.state.vels[c.node_id]);
                lc.gap = -c.pen;
                lc = lc.with_solref_imp(sri).with_pre_vel_n(vn_pre);
            } else {
                // Baumgarte penetration stabilization: bias normal free velocity inward by -beta * pen / dt
                lc.gap = -(baumgarte_beta.clamp(0.0, 1.0) * c.pen / dt);
            }
            constraints.push(lc);
            jac_rows.push((c.node_id, [j_n, j_t1, j_t2]));
        }

        // Assemble exact 3m x 3m Delassus compliance matrix W = J_c H^{-1} J_c^T
        // using cached factored bipartite operator (ZERO re-factorization!)
        let dim = 3 * m;
        let mut w_mat = vec![0.0; dim * dim];
        let mut rhs_col = vec![Vec6::ZERO; total];
        for i in 0..m {
            let (nid_i, j_i) = jac_rows[i];
            for d in 0..3 {
                rhs_col[nid_i] = j_i[d];
                let dv_col = factored.solve_rhs(&rhs_col, &self.analysis)?;
                rhs_col[nid_i] = Vec6::ZERO;
                let col_idx = 3 * i + d;
                for j in 0..m {
                    let (nid_j, j_j) = jac_rows[j];
                    let dv_j = dv_col[nid_j];
                    for r in 0..3 {
                        w_mat[(3 * j + r) * dim + col_idx] = j_j[r].dot(dv_j);
                    }
                }
            }
        }
        // Symmetrize W numerically to suppress FP64 roundoff asymmetry
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (w_mat[i * dim + j] + w_mat[j * dim + i]);
                w_mat[i * dim + j] = avg;
                w_mat[j * dim + i] = avg;
            }
        }

        let sys = DelassusSystem::new(constraints, w_mat);
        let warm = if self.state.lcp_warm_start.len() == m {
            Some(self.state.lcp_warm_start.as_slice())
        } else {
            None
        };
        let mut lcp_eff = *lcp_cfg;
        lcp_eff.solref_dt = dt;
        let sol = sys.solve(&lcp_eff, warm);
        self.state.lcp_warm_start = sol.impulses.clone();

        // Accumulate generalized contact impulses J_c^T * lambda and solve H * dv = J_c^T * lambda
        let mut imp_rhs = vec![Vec6::ZERO; total];
        let mut site_normals = Vec::new();
        for (i, &cand_idx) in kept.iter().enumerate() {
            let (nid, j_i) = jac_rows[i];
            let lam = sol.impulses[i];
            imp_rhs[nid] += j_i[0] * lam[0] + j_i[1] * lam[1] + j_i[2] * lam[2];
            if let Some(si) = cands[cand_idx].site_idx {
                site_normals.push((si, lam[0] / dt));
            }
        }
        let dv_corr = factored.solve_rhs(&imp_rhs, &self.analysis)?;
        let res = sol.normal_complementarity_res.max(sol.friction_cone_res);
        Ok((dv_corr, m, sol.iterations, res, site_normals))
    }

    /// Compute exact mechanical energy (kinetic, elastic, actuation, gravity) and exact
    /// spatial linear & angular momentum for physics budget and conservation verification.
    pub fn energy_and_momentum(&self) -> EnergyMomentumReport {
        let nb = self.graph.bodies.len();
        let g = self.graph.gravity;
        let mut kinetic = 0.0;
        let mut elastic = 0.0;
        let mut gravity_pot = 0.0;
        let mut lin_mom = Vec3::ZERO;
        let mut ang_mom = Vec3::ZERO;

        // Rigid bodies
        for bi in 0..nb {
            let body = &self.graph.bodies[bi];
            if body.is_fixed {
                continue;
            }
            let pose = self.state.poses[bi];
            let vel = self.state.vels[bi];
            let m6 = self.body_inertia_mat6(bi, pose.rot);
            let p6 = m6.mul_vec(vel);
            kinetic += 0.5 * vel.dot(p6);
            lin_mom += p6.lin();
            ang_mom += pose.pos.cross(p6.lin()) + p6.ang();
            let com_w = pose.transform_point(body.com_local);
            gravity_pot -= body.mass * g.dot(com_w);
        }

        // Beams (Corotational & DER)
        for (s, br) in self.beams.iter().enumerate() {
            let map = &self.analysis.indexing.beam_maps[s];
            match br {
                BeamRuntime::Corotational(data) => {
                    let ev = data.evaluate(map, &self.state.poses, self.config.use_strain_jacobian);
                    elastic += ev.energy;
                    for e in 0..data.n_elements {
                        let n0 = map.chain_node_ids[e];
                        let n1 = map.chain_node_ids[e + 1];
                        let v12 = Vec12::from_halves(self.state.vels[n0], self.state.vels[n1]);
                        let p12 = ev.elem_m[e].mul_vec(v12);
                        kinetic += 0.5 * v12.dot(p12);
                        lin_mom += p12.head().lin() + p12.tail().lin();
                        ang_mom += self.state.poses[n0].pos.cross(p12.head().lin())
                            + p12.head().ang()
                            + self.state.poses[n1].pos.cross(p12.tail().lin())
                            + p12.tail().ang();
                        let g12 = Vec12::from_halves(
                            Vec6::from_parts(g, Vec3::ZERO),
                            Vec6::from_parts(g, Vec3::ZERO),
                        );
                        let fg = ev.elem_m[e].mul_vec(g12);
                        gravity_pot -= self.state.poses[n0].pos.dot(fg.head().lin())
                            + self.state.poses[n1].pos.dot(fg.tail().lin());
                    }
                }
                BeamRuntime::Der(data) => {
                    if let Some(ref dst) = self.state.der_states[s] {
                        let zero_f = vec![Vec3::ZERO; map.free_node_ids.len()];
                        let zero_k = vec![Mat3::ZERO; map.free_node_ids.len()];
                        if let Ok(out) = data.assemble_and_condense(
                            map,
                            &self.state.poses,
                            &self.state.vels,
                            None,
                            None,
                            dst,
                            g,
                            &zero_f,
                            &zero_k,
                            1e-3,
                            1.0,
                            1e-6,
                            true,
                            false,
                        ) {
                            elastic += out.energy;
                        }
                        // Clamp endpoint lumped masses (matching dynamics in DerBeamData::assemble_and_condense)
                        if !self.graph.bodies[data.body_a].is_fixed && data.end_a_mass > 0.0 {
                            let xa = self.state.poses[data.body_a];
                            let off_a = xa.rot.rotate(data.clamp_a.pos);
                            let pa = xa.pos + off_a;
                            let va = self.state.vels[data.body_a].lin()
                                + self.state.vels[data.body_a].ang().cross(off_a);
                            let mom_a = va * data.end_a_mass;
                            kinetic += 0.5 * data.end_a_mass * va.norm_sq();
                            lin_mom += mom_a;
                            ang_mom += pa.cross(mom_a);
                            gravity_pot -= data.end_a_mass * g.dot(pa);
                        }
                        if let Some(bb) = data.body_b {
                            if !self.graph.bodies[bb].is_fixed && data.end_b_mass > 0.0 {
                                let xb = self.state.poses[bb];
                                let cb = data.clamp_b.unwrap_or(Se3::IDENTITY);
                                let off_b = xb.rot.rotate(cb.pos);
                                let pb = xb.pos + off_b;
                                let vb = self.state.vels[bb].lin()
                                    + self.state.vels[bb].ang().cross(off_b);
                                let mom_b = vb * data.end_b_mass;
                                kinetic += 0.5 * data.end_b_mass * vb.norm_sq();
                                lin_mom += mom_b;
                                ang_mom += pb.cross(mom_b);
                                gravity_pot -= data.end_b_mass * g.dot(pb);
                            }
                        }
                        for (li, &nid) in map.free_node_ids.iter().enumerate() {
                            let m = data.free_vertex_mass[li];
                            let v = self.state.vels[nid].lin();
                            let p = v * m;
                            kinetic += 0.5 * m * v.norm_sq();
                            lin_mom += p;
                            ang_mom += self.state.poses[nid].pos.cross(p);
                            gravity_pot -= m * g.dot(self.state.poses[nid].pos);
                        }
                        for e in 0..data.n_elements {
                            kinetic += 0.5 * data.edge_twist_inertia[e] * dst.theta_vel[e].powi(2);
                        }
                    }
                }
            }
        }

        // Actuation stored potential energy (non-mutating scratch evaluation)
        let mut act_scratch = self.state.actuation.clone();
        let body_masses = self.body_masses();
        let act_ev = evaluate_actuation_with_masses(
            nb,
            &self.graph.tendons,
            &self.graph.joints,
            &self.rest_poses,
            &self.state.poses,
            &self.state.vels,
            &mut act_scratch,
            0.0,
            0.0,
            false,
            false,
            self.config.joint_stribeck_friction,
            Some(&body_masses),
            0.0,
        );

        let total_energy = kinetic + elastic + act_ev.potential_energy + gravity_pot;
        EnergyMomentumReport {
            kinetic_energy: kinetic,
            elastic_energy: elastic,
            actuation_potential_energy: act_ev.potential_energy,
            gravity_potential_energy: gravity_pot,
            total_energy,
            linear_momentum: lin_mom,
            angular_momentum: ang_mom,
        }
    }
}
