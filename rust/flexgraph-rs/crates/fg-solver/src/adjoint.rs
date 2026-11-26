//! Differentiable Physics via Implicit Function Theorem (IFT) Reverse-Mode Adjoints
//! (DiffPD / Dojo paradigm) for `flexgraph-rs`.
//!
//! Computes exact trajectory gradients $\nabla_{u} \mathcal{L}$, $\nabla_{\theta} \mathcal{L}$,
//! and $\nabla_{(x_0, v_0)} \mathcal{L}$ across multi-step rigid-flexible continuum rollouts
//! with active contact by **reusing the cached [`FactoredBipartiteSystem`] from each forward step**
//! ($O(E + \mathrm{nnz}(L))$ back-solve per step, zero matrix re-factorization, zero unrolled
//! inner solver tape!).
//!
//! Supports all 7 [`DiffParamKind`] physical parameter families (`BeamYoungsModulus`,
//! `BeamDampingBeta`, `ContactMuFwd`, `ContactMuRev`, `ContactStiffness`, `TendonStiffness`,
//! `BodyMass`), actuator commands `u_k`, and $SE(3)$ Lie-group pose/velocity cotangents
//! using exact $\mathrm{dexp}(\Delta t\,\omega)$ and $\mathrm{Ad}_{\exp(\Delta t\,\omega)}$ maps.

use crate::actuation::{
    advance_tendon_servos, evaluate_actuation_with_masses, ActuationState,
};
use crate::continuum::DerBeamState;
use crate::integrator::{ContactBackend, FactoredBipartiteSystem, SolverError, WorldState};
use crate::world::{BeamRuntime, FlexGraphWorld};
use fg_contact::{
    eval_arpc_site, eval_arpc_site_with_mass, eval_sdf_surface_contact, implicit_wheel_contact,
    ipc_mollifier_friction_eval, ipc_point_sdf_barrier, ArpcSiteState,
};
use fg_ir::{ContactAttachment, ContactMode, DiffParamKind, DiffParamSpec};
use fg_math::{
    beam_frame_transform, beam_local_strain, beam_strain_jacobian, cubic_hermite_midpoint,
    parallel_transport, so3_dexp, Mat12, Mat3, Mat6, Quat, Se3, SpatialInertia, Vec12, Vec3, Vec6,
};
use fg_sdf::{NormalContactModel, SdfField};
use serde::{Deserialize, Serialize};

/// Differentiable scalar trajectory loss evaluated at each step `k = 0..=T`.
///
/// Return `(loss_k, dL_dposes_k, dL_dvels_k, dL_du_k)` where:
/// - `dL_dposes_k` has length `poses.len()` in left-trivialized $SE(3)$ tangent coordinates
///   (`dL/dxi` under `pose' = pose.integrate_decoupled(xi, 1.0)`),
/// - `dL_dvels_k` has length `vels.len()`,
/// - `dL_du_k` has length `controls.map_or(0, |u| u.len())` (empty at terminal step `T`).
pub trait TrajectoryLoss {
    fn step_loss_and_grad(
        &self,
        step_idx: usize,
        poses: &[Se3],
        vels: &[Vec6],
        controls: Option<&[f64]>,
    ) -> (f64, Vec<Vec6>, Vec<Vec6>, Vec<f64>);
}

/// Built-in MSE trajectory tracking + control regularization loss (ready for SysID & FFI).
///
/// Evaluates:
/// $$\mathcal{L} = \sum_{k=0}^{T} \left[ \sum_{b \in \text{bodies}} \left(
/// \frac{w_p}{2}\|p_k[b] - p^*_k[b]\|^2 + \frac{w_r}{2}\|\log(q_k[b]\,(q^*_k[b])^{-1})\|^2
/// + \frac{w_v}{2}\|v_k[b] - v^*_k[b]\|^2 \right) + \frac{w_u}{2}\|u_k\|^2 \right].$$
#[derive(Clone, Debug)]
pub struct PoseTrackingLoss {
    /// Target poses indexed `[step][body_id]` for `step in 0..=T` (only first `body_ids` tracked).
    pub target_poses: Vec<Vec<Se3>>,
    /// Optional target velocities indexed `[step][body_id]`.
    pub target_vels: Option<Vec<Vec<Vec6>>>,
    /// Which node indices to track (defaults to `0..target_poses[0].len()` when empty).
    pub tracked_nodes: Vec<usize>,
    pub weight_pos: f64,
    pub weight_rot: f64,
    pub weight_vel: f64,
    pub weight_ctrl: f64,
}

impl PoseTrackingLoss {
    /// Construct a position-and-orientation tracking loss over all rigid bodies present in `target_poses`.
    pub fn new(target_poses: Vec<Vec<Se3>>, weight_pos: f64, weight_rot: f64) -> Self {
        Self {
            target_poses,
            target_vels: None,
            tracked_nodes: Vec::new(),
            weight_pos,
            weight_rot,
            weight_vel: 0.0,
            weight_ctrl: 0.0,
        }
    }
}

impl TrajectoryLoss for PoseTrackingLoss {
    fn step_loss_and_grad(
        &self,
        step_idx: usize,
        poses: &[Se3],
        vels: &[Vec6],
        controls: Option<&[f64]>,
    ) -> (f64, Vec<Vec6>, Vec<Vec6>, Vec<f64>) {
        let mut loss = 0.0;
        let mut gx = vec![Vec6::ZERO; poses.len()];
        let mut gv = vec![Vec6::ZERO; vels.len()];
        let mut gu = controls.map_or_else(Vec::new, |u| vec![0.0; u.len()]);

        if let Some(targets_k) = self.target_poses.get(step_idx) {
            let n_track = if self.tracked_nodes.is_empty() {
                targets_k.len().min(poses.len())
            } else {
                self.tracked_nodes.len()
            };
            for idx in 0..n_track {
                let nid = if self.tracked_nodes.is_empty() {
                    idx
                } else {
                    self.tracked_nodes[idx]
                };
                if nid >= poses.len() || idx >= targets_k.len() {
                    continue;
                }
                let tgt = targets_k[idx];
                if self.weight_pos > 0.0 {
                    let dp = poses[nid].pos - tgt.pos;
                    loss += 0.5 * self.weight_pos * dp.norm_sq();
                    let g_lin = dp * self.weight_pos;
                    gx[nid] += Vec6::from_parts(g_lin, Vec3::ZERO);
                }
                if self.weight_rot > 0.0 {
                    // Left-invariant rotation error theta = log(q * q_tgt^{-1});
                    // since dexp_inv(theta)^T * theta == theta identically on SO(3),
                    // d/dphi (0.5 ||theta||^2) = theta exactly!
                    let theta = poses[nid].rot.mul(tgt.rot.conj()).log();
                    loss += 0.5 * self.weight_rot * theta.norm_sq();
                    let g_ang = theta * self.weight_rot;
                    gx[nid] += Vec6::from_parts(Vec3::ZERO, g_ang);
                }
                if self.weight_vel > 0.0 {
                    if let Some(ref tv_all) = self.target_vels {
                        if let Some(tv_k) = tv_all.get(step_idx) {
                            if idx < tv_k.len() {
                                let dv = vels[nid] - tv_k[idx];
                                loss += 0.5 * self.weight_vel * dv.dot(dv);
                                gv[nid] += dv * self.weight_vel;
                            }
                        }
                    }
                }
            }
        }

        if self.weight_ctrl > 0.0 {
            if let Some(u) = controls {
                for (i, &ui) in u.iter().enumerate() {
                    loss += 0.5 * self.weight_ctrl * ui * ui;
                    gu[i] = self.weight_ctrl * ui;
                }
            }
        }

        (loss, gx, gv, gu)
    }
}

/// Output of [`compute_trajectory_adjoint`]: scalar trajectory loss and exact IFT gradients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdjointResult {
    /// Total scalar trajectory loss $\mathcal{L}$.
    pub loss: f64,
    /// Reverse-mode gradient w.r.t. per-step actuator controls `grad_controls[step][act]`.
    pub grad_controls: Vec<Vec<f64>>,
    /// Reverse-mode gradient w.r.t. each [`DiffParamSpec`] (`dL/dtheta_p * spec.scale`).
    pub grad_params: Vec<f64>,
    /// Reverse-mode gradient w.r.t. initial node poses `x_0` (left-trivialized $SE(3)$ cotangent).
    pub grad_init_poses: Vec<Vec6>,
    /// Reverse-mode gradient w.r.t. initial node velocities `v_0`.
    pub grad_init_vels: Vec<Vec6>,
    /// Final forward state at step `T`.
    pub final_state: WorldState,
}

/// Read the current physical scalar value of a [`DiffParamSpec`] from `world`.
pub fn get_diff_param_value(world: &FlexGraphWorld, spec: &DiffParamSpec) -> f64 {
    let first_id = spec.target_ids.first().copied().unwrap_or(0);
    match spec.kind {
        DiffParamKind::BeamYoungsModulus => world
            .graph
            .beams
            .get(first_id)
            .map_or(0.0, |b| b.material.youngs_modulus),
        DiffParamKind::BeamDampingBeta => world
            .graph
            .beams
            .get(first_id)
            .map_or(0.0, |b| b.material.rayleigh_beta),
        DiffParamKind::ContactMuFwd => {
            if let ContactBackend::SdfBarrier { mu, .. } = world.config.contact {
                return mu;
            }
            world
                .graph
                .contact_sites
                .get(first_id)
                .map_or(0.0, |s| match s.mode {
                    ContactMode::RatchetWheel { mu_fwd, .. }
                    | ContactMode::AnisotropicScale { mu_fwd, .. } => mu_fwd,
                    ContactMode::IsotropicPad { mu, .. } => {
                        if mu > 0.0 {
                            mu
                        } else {
                            world.sdf.materials.get(s.material_profile_id).mu
                        }
                    }
                })
        }
        DiffParamKind::ContactMuRev => world
            .graph
            .contact_sites
            .get(first_id)
            .map_or(0.0, |s| match s.mode {
                ContactMode::AnisotropicScale { mu_rev, .. } => mu_rev,
                _ => 0.0,
            }),
        DiffParamKind::ContactStiffness => match world.config.contact {
            ContactBackend::GpuCoreKelvinParity { kn, .. } => kn,
            ContactBackend::SdfBarrier { barrier, .. } => barrier.stiffness,
            _ => {
                let mat_id = world
                    .graph
                    .contact_sites
                    .get(first_id)
                    .map_or(0, |s| s.material_profile_id);
                match world.sdf.materials.get(mat_id).normal_model {
                    NormalContactModel::KelvinVoigt { kn, .. } => kn,
                    NormalContactModel::HuntCrossleyBekker { hc_k, .. } => hc_k,
                }
            }
        },
        DiffParamKind::TendonStiffness => world
            .graph
            .tendons
            .get(first_id)
            .map_or(0.0, |t| t.stiffness),
        DiffParamKind::BodyMass => world.graph.bodies.get(first_id).map_or(0.0, |b| b.mass),
    }
}

/// Set the physical scalar value of a [`DiffParamSpec`] on `world` (respecting `spec.bounds`
/// when `clamp_bounds == true` and automatically rebuilding cached beam matrices if needed).
pub fn set_diff_param_value(
    world: &mut FlexGraphWorld,
    spec: &DiffParamSpec,
    raw_value: f64,
    clamp_bounds: bool,
) {
    let val = if clamp_bounds {
        if let Some([lo, hi]) = spec.bounds {
            raw_value.clamp(lo, hi)
        } else {
            raw_value
        }
    } else {
        raw_value
    };

    match spec.kind {
        DiffParamKind::BeamYoungsModulus => {
            let ids = resolve_targets(&spec.target_ids, world.graph.beams.len());
            for id in ids {
                world.graph.beams[id].material.youngs_modulus = val;
            }
            world.rebuild_beam_data();
        }
        DiffParamKind::BeamDampingBeta => {
            let ids = resolve_targets(&spec.target_ids, world.graph.beams.len());
            for id in ids {
                world.graph.beams[id].material.rayleigh_beta = val;
            }
            world.rebuild_beam_data();
        }
        DiffParamKind::ContactMuFwd => {
            if let ContactBackend::SdfBarrier { ref mut mu, .. } = world.config.contact {
                *mu = val;
                return;
            }
            let ids = resolve_targets(&spec.target_ids, world.graph.contact_sites.len());
            for id in ids {
                let mat_id = world.graph.contact_sites[id].material_profile_id;
                match &mut world.graph.contact_sites[id].mode {
                    ContactMode::RatchetWheel { mu_fwd, .. }
                    | ContactMode::AnisotropicScale { mu_fwd, .. } => *mu_fwd = val,
                    ContactMode::IsotropicPad { mu, .. } => {
                        if *mu > 0.0 {
                            *mu = val;
                        } else if let Some(mat) = world.sdf.materials.get_mut(mat_id) {
                            mat.mu = val;
                        }
                    }
                }
            }
        }
        DiffParamKind::ContactMuRev => {
            let ids = resolve_targets(&spec.target_ids, world.graph.contact_sites.len());
            for id in ids {
                if let ContactMode::AnisotropicScale { mu_rev, .. } =
                    &mut world.graph.contact_sites[id].mode
                {
                    *mu_rev = val;
                }
            }
        }
        DiffParamKind::ContactStiffness => match &mut world.config.contact {
            ContactBackend::GpuCoreKelvinParity { kn, .. } => *kn = val,
            ContactBackend::SdfBarrier { barrier, .. } => barrier.stiffness = val,
            _ => {
                let mat_ids: Vec<usize> = if spec.target_ids.is_empty() {
                    (0..world.sdf.materials.len()).collect()
                } else {
                    let mut ms: Vec<usize> = spec
                        .target_ids
                        .iter()
                        .filter_map(|&si| {
                            world
                                .graph
                                .contact_sites
                                .get(si)
                                .map(|s| s.material_profile_id)
                        })
                        .collect();
                    ms.sort_unstable();
                    ms.dedup();
                    ms
                };
                for mid in mat_ids {
                    if let Some(mat) = world.sdf.materials.get_mut(mid) {
                        match &mut mat.normal_model {
                            NormalContactModel::KelvinVoigt { kn, .. } => *kn = val,
                            NormalContactModel::HuntCrossleyBekker { hc_k, .. } => *hc_k = val,
                        }
                    }
                }
            }
        },
        DiffParamKind::TendonStiffness => {
            let ids = resolve_targets(&spec.target_ids, world.graph.tendons.len());
            for id in ids {
                world.graph.tendons[id].stiffness = val;
            }
        }
        DiffParamKind::BodyMass => {
            let ids = resolve_targets(&spec.target_ids, world.graph.bodies.len());
            for id in ids {
                world.graph.bodies[id].mass = val;
            }
        }
    }
}

fn resolve_targets(target_ids: &[usize], total: usize) -> Vec<usize> {
    if target_ids.is_empty() {
        (0..total).collect()
    } else {
        target_ids.iter().copied().filter(|&i| i < total).collect()
    }
}

/// Lightweight per-step checkpoint recorded during forward rollout (zero unrolled solver tape!).
struct StepTape {
    poses_k: Vec<Se3>,
    poses_next: Vec<Se3>,
    vels_k: Vec<Vec6>,
    act_before: ActuationState,
    act_after_servo: ActuationState,
    con_before: Vec<ArpcSiteState>,
    der_states_k: Vec<Option<DerBeamState>>,
    der_states_next: Vec<Option<DerBeamState>>,
    v_next: Vec<Vec6>,
    factored: FactoredBipartiteSystem,
}

#[inline]
fn pack_arpc_state(s: ArpcSiteState) -> [f64; 7] {
    [
        s.spin,
        s.wheel_angle,
        s.memory_world.x(),
        s.memory_world.y(),
        s.memory_world.z(),
        s.ratchet_play,
        s.soil_sinkage,
    ]
}

#[inline]
fn unpack_arpc_state(a: [f64; 7]) -> ArpcSiteState {
    ArpcSiteState {
        spin: a[0],
        wheel_angle: a[1],
        memory_world: Vec3::new(a[2], a[3], a[4]),
        ratchet_play: a[5],
        soil_sinkage: a[6],
    }
}

#[inline]
fn dot7(a: [f64; 7], b: [f64; 7]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6]
}

/// Evaluate a single contact site `si` (100% consistent with `FlexGraphWorld::evaluate_contacts`)
/// returning `(body_wrench, body_k_tan, beam_force, beam_k3, next_site_state)`.
fn eval_single_site_contact(
    world: &FlexGraphWorld,
    si: usize,
    pose_node: Se3,
    vel_node: Vec6,
    prev_st: ArpcSiteState,
    dt: f64,
) -> (Vec6, Mat6, Vec3, Mat3, ArpcSiteState) {
    let site = &world.graph.contact_sites[si];
    match &world.config.contact {
        ContactBackend::None | ContactBackend::DelassusLcp { .. } => {
            (Vec6::ZERO, Mat6::ZERO, Vec3::ZERO, Mat3::ZERO, prev_st)
        }
        ContactBackend::Arpc { implicit } => match site.attachment {
            ContactAttachment::RigidBody { body_id } => {
                let ev = if site.solref_imp.is_some() {
                    eval_arpc_site_with_mass(
                        site,
                        pose_node,
                        vel_node,
                        &world.sdf,
                        &world.sdf.materials,
                        &prev_st,
                        dt,
                        *implicit,
                        world.graph.bodies[body_id].mass,
                    )
                } else {
                    eval_arpc_site(
                        site,
                        pose_node,
                        vel_node,
                        &world.sdf,
                        &world.sdf.materials,
                        &prev_st,
                        dt,
                        *implicit,
                    )
                };
                let w = if ev.active { ev.wrench } else { Vec6::ZERO };
                let k = if ev.active && *implicit {
                    ev.k_tan
                } else {
                    Mat6::ZERO
                };
                (w, k, Vec3::ZERO, Mat3::ZERO, ev.next_state)
            }
            ContactAttachment::BeamInteriorNode { .. } => {
                let pos = pose_node.transform_point(site.local_pos);
                let vel = vel_node.lin();
                let ev = eval_sdf_surface_contact(
                    pos,
                    vel,
                    site.radius,
                    &world.sdf,
                    &world.sdf.materials,
                    dt,
                    *implicit,
                    1.0,
                );
                let f = if ev.active { ev.force_world } else { Vec3::ZERO };
                let k = if ev.active && *implicit {
                    ev.k_tan_3x3
                } else {
                    Mat3::ZERO
                };
                (Vec6::ZERO, Mat6::ZERO, f, k, prev_st)
            }
        },
        ContactBackend::GpuCoreKelvinParity { kn, cn } => {
            if let ContactAttachment::RigidBody { body_id } = site.attachment {
                if let ContactMode::RatchetWheel {
                    wheel_mass,
                    rolling_resistance,
                    mu_fwd,
                    axle_axis_local,
                    ..
                } = site.mode
                {
                    let axle = pose_node.rot.rotate(axle_axis_local);
                    let up0 = Vec3::EZ;
                    let rim = -site.radius * (up0 - axle * axle.z()).normalized_or(Vec3::EZ);
                    let offset = pose_node.rot.rotate(site.local_pos);
                    let arm = offset + rim;
                    let contact_pos = pose_node.pos + arm;
                    let velocity = vel_node.lin() + vel_node.ang().cross(arm);

                    let q_sdf = world.sdf.query(contact_pos);
                    let up_g = q_sdf.normal.normalized_or(Vec3::EZ);
                    let penetration = -q_sdf.phi;
                    let forward = up_g.cross(axle).normalized_or(Vec3::EX);
                    let lateral = up_g.cross(forward);
                    let vn = velocity.dot(up_g);
                    let tv = [velocity.dot(forward), velocity.dot(lateral)];
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
                    let next_st = ArpcSiteState {
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
                    let w = if active {
                        let force =
                            forward * eval.force[0] + lateral * eval.force[1] + up_g * normal;
                        let rt = rim.cross(force);
                        let torque = offset.cross(force) + rt - axle * rt.dot(axle)
                            + axle * eval.reaction;
                        Vec6::from_parts(force, torque)
                    } else {
                        Vec6::ZERO
                    };
                    return (w, Mat6::ZERO, Vec3::ZERO, Mat3::ZERO, next_st);
                }
                let ev = if site.solref_imp.is_some() {
                    eval_arpc_site_with_mass(
                        site,
                        pose_node,
                        vel_node,
                        &world.sdf,
                        &world.sdf.materials,
                        &prev_st,
                        dt,
                        false,
                        world.graph.bodies[body_id].mass,
                    )
                } else {
                    eval_arpc_site(
                        site,
                        pose_node,
                        vel_node,
                        &world.sdf,
                        &world.sdf.materials,
                        &prev_st,
                        dt,
                        false,
                    )
                };
                let w = if ev.active { ev.wrench } else { Vec6::ZERO };
                return (w, Mat6::ZERO, Vec3::ZERO, Mat3::ZERO, ev.next_state);
            }
            (Vec6::ZERO, Mat6::ZERO, Vec3::ZERO, Mat3::ZERO, prev_st)
        }
        ContactBackend::SdfBarrier {
            barrier,
            mu,
            friction_eps_v,
            ..
        } => {
            if let ContactAttachment::RigidBody { .. } = site.attachment {
                let offset = pose_node.rot.rotate(site.local_pos);
                let p = pose_node.pos + offset;
                let mut cfg_site = *barrier;
                cfg_site.thickness = cfg_site.thickness.max(site.radius);
                let ev = ipc_point_sdf_barrier(&world.sdf, p, &cfg_site);
                if ev.active {
                    let mut f = -ev.grad;
                    let mut k3 = ev.hess_spd * (dt * dt);
                    let lam_n = f.dot(ev.normal).max(0.0);
                    if *mu > 0.0 && lam_n > 0.0 {
                        let eps_u = (friction_eps_v * dt).max(1e-9);
                        let v_pt = vel_node.lin() + vel_node.ang().cross(offset);
                        let fr = ipc_mollifier_friction_eval(v_pt * dt, ev.normal, lam_n, *mu, eps_u);
                        f -= fr.grad;
                        k3 += fr.hess * (dt * dt);
                    }
                    let w = Vec6::from_parts(f, offset.cross(f));
                    let j = Mat6::from_blocks(
                        Mat3::IDENTITY,
                        -offset.skew(),
                        Mat3::ZERO,
                        Mat3::ZERO,
                    );
                    let k6 = Mat6::from_blocks(k3, Mat3::ZERO, Mat3::ZERO, Mat3::ZERO);
                    let k_tan = j.transpose().mul_mat(k6).mul_mat(j);
                    return (w, k_tan, Vec3::ZERO, Mat3::ZERO, prev_st);
                }
            }
            (Vec6::ZERO, Mat6::ZERO, Vec3::ZERO, Mat3::ZERO, prev_st)
        }
    }
}

/// Evaluate local scalar contraction `Psi_site = dt * lam^T w - lam^T K v_next + bar_st^T next_st`.
#[allow(clippy::too_many_arguments)]
fn site_psi(
    world: &FlexGraphWorld,
    si: usize,
    pose_node: Se3,
    vel_node: Vec6,
    prev_st: ArpcSiteState,
    dt: f64,
    lam: Vec6,
    v_next: Vec6,
    bar_next_st: [f64; 7],
) -> f64 {
    let (w, k_tan, f_beam, k_beam, next_st) =
        eval_single_site_contact(world, si, pose_node, vel_node, prev_st, dt);
    let mut psi = dot7(bar_next_st, pack_arpc_state(next_st));
    match world.graph.contact_sites[si].attachment {
        ContactAttachment::RigidBody { .. } => {
            psi += dt * lam.dot(w) - lam.dot(k_tan.mul_vec(v_next));
        }
        ContactAttachment::BeamInteriorNode { .. } => {
            psi += dt * lam.lin().dot(f_beam) - lam.lin().dot(k_beam.mul_vec(v_next.lin()));
        }
    }
    psi
}

/// Evaluate local scalar contraction for a single beam interior node SDF contact.
#[allow(clippy::too_many_arguments)]
fn beam_interior_contact_psi(
    world: &FlexGraphWorld,
    radius: f64,
    pos: Vec3,
    vel: Vec3,
    dt: f64,
    lam_lin: Vec3,
    v_next_lin: Vec3,
) -> f64 {
    match &world.config.contact {
        ContactBackend::None | ContactBackend::DelassusLcp { .. } => 0.0,
        ContactBackend::SdfBarrier {
            barrier,
            mu,
            friction_eps_v,
            ..
        } => {
            let mut cfg_b = *barrier;
            cfg_b.thickness = cfg_b.thickness.max(radius);
            let ev = ipc_point_sdf_barrier(&world.sdf, pos, &cfg_b);
            if !ev.active {
                return 0.0;
            }
            let mut f = -ev.grad;
            let mut k3 = ev.hess_spd * (dt * dt);
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
                k3 += fr.hess * (dt * dt);
            }
            dt * lam_lin.dot(f) - lam_lin.dot(k3.mul_vec(v_next_lin))
        }
        ContactBackend::Arpc { implicit } => {
            let ev = eval_sdf_surface_contact(
                pos,
                vel,
                radius,
                &world.sdf,
                &world.sdf.materials,
                dt,
                *implicit,
                0.25,
            );
            if !ev.active {
                return 0.0;
            }
            let k_term = if *implicit {
                lam_lin.dot(ev.k_tan_3x3.mul_vec(v_next_lin))
            } else {
                0.0
            };
            dt * lam_lin.dot(ev.force_world) - k_term
        }
        ContactBackend::GpuCoreKelvinParity { .. } => {
            let ev = eval_sdf_surface_contact(
                pos,
                vel,
                radius,
                &world.sdf,
                &world.sdf.materials,
                dt,
                false,
                0.25,
            );
            if !ev.active {
                0.0
            } else {
                dt * lam_lin.dot(ev.force_world)
            }
        }
    }
}

/// Evaluate the smooth geometric (non-strain-differentiated) scalar contraction for a single
/// corotational beam element `(s, e)` with frozen local elastic force `f_loc = K_loc (u_k - u_0)`
/// and frozen local stiffness/mass contractions.
///
/// Separating the stiff elastic strain term `-dt * J_k^T (K_loc A_k lam)` analytically eliminates
/// catastrophic FP64 cancellation (`EA/L ~ 1e7 N/m` divided by `eps_elem`) while remaining 100%
/// exact for both `use_strain_jacobian = true` and `false`.
#[allow(clippy::too_many_arguments)]
fn corotational_elem_geom_psi(
    data: &crate::continuum::CorotationalBeamData,
    e: usize,
    x0: Se3,
    x1: Se3,
    v_rhs: Vec12,
    v_n: Vec12,
    lam: Vec12,
    f_loc: Vec12,
    k_a_v: Vec12,
    k_a_lam: Vec12,
    m_t_vrhs: Vec12,
    m_t_lam: Vec12,
    dt: f64,
    ck: f64,
    use_strain_jacobian: bool,
) -> f64 {
    let tf0 = data.elem_transforms[e][0];
    let tf1 = data.elem_transforms[e][1];
    let q0 = x0.rot.mul(tf0.rot);
    let q1 = x1.rot.mul(tf1.rot);
    let off0 = x0.rot.rotate(tf0.pos);
    let off1 = x1.rot.rotate(tf1.pos);
    let p0 = x0.pos + off0;
    let p1 = x1.pos + off1;
    let l = data.lengths[e];

    let frame = cubic_hermite_midpoint(p0, q0, p1, q1, l);
    let t_mat = beam_frame_transform(frame.rot, off0, off1);
    let a_mat: Mat12 = if use_strain_jacobian {
        beam_strain_jacobian(p0, q0, off0, p1, q1, off1, l)
    } else {
        t_mat
    };

    let m_geom = (t_mat.mul_vec(lam)).dot(m_t_vrhs) + (t_mat.mul_vec(v_rhs)).dot(m_t_lam);
    let a_lam = a_mat.mul_vec(lam);
    let a_vn = a_mat.mul_vec(v_n);
    let f_geom = -dt * a_lam.dot(f_loc);
    let k_geom = -ck * (a_lam.dot(k_a_v) + a_vn.dot(k_a_lam));
    m_geom + f_geom + k_geom
}

/// Compute exact reverse-mode IFT trajectory gradients with respect to actuator controls,
/// [`DiffParamSpec`] physical parameters, and initial conditions `(x_0, v_0)`.
///
/// Restores `world.state` cleanly to its entry state upon completion (even if an error occurs).
pub fn compute_trajectory_adjoint(
    world: &mut FlexGraphWorld,
    controls: &[Vec<f64>],
    dt: f64,
    diff_params: &[DiffParamSpec],
    loss: &dyn TrajectoryLoss,
) -> Result<AdjointResult, SolverError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(SolverError::InvalidDt(dt));
    }
    let saved_state = world.state.clone();
    let res = compute_trajectory_adjoint_inner(world, controls, dt, diff_params, loss);
    world.state = saved_state;
    res
}

fn compute_trajectory_adjoint_inner(
    world: &mut FlexGraphWorld,
    controls: &[Vec<f64>],
    dt: f64,
    diff_params: &[DiffParamSpec],
    loss: &dyn TrajectoryLoss,
) -> Result<AdjointResult, SolverError> {
    let n_steps = controls.len();
    let nb = world.graph.bodies.len();
    let total = world.analysis.indexing.total_nodes;
    let n_sites = world.graph.contact_sites.len();
    let n_tendons = world.graph.tendons.len();
    let n_joints = world.graph.joints.len();
    let n_servo_slots = world.state.actuation.tendon_servo_pos.len();
    let g6 = Vec6::from_parts(world.graph.gravity, Vec3::ZERO);

    // --- 1. Forward Pass: Rollout trajectory & cache per-step factored bipartite operators ---
    let mut total_loss = 0.0_f64;
    let mut step_loss_grads: Vec<(Vec<Vec6>, Vec<Vec6>, Vec<f64>)> = Vec::with_capacity(n_steps);
    let mut tapes: Vec<StepTape> = Vec::with_capacity(n_steps);

    for (k, u_k) in controls.iter().enumerate() {
        let (lk, gxk, gvk, guk) =
            loss.step_loss_and_grad(k, &world.state.poses, &world.state.vels, Some(u_k));
        total_loss += lk;
        step_loss_grads.push((gxk, gvk, guk));

        world.set_all_targets(u_k);
        let poses_k = world.state.poses.clone();
        let vels_k = world.state.vels.clone();
        let act_before = world.state.actuation.clone();
        let con_before = world.state.contact_states.clone();
        let der_states_k = world.state.der_states.clone();

        // Snapshot post-servo actuation state matching step_linearly_implicit_euler line 1
        let mut act_after_servo = act_before.clone();
        advance_tendon_servos(&world.graph.tendons, &mut act_after_servo, dt);

        let (_diag, factored) = world.step_linearly_implicit_euler_with_factor(dt)?;
        let poses_next = world.state.poses.clone();
        let der_states_next = world.state.der_states.clone();
        let v_next = world.state.vels.clone();

        tapes.push(StepTape {
            poses_k,
            poses_next,
            vels_k,
            act_before,
            act_after_servo,
            con_before,
            der_states_k,
            der_states_next,
            v_next,
            factored,
        });
    }

    // Terminal loss at step T
    let final_state = world.state.clone();
    let (l_term, mut bar_x, mut bar_v, _) =
        loss.step_loss_and_grad(n_steps, &final_state.poses, &final_state.vels, None);
    total_loss += l_term;

    // Running reverse-mode cotangents across persistent internal states
    let mut bar_servo_pos = vec![0.0_f64; n_servo_slots];
    let mut bar_tendon_tension = vec![0.0_f64; n_tendons];
    let mut bar_seg_tensions: Vec<Vec<f64>> = world
        .state
        .actuation
        .tendons
        .iter()
        .map(|t| vec![0.0_f64; t.segment_tensions.len()])
        .collect();
    let mut bar_prev_seg_lens: Vec<Vec<f64>> = world
        .state
        .actuation
        .tendons
        .iter()
        .map(|t| vec![0.0_f64; t.prev_segment_lengths.len()])
        .collect();
    let mut bar_con_state = vec![[0.0_f64; 7]; n_sites];
    let mut bar_der_theta: Vec<Vec<f64>> = world
        .beams
        .iter()
        .map(|br| match br {
            BeamRuntime::Der(data) => vec![0.0; data.n_elements],
            BeamRuntime::Corotational(_) => Vec::new(),
        })
        .collect();
    let mut bar_der_theta_vel: Vec<Vec<f64>> = world
        .beams
        .iter()
        .map(|br| match br {
            BeamRuntime::Der(data) => vec![0.0; data.n_elements],
            BeamRuntime::Corotational(_) => Vec::new(),
        })
        .collect();
    let mut bar_der_bishop_t: Vec<Vec<Vec3>> = world
        .beams
        .iter()
        .map(|br| match br {
            BeamRuntime::Der(data) => vec![Vec3::ZERO; data.n_elements],
            BeamRuntime::Corotational(_) => Vec::new(),
        })
        .collect();
    let mut bar_der_bishop_u: Vec<Vec<Vec3>> = world
        .beams
        .iter()
        .map(|br| match br {
            BeamRuntime::Der(data) => vec![Vec3::ZERO; data.n_elements],
            BeamRuntime::Corotational(_) => Vec::new(),
        })
        .collect();

    let mut grad_controls: Vec<Vec<f64>> = controls.iter().map(|u| vec![0.0; u.len()]).collect();
    let mut grad_params = vec![0.0_f64; diff_params.len()];

    // Precompute actuator ID -> packed control slice index maps (matching `set_all_targets`)
    let n_ten_act = world.tendon_actuator_ids.len();
    let mut tendon_aid_to_u_idx: Vec<Option<usize>> = vec![None; n_servo_slots];
    for (slot, &aid) in world.tendon_actuator_ids.iter().enumerate() {
        if aid < tendon_aid_to_u_idx.len() {
            tendon_aid_to_u_idx[aid] = Some(slot);
        }
    }
    let n_jnt_slots = world.state.actuation.joint_targets.len();
    let mut joint_aid_to_u_idx: Vec<Option<usize>> = vec![None; n_jnt_slots];
    for (slot, &aid) in world.joint_actuator_ids.iter().enumerate() {
        if aid < joint_aid_to_u_idx.len() {
            joint_aid_to_u_idx[aid] = Some(n_ten_act + slot);
        }
    }

    // --- 2. Reverse Pass (k = T-1 .. 0): Single O(N) IFT back-solve + local entity VJPs ---
    let eps_elem = 1.0e-6;
    let eps_local = 1.0e-7;
    let body_masses = world.body_masses();

    for k in (0..n_steps).rev() {
        let tape = &tapes[k];
        let (ref gxk_loss, ref gvk_loss, ref guk_loss) = step_loss_grads[k];

        // Step A0: Backprop through DER Bishop-frame update & synced free-node orientations
        let mut next_bar_der_bishop_t: Vec<Vec<Vec3>> = world
            .beams
            .iter()
            .map(|br| match br {
                BeamRuntime::Der(data) => vec![Vec3::ZERO; data.n_elements],
                BeamRuntime::Corotational(_) => Vec::new(),
            })
            .collect();
        let mut next_bar_der_bishop_u: Vec<Vec<Vec3>> = world
            .beams
            .iter()
            .map(|br| match br {
                BeamRuntime::Der(data) => vec![Vec3::ZERO; data.n_elements],
                BeamRuntime::Corotational(_) => Vec::new(),
            })
            .collect();

        for (s, br) in world.beams.iter().enumerate() {
            let BeamRuntime::Der(data) = br else {
                continue;
            };
            let (Some(dst_k), Some(dst_next)) = (
                tape.der_states_k[s].as_ref(),
                tape.der_states_next[s].as_ref(),
            ) else {
                continue;
            };
            let map = &world.analysis.indexing.beam_maps[s];

            // 1. Pull back free-node rotation cotangents from `poses_next[nid].rot` synced from
            //    `(t_next, u_next, theta_next)`: exact SO(3) identity
            //    dphi = t x dt + (v . du + dtheta) * t
            for (idx, &nid) in map.free_node_ids.iter().enumerate() {
                let g_rot = bar_x[nid].ang();
                bar_x[nid] = Vec6::from_parts(bar_x[nid].lin(), Vec3::ZERO);
                if g_rot.norm_sq() > 0.0 {
                    let e = idx.min(data.n_elements - 1);
                    let t_e = dst_next.bishop_t[e];
                    let u_e = dst_next.bishop_u[e];
                    let v_e = t_e.cross(u_e);
                    let g_dot_t = g_rot.dot(t_e);
                    bar_der_bishop_t[s][e] += g_rot.cross(t_e);
                    bar_der_bishop_u[s][e] += v_e * g_dot_t;
                    bar_der_theta[s][e] += g_dot_t;
                }
            }

            // 2. Pull back `(bar_der_bishop_t, bar_der_bishop_u)` through parallel transport
            //    `u_next = parallel_transport(u_k, t_k, t_next)` and edge normalization.
            let pts_next = data.vertex_positions(map, &tape.poses_next);
            let off_a_next = tape.poses_next[data.body_a].rot.rotate(data.clamp_a.pos);
            let off_b_next = data.body_b.map_or(Vec3::ZERO, |bb| {
                tape.poses_next[bb]
                    .rot
                    .rotate(data.clamp_b.unwrap_or(Se3::IDENTITY).pos)
            });
            let n_free_verts = map.free_node_ids.len();

            for e in 0..data.n_elements {
                let bt = bar_der_bishop_t[s][e];
                let bu = bar_der_bishop_u[s][e];
                if bt.norm_sq() == 0.0 && bu.norm_sq() == 0.0 {
                    continue;
                }
                let t_k = dst_k.bishop_t[e];
                let u_k = dst_k.bishop_u[e];
                let t_next = dst_next.bishop_t[e];

                let mut g_t_next = bt;
                if bu.norm_sq() > 0.0 {
                    for a in 0..3 {
                        let mut d = Vec3::ZERO;
                        d[a] = eps_local;
                        let up = bu.dot(parallel_transport(u_k + d, t_k, t_next));
                        let um = bu.dot(parallel_transport(u_k - d, t_k, t_next));
                        next_bar_der_bishop_u[s][e][a] += (up - um) / (2.0 * eps_local);

                        let tkp = bu.dot(parallel_transport(u_k, t_k + d, t_next));
                        let tkm = bu.dot(parallel_transport(u_k, t_k - d, t_next));
                        next_bar_der_bishop_t[s][e][a] += (tkp - tkm) / (2.0 * eps_local);

                        let tnp = bu.dot(parallel_transport(u_k, t_k, t_next + d));
                        let tnm = bu.dot(parallel_transport(u_k, t_k, t_next - d));
                        g_t_next[a] += (tnp - tnm) / (2.0 * eps_local);
                    }
                }

                let edge_vec = pts_next[e + 1] - pts_next[e];
                let len = edge_vec.norm();
                if len > 1e-12 {
                    let g_edge = (g_t_next - t_next * g_t_next.dot(t_next)) * (1.0 / len);
                    if e == 0 {
                        if !world.graph.bodies[data.body_a].is_fixed {
                            let f = -g_edge;
                            bar_x[data.body_a] += Vec6::from_parts(f, off_a_next.cross(f));
                        }
                    } else {
                        let nid = map.free_node_ids[e - 1];
                        bar_x[nid] -= Vec6::from_parts(g_edge, Vec3::ZERO);
                    }
                    if e < n_free_verts {
                        let nid = map.free_node_ids[e];
                        bar_x[nid] += Vec6::from_parts(g_edge, Vec3::ZERO);
                    } else if let Some(bb) = data.body_b {
                        if !world.graph.bodies[bb].is_fixed {
                            bar_x[bb] += Vec6::from_parts(g_edge, off_b_next.cross(g_edge));
                        }
                    }
                }
            }
        }

        // Step A: Backprop through exact SE(3) decoupled pose integration & DER twist integration
        let mut v_tilde = vec![Vec6::ZERO; total];
        let mut next_bar_x = vec![Vec6::ZERO; total];
        let mut next_bar_v = vec![Vec6::ZERO; total];

        for i in 0..total {
            if i < nb && world.graph.bodies[i].is_fixed {
                continue;
            }
            let dth = tape.v_next[i].ang() * dt;
            let rot_dth = Quat::exp(dth).to_mat3();
            let dexp_dth = so3_dexp(dth);
            v_tilde[i] = Vec6::from_parts(
                bar_v[i].lin() + dt * bar_x[i].lin(),
                bar_v[i].ang() + dt * dexp_dth.transpose().mul_vec(bar_x[i].ang()),
            );
            next_bar_x[i] = Vec6::from_parts(
                bar_x[i].lin() + gxk_loss[i].lin(),
                rot_dth.transpose().mul_vec(bar_x[i].ang()) + gxk_loss[i].ang(),
            );
            next_bar_v[i] = gvk_loss[i];
        }

        let mut theta_tilde: Vec<Option<Vec<f64>>> = vec![None; world.beams.len()];
        let mut next_bar_der_theta: Vec<Vec<f64>> = vec![Vec::new(); world.beams.len()];
        let mut next_bar_der_theta_vel: Vec<Vec<f64>> = vec![Vec::new(); world.beams.len()];
        for (s, br) in world.beams.iter().enumerate() {
            if let BeamRuntime::Der(data) = br {
                let mut tt = vec![0.0; data.n_elements];
                for e in 0..data.n_elements {
                    tt[e] = bar_der_theta_vel[s][e] + dt * bar_der_theta[s][e];
                }
                theta_tilde[s] = Some(tt);
                next_bar_der_theta[s] = bar_der_theta[s].clone();
                next_bar_der_theta_vel[s] = vec![0.0; data.n_elements];
            }
        }

        // Seed control gradient with direct step-k loss derivative
        for (idx, &g) in guk_loss.iter().enumerate() {
            if idx < grad_controls[k].len() {
                grad_controls[k][idx] += g;
            }
        }

        // Step B: Single O(N) IFT Adjoint Linear Solve reusing cached FactoredBipartiteSystem!
        // Since H_k is symmetric in LinearlyImplicitEuler, H_k^T lambda_k = v_tilde.
        let theta_tilde_refs: Vec<Option<&[f64]>> =
            theta_tilde.iter().map(|o| o.as_deref()).collect();
        let (lambda, lambda_der_thetas) = tape.factored.solve_rhs_with_der_thetas(
            &v_tilde,
            &theta_tilde_refs,
            &world.analysis,
        )?;

        // --- Step C1: Rigid-body inertia & gyroscopic VJPs (exact analytical + 3x rot) ---
        for bi in 0..nb {
            if world.graph.bodies[bi].is_fixed {
                continue;
            }
            let q = tape.poses_k[bi].rot;
            let v = tape.vels_k[bi];
            let v_next = tape.v_next[bi];
            let lam = lambda[bi];
            let alpha = world.graph.bodies[bi].rayleigh_alpha;
            let m6 = world.body_inertia_mat6(bi, q);

            // Analytical dPsi / dv_k = M_b * lam - dt * (d tau_gyro / d w)^T lam_ang
            next_bar_v[bi] += m6.mul_vec(lam);
            if !world.config.world_fixed_inertia {
                let iw = m6.block(1, 1);
                let w = v.ang();
                let iw_w = iw.mul_vec(w);
                let lam_w = lam.ang();
                let d_gyro_t_lam = iw_w.cross(lam_w) - iw.mul_vec(w.cross(lam_w));
                next_bar_v[bi] -= Vec6::from_parts(Vec3::ZERO, d_gyro_t_lam * dt);

                // Orientation derivative of M_b(q) and tau_gyro(q, v) (only 3 rotation scalars)
                let body_psi_q = |q_eval: Quat| -> f64 {
                    let mq = world.body_inertia_mat6(bi, q_eval);
                    let gyro = world.body_gyro_wrench(bi, q_eval, v);
                    let rhs_minus_lhs = mq.mul_vec(v + dt * g6 - v_next * (1.0 + dt * alpha))
                        - gyro * dt;
                    lam.dot(rhs_minus_lhs)
                };
                for a in 0..3 {
                    let mut d = Vec3::ZERO;
                    d[a] = eps_local;
                    let qp = Quat::exp(d).mul(q).normalized();
                    let qm = Quat::exp(-d).mul(q).normalized();
                    next_bar_x[bi][3 + a] += (body_psi_q(qp) - body_psi_q(qm)) / (2.0 * eps_local);
                }
            }
        }

        // --- Step C2: Continuum Beam VJPs (Corotational & DER) + interior beam SDF contact ---
        for (s, br) in world.beams.iter().enumerate() {
            let map = &world.analysis.indexing.beam_maps[s];
            match br {
                BeamRuntime::Corotational(data) => {
                    let cm = 1.0 + dt * data.rayleigh_alpha;
                    let ck = dt * data.rayleigh_beta + dt * dt;
                    for e in 0..data.n_elements {
                        let n0 = map.chain_node_ids[e];
                        let n1 = map.chain_node_ids[e + 1];
                        let x0 = tape.poses_k[n0];
                        let x1 = tape.poses_k[n1];
                        let lam0 = lambda[n0];
                        let lam1 = lambda[n1];
                        if lam0.max_abs() == 0.0 && lam1.max_abs() == 0.0 {
                            continue;
                        }
                        let tf0 = data.elem_transforms[e][0];
                        let tf1 = data.elem_transforms[e][1];
                        let q0 = x0.rot.mul(tf0.rot);
                        let q1 = x1.rot.mul(tf1.rot);
                        let off0 = x0.rot.rotate(tf0.pos);
                        let off1 = x1.rot.rotate(tf1.pos);
                        let p0 = x0.pos + off0;
                        let p1 = x1.pos + off1;
                        let l = data.lengths[e];

                        let (u_k, frame_k) =
                            beam_local_strain(p0, q0, p1, q1, l, world.config.use_strain_jacobian);
                        let du = u_k - data.rest_local[e];
                        let f_loc = data.k_local[e].mul_vec(du);
                        let t_k = beam_frame_transform(frame_k.rot, off0, off1);
                        let j_k = beam_strain_jacobian(p0, q0, off0, p1, q1, off1, l);
                        let a_k = if world.config.use_strain_jacobian {
                            j_k
                        } else {
                            t_k
                        };

                        let lam = Vec12::from_halves(lam0, lam1);
                        let v_n = Vec12::from_halves(tape.v_next[n0], tape.v_next[n1]);
                        let v_k = Vec12::from_halves(tape.vels_k[n0] + dt * g6, tape.vels_k[n1] + dt * g6);
                        let v_rhs = v_k - v_n * cm;

                        let t_lam = t_k.mul_vec(lam);
                        let m_t_lam = data.m_local[e].mul_vec(t_lam);
                        let m_t_vrhs = data.m_local[e].mul_vec(t_k.mul_vec(v_rhs));
                        // Analytical velocity VJP: +M_w * lam = T^T M_loc T * lam
                        let m_lam = t_k.transpose().mul_vec(m_t_lam);
                        if !(n0 < nb && world.graph.bodies[n0].is_fixed) {
                            next_bar_v[n0] += m_lam.head();
                        }
                        if !(n1 < nb && world.graph.bodies[n1].is_fixed) {
                            next_bar_v[n1] += m_lam.tail();
                        }

                        // Exact analytical stiff elastic strain VJP: -dt * J_k^T (K_loc A_k lam)
                        // (zero cancellation even for E = 200 GPa steel strips!)
                        let k_a_lam = data.k_local[e].mul_vec(a_k.mul_vec(lam));
                        let k_a_v = data.k_local[e].mul_vec(a_k.mul_vec(v_n));
                        let stiff_grad = -(j_k.transpose().mul_vec(k_a_lam)) * dt;
                        if !(n0 < nb && world.graph.bodies[n0].is_fixed) {
                            next_bar_x[n0] += stiff_grad.head();
                        }
                        if !(n1 < nb && world.graph.bodies[n1].is_fixed) {
                            next_bar_x[n1] += stiff_grad.tail();
                        }

                        // Smooth O(1) geometric frame/Jacobian pose VJP (frozen f_loc, k_a_v, k_a_lam)
                        if !(n0 < nb && world.graph.bodies[n0].is_fixed) {
                            for d in 0..6 {
                                let mut tw = Vec6::ZERO;
                                tw[d] = 1.0;
                                let p_p = corotational_elem_geom_psi(
                                    data,
                                    e,
                                    x0.integrate_decoupled(tw, eps_elem),
                                    x1,
                                    v_rhs,
                                    v_n,
                                    lam,
                                    f_loc,
                                    k_a_v,
                                    k_a_lam,
                                    m_t_vrhs,
                                    m_t_lam,
                                    dt,
                                    ck,
                                    world.config.use_strain_jacobian,
                                );
                                let p_m = corotational_elem_geom_psi(
                                    data,
                                    e,
                                    x0.integrate_decoupled(tw, -eps_elem),
                                    x1,
                                    v_rhs,
                                    v_n,
                                    lam,
                                    f_loc,
                                    k_a_v,
                                    k_a_lam,
                                    m_t_vrhs,
                                    m_t_lam,
                                    dt,
                                    ck,
                                    world.config.use_strain_jacobian,
                                );
                                next_bar_x[n0][d] += (p_p - p_m) / (2.0 * eps_elem);
                            }
                        }
                        if !(n1 < nb && world.graph.bodies[n1].is_fixed) {
                            for d in 0..6 {
                                let mut tw = Vec6::ZERO;
                                tw[d] = 1.0;
                                let p_p = corotational_elem_geom_psi(
                                    data,
                                    e,
                                    x0,
                                    x1.integrate_decoupled(tw, eps_elem),
                                    v_rhs,
                                    v_n,
                                    lam,
                                    f_loc,
                                    k_a_v,
                                    k_a_lam,
                                    m_t_vrhs,
                                    m_t_lam,
                                    dt,
                                    ck,
                                    world.config.use_strain_jacobian,
                                );
                                let p_m = corotational_elem_geom_psi(
                                    data,
                                    e,
                                    x0,
                                    x1.integrate_decoupled(tw, -eps_elem),
                                    v_rhs,
                                    v_n,
                                    lam,
                                    f_loc,
                                    k_a_v,
                                    k_a_lam,
                                    m_t_vrhs,
                                    m_t_lam,
                                    dt,
                                    ck,
                                    world.config.use_strain_jacobian,
                                );
                                next_bar_x[n1][d] += (p_p - p_m) / (2.0 * eps_elem);
                            }
                        }
                    }
                }
                BeamRuntime::Der(data) => {
                    let cm = 1.0 + dt * data.rayleigh_alpha;
                    let ck = dt * data.rayleigh_beta + dt * dt;
                    let lam_th = lambda_der_thetas[s].as_deref().unwrap_or(&[]);

                    // 1. Lumped translational & twist-inertia velocity VJPs on DER free vertices, edges & clamps
                    for (li, &nid) in map.free_node_ids.iter().enumerate() {
                        let m = data.free_vertex_mass[li];
                        next_bar_v[nid] += Vec6::from_parts(lambda[nid].lin() * m, Vec3::ZERO);
                    }
                    for (e, &lt) in lam_th.iter().enumerate() {
                        next_bar_der_theta_vel[s][e] += data.edge_twist_inertia[e] * lt;
                    }
                    if !world.graph.bodies[data.body_a].is_fixed && data.end_a_mass > 0.0 {
                        let off_a = tape.poses_k[data.body_a].rot.rotate(data.clamp_a.pos);
                        let lam_pt =
                            lambda[data.body_a].lin() + lambda[data.body_a].ang().cross(off_a);
                        let f = lam_pt * data.end_a_mass;
                        next_bar_v[data.body_a] += Vec6::from_parts(f, off_a.cross(f));
                    }
                    if let Some(bb) = data.body_b {
                        if !world.graph.bodies[bb].is_fixed && data.end_b_mass > 0.0 {
                            let cb = data.clamp_b.unwrap_or(Se3::IDENTITY);
                            let off_b = tape.poses_k[bb].rot.rotate(cb.pos);
                            let lam_pt = lambda[bb].lin() + lambda[bb].ang().cross(off_b);
                            let f = lam_pt * data.end_b_mass;
                            next_bar_v[bb] += Vec6::from_parts(f, off_b.cross(f));
                        }
                    }

                    // 2. Full DER elastic pose / twist / Bishop-frame & clamp-mass VJPs
                    let (Some(dst_k), Some(dst_next)) = (
                        tape.der_states_k[s].as_ref(),
                        tape.der_states_next[s].as_ref(),
                    ) else {
                        continue;
                    };
                    let vn_th = dst_next.theta_vel.as_slice();
                    let lam_active = lambda[data.body_a].max_abs() > 0.0
                        || data.body_b.is_some_and(|bb| lambda[bb].max_abs() > 0.0)
                        || map
                            .free_node_ids
                            .iter()
                            .any(|&nid| lambda[nid].lin().norm_sq() > 0.0)
                        || lam_th.iter().any(|&x| x != 0.0);

                    if lam_active {
                        let n_el = data.n_elements;
                        let spd_cache = world.config.project_der_hessian_spd.then(|| {
                            data.build_elastic_vjp_cache(
                                map,
                                &tape.poses_k,
                                dst_k,
                                &lambda,
                                lam_th,
                                &tape.v_next,
                                vn_th,
                            )
                        });
                        let der_psi_subset =
                            |poses_loc: &[Se3],
                             dst_loc: &DerBeamState,
                             stretch_range: std::ops::Range<usize>,
                             interior_range: std::ops::Range<usize>,
                             inc_a: bool,
                             inc_b: bool,
                             inc_mass_a: bool,
                             inc_mass_b: bool|
                             -> f64 {
                                let (grad_dot, hess_dot) = data.eval_elastic_contraction_subset(
                                    map,
                                    poses_loc,
                                    dst_loc,
                                    &lambda,
                                    lam_th,
                                    &tape.v_next,
                                    vn_th,
                                    world.config.project_der_hessian_spd,
                                    spd_cache.as_ref(),
                                    stretch_range,
                                    interior_range,
                                    inc_a,
                                    inc_b,
                                );
                                let mut val = -dt * grad_dot - ck * hess_dot;
                                if inc_mass_a
                                    && !world.graph.bodies[data.body_a].is_fixed
                                    && data.end_a_mass > 0.0
                                    && data.clamp_a.pos.norm_sq() > 0.0
                                {
                                    let oa = poses_loc[data.body_a].rot.rotate(data.clamp_a.pos);
                                    let la = lambda[data.body_a].lin()
                                        + lambda[data.body_a].ang().cross(oa);
                                    let va_k = tape.vels_k[data.body_a].lin()
                                        + tape.vels_k[data.body_a].ang().cross(oa);
                                    let va_n = tape.v_next[data.body_a].lin()
                                        + tape.v_next[data.body_a].ang().cross(oa);
                                    val += data.end_a_mass
                                        * la.dot(va_k + dt * world.graph.gravity - cm * va_n);
                                }
                                if inc_mass_b {
                                    if let Some(bb) = data.body_b {
                                        let cb = data.clamp_b.unwrap_or(Se3::IDENTITY);
                                        if !world.graph.bodies[bb].is_fixed
                                            && data.end_b_mass > 0.0
                                            && cb.pos.norm_sq() > 0.0
                                        {
                                            let ob = poses_loc[bb].rot.rotate(cb.pos);
                                            let lb = lambda[bb].lin() + lambda[bb].ang().cross(ob);
                                            let vb_k = tape.vels_k[bb].lin()
                                                + tape.vels_k[bb].ang().cross(ob);
                                            let vb_n = tape.v_next[bb].lin()
                                                + tape.v_next[bb].ang().cross(ob);
                                            val += data.end_b_mass
                                                * lb.dot(
                                                    vb_k + dt * world.graph.gravity - cm * vb_n,
                                                );
                                        }
                                    }
                                }
                                val
                            };

                        let mut poses_tmp = tape.poses_k.clone();
                        let mut dst_tmp = dst_k.clone();
                        let eps_der = 1.0e-7;

                        // Clamp A 6-DOF pose VJP (touches stretch 0, interior 1, clamp A, clamp B if n_el==1)
                        if !world.graph.bodies[data.body_a].is_fixed {
                            let orig = poses_tmp[data.body_a];
                            let sr = 0..1.min(n_el);
                            let ir = 1..2.min(n_el);
                            let inc_b = data.body_b.is_some() && n_el == 1;
                            for d in 0..6 {
                                let mut tw = Vec6::ZERO;
                                tw[d] = 1.0;
                                poses_tmp[data.body_a] = orig.integrate_decoupled(tw, eps_der);
                                let pp = der_psi_subset(
                                    &poses_tmp,
                                    &dst_tmp,
                                    sr.clone(),
                                    ir.clone(),
                                    true,
                                    inc_b,
                                    true,
                                    false,
                                );
                                poses_tmp[data.body_a] = orig.integrate_decoupled(tw, -eps_der);
                                let pm = der_psi_subset(
                                    &poses_tmp,
                                    &dst_tmp,
                                    sr.clone(),
                                    ir.clone(),
                                    true,
                                    inc_b,
                                    true,
                                    false,
                                );
                                next_bar_x[data.body_a][d] += (pp - pm) / (2.0 * eps_der);
                            }
                            poses_tmp[data.body_a] = orig;
                        }
                        // Clamp B 6-DOF pose VJP (touches stretch n_el-1, interior n_el-1, clamp B, clamp A if n_el==1)
                        if let Some(bb) = data.body_b {
                            if !world.graph.bodies[bb].is_fixed {
                                let orig = poses_tmp[bb];
                                let sr = (n_el - 1)..n_el;
                                let ir = (n_el - 1).max(1)..n_el;
                                let inc_a = n_el == 1;
                                for d in 0..6 {
                                    let mut tw = Vec6::ZERO;
                                    tw[d] = 1.0;
                                    poses_tmp[bb] = orig.integrate_decoupled(tw, eps_der);
                                    let pp = der_psi_subset(
                                        &poses_tmp,
                                        &dst_tmp,
                                        sr.clone(),
                                        ir.clone(),
                                        inc_a,
                                        true,
                                        false,
                                        true,
                                    );
                                    poses_tmp[bb] = orig.integrate_decoupled(tw, -eps_der);
                                    let pm = der_psi_subset(
                                        &poses_tmp,
                                        &dst_tmp,
                                        sr.clone(),
                                        ir.clone(),
                                        inc_a,
                                        true,
                                        false,
                                        true,
                                    );
                                    next_bar_x[bb][d] += (pp - pm) / (2.0 * eps_der);
                                }
                                poses_tmp[bb] = orig;
                            }
                        }
                        // Free vertex 3-DOF translational pose VJP (O(1) local stencils per vertex!)
                        for (li, &nid) in map.free_node_ids.iter().enumerate() {
                            let v = li + 1;
                            let sr = (v - 1)..(v + 1).min(n_el);
                            let ir = v.saturating_sub(1).max(1)..(v + 2).min(n_el);
                            let inc_a = v == 1;
                            let inc_b = data.body_b.is_some() && v + 1 == n_el;
                            let orig_pos = poses_tmp[nid].pos;
                            for a in 0..3 {
                                let mut dp = Vec3::ZERO;
                                dp[a] = eps_der;
                                poses_tmp[nid].pos = orig_pos + dp;
                                let pp = der_psi_subset(
                                    &poses_tmp,
                                    &dst_tmp,
                                    sr.clone(),
                                    ir.clone(),
                                    inc_a,
                                    inc_b,
                                    false,
                                    false,
                                );
                                poses_tmp[nid].pos = orig_pos - dp;
                                let pm = der_psi_subset(
                                    &poses_tmp,
                                    &dst_tmp,
                                    sr.clone(),
                                    ir.clone(),
                                    inc_a,
                                    inc_b,
                                    false,
                                    false,
                                );
                                next_bar_x[nid][a] += (pp - pm) / (2.0 * eps_der);
                            }
                            poses_tmp[nid].pos = orig_pos;
                        }
                        // Twist angles & Bishop frames (O(1) local stencils per edge, only for k > 0)
                        if k > 0 {
                            for e in 0..n_el {
                                let ir = e.max(1)..(e + 2).min(n_el);
                                let inc_a = e == 0;
                                let inc_b = data.body_b.is_some() && e + 1 == n_el;

                                let th0 = dst_tmp.theta[e];
                                dst_tmp.theta[e] = th0 + eps_der;
                                let tp = der_psi_subset(
                                    &poses_tmp,
                                    &dst_tmp,
                                    0..0,
                                    ir.clone(),
                                    inc_a,
                                    inc_b,
                                    false,
                                    false,
                                );
                                dst_tmp.theta[e] = th0 - eps_der;
                                let tm = der_psi_subset(
                                    &poses_tmp,
                                    &dst_tmp,
                                    0..0,
                                    ir.clone(),
                                    inc_a,
                                    inc_b,
                                    false,
                                    false,
                                );
                                dst_tmp.theta[e] = th0;
                                next_bar_der_theta[s][e] += (tp - tm) / (2.0 * eps_der);

                                let bt0 = dst_tmp.bishop_t[e];
                                let bu0 = dst_tmp.bishop_u[e];
                                for a in 0..3 {
                                    let mut d = Vec3::ZERO;
                                    d[a] = eps_der;
                                    dst_tmp.bishop_t[e] = bt0 + d;
                                    let btp = der_psi_subset(
                                        &poses_tmp,
                                        &dst_tmp,
                                        0..0,
                                        ir.clone(),
                                        inc_a,
                                        inc_b,
                                        false,
                                        false,
                                    );
                                    dst_tmp.bishop_t[e] = bt0 - d;
                                    let btm = der_psi_subset(
                                        &poses_tmp,
                                        &dst_tmp,
                                        0..0,
                                        ir.clone(),
                                        inc_a,
                                        inc_b,
                                        false,
                                        false,
                                    );
                                    next_bar_der_bishop_t[s][e][a] +=
                                        (btp - btm) / (2.0 * eps_der);
                                }
                                dst_tmp.bishop_t[e] = bt0;

                                for a in 0..3 {
                                    let mut d = Vec3::ZERO;
                                    d[a] = eps_der;
                                    dst_tmp.bishop_u[e] = bu0 + d;
                                    let bup = der_psi_subset(
                                        &poses_tmp,
                                        &dst_tmp,
                                        0..0,
                                        ir.clone(),
                                        inc_a,
                                        inc_b,
                                        false,
                                        false,
                                    );
                                    dst_tmp.bishop_u[e] = bu0 - d;
                                    let bum = der_psi_subset(
                                        &poses_tmp,
                                        &dst_tmp,
                                        0..0,
                                        ir.clone(),
                                        inc_a,
                                        inc_b,
                                        false,
                                        false,
                                    );
                                    next_bar_der_bishop_u[s][e][a] +=
                                        (bup - bum) / (2.0 * eps_der);
                                }
                                dst_tmp.bishop_u[e] = bu0;
                            }
                        }
                    }
                }
            }

            // Beam interior node SDF contact VJP
            if let Some(radius) = world.graph.beams[s].interior_node_contact_radius {
                for &nid in &map.free_node_ids {
                    let lam_l = lambda[nid].lin();
                    if lam_l.norm_sq() == 0.0 {
                        continue;
                    }
                    let pos = tape.poses_k[nid].pos;
                    let vel = tape.vels_k[nid].lin();
                    let vn = tape.v_next[nid].lin();
                    if beam_interior_contact_psi(world, radius, pos, vel, dt, lam_l, vn) == 0.0 {
                        continue;
                    }
                    for a in 0..3 {
                        let mut dp = Vec3::ZERO;
                        dp[a] = eps_local;
                        let pp =
                            beam_interior_contact_psi(world, radius, pos + dp, vel, dt, lam_l, vn);
                        let pm =
                            beam_interior_contact_psi(world, radius, pos - dp, vel, dt, lam_l, vn);
                        next_bar_x[nid][a] += (pp - pm) / (2.0 * eps_local);

                        let vp =
                            beam_interior_contact_psi(world, radius, pos, vel + dp, dt, lam_l, vn);
                        let vm =
                            beam_interior_contact_psi(world, radius, pos, vel - dp, dt, lam_l, vn);
                        next_bar_v[nid][a] += (vp - vm) / (2.0 * eps_local);
                    }
                }
            }
        }

        // --- Step C3: Tendon VJPs (local per tendon -> body poses/vels + servo_pos/vel_mid + stick-slip state) ---
        let mut bar_servo_pos_mid = bar_servo_pos.clone();
        let mut bar_servo_vel_mid = vec![0.0_f64; n_servo_slots];
        let mut next_bar_tendon_tension = vec![0.0_f64; n_tendons];
        let mut next_bar_seg_tensions: Vec<Vec<f64>> = bar_seg_tensions
            .iter()
            .map(|v| vec![0.0_f64; v.len()])
            .collect();
        let mut next_bar_prev_seg_lens: Vec<Vec<f64>> = bar_prev_seg_lens
            .iter()
            .map(|v| vec![0.0_f64; v.len()])
            .collect();

        for ti in 0..n_tendons {
            let spec = &world.graph.tendons[ti];
            let aid = spec.actuator_id;
            let mut uniq_bodies: Vec<usize> = spec.waypoints.iter().map(|w| w.body_id).collect();
            uniq_bodies.sort_unstable();
            uniq_bodies.dedup();

            let ten_psi = |poses_loc: &[Se3], vels_loc: &[Vec6], act_in: &ActuationState| -> f64 {
                let mut act_tmp = act_in.clone();
                let ev = evaluate_actuation_with_masses(
                    nb,
                    std::slice::from_ref(spec),
                    &[],
                    &world.rest_poses,
                    poses_loc,
                    vels_loc,
                    &mut act_tmp,
                    0.0,
                    0.0,
                    false,
                    true,
                    world.config.joint_stribeck_friction,
                    Some(&body_masses),
                    dt,
                );
                let mut val = bar_tendon_tension[ti] * act_tmp.tendons[0].tension;
                if spec.stick_slip_enabled {
                    for (&b, &v) in bar_seg_tensions[ti]
                        .iter()
                        .zip(act_tmp.tendons[0].segment_tensions.iter())
                    {
                        val += b * v;
                    }
                    for (&b, &v) in bar_prev_seg_lens[ti]
                        .iter()
                        .zip(act_tmp.tendons[0].prev_segment_lengths.iter())
                    {
                        val += b * v;
                    }
                }
                for &bi in &uniq_bodies {
                    val += dt * lambda[bi].dot(ev.body_wrenches[bi]);
                }
                val
            };

            // Slice ActuationState to single tendon entry `ti`
            let mut single_act = ActuationState {
                tendon_servo_pos: tape.act_after_servo.tendon_servo_pos.clone(),
                tendon_servo_target: tape.act_after_servo.tendon_servo_target.clone(),
                tendon_servo_vel: tape.act_after_servo.tendon_servo_vel.clone(),
                tendons: vec![tape.act_after_servo.tendons[ti].clone()],
                joint_targets: Vec::new(),
                last_joint_torque: Vec::new(),
                last_joint_torque_cap: Vec::new(),
            };

            let mut poses_tmp = tape.poses_k.clone();
            let mut vels_tmp = tape.vels_k.clone();
            for &bi in &uniq_bodies {
                if world.graph.bodies[bi].is_fixed {
                    continue;
                }
                let orig_p = poses_tmp[bi];
                for d in 0..6 {
                    let mut tw = Vec6::ZERO;
                    tw[d] = 1.0;
                    poses_tmp[bi] = orig_p.integrate_decoupled(tw, eps_local);
                    let pp = ten_psi(&poses_tmp, &vels_tmp, &single_act);
                    poses_tmp[bi] = orig_p.integrate_decoupled(tw, -eps_local);
                    let pm = ten_psi(&poses_tmp, &vels_tmp, &single_act);
                    next_bar_x[bi][d] += (pp - pm) / (2.0 * eps_local);
                }
                poses_tmp[bi] = orig_p;

                let orig_v = vels_tmp[bi];
                for d in 0..6 {
                    vels_tmp[bi][d] = orig_v[d] + eps_local;
                    let vp = ten_psi(&poses_tmp, &vels_tmp, &single_act);
                    vels_tmp[bi][d] = orig_v[d] - eps_local;
                    let vm = ten_psi(&poses_tmp, &vels_tmp, &single_act);
                    next_bar_v[bi][d] += (vp - vm) / (2.0 * eps_local);
                }
                vels_tmp[bi] = orig_v;
            }

            // Derivative wrt servo_pos_mid[aid] and servo_vel_mid[aid]
            let s0 = single_act.tendon_servo_pos[aid];
            single_act.tendon_servo_pos[aid] = s0 + eps_local;
            let sp = ten_psi(&poses_tmp, &vels_tmp, &single_act);
            single_act.tendon_servo_pos[aid] = s0 - eps_local;
            let sm = ten_psi(&poses_tmp, &vels_tmp, &single_act);
            single_act.tendon_servo_pos[aid] = s0;
            bar_servo_pos_mid[aid] += (sp - sm) / (2.0 * eps_local);

            let sv0 = single_act.tendon_servo_vel[aid];
            single_act.tendon_servo_vel[aid] = sv0 + eps_local;
            let svp = ten_psi(&poses_tmp, &vels_tmp, &single_act);
            single_act.tendon_servo_vel[aid] = sv0 - eps_local;
            let svm = ten_psi(&poses_tmp, &vels_tmp, &single_act);
            single_act.tendon_servo_vel[aid] = sv0;
            bar_servo_vel_mid[aid] += (svp - svm) / (2.0 * eps_local);

            // Derivative wrt previous stick-slip Capstan segment tensions & lengths
            if spec.stick_slip_enabled && k > 0 {
                let n_seg = single_act.tendons[0].segment_tensions.len();
                for j in 0..n_seg {
                    let t0 = single_act.tendons[0].segment_tensions[j];
                    let t_hi = t0 + eps_local;
                    let t_lo = (t0 - eps_local).max(0.0);
                    single_act.tendons[0].segment_tensions[j] = t_hi;
                    let tp = ten_psi(&poses_tmp, &vels_tmp, &single_act);
                    single_act.tendons[0].segment_tensions[j] = t_lo;
                    let tm = ten_psi(&poses_tmp, &vels_tmp, &single_act);
                    single_act.tendons[0].segment_tensions[j] = t0;
                    if t_hi > t_lo {
                        next_bar_seg_tensions[ti][j] = (tp - tm) / (t_hi - t_lo);
                    }
                }
                let n_len = single_act.tendons[0].prev_segment_lengths.len();
                for j in 0..n_len {
                    let l0 = single_act.tendons[0].prev_segment_lengths[j];
                    single_act.tendons[0].prev_segment_lengths[j] = l0 + eps_local;
                    let lp = ten_psi(&poses_tmp, &vels_tmp, &single_act);
                    single_act.tendons[0].prev_segment_lengths[j] = l0 - eps_local;
                    let lm = ten_psi(&poses_tmp, &vels_tmp, &single_act);
                    single_act.tendons[0].prev_segment_lengths[j] = l0;
                    next_bar_prev_seg_lens[ti][j] = (lp - lm) / (2.0 * eps_local);
                }
            }
        }

        // --- Step C4: Articulated & Compliant Joint VJPs (local per joint -> bodies + u_k) ---
        for ji in 0..n_joints {
            let jspec = &world.graph.joints[ji];
            let ba = jspec.body_a;
            let bb = jspec.body_b;
            if lambda[ba].max_abs() == 0.0 && lambda[bb].max_abs() == 0.0 {
                continue;
            }

            let joint_psi = |poses_loc: &[Se3], vels_loc: &[Vec6], act_in: &ActuationState| -> f64 {
                let mut act_tmp = act_in.clone();
                let ev = evaluate_actuation_with_masses(
                    nb,
                    &[],
                    std::slice::from_ref(jspec),
                    &world.rest_poses,
                    poses_loc,
                    vels_loc,
                    &mut act_tmp,
                    dt * dt,
                    dt,
                    false,
                    false,
                    world.config.joint_stribeck_friction,
                    Some(&body_masses),
                    dt,
                );
                let mut val = dt
                    * (lambda[ba].dot(ev.body_wrenches[ba]) + lambda[bb].dot(ev.body_wrenches[bb]));
                for &(u, v, blk) in &ev.implicit_blocks {
                    if world.analysis.body_to_free_slot[u].is_some()
                        && world.analysis.body_to_free_slot[v].is_some()
                    {
                        val -= lambda[u].dot(blk.mul_vec(tape.v_next[v]));
                    }
                }
                val
            };

            let mut act_j = tape.act_after_servo.clone();
            let mut poses_tmp = tape.poses_k.clone();
            let mut vels_tmp = tape.vels_k.clone();

            for &bi in &[ba, bb] {
                if world.graph.bodies[bi].is_fixed {
                    continue;
                }
                let orig_p = poses_tmp[bi];
                for d in 0..6 {
                    let mut tw = Vec6::ZERO;
                    tw[d] = 1.0;
                    poses_tmp[bi] = orig_p.integrate_decoupled(tw, eps_local);
                    let pp = joint_psi(&poses_tmp, &vels_tmp, &act_j);
                    poses_tmp[bi] = orig_p.integrate_decoupled(tw, -eps_local);
                    let pm = joint_psi(&poses_tmp, &vels_tmp, &act_j);
                    next_bar_x[bi][d] += (pp - pm) / (2.0 * eps_local);
                }
                poses_tmp[bi] = orig_p;

                let orig_v = vels_tmp[bi];
                for d in 0..6 {
                    vels_tmp[bi][d] = orig_v[d] + eps_local;
                    let vp = joint_psi(&poses_tmp, &vels_tmp, &act_j);
                    vels_tmp[bi][d] = orig_v[d] - eps_local;
                    let vm = joint_psi(&poses_tmp, &vels_tmp, &act_j);
                    next_bar_v[bi][d] += (vp - vm) / (2.0 * eps_local);
                }
                vels_tmp[bi] = orig_v;
            }

            // Direct control gradient wrt joint target u_k
            if let Some(aid) = jspec.actuator_id {
                if let Some(Some(u_idx)) = joint_aid_to_u_idx.get(aid) {
                    if *u_idx < grad_controls[k].len() && aid < act_j.joint_targets.len() {
                        let t0 = act_j.joint_targets[aid];
                        act_j.joint_targets[aid] = t0 + eps_local;
                        let tp = joint_psi(&poses_tmp, &vels_tmp, &act_j);
                        act_j.joint_targets[aid] = t0 - eps_local;
                        let tm = joint_psi(&poses_tmp, &vels_tmp, &act_j);
                        act_j.joint_targets[aid] = t0;
                        grad_controls[k][*u_idx] += (tp - tm) / (2.0 * eps_local);
                    }
                }
            }
        }

        // --- Step C5: Contact Site VJPs (local per site -> node pose/vel + persistent ArpcSiteState) ---
        let mut next_bar_con_state = vec![[0.0_f64; 7]; n_sites];
        for si in 0..n_sites {
            let nid = match world.graph.contact_sites[si].attachment {
                ContactAttachment::RigidBody { body_id } => body_id,
                ContactAttachment::BeamInteriorNode {
                    beam_id,
                    node_local_idx,
                } => world.analysis.indexing.beam_maps[beam_id].free_node_ids[node_local_idx],
            };
            let is_fixed = nid < nb && world.graph.bodies[nid].is_fixed;
            let lam = lambda[nid];
            let vn = tape.v_next[nid];
            let b_st = bar_con_state[si];
            if lam.max_abs() == 0.0 && b_st.iter().all(|&x| x == 0.0) {
                continue;
            }

            let pose0 = tape.poses_k[nid];
            let vel0 = tape.vels_k[nid];
            let st0 = tape.con_before[si];

            if !is_fixed {
                for d in 0..6 {
                    let mut tw = Vec6::ZERO;
                    tw[d] = 1.0;
                    let pp = site_psi(
                        world,
                        si,
                        pose0.integrate_decoupled(tw, eps_local),
                        vel0,
                        st0,
                        dt,
                        lam,
                        vn,
                        b_st,
                    );
                    let pm = site_psi(
                        world,
                        si,
                        pose0.integrate_decoupled(tw, -eps_local),
                        vel0,
                        st0,
                        dt,
                        lam,
                        vn,
                        b_st,
                    );
                    next_bar_x[nid][d] += (pp - pm) / (2.0 * eps_local);

                    let mut vp_v = vel0;
                    let mut vm_v = vel0;
                    vp_v[d] += eps_local;
                    vm_v[d] -= eps_local;
                    let vp = site_psi(world, si, pose0, vp_v, st0, dt, lam, vn, b_st);
                    let vm = site_psi(world, si, pose0, vm_v, st0, dt, lam, vn, b_st);
                    next_bar_v[nid][d] += (vp - vm) / (2.0 * eps_local);
                }
            }

            let st_arr = pack_arpc_state(st0);
            for d in 0..7 {
                let mut sp = st_arr;
                let mut sm = st_arr;
                sp[d] += eps_local;
                sm[d] -= eps_local;
                let vp = site_psi(world, si, pose0, vel0, unpack_arpc_state(sp), dt, lam, vn, b_st);
                let vm = site_psi(world, si, pose0, vel0, unpack_arpc_state(sm), dt, lam, vn, b_st);
                next_bar_con_state[si][d] = (vp - vm) / (2.0 * eps_local);
            }
        }

        // --- Step C6: Backprop Tendon Servo Update (`advance_tendon_servos`) ---
        let mut next_bar_servo_pos = vec![0.0_f64; n_servo_slots];
        if n_tendons > 0 {
            let servo_psi = |act_in: &ActuationState| -> f64 {
                let mut tmp = act_in.clone();
                advance_tendon_servos(&world.graph.tendons, &mut tmp, dt);
                let mut val = 0.0;
                for aid in 0..n_servo_slots {
                    val += bar_servo_pos_mid[aid] * tmp.tendon_servo_pos[aid]
                        + bar_servo_vel_mid[aid] * tmp.tendon_servo_vel[aid];
                }
                val
            };

            let mut act_s = tape.act_before.clone();
            for aid in 0..n_servo_slots {
                if bar_servo_pos_mid[aid] == 0.0 && bar_servo_vel_mid[aid] == 0.0 {
                    continue;
                }
                // Derivative wrt tendon servo command u_k
                if let Some(Some(u_idx)) = tendon_aid_to_u_idx.get(aid) {
                    if *u_idx < grad_controls[k].len() {
                        let t0 = act_s.tendon_servo_target[aid];
                        act_s.tendon_servo_target[aid] = t0 + eps_local;
                        let tp = servo_psi(&act_s);
                        act_s.tendon_servo_target[aid] = t0 - eps_local;
                        let tm = servo_psi(&act_s);
                        act_s.tendon_servo_target[aid] = t0;
                        grad_controls[k][*u_idx] += (tp - tm) / (2.0 * eps_local);
                    }
                }
                // Derivative wrt previous servo_pos[aid]
                let p0 = act_s.tendon_servo_pos[aid];
                act_s.tendon_servo_pos[aid] = p0 + eps_local;
                let pp = servo_psi(&act_s);
                act_s.tendon_servo_pos[aid] = p0 - eps_local;
                let pm = servo_psi(&act_s);
                act_s.tendon_servo_pos[aid] = p0;
                next_bar_servo_pos[aid] = (pp - pm) / (2.0 * eps_local);
            }
            // Derivative wrt previous tendon tension (DC torque-speed derating chain!)
            for ti in 0..n_tendons {
                let aid = world.graph.tendons[ti].actuator_id;
                if bar_servo_pos_mid[aid] == 0.0 && bar_servo_vel_mid[aid] == 0.0 {
                    continue;
                }
                let ten0 = act_s.tendons[ti].tension;
                act_s.tendons[ti].tension = ten0 + eps_local;
                let tp = servo_psi(&act_s);
                act_s.tendons[ti].tension = (ten0 - eps_local).max(0.0);
                let tm = servo_psi(&act_s);
                let denom = (ten0 + eps_local) - (ten0 - eps_local).max(0.0);
                act_s.tendons[ti].tension = ten0;
                if denom > 0.0 {
                    next_bar_tendon_tension[ti] = (tp - tm) / denom;
                }
            }
        }

        // --- Step C7: Physical Parameter Gradients dL/dtheta_p (analytical + local entity VJP) ---
        for (pi, spec) in diff_params.iter().enumerate() {
            let dpsi = accumulate_param_grad_step(
                world,
                spec,
                tape,
                &lambda,
                &lambda_der_thetas,
                &bar_tendon_tension,
                &bar_seg_tensions,
                &bar_prev_seg_lens,
                &bar_con_state,
                g6,
                dt,
            );
            grad_params[pi] += dpsi * spec.scale;
        }

        bar_x = next_bar_x;
        bar_v = next_bar_v;
        bar_servo_pos = next_bar_servo_pos;
        bar_tendon_tension = next_bar_tendon_tension;
        bar_seg_tensions = next_bar_seg_tensions;
        bar_prev_seg_lens = next_bar_prev_seg_lens;
        bar_con_state = next_bar_con_state;
        bar_der_theta = next_bar_der_theta;
        bar_der_theta_vel = next_bar_der_theta_vel;
        bar_der_bishop_t = next_bar_der_bishop_t;
        bar_der_bishop_u = next_bar_der_bishop_u;
    }

    Ok(AdjointResult {
        loss: total_loss,
        grad_controls,
        grad_params,
        grad_init_poses: bar_x,
        grad_init_vels: bar_v,
        final_state,
    })
}

/// Evaluate exact step-`k` partial derivative `d Psi_k / d theta_p` for one [`DiffParamSpec`].
#[allow(clippy::too_many_arguments)]
fn accumulate_param_grad_step(
    world: &mut FlexGraphWorld,
    spec: &DiffParamSpec,
    tape: &StepTape,
    lambda: &[Vec6],
    lambda_der_thetas: &[Option<Vec<f64>>],
    bar_tendon_tension: &[f64],
    bar_seg_tensions: &[Vec<f64>],
    bar_prev_seg_lens: &[Vec<f64>],
    bar_con_state: &[[f64; 7]],
    g6: Vec6,
    dt: f64,
) -> f64 {
    let nb = world.graph.bodies.len();
    match spec.kind {
        // Exact analytical 1/E homogeneity of beam stiffness & elastic wrenches (Corotational & DER)!
        DiffParamKind::BeamYoungsModulus => {
            let ids = resolve_targets(&spec.target_ids, world.graph.beams.len());
            let mut acc = 0.0;
            for s in ids {
                let e_mod = world.graph.beams[s].material.youngs_modulus;
                if e_mod.abs() < 1e-30 {
                    continue;
                }
                let map = &world.analysis.indexing.beam_maps[s];
                match &world.beams[s] {
                    BeamRuntime::Corotational(data) => {
                        let ck = dt * data.rayleigh_beta + dt * dt;
                        for e in 0..data.n_elements {
                            let n0 = map.chain_node_ids[e];
                            let n1 = map.chain_node_ids[e + 1];
                            let lam = Vec12::from_halves(lambda[n0], lambda[n1]);
                            if lam.max_abs() == 0.0 {
                                continue;
                            }
                            let (_en, f_el, k_w, _m_w) = data.evaluate_element(
                                e,
                                tape.poses_k[n0],
                                tape.poses_k[n1],
                                world.config.use_strain_jacobian,
                            );
                            let v_n = Vec12::from_halves(tape.v_next[n0], tape.v_next[n1]);
                            acc += lam.dot(f_el * dt - k_w.mul_vec(v_n * ck)) / e_mod;
                        }
                    }
                    BeamRuntime::Der(data) => {
                        if let (Some(dst_k), Some(dst_next), Some(lam_th)) = (
                            tape.der_states_k[s].as_ref(),
                            tape.der_states_next[s].as_ref(),
                            lambda_der_thetas[s].as_deref(),
                        ) {
                            let ck = dt * data.rayleigh_beta + dt * dt;
                            let (grad_dot, hess_dot) = data.eval_elastic_contraction(
                                map,
                                &tape.poses_k,
                                dst_k,
                                lambda,
                                lam_th,
                                &tape.v_next,
                                &dst_next.theta_vel,
                                world.config.project_der_hessian_spd,
                            );
                            acc += (-dt * grad_dot - ck * hess_dot) / e_mod;
                        }
                    }
                }
            }
            acc
        }
        // Exact analytical Rayleigh beta derivative: d(ck)/dbeta = dt => -dt * lam^T K v_next
        DiffParamKind::BeamDampingBeta => {
            let ids = resolve_targets(&spec.target_ids, world.graph.beams.len());
            let mut acc = 0.0;
            for s in ids {
                let map = &world.analysis.indexing.beam_maps[s];
                match &world.beams[s] {
                    BeamRuntime::Corotational(data) => {
                        for e in 0..data.n_elements {
                            let n0 = map.chain_node_ids[e];
                            let n1 = map.chain_node_ids[e + 1];
                            let lam = Vec12::from_halves(lambda[n0], lambda[n1]);
                            if lam.max_abs() == 0.0 {
                                continue;
                            }
                            let (_en, _f_el, k_w, _m_w) = data.evaluate_element(
                                e,
                                tape.poses_k[n0],
                                tape.poses_k[n1],
                                world.config.use_strain_jacobian,
                            );
                            let v_n = Vec12::from_halves(tape.v_next[n0], tape.v_next[n1]);
                            acc -= dt * lam.dot(k_w.mul_vec(v_n));
                        }
                    }
                    BeamRuntime::Der(data) => {
                        if let (Some(dst_k), Some(dst_next), Some(lam_th)) = (
                            tape.der_states_k[s].as_ref(),
                            tape.der_states_next[s].as_ref(),
                            lambda_der_thetas[s].as_deref(),
                        ) {
                            let (_grad_dot, hess_dot) = data.eval_elastic_contraction(
                                map,
                                &tape.poses_k,
                                dst_k,
                                lambda,
                                lam_th,
                                &tape.v_next,
                                &dst_next.theta_vel,
                                world.config.project_der_hessian_spd,
                            );
                            acc -= dt * hess_dot;
                        }
                    }
                }
            }
            acc
        }
        // Exact BodyMass derivative: rigid inertia/gyroscopic + positive-mode SolRefImp sites & actuation
        DiffParamKind::BodyMass => {
            let ids = resolve_targets(&spec.target_ids, nb);
            let mut acc = 0.0;
            for bi in ids {
                if world.graph.bodies[bi].is_fixed {
                    continue;
                }
                let lam = lambda[bi];
                if lam.max_abs() > 0.0 {
                    let q = tape.poses_k[bi].rot;
                    let v = tape.vels_k[bi];
                    let v_next = tape.v_next[bi];
                    let alpha = world.graph.bodies[bi].rayleigh_alpha;
                    let unit_si = SpatialInertia {
                        mass: 1.0,
                        com: world.graph.bodies[bi].com_local,
                        rot_inertia_com: Mat3::ZERO,
                    };
                    let dm6 = if world.config.world_fixed_inertia {
                        unit_si.to_mat6()
                    } else {
                        unit_si.world_mat6(q)
                    };
                    let d_gyro = if world.config.world_fixed_inertia {
                        Vec6::ZERO
                    } else {
                        let diw = dm6.block(1, 1);
                        let w = v.ang();
                        Vec6::from_parts(Vec3::ZERO, w.cross(diw.mul_vec(w)))
                    };
                    let rhs_minus_lhs =
                        dm6.mul_vec(v + dt * g6 - v_next * (1.0 + dt * alpha)) - d_gyro * dt;
                    acc += lam.dot(rhs_minus_lhs);
                }

                // Positive-mode SolRefImp contact sites, tendons, and joints scale with body mass!
                let m0 = world.graph.bodies[bi].mass;
                let eps = 1e-6 * m0.abs().max(1e-6);

                // 1. Attached RigidBody contact sites with positive-mode SolRefImp
                let affected_sites: Vec<usize> = match &world.config.contact {
                    ContactBackend::Arpc { .. } => world
                        .graph
                        .contact_sites
                        .iter()
                        .enumerate()
                        .filter_map(|(si, s)| {
                            let attached =
                                matches!(s.attachment, ContactAttachment::RigidBody { body_id } if body_id == bi);
                            if attached && s.solref_imp.is_some_and(|sri| !sri.is_direct_mode()) {
                                Some(si)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    ContactBackend::GpuCoreKelvinParity { .. } => world
                        .graph
                        .contact_sites
                        .iter()
                        .enumerate()
                        .filter_map(|(si, s)| {
                            let attached =
                                matches!(s.attachment, ContactAttachment::RigidBody { body_id } if body_id == bi);
                            let non_wheel = !matches!(s.mode, ContactMode::RatchetWheel { .. });
                            if attached
                                && non_wheel
                                && s.solref_imp.is_some_and(|sri| !sri.is_direct_mode())
                            {
                                Some(si)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                if !affected_sites.is_empty() {
                    let eval_mass_sites = |w: &mut FlexGraphWorld, m_val: f64| -> f64 {
                        w.graph.bodies[bi].mass = m_val;
                        let mut sum = 0.0;
                        for &si in &affected_sites {
                            sum += site_psi(
                                w,
                                si,
                                tape.poses_k[bi],
                                tape.vels_k[bi],
                                tape.con_before[si],
                                dt,
                                lambda[bi],
                                tape.v_next[bi],
                                bar_con_state[si],
                            );
                        }
                        sum
                    };
                    let pp = eval_mass_sites(world, m0 + eps);
                    let pm = eval_mass_sites(world, m0 - eps);
                    world.graph.bodies[bi].mass = m0;
                    acc += (pp - pm) / (2.0 * eps);
                }

                // 2. Attached tendons using positive-mode SolRefImp (reduced pair mass depends on m_bi)
                let affected_tendons: Vec<usize> = world
                    .graph
                    .tendons
                    .iter()
                    .enumerate()
                    .filter_map(|(ti, t)| {
                        let touches_bi = t
                            .waypoints
                            .first()
                            .is_some_and(|w| w.body_id == bi)
                            || t.waypoints.last().is_some_and(|w| w.body_id == bi);
                        if touches_bi && t.solref_imp.is_some_and(|sri| !sri.is_direct_mode()) {
                            Some(ti)
                        } else {
                            None
                        }
                    })
                    .collect();
                if !affected_tendons.is_empty() {
                    let eval_mass_tendons = |w: &mut FlexGraphWorld, m_val: f64| -> f64 {
                        w.graph.bodies[bi].mass = m_val;
                        let bm = w.body_masses();
                        let mut sum = 0.0;
                        for &ti in &affected_tendons {
                            let tspec = &w.graph.tendons[ti];
                            let mut single_act = ActuationState {
                                tendon_servo_pos: tape.act_after_servo.tendon_servo_pos.clone(),
                                tendon_servo_target: tape
                                    .act_after_servo
                                    .tendon_servo_target
                                    .clone(),
                                tendon_servo_vel: tape.act_after_servo.tendon_servo_vel.clone(),
                                tendons: vec![tape.act_after_servo.tendons[ti].clone()],
                                joint_targets: Vec::new(),
                                last_joint_torque: Vec::new(),
                                last_joint_torque_cap: Vec::new(),
                            };
                            let ev = evaluate_actuation_with_masses(
                                nb,
                                std::slice::from_ref(tspec),
                                &[],
                                &w.rest_poses,
                                &tape.poses_k,
                                &tape.vels_k,
                                &mut single_act,
                                0.0,
                                0.0,
                                false,
                                true,
                                w.config.joint_stribeck_friction,
                                Some(&bm),
                                dt,
                            );
                            sum += bar_tendon_tension[ti] * single_act.tendons[0].tension;
                            if tspec.stick_slip_enabled {
                                for (&b, &v) in bar_seg_tensions[ti]
                                    .iter()
                                    .zip(single_act.tendons[0].segment_tensions.iter())
                                {
                                    sum += b * v;
                                }
                                for (&b, &v) in bar_prev_seg_lens[ti]
                                    .iter()
                                    .zip(single_act.tendons[0].prev_segment_lengths.iter())
                                {
                                    sum += b * v;
                                }
                            }
                            let mut uniq: Vec<usize> =
                                tspec.waypoints.iter().map(|wp| wp.body_id).collect();
                            uniq.sort_unstable();
                            uniq.dedup();
                            for b_u in uniq {
                                sum += dt * lambda[b_u].dot(ev.body_wrenches[b_u]);
                            }
                        }
                        sum
                    };
                    let pp = eval_mass_tendons(world, m0 + eps);
                    let pm = eval_mass_tendons(world, m0 - eps);
                    world.graph.bodies[bi].mass = m0;
                    acc += (pp - pm) / (2.0 * eps);
                }

                // 3. Attached joints using positive-mode SolRefImp (reduced pair mass depends on m_bi)
                let affected_joints: Vec<usize> = world
                    .graph
                    .joints
                    .iter()
                    .enumerate()
                    .filter_map(|(ji, j)| {
                        if (j.body_a == bi || j.body_b == bi)
                            && j.solref_imp.is_some_and(|sri| !sri.is_direct_mode())
                        {
                            Some(ji)
                        } else {
                            None
                        }
                    })
                    .collect();
                if !affected_joints.is_empty() {
                    let eval_mass_joints = |w: &mut FlexGraphWorld, m_val: f64| -> f64 {
                        w.graph.bodies[bi].mass = m_val;
                        let bm = w.body_masses();
                        let mut sum = 0.0;
                        for &ji in &affected_joints {
                            let jspec = &w.graph.joints[ji];
                            let ba = jspec.body_a;
                            let bb = jspec.body_b;
                            let mut act_tmp = tape.act_after_servo.clone();
                            let ev = evaluate_actuation_with_masses(
                                nb,
                                &[],
                                std::slice::from_ref(jspec),
                                &w.rest_poses,
                                &tape.poses_k,
                                &tape.vels_k,
                                &mut act_tmp,
                                dt * dt,
                                dt,
                                false,
                                false,
                                w.config.joint_stribeck_friction,
                                Some(&bm),
                                dt,
                            );
                            sum += dt
                                * (lambda[ba].dot(ev.body_wrenches[ba])
                                    + lambda[bb].dot(ev.body_wrenches[bb]));
                            for &(u, v, blk) in &ev.implicit_blocks {
                                if w.analysis.body_to_free_slot[u].is_some()
                                    && w.analysis.body_to_free_slot[v].is_some()
                                {
                                    sum -= lambda[u].dot(blk.mul_vec(tape.v_next[v]));
                                }
                            }
                        }
                        sum
                    };
                    let pp = eval_mass_joints(world, m0 + eps);
                    let pm = eval_mass_joints(world, m0 - eps);
                    world.graph.bodies[bi].mass = m0;
                    acc += (pp - pm) / (2.0 * eps);
                }
            }
            acc
        }
        // Local TendonStiffness VJP (only touches targeted tendons!)
        DiffParamKind::TendonStiffness => {
            let ids = resolve_targets(&spec.target_ids, world.graph.tendons.len());
            let cur_val = get_diff_param_value(world, spec);
            let eps = 1e-6 * cur_val.abs().max(1.0);
            let bm = world.body_masses();
            let eval_ten = |w: &mut FlexGraphWorld, val: f64| -> f64 {
                set_diff_param_value(w, spec, val, false);
                let mut sum = 0.0;
                for &ti in &ids {
                    let tspec = &w.graph.tendons[ti];
                    let mut single_act = ActuationState {
                        tendon_servo_pos: tape.act_after_servo.tendon_servo_pos.clone(),
                        tendon_servo_target: tape.act_after_servo.tendon_servo_target.clone(),
                        tendon_servo_vel: tape.act_after_servo.tendon_servo_vel.clone(),
                        tendons: vec![tape.act_after_servo.tendons[ti].clone()],
                        joint_targets: Vec::new(),
                        last_joint_torque: Vec::new(),
                        last_joint_torque_cap: Vec::new(),
                    };
                    let ev = evaluate_actuation_with_masses(
                        nb,
                        std::slice::from_ref(tspec),
                        &[],
                        &w.rest_poses,
                        &tape.poses_k,
                        &tape.vels_k,
                        &mut single_act,
                        0.0,
                        0.0,
                        false,
                        true,
                        w.config.joint_stribeck_friction,
                        Some(&bm),
                        dt,
                    );
                    sum += bar_tendon_tension[ti] * single_act.tendons[0].tension;
                    if tspec.stick_slip_enabled {
                        for (&b, &v) in bar_seg_tensions[ti]
                            .iter()
                            .zip(single_act.tendons[0].segment_tensions.iter())
                        {
                            sum += b * v;
                        }
                        for (&b, &v) in bar_prev_seg_lens[ti]
                            .iter()
                            .zip(single_act.tendons[0].prev_segment_lengths.iter())
                        {
                            sum += b * v;
                        }
                    }
                    let mut uniq: Vec<usize> = tspec.waypoints.iter().map(|wp| wp.body_id).collect();
                    uniq.sort_unstable();
                    uniq.dedup();
                    for bi in uniq {
                        sum += dt * lambda[bi].dot(ev.body_wrenches[bi]);
                    }
                }
                sum
            };
            let pp = eval_ten(world, cur_val + eps);
            let pm = eval_ten(world, cur_val - eps);
            set_diff_param_value(world, spec, cur_val, false);
            (pp - pm) / (2.0 * eps)
        }
        // Local ContactMuFwd / ContactMuRev / ContactStiffness VJP (sites + beam interior SDF contacts!)
        DiffParamKind::ContactMuFwd
        | DiffParamKind::ContactMuRev
        | DiffParamKind::ContactStiffness => {
            let cur_val = get_diff_param_value(world, spec);
            let eps = 1e-6 * cur_val.abs().max(1.0);
            let is_global_contact = matches!(
                (&world.config.contact, spec.kind),
                (
                    ContactBackend::SdfBarrier { .. },
                    DiffParamKind::ContactStiffness | DiffParamKind::ContactMuFwd
                ) | (
                    ContactBackend::GpuCoreKelvinParity { .. },
                    DiffParamKind::ContactStiffness
                ) | (ContactBackend::Arpc { .. }, DiffParamKind::ContactStiffness)
            ) || (spec.kind == DiffParamKind::ContactMuFwd
                && resolve_targets(&spec.target_ids, world.graph.contact_sites.len())
                    .iter()
                    .any(|&si| {
                        matches!(
                            world.graph.contact_sites[si].mode,
                            ContactMode::IsotropicPad { mu, .. } if mu <= 0.0
                        )
                    }));
            let site_ids = if is_global_contact || spec.target_ids.is_empty() {
                (0..world.graph.contact_sites.len()).collect()
            } else {
                resolve_targets(&spec.target_ids, world.graph.contact_sites.len())
            };
            let check_beam_interior = matches!(
                spec.kind,
                DiffParamKind::ContactStiffness | DiffParamKind::ContactMuFwd
            );
            let eval_sites = |w: &mut FlexGraphWorld, val: f64| -> f64 {
                set_diff_param_value(w, spec, val, false);
                let mut sum = 0.0;
                for &si in &site_ids {
                    let nid = match w.graph.contact_sites[si].attachment {
                        ContactAttachment::RigidBody { body_id } => body_id,
                        ContactAttachment::BeamInteriorNode {
                            beam_id,
                            node_local_idx,
                        } => w.analysis.indexing.beam_maps[beam_id].free_node_ids[node_local_idx],
                    };
                    sum += site_psi(
                        w,
                        si,
                        tape.poses_k[nid],
                        tape.vels_k[nid],
                        tape.con_before[si],
                        dt,
                        lambda[nid],
                        tape.v_next[nid],
                        bar_con_state[si],
                    );
                }
                if check_beam_interior {
                    for (s, bspec) in w.graph.beams.iter().enumerate() {
                        let Some(radius) = bspec.interior_node_contact_radius else {
                            continue;
                        };
                        let map = &w.analysis.indexing.beam_maps[s];
                        for &nid in &map.free_node_ids {
                            let lam_l = lambda[nid].lin();
                            if lam_l.norm_sq() == 0.0 {
                                continue;
                            }
                            sum += beam_interior_contact_psi(
                                w,
                                radius,
                                tape.poses_k[nid].pos,
                                tape.vels_k[nid].lin(),
                                dt,
                                lam_l,
                                tape.v_next[nid].lin(),
                            );
                        }
                    }
                }
                sum
            };
            let pp = eval_sites(world, cur_val + eps);
            let pm = eval_sites(world, cur_val - eps);
            set_diff_param_value(world, spec, cur_val, false);
            (pp - pm) / (2.0 * eps)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fg_ir::{
        BeamEdgeSpec, BeamFormulation, ContactSiteSpec, JointEdgeSpec, JointType, RigidBodySpec,
        RigidFlexGraph, TendonDriveMode, TendonEdgeSpec, TendonWaypoint,
    };
    use fg_math::{BeamMaterial, BeamSection};
    use fg_sdf::{MaterialLibrary, PlaneSdf, WorldSdf, MAT_HARD_FLOOR};
    use std::sync::Arc;

    fn make_body(id: usize, pos: Vec3, is_fixed: bool, mass: f64) -> RigidBodySpec {
        RigidBodySpec {
            id,
            name: format!("b{id}"),
            is_fixed,
            mass,
            inertia: Mat3::diag([4.0e-4, 2.5e-4, 2.5e-4]),
            com_local: Vec3::ZERO,
            rest_pose: Se3::new(pos, Quat::IDENTITY),
            collision_radius: 0.015,
            rayleigh_alpha: 0.05,
            tags: Vec::new(),
        }
    }

    fn make_beam_strip(
        p0: Vec3,
        p1: Vec3,
        n_el: usize,
        youngs_modulus: f64,
        beta: f64,
    ) -> BeamEdgeSpec {
        let bow = 0.02;
        let poses: Vec<Se3> = (0..=n_el)
            .map(|i| {
                let u = i as f64 / n_el as f64;
                let pos = p0 + (p1 - p0) * u + Vec3::new(0.0, 0.0, 4.0 * bow * u * (1.0 - u));
                let tang = ((p1 - p0) + Vec3::new(0.0, 0.0, 4.0 * bow * (1.0 - 2.0 * u))).normalized();
                let w = Vec3::EY;
                let n = tang.cross(w).normalized();
                Se3::new(pos, Quat::from_mat3(Mat3::from_cols(tang, w, n)))
            })
            .collect();
        BeamEdgeSpec {
            id: 0,
            name: "strip_0".into(),
            body_a: 0,
            clamp_a: Se3::new(Vec3::ZERO, poses[0].rot),
            body_b: Some(1),
            clamp_b: Some(Se3::new(Vec3::ZERO, poses[n_el].rot)),
            n_elements: n_el,
            formulation: BeamFormulation::CorotationalEulerBernoulli,
            section: BeamSection::Rectangular {
                width: 0.016,
                thickness: 0.00025,
            },
            material: BeamMaterial {
                youngs_modulus,
                rayleigh_alpha: 0.05,
                rayleigh_beta: beta,
                ..BeamMaterial::SPRING_STEEL
            },
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses,
            element_lengths: Vec::new(),
            element_transforms: None,
            interior_node_contact_radius: None,
            enable_self_contact: false,
        }
    }

    fn rollout_loss(
        world: &mut FlexGraphWorld,
        controls: &[Vec<f64>],
        dt: f64,
        loss: &dyn TrajectoryLoss,
    ) -> f64 {
        let saved = world.state.clone();
        let mut total = 0.0;
        for (k, u_k) in controls.iter().enumerate() {
            let (lk, _, _, _) =
                loss.step_loss_and_grad(k, &world.state.poses, &world.state.vels, Some(u_k));
            total += lk;
            world.set_all_targets(u_k);
            world.step(dt).unwrap();
        }
        let (lt, _, _, _) =
            loss.step_loss_and_grad(controls.len(), &world.state.poses, &world.state.vels, None);
        total += lt;
        world.state = saved;
        total
    }

    #[test]
    fn test_ift_adjoint_free_flight_matches_central_fd_to_1e5() {
        // 2-plate flexible steel-arch system driven by tendon + yaw revolute in 3D free flight
        let p0 = Vec3::new(0.0, 0.0, 0.2);
        let p1 = Vec3::new(0.12, 0.0, 0.2);
        let strip = make_beam_strip(p0, p1, 4, 8.0e10, 0.002);
        let tendon = TendonEdgeSpec {
            id: 0,
            name: "t0".into(),
            actuator_id: 0,
            waypoints: vec![
                TendonWaypoint {
                    body_id: 0,
                    local_pos: Vec3::new(0.0, 0.015, -0.005),
                    wrap_angle_rad: 0.0,
                },
                TendonWaypoint {
                    body_id: 1,
                    local_pos: Vec3::new(0.0, 0.015, -0.005),
                    wrap_angle_rad: 0.0,
                },
            ],
            drive_mode: TendonDriveMode::Linear,
            rest_length: 0.115,
            slack: 0.0,
            stiffness: 1200.0,
            damping: 0.4,
            capstan_mu_theta: 0.0,
            stick_slip_enabled: false,
            servo_torque_limit: 50.0,
            servo_speed_limit: 5.0,
            servo_deadband: 0.0,
            angle_limits: [0.0, 0.05],
            solref_imp: None,
        };
        let joint = JointEdgeSpec {
            id: 0,
            name: "yaw".into(),
            body_a: 0,
            body_b: 1,
            joint_type: JointType::Revolute {
                axis_local_a: Vec3::EZ,
                anchor_a: Vec3::new(0.12, 0.0, 0.0),
                anchor_b: Vec3::ZERO,
                pos_stiffness: 0.0,
                pos_damping: 0.0,
                align_stiffness: 0.0,
                align_damping: 0.0,
            },
            actuator_id: Some(1),
            kp: 1.5,
            kd: 0.05,
            torque_limit: 5.0,
            speed_limit: 0.0,
            static_friction: 0.0,
            deadband: 0.0,
            limits: None,
            implicit_coupling: false,
            solref_imp: None,
        };
        let graph = RigidFlexGraph {
            name: "adj_free_flight".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_body(0, p0, false, 0.25), make_body(1, p1, false, 0.28)],
            beams: vec![strip],
            tendons: vec![tendon],
            joints: vec![joint],
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = crate::SolverConfig {
            integrator: crate::IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::None,
            use_strain_jacobian: true,
            world_fixed_inertia: false,
            ..crate::SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
        world.state.vels[0] = Vec6::from_parts(Vec3::new(0.2, -0.1, 0.05), Vec3::new(0.3, -0.2, 0.4));
        world.state.vels[1] = Vec6::from_parts(Vec3::new(-0.1, 0.15, 0.02), Vec3::new(-0.2, 0.3, -0.1));

        let dt = 0.001;
        let n_steps = 12;
        let controls: Vec<Vec<f64>> = (0..n_steps)
            .map(|k| vec![0.002 + 0.0005 * (k as f64), 0.15 + 0.02 * (k as f64)])
            .collect();

        // Build shifted target trajectory for non-trivial loss
        let targets: Vec<Vec<Se3>> = (0..=n_steps)
            .map(|k| {
                let t = k as f64 * dt;
                vec![
                    Se3::new(p0 + Vec3::new(0.02 * t, 0.005, -0.002), Quat::from_axis_angle(Vec3::EX, 0.05)),
                    Se3::new(p1 + Vec3::new(-0.01 * t, -0.004, 0.003), Quat::from_axis_angle(Vec3::EZ, -0.08)),
                ]
            })
            .collect();
        let mut loss = PoseTrackingLoss::new(targets, 100.0, 2.0);
        loss.weight_ctrl = 0.1;

        let diff_params = vec![
            DiffParamSpec::shared("E", DiffParamKind::BeamYoungsModulus),
            DiffParamSpec::shared("beta", DiffParamKind::BeamDampingBeta),
            DiffParamSpec::shared("k_ten", DiffParamKind::TendonStiffness),
            DiffParamSpec::targeted("m0", DiffParamKind::BodyMass, vec![0]),
        ];

        let adj = compute_trajectory_adjoint(&mut world, &controls, dt, &diff_params, &loss).unwrap();

        // Verify ALL physical parameter gradients against central FD (< 1e-5 relative error!)
        for (pi, spec) in diff_params.iter().enumerate() {
            let v0 = get_diff_param_value(&world, spec);
            let eps = 1e-5 * v0.abs().max(1.0);
            set_diff_param_value(&mut world, spec, v0 + eps, false);
            let lp = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, spec, v0 - eps, false);
            let lm = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, spec, v0, false);
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj.grad_params[pi];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-8);
            assert!(
                rel < 1e-5,
                "param {} free-flight IFT vs FD mismatch: an={an:e} fd={fd:e} rel={rel:e}",
                spec.name
            );
        }

        // Verify per-step control gradients (both tendon servo & revolute joint) against central FD
        for &step_chk in &[0usize, 5, n_steps - 1] {
            for a in 0..2 {
                let eps = 1e-5;
                let mut u_p = controls.clone();
                let mut u_m = controls.clone();
                u_p[step_chk][a] += eps;
                u_m[step_chk][a] -= eps;
                let lp = rollout_loss(&mut world, &u_p, dt, &loss);
                let lm = rollout_loss(&mut world, &u_m, dt, &loss);
                let fd = (lp - lm) / (2.0 * eps);
                let an = adj.grad_controls[step_chk][a];
                let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-8);
                assert!(
                    rel < 1e-5,
                    "control u[{step_chk}][{a}] IFT vs FD mismatch: an={an:e} fd={fd:e} rel={rel:e}"
                );
            }
        }
        // Verify initial pose & velocity cotangents (`grad_init_poses`, `grad_init_vels`) vs central FD
        for d in 0..6 {
            let eps = 1e-5;
            let orig_p = world.state.poses[0];
            let mut tw = Vec6::ZERO;
            tw[d] = 1.0;
            world.state.poses[0] = orig_p.integrate_decoupled(tw, eps);
            let lp = rollout_loss(&mut world, &controls, dt, &loss);
            world.state.poses[0] = orig_p.integrate_decoupled(tw, -eps);
            let lm = rollout_loss(&mut world, &controls, dt, &loss);
            world.state.poses[0] = orig_p;
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj.grad_init_poses[0][d];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-8);
            assert!(
                rel < 1e-5,
                "init pose[0][{d}] IFT vs FD mismatch: an={an:e} fd={fd:e} rel={rel:e}"
            );

            let orig_v = world.state.vels[0];
            world.state.vels[0][d] = orig_v[d] + eps;
            let vp = rollout_loss(&mut world, &controls, dt, &loss);
            world.state.vels[0][d] = orig_v[d] - eps;
            let vm = rollout_loss(&mut world, &controls, dt, &loss);
            world.state.vels[0] = orig_v;
            let fd_v = (vp - vm) / (2.0 * eps);
            let an_v = adj.grad_init_vels[0][d];
            let rel_v = (an_v - fd_v).abs() / fd_v.abs().max(an_v.abs()).max(1e-8);
            assert!(
                rel_v < 1e-5,
                "init vel[0][{d}] IFT vs FD mismatch: an={an_v:e} fd={fd_v:e} rel={rel_v:e}"
            );
        }
    }

    #[test]
    fn test_ift_adjoint_active_contact_matches_central_fd_and_recovers_sysid() {
        // Active-contact 2-body flexible beam sliding on flat floor under ARPC implicit contact!
        let make_contact_world = |e_mod: f64, mu_val: f64| -> FlexGraphWorld {
            let p0 = Vec3::new(0.0, 0.0, 0.0185); // 1.5 mm active penetration into z=0 (r=0.02)
            let p1 = Vec3::new(0.12, 0.0, 0.0185);
            let strip = make_beam_strip(p0, p1, 4, e_mod, 0.002);
            let sites = vec![
                ContactSiteSpec {
                    id: 0,
                    name: "pad0".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 0 },
                    local_pos: Vec3::ZERO,
                    local_frame: Quat::IDENTITY,
                    radius: 0.02,
                    patch_radius: 0.005,
                    mode: ContactMode::IsotropicPad {
                        mu: mu_val,
                        pivot_mu: 0.1,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: None,
                },
                ContactSiteSpec {
                    id: 1,
                    name: "pad1".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 1 },
                    local_pos: Vec3::ZERO,
                    local_frame: Quat::IDENTITY,
                    radius: 0.02,
                    patch_radius: 0.005,
                    mode: ContactMode::IsotropicPad {
                        mu: mu_val,
                        pivot_mu: 0.1,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: None,
                },
            ];
            let tendon = TendonEdgeSpec {
                id: 0,
                name: "t0".into(),
                actuator_id: 0,
                waypoints: vec![
                    TendonWaypoint {
                        body_id: 0,
                        local_pos: Vec3::ZERO,
                        wrap_angle_rad: 0.0,
                    },
                    TendonWaypoint {
                        body_id: 1,
                        local_pos: Vec3::ZERO,
                        wrap_angle_rad: 0.0,
                    },
                ],
                drive_mode: TendonDriveMode::Linear,
                rest_length: 0.12,
                slack: 0.0,
                stiffness: 1500.0,
                damping: 0.5,
                capstan_mu_theta: 0.0,
                stick_slip_enabled: false,
                servo_torque_limit: 50.0,
                servo_speed_limit: 5.0,
                servo_deadband: 0.0,
                angle_limits: [0.0, 0.04],
                solref_imp: None,
            };
            let graph = RigidFlexGraph {
                name: "adj_active_contact".into(),
                gravity: Vec3::new(0.0, 0.0, -9.81),
                bodies: vec![make_body(0, p0, false, 0.30), make_body(1, p1, false, 0.30)],
                beams: vec![strip],
                tendons: vec![tendon],
                joints: Vec::new(),
                contact_sites: sites,
                symmetry: Default::default(),
                solver_hints: Default::default(),
                diff_params: Vec::new(),
                mesh_colliders: Vec::new(),
            };
            let mut sdf = WorldSdf::new(MaterialLibrary::standard());
            sdf.add_unbounded(Arc::new(PlaneSdf::flat_ground(0.0, MAT_HARD_FLOOR)));
            let cfg = crate::SolverConfig {
                integrator: crate::IntegratorKind::LinearlyImplicitEuler,
                contact: ContactBackend::Arpc { implicit: true },
                use_strain_jacobian: true,
                world_fixed_inertia: false,
                ..crate::SolverConfig::default()
            };
            let mut w = FlexGraphWorld::new(graph, sdf, cfg).unwrap();
            // Steady sliding velocity in +X so Coulomb friction is smoothly active
            w.state.vels[0] = Vec6::from_parts(Vec3::new(0.35, 0.0, 0.0), Vec3::ZERO);
            w.state.vels[1] = Vec6::from_parts(Vec3::new(0.35, 0.0, 0.0), Vec3::ZERO);
            w
        };

        let dt = 0.001;
        let n_steps = 15;
        let controls: Vec<Vec<f64>> = (0..n_steps)
            .map(|k| vec![0.001 + 0.0008 * (k as f64)])
            .collect();

        // 1. Record ground-truth observation trajectory at true (E_true = 60 GPa, mu_true = 0.45)
        let e_true = 6.0e10;
        let mu_true = 0.45;
        let mut truth_world = make_contact_world(e_true, mu_true);
        let mut target_poses = vec![vec![truth_world.state.poses[0], truth_world.state.poses[1]]];
        let mut target_vels = vec![vec![truth_world.state.vels[0], truth_world.state.vels[1]]];
        for u_k in &controls {
            truth_world.set_all_targets(u_k);
            truth_world.step(dt).unwrap();
            target_poses.push(vec![truth_world.state.poses[0], truth_world.state.poses[1]]);
            target_vels.push(vec![truth_world.state.vels[0], truth_world.state.vels[1]]);
        }
        assert!(truth_world.state.contact_normal_forces[0] > 0.5);

        let mut loss = PoseTrackingLoss::new(target_poses, 1.0e4, 10.0);
        loss.target_vels = Some(target_vels);
        loss.weight_vel = 10.0;

        // 2. Verify active-contact IFT adjoint vs central FD at wrong initial guess (E0 = 85 GPa, mu0 = 0.28)
        let mut world = make_contact_world(8.5e10, 0.28);
        let diff_params = vec![
            DiffParamSpec {
                name: "E".into(),
                kind: DiffParamKind::BeamYoungsModulus,
                target_ids: Vec::new(),
                bounds: Some([1.0e10, 3.0e11]),
                scale: 1.0,
            },
            DiffParamSpec {
                name: "mu".into(),
                kind: DiffParamKind::ContactMuFwd,
                target_ids: Vec::new(),
                bounds: Some([0.05, 1.5]),
                scale: 1.0,
            },
            DiffParamSpec::shared("k_con", DiffParamKind::ContactStiffness),
        ];

        let adj = compute_trajectory_adjoint(&mut world, &controls, dt, &diff_params, &loss).unwrap();
        for (pi, spec) in diff_params.iter().enumerate() {
            let v0 = get_diff_param_value(&world, spec);
            let eps = 1e-5 * v0.abs().max(1.0);
            set_diff_param_value(&mut world, spec, v0 + eps, false);
            let lp = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, spec, v0 - eps, false);
            let lm = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, spec, v0, false);
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj.grad_params[pi];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-8);
            assert!(
                rel < 1e-5,
                "active-contact param {} IFT vs FD mismatch: an={an:e} fd={fd:e} rel={rel:e}",
                spec.name
            );
        }

        // 3. Sim2Real SysID recovery: recover unknown (E_true, mu_true) from (+41%, -38%) wrong guess to < 1%!
        let sysid_params = vec![
            DiffParamSpec {
                name: "E_norm".into(),
                kind: DiffParamKind::BeamYoungsModulus,
                target_ids: Vec::new(),
                bounds: Some([1.0e10, 3.0e11]),
                scale: 1.0e10, // nondimensionalize E in 10-GPa units for well-conditioned Adam!
            },
            DiffParamSpec {
                name: "mu".into(),
                kind: DiffParamKind::ContactMuFwd,
                target_ids: Vec::new(),
                bounds: Some([0.05, 1.5]),
                scale: 1.0,
            },
        ];
        let mut theta = [8.5_f64, 0.28_f64]; // E = 85 GPa (+41.7%), mu = 0.28 (-37.8%)
        let mut m = [0.0_f64; 2];
        let mut v = [0.0_f64; 2];
        let lr = 0.15;
        let beta1 = 0.85;
        let beta2 = 0.99;

        for iter in 1..=120 {
            set_diff_param_value(&mut world, &sysid_params[0], theta[0] * 1.0e10, true);
            set_diff_param_value(&mut world, &sysid_params[1], theta[1], true);
            let res =
                compute_trajectory_adjoint(&mut world, &controls, dt, &sysid_params, &loss).unwrap();
            if res.loss < 1e-12 {
                break;
            }
            for j in 0..2 {
                let g = res.grad_params[j];
                m[j] = beta1 * m[j] + (1.0 - beta1) * g;
                v[j] = beta2 * v[j] + (1.0 - beta2) * g * g;
                let m_hat = m[j] / (1.0 - beta1.powi(iter));
                let v_hat = v[j] / (1.0 - beta2.powi(iter));
                theta[j] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }
            theta[0] = theta[0].clamp(1.0, 30.0);
            theta[1] = theta[1].clamp(0.05, 1.5);
        }

        let e_est = theta[0] * 1.0e10;
        let mu_est = theta[1];
        let e_rel_err = (e_est - e_true).abs() / e_true;
        let mu_rel_err = (mu_est - mu_true).abs() / mu_true;
        assert!(
            e_rel_err < 0.01,
            "SysID failed to recover E to < 1%: e_est={e_est:e} e_true={e_true:e} rel={e_rel_err:e}"
        );
        assert!(
            mu_rel_err < 0.01,
            "SysID failed to recover mu to < 1%: mu_est={mu_est} mu_true={mu_true} rel={mu_rel_err:e}"
        );

        // 4. Verify ContactMuRev (7th DiffParamKind) on an AnisotropicScale site sliding backward (-X)
        let p_rev = Vec3::new(0.0, 0.0, 0.0185);
        let graph_rev = RigidFlexGraph {
            name: "adj_aniso_rev".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_body(0, p_rev, false, 0.30)],
            beams: Vec::new(),
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: vec![ContactSiteSpec {
                id: 0,
                name: "scale0".into(),
                attachment: ContactAttachment::RigidBody { body_id: 0 },
                local_pos: Vec3::ZERO,
                local_frame: Quat::IDENTITY,
                radius: 0.02,
                patch_radius: 0.005,
                mode: ContactMode::AnisotropicScale {
                    mu_fwd: 0.15,
                    mu_rev: 0.72,
                    mu_lat: 0.40,
                    pivot_mu: 0.05,
                    static_mu_ratio: 1.0,
                    fwd_axis_local: Vec3::EX,
                },
                material_profile_id: MAT_HARD_FLOOR,
                solref_imp: None,
            }],
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let mut w_rev = FlexGraphWorld::new(
            graph_rev,
            WorldSdf::flat_ground(),
            crate::SolverConfig {
                integrator: crate::IntegratorKind::LinearlyImplicitEuler,
                contact: ContactBackend::Arpc { implicit: true },
                ..crate::SolverConfig::default()
            },
        )
        .unwrap();
        w_rev.state.vels[0] = Vec6::from_parts(Vec3::new(-0.35, 0.0, 0.0), Vec3::ZERO);
        let u_empty: Vec<Vec<f64>> = vec![Vec::new(); 8];
        let tgt_rev: Vec<Vec<Se3>> = (0..=8)
            .map(|_| vec![Se3::new(Vec3::new(0.05, 0.0, 0.02), Quat::IDENTITY)])
            .collect();
        let loss_rev = PoseTrackingLoss::new(tgt_rev, 1000.0, 1.0);
        let p_mu_rev = vec![DiffParamSpec::shared("mu_rev", DiffParamKind::ContactMuRev)];
        let adj_rev =
            compute_trajectory_adjoint(&mut w_rev, &u_empty, dt, &p_mu_rev, &loss_rev).unwrap();
        let v0 = get_diff_param_value(&w_rev, &p_mu_rev[0]);
        let eps = 1e-6;
        set_diff_param_value(&mut w_rev, &p_mu_rev[0], v0 + eps, false);
        let lp = rollout_loss(&mut w_rev, &u_empty, dt, &loss_rev);
        set_diff_param_value(&mut w_rev, &p_mu_rev[0], v0 - eps, false);
        let lm = rollout_loss(&mut w_rev, &u_empty, dt, &loss_rev);
        set_diff_param_value(&mut w_rev, &p_mu_rev[0], v0, false);
        let fd = (lp - lm) / (2.0 * eps);
        let an = adj_rev.grad_params[0];
        let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-8);
        assert!(
            rel < 1e-5 && an.abs() > 1e-6,
            "ContactMuRev IFT vs FD mismatch: an={an:e} fd={fd:e} rel={rel:e}"
        );
    }

    #[test]
    fn test_ift_adjoint_sdf_barrier_beam_interior_contact_stiffness_and_mu_matches_fd() {
        // Regression for Bug 1 (#134): Corotational beam with `interior_node_contact_radius = Some(...)`
        // + body contact site under `ContactBackend::SdfBarrier` (`IpcLogBarrier`).
        let p0 = Vec3::new(0.0, 0.0, 0.021);
        let p1 = Vec3::new(0.12, 0.0, 0.021);
        let mut strip = make_beam_strip(p0, p1, 4, 5.0e9, 0.002);
        // Dip interior centerline nodes close to z = 0 so beam interior SDF barrier is strongly active!
        for i in 1..4 {
            let u = i as f64 / 4.0;
            strip.centerline_rest_poses[i].pos =
                Vec3::new(u * 0.12, 0.0, 0.0155);
            strip.centerline_rest_poses[i].rot = Quat::IDENTITY;
        }
        strip.clamp_a = Se3::IDENTITY;
        strip.clamp_b = Some(Se3::IDENTITY);
        strip.centerline_rest_poses[0] = Se3::new(p0, Quat::IDENTITY);
        strip.centerline_rest_poses[4] = Se3::new(p1, Quat::IDENTITY);
        strip.interior_node_contact_radius = Some(0.010);

        let graph = RigidFlexGraph {
            name: "sdf_barrier_beam_int_adj".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_body(0, p0, false, 0.25), make_body(1, p1, false, 0.25)],
            beams: vec![strip],
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: vec![ContactSiteSpec {
                id: 0,
                name: "pad0".into(),
                attachment: ContactAttachment::RigidBody { body_id: 0 },
                local_pos: Vec3::ZERO,
                local_frame: Quat::IDENTITY,
                radius: 0.015,
                patch_radius: 0.0,
                mode: ContactMode::IsotropicPad {
                    mu: 0.35,
                    pivot_mu: 0.0,
                    static_mu_ratio: 1.0,
                    sdf_only: true,
                },
                material_profile_id: MAT_HARD_FLOOR,
                solref_imp: None,
            }],
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = crate::SolverConfig {
            integrator: crate::IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::SdfBarrier {
                barrier: fg_contact::IpcBarrierConfig {
                    kind: fg_contact::BarrierKind::IpcLogBarrier,
                    stiffness: 500.0,
                    d_hat: 0.012,
                    thickness: 0.002,
                    mollifier_eps_x: 1e-5,
                },
                mu: 0.35,
                friction_eps_v: 0.05,
                use_ccd: false,
            },
            use_strain_jacobian: true,
            world_fixed_inertia: false,
            ..crate::SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
        for v in &mut world.state.vels {
            *v = Vec6::from_parts(Vec3::new(0.40, 0.0, -0.05), Vec3::ZERO);
        }

        let dt = 0.001;
        let n_steps = 8;
        let controls: Vec<Vec<f64>> = vec![Vec::new(); n_steps];
        let targets: Vec<Vec<Se3>> = (0..=n_steps)
            .map(|_| {
                vec![
                    Se3::new(p0 + Vec3::new(0.05, 0.0, 0.01), Quat::IDENTITY),
                    Se3::new(p1 + Vec3::new(0.05, 0.0, 0.01), Quat::IDENTITY),
                ]
            })
            .collect();
        let loss = PoseTrackingLoss::new(targets, 1000.0, 5.0);

        let diff_params = vec![
            DiffParamSpec::shared("k_bar", DiffParamKind::ContactStiffness),
            DiffParamSpec::shared("mu_bar", DiffParamKind::ContactMuFwd),
            DiffParamSpec::targeted("k_bar_tgt", DiffParamKind::ContactStiffness, vec![0]),
            DiffParamSpec::targeted("mu_bar_tgt", DiffParamKind::ContactMuFwd, vec![0]),
        ];
        let adj = compute_trajectory_adjoint(&mut world, &controls, dt, &diff_params, &loss).unwrap();

        for (pi, spec) in diff_params.iter().enumerate() {
            let v0 = get_diff_param_value(&world, spec);
            let eps = 1e-5 * v0.abs().max(1.0);
            set_diff_param_value(&mut world, spec, v0 + eps, false);
            let lp = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, spec, v0 - eps, false);
            let lm = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, spec, v0, false);
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj.grad_params[pi];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-10);
            assert!(
                rel < 1e-5 && an.abs() > 1e-8,
                "SdfBarrier beam-interior param {} mismatch: an={an:e} fd={fd:e} rel={rel:e}",
                spec.name
            );
        }
    }

    #[test]
    fn test_ift_adjoint_body_mass_with_positive_solref_imp_matches_fd() {
        // Regression for Bug 2 (#134): BodyMass IFT VJP with positive-mode SolRefImp on
        // ARPC contact site, non-wheel GpuCoreKelvinParity site, AND positive-mode SolRefImp tendon!
        let sri_contact =
            fg_math::SolRefImp::from_mujoco([0.02, 1.0], [0.9, 0.95, 0.005, 0.5, 2.0]);
        let sri_tendon =
            fg_math::SolRefImp::from_mujoco([0.03, 0.8], [0.9, 0.95, 0.004, 0.5, 2.0]);

        for backend in [
            ContactBackend::Arpc { implicit: true },
            ContactBackend::GpuCoreKelvinParity {
                kn: 5000.0,
                cn: 8.0,
            },
        ] {
            let p0 = Vec3::new(0.0, 0.0, 0.018);
            let p1 = Vec3::new(0.11, 0.0, 0.050);
            let graph = RigidFlexGraph {
                name: "body_mass_solref_adj".into(),
                gravity: Vec3::new(0.0, 0.0, -9.81),
                bodies: vec![make_body(0, p0, false, 0.5), make_body(1, p1, true, 1.0)],
                beams: Vec::new(),
                tendons: vec![TendonEdgeSpec {
                    id: 0,
                    name: "t_sri".into(),
                    actuator_id: 0,
                    waypoints: vec![
                        TendonWaypoint {
                            body_id: 0,
                            local_pos: Vec3::ZERO,
                            wrap_angle_rad: 0.0,
                        },
                        TendonWaypoint {
                            body_id: 1,
                            local_pos: Vec3::ZERO,
                            wrap_angle_rad: 0.0,
                        },
                    ],
                    drive_mode: TendonDriveMode::Linear,
                    rest_length: 0.10,
                    slack: 0.0,
                    stiffness: 1000.0,
                    damping: 5.0,
                    capstan_mu_theta: 0.0,
                    stick_slip_enabled: false,
                    servo_torque_limit: 50.0,
                    servo_speed_limit: 5.0,
                    servo_deadband: 0.0,
                    angle_limits: [0.0, 0.05],
                    solref_imp: Some(sri_tendon),
                }],
                joints: Vec::new(),
                contact_sites: vec![ContactSiteSpec {
                    id: 0,
                    name: "sri_pad".into(),
                    attachment: ContactAttachment::RigidBody { body_id: 0 },
                    local_pos: Vec3::ZERO,
                    local_frame: Quat::IDENTITY,
                    radius: 0.02,
                    patch_radius: 0.005,
                    mode: ContactMode::IsotropicPad {
                        mu: 0.45,
                        pivot_mu: 0.1,
                        static_mu_ratio: 1.0,
                        sdf_only: false,
                    },
                    material_profile_id: MAT_HARD_FLOOR,
                    solref_imp: Some(sri_contact),
                }],
                symmetry: Default::default(),
                solver_hints: Default::default(),
                diff_params: Vec::new(),
                mesh_colliders: Vec::new(),
            };
            let cfg = crate::SolverConfig {
                integrator: crate::IntegratorKind::LinearlyImplicitEuler,
                contact: backend.clone(),
                world_fixed_inertia: false,
                ..crate::SolverConfig::default()
            };
            let mut world = FlexGraphWorld::new(graph, WorldSdf::flat_ground(), cfg).unwrap();
            world.state.vels[0] = Vec6::from_parts(Vec3::new(0.35, 0.0, -0.05), Vec3::ZERO);

            let dt = 0.001;
            let n_steps = 10;
            let controls: Vec<Vec<f64>> = vec![vec![0.005]; n_steps];
            let targets: Vec<Vec<Se3>> = (0..=n_steps)
                .map(|_| vec![Se3::new(Vec3::new(0.05, 0.0, 0.025), Quat::IDENTITY)])
                .collect();
            let loss = PoseTrackingLoss::new(targets, 500.0, 2.0);
            let params = vec![DiffParamSpec::targeted("m0", DiffParamKind::BodyMass, vec![0])];

            let adj = compute_trajectory_adjoint(&mut world, &controls, dt, &params, &loss).unwrap();
            let v0 = get_diff_param_value(&world, &params[0]);
            let eps = 1e-6 * v0;
            set_diff_param_value(&mut world, &params[0], v0 + eps, false);
            let lp = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, &params[0], v0 - eps, false);
            let lm = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, &params[0], v0, false);
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj.grad_params[0];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-8);
            assert!(
                rel < 1e-5 && an.abs() > 1e-5,
                "BodyMass SolRefImp ({backend:?}) mismatch: an={an:e} fd={fd:e} rel={rel:e}"
            );
        }
    }

    #[test]
    fn test_ift_adjoint_der_rod_youngs_beta_controls_and_init_state_matches_fd() {
        // Regression for Bug 3 (#134): Full DER IFT adjoint across multi-section SoftOctopusArm-DER
        // AND free-base cantilever DER rod (with non-zero clamp offset, twist, & Bishop frame transport).

        // --- Case A: 2-section tapered DER arm (SoftOctopusArm-DER style: fixed base + 2 rings + tendon) ---
        let n_el = 4;
        let sec_len = 0.06;
        let dl = sec_len / n_el as f64;
        let z0 = 0.20;
        let make_der_sec = |id: usize, ba: usize, bb: Option<usize>, x0: f64, radius: f64| {
            let poses: Vec<Se3> = (0..=n_el)
                .map(|j| Se3::new(Vec3::new(x0 + j as f64 * dl, 0.0, z0), Quat::IDENTITY))
                .collect();
            BeamEdgeSpec {
                id,
                name: format!("der_sec_{id}"),
                body_a: ba,
                clamp_a: Se3::IDENTITY,
                body_b: bb,
                clamp_b: bb.map(|_| Se3::IDENTITY),
                n_elements: n_el,
                formulation: BeamFormulation::DiscreteElasticRod,
                section: BeamSection::Circular { radius },
                material: BeamMaterial {
                    youngs_modulus: 6.0e6,
                    poisson_ratio: 0.45,
                    density: 1120.0,
                    rayleigh_alpha: 0.15,
                    rayleigh_beta: 0.004,
                },
                thin_shell_compat: false,
                prestress_frac: 0.0,
                centerline_rest_poses: poses,
                element_lengths: vec![dl; n_el],
                element_transforms: None,
                interior_node_contact_radius: None,
                enable_self_contact: false,
            }
        };
        let b0 = make_body(0, Vec3::new(0.0, 0.0, z0), true, 0.12);
        let b1 = make_body(1, Vec3::new(sec_len, 0.0, z0), false, 0.012);
        let b2 = make_body(2, Vec3::new(2.0 * sec_len, 0.0, z0), false, 0.008);
        let tendon = TendonEdgeSpec {
            id: 0,
            name: "muscle_z".into(),
            actuator_id: 0,
            waypoints: vec![
                TendonWaypoint {
                    body_id: 0,
                    local_pos: Vec3::new(0.0, 0.006, 0.008),
                    wrap_angle_rad: 0.0,
                },
                TendonWaypoint {
                    body_id: 2,
                    local_pos: Vec3::new(0.0, 0.004, 0.006),
                    wrap_angle_rad: 0.0,
                },
            ],
            drive_mode: TendonDriveMode::Linear,
            rest_length: 2.0 * sec_len,
            slack: 0.0,
            stiffness: 400.0,
            damping: 0.4,
            capstan_mu_theta: 0.0,
            stick_slip_enabled: false,
            servo_torque_limit: 20.0,
            servo_speed_limit: 2.0,
            servo_deadband: 0.0,
            angle_limits: [0.0, 0.04],
            solref_imp: None,
        };
        let graph_arm = RigidFlexGraph {
            name: "der_octopus_arm_adj".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![b0, b1, b2],
            beams: vec![
                make_der_sec(0, 0, Some(1), 0.0, 0.010),
                make_der_sec(1, 1, Some(2), sec_len, 0.007),
            ],
            tendons: vec![tendon],
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let cfg = crate::SolverConfig {
            integrator: crate::IntegratorKind::LinearlyImplicitEuler,
            contact: ContactBackend::None,
            world_fixed_inertia: false,
            project_der_hessian_spd: true,
            ..crate::SolverConfig::default()
        };
        let mut world = FlexGraphWorld::new(graph_arm, WorldSdf::flat_ground(), cfg.clone()).unwrap();
        // Seed 3D out-of-plane velocity + twist so stretching, bending, twisting, and Bishop transport are all active!
        world.state.vels[1] = Vec6::from_parts(Vec3::new(0.05, 0.18, -0.12), Vec3::new(1.5, -0.8, 0.6));
        world.state.vels[2] = Vec6::from_parts(Vec3::new(-0.04, -0.15, -0.20), Vec3::new(-1.2, 0.9, -0.7));
        if let Some(ref mut ds) = world.state.der_states[0] {
            ds.theta[1] = 0.15;
            ds.theta_vel[2] = 0.8;
        }

        let dt = 0.001;
        let n_steps = 10;
        let controls: Vec<Vec<f64>> = (0..n_steps)
            .map(|k| vec![0.004 + 0.001 * k as f64])
            .collect();
        let mid_free = world.analysis.indexing.beam_maps[1].free_node_ids[1];
        let targets: Vec<Vec<Se3>> = (0..=n_steps)
            .map(|_| {
                vec![
                    Se3::new(Vec3::new(sec_len, 0.01, z0 + 0.01), Quat::from_axis_angle(Vec3::EX, 0.1)),
                    Se3::new(Vec3::new(2.0 * sec_len, -0.01, z0 + 0.02), Quat::from_axis_angle(Vec3::EY, -0.12)),
                    Se3::new(Vec3::new(1.5 * sec_len, 0.005, z0 - 0.01), Quat::from_axis_angle(Vec3::EZ, 0.08)),
                ]
            })
            .collect();
        let mut loss = PoseTrackingLoss::new(targets, 200.0, 5.0);
        loss.tracked_nodes = vec![1, 2, mid_free];

        let diff_params = vec![
            DiffParamSpec::shared("E", DiffParamKind::BeamYoungsModulus),
            DiffParamSpec::shared("beta", DiffParamKind::BeamDampingBeta),
            DiffParamSpec::targeted("m2", DiffParamKind::BodyMass, vec![2]),
        ];
        let adj = compute_trajectory_adjoint(&mut world, &controls, dt, &diff_params, &loss).unwrap();

        for (pi, spec) in diff_params.iter().enumerate() {
            let v0 = get_diff_param_value(&world, spec);
            let eps = 1e-6 * v0.abs().max(1e-4);
            set_diff_param_value(&mut world, spec, v0 + eps, false);
            let lp = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, spec, v0 - eps, false);
            let lm = rollout_loss(&mut world, &controls, dt, &loss);
            set_diff_param_value(&mut world, spec, v0, false);
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj.grad_params[pi];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-10);
            assert!(
                rel < 1e-5 && an.abs() > 1e-12,
                "DER arm param {} IFT vs FD mismatch: an={an:e} fd={fd:e} rel={rel:e}",
                spec.name
            );
        }

        // Control gradient check on DER arm
        for &step_chk in &[0usize, 4, n_steps - 1] {
            let eps = 1e-6;
            let mut u_p = controls.clone();
            let mut u_m = controls.clone();
            u_p[step_chk][0] += eps;
            u_m[step_chk][0] -= eps;
            let lp = rollout_loss(&mut world, &u_p, dt, &loss);
            let lm = rollout_loss(&mut world, &u_m, dt, &loss);
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj.grad_controls[step_chk][0];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-10);
            assert!(
                rel < 1e-5,
                "DER arm control u[{step_chk}][0] mismatch: an={an:e} fd={fd:e} rel={rel:e}"
            );
        }

        // Advance 2 steps off the exact zero-strain rest kink (`max(k_perp, 0)` in SPD projection)
        // to verify initial pose & velocity cotangents on a smooth 3D flexed/twisted DER state!
        world.step_n(2, dt).unwrap();
        let adj_mid =
            compute_trajectory_adjoint(&mut world, &controls, dt, &diff_params, &loss).unwrap();
        for &nid in &[1usize, mid_free] {
            let n_dofs = if nid < 3 { 6 } else { 3 };
            for d in 0..n_dofs {
                let eps_p = 1e-7;
                let orig_p = world.state.poses[nid];
                let mut tw = Vec6::ZERO;
                tw[d] = 1.0;
                world.state.poses[nid] = orig_p.integrate_decoupled(tw, eps_p);
                let lp = rollout_loss(&mut world, &controls, dt, &loss);
                world.state.poses[nid] = orig_p.integrate_decoupled(tw, -eps_p);
                let lm = rollout_loss(&mut world, &controls, dt, &loss);
                world.state.poses[nid] = orig_p;
                let fd = (lp - lm) / (2.0 * eps_p);
                let an = adj_mid.grad_init_poses[nid][d];
                let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-8);
                assert!(
                    rel < 1e-5,
                    "DER init pose[{nid}][{d}] mismatch: an={an:e} fd={fd:e} rel={rel:e}"
                );

                let eps_v = 1e-5;
                let orig_v = world.state.vels[nid];
                world.state.vels[nid][d] = orig_v[d] + eps_v;
                let vp = rollout_loss(&mut world, &controls, dt, &loss);
                world.state.vels[nid][d] = orig_v[d] - eps_v;
                let vm = rollout_loss(&mut world, &controls, dt, &loss);
                world.state.vels[nid] = orig_v;
                let fd_v = (vp - vm) / (2.0 * eps_v);
                let an_v = adj_mid.grad_init_vels[nid][d];
                let rel_v = (an_v - fd_v).abs() / fd_v.abs().max(an_v.abs()).max(1e-8);
                assert!(
                    rel_v < 1e-5,
                    "DER init vel[{nid}][{d}] mismatch: an={an_v:e} fd={fd_v:e} rel={rel_v:e}"
                );
            }
        }

        // --- Case B: Free-base DER cantilever with non-zero clamp offset (tests clamp_a inertia + free tip) ---
        let clamp_off = Vec3::new(0.015, -0.010, 0.005);
        let poses_cant: Vec<Se3> = (0..=n_el)
            .map(|j| {
                Se3::new(
                    Vec3::new(0.0, 0.0, z0) + clamp_off + Vec3::new(j as f64 * dl, 0.0, 0.0),
                    Quat::IDENTITY,
                )
            })
            .collect();
        let cant_beam = BeamEdgeSpec {
            id: 0,
            name: "der_cant".into(),
            body_a: 0,
            clamp_a: Se3::new(clamp_off, Quat::IDENTITY),
            body_b: None,
            clamp_b: None,
            n_elements: n_el,
            formulation: BeamFormulation::DiscreteElasticRod,
            section: BeamSection::Circular { radius: 0.008 },
            material: BeamMaterial {
                youngs_modulus: 8.0e6,
                poisson_ratio: 0.40,
                density: 1200.0,
                rayleigh_alpha: 0.1,
                rayleigh_beta: 0.003,
            },
            thin_shell_compat: false,
            prestress_frac: 0.0,
            centerline_rest_poses: poses_cant,
            element_lengths: vec![dl; n_el],
            element_transforms: None,
            interior_node_contact_radius: None,
            enable_self_contact: false,
        };
        let graph_cant = RigidFlexGraph {
            name: "der_cant_adj".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![make_body(0, Vec3::new(0.0, 0.0, z0), false, 0.08)],
            beams: vec![cant_beam],
            tendons: Vec::new(),
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        let mut w_cant = FlexGraphWorld::new(graph_cant, WorldSdf::flat_ground(), cfg).unwrap();
        w_cant.state.vels[0] = Vec6::from_parts(Vec3::new(0.1, -0.2, 0.05), Vec3::new(1.2, -1.5, 0.9));
        // Advance 2 warm-up steps off exact zero-strain rest kink for smooth init-pose FD
        w_cant.step_n(2, dt).unwrap();
        let tip_nid = *w_cant.analysis.indexing.beam_maps[0]
            .free_node_ids
            .last()
            .unwrap();
        let u_empty: Vec<Vec<f64>> = vec![Vec::new(); 8];
        let tgt_cant: Vec<Vec<Se3>> = (0..=8)
            .map(|_| {
                vec![
                    Se3::new(Vec3::new(0.01, -0.01, z0 + 0.02), Quat::from_axis_angle(Vec3::EY, 0.15)),
                    Se3::new(Vec3::new(sec_len, 0.02, z0 - 0.02), Quat::from_axis_angle(Vec3::EX, -0.1)),
                ]
            })
            .collect();
        let mut loss_cant = PoseTrackingLoss::new(tgt_cant, 150.0, 4.0);
        loss_cant.tracked_nodes = vec![0, tip_nid];
        let cant_params = vec![
            DiffParamSpec::shared("E_cant", DiffParamKind::BeamYoungsModulus),
            DiffParamSpec::shared("beta_cant", DiffParamKind::BeamDampingBeta),
        ];
        let adj_cant =
            compute_trajectory_adjoint(&mut w_cant, &u_empty, dt, &cant_params, &loss_cant)
                .unwrap();

        for (pi, spec) in cant_params.iter().enumerate() {
            let v0 = get_diff_param_value(&w_cant, spec);
            let eps = 1e-6 * v0.abs().max(1e-4);
            set_diff_param_value(&mut w_cant, spec, v0 + eps, false);
            let lp = rollout_loss(&mut w_cant, &u_empty, dt, &loss_cant);
            set_diff_param_value(&mut w_cant, spec, v0 - eps, false);
            let lm = rollout_loss(&mut w_cant, &u_empty, dt, &loss_cant);
            set_diff_param_value(&mut w_cant, spec, v0, false);
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj_cant.grad_params[pi];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-10);
            assert!(
                rel < 1e-5 && an.abs() > 1e-12,
                "DER cantilever param {} mismatch: an={an:e} fd={fd:e} rel={rel:e}",
                spec.name
            );
        }
        for d in 0..6 {
            let eps = 1e-7;
            let orig_p = w_cant.state.poses[0];
            let mut tw = Vec6::ZERO;
            tw[d] = 1.0;
            w_cant.state.poses[0] = orig_p.integrate_decoupled(tw, eps);
            let lp = rollout_loss(&mut w_cant, &u_empty, dt, &loss_cant);
            w_cant.state.poses[0] = orig_p.integrate_decoupled(tw, -eps);
            let lm = rollout_loss(&mut w_cant, &u_empty, dt, &loss_cant);
            w_cant.state.poses[0] = orig_p;
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj_cant.grad_init_poses[0][d];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-8);
            assert!(
                rel < 1e-5,
                "DER cantilever init pose[0][{d}] mismatch: an={an:e} fd={fd:e} rel={rel:e}"
            );
        }

        // --- Case C: SoftOctopusArm-DER style 2-section DER arm with 3-waypoint stick-slip Capstan tendon ---
        let mut graph_ss = RigidFlexGraph {
            name: "der_octopus_stick_slip".into(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: vec![
                make_body(0, Vec3::new(0.0, 0.0, z0), true, 0.12),
                make_body(1, Vec3::new(sec_len, 0.0, z0), false, 0.012),
                make_body(2, Vec3::new(2.0 * sec_len, 0.0, z0), false, 0.008),
            ],
            beams: vec![
                make_der_sec(0, 0, Some(1), 0.0, 0.010),
                make_der_sec(1, 1, Some(2), sec_len, 0.007),
            ],
            tendons: vec![TendonEdgeSpec {
                id: 0,
                name: "muscle_ss".into(),
                actuator_id: 0,
                waypoints: vec![
                    TendonWaypoint {
                        body_id: 0,
                        local_pos: Vec3::new(0.0, 0.0, 0.010),
                        wrap_angle_rad: 0.0,
                    },
                    TendonWaypoint {
                        body_id: 1,
                        local_pos: Vec3::new(0.0, 0.0, 0.008),
                        wrap_angle_rad: 0.06,
                    },
                    TendonWaypoint {
                        body_id: 2,
                        local_pos: Vec3::new(0.0, 0.0, 0.006),
                        wrap_angle_rad: 0.0,
                    },
                ],
                drive_mode: TendonDriveMode::Linear,
                rest_length: 0.0,
                slack: 0.0,
                stiffness: 600.0,
                damping: 0.8,
                capstan_mu_theta: 0.10,
                stick_slip_enabled: true,
                servo_torque_limit: 5.0,
                servo_speed_limit: 0.5,
                servo_deadband: 0.0,
                angle_limits: [0.0, 0.04],
                solref_imp: None,
            }],
            joints: Vec::new(),
            contact_sites: Vec::new(),
            symmetry: Default::default(),
            solver_hints: Default::default(),
            diff_params: Vec::new(),
            mesh_colliders: Vec::new(),
        };
        // Slight initial pre-tension via rest_length slightly shorter than geometric length
        graph_ss.tendons[0].rest_length = 2.0 * sec_len - 0.002;
        let mut w_ss = FlexGraphWorld::new(
            graph_ss,
            WorldSdf::flat_ground(),
            crate::SolverConfig {
                integrator: crate::IntegratorKind::LinearlyImplicitEuler,
                contact: ContactBackend::None,
                world_fixed_inertia: false,
                project_der_hessian_spd: true,
                ..crate::SolverConfig::default()
            },
        )
        .unwrap();
        let u_ss: Vec<Vec<f64>> = (0..8).map(|k| vec![0.003 + 0.0005 * k as f64]).collect();
        let tgt_ss: Vec<Vec<Se3>> = (0..=8)
            .map(|_| {
                vec![
                    Se3::new(Vec3::new(sec_len, 0.0, z0 + 0.01), Quat::IDENTITY),
                    Se3::new(Vec3::new(2.0 * sec_len, 0.0, z0 + 0.02), Quat::IDENTITY),
                ]
            })
            .collect();
        let mut loss_ss = PoseTrackingLoss::new(tgt_ss, 200.0, 5.0);
        loss_ss.tracked_nodes = vec![1, 2];
        let ss_params = vec![
            DiffParamSpec::shared("E_ss", DiffParamKind::BeamYoungsModulus),
            DiffParamSpec::shared("beta_ss", DiffParamKind::BeamDampingBeta),
            DiffParamSpec::shared("kten_ss", DiffParamKind::TendonStiffness),
        ];
        let adj_ss =
            compute_trajectory_adjoint(&mut w_ss, &u_ss, dt, &ss_params, &loss_ss).unwrap();
        for (pi, spec) in ss_params.iter().enumerate() {
            let v0 = get_diff_param_value(&w_ss, spec);
            let eps = 1e-5 * v0.abs().max(1e-4);
            set_diff_param_value(&mut w_ss, spec, v0 + eps, false);
            let lp = rollout_loss(&mut w_ss, &u_ss, dt, &loss_ss);
            set_diff_param_value(&mut w_ss, spec, v0 - eps, false);
            let lm = rollout_loss(&mut w_ss, &u_ss, dt, &loss_ss);
            set_diff_param_value(&mut w_ss, spec, v0, false);
            let fd = (lp - lm) / (2.0 * eps);
            let an = adj_ss.grad_params[pi];
            let rel = (an - fd).abs() / fd.abs().max(an.abs()).max(1e-10);
            assert!(
                rel < 1e-5 && an.abs() > 1e-12,
                "DER stick-slip param {} mismatch: an={an:e} fd={fd:e} rel={rel:e}",
                spec.name
            );
        }
    }
}
