//! Rich Actuation Mechanics for `flexgraph-rs`:
//! 1. **Multi-Waypoint Capstan Tendons** (`EccentricCrank`, `LegacyWrap`, `Capstan`, `Linear`):
//!    - Exact shared-servo load summation, deadband, speed limit, and DC torque-speed derating
//!      matching `serpworm.physics.gpu_core` to `< 1e-12` FP64.
//!    - Multi-waypoint routing across arbitrary rigid bodies with exact **Stick-Slip Capstan
//!      Hysteresis** (friction-cone proximal projection `[T_up * exp(-mu*theta), T_up * exp(+mu*theta)]`)
//!      capturing pull/release tension hysteresis and exact zero-resultant internal forces.
//! 2. **Articulated & Compliant Joints** (`Revolute`, `Prismatic`, `Spherical`, `Weld`,
//!    `AxialStopGap`, `CableSpring`):
//!    - Exact Stribeck static-to-kinetic friction, deadband, joint limits, and DC motor
//!      torque-speed envelopes.
//!    - Exact analytical `12x12` tangent blocks (SPD-projected when requested) for implicit
//!      Schur coupling (`joint.implicit_coupling = true`, e.g., cyclic `Tensegrity-6Bar` cables!).

use fg_ir::{JointEdgeSpec, JointType, TendonDriveMode, TendonEdgeSpec};
use fg_math::{Mat3, Mat6, Se3, SolRefImp, Vec3, Vec6};
use serde::{Deserialize, Serialize};

/// Compute reduced effective translational mass for a 2-body constraint (or unit mass when
/// masses are omitted or both bodies are fixed).
#[inline]
fn reduced_pair_mass(ba: usize, bb: usize, body_masses: Option<&[(f64, bool)]>) -> f64 {
    let Some(masses) = body_masses else {
        return 1.0;
    };
    let (ma, fa) = masses.get(ba).copied().unwrap_or((1.0, false));
    let (mb, fb) = masses.get(bb).copied().unwrap_or((1.0, false));
    if ba == bb {
        return if fa { 1.0 } else { ma.max(1e-9) };
    }
    match (fa, fb) {
        (false, false) => (ma * mb / (ma + mb).max(1e-12)).max(1e-9),
        (true, false) => mb.max(1e-9),
        (false, true) => ma.max(1e-9),
        (true, true) => 1.0,
    }
}

/// Resolve effective `(stiffness, damping)` honoring optional MuJoCo `SolRefImp`.
///
/// - When `solref_imp` is `None`, returns `(base_k, base_c)` untouched for exact v1 parity.
/// - When `solref_imp` is `Some(sri)`, delegates directly to
///   [`SolRefImp::to_force_stiffness_damping`] so both direct negative `(-k, -b)` mode and
///   positive `(timeconst, dampratio)` mode (scaled by reduced body mass `eff_mass` and clamped
///   by `2 * dt`) accurately govern the soft-constraint dynamics.
#[inline]
fn resolve_solref_kc(
    solref_imp: Option<SolRefImp>,
    violation: f64,
    base_k: f64,
    base_c: f64,
    eff_mass: f64,
    dt: f64,
) -> (f64, f64) {
    let Some(sri) = solref_imp else {
        return (base_k, base_c);
    };
    sri.to_force_stiffness_damping(violation, eff_mass, dt)
}

/// Per-tendon persistent state (tension + per-segment stick-slip Capstan memory).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TendonRuntimeState {
    /// Proximal (actuator-side) cable tension (N).
    pub tension: f64,
    /// Per-segment tension along multi-waypoint spans (len `waypoints.len() - 1`).
    pub segment_tensions: Vec<f64>,
    /// Previous total geometric path length (for incremental stick-slip elastic update).
    pub prev_segment_lengths: Vec<f64>,
}

/// Persistent actuation state across all servos, tendons, and joint actuators.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ActuationState {
    /// Current tendon servo positions (rad or m), indexed by `tendon.actuator_id`.
    pub tendon_servo_pos: Vec<f64>,
    /// Commanded tendon servo targets, indexed by `tendon.actuator_id`.
    pub tendon_servo_target: Vec<f64>,
    /// Last-step tendon servo velocities.
    pub tendon_servo_vel: Vec<f64>,
    /// Per-tendon runtime tension & stick-slip Capstan history.
    pub tendons: Vec<TendonRuntimeState>,
    /// Commanded articulated joint actuator targets, indexed by `joint.actuator_id`.
    pub joint_targets: Vec<f64>,
    /// Last-step measured joint actuator torque/force (for telemetry/RL observations).
    pub last_joint_torque: Vec<f64>,
    /// Last-step DC torque-speed cap on each joint.
    pub last_joint_torque_cap: Vec<f64>,
}

impl ActuationState {
    /// Initialize actuation state from graph tendon & joint specs and rest poses.
    pub fn from_specs(
        tendons: &[TendonEdgeSpec],
        joints: &[JointEdgeSpec],
        rest_poses: &[Se3],
    ) -> Self {
        let n_tendon_act = tendons
            .iter()
            .map(|t| t.actuator_id + 1)
            .max()
            .unwrap_or(0);
        let n_joint_act = joints
            .iter()
            .filter_map(|j| j.actuator_id.map(|id| id + 1))
            .max()
            .unwrap_or(0);

        let tendon_states = tendons
            .iter()
            .map(|t| {
                let n_seg = t.waypoints.len().saturating_sub(1);
                let mut seg_lens = Vec::with_capacity(n_seg);
                for k in 0..n_seg {
                    let wp0 = t.waypoints[k];
                    let wp1 = t.waypoints[k + 1];
                    let p0 = rest_poses[wp0.body_id].transform_point(wp0.local_pos);
                    let p1 = rest_poses[wp1.body_id].transform_point(wp1.local_pos);
                    seg_lens.push((p1 - p0).norm());
                }
                TendonRuntimeState {
                    tension: 0.0,
                    segment_tensions: vec![0.0; n_seg],
                    prev_segment_lengths: seg_lens,
                }
            })
            .collect();

        Self {
            tendon_servo_pos: vec![0.0; n_tendon_act],
            tendon_servo_target: vec![0.0; n_tendon_act],
            tendon_servo_vel: vec![0.0; n_tendon_act],
            tendons: tendon_states,
            joint_targets: vec![0.0; n_joint_act],
            last_joint_torque: vec![0.0; joints.len()],
            last_joint_torque_cap: joints.iter().map(|j| j.torque_limit).collect(),
        }
    }
}

/// Output of tendon & joint evaluation for one substep / Newton residual evaluation.
#[derive(Clone, Debug)]
pub struct ActuationEval {
    /// Potential energy stored in tendons and joints.
    pub potential_energy: f64,
    /// Generalized wrenches added to rigid bodies `0..num_bodies`.
    pub body_wrenches: Vec<Vec6>,
    /// Implicit `6x6` tangent stiffness/damping blocks `(body_u, body_v, K_block)` for
    /// joints with `implicit_coupling == true` (or implicit Newton linearization).
    pub implicit_blocks: Vec<(usize, usize, Mat6)>,
}

/// Compute instantaneous lever arm (d takeup / d servo) for a tendon drive mode.
pub fn tendon_lever_arm(mode: TendonDriveMode, servo: f64) -> f64 {
    match mode {
        TendonDriveMode::EccentricCrank {
            horn_radius,
            guide_dist,
        } => {
            let chord = (guide_dist * guide_dist + horn_radius * horn_radius
                - 2.0 * guide_dist * horn_radius * servo.cos())
            .max(1e-18)
            .sqrt()
            .max(1e-9);
            (horn_radius * guide_dist * servo.sin().abs() / chord).min(horn_radius)
        }
        TendonDriveMode::LegacyWrap {
            horn_radius,
            guide_dist,
        } => {
            let chord = (guide_dist * guide_dist + horn_radius * horn_radius
                - 2.0 * guide_dist * horn_radius * servo.cos())
            .max(1e-18)
            .sqrt()
            .max(1e-9);
            let alpha = (horn_radius / guide_dist).clamp(-1.0, 1.0).acos();
            if servo <= alpha {
                horn_radius * guide_dist * servo.sin() / chord
            } else {
                horn_radius
            }
        }
        TendonDriveMode::Capstan { drum_radius } => drum_radius,
        TendonDriveMode::Linear => 1.0,
    }
}

/// Compute cable retraction length `takeup(servo)` for a tendon drive mode.
pub fn tendon_takeup(mode: TendonDriveMode, servo: f64) -> f64 {
    match mode {
        TendonDriveMode::EccentricCrank {
            horn_radius,
            guide_dist,
        } => {
            let chord = (guide_dist * guide_dist + horn_radius * horn_radius
                - 2.0 * guide_dist * horn_radius * servo.cos())
            .max(0.0)
            .sqrt();
            chord - (guide_dist - horn_radius)
        }
        TendonDriveMode::LegacyWrap {
            horn_radius,
            guide_dist,
        } => {
            let chord = (guide_dist * guide_dist + horn_radius * horn_radius
                - 2.0 * guide_dist * horn_radius * servo.cos())
            .max(0.0)
            .sqrt();
            let alpha = (horn_radius / guide_dist).clamp(-1.0, 1.0).acos();
            let raw = if servo <= alpha {
                chord
            } else {
                (guide_dist * guide_dist - horn_radius * horn_radius)
                    .max(0.0)
                    .sqrt()
                    + horn_radius * (servo - alpha)
            };
            raw - (guide_dist - horn_radius)
        }
        TendonDriveMode::Capstan { drum_radius } => drum_radius * servo,
        TendonDriveMode::Linear => servo,
    }
}

/// Advance tendon servo positions once per time step (exact DC motor torque-speed derating
/// and deadband matching `gpu_core.py:501-516`).
pub fn advance_tendon_servos(
    tendons: &[TendonEdgeSpec],
    act: &mut ActuationState,
    dt: f64,
) {
    let n_act = act.tendon_servo_pos.len();
    if n_act == 0 {
        return;
    }
    let mut load = vec![0.0; n_act];
    let mut speed_lim = vec![0.0; n_act];
    let mut torque_lim = vec![0.0; n_act];
    let mut deadband = vec![0.0; n_act];
    let mut limits = vec![[f64::NEG_INFINITY, f64::INFINITY]; n_act];

    for (ti, spec) in tendons.iter().enumerate() {
        let aid = spec.actuator_id;
        let servo = act.tendon_servo_pos[aid];
        let lever = tendon_lever_arm(spec.drive_mode, servo);
        let cap_mt = spec.capstan_mu_theta.clamp(0.0, 0.6);
        load[aid] += act.tendons[ti].tension * lever * cap_mt.exp();
        speed_lim[aid] = spec.servo_speed_limit;
        torque_lim[aid] = spec.servo_torque_limit;
        deadband[aid] = spec.servo_deadband.max(0.0);
        if spec.angle_limits[0] < spec.angle_limits[1] {
            limits[aid] = spec.angle_limits;
        }
    }

    for aid in 0..n_act {
        let s_max = speed_lim[aid] * dt;
        let raw_err = act.tendon_servo_target[aid] - act.tendon_servo_pos[aid];
        let eff_err = raw_err.signum() * (raw_err.abs() - deadband[aid]).max(0.0);
        let mut inc = eff_err.clamp(-s_max, s_max);
        if inc > 0.0 && torque_lim[aid] > 0.0 {
            inc *= (1.0 - load[aid] / torque_lim[aid]).clamp(0.0, 1.0);
        }
        let next_pos = (act.tendon_servo_pos[aid] + inc).clamp(limits[aid][0], limits[aid][1]);
        let actual_inc = next_pos - act.tendon_servo_pos[aid];
        act.tendon_servo_vel[aid] = actual_inc / dt;
        act.tendon_servo_pos[aid] = next_pos;
    }
}

/// Evaluate all tendons and articulated/compliant joints at `(poses, vels)`.
///
/// - When `commit_state == true`, updates `act.tendons[i]` stick-slip memory and
///   `act.last_joint_torque`. Set `commit_state = false` during Newton line-search probes.
/// - When `implicit_scale_k > 0.0` (e.g., `h^2` in velocity form or `1.0` in Newton position form)
///   and `implicit_scale_c > 0.0` (e.g., `h` in velocity form or `c_v` in Newton form),
///   populates `implicit_blocks` for every joint with `implicit_coupling == true` (or all joints
///   if `assemble_all_joint_tangents == true`).
/// - For positive-mode MuJoCo [`SolRefImp`] (`timeconst > 0, dampratio > 0`), uses unit effective
///   mass (`1.0 kg`) and `dt = 0.0`; call [`evaluate_actuation_with_masses`] to pass exact body
///   masses and timestep `dt`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_actuation(
    num_bodies: usize,
    tendons: &[TendonEdgeSpec],
    joints: &[JointEdgeSpec],
    rest_poses: &[Se3],
    poses: &[Se3],
    vels: &[Vec6],
    act: &mut ActuationState,
    implicit_scale_k: f64,
    implicit_scale_c: f64,
    assemble_all_joint_tangents: bool,
    commit_state: bool,
    stribeck_enabled: bool,
) -> ActuationEval {
    evaluate_actuation_with_masses(
        num_bodies,
        tendons,
        joints,
        rest_poses,
        poses,
        vels,
        act,
        implicit_scale_k,
        implicit_scale_c,
        assemble_all_joint_tangents,
        commit_state,
        stribeck_enabled,
        None,
        0.0,
    )
}

/// Evaluate all tendons and articulated/compliant joints at `(poses, vels)` with optional
/// per-body `(mass, is_fixed)` metadata and timestep `dt` for MuJoCo positive-mode [`SolRefImp`]
/// reduced-mass soft-constraint scaling.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_actuation_with_masses(
    num_bodies: usize,
    tendons: &[TendonEdgeSpec],
    joints: &[JointEdgeSpec],
    rest_poses: &[Se3],
    poses: &[Se3],
    vels: &[Vec6],
    act: &mut ActuationState,
    implicit_scale_k: f64,
    implicit_scale_c: f64,
    assemble_all_joint_tangents: bool,
    commit_state: bool,
    stribeck_enabled: bool,
    body_masses: Option<&[(f64, bool)]>,
    dt: f64,
) -> ActuationEval {
    let mut body_wrenches = vec![Vec6::ZERO; num_bodies];
    let mut implicit_blocks = Vec::new();
    let mut potential_energy = 0.0;

    // --- 1. Evaluate Tendons ---
    for (ti, spec) in tendons.iter().enumerate() {
        let aid = spec.actuator_id;
        let servo = act.tendon_servo_pos[aid];
        let sv_vel = act.tendon_servo_vel[aid];
        let takeup = tendon_takeup(spec.drive_mode, servo);

        let n_wp = spec.waypoints.len();
        let n_seg = n_wp - 1;

        // Compute rest length if <= 0.0
        let base_rest_len = if spec.rest_length > 0.0 {
            spec.rest_length
        } else {
            let mut sum = 0.0;
            for k in 0..n_seg {
                let wp0 = spec.waypoints[k];
                let wp1 = spec.waypoints[k + 1];
                let p0 = rest_poses[wp0.body_id].transform_point(wp0.local_pos);
                let p1 = rest_poses[wp1.body_id].transform_point(wp1.local_pos);
                sum += (p1 - p0).norm();
            }
            sum
        };

        // Fast exact GPUCore parity path for 2-waypoint non-stick-slip tendons
        if n_wp == 2 && !spec.stick_slip_enabled {
            let wp0 = spec.waypoints[0];
            let wp1 = spec.waypoints[1];
            let xa = poses[wp0.body_id];
            let xb = poses[wp1.body_id];
            let ar = xa.rot.rotate(wp0.local_pos);
            let br = xb.rot.rotate(wp1.local_pos);
            let delta = (xb.pos + br) - (xa.pos + ar);
            let length = delta.norm();
            let dir = delta / length.max(1e-15);

            let va = vels[wp0.body_id].lin() + vels[wp0.body_id].ang().cross(ar);
            let vb = vels[wp1.body_id].lin() + vels[wp1.body_id].ang().cross(br);
            let rest = (base_rest_len + spec.slack) - takeup;
            let extension = length - rest;
            let speed = (vb - va).dot(dir);

            let m_eff = reduced_pair_mass(wp0.body_id, wp1.body_id, body_masses);
            let (k_ten, c_ten) = resolve_solref_kc(
                spec.solref_imp,
                extension.max(0.0),
                spec.stiffness,
                spec.damping,
                m_eff,
                dt,
            );
            let cap_mt = spec.capstan_mu_theta.clamp(0.0, 0.6);
            let c_eff = c_ten + 12.0 * cap_mt;
            let ten_raw = if extension >= 0.0 {
                (k_ten * extension + c_eff * speed).max(0.0)
            } else {
                0.0
            };
            let pull_frac = (sv_vel.max(0.0) / 0.35).tanh();
            let tension = ten_raw * (-cap_mt * pull_frac).exp();
            if extension > 0.0 {
                potential_energy += 0.5 * k_ten * extension * extension;
            }
            if commit_state {
                act.tendons[ti].tension = tension;
                if let Some(s0) = act.tendons[ti].segment_tensions.get_mut(0) {
                    *s0 = tension;
                }
                if let Some(l0) = act.tendons[ti].prev_segment_lengths.get_mut(0) {
                    *l0 = length;
                }
            }
            let force = tension * dir;
            body_wrenches[wp0.body_id] += Vec6::from_parts(force, ar.cross(force));
            body_wrenches[wp1.body_id] -= Vec6::from_parts(force, br.cross(force));
            if assemble_all_joint_tangents
                && extension >= 0.0
                && (implicit_scale_k > 0.0 || implicit_scale_c > 0.0)
            {
                let dd = dir.outer(dir);
                let att = (-cap_mt * pull_frac).exp();
                let k_par = (implicit_scale_k * k_ten + implicit_scale_c * c_eff) * att;
                let k_perp = implicit_scale_k * (tension / length.max(1e-15)).max(0.0);
                let k3 = k_par * dd + k_perp * (Mat3::IDENTITY - dd);
                push_linear_pair_blocks(
                    &mut implicit_blocks,
                    wp0.body_id,
                    wp1.body_id,
                    &point_wrench_mat6(ar),
                    &point_wrench_mat6(br),
                    k3,
                );
            }
            continue;
        }

        // General multi-waypoint path with optional Stick-Slip Capstan Hysteresis
        let mut r_world = Vec::with_capacity(n_wp);
        let mut p_world = Vec::with_capacity(n_wp);
        let mut v_world = Vec::with_capacity(n_wp);
        for wp in &spec.waypoints {
            let x = poses[wp.body_id];
            let v = vels[wp.body_id];
            let rw = x.rot.rotate(wp.local_pos);
            r_world.push(rw);
            p_world.push(x.pos + rw);
            v_world.push(v.lin() + v.ang().cross(rw));
        }

        let mut seg_lens = Vec::with_capacity(n_seg);
        let mut seg_dirs = Vec::with_capacity(n_seg);
        let mut total_len = 0.0;
        let mut total_speed = 0.0;
        for k in 0..n_seg {
            let d = p_world[k + 1] - p_world[k];
            let l = d.norm();
            let dir = d / l.max(1e-15);
            let sp = (v_world[k + 1] - v_world[k]).dot(dir);
            seg_lens.push(l);
            seg_dirs.push(dir);
            total_len += l;
            total_speed += sp;
        }

        let rest = (base_rest_len + spec.slack) - takeup;
        let extension = total_len - rest;
        let m_eff = reduced_pair_mass(
            spec.waypoints[0].body_id,
            spec.waypoints[n_wp - 1].body_id,
            body_masses,
        );
        let (k_ten, c_ten) = resolve_solref_kc(
            spec.solref_imp,
            extension.max(0.0),
            spec.stiffness,
            spec.damping,
            m_eff,
            dt,
        );
        if extension > 0.0 {
            potential_energy += 0.5 * k_ten * extension * extension;
        }

        let cap_mu = spec.capstan_mu_theta.max(0.0);
        let base_t = if extension >= 0.0 {
            (k_ten * extension + c_ten * total_speed).max(0.0)
        } else {
            0.0
        };

        let mut seg_tensions = vec![base_t; n_seg];
        if spec.stick_slip_enabled && cap_mu > 0.0 {
            // Entry capstan stick-slip hysteresis on segment 0 (between actuator horn and first span)
            let entry_theta = if spec.waypoints[0].wrap_angle_rad > 0.0 {
                spec.waypoints[0].wrap_angle_rad
            } else {
                1.0
            };
            let att0 = (cap_mu * entry_theta).clamp(0.0, 2.0).exp();
            let prev_t0 = act.tendons[ti].segment_tensions.first().copied().unwrap_or(base_t);
            let prev_l0 = act.tendons[ti]
                .prev_segment_lengths
                .first()
                .copied()
                .unwrap_or(seg_lens[0]);
            let dl0 = seg_lens[0] - prev_l0;
            // Elastic trial tension when stuck, projected onto Capstan friction cone [base_t/att, base_t*att]
            let t0_trial = (prev_t0 + k_ten * dl0).max(0.0);
            let t0 = if base_t <= 0.0 {
                0.0
            } else {
                t0_trial.clamp(base_t / att0, base_t * att0)
            };
            seg_tensions[0] = t0;

            // Propagate stick-slip Capstan cone sequentially across intermediate guide waypoints
            for k in 1..n_seg {
                let bend_angle = if spec.waypoints[k].wrap_angle_rad > 0.0 {
                    spec.waypoints[k].wrap_angle_rad
                } else {
                    seg_dirs[k - 1].dot(seg_dirs[k]).clamp(-1.0, 1.0).acos()
                };
                let att = (cap_mu * bend_angle).clamp(0.0, 2.0).exp();
                let t_up = seg_tensions[k - 1];
                let prev_tk = act.tendons[ti]
                    .segment_tensions
                    .get(k)
                    .copied()
                    .unwrap_or(t_up);
                let prev_lk = act.tendons[ti]
                    .prev_segment_lengths
                    .get(k)
                    .copied()
                    .unwrap_or(seg_lens[k]);
                let dlk = seg_lens[k] - prev_lk;
                let tk_trial = (prev_tk + k_ten * dlk).max(0.0);
                seg_tensions[k] = if t_up <= 0.0 {
                    0.0
                } else {
                    tk_trial.clamp(t_up / att, t_up * att)
                };
            }
        } else if cap_mu > 0.0 {
            // Smooth pull/release exponential attenuation without stick-slip state
            let pull_frac = (sv_vel / 0.35).tanh();
            let mut running_t = base_t * (-cap_mu * pull_frac.max(0.0)).exp();
            seg_tensions[0] = running_t;
            for k in 1..n_seg {
                let bend_angle = if spec.waypoints[k].wrap_angle_rad > 0.0 {
                    spec.waypoints[k].wrap_angle_rad
                } else {
                    seg_dirs[k - 1].dot(seg_dirs[k]).clamp(-1.0, 1.0).acos()
                };
                running_t *= (-cap_mu * bend_angle * pull_frac).exp();
                seg_tensions[k] = running_t.max(0.0);
            }
        }

        if commit_state {
            act.tendons[ti].tension = base_t;
            act.tendons[ti].segment_tensions.clone_from(&seg_tensions);
            act.tendons[ti].prev_segment_lengths.clone_from(&seg_lens);
        }

        // Apply exact equal-and-opposite segment forces at every waypoint pair (zero net internal wrench!)
        for k in 0..n_seg {
            let f_seg = seg_tensions[k] * seg_dirs[k];
            let ba = spec.waypoints[k].body_id;
            let bb = spec.waypoints[k + 1].body_id;
            body_wrenches[ba] += Vec6::from_parts(f_seg, r_world[k].cross(f_seg));
            body_wrenches[bb] -= Vec6::from_parts(f_seg, r_world[k + 1].cross(f_seg));
            if assemble_all_joint_tangents
                && extension >= 0.0
                && (implicit_scale_k > 0.0 || implicit_scale_c > 0.0)
            {
                let dd = seg_dirs[k].outer(seg_dirs[k]);
                let k_par = implicit_scale_k * k_ten + implicit_scale_c * c_ten;
                let k_perp = implicit_scale_k * (seg_tensions[k] / seg_lens[k].max(1e-15)).max(0.0);
                let k3 = k_par * dd + k_perp * (Mat3::IDENTITY - dd);
                push_linear_pair_blocks(
                    &mut implicit_blocks,
                    ba,
                    bb,
                    &point_wrench_mat6(r_world[k]),
                    &point_wrench_mat6(r_world[k + 1]),
                    k3,
                );
            }
        }
    }

    // --- 2. Evaluate Articulated & Compliant Joints ---
    for (ji, joint) in joints.iter().enumerate() {
        let ba = joint.body_a;
        let bb = joint.body_b;
        let xa = poses[ba];
        let xb = poses[bb];
        let va = vels[ba];
        let vb = vels[bb];
        let ra_mat = xa.rot.to_mat3();
        let rb_mat = xb.rot.to_mat3();
        let want_implicit = (joint.implicit_coupling || assemble_all_joint_tangents)
            && (implicit_scale_k > 0.0 || implicit_scale_c > 0.0);

        let m_joint = reduced_pair_mass(ba, bb, body_masses);

        match joint.joint_type {
            JointType::AxialStopGap {
                axis_local_a,
                min_gap,
                stiffness,
            } => {
                let sep = xb.pos - xa.pos;
                let axis = ra_mat.mul_vec(axis_local_a);
                let gap = sep.dot(axis);
                let pen = (min_gap - gap).max(0.0);
                if pen > 0.0 {
                    let (k_stop, _c_stop) =
                        resolve_solref_kc(joint.solref_imp, pen, stiffness, 0.0, m_joint, dt);
                    potential_energy += 0.5 * k_stop * pen * pen;
                    let push = (k_stop * pen) * axis;
                    body_wrenches[bb] += Vec6::from_parts(push, Vec3::ZERO);
                    body_wrenches[ba] -= Vec6::from_parts(push, sep.cross(push));
                    if want_implicit && implicit_scale_k > 0.0 {
                        let k_lin = (implicit_scale_k * k_stop) * axis.outer(axis);
                        let j_a = point_wrench_mat6(sep);
                        let j_b = Mat6::from_blocks(Mat3::IDENTITY, Mat3::ZERO, Mat3::ZERO, Mat3::ZERO);
                        push_linear_pair_blocks(
                            &mut implicit_blocks,
                            ba,
                            bb,
                            &j_a,
                            &j_b,
                            k_lin,
                        );
                    }
                }
            }
            JointType::Revolute {
                axis_local_a,
                anchor_a,
                anchor_b,
                pos_stiffness,
                pos_damping,
                align_stiffness,
                align_damping,
            } => {
                let arm_a = xa.rot.rotate(anchor_a);
                let arm_b = xb.rot.rotate(anchor_b);
                let anchor_b_zero = anchor_b.norm_sq() == 0.0;
                // Position constraint (exact parity with gpu_core.py:544-547 when anchor_b == 0)
                let (err, relv, moment_a, moment_b) = if anchor_b_zero {
                    let sep = xb.pos - xa.pos;
                    let e = sep - arm_a;
                    let rv = vb.lin() - va.lin() - va.ang().cross(sep);
                    (e, rv, sep, Vec3::ZERO)
                } else {
                    let pa = xa.pos + arm_a;
                    let pb = xb.pos + arm_b;
                    let e = pb - pa;
                    let rv = (vb.lin() + vb.ang().cross(arm_b)) - (va.lin() + va.ang().cross(arm_a));
                    (e, rv, pb - xa.pos, arm_b)
                };
                let (pos_k, pos_c) = resolve_solref_kc(
                    joint.solref_imp,
                    err.norm(),
                    pos_stiffness,
                    pos_damping,
                    m_joint,
                    dt,
                );
                potential_energy += 0.5 * pos_k * err.norm_sq();
                let force = -pos_k * err - pos_c * relv;
                body_wrenches[bb] += Vec6::from_parts(force, moment_b.cross(force));
                body_wrenches[ba] -= Vec6::from_parts(force, moment_a.cross(force));

                // Hinge angle, rate, actuator PD + Stribeck/Coulomb static friction + DC speed-torque envelope
                let axis_unit_local = axis_local_a.normalized_or(Vec3::EZ);
                let axis_w = ra_mat.mul_vec(axis_unit_local);
                let axis_b_w = rb_mat.mul_vec(axis_unit_local);
                let rel_r = ra_mat.transpose().mul_mat(rb_mat);
                let angle = if (axis_unit_local - Vec3::EZ).norm() < 1e-12 {
                    // Exact match to gpu_core.py:548
                    rel_r.0[1][0].atan2(rel_r.0[0][0])
                } else {
                    let q_rel = xa.rot.conj().mul(xb.rot).canonical();
                    let proj = q_rel.vec().dot(axis_unit_local);
                    2.0 * proj.atan2(q_rel.w())
                };

                let omega = vb.ang() - va.ang();
                let rate = omega.dot(axis_w);
                let target = joint
                    .actuator_id
                    .and_then(|id| act.joint_targets.get(id).copied())
                    .unwrap_or(0.0);

                let raw_err = target - angle;
                let eff_err = if joint.deadband > 0.0 {
                    raw_err.signum() * (raw_err.abs() - joint.deadband).max(0.0)
                } else {
                    raw_err
                };
                let mut raw_t = joint.kp * eff_err - joint.kd * rate;

                // Joint angle soft limits if configured
                if let Some([lo, hi]) = joint.limits {
                    let k_lim = align_stiffness.max(20.0);
                    if angle < lo {
                        raw_t += k_lim * (lo - angle);
                        potential_energy += 0.5 * k_lim * (lo - angle).powi(2);
                    } else if angle > hi {
                        raw_t += k_lim * (hi - angle);
                        potential_energy += 0.5 * k_lim * (hi - angle).powi(2);
                    }
                }

                // Static friction (exact Coulomb deadband in parity mode, or Stribeck boost when enabled)
                let sf = if stribeck_enabled && joint.static_friction > 0.0 {
                    joint.static_friction * (1.0 + 0.35 * (-(rate / 0.05).powi(2)).exp())
                } else {
                    joint.static_friction.max(0.0)
                };
                let tlim = if joint.torque_limit > 0.0 {
                    joint.torque_limit
                } else {
                    f64::INFINITY
                };
                let mut jt_torque =
                    (raw_t.signum() * (raw_t.abs() - sf).max(0.0)).clamp(-tlim, tlim);
                let cap = if joint.speed_limit > 0.0 && tlim.is_finite() {
                    if jt_torque * rate > 0.0 {
                        tlim * (1.0 - rate.abs() / joint.speed_limit.max(1e-9)).clamp(0.0, 1.0)
                    } else {
                        tlim
                    }
                } else {
                    tlim
                };
                jt_torque = jt_torque.clamp(-cap, cap);
                if commit_state {
                    act.last_joint_torque[ji] = jt_torque;
                    act.last_joint_torque_cap[ji] = cap;
                }

                potential_energy += align_stiffness * (1.0 - axis_w.dot(axis_b_w));
                let align_torque = align_stiffness * axis_b_w.cross(axis_w)
                    - align_damping * (omega - rate * axis_w);
                let total_torque = jt_torque * axis_w + align_torque;
                body_wrenches[bb] += Vec6::from_parts(Vec3::ZERO, total_torque);
                body_wrenches[ba] -= Vec6::from_parts(Vec3::ZERO, total_torque);

                if want_implicit {
                    let kp_eff = implicit_scale_k * pos_k + implicit_scale_c * pos_c;
                    let k_lin = Mat3::diag([kp_eff, kp_eff, kp_eff]);
                    let j_a = point_wrench_mat6(moment_a);
                    let j_b = point_wrench_mat6(moment_b);
                    push_linear_pair_blocks(&mut implicit_blocks, ba, bb, &j_a, &j_b, k_lin);

                    // Angular alignment + PD tangent stiffness
                    let aa = axis_w.outer(axis_w);
                    let perp = Mat3::IDENTITY - aa;
                    let k_ax = implicit_scale_k * joint.kp + implicit_scale_c * joint.kd;
                    let k_al = implicit_scale_k * align_stiffness + implicit_scale_c * align_damping;
                    let k_rot = k_ax * aa + k_al * perp;
                    let blk_rot = Mat6::from_blocks(Mat3::ZERO, Mat3::ZERO, Mat3::ZERO, k_rot);
                    implicit_blocks.push((ba, ba, blk_rot));
                    implicit_blocks.push((bb, bb, blk_rot));
                    implicit_blocks.push((ba, bb, -1.0 * blk_rot));
                    implicit_blocks.push((bb, ba, -1.0 * blk_rot));
                }
            }
            JointType::Prismatic {
                axis_local_a,
                anchor_a,
                anchor_b,
                perp_stiffness,
                perp_damping,
            } => {
                let arm_a = xa.rot.rotate(anchor_a);
                let arm_b = xb.rot.rotate(anchor_b);
                let pa = xa.pos + arm_a;
                let pb = xb.pos + arm_b;
                let axis_w = ra_mat.mul_vec(axis_local_a.normalized_or(Vec3::EX));
                let diff = pb - pa;
                let slide = diff.dot(axis_w);
                let perp_err = diff - slide * axis_w;
                let rv = (vb.lin() + vb.ang().cross(arm_b)) - (va.lin() + va.ang().cross(arm_a));
                let rate = rv.dot(axis_w);
                let perp_rv = rv - rate * axis_w;

                let target = joint
                    .actuator_id
                    .and_then(|id| act.joint_targets.get(id).copied())
                    .unwrap_or(0.0);
                let raw_err = target - slide;
                let eff_err = raw_err.signum() * (raw_err.abs() - joint.deadband.max(0.0)).max(0.0);
                let mut raw_f = joint.kp * eff_err - joint.kd * rate;
                if let Some([lo, hi]) = joint.limits {
                    if slide < lo {
                        raw_f += perp_stiffness * (lo - slide);
                    } else if slide > hi {
                        raw_f += perp_stiffness * (hi - slide);
                    }
                }
                let sf = joint.static_friction.max(0.0);
                let flim = if joint.torque_limit > 0.0 {
                    joint.torque_limit
                } else {
                    f64::INFINITY
                };
                let mut ax_f = (raw_f.signum() * (raw_f.abs() - sf).max(0.0)).clamp(-flim, flim);
                if joint.speed_limit > 0.0 && flim.is_finite() && ax_f * rate > 0.0 {
                    let cap = flim * (1.0 - rate.abs() / joint.speed_limit.max(1e-9)).clamp(0.0, 1.0);
                    ax_f = ax_f.clamp(-cap, cap);
                }
                if commit_state {
                    act.last_joint_torque[ji] = ax_f;
                    act.last_joint_torque_cap[ji] = flim;
                }

                let (pk, pc) = resolve_solref_kc(
                    joint.solref_imp,
                    perp_err.norm(),
                    perp_stiffness,
                    perp_damping,
                    m_joint,
                    dt,
                );
                let force = ax_f * axis_w - pk * perp_err - pc * perp_rv;
                // Lock relative orientation via SO(3) log restoring torque
                let phi_rel = xa.rot.rotate(xa.rot.conj().mul(xb.rot).log());
                let omega = vb.ang() - va.ang();
                let rot_k = pk * 0.02;
                let rot_c = pc * 0.02;
                let torque = -rot_k * phi_rel - rot_c * omega;
                potential_energy += 0.5 * pk * perp_err.norm_sq() + 0.5 * rot_k * phi_rel.norm_sq();

                body_wrenches[bb] += Vec6::from_parts(force, arm_b.cross(force) + torque);
                body_wrenches[ba] -= Vec6::from_parts(force, (pb - xa.pos).cross(force) + torque);

                if want_implicit {
                    let aa = axis_w.outer(axis_w);
                    let k_ax = implicit_scale_k * joint.kp + implicit_scale_c * joint.kd;
                    let k_p = implicit_scale_k * pk + implicit_scale_c * pc;
                    let k_lin = k_ax * aa + k_p * (Mat3::IDENTITY - aa);
                    push_linear_pair_blocks(
                        &mut implicit_blocks,
                        ba,
                        bb,
                        &point_wrench_mat6(pb - xa.pos),
                        &point_wrench_mat6(arm_b),
                        k_lin,
                    );
                    let kr = implicit_scale_k * rot_k + implicit_scale_c * rot_c;
                    let blk_rot =
                        Mat6::from_blocks(Mat3::ZERO, Mat3::ZERO, Mat3::ZERO, Mat3::diag([kr; 3]));
                    implicit_blocks.push((ba, ba, blk_rot));
                    implicit_blocks.push((bb, bb, blk_rot));
                    implicit_blocks.push((ba, bb, -1.0 * blk_rot));
                    implicit_blocks.push((bb, ba, -1.0 * blk_rot));
                }
            }
            JointType::Spherical {
                anchor_a,
                anchor_b,
                pos_stiffness,
                pos_damping,
            } => {
                let arm_a = xa.rot.rotate(anchor_a);
                let arm_b = xb.rot.rotate(anchor_b);
                let pa = xa.pos + arm_a;
                let pb = xb.pos + arm_b;
                let err = pb - pa;
                let rv = (vb.lin() + vb.ang().cross(arm_b)) - (va.lin() + va.ang().cross(arm_a));
                let (pos_k, pos_c) = resolve_solref_kc(
                    joint.solref_imp,
                    err.norm(),
                    pos_stiffness,
                    pos_damping,
                    m_joint,
                    dt,
                );
                potential_energy += 0.5 * pos_k * err.norm_sq();
                let force = -pos_k * err - pos_c * rv;
                body_wrenches[bb] += Vec6::from_parts(force, arm_b.cross(force));
                body_wrenches[ba] -= Vec6::from_parts(force, (pb - xa.pos).cross(force));

                if want_implicit {
                    let kp = implicit_scale_k * pos_k + implicit_scale_c * pos_c;
                    push_linear_pair_blocks(
                        &mut implicit_blocks,
                        ba,
                        bb,
                        &point_wrench_mat6(pb - xa.pos),
                        &point_wrench_mat6(arm_b),
                        Mat3::diag([kp; 3]),
                    );
                }
            }
            JointType::Weld {
                rel_pose,
                pos_stiffness,
                rot_stiffness,
            } => {
                let target_pos_b = xa.transform_point(rel_pose.pos);
                let target_rot_b = xa.rot.mul(rel_pose.rot);
                let err = xb.pos - target_pos_b;
                let sep = xb.pos - xa.pos;
                let rv = vb.lin() - va.lin() - va.ang().cross(sep);
                let base_pos_damping = 0.01 * pos_stiffness.sqrt();
                let (pos_k, pos_damping) = resolve_solref_kc(
                    joint.solref_imp,
                    err.norm(),
                    pos_stiffness,
                    base_pos_damping,
                    m_joint,
                    dt,
                );
                let force = -pos_k * err - pos_damping * rv;

                let phi_err = target_rot_b.rotate(target_rot_b.conj().mul(xb.rot).log());
                let omega = vb.ang() - va.ang();
                let base_rot_damping = 0.01 * rot_stiffness.sqrt();
                let (rot_k, rot_damping) = resolve_solref_kc(
                    joint.solref_imp,
                    phi_err.norm(),
                    rot_stiffness,
                    base_rot_damping,
                    m_joint,
                    dt,
                );
                let torque = -rot_k * phi_err - rot_damping * omega;
                potential_energy +=
                    0.5 * pos_k * err.norm_sq() + 0.5 * rot_k * phi_err.norm_sq();

                body_wrenches[bb] += Vec6::from_parts(force, torque);
                body_wrenches[ba] -= Vec6::from_parts(force, sep.cross(force) + torque);

                if want_implicit {
                    let kp = implicit_scale_k * pos_k + implicit_scale_c * pos_damping;
                    let kr = implicit_scale_k * rot_k + implicit_scale_c * rot_damping;
                    push_linear_pair_blocks(
                        &mut implicit_blocks,
                        ba,
                        bb,
                        &point_wrench_mat6(sep),
                        &point_wrench_mat6(Vec3::ZERO),
                        Mat3::diag([kp; 3]),
                    );
                    let blk_rot =
                        Mat6::from_blocks(Mat3::ZERO, Mat3::ZERO, Mat3::ZERO, Mat3::diag([kr; 3]));
                    implicit_blocks.push((ba, ba, blk_rot));
                    implicit_blocks.push((bb, bb, blk_rot));
                    implicit_blocks.push((ba, bb, -1.0 * blk_rot));
                    implicit_blocks.push((bb, ba, -1.0 * blk_rot));
                }
            }
            JointType::CableSpring {
                anchor_a,
                anchor_b,
                rest_length,
                stiffness,
                damping,
                compression_allowed,
            } => {
                let arm_a = xa.rot.rotate(anchor_a);
                let arm_b = xb.rot.rotate(anchor_b);
                let pa = xa.pos + arm_a;
                let pb = xb.pos + arm_b;
                let delta = pb - pa;
                let len = delta.norm().max(1e-15);
                let dir = delta / len;
                let ext = len - rest_length;

                let active = compression_allowed || ext > 0.0;
                if active {
                    let va_p = va.lin() + va.ang().cross(arm_a);
                    let vb_p = vb.lin() + vb.ang().cross(arm_b);
                    let speed = (vb_p - va_p).dot(dir);
                    let (k_cab, c_cab) = resolve_solref_kc(
                        joint.solref_imp,
                        ext.abs(),
                        stiffness,
                        damping,
                        m_joint,
                        dt,
                    );
                    let raw_ten = k_cab * ext + c_cab * speed;
                    let tension = if compression_allowed {
                        raw_ten
                    } else {
                        raw_ten.max(0.0)
                    };
                    potential_energy += 0.5 * k_cab * ext * ext;
                    let f = tension * dir;
                    body_wrenches[ba] += Vec6::from_parts(f, arm_a.cross(f));
                    body_wrenches[bb] -= Vec6::from_parts(f, arm_b.cross(f));
                    if commit_state {
                        act.last_joint_torque[ji] = tension;
                    }

                    if want_implicit {
                        // Exact SPD-projected 3x3 spring-damper tangent:
                        // K_3 = (sk * k + sc * c) d d^T + sk * max(tension / len, 0) * (I - d d^T)
                        let dd = dir.outer(dir);
                        let k_par = implicit_scale_k * k_cab + implicit_scale_c * c_cab;
                        let k_perp = implicit_scale_k * (tension / len).max(0.0);
                        let k3 = k_par * dd + k_perp * (Mat3::IDENTITY - dd);
                        push_linear_pair_blocks(
                            &mut implicit_blocks,
                            ba,
                            bb,
                            &point_wrench_mat6(arm_a),
                            &point_wrench_mat6(arm_b),
                            k3,
                        );
                    }
                } else if commit_state {
                    act.last_joint_torque[ji] = 0.0;
                }
            }
        }
    }

    ActuationEval {
        potential_energy,
        body_wrenches,
        implicit_blocks,
    }
}

/// Build the `3x6` Jacobian mapping body twist `[v; w]` to point velocity `v + w x arm`,
/// stored in the top 3 rows of a `Mat6` (bottom 3 rows zero).
#[inline]
fn point_wrench_mat6(arm: Vec3) -> Mat6 {
    Mat6::from_blocks(Mat3::IDENTITY, -arm.skew(), Mat3::ZERO, Mat3::ZERO)
}

/// Scatter SPD pair stiffness blocks `J_a^T K J_a`, `-J_a^T K J_b`, `-J_b^T K J_a`, `J_b^T K J_b`
/// for a linear force constraint connecting `body_a` and `body_b`.
fn push_linear_pair_blocks(
    out: &mut Vec<(usize, usize, Mat6)>,
    ba: usize,
    bb: usize,
    ja: &Mat6,
    jb: &Mat6,
    k3: Mat3,
) {
    let k6 = Mat6::from_blocks(k3, Mat3::ZERO, Mat3::ZERO, Mat3::ZERO);
    let k_aa = ja.transpose().mul_mat(k6).mul_mat(*ja);
    let k_bb = jb.transpose().mul_mat(k6).mul_mat(*jb);
    let k_ab = (-1.0) * ja.transpose().mul_mat(k6).mul_mat(*jb);
    let k_ba = k_ab.transpose();
    out.push((ba, ba, k_aa));
    out.push((bb, bb, k_bb));
    out.push((ba, bb, k_ab));
    out.push((bb, ba, k_ba));
}

#[cfg(test)]
mod tests {
    use super::*;
    use fg_ir::TendonWaypoint;
    use fg_math::Quat;

    #[test]
    fn test_multi_waypoint_capstan_stick_slip_hysteresis_and_zero_net_wrench() {
        let poses = vec![
            Se3::new(Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY),
            Se3::new(Vec3::new(0.10, 0.04, 0.0), Quat::from_axis_angle(Vec3::EZ, 0.2)),
            Se3::new(Vec3::new(0.18, 0.12, 0.02), Quat::from_axis_angle(Vec3::EY, -0.1)),
        ];
        let vels = vec![Vec6::ZERO; 3];
        let tendon = TendonEdgeSpec {
            id: 0,
            name: "catheter_t0".into(),
            actuator_id: 0,
            waypoints: vec![
                TendonWaypoint {
                    body_id: 0,
                    local_pos: Vec3::new(0.0, 0.01, 0.0),
                    wrap_angle_rad: 0.4,
                },
                TendonWaypoint {
                    body_id: 1,
                    local_pos: Vec3::new(0.0, 0.01, 0.0),
                    wrap_angle_rad: 0.5,
                },
                TendonWaypoint {
                    body_id: 2,
                    local_pos: Vec3::new(0.0, 0.01, 0.0),
                    wrap_angle_rad: 0.0,
                },
            ],
            drive_mode: TendonDriveMode::Capstan { drum_radius: 0.01 },
            rest_length: 0.0,
            slack: 0.0,
            stiffness: 2000.0,
            damping: 0.5,
            capstan_mu_theta: 0.35,
            stick_slip_enabled: true,
            servo_torque_limit: 2.0,
            servo_speed_limit: 5.0,
            servo_deadband: 0.0,
            angle_limits: [0.0, 3.0],
            solref_imp: None,
        };

        let mut act = ActuationState::from_specs(std::slice::from_ref(&tendon), &[], &poses);
        let dt = 0.001;

        // Pull phase: retract cable to 0.5 rad
        act.tendon_servo_target[0] = 0.5;
        for _ in 0..120 {
            advance_tendon_servos(std::slice::from_ref(&tendon), &mut act, dt);
            let eval = evaluate_actuation(
                3,
                std::slice::from_ref(&tendon),
                &[],
                &poses,
                &poses,
                &vels,
                &mut act,
                0.0,
                0.0,
                false,
                true,
                false,
            );
            // Check exact internal force & world torque balance (sum == 0)
            let mut sum_f = Vec3::ZERO;
            let mut sum_tau = Vec3::ZERO;
            for (bi, w) in eval.body_wrenches.iter().enumerate() {
                sum_f += w.lin();
                sum_tau += poses[bi].pos.cross(w.lin()) + w.ang();
            }
            assert!(sum_f.norm() < 1e-12, "net internal force = {}", sum_f.norm());
            assert!(
                sum_tau.norm() < 1e-12,
                "net internal torque = {}",
                sum_tau.norm()
            );
        }
        let pull_prox = act.tendons[0].tension;
        let pull_dist = act.tendons[0].segment_tensions[1];
        assert!(pull_prox > 5.0);
        // During pull, distal tension is attenuated strictly below proximal tension
        assert!(
            pull_dist < pull_prox * 0.95,
            "pull distal {pull_dist} should be attenuated below proximal {pull_prox}"
        );

        // Partial release phase: drop target slightly -> distal segment sticks at higher tension than pull!
        act.tendon_servo_target[0] = 0.35;
        for _ in 0..40 {
            advance_tendon_servos(std::slice::from_ref(&tendon), &mut act, dt);
            let _ = evaluate_actuation(
                3,
                std::slice::from_ref(&tendon),
                &[],
                &poses,
                &poses,
                &vels,
                &mut act,
                0.0,
                0.0,
                false,
                true,
                false,
            );
        }
        let rel_prox = act.tendons[0].tension;
        let rel_dist = act.tendons[0].segment_tensions[1];
        // Hysteresis signature: during release, distal tension exceeds proximal tension (trapped by Capstan friction!)
        assert!(
            rel_dist > rel_prox,
            "stick-slip hysteresis failed: rel_dist ({rel_dist}) <= rel_prox ({rel_prox})"
        );
    }

    #[test]
    fn test_cable_spring_implicit_tangent_matches_fd() {
        let poses = vec![
            Se3::new(Vec3::new(0.0, 0.0, 0.0), Quat::from_axis_angle(Vec3::EX, 0.1)),
            Se3::new(Vec3::new(0.15, 0.05, -0.02), Quat::from_axis_angle(Vec3::EZ, -0.2)),
        ];
        let vels = vec![Vec6::ZERO; 2];
        let joint = JointEdgeSpec {
            id: 0,
            name: "cable".into(),
            body_a: 0,
            body_b: 1,
            joint_type: JointType::CableSpring {
                anchor_a: Vec3::new(0.01, 0.02, -0.01),
                anchor_b: Vec3::new(-0.02, 0.01, 0.03),
                rest_length: 0.10,
                stiffness: 800.0,
                damping: 0.0,
                compression_allowed: false,
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
        };
        let mut act = ActuationState::from_specs(&[], std::slice::from_ref(&joint), &poses);
        let eval = evaluate_actuation(
            2,
            &[],
            std::slice::from_ref(&joint),
            &poses,
            &poses,
            &vels,
            &mut act,
            1.0,
            0.0,
            true,
            false,
            false,
        );
        // Assemble 12x12 analytic K and verify against FD of -body_wrenches on linear DOFs
        let mut k_aa = Mat6::ZERO;
        let mut k_ab = Mat6::ZERO;
        for &(u, v, blk) in &eval.implicit_blocks {
            if u == 0 && v == 0 {
                k_aa += blk;
            } else if u == 0 && v == 1 {
                k_ab += blk;
            }
        }
        let eps = 1e-7;
        for d in 0..3 {
            let mut twist = Vec6::ZERO;
            twist[d] = 1.0;
            let mut p_plus = poses.clone();
            let mut p_minus = poses.clone();
            p_plus[0] = poses[0].integrate_decoupled(twist, eps);
            p_minus[0] = poses[0].integrate_decoupled(twist, -eps);
            let e_p = evaluate_actuation(
                2,
                &[],
                std::slice::from_ref(&joint),
                &poses,
                &p_plus,
                &vels,
                &mut act,
                0.0,
                0.0,
                false,
                false,
                false,
            );
            let e_m = evaluate_actuation(
                2,
                &[],
                std::slice::from_ref(&joint),
                &poses,
                &p_minus,
                &vels,
                &mut act,
                0.0,
                0.0,
                false,
                false,
                false,
            );
            let col_fd = -(e_p.body_wrenches[0] - e_m.body_wrenches[0]) * (0.5 / eps);
            for r in 0..6 {
                let err = (col_fd[r] - k_aa.0[r][d]).abs();
                assert!(err < 1e-4, "K_aa[{r},{d}] fd={} an={}", col_fd[r], k_aa.0[r][d]);
            }
            let col_ab_fd = -(e_p.body_wrenches[1] - e_m.body_wrenches[1]) * (0.5 / eps);
            for r in 0..6 {
                let err = (col_ab_fd[r] - k_ab.transpose().0[r][d]).abs();
                assert!(
                    err < 1e-4,
                    "K_ba[{r},{d}] fd={} an={}",
                    col_ab_fd[r],
                    k_ab.transpose().0[r][d]
                );
            }
        }
    }

    #[test]
    fn test_resolve_solref_kc_positive_mujoco_mode_governs_dynamics_and_none_preserves_v1() {
        let rest = vec![
            Se3::new(Vec3::ZERO, Quat::IDENTITY),
            Se3::new(Vec3::new(0.10, 0.0, 0.0), Quat::IDENTITY),
        ];
        // Extend tendon by 0.01 m and move body 1 along +X at 0.2 m/s
        let poses = vec![
            Se3::new(Vec3::ZERO, Quat::IDENTITY),
            Se3::new(Vec3::new(0.11, 0.0, 0.0), Quat::IDENTITY),
        ];
        let vels = vec![
            Vec6::ZERO,
            Vec6::from_parts(Vec3::new(0.2, 0.0, 0.0), Vec3::ZERO),
        ];
        let masses = [(1.0_f64, true), (0.5_f64, false)];
        let dt = 0.001;

        let make_tendon = |solref_imp: Option<SolRefImp>| TendonEdgeSpec {
            id: 0,
            name: "t_solref".into(),
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
            solref_imp,
        };

        // 1. solref_imp == None preserves exact (base_k, base_c) untouched for v1 parity
        let t_none = make_tendon(None);
        let mut act_none = ActuationState::from_specs(std::slice::from_ref(&t_none), &[], &rest);
        let ev_none = evaluate_actuation_with_masses(
            2,
            std::slice::from_ref(&t_none),
            &[],
            &rest,
            &poses,
            &vels,
            &mut act_none,
            0.0,
            0.0,
            false,
            true,
            false,
            Some(&masses),
            dt,
        );
        let expected_none = 1000.0 * 0.01 + 5.0 * 0.2;
        assert!((ev_none.body_wrenches[0].lin().x() - expected_none).abs() < 1e-12);

        // 2. Positive MuJoCo mode: [0.005, 0.5] vs [0.5, 5.0] must produce dramatically different forces
        //    governed by (timeconst, dampratio) and reduced mass m_eff = 0.5 kg!
        let sri_stiff = SolRefImp::from_mujoco([0.005, 0.5], [0.9, 0.95, 0.002, 0.5, 2.0]);
        let sri_soft = SolRefImp::from_mujoco([0.5, 5.0], [0.9, 0.95, 0.002, 0.5, 2.0]);
        let t_stiff = make_tendon(Some(sri_stiff));
        let t_soft = make_tendon(Some(sri_soft));
        let mut act_stiff = ActuationState::from_specs(std::slice::from_ref(&t_stiff), &[], &rest);
        let mut act_soft = ActuationState::from_specs(std::slice::from_ref(&t_soft), &[], &rest);

        let ev_stiff = evaluate_actuation_with_masses(
            2,
            std::slice::from_ref(&t_stiff),
            &[],
            &rest,
            &poses,
            &vels,
            &mut act_stiff,
            0.0,
            0.0,
            false,
            true,
            false,
            Some(&masses),
            dt,
        );
        let ev_soft = evaluate_actuation_with_masses(
            2,
            std::slice::from_ref(&t_soft),
            &[],
            &rest,
            &poses,
            &vels,
            &mut act_soft,
            0.0,
            0.0,
            false,
            true,
            false,
            Some(&masses),
            dt,
        );
        let (k_stiff, c_stiff) = sri_stiff.to_force_stiffness_damping(0.01, 0.5, dt);
        let (k_soft, c_soft) = sri_soft.to_force_stiffness_damping(0.01, 0.5, dt);
        let f_stiff = ev_stiff.body_wrenches[0].lin().x();
        let f_soft = ev_soft.body_wrenches[0].lin().x();
        assert!((f_stiff - (k_stiff * 0.01 + c_stiff * 0.2)).abs() < 1e-10);
        assert!((f_soft - (k_soft * 0.01 + c_soft * 0.2)).abs() < 1e-10);
        assert!(
            f_stiff > 100.0 * f_soft,
            "positive SolRefImp must distinguish [0.005, 0.5] vs [0.5, 5.0]: stiff={f_stiff}, soft={f_soft}"
        );

        // 3. Direct negative mode ignores body mass and matches SolRefImp::direct
        let sri_direct = SolRefImp::direct(800.0, 12.0);
        let t_dir = make_tendon(Some(sri_direct));
        let mut act_dir = ActuationState::from_specs(std::slice::from_ref(&t_dir), &[], &rest);
        let ev_dir = evaluate_actuation_with_masses(
            2,
            std::slice::from_ref(&t_dir),
            &[],
            &rest,
            &poses,
            &vels,
            &mut act_dir,
            0.0,
            0.0,
            false,
            true,
            false,
            Some(&masses),
            dt,
        );
        let (k_dir, c_dir) = sri_direct.to_force_stiffness_damping(0.01, 0.5, dt);
        assert!((ev_dir.body_wrenches[0].lin().x() - (k_dir * 0.01 + c_dir * 0.2)).abs() < 1e-10);

        // 4. Reduced pair mass for two free bodies on CableSpring: m_eff = m0*m1/(m0+m1)
        let free_masses = [(0.6_f64, false), (0.3_f64, false)];
        let m_red = (0.6 * 0.3) / (0.6 + 0.3);
        let cable = JointEdgeSpec {
            id: 0,
            name: "c_solref".into(),
            body_a: 0,
            body_b: 1,
            joint_type: JointType::CableSpring {
                anchor_a: Vec3::ZERO,
                anchor_b: Vec3::ZERO,
                rest_length: 0.10,
                stiffness: 1000.0,
                damping: 5.0,
                compression_allowed: false,
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
            solref_imp: Some(sri_stiff),
        };
        let mut act_c = ActuationState::from_specs(&[], std::slice::from_ref(&cable), &rest);
        let ev_c = evaluate_actuation_with_masses(
            2,
            &[],
            std::slice::from_ref(&cable),
            &rest,
            &poses,
            &vels,
            &mut act_c,
            0.0,
            0.0,
            false,
            false,
            false,
            Some(&free_masses),
            dt,
        );
        let (kc, cc) = sri_stiff.to_force_stiffness_damping(0.01, m_red, dt);
        assert!((ev_c.body_wrenches[0].lin().x() - (kc * 0.01 + cc * 0.2)).abs() < 1e-10);
    }
}
