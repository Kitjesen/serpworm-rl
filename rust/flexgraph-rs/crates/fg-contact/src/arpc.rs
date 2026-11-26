//! Unified Implicit Anisotropic Patch Contact (ARPC) engine.
//!
//! Unifies:
//! - **`RatchetWheel`**: spin inertia, rolling resistance, one-way pawl backlash, tangential
//!   spring-damper bristle memory, Stribeck friction ellipse, circular patch pivoting torque,
//!   and bit-level FP64 parity with `serpworm.physics.wheel_stiction` & `gpu_core.py`;
//! - **`AnisotropicScale`**: wheelless ventral-scale / bristle crawling ($\mu_{\mathrm{fwd}} \ll
//!   \mu_{\mathrm{rev}}, \mu_{\mathrm{lat}}$, grounded in Gidoni 2017 stasis-domain mechanics);
//! - **`IsotropicPad`**: rubber/metal footpad and continuum wall contact;
//! - **Hunt-Crossley + Bekker elasto-plastic soil rutting** & lateral rut-wall bulldozing;
//! - **Exact SPD $6\times 6$ `Contact-Schur` tangent assembly** (`k_tan`) + explicit-damping
//!   RHS compensation (`wrench_comp`).

use fg_ir::{ContactMode, ContactSiteSpec};
use fg_math::{Mat3, Mat6, Se3, Vec3, Vec6};
use fg_sdf::{MaterialLibrary, NormalContactModel, SdfField};
use serde::{Deserialize, Serialize};

/// Persistent per-contact-site internal state across timesteps (spin, tangential stick-slip
/// memory, ratchet backlash play, and Bekker elasto-plastic soil rut sinkage).
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArpcSiteState {
    pub spin: f64,
    pub wheel_angle: f64,
    /// Tangential elastic bristle memory stored in world coordinates (matching `gpu_core.memory`).
    pub memory_world: Vec3,
    /// One-way ratchet pawl angular backlash play in `[0, backlash_rad]`.
    pub ratchet_play: f64,
    /// Plastic Bekker soil rut depth (m).
    pub soil_sinkage: f64,
}

/// Output of [`implicit_wheel_contact`] / [`legacy_wheel_contact`] (exact `wheel_stiction.py` parity).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WheelContactEval {
    pub force: [f64; 2],
    pub omega: f64,
    pub memory: [f64; 2],
    pub reaction: f64,
    pub slip: [f64; 2],
}

/// Exact FP64 port of `serpworm.physics.wheel_stiction.contact` (coupled spin + zero-speed stiction).
#[allow(clippy::too_many_arguments)]
pub fn implicit_wheel_contact(
    velocity: [f64; 2],
    spin: f64,
    memory: [f64; 2],
    normal: f64,
    dt: f64,
    radius: f64,
    mu: f64,
    rolling: f64,
    wheel_mass: f64,
) -> WheelContactEval {
    let inertia = 0.5 * wheel_mass * radius * radius;
    let stiffness = 2000.0;
    let damping = 8.0;
    let viscous = 2.0e-6;
    let gain = stiffness * dt + damping;
    let base = [
        -stiffness * memory[0] - gain * velocity[0],
        -stiffness * memory[1] - gain * velocity[1],
    ];
    let limit = rolling * normal * radius;
    let omega_unc = ((inertia * spin / dt - radius * base[0] - limit)
        / (inertia / dt + radius * radius * gain + viscous))
        .max(0.0);
    let mut f0 = base[0] + gain * radius * omega_unc;
    let mut f1 = base[1];
    let mag = f0.hypot(f1);
    let cap = mu * normal;
    let sliding = mag > cap;
    let scale = (cap / mag.max(1e-15)).min(1.0);
    f0 *= scale;
    f1 *= scale;

    let omega = ((inertia * spin / dt - radius * f0 - limit) / (inertia / dt + viscous)).max(0.0);
    let reaction = inertia * (omega - spin) / dt + radius * f0;
    let slip = [velocity[0] - radius * omega, velocity[1]];
    let new_mem = if sliding {
        [
            -(f0 + damping * slip[0]) / stiffness,
            -(f1 + damping * slip[1]) / stiffness,
        ]
    } else {
        [memory[0] + dt * slip[0], memory[1] + dt * slip[1]]
    };

    WheelContactEval {
        force: [f0, f1],
        omega,
        memory: new_mem,
        reaction,
        slip,
    }
}

/// Exact FP64 port of `serpworm.physics.wheel_stiction.old_contact` (legacy parity option).
#[allow(clippy::too_many_arguments)]
pub fn legacy_wheel_contact(
    velocity: [f64; 2],
    spin: f64,
    memory: [f64; 2],
    normal: f64,
    dt: f64,
    radius: f64,
    mu: f64,
    rolling: f64,
) -> WheelContactEval {
    let inertia = 0.5 * 0.009 * radius * radius;
    let gain = 2000.0 * dt + 8.0;
    let resistance = (rolling * normal * radius + 2.0e-6 * spin).min(inertia * spin / dt);
    let base = [
        -2000.0 * memory[0] - gain * velocity[0],
        -2000.0 * memory[1] - gain * velocity[1],
    ];
    let omega_unc = ((spin - dt * (radius * base[0] + resistance) / inertia)
        / (1.0 + dt * radius * radius * gain / inertia))
        .max(0.0);
    let mut f0 = base[0] + gain * radius * omega_unc;
    let mut f1 = base[1];
    let mag = f0.hypot(f1);
    let cap = mu * normal;
    let sliding = mag > cap;
    let scale = (cap / mag.max(1e-15)).min(1.0);
    f0 *= scale;
    f1 *= scale;

    let free = spin - dt * (radius * f0 + resistance) / inertia;
    let omega = free.max(0.0);
    let reaction = inertia * (omega - free) / dt - resistance;
    let slip = [velocity[0] - radius * omega, velocity[1]];
    let new_mem = if sliding {
        [-(f0 + 8.0 * slip[0]) / 2000.0, -(f1 + 8.0 * slip[1]) / 2000.0]
    } else {
        [memory[0] + dt * slip[0], memory[1] + dt * slip[1]]
    };

    WheelContactEval {
        force: [f0, f1],
        omega,
        memory: new_mem,
        reaction,
        slip,
    }
}

/// Output of [`arpc_normal_and_rut`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ArpcNormalEval {
    pub normal: f64,
    pub new_rut: f64,
    pub kn_tan: f64,
    pub cn_tan: f64,
}

/// Exact FP64 port of `serpworm.physics.wheel_stiction.arpc_normal_and_rut`:
/// Smooth zero-impulse Hunt-Crossley normal force with Bekker elasto-plastic soil rut hysteresis.
#[allow(clippy::too_many_arguments)]
pub fn arpc_normal_and_rut(
    pen: f64,
    vz: f64,
    v_horiz_speed: f64,
    rut: f64,
    dt: f64,
    hc_k: f64,
    hc_alpha: f64,
    soil_exp: f64,
    unload_ratio: f64,
    rut_relax_m: f64,
) -> ArpcNormalEval {
    let pen_pos = pen.max(0.0);
    let has_unload = unload_ratio > 1.0;
    let (rut_rel, new_rut) = if has_unload {
        let inv_relax = 1.0 / rut_relax_m.max(1e-6);
        let rr = rut * (-v_horiz_speed * dt * inv_relax).exp();
        let plastic_frac = 1.0 - 1.0 / unload_ratio.max(1.0);
        (rr, rr.max(plastic_frac * pen_pos))
    } else {
        (0.0, 0.0)
    };

    let in_contact = pen > rut_rel && pen > 0.0;
    if !in_contact {
        return ArpcNormalEval {
            normal: 0.0,
            new_rut,
            kn_tan: 0.0,
            cn_tan: 0.0,
        };
    }

    let pen_pm1 = pen.powf(soil_exp - 1.0);
    let pen_pm2 = pen_pm1 / pen;
    let virgin_el = hc_k * pen * pen_pm1;
    let dpen = (pen - rut_rel).max(0.0);
    let unload_el = unload_ratio * hc_k * dpen * pen_pm1;
    let is_unload = has_unload && (unload_el < virgin_el);

    let f_el = if is_unload { unload_el } else { virgin_el };
    let df_el = if is_unload {
        unload_ratio * hc_k * (pen_pm1 + (soil_exp - 1.0) * dpen * pen_pm2)
    } else {
        soil_exp * hc_k * pen_pm1
    };

    let damp_raw = 1.0 - hc_alpha * vz;
    if damp_raw <= 0.0 {
        return ArpcNormalEval {
            normal: 0.0,
            new_rut,
            kn_tan: 0.0,
            cn_tan: 0.0,
        };
    }

    ArpcNormalEval {
        normal: f_el * damp_raw,
        new_rut,
        kn_tan: df_el * damp_raw,
        cn_tan: hc_alpha * f_el,
    }
}

/// Output of [`arpc_tangential_contact`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ArpcTangentialEval {
    pub force_2d: [f64; 2],
    pub omega: f64,
    pub new_memory: [f64; 2],
    pub new_play: f64,
    pub reaction: f64,
    pub slip: [f64; 2],
    pub tau_z: f64,
    pub ct_fwd_tan: f64,
    pub ct_lat_tan: f64,
    pub c_piv_tan: f64,
}

/// Exact FP64 port of `serpworm.physics.wheel_stiction.arpc_tangential_contact`:
/// Anisotropic Ratchet-Patch Contact tangential force, backlash spin, Stribeck stiction,
/// lateral rut-wall bulldozing, circular patch pivoting torque, and implicit tangents.
#[allow(clippy::too_many_arguments)]
pub fn arpc_tangential_contact(
    tangent_v: [f64; 2],
    spin: f64,
    memory: [f64; 2],
    play: f64,
    normal: f64,
    rut: f64,
    wz_plate: f64,
    dt: f64,
    radius: f64,
    mu_fwd: f64,
    mu_lat: f64,
    rolling: f64,
    backlash_rad: f64,
    patch_radius: f64,
    pivot_mu: f64,
    rut_lat_k: f64,
    static_mu_ratio: f64,
    wheel_mass: f64,
) -> ArpcTangentialEval {
    let inertia = 0.5 * wheel_mass * radius * radius;
    let stiffness = 2000.0;
    let damping = 8.0;
    let viscous = 2.0e-6;
    let gain = stiffness * dt + damping;
    let inv_dt_inertia = inertia / dt;
    let denom_coup = inv_dt_inertia + (radius * radius) * gain + viscous;
    let denom_sat = inv_dt_inertia + viscous;

    let has_bl = backlash_rad > 0.0;
    let base0 = -stiffness * memory[0] - gain * tangent_v[0];
    let base1 = -stiffness * memory[1] - gain * tangent_v[1];
    let limit = rolling * normal * radius;
    let omega_min = if has_bl { -play / dt } else { 0.0 };

    let n_drive = inv_dt_inertia * spin - radius * base0;
    let drive_net = signum_zero(n_drive) * (n_drive.abs() - limit).max(0.0);
    let omega_unc = drive_net / denom_coup;
    let omega_pre = omega_unc.max(omega_min);

    let mut f0 = base0 + gain * radius * omega_pre;
    let mut f1 = base1;

    let smr_excess = (static_mu_ratio - 1.0).max(0.0);
    let v_slip_est = (tangent_v[0] - radius * omega_pre).hypot(tangent_v[1]);
    let stribeck = 1.0 + smr_excess * (-(v_slip_est / 0.03).powi(2)).exp();

    let cap_fwd = mu_fwd * stribeck * normal;
    let cap_lat = mu_lat * stribeck * normal;
    let has_normal = cap_fwd > 1e-15 && cap_lat > 1e-15;
    let (sliding, scale) = if has_normal {
        let u0 = f0 / cap_fwd;
        let u1 = f1 / cap_lat;
        let ell = u0.hypot(u1);
        (ell > 1.0, (1.0 / ell.max(1e-15)).min(1.0))
    } else {
        (true, 0.0)
    };

    f0 *= scale;
    f1 *= scale;

    let n_drive_sat = inv_dt_inertia * spin - radius * f0;
    let drive_sat_net = signum_zero(n_drive_sat) * (n_drive_sat.abs() - limit).max(0.0);
    let omega = (drive_sat_net / denom_sat).max(omega_min);

    let reaction = inv_dt_inertia * (omega - spin) + radius * f0;
    let slip0 = tangent_v[0] - radius * omega;
    let slip1 = tangent_v[1];

    let mem0 = if sliding {
        -(f0 + damping * slip0) / stiffness
    } else {
        memory[0] + dt * slip0
    };
    let mem1 = if sliding {
        -(f1 + damping * slip1) / stiffness
    } else {
        memory[1] + dt * slip1
    };

    let new_play = if has_bl {
        (play + omega * dt).max(0.0).min(backlash_rad)
    } else {
        0.0
    };

    let rlk = rut_lat_k.max(0.0);
    let th_lat = (tangent_v[1] / 0.02).tanh();
    let rut_active = if has_normal { rut } else { 0.0 };
    let f_rut_lat = -(rlk * rut_active) * th_lat;
    let c_rut_lat = (rlk * rut_active / 0.02) * (1.0 - th_lat * th_lat).max(0.0);

    let pr = patch_radius.max(0.0);
    let piv_cap = (2.0 / 3.0) * pivot_mu.max(0.0) * pr * normal;
    let th_wz = (wz_plate / 0.15).tanh();
    let tau_z = -piv_cap * th_wz;
    let c_piv_tan = (piv_cap / 0.15) * (1.0 - th_wz * th_wz).max(0.0);

    let free_rolling = (n_drive.abs() > limit) && (omega_unc > omega_min);
    let g_roll = gain * (denom_sat / denom_coup);
    let g_fwd = if free_rolling { g_roll } else { gain };
    let ct_fwd_tan = g_fwd * scale;
    let ct_lat_tan = gain * scale + c_rut_lat;

    ArpcTangentialEval {
        force_2d: [f0, f1 + f_rut_lat],
        omega,
        new_memory: [mem0, mem1],
        new_play,
        reaction,
        slip: [slip0, slip1],
        tau_z,
        ct_fwd_tan,
        ct_lat_tan,
        c_piv_tan,
    }
}

/// Complete unified ARPC evaluation for one [`ContactSiteSpec`] attached to a rigid body.
#[derive(Copy, Clone, Debug)]
pub struct ArpcSiteEval {
    pub active: bool,
    pub normal_force: f64,
    pub mat_id: usize,
    pub contact_pos_world: Vec3,
    pub contact_normal_world: Vec3,
    pub force_world: Vec3,
    /// Total 6D spatial wrench `[force; torque]` about the body origin (INCLUDING explicit-damping
    /// backward-Euler compensation `wrench_comp` when `implicit = true`, ready to add to body RHS).
    pub wrench: Vec6,
    /// Explicit-damping compensation wrench alone (already added into `wrench` when `implicit = true`).
    pub wrench_comp: Vec6,
    /// Strictly SPD 6x6 `Contact-Schur` LHS stiffness matrix `K_c6` (`Mat6::ZERO` when `implicit = false`).
    pub k_tan: Mat6,
    pub slip_2d: [f64; 2],
    pub next_state: ArpcSiteState,
}

/// Unified Implicit Anisotropic Patch Contact evaluator supporting `RatchetWheel`,
/// `AnisotropicScale`, and `IsotropicPad` against any 3D [`SdfField`] and [`MaterialLibrary`].
///
/// When `site.solref_imp` is `Some(sri)`, uses unit effective mass (`1.0 kg`) to convert
/// positive `(timeconst, dampratio)` mode to force-space stiffness/damping; call
/// [`eval_arpc_site_with_mass`] to supply the exact body effective mass.
#[allow(clippy::too_many_arguments)]
pub fn eval_arpc_site(
    site: &ContactSiteSpec,
    body_pose: Se3,
    body_twist: Vec6,
    sdf: &dyn SdfField,
    materials: &MaterialLibrary,
    state: &ArpcSiteState,
    dt: f64,
    implicit: bool,
) -> ArpcSiteEval {
    eval_arpc_site_with_mass(
        site,
        body_pose,
        body_twist,
        sdf,
        materials,
        state,
        dt,
        implicit,
        1.0,
    )
}

/// Unified Implicit Anisotropic Patch Contact evaluator with explicit effective contact mass
/// `eff_mass` (kg) for MuJoCo [`fg_math::SolRefImp`] soft-constraint regularization.
#[allow(clippy::too_many_arguments)]
pub fn eval_arpc_site_with_mass(
    site: &ContactSiteSpec,
    body_pose: Se3,
    body_twist: Vec6,
    sdf: &dyn SdfField,
    materials: &MaterialLibrary,
    state: &ArpcSiteState,
    dt: f64,
    implicit: bool,
    eff_mass: f64,
) -> ArpcSiteEval {
    let offset_world = body_pose.rot.rotate(site.local_pos);
    let center_world = body_pose.pos + offset_world;

    // Query SDF at contact site center to get surface normal, material, and penetration depth.
    // For RatchetWheel, shift rim along `-radius * unit(up - axle*(axle·up))` matching `gpu_core.py:565`.
    let q_center = sdf.query(center_world);
    let up_g = q_center.normal.normalized_or(Vec3::EZ);

    let (rim_world, axle_world_opt, fwd_axis_world) = match site.mode {
        ContactMode::RatchetWheel {
            axle_axis_local, ..
        } => {
            let axle_w = body_pose
                .rot
                .rotate(axle_axis_local)
                .normalized_or(Vec3::EY);
            let radial = up_g - axle_w * up_g.dot(axle_w);
            let rim = -site.radius * radial.normalized_or(up_g);
            let fwd = up_g.cross(axle_w).normalized_or(Vec3::EX);
            (rim, Some(axle_w), fwd)
        }
        ContactMode::AnisotropicScale { fwd_axis_local, .. } => {
            let fwd_raw = body_pose.rot.rotate(fwd_axis_local);
            let fwd_proj = fwd_raw - up_g * fwd_raw.dot(up_g);
            let fwd = fwd_proj.normalized_or(fallback_tangent(up_g));
            (-site.radius * up_g, None, fwd)
        }
        ContactMode::IsotropicPad { .. } => {
            let fwd = fallback_tangent(up_g);
            (-site.radius * up_g, None, fwd)
        }
    };

    let arm_world = offset_world + rim_world;
    let contact_pos_world = body_pose.pos + arm_world;
    let q_contact = sdf.query(contact_pos_world);
    let up_c = q_contact.normal.normalized_or(up_g);
    let pen = -q_contact.phi;
    let mat = materials.get(q_contact.mat_id);

    let vel_contact = body_twist.lin() + body_twist.ang().cross(arm_world);
    let vn = vel_contact.dot(up_c);

    // Orthonormal local tangent frame `(u0 = forward, u1 = lateral, u2 = up_c)`
    let u0 = (fwd_axis_world - up_c * fwd_axis_world.dot(up_c)).normalized_or(fallback_tangent(up_c));
    let u1 = up_c.cross(u0).normalized();
    let tv0 = vel_contact.dot(u0);
    let tv1 = vel_contact.dot(u1);
    let v_horiz = tv0.hypot(tv1);
    let mem0 = state.memory_world.dot(u0);
    let mem1 = state.memory_world.dot(u1);

    // 1. Normal contact force + optional MuJoCo SolRefImp regularization + Bekker soil rutting
    let (normal_force, new_rut, kn_tan, cn_tan) = if let Some(sri) = site.solref_imp {
        if pen > 0.0 {
            let [k_kb, b_kb, imp, imp_p] = sri.kbip(pen, 2.0 * dt.max(0.0));
            let m_scale = if sri.is_direct_mode() {
                1.0
            } else {
                eff_mass.max(0.0)
            };
            let kn = m_scale * imp * imp * k_kb;
            let cn = m_scale * imp * b_kb;
            let raw = kn * pen - cn * vn;
            if raw > 0.0 {
                let kn_t = (m_scale * k_kb * (imp * imp + 2.0 * imp * imp_p * pen)).max(kn);
                (raw, 0.0, kn_t, cn)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    } else {
        match mat.normal_model {
            NormalContactModel::KelvinVoigt { kn, cn } => {
                let raw = kn * pen - cn * vn;
                if pen > 0.0 && raw > 0.0 {
                    (raw, 0.0, kn, cn)
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                }
            }
            NormalContactModel::HuntCrossleyBekker {
                hc_k,
                hc_alpha,
                soil_exp,
                unload_ratio,
                rut_relax_m,
            } => {
                let res = arpc_normal_and_rut(
                    pen,
                    vn,
                    v_horiz,
                    state.soil_sinkage,
                    dt,
                    hc_k,
                    hc_alpha,
                    soil_exp,
                    unload_ratio,
                    rut_relax_m,
                );
                (res.normal, res.new_rut, res.kn_tan, res.cn_tan)
            }
        }
    };

    let active = normal_force > 0.0;
    let wz_n = body_twist.ang().dot(up_c);
    let same_mat = q_contact.mat_id == site.material_profile_id;
    let base_mat = materials.get(site.material_profile_id);
    let patch_r = if same_mat {
        site.patch_radius
    } else if site.patch_radius > 0.0 && base_mat.patch_radius > 1e-9 {
        site.patch_radius * (mat.patch_radius / base_mat.patch_radius)
    } else {
        mat.patch_radius
    };

    // 2. Mode-specific tangential tribology & spin/memory update
    match site.mode {
        ContactMode::RatchetWheel {
            wheel_mass,
            rolling_resistance,
            backlash_rad,
            pivot_mu,
            mu_fwd,
            mu_lat_ratio,
            static_mu_ratio,
            ..
        } => {
            let axle_w = axle_world_opt.unwrap_or(Vec3::EY);
            // Exact site parameters on nominal material (bit-exact `gpu_core` parity);
            // smooth multiplicative material scaling when crossing into a different terrain zone!
            let (mu_eff_fwd, mu_eff_lat, roll_eff, piv_mu_eff, smr_eff) = if same_mat {
                (
                    mu_fwd.max(0.0),
                    (mu_fwd * mu_lat_ratio).max(0.0),
                    rolling_resistance.max(0.0),
                    pivot_mu.max(0.0),
                    static_mu_ratio.max(1.0),
                )
            } else {
                let mu_scale = mat.mu / base_mat.mu.max(1e-6);
                let lat_scale = mat.mu_lat_ratio / base_mat.mu_lat_ratio.max(1e-6);
                let roll_scale = mat.rolling_resistance / base_mat.rolling_resistance.max(1e-6);
                let mf = (mu_fwd * mu_scale).max(0.0);
                (
                    mf,
                    (mf * mu_lat_ratio * lat_scale).max(0.0),
                    (rolling_resistance * roll_scale).max(0.0),
                    mat.pivot_mu.max(0.0),
                    static_mu_ratio.max(mat.static_mu_ratio),
                )
            };

            let tang = arpc_tangential_contact(
                [tv0, tv1],
                state.spin,
                [mem0, mem1],
                state.ratchet_play,
                normal_force,
                new_rut,
                wz_n,
                dt,
                site.radius,
                mu_eff_fwd,
                mu_eff_lat,
                roll_eff,
                backlash_rad,
                patch_r,
                piv_mu_eff,
                mat.rut_lat_k,
                smr_eff,
                wheel_mass,
            );

            let inertia = 0.5 * wheel_mass * site.radius * site.radius;
            let next_spin = if active {
                tang.omega
            } else {
                state.spin * (1.0 - dt * 2.0e-6 / inertia.max(1e-12)).max(0.0)
            };
            let next_state = ArpcSiteState {
                spin: next_spin,
                wheel_angle: state.wheel_angle + dt * next_spin,
                memory_world: if active {
                    u0 * tang.new_memory[0] + u1 * tang.new_memory[1]
                } else {
                    Vec3::ZERO
                },
                ratchet_play: if active {
                    tang.new_play
                } else {
                    state.ratchet_play
                },
                soil_sinkage: new_rut,
            };

            if !active {
                return ArpcSiteEval {
                    active: false,
                    normal_force: 0.0,
                    mat_id: q_contact.mat_id,
                    contact_pos_world,
                    contact_normal_world: up_c,
                    force_world: Vec3::ZERO,
                    wrench: Vec6::ZERO,
                    wrench_comp: Vec6::ZERO,
                    k_tan: Mat6::ZERO,
                    slip_2d: tang.slip,
                    next_state,
                };
            }

            let force_world = u0 * tang.force_2d[0] + u1 * tang.force_2d[1] + up_c * normal_force;
            let rt = rim_world.cross(force_world);
            // Subtract rim torque along free wheel axle and add exact axle reaction + patch pivot torque
            let torque_world = offset_world.cross(force_world) + rt
                - axle_w * rt.dot(axle_w)
                + axle_w * tang.reaction
                + up_c * tang.tau_z;
            let base_wrench = Vec6::from_parts(force_world, torque_world);

            let (wrench_comp, k_tan) = if implicit {
                assemble_contact_schur_6x6(
                    arm_world,
                    u0,
                    u1,
                    up_c,
                    tv0,
                    tv1,
                    vn,
                    wz_n,
                    tang.ct_fwd_tan,
                    tang.ct_lat_tan,
                    cn_tan,
                    kn_tan,
                    tang.c_piv_tan,
                    dt,
                )
            } else {
                (Vec6::ZERO, Mat6::ZERO)
            };

            ArpcSiteEval {
                active: true,
                normal_force,
                mat_id: q_contact.mat_id,
                contact_pos_world,
                contact_normal_world: up_c,
                force_world,
                wrench: base_wrench + wrench_comp,
                wrench_comp,
                k_tan,
                slip_2d: tang.slip,
                next_state,
            }
        }
        ContactMode::AnisotropicScale {
            mu_fwd,
            mu_rev,
            mu_lat,
            pivot_mu,
            static_mu_ratio,
            ..
        } => {
            eval_patch_bristle(
                arm_world,
                contact_pos_world,
                u0,
                u1,
                up_c,
                tv0,
                tv1,
                vn,
                wz_n,
                mem0,
                mem1,
                active,
                normal_force,
                new_rut,
                kn_tan,
                cn_tan,
                mu_fwd,
                mu_rev,
                mu_lat,
                patch_r,
                if pivot_mu > 0.0 { pivot_mu } else { mat.pivot_mu },
                mat.rut_lat_k,
                static_mu_ratio.max(mat.static_mu_ratio),
                q_contact.mat_id,
                dt,
                implicit,
            )
        }
        ContactMode::IsotropicPad {
            mu,
            pivot_mu,
            static_mu_ratio,
            ..
        } => {
            let mu_eff = if mu > 0.0 { mu } else { mat.mu };
            eval_patch_bristle(
                arm_world,
                contact_pos_world,
                u0,
                u1,
                up_c,
                tv0,
                tv1,
                vn,
                wz_n,
                mem0,
                mem1,
                active,
                normal_force,
                new_rut,
                kn_tan,
                cn_tan,
                mu_eff,
                mu_eff,
                mu_eff * mat.mu_lat_ratio,
                patch_r,
                if pivot_mu > 0.0 { pivot_mu } else { mat.pivot_mu },
                mat.rut_lat_k,
                static_mu_ratio.max(mat.static_mu_ratio),
                q_contact.mat_id,
                dt,
                implicit,
            )
        }
    }
}

/// Shared implicit bristle-memory tangential patch evaluator for `AnisotropicScale`
/// (`mu_fwd != mu_rev`) and `IsotropicPad` (`mu_fwd == mu_rev`).
#[allow(clippy::too_many_arguments)]
fn eval_patch_bristle(
    arm_world: Vec3,
    contact_pos_world: Vec3,
    u0: Vec3,
    u1: Vec3,
    up_c: Vec3,
    tv0: f64,
    tv1: f64,
    vn: f64,
    wz_n: f64,
    mem0: f64,
    mem1: f64,
    active: bool,
    normal_force: f64,
    new_rut: f64,
    kn_tan: f64,
    cn_tan: f64,
    mu_fwd: f64,
    mu_rev: f64,
    mu_lat: f64,
    patch_radius: f64,
    pivot_mu: f64,
    rut_lat_k: f64,
    static_mu_ratio: f64,
    mat_id: usize,
    dt: f64,
    implicit: bool,
) -> ArpcSiteEval {
    if !active {
        return ArpcSiteEval {
            active: false,
            normal_force: 0.0,
            mat_id,
            contact_pos_world,
            contact_normal_world: up_c,
            force_world: Vec3::ZERO,
            wrench: Vec6::ZERO,
            wrench_comp: Vec6::ZERO,
            k_tan: Mat6::ZERO,
            slip_2d: [tv0, tv1],
            next_state: ArpcSiteState {
                soil_sinkage: new_rut,
                ..Default::default()
            },
        };
    }

    let stiffness = 2000.0;
    let damping = 8.0;
    let gain = stiffness * dt + damping;
    let f0_raw = -stiffness * mem0 - gain * tv0;
    let f1_raw = -stiffness * mem1 - gain * tv1;

    // Smooth directional friction coefficient along forward/reverse axis:
    // Forward motion (tv0 > 0, resisting force f0_raw < 0) uses `mu_fwd`;
    // Reverse motion (tv0 < 0, resisting force f0_raw > 0) uses `mu_rev` (ventral scale anchor!).
    let w_fwd = 0.5 * (1.0 + (-f0_raw / (0.05 * normal_force.max(1e-6) + 1e-6)).tanh());
    let mu_long = mu_rev + (mu_fwd - mu_rev) * w_fwd;

    let smr_excess = (static_mu_ratio - 1.0).max(0.0);
    let speed = tv0.hypot(tv1);
    let stribeck = 1.0 + smr_excess * (-(speed / 0.03).powi(2)).exp();

    let cap_long = (mu_long * stribeck * normal_force).max(1e-15);
    let cap_lat = (mu_lat * stribeck * normal_force).max(1e-15);
    let ell = (f0_raw / cap_long).hypot(f1_raw / cap_lat);
    let sliding = ell > 1.0;
    let scale = (1.0 / ell.max(1e-15)).min(1.0);

    let f0 = f0_raw * scale;
    let f1 = f1_raw * scale;

    let new_mem0 = if sliding {
        -(f0 + damping * tv0) / stiffness
    } else {
        mem0 + dt * tv0
    };
    let new_mem1 = if sliding {
        -(f1 + damping * tv1) / stiffness
    } else {
        mem1 + dt * tv1
    };

    // Lateral rut-wall bulldozing resistance on soft soil
    let rlk = rut_lat_k.max(0.0);
    let th_lat = (tv1 / 0.02).tanh();
    let f_rut_lat = -(rlk * new_rut) * th_lat;
    let c_rut_lat = (rlk * new_rut / 0.02) * (1.0 - th_lat * th_lat).max(0.0);

    // Circular contact-patch pivoting torque
    let piv_cap = (2.0 / 3.0) * pivot_mu.max(0.0) * patch_radius.max(0.0) * normal_force;
    let th_wz = (wz_n / 0.15).tanh();
    let tau_piv = -piv_cap * th_wz;
    let c_piv_tan = (piv_cap / 0.15) * (1.0 - th_wz * th_wz).max(0.0);

    let force_world = u0 * f0 + u1 * (f1 + f_rut_lat) + up_c * normal_force;
    let torque_world = arm_world.cross(force_world) + up_c * tau_piv;
    let base_wrench = Vec6::from_parts(force_world, torque_world);

    let ct_fwd = gain * scale;
    let ct_lat = gain * scale + c_rut_lat;
    let (wrench_comp, k_tan) = if implicit {
        assemble_contact_schur_6x6(
            arm_world, u0, u1, up_c, tv0, tv1, vn, wz_n, ct_fwd, ct_lat, cn_tan, kn_tan, c_piv_tan,
            dt,
        )
    } else {
        (Vec6::ZERO, Mat6::ZERO)
    };

    ArpcSiteEval {
        active: true,
        normal_force,
        mat_id,
        contact_pos_world,
        contact_normal_world: up_c,
        force_world,
        wrench: base_wrench + wrench_comp,
        wrench_comp,
        k_tan,
        slip_2d: [tv0, tv1],
        next_state: ArpcSiteState {
            spin: 0.0,
            wheel_angle: 0.0,
            memory_world: u0 * new_mem0 + u1 * new_mem1,
            ratchet_play: 0.0,
            soil_sinkage: new_rut,
        },
    }
}

/// Assemble exact SPD 6x6 `Contact-Schur` tangent `K_c6` and backward-Euler explicit-damping
/// RHS compensation `f_comp` matching `gpu_core.py:654-682` verbatim!
#[allow(clippy::too_many_arguments)]
fn assemble_contact_schur_6x6(
    arm: Vec3,
    u0: Vec3,
    u1: Vec3,
    u2: Vec3,
    tv0: f64,
    tv1: f64,
    vn: f64,
    wz_n: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    k2: f64,
    c_piv: f64,
    dt: f64,
) -> (Vec6, Mat6) {
    let j0 = Vec6::from_parts(u0, arm.cross(u0));
    let j1 = Vec6::from_parts(u1, arm.cross(u1));
    let j2 = Vec6::from_parts(u2, arm.cross(u2));
    let jp = Vec6::from_parts(Vec3::ZERO, u2);

    let f_comp = j0 * (c0 * tv0) + j1 * (c1 * tv1) + j2 * (c2 * vn) + jp * (c_piv * wz_n);

    let w0 = dt * c0.max(0.0);
    let w1 = dt * c1.max(0.0);
    let w2 = dt * c2.max(0.0) + dt * dt * k2.max(0.0);
    let wp = dt * c_piv.max(0.0);

    let k_c6 = j0.outer(j0) * w0 + j1.outer(j1) * w1 + j2.outer(j2) * w2 + jp.outer(jp) * wp;
    (f_comp, k_c6.symmetrize())
}

/// Result of [`eval_sdf_surface_contact`] for beam interior nodes or spherical body shells
/// contacting an SDF wall/pipe/obstacle (exact mirror of `pipe_geometry.py::_accumulate_surface_contact`).
#[derive(Copy, Clone, Debug)]
pub struct SdfSurfaceContactEval {
    pub active: bool,
    pub normal_force: f64,
    /// Total 3D world force (including explicit-damping RHS compensation when `implicit = true`).
    pub force_world: Vec3,
    /// SPD 3x3 implicit tangent block `wn n n^T + wt (I - n n^T)` (`Mat3::ZERO` when `implicit = false`).
    pub k_tan_3x3: Mat3,
    pub mat_id: usize,
}

/// Evaluate exact $C^1$ surface contact force & SPD 3x3 implicit tangent for a sphere of radius
/// `node_radius` at `(pos, vel)` against any [`SdfField`] and [`MaterialLibrary`].
#[allow(clippy::too_many_arguments)]
pub fn eval_sdf_surface_contact(
    pos: Vec3,
    vel: Vec3,
    node_radius: f64,
    sdf: &dyn SdfField,
    materials: &MaterialLibrary,
    dt: f64,
    implicit: bool,
    k_scale: f64,
) -> SdfSurfaceContactEval {
    let q = sdf.query(pos);
    let pen = (node_radius - q.phi).max(0.0);
    if pen <= 0.0 {
        return SdfSurfaceContactEval {
            active: false,
            normal_force: 0.0,
            force_world: Vec3::ZERO,
            k_tan_3x3: Mat3::ZERO,
            mat_id: q.mat_id,
        };
    }
    let mat = materials.get(q.mat_id);
    let n = q.normal.normalized_or(Vec3::EZ);
    let vn = vel.dot(n);
    let kw = mat.wall_stiffness * k_scale;
    let cw = mat.wall_damping;
    let raw_fn = kw * pen - cw * vn;
    if raw_fn <= 0.0 {
        return SdfSurfaceContactEval {
            active: false,
            normal_force: 0.0,
            force_world: Vec3::ZERO,
            k_tan_3x3: Mat3::ZERO,
            mat_id: q.mat_id,
        };
    }

    let vt = vel - n * vn;
    let vt_mag = vt.norm();
    let v_reg = mat.wall_v_reg.max(1e-6);
    let th = (vt_mag / v_reg).tanh();
    let inv_vt = if vt_mag > 1e-12 {
        th / vt_mag
    } else {
        1.0 / v_reg
    };
    let f_fric = -vt * (mat.mu * raw_fn * inv_vt);
    let mut force_world = n * raw_fn + f_fric;

    let k_tan_3x3 = if implicit {
        let c_t_eff = (mat.mu * raw_fn / v_reg) * (1.0 - th * th).max(0.0);
        force_world += n * (cw * vn) + vt * c_t_eff;
        let wn = dt * cw + dt * dt * kw;
        let wt = dt * c_t_eff;
        let nn = n.outer(n);
        (nn * wn + (Mat3::IDENTITY - nn) * wt).symmetrize()
    } else {
        Mat3::ZERO
    };

    SdfSurfaceContactEval {
        active: true,
        normal_force: raw_fn,
        force_world,
        k_tan_3x3,
        mat_id: q.mat_id,
    }
}

#[inline]
fn signum_zero(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline]
fn fallback_tangent(n: Vec3) -> Vec3 {
    let a = if n.x().abs() < 0.9 { Vec3::EX } else { Vec3::EY };
    (a - n * a.dot(n)).normalized_or(Vec3::EX)
}
