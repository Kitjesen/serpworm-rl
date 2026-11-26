"""Rolling torque and wheel spin solved together, including zero-speed stiction."""
import torch

def contact(velocity,spin,memory,normal,dt,radius=.018,mu=.8,rolling=.06,wheel_mass=.009):
    inertia=.5*wheel_mass*radius**2;stiffness=2000.;damping=8.;viscous=2e-6
    gain=stiffness*dt+damping;base=-stiffness*memory-gain*velocity
    limit=rolling*normal*radius
    omega=((inertia*spin/dt-radius*base[...,0]-limit)/(inertia/dt+radius**2*gain+viscous)).clamp_min(0)
    force=base.clone();force[...,0]+=gain*radius*omega
    magnitude=torch.linalg.vector_norm(force,dim=-1);cap=mu*normal;sliding=magnitude>cap
    force=force*torch.minimum(torch.ones_like(magnitude),cap/magnitude.clamp_min(1e-15))[...,None]
    omega=((inertia*spin/dt-radius*force[...,0]-limit)/(inertia/dt+viscous)).clamp_min(0)
    # Combined axle torque (rolling resistance + one-way clutch). It transmits
    # load even at zero speed, without applying an unbalanced impulsive brake.
    reaction=inertia*(omega-spin)/dt+radius*force[...,0]
    slip=velocity.clone();slip[...,0]-=radius*omega
    memory=torch.where(sliding[...,None],-(force+damping*slip)/stiffness,memory+dt*slip)
    return force,omega,memory,reaction,slip

def old_contact(velocity,spin,memory,normal,dt,radius=.018,mu=.8,rolling=.06):
    inertia=.5*.009*radius**2;gain=2000*dt+8
    resistance=torch.minimum(rolling*normal*radius+2e-6*spin,inertia*spin/dt)
    base=-2000*memory-gain*velocity
    omega=((spin-dt*(radius*base[...,0]+resistance)/inertia)/(1+dt*radius**2*gain/inertia)).clamp_min(0)
    force=base.clone();force[...,0]+=gain*radius*omega
    magnitude=torch.linalg.vector_norm(force,dim=-1);cap=mu*normal;sliding=magnitude>cap
    force*=torch.minimum(torch.ones_like(magnitude),cap/magnitude.clamp_min(1e-15))[...,None]
    free=spin-dt*(radius*force[...,0]+resistance)/inertia;omega=free.clamp_min(0)
    reaction=inertia*(omega-free)/dt-resistance
    slip=velocity.clone();slip[...,0]-=radius*omega
    memory=torch.where(sliding[...,None],-(force+8*slip)/2000,memory+dt*slip)
    return force,omega,memory,reaction,slip


def arpc_normal_and_rut(
    pen,
    vz,
    v_horiz_speed,
    rut,
    dt,
    hc_k,
    hc_alpha,
    soil_exp,
    unload_ratio,
    rut_relax_m,
):
    """Smooth zero-impulse Hunt-Crossley normal force with elasto-plastic soil rut hysteresis.

    Loading branch:    F_el = hc_k * pen^p
    Unloading branch:  F_el = unload_ratio * hc_k * (pen - rut_rel) * pen^(p - 1)  (C^0 at pen = rut_rel / (1 - 1/gamma))
    Hunt-Crossley:     F_n  = F_el * max(0, 1 - hc_alpha * vz)
    Plastic rut state: rut_rel = rut * exp(-v_horiz_speed * dt / rut_relax_m),
                       new_rut = max(rut_rel, (1 - 1/unload_ratio) * max(pen, 0)) if unload_ratio > 1 else 0.

    Returns:
        (normal, new_rut, kn_tan, cn_tan) with kn_tan >= 0 and cn_tan >= 0 for implicit Contact-Schur coupling.
    """
    zeros = torch.zeros_like(pen)
    pen_pos = pen.clamp_min(0.0)
    ur_t = torch.as_tensor(unload_ratio, device=pen.device, dtype=pen.dtype)
    has_unload = ur_t > 1.0
    inv_relax = 1.0 / max(float(rut_relax_m), 1e-6)
    rut_rel_u = rut * torch.exp(-v_horiz_speed * dt * inv_relax)
    ur_safe = ur_t.clamp_min(1.0)
    plastic_frac = 1.0 - 1.0 / ur_safe
    new_rut_u = torch.maximum(rut_rel_u, plastic_frac * pen_pos)
    rut_rel = torch.where(has_unload, rut_rel_u, zeros)
    new_rut = torch.where(has_unload, new_rut_u, zeros)

    in_contact = (pen > rut_rel) & (pen > 0.0)
    pen_safe = torch.where(in_contact, pen, torch.ones_like(pen))
    pen_pm1 = torch.pow(pen_safe, soil_exp - 1.0)
    pen_pm2 = pen_pm1 / pen_safe

    virgin_el = hc_k * pen_safe * pen_pm1
    dpen = (pen_safe - rut_rel).clamp_min(0.0)
    unload_el = ur_t * hc_k * dpen * pen_pm1
    is_unload = has_unload & (unload_el < virgin_el)
    f_el = torch.where(in_contact, torch.where(is_unload, unload_el, virgin_el), zeros)
    df_unload = ur_t * hc_k * (pen_pm1 + (soil_exp - 1.0) * dpen * pen_pm2)
    df_virgin = soil_exp * hc_k * pen_pm1
    df_el = torch.where(in_contact, torch.where(is_unload, df_unload, df_virgin), zeros)

    damp_raw = 1.0 - hc_alpha * vz
    damp_pos = damp_raw > 0.0
    damp_mult = torch.where(damp_pos, damp_raw, zeros)

    normal = f_el * damp_mult
    active_tan = in_contact & damp_pos
    kn_tan = torch.where(active_tan, df_el * damp_mult, zeros)
    cn_tan = torch.where(active_tan, hc_alpha * f_el, zeros)
    return normal, new_rut, kn_tan, cn_tan


def arpc_tangential_contact(
    tangent_v,
    spin,
    memory,
    play,
    normal,
    rut,
    wz_plate,
    dt,
    radius=0.018,
    mu_fwd=0.8,
    mu_lat=0.8,
    rolling=0.06,
    backlash_rad=0.05,
    patch_radius=0.008,
    pivot_mu=0.65,
    rut_lat_k=0.0,
    static_mu_ratio=1.0,
    wheel_mass=0.009,
):
    """Anisotropic Ratchet-Patch Contact (ARPC) tangential force, backlash spin, Stribeck stiction,
    pivoting torque, and implicit tangents. Supports both scalar and per-env tensor parameters.

    Returns:
        (force_2d, omega, new_memory, new_play, reaction, slip, tau_z, ct_fwd_tan, ct_lat_tan, c_piv_tan)
    """
    inertia = 0.5 * wheel_mass * radius ** 2
    stiffness = 2000.0
    damping = 8.0
    viscous = 2e-6
    gain = stiffness * dt + damping
    inv_dt_inertia = inertia / dt
    denom_coup = inv_dt_inertia + (radius ** 2) * gain + viscous
    denom_sat = inv_dt_inertia + viscous

    zeros = torch.zeros_like(normal)
    bl_t = torch.as_tensor(backlash_rad, device=play.device, dtype=play.dtype)
    has_bl = bl_t > 0.0

    base = -stiffness * memory - gain * tangent_v
    limit = rolling * normal * radius
    omega_min = torch.where(has_bl, -play / dt, zeros)

    n_drive = inv_dt_inertia * spin - radius * base[..., 0]
    drive_net = torch.sign(n_drive) * (n_drive.abs() - limit).clamp_min(0.0)
    omega_unc = drive_net / denom_coup
    omega = torch.maximum(omega_unc, omega_min)

    f0 = base[..., 0] + gain * radius * omega
    f1 = base[..., 1]

    # Stribeck static-to-kinetic friction boost (branchless for CUDA Graph capture; exact 1.0 when ratio == 1.0)
    smr_t = torch.as_tensor(static_mu_ratio, device=normal.device, dtype=normal.dtype)
    smr_excess = (smr_t - 1.0).clamp_min(0.0)
    v_slip_est = torch.hypot(tangent_v[..., 0] - radius * omega, tangent_v[..., 1])
    stribeck = 1.0 + smr_excess * torch.exp(-(v_slip_est / 0.03).square())

    cap_fwd = mu_fwd * stribeck * normal
    cap_lat = mu_lat * stribeck * normal
    has_normal = (cap_fwd > 1e-15) & (cap_lat > 1e-15)
    u0 = torch.where(has_normal, f0 / cap_fwd.clamp_min(1e-15), zeros)
    u1 = torch.where(has_normal, f1 / cap_lat.clamp_min(1e-15), zeros)
    ell = torch.hypot(u0, u1)
    sliding = (~has_normal) | (ell > 1.0)
    scale = torch.where(has_normal, torch.minimum(torch.ones_like(ell), 1.0 / ell.clamp_min(1e-15)), zeros)

    f0 = f0 * scale
    f1 = f1 * scale

    n_drive_sat = inv_dt_inertia * spin - radius * f0
    drive_sat_net = torch.sign(n_drive_sat) * (n_drive_sat.abs() - limit).clamp_min(0.0)
    omega = torch.maximum(drive_sat_net / denom_sat, omega_min)

    reaction = inv_dt_inertia * (omega - spin) + radius * f0
    slip0 = tangent_v[..., 0] - radius * omega
    slip1 = tangent_v[..., 1]
    slip = torch.stack((slip0, slip1), dim=-1)

    mem0 = torch.where(sliding, -(f0 + damping * slip0) / stiffness, memory[..., 0] + dt * slip0)
    mem1 = torch.where(sliding, -(f1 + damping * slip1) / stiffness, memory[..., 1] + dt * slip1)
    new_memory = torch.stack((mem0, mem1), dim=-1)

    new_play = torch.where(has_bl, torch.minimum((play + omega * dt).clamp_min(0.0), bl_t), zeros)

    # Optional lateral rut-wall bulldozing resistance on soft soil (branchless for CUDA Graph capture)
    rlk_t = torch.as_tensor(rut_lat_k, device=normal.device, dtype=normal.dtype).clamp_min(0.0)
    th_lat = torch.tanh(tangent_v[..., 1] / 0.02)
    rut_active = torch.where(has_normal, rut, zeros)
    f_rut_lat = -(rlk_t * rut_active) * th_lat
    c_rut_lat = (rlk_t * rut_active / 0.02) * (1.0 - th_lat * th_lat).clamp_min(0.0)

    force_2d = torch.stack((f0, f1 + f_rut_lat), dim=-1)

    # Smooth rubber contact-patch vertical pivoting friction torque & positive rotational damping tangent
    pr_t = torch.as_tensor(patch_radius, device=normal.device, dtype=normal.dtype).clamp_min(0.0)
    piv_cap = (2.0 / 3.0) * max(float(pivot_mu), 0.0) * pr_t * normal
    th_wz = torch.tanh(wz_plate / 0.15)
    tau_z = -piv_cap * th_wz
    c_piv_tan = (piv_cap / 0.15) * (1.0 - th_wz * th_wz).clamp_min(0.0)

    # Positive tangential damping tangents for implicit Contact-Schur stabilization
    free_rolling = (n_drive.abs() > limit) & (omega_unc > omega_min)
    g_roll = gain * (denom_sat / denom_coup)
    g_fwd = torch.where(free_rolling, torch.full_like(normal, g_roll), torch.full_like(normal, gain))
    ct_fwd_tan = g_fwd * scale
    ct_lat_tan = gain * scale + c_rut_lat

    return force_2d, omega, new_memory, new_play, reaction, slip, tau_z, ct_fwd_tan, ct_lat_tan, c_piv_tan

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    report=[]
    for dt in (.0005,.00025,.000125):
        for name,fn in [('legacy',old_contact),('implicit',contact)]:
            # Under rolling threshold; above threshold; opposing ratchet;
            # contact Coulomb overload; passive coasting.
            drive=torch.tensor([.04,.12,-.04,-2.,0.]);velocity=torch.tensor([0.,0.,0.,0.,.1]);position=torch.zeros(5)
            spin=velocity/.018;memory=torch.zeros((5,2));normal=torch.ones(5)*1.4
            for _ in range(round(1.5/dt)):
                vv=torch.stack((velocity,torch.zeros_like(velocity)),-1)
                force,spin,memory,_,_=fn(vv,spin,memory,normal,dt)
                velocity+=dt*(drive+force[:,0])/.14;position+=dt*velocity
            record=dict(dt=dt,mode=name,velocity_m_s=velocity.tolist(),position_m=position.tolist(),spin_rad_s=spin.tolist())
            if name=='implicit':
                assert abs(velocity[0])<1e-8 and abs(velocity[2])<1e-8,record
                assert abs(position[0]-.04/2000)<1e-6,record
                assert velocity[1]>.01 and velocity[3]<-.1 and abs(velocity[4])<1e-5,record
            report.append(record)
    Path(__file__).with_name('wheel_stiction_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
