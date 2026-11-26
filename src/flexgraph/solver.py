"""Universal Lossless FP64 Bipartite Graph-Schur Rigid-Flexible Physics Solver.

Compiles ANY ``RigidFlexGraph`` + ``ComposableSDFWorld`` into a vectorized FP64 PyTorch solver
with zero robot-specific branches:
  - Step 1: Batched 6x6 Block-Thomas elimination per beam-discretization bucket
            (supports clamped-clamped, fixed-base, AND free cantilever beams ``body_b = -1``)
  - Step 2: Adaptive Rigid Schur solver automatically routed by ``graph.analyze_sparsity()``
            (``block_diagonal_pairs`` fast batched 12x12 LU/Cholesky, or exact ``6*N_free`` Schur solve)
  - Step 3: O(E) parallel back-substitution + exact SO(3) exponential map pose update

Supports:
  * Multi-waypoint tendons (crank, legacy_wrap, capstan, linear) with Capstan friction
  * Revolute / stop-gap / connector joints with static friction & DC torque-speed curves
  * Unified AnisotropicPatchContact (ratchet_wheel, anisotropic_scale, isotropic_pad)
  * Beam interior-node implicit SDF contact + optional Py-DiSMech C2 self-contact
  * Exact machine-precision (< 1e-10) parity with ``serpworm.physics.gpu_core.GPUCore``
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from flexgraph.contact import (
    CONTACT_MODE_MAP,
    ContactState,
    eval_anisotropic_patch_contact,
    eval_c2_self_contact,
)
from flexgraph.graph import RigidFlexGraph
from flexgraph.math3d import (
    beam_local_matrices,
    compute_section_properties,
    cross,
    midpoint_frame,
    norm,
    quat_conj,
    quat_exp_step,
    quat_log,
    quat_mul,
    quat_rotate,
    quat_to_mat,
    skew,
    unit,
)
from flexgraph.sdf_world import ComposableSDFWorld, HeightFieldSDF


class _BeamBucket:
    """Vectorized storage for all beams sharing the same element count E and endpoint type."""

    def __init__(
        self,
        n_elem: int,
        is_cantilever: bool,
        beam_indices: list[int],
        nodes: torch.Tensor,        # (S_b, E+1) global node IDs
        edges: torch.Tensor,        # (S_b, E, 2) global node ID pairs
        lengths: torch.Tensor,      # (S_b, E)
        transforms: torch.Tensor,   # (S_b, E, 2, 7)
        K_loc: torch.Tensor,        # (S_b, E, 12, 12)
        M_loc: torch.Tensor,        # (S_b, E, 12, 12)
        rayleigh_alpha: torch.Tensor,  # (S_b,)
        rayleigh_beta: torch.Tensor,   # (S_b,)
        body_a: torch.Tensor,       # (S_b,) rigid body index
        body_b: torch.Tensor,       # (S_b,) rigid body index (-1 if cantilever)
        interior_nodes: torch.Tensor,  # (S_b, n_int)
        contact_enabled: torch.Tensor, # (S_b,) bool
        contact_radius: torch.Tensor,  # (S_b,)
    ):
        self.E = n_elem
        self.is_cantilever = is_cantilever
        self.beam_indices = beam_indices
        self.S = len(beam_indices)
        self.nodes = nodes
        self.edges = edges
        self.lengths = lengths
        self.transforms = transforms
        self.K = K_loc
        self.M = M_loc
        self.rayleigh_alpha = rayleigh_alpha
        self.rayleigh_beta = rayleigh_beta
        self.body_a = body_a
        self.body_b = body_b
        self.interior_nodes = interior_nodes
        self.contact_enabled = contact_enabled
        self.contact_radius = contact_radius
        self.rest_local: torch.Tensor | None = None


class GraphSchurSolver:
    """Generic FP64 implicit bipartite rigid-flexible solver."""

    def __init__(
        self,
        graph: RigidFlexGraph,
        world: ComposableSDFWorld | None = None,
        *,
        batch: int = 1,
        dt: float = 0.001,
        device: str | torch.device = 'cpu',
        dtype: torch.dtype = torch.float64,
        implicit_contact: bool = True,
        use_kelvin_normal: bool = False,
        enable_self_contact: bool = False,
        self_contact_stiffness: float = 1500.0,
    ):
        self.graph = graph
        self.world = world if world is not None else ComposableSDFWorld(primitives=[HeightFieldSDF()])
        self.batch = int(batch)
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype
        self.implicit_contact = bool(implicit_contact)
        self.use_kelvin_normal = bool(use_kelvin_normal)
        self.enable_self_contact = bool(enable_self_contact)
        self.self_contact_stiffness = float(self_contact_stiffness)

        self.sparsity = graph.analyze_sparsity()
        self.route = self.sparsity.recommended_route
        self.P = len(graph.bodies)
        self.fixed_mask = torch.tensor([b.fixed for b in graph.bodies], device=self.device, dtype=torch.bool)
        self.free_body_ids = torch.where(~self.fixed_mask)[0]
        self.n_free = int(self.free_body_ids.numel())
        self.free_dof_ids = (self.free_body_ids[:, None] * 6 + torch.arange(6, device=self.device)[None, :]).reshape(-1)

        # Build global node array: [0..P-1] rigid bodies, [P..D-1] beam interior/cantilever nodes
        poses: list[list[float]] = [list(b.rest_pose_7d) for b in graph.bodies]
        beam_node_lists: list[list[int]] = []
        interior_beam_owner: list[int] = []
        for sid, beam in enumerate(graph.beams):
            nds = [int(beam.body_a)]
            n_e = int(beam.n_elem)
            if beam.body_b >= 0:
                for j in range(1, n_e):
                    nds.append(len(poses))
                    poses.append(list(beam.node_poses_7d[j]))
                    interior_beam_owner.append(sid)
                nds.append(int(beam.body_b))
            else:
                # Free cantilever tip: all nodes 1..n_e are free beam nodes!
                for j in range(1, n_e + 1):
                    nds.append(len(poses))
                    poses.append(list(beam.node_poses_7d[j]))
                    interior_beam_owner.append(sid)
            beam_node_lists.append(nds)

        self.D = len(poses)
        self.interior_beam_owner = torch.tensor(interior_beam_owner, device=self.device, dtype=torch.long)
        rest_t = torch.tensor(poses, device=self.device, dtype=self.dtype)
        self.rest = rest_t[None].repeat(self.batch, 1, 1)
        self.x = self.rest.clone()
        self.v = torch.zeros((self.batch, self.D, 6), device=self.device, dtype=self.dtype)

        g_vec = torch.tensor(self.world.gravity_m_s2, device=self.device, dtype=self.dtype)
        self.gravity = torch.zeros_like(self.v)
        self.gravity[..., :3] = g_vec

        # Rigid body mass matrices (P, 6, 6)
        pm = torch.zeros((self.P, 6, 6), device=self.device, dtype=self.dtype)
        eye3 = torch.eye(3, device=self.device, dtype=self.dtype)
        for i, bd in enumerate(graph.bodies):
            pm[i, :3, :3] = float(bd.mass_kg) * eye3
            pm[i, 3:, 3:] = torch.diag(torch.tensor(bd.inertia_diag, device=self.device, dtype=self.dtype))
        self.body_mass = pm[None].repeat(self.batch, 1, 1, 1)
        # Body rayleigh alpha = first beam's alpha or 0.05
        self.body_rayleigh_alpha = float(graph.beams[0].rayleigh_alpha) if graph.beams else 0.05

        # Group beams into vectorized buckets by (n_elem, is_cantilever)
        self.buckets: list[_BeamBucket] = []
        bucket_map: dict[tuple[int, bool], list[int]] = {}
        for sid, beam in enumerate(graph.beams):
            key = (int(beam.n_elem), beam.body_b < 0)
            bucket_map.setdefault(key, []).append(sid)

        for (n_e, is_cant), sids in sorted(bucket_map.items()):
            S_b = len(sids)
            nodes_arr = torch.tensor([beam_node_lists[s] for s in sids], device=self.device, dtype=torch.long)
            edges_arr = torch.stack((nodes_arr[:, :-1], nodes_arr[:, 1:]), dim=-1)  # (S_b, E, 2)
            lens_arr = torch.tensor([graph.beams[s].element_lengths_m for s in sids], device=self.device, dtype=self.dtype)
            tf_arr = torch.zeros((S_b, n_e, 2, 7), device=self.device, dtype=self.dtype)
            tf_arr[..., 6] = 1.0
            A_l, Iy_l, Iz_l, J_l, E_l, nu_l, rho_l, w_l, ts_l = [], [], [], [], [], [], [], [], []
            ra_l, rb_l, ba_l, bb_l, con_en, con_r, pf_l = [], [], [], [], [], [], []
            for bi, s in enumerate(sids):
                b = graph.beams[s]
                tf_arr[bi, 0, 0] = torch.tensor(b.clamp_a_7d, device=self.device, dtype=self.dtype)
                if not is_cant:
                    tf_arr[bi, -1, 1] = torch.tensor(b.clamp_b_7d, device=self.device, dtype=self.dtype)
                A_v, Iy_v, Iz_v, J_v = compute_section_properties(
                    b.section_type, b.width_m, b.thickness_m, b.radius_m, b.area_m2, b.iy_m4, b.iz_m4, b.j_m4
                )
                A_l.append(A_v)
                Iy_l.append(Iy_v)
                Iz_l.append(Iz_v)
                J_l.append(J_v)
                E_l.append(b.youngs_modulus_pa)
                nu_l.append(b.poisson_ratio)
                rho_l.append(b.density_kg_m3)
                w_l.append(b.width_m)
                ts_l.append(b.thin_shell_compat)
                ra_l.append(b.rayleigh_alpha)
                rb_l.append(b.rayleigh_beta)
                ba_l.append(b.body_a)
                bb_l.append(b.body_b)
                con_en.append(b.enable_node_contact)
                con_r.append(b.contact_node_radius_m)
                pf_l.append(b.prestress_frac)

            def _exp(vals, dt_override=None):
                t = torch.tensor(vals, device=self.device, dtype=dt_override or self.dtype)
                return t[:, None].expand(S_b, n_e)

            K_loc, M_loc = beam_local_matrices(
                lens_arr, _exp(A_l), _exp(Iy_l), _exp(Iz_l), _exp(J_l),
                _exp(E_l), _exp(nu_l), _exp(rho_l),
                thin_shell_compat=_exp(ts_l, torch.bool),
                width_m=_exp(w_l),
            )
            int_nodes = nodes_arr[:, 1:] if is_cant else nodes_arr[:, 1:-1]
            bkt = _BeamBucket(
                n_elem=n_e,
                is_cantilever=is_cant,
                beam_indices=sids,
                nodes=nodes_arr,
                edges=edges_arr,
                lengths=lens_arr,
                transforms=tf_arr,
                K_loc=K_loc,
                M_loc=M_loc,
                rayleigh_alpha=torch.tensor(ra_l, device=self.device, dtype=self.dtype),
                rayleigh_beta=torch.tensor(rb_l, device=self.device, dtype=self.dtype),
                body_a=torch.tensor(ba_l, device=self.device, dtype=torch.long),
                body_b=torch.tensor(bb_l, device=self.device, dtype=torch.long),
                interior_nodes=int_nodes,
                contact_enabled=torch.tensor(con_en, device=self.device, dtype=torch.bool),
                contact_radius=torch.tensor(con_r, device=self.device, dtype=self.dtype),
            )
            # Capture stress-free rest local frames (and optional flat-ribbon prestress blend)
            u_rest, _ = self._bucket_frames(bkt)
            pf_t = torch.tensor(pf_l, device=self.device, dtype=self.dtype)[:, None, None]
            if (pf_t > 0.0).any():
                flat = torch.zeros_like(u_rest)
                flat[..., 0] = -0.5 * lens_arr[None]
                flat[..., 6] = +0.5 * lens_arr[None]
                u_rest = (1.0 - pf_t[None]) * u_rest + pf_t[None] * flat
            bkt.rest_local = u_rest
            self.buckets.append(bkt)

        # Actuators state: unified vector (B, n_actuators)
        self.n_actuators = len(graph.actuators)
        self.act_state = torch.zeros((self.batch, self.n_actuators), device=self.device, dtype=self.dtype)
        self.act_target = torch.zeros((self.batch, self.n_actuators), device=self.device, dtype=self.dtype)
        self.tendon_tensions = torch.zeros((self.batch, len(graph.tendons)), device=self.device, dtype=self.dtype)
        self.joint_torques = torch.zeros((self.batch, len(graph.joints)), device=self.device, dtype=self.dtype)

        # Contact sites setup
        self.n_sites = len(graph.contact_sites)
        self.contact_state = ContactState.zeros(self.batch, self.n_sites, self.device, self.dtype)
        self.last_site_normal = torch.zeros((self.batch, self.n_sites), device=self.device, dtype=self.dtype)
        self.last_self_contact_energy = torch.zeros((self.batch,), device=self.device, dtype=self.dtype)
        if self.n_sites > 0:
            self._site_owner = torch.tensor([s.owner_body for s in graph.contact_sites], device=self.device, dtype=torch.long)
            self._site_lpos = torch.tensor([s.local_pos for s in graph.contact_sites], device=self.device, dtype=self.dtype)
            self._site_axle = torch.tensor([s.axle_local for s in graph.contact_sites], device=self.device, dtype=self.dtype)
            self._site_fwd = torch.tensor([s.fwd_local for s in graph.contact_sites], device=self.device, dtype=self.dtype)
            self._site_mode = torch.tensor([CONTACT_MODE_MAP[s.contact_mode] for s in graph.contact_sites], device=self.device, dtype=torch.long)
            self._site_rad = torch.tensor([s.radius_m for s in graph.contact_sites], device=self.device, dtype=self.dtype)
            self._site_prad = torch.tensor([s.patch_radius_m for s in graph.contact_sites], device=self.device, dtype=self.dtype)
            self._site_muf = torch.tensor([s.mu_fwd for s in graph.contact_sites], device=self.device, dtype=self.dtype)[None].repeat(self.batch, 1)
            self._site_mur = torch.tensor([s.mu_rev for s in graph.contact_sites], device=self.device, dtype=self.dtype)[None].repeat(self.batch, 1)
            self._site_mul = torch.tensor([s.mu_lat for s in graph.contact_sites], device=self.device, dtype=self.dtype)[None].repeat(self.batch, 1)
            self._site_roll = torch.tensor([s.rolling for s in graph.contact_sites], device=self.device, dtype=self.dtype)[None].repeat(self.batch, 1)
            self._site_wmass = torch.tensor([s.wheel_mass_kg for s in graph.contact_sites], device=self.device, dtype=self.dtype)
            self._site_backlash = torch.tensor([s.backlash_rad for s in graph.contact_sites], device=self.device, dtype=self.dtype)[None].repeat(self.batch, 1)
            self._site_statr = torch.tensor([s.static_mu_ratio for s in graph.contact_sites], device=self.device, dtype=self.dtype)[None].repeat(self.batch, 1)
            self._site_pivmu = torch.tensor([s.pivot_mu for s in graph.contact_sites], device=self.device, dtype=self.dtype)

        # Pre-pack material library tables on device for vectorized SDF lookup
        mats = self.world.materials
        self._mat_hc_k = torch.tensor([m.hc_k for m in mats], device=self.device, dtype=self.dtype)
        self._mat_hc_alpha = torch.tensor([m.hc_alpha for m in mats], device=self.device, dtype=self.dtype)
        self._mat_soil_exp = torch.tensor([m.soil_exp for m in mats], device=self.device, dtype=self.dtype)
        self._mat_unload = torch.tensor([m.unload_ratio for m in mats], device=self.device, dtype=self.dtype)
        self._mat_relax = torch.tensor([m.rut_relax_m for m in mats], device=self.device, dtype=self.dtype)
        self._mat_rlatk = torch.tensor([m.rut_lat_k for m in mats], device=self.device, dtype=self.dtype)
        self._mat_mu_scale = torch.tensor([m.mu_scale for m in mats], device=self.device, dtype=self.dtype)
        self._mat_lkn = torch.tensor([m.linear_kn for m in mats], device=self.device, dtype=self.dtype)
        self._mat_lcn = torch.tensor([m.linear_cn for m in mats], device=self.device, dtype=self.dtype)

        self.failed = torch.zeros((), device=self.device, dtype=torch.bool)

    def reset(self) -> None:
        self.x.copy_(self.rest)
        self.v.zero_()
        self.act_state.zero_()
        self.act_target.zero_()
        self.tendon_tensions.zero_()
        self.joint_torques.zero_()
        self.contact_state = ContactState.zeros(self.batch, self.n_sites, self.device, self.dtype)
        self.failed.zero_()

    def _bucket_frames(self, bkt: _BeamBucket) -> tuple[torch.Tensor, torch.Tensor]:
        xx = self.x[:, bkt.edges].reshape(self.batch, bkt.S, bkt.E, 2, 7)
        q = quat_mul(xx[..., 3:], bkt.transforms[..., 3:].expand(self.batch, -1, -1, -1, -1))
        offset = quat_rotate(xx[..., 3:], bkt.transforms[..., :3].expand(self.batch, -1, -1, -1, -1))
        p = xx[..., :3] + offset
        center, r = midpoint_frame(p, q, bkt.lengths)
        inv = quat_conj(r)[..., None, :].expand(-1, -1, -1, 2, -1)
        u = torch.cat((quat_rotate(inv, p - center[..., None, :]), quat_log(quat_mul(inv, q))), dim=-1).reshape(
            self.batch, bkt.S, bkt.E, 12
        )
        rt = quat_to_mat(r).transpose(-1, -2)[..., None, :, :].expand(-1, -1, -1, 2, -1, -1)
        zero = torch.zeros_like(rt)
        trans = torch.cat((torch.cat((rt, -rt @ skew(offset)), dim=-1), torch.cat((zero, rt), dim=-1)), dim=-2)
        T = torch.zeros((self.batch, bkt.S, bkt.E, 12, 12), device=self.device, dtype=self.dtype)
        T[..., :6, :6] = trans[..., 0, :, :]
        T[..., 6:, 6:] = trans[..., 1, :, :]
        return u, T

    def _accumulate_tendons(self, f: torch.Tensor, h: float) -> None:
        if not self.graph.tendons:
            return
        B = self.batch
        x_b = self.x[:, :self.P]
        v_b = self.v[:, :self.P]
        q_b = x_b[..., 3:]

        # Group tendons by actuator_idx so shared crank/capstan servo sums load across its tendon pair
        by_act: dict[int, list[int]] = {}
        for tid, ten in enumerate(self.graph.tendons):
            by_act.setdefault(ten.actuator_idx, []).append(tid)

        for act_id, tids in by_act.items():
            t0 = self.graph.tendons[tids[0]]
            sv = self.act_state[:, act_id]
            tgt = self.act_target[:, act_id]
            r, g = float(t0.horn_radius_m), float(t0.guide_dist_m)
            cap_mt = float(t0.capstan_mu_theta)

            if t0.drive_mode in ('crank', 'legacy_wrap'):
                chord = torch.sqrt((g * g + r * r - 2.0 * g * r * torch.cos(sv)).clamp_min(1e-18))
                if t0.drive_mode == 'crank':
                    lever = (r * g * torch.sin(sv).abs() / chord).clamp_max(r)
                else:
                    alpha = math.acos(min(1.0, max(-1.0, r / g)))
                    lever = torch.where(sv <= alpha, r * g * torch.sin(sv) / chord, torch.full_like(sv, r))
                load = self.tendon_tensions[:, tids].sum(-1) * lever * math.exp(cap_mt)
            elif t0.drive_mode == 'capstan':
                load = self.tendon_tensions[:, tids].sum(-1) * r * math.exp(cap_mt)
            else:  # 'linear'
                load = self.tendon_tensions[:, tids].sum(-1) * 0.01 * math.exp(cap_mt)

            s_max = float(t0.speed_limit_rad_s) * h
            raw_err = tgt - sv
            db = float(t0.deadband_rad)
            eff_err = torch.sign(raw_err) * (raw_err.abs() - db).clamp_min(0.0)
            inc = eff_err.clamp(-s_max, s_max)
            inc = inc * torch.where(inc > 0.0, (1.0 - load / max(float(t0.torque_limit_nm), 1e-9)).clamp(0.0, 1.0), 1.0)
            sv_vel = inc / h
            self.act_state[:, act_id] = sv + inc
            sv_new = self.act_state[:, act_id]

            if t0.drive_mode == 'crank':
                chord2 = torch.sqrt((g * g + r * r - 2.0 * g * r * torch.cos(sv_new)).clamp_min(1e-18))
                takeup = chord2 - (g - r)
            elif t0.drive_mode == 'legacy_wrap':
                chord2 = torch.sqrt((g * g + r * r - 2.0 * g * r * torch.cos(sv_new)).clamp_min(1e-18))
                alpha = math.acos(min(1.0, max(-1.0, r / g)))
                takeup = torch.where(
                    sv_new <= alpha, chord2, math.sqrt(max(g * g - r * r, 0.0)) + r * (sv_new - alpha)
                ) - (g - r)
            elif t0.drive_mode == 'capstan':
                takeup = r * sv_new
            else:
                takeup = sv_new  # linear stroke (m)

            pull_frac = torch.tanh(sv_vel.clamp_min(0.0) / 0.35)
            for tid in tids:
                ten = self.graph.tendons[tid]
                wps = ten.waypoints
                # Compute multi-waypoint path length & relative extension speed
                wp_pos = []
                wp_vel = []
                wp_arm = []
                for b_idx, l_xyz in wps:
                    lp = torch.tensor(l_xyz, device=self.device, dtype=self.dtype)[None].expand(B, 3)
                    arm = quat_rotate(q_b[:, b_idx], lp)
                    wp_arm.append(arm)
                    wp_pos.append(x_b[:, b_idx, :3] + arm)
                    wp_vel.append(v_b[:, b_idx, :3] + cross(v_b[:, b_idx, 3:], arm))

                total_len = torch.zeros((B,), device=self.device, dtype=self.dtype)
                total_spd = torch.zeros((B,), device=self.device, dtype=self.dtype)
                seg_dirs = []
                for k in range(len(wps) - 1):
                    delta = wp_pos[k + 1] - wp_pos[k]
                    seg_l = norm(delta)[..., 0]
                    d_hat = delta / seg_l[..., None].clamp_min(1e-15)
                    seg_dirs.append(d_hat)
                    total_len = total_len + seg_l
                    total_spd = total_spd + ((wp_vel[k + 1] - wp_vel[k]) * d_hat).sum(-1)

                rest_l = (float(ten.rest_length_m) + float(ten.slack_m)) - takeup
                ext = total_len - rest_l
                c_eff = float(ten.damping_ns_m) + 12.0 * float(ten.capstan_mu_theta)
                ten_raw = torch.where(ext >= 0.0, (float(ten.stiffness_n_m) * ext + c_eff * total_spd).clamp_min(0.0), 0.0)
                ten_eff = ten_raw * torch.exp(-float(ten.capstan_mu_theta) * pull_frac)
                self.tendon_tensions[:, tid] = ten_eff

                for k in range(len(wps) - 1):
                    ba, _ = wps[k]
                    bb, _ = wps[k + 1]
                    force_k = ten_eff[..., None] * seg_dirs[k]
                    f[:, ba, :3] = f[:, ba, :3] + force_k
                    f[:, bb, :3] = f[:, bb, :3] - force_k
                    f[:, ba, 3:] = f[:, ba, 3:] + cross(wp_arm[k], force_k)
                    f[:, bb, 3:] = f[:, bb, 3:] - cross(wp_arm[k + 1], force_k)

    def _accumulate_joints(
        self,
        f: torch.Tensor,
        h: float,
    ) -> list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        implicit_blocks: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if not self.graph.joints:
            return implicit_blocks
        B = self.batch
        x_b = self.x[:, :self.P]
        v_b = self.v[:, :self.P]
        q_b = x_b[..., 3:]
        R_b = quat_to_mat(q_b)
        I3 = torch.eye(3, device=self.device, dtype=self.dtype)[None].expand(B, 3, 3)

        for jid, jt in enumerate(self.graph.joints):
            a, b = jt.body_a, jt.body_b
            if jt.joint_type == 'stop_gap':
                sep = x_b[:, b, :3] - x_b[:, a, :3]
                axis = -R_b[:, a, :, 0]
                gap = (sep * axis).sum(-1)
                push = float(jt.k_stop) * (float(jt.stop_gap_m) - gap).clamp_min(0.0)[..., None] * axis
                f[:, b, :3] = f[:, b, :3] + push
                f[:, a, :3] = f[:, a, :3] - push
                f[:, a, 3:] = f[:, a, 3:] - cross(sep, push)
            elif jt.joint_type in ('connector_revolute', 'revolute'):
                arm_loc = torch.tensor(jt.anchor_a, device=self.device, dtype=self.dtype)[None].expand(B, 3)
                arm = quat_rotate(q_b[:, a], arm_loc)
                sep = x_b[:, b, :3] - x_b[:, a, :3]
                err = sep - arm
                relv = v_b[:, b, :3] - v_b[:, a, :3] - cross(v_b[:, a, 3:], sep)

                # Local hinge axis in body_a & body_b frames
                ax_loc = torch.tensor(jt.axis_local, device=self.device, dtype=self.dtype)
                ax_loc = ax_loc / ax_loc.norm().clamp_min(1e-12)
                axis_a = torch.einsum('bij,j->bi', R_b[:, a], ax_loc)
                axis_b = torch.einsum('bij,j->bi', R_b[:, b], ax_loc)
                R_rel = R_b[:, a].transpose(-1, -2) @ R_b[:, b]
                # Extract signed angle around ax_loc
                if abs(float(ax_loc[2])) > 0.9:
                    ang = torch.atan2(R_rel[:, 1, 0], R_rel[:, 0, 0])
                elif abs(float(ax_loc[1])) > 0.9:
                    ang = torch.atan2(R_rel[:, 0, 2], R_rel[:, 2, 2])
                else:
                    ang = torch.atan2(R_rel[:, 2, 1], R_rel[:, 1, 1])

                omega = v_b[:, b, 3:] - v_b[:, a, 3:]
                rate = (omega * axis_a).sum(-1)
                tgt = self.act_target[:, jt.actuator_idx] if jt.actuator_idx >= 0 else torch.zeros_like(ang)
                if jt.actuator_idx >= 0:
                    self.act_state[:, jt.actuator_idx] = ang

                raw_t = float(jt.kp) * (tgt - ang) - float(jt.kd) * rate
                sf = max(0.0, float(jt.tau_static_fric))
                tlim = float(jt.tau_max)
                slim = float(jt.omega_max)
                tau = (torch.sign(raw_t) * (raw_t.abs() - sf).clamp_min(0.0)).clamp(-tlim, tlim)
                if slim > 0.0:
                    cap = torch.where(tau * rate > 0.0, tlim * (1.0 - rate.abs() / slim).clamp(0.0, 1.0), tlim)
                    tau = torch.maximum(torch.minimum(tau, cap), -cap)
                else:
                    cap = torch.full_like(tau, tlim)
                self.joint_torques[:, jid] = tau

                if not jt.implicit_coupling:
                    f_pos = -float(jt.k_pos) * err - float(jt.c_pos) * relv
                    f[:, b, :3] = f[:, b, :3] + f_pos
                    f[:, a, :3] = f[:, a, :3] - f_pos
                    f[:, a, 3:] = f[:, a, 3:] - cross(sep, f_pos)
                    torque = (
                        tau[..., None] * axis_a
                        + float(jt.k_align) * cross(axis_b, axis_a)
                        - float(jt.c_align) * (omega - axis_a * rate[..., None])
                    )
                    f[:, b, 3:] = f[:, b, 3:] + torque
                    f[:, a, 3:] = f[:, a, 3:] - torque
                else:
                    # Unconditionally L-stable linearly-implicit backward Euler coupling:
                    # elastic terms enter RHS; linear damping & h*K stiffness enter globalA as exact SPD 12x12 blocks.
                    f_pos_el = -float(jt.k_pos) * err
                    f[:, b, :3] = f[:, b, :3] + f_pos_el
                    f[:, a, :3] = f[:, a, :3] - f_pos_el
                    f[:, a, 3:] = f[:, a, 3:] - cross(sep, f_pos_el)

                    raw_el = float(jt.kp) * (tgt - ang)
                    tau_el = (torch.sign(raw_el) * (raw_el.abs() - sf).clamp_min(0.0)).clamp(-tlim, tlim)
                    if slim > 0.0:
                        cap_el = torch.where(
                            tau_el * rate > 0.0, tlim * (1.0 - rate.abs() / slim).clamp(0.0, 1.0), tlim
                        )
                        tau_el = torch.maximum(torch.minimum(tau_el, cap_el), -cap_el)
                    torque_el = tau_el[..., None] * axis_a + float(jt.k_align) * cross(axis_b, axis_a)
                    f[:, b, 3:] = f[:, b, 3:] + torque_el
                    f[:, a, 3:] = f[:, a, 3:] - torque_el

                    # SPD position Jacobian J_p = [-I, S(sep), I, 0] => K_pos = h*(h*k_pos + c_pos) * J_p^T J_p
                    kp_eff = h * (h * float(jt.k_pos) + float(jt.c_pos))
                    S = skew(sep)  # (B, 3, 3)
                    K_aa = torch.zeros((B, 6, 6), device=self.device, dtype=self.dtype)
                    K_bb = torch.zeros((B, 6, 6), device=self.device, dtype=self.dtype)
                    K_ab = torch.zeros((B, 6, 6), device=self.device, dtype=self.dtype)
                    K_ba = torch.zeros((B, 6, 6), device=self.device, dtype=self.dtype)

                    K_aa[:, :3, :3] = kp_eff * I3
                    K_aa[:, :3, 3:] = -kp_eff * S
                    K_aa[:, 3:, :3] = kp_eff * S
                    K_aa[:, 3:, 3:] = -kp_eff * (S @ S)
                    K_bb[:, :3, :3] = kp_eff * I3
                    K_ab[:, :3, :3] = -kp_eff * I3
                    K_ab[:, 3:, :3] = -kp_eff * S
                    K_ba[:, :3, :3] = -kp_eff * I3
                    K_ba[:, :3, 3:] = kp_eff * S

                    # SPD rotational alignment + hinge PD tangent: K_rot = k_ax * aa^T + k_al * (I - aa^T)
                    aa = axis_a[..., :, None] * axis_a[..., None, :]
                    perp = I3 - aa
                    unsat = ((raw_t.abs() <= cap) & (raw_t.abs() > sf)).to(dtype=self.dtype)
                    k_ax = h * (float(jt.kd) + h * float(jt.kp) * unsat)
                    k_al = h * (h * float(jt.k_align) + float(jt.c_align))
                    K_rot = k_ax[:, None, None] * aa + k_al * perp
                    K_aa[:, 3:, 3:] = K_aa[:, 3:, 3:] + K_rot
                    K_bb[:, 3:, 3:] = K_bb[:, 3:, 3:] + K_rot
                    K_ab[:, 3:, 3:] = K_ab[:, 3:, 3:] - K_rot
                    K_ba[:, 3:, 3:] = K_ba[:, 3:, 3:] - K_rot
                    implicit_blocks.append((a, b, K_aa, K_bb, K_ab, K_ba))
        return implicit_blocks

    def _accumulate_site_contacts(self, f: torch.Tensor, h: float) -> torch.Tensor:
        """Evaluate unified AnisotropicPatchContact for all rigid-body contact sites against ComposableSDFWorld."""
        plate_K = torch.zeros((self.batch, self.P, 6, 6), device=self.device, dtype=self.dtype)
        if self.n_sites == 0:
            return plate_K
        B, M = self.batch, self.n_sites
        owner = self._site_owner
        q_o = self.x[:, owner, 3:]  # (B, M, 4)
        p_o = self.x[:, owner, :3]
        v_o = self.v[:, owner, :3]
        w_o = self.v[:, owner, 3:]

        offset = quat_rotate(q_o, self._site_lpos[None].expand(B, M, 3))
        axle_w = quat_rotate(q_o, self._site_axle[None].expand(B, M, 3))
        fwd_w = quat_rotate(q_o, self._site_fwd[None].expand(B, M, 3))

        # Query SDF at site center first to obtain outward normal, then project wheel rim along -normal
        q_center = self.world.query(p_o + offset)
        n_w = q_center.normal
        is_wheel = (self._site_mode == 0)[None, :]
        # Rim vector for wheels lies perpendicular to axle pointing along -normal_w; for pads along -normal_w
        rad_dir_wheel = unit(n_w - axle_w * (n_w * axle_w).sum(-1, keepdim=True))
        rim_dir = torch.where(is_wheel[..., None], rad_dir_wheel, n_w)
        rim_w = -self._site_rad[None, :, None] * rim_dir
        arm_w = offset + rim_w
        contact_pos = p_o + arm_w

        # Exact penetration at contact patch
        q_patch = self.world.query(contact_pos)
        pen = -q_patch.phi
        mat_id = q_patch.mat_id.clamp(0, len(self.world.materials) - 1)
        vel_site = v_o + cross(w_o, arm_w)

        f_w, t_w, K_6x6, normal_n, next_st = eval_anisotropic_patch_contact(
            pen=pen,
            normal_w=q_patch.normal,
            vel_site=vel_site,
            omega_owner=w_o,
            arm_w=arm_w,
            rim_w=torch.where(is_wheel[..., None], rim_w, torch.zeros_like(rim_w)),
            axle_w=axle_w,
            fwd_hint_w=fwd_w,
            mode_id=self._site_mode,
            radius_m=self._site_rad,
            patch_radius_m=self._site_prad,
            mu_fwd=self._site_muf * self._mat_mu_scale[mat_id],
            mu_rev=self._site_mur * self._mat_mu_scale[mat_id],
            mu_lat=self._site_mul * self._mat_mu_scale[mat_id],
            rolling=self._site_roll,
            wheel_mass_kg=self._site_wmass,
            backlash_rad=self._site_backlash,
            static_mu_ratio=self._site_statr,
            pivot_mu=self._site_pivmu,
            hc_k=self._mat_hc_k[mat_id],
            hc_alpha=self._mat_hc_alpha[mat_id],
            soil_exp=self._mat_soil_exp[mat_id],
            unload_ratio=self._mat_unload[mat_id],
            rut_relax_m=self._mat_relax[mat_id],
            rut_lat_k=self._mat_rlatk[mat_id],
            state=self.contact_state,
            dt=h,
            use_kelvin_normal=self.use_kelvin_normal,
        )
        self.contact_state = next_st
        self.last_site_normal = normal_n
        f.index_add_(1, owner, torch.cat((f_w, t_w), dim=-1))
        if self.implicit_contact:
            plate_K.index_add_(1, owner, K_6x6)
        return plate_K

    def _beam_node_contacts(self, h: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Implicit SDF wall/ground contact + optional Py-DiSMech C2 self-contact on all beam interior nodes."""
        n_int = self.D - self.P
        f_int = torch.zeros((self.batch, n_int, 3), device=self.device, dtype=self.dtype)
        K_int = torch.zeros((self.batch, n_int, 3, 3), device=self.device, dtype=self.dtype)
        if n_int == 0:
            return f_int, K_int

        # Per-bucket SDF contact on enabled beams
        for bkt in self.buckets:
            if not bkt.contact_enabled.any():
                continue
            ids = bkt.interior_nodes[bkt.contact_enabled].reshape(-1)
            r_nodes = bkt.contact_radius[bkt.contact_enabled][:, None].expand(
                -1, bkt.interior_nodes.shape[1]
            ).reshape(-1)
            pos = self.x[:, ids, :3]
            vel = self.v[:, ids, :3]
            q = self.world.query(pos)
            pen = (r_nodes[None, :] - q.phi).clamp_min(0.0)
            active = pen > 0.0
            if not active.any():
                continue
            mat_id = q.mat_id.clamp(0, len(self.world.materials) - 1)
            k_w = self._mat_lkn[mat_id] * 0.20
            c_w = 4.0
            vn = (vel * q.normal).sum(-1)
            fn = torch.where(active, (k_w * pen - c_w * vn).clamp_min(0.0), torch.zeros_like(pen))
            vt = vel - vn[..., None] * q.normal
            vt_n = norm(vt)[..., 0]
            mu_w = 0.25 * self._mat_mu_scale[mat_id]
            ft = -mu_w[..., None] * fn[..., None] * (vt / vt_n[..., None].clamp_min(0.015))
            f_tot = fn[..., None] * q.normal + ft
            kn_imp = torch.where(active, h * (h * k_w + c_w), torch.zeros_like(pen))
            k_mat = kn_imp[..., None, None] * (q.normal[..., :, None] * q.normal[..., None, :])
            rel_ids = ids - self.P
            f_int.index_add_(1, rel_ids, f_tot)
            K_int.index_add_(1, rel_ids, k_mat)

        # Optional Py-DiSMech C2 smooth self-contact between different beams
        if self.enable_self_contact and len(self.graph.beams) > 1:
            pos_all = self.x[:, self.P:, :3]
            f_sc, k_sc, e_sc = eval_c2_self_contact(
                pos_all,
                self.interior_beam_owner,
                radius_m=0.002,
                delta_m=0.002,
                stiffness_k=self.self_contact_stiffness,
                dt=h,
            )
            f_int = f_int + f_sc
            K_int = K_int + k_sc
            self.last_self_contact_energy = e_sc
        return f_int, K_int

    def _eliminate_bucket(
        self,
        bkt: _BeamBucket,
        A: torch.Tensor,         # (B, S_b, E, 12, 12)
        rhs: torch.Tensor,       # (B, D, 6)
        K_int_all: torch.Tensor, # (B, D-P, 3, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact batched 6x6 Block-Thomas elimination for clamped-clamped OR free-cantilever beams.

        Returns ``(z_sol, endA, endr)``:
          - Clamped-clamped: ``z_sol`` (B, S_b, E-1, 6, 13), ``endA`` (B, S_b, 12, 12), ``endr`` (B, S_b, 12)
          - Free cantilever: ``z_sol`` (B, S_b, E, 6, 7),    ``endA`` (B, S_b, 6, 6),   ``endr`` (B, S_b, 6)
        """
        B, S_b, E = self.batch, bkt.S, bkt.E
        rel_ids = bkt.interior_nodes - self.P
        node_K = K_int_all[:, rel_ids]  # (B, S_b, n_int, 3, 3)

        if not bkt.is_cantilever:
            n_int = E - 1
            diag = (A[:, :, :-1, 6:, 6:] + A[:, :, 1:, :6, :6]).clone()
            diag[..., :3, :3] = diag[..., :3, :3] + node_K
            lower = A[:, :, 1:-1, 6:, :6]
            upper = A[:, :, 1:-1, :6, 6:]
            r = torch.zeros((B, S_b, n_int, 6, 13), device=self.device, dtype=self.dtype)
            r[..., 0] = rhs[:, bkt.interior_nodes]
            r[:, :, 0, :, 1:7] = A[:, :, 0, 6:, :6]
            r[:, :, -1, :, 7:13] = A[:, :, -1, :6, 6:]

            ds, rs = [], []
            for i in range(n_int):
                d = diag[:, :, i]
                b_rhs = r[:, :, i]
                if i:
                    fac = lower[:, :, i - 1] @ ds[-1]
                    d = d - fac @ upper[:, :, i - 1]
                    b_rhs = b_rhs - fac @ rs[-1]
                inv, info = torch.linalg.inv_ex(d, check_errors=False)
                self.failed.logical_or_((info != 0).any())
                ds.append(inv)
                rs.append(b_rhs)
            sol = [None] * n_int
            sol[-1] = ds[-1] @ rs[-1]
            for i in range(n_int - 2, -1, -1):
                sol[i] = ds[i] @ (rs[i] - upper[:, :, i] @ sol[i + 1])
            z = torch.stack(sol, dim=2)  # (B, S_b, E-1, 6, 13)

            endA = torch.zeros((B, S_b, 12, 12), device=self.device, dtype=self.dtype)
            endA[..., :6, :6] = A[:, :, 0, :6, :6]
            endA[..., 6:, 6:] = A[:, :, -1, 6:, 6:]
            left = A[:, :, 0, :6, 6:]
            right = A[:, :, -1, 6:, :6]
            endA[..., :6, :] = endA[..., :6, :] - left @ z[:, :, 0, :, 1:]
            endA[..., 6:, :] = endA[..., 6:, :] - right @ z[:, :, -1, :, 1:]
            endr = -torch.cat(
                ((left @ z[:, :, 0, :, 0:1])[..., 0], (right @ z[:, :, -1, :, 0:1])[..., 0]),
                dim=-1,
            )
            return z, endA, endr

        # Free cantilever beam: interior nodes 1..E-1 plus free tip node E (total n_int = E)
        n_int = E
        diag = torch.empty((B, S_b, n_int, 6, 6), device=self.device, dtype=self.dtype)
        if E > 1:
            diag[:, :, :-1] = A[:, :, :-1, 6:, 6:] + A[:, :, 1:, :6, :6]
        diag[:, :, -1] = A[:, :, -1, 6:, 6:]
        diag[..., :3, :3] = diag[..., :3, :3] + node_K
        lower = A[:, :, 1:, 6:, :6]  # (B, S_b, E-1, 6, 6)
        upper = A[:, :, 1:, :6, 6:]  # (B, S_b, E-1, 6, 6)
        r = torch.zeros((B, S_b, n_int, 6, 7), device=self.device, dtype=self.dtype)
        r[..., 0] = rhs[:, bkt.interior_nodes]
        r[:, :, 0, :, 1:7] = A[:, :, 0, 6:, :6]

        ds, rs = [], []
        for i in range(n_int):
            d = diag[:, :, i]
            b_rhs = r[:, :, i]
            if i:
                fac = lower[:, :, i - 1] @ ds[-1]
                d = d - fac @ upper[:, :, i - 1]
                b_rhs = b_rhs - fac @ rs[-1]
            inv, info = torch.linalg.inv_ex(d, check_errors=False)
            self.failed.logical_or_((info != 0).any())
            ds.append(inv)
            rs.append(b_rhs)
        sol = [None] * n_int
        sol[-1] = ds[-1] @ rs[-1]
        for i in range(n_int - 2, -1, -1):
            sol[i] = ds[i] @ (rs[i] - upper[:, :, i] @ sol[i + 1])
        z = torch.stack(sol, dim=2)  # (B, S_b, E, 6, 7)

        left = A[:, :, 0, :6, 6:]
        endA = A[:, :, 0, :6, :6] - left @ z[:, :, 0, :, 1:]  # (B, S_b, 6, 6)
        endr = -(left @ z[:, :, 0, :, 0:1])[..., 0]           # (B, S_b, 6)
        return z, endA, endr

    def step(self, targets: torch.Tensor | None = None, steps: int = 1) -> None:
        """Advance batched rigid-flexible physics by ``steps`` implicit FP64 substeps."""
        if targets is not None:
            tgt = torch.as_tensor(targets, device=self.device, dtype=self.dtype)
            if tgt.ndim == 1:
                tgt = tgt[None].expand(self.batch, -1)
            self.act_target.copy_(tgt)
        for _ in range(int(steps)):
            self._substep()

    def _substep(self) -> None:
        h = self.dt
        B = self.batch
        f_ext = torch.zeros_like(self.v)
        self._accumulate_tendons(f_ext, h)
        joint_imp_blocks = self._accumulate_joints(f_ext, h)
        plate_K_con = self._accumulate_site_contacts(f_ext, h)

        rhs = h * f_ext
        # Rigid body inertia RHS
        vg_b = self.v[:, :self.P] + h * self.gravity[:, :self.P]
        rhs[:, :self.P] = rhs[:, :self.P] + (self.body_mass @ vg_b[..., None])[..., 0]

        # Beam interior node contacts (SDF + optional Py-DiSMech C2 self-contact)
        f_int_con, K_int_con = self._beam_node_contacts(h)
        if self.D > self.P:
            rhs[:, self.P:, :3] = rhs[:, self.P:, :3] + h * f_int_con

        # Assemble per-bucket beam elastic forces, K, M, and A
        bucket_A: list[torch.Tensor] = []
        for bkt in self.buckets:
            u, T = self._bucket_frames(bkt)
            Tt = T.transpose(-1, -2)
            elastic_e = -(Tt @ (bkt.K @ (u - bkt.rest_local)[..., None]))[..., 0]  # (B, S_b, E, 12)
            K_w = Tt @ bkt.K @ T
            M_w = Tt @ bkt.M @ T
            vg_e = (self.v + h * self.gravity)[:, bkt.edges].reshape(B, bkt.S, bkt.E, 12)
            rhs_e = (M_w @ vg_e[..., None])[..., 0] + h * elastic_e
            rhs.index_add_(1, bkt.edges.reshape(-1), rhs_e.reshape(B, -1, 6))
            ra = bkt.rayleigh_alpha[None, :, None, None, None]
            rb = bkt.rayleigh_beta[None, :, None, None, None]
            A_b = (1.0 + h * ra) * M_w + (h * rb + h * h) * K_w
            bucket_A.append(A_b)

        # Stage 1: Eliminate all beam interior nodes via batched 6x6 Block-Thomas
        P6 = 6 * self.P
        globalA = torch.zeros((B, P6, P6), device=self.device, dtype=self.dtype)
        globalr = rhs[:, :self.P].reshape(B, P6).clone()

        for p in range(self.P):
            globalA[:, 6 * p:6 * p + 6, 6 * p:6 * p + 6] = (
                (1.0 + h * self.body_rayleigh_alpha) * self.body_mass[:, p] + plate_K_con[:, p]
            )
        for a_id, b_id, K_aa, K_bb, K_ab, K_ba in joint_imp_blocks:
            sa, sb = 6 * a_id, 6 * b_id
            globalA[:, sa:sa + 6, sa:sa + 6] = globalA[:, sa:sa + 6, sa:sa + 6] + K_aa
            globalA[:, sb:sb + 6, sb:sb + 6] = globalA[:, sb:sb + 6, sb:sb + 6] + K_bb
            globalA[:, sa:sa + 6, sb:sb + 6] = globalA[:, sa:sa + 6, sb:sb + 6] + K_ab
            globalA[:, sb:sb + 6, sa:sa + 6] = globalA[:, sb:sb + 6, sa:sa + 6] + K_ba

        elim_cache = []
        for bkt, A_b in zip(self.buckets, bucket_A):
            z, endA, endr = self._eliminate_bucket(bkt, A_b, rhs, K_int_con)
            elim_cache.append((bkt, z))
            if not bkt.is_cantilever:
                dofs = torch.stack((bkt.body_a, bkt.body_b), dim=-1)[..., None] * 6 + torch.arange(
                    6, device=self.device
                )
                dofs = dofs.reshape(bkt.S, 12)
                scat_m = (dofs[:, :, None] * P6 + dofs[:, None, :]).reshape(-1)
                globalA.reshape(B, P6 * P6).index_add_(1, scat_m, endA.reshape(B, -1))
                globalr.index_add_(1, dofs.reshape(-1), endr.reshape(B, -1))
            else:
                dofs = (bkt.body_a[:, None] * 6 + torch.arange(6, device=self.device)[None, :])  # (S_b, 6)
                scat_m = (dofs[:, :, None] * P6 + dofs[:, None, :]).reshape(-1)
                globalA.reshape(B, P6 * P6).index_add_(1, scat_m, endA.reshape(B, -1))
                globalr.index_add_(1, dofs.reshape(-1), endr.reshape(B, -1))

        # Stage 2: Adaptive Rigid Schur Solve (respecting Dirichlet fixed bodies)
        vp = torch.zeros((B, P6), device=self.device, dtype=self.dtype)
        if self.n_free > 0:
            if self.route == 'block_diagonal_pairs':
                n_pairs = self.P // 2
                pair_ids = torch.arange(n_pairs, device=self.device)
                blocks = globalA.reshape(B, n_pairs, 12, n_pairs, 12).permute(0, 1, 3, 2, 4)
                local = blocks[:, pair_ids, pair_ids].reshape(B * n_pairs, 12, 12)
                sol_p, info = torch.linalg.solve_ex(local, globalr.reshape(B * n_pairs, 12, 1), check_errors=False)
                self.failed.logical_or_((info != 0).any())
                vp = sol_p.reshape(B, P6)
            else:
                # Exact Schur solve over all free rigid DoFs (fixed bodies pinned identically to 0 velocity)
                idx = self.free_dof_ids
                A_ff = globalA[:, idx][:, :, idx]
                r_ff = globalr[:, idx, None]
                sol_f, info = torch.linalg.solve_ex(A_ff, r_ff, check_errors=False)
                self.failed.logical_or_((info != 0).any())
                vp[:, idx] = sol_f[..., 0]

        # Stage 3: Back-substitute rigid velocities to all beam interior nodes + SO(3) exponential update
        vnew = torch.zeros_like(self.v)
        vnew[:, :self.P] = vp.reshape(B, self.P, 6)
        for bkt, z in elim_cache:
            if not bkt.is_cantilever:
                dofs = (
                    torch.stack((bkt.body_a, bkt.body_b), dim=-1)[..., None] * 6
                    + torch.arange(6, device=self.device)
                ).reshape(bkt.S, 12)
                vp_end = vp[:, dofs][..., None, :, None]  # (B, S_b, 1, 12, 1)
            else:
                dofs = bkt.body_a[:, None] * 6 + torch.arange(6, device=self.device)[None, :]
                vp_end = vp[:, dofs][..., None, :, None]  # (B, S_b, 1, 6, 1)
            vi = z[..., 0] - (z[..., 1:] @ vp_end)[..., 0]
            vnew[:, bkt.interior_nodes] = vi

        # Pin fixed bodies strictly to rest
        if self.fixed_mask.any():
            vnew[:, :self.P] = torch.where(self.fixed_mask[None, :, None], torch.zeros_like(vnew[:, :self.P]), vnew[:, :self.P])

        self.v.copy_(vnew)
        self.x[..., :3].add_(h * vnew[..., :3])
        self.x[..., 3:].copy_(quat_exp_step(self.x[..., 3:], vnew[..., 3:], h))
        if self.fixed_mask.any():
            self.x[:, :self.P] = torch.where(
                self.fixed_mask[None, :, None], self.rest[:, :self.P], self.x[:, :self.P]
            )
