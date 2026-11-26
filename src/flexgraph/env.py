"""Unified Morphology-Agnostic Closed-Loop RL Environment & Universal SafetyGuard (Layers 4-5).

Bridges ANY ``RigidFlexGraph`` + ``ComposableSDFWorld`` + ``GraphSchurSolver`` / ``RustGraphSolverBackend``
directly to ``TokenizedHIMDreamWaQ`` / ``FiniteGroupEquivariantWrapper`` with:
  - Automatic entity-token construction ``(B, H, N_act, 8)`` including universal software-torque ``q_err``
  - Global body IMU + velocity-command history ``(B, H, 9)``
  - Morphology-invariant 16D privileged critic summary & 7D EstNet ground-truth targets
  - Vectorized partial resets (``reset_done``), NaN/Inf sanitization, and clean Gymnasium-like batch API
  - Live actuator-fault injection (freezes failed actuators & updates ``active_mask``)
  - Universal SDF arc-length progress telemetry & ``UniversalSafetyGuard`` deployment manifest
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, NamedTuple

import torch

from flexgraph.graph import RigidFlexGraph
from flexgraph.math3d import quat_conj, quat_rotate
from flexgraph.rust_backend import create_solver
from flexgraph.sdf_world import ComposableSDFWorld


@dataclass
class SafetyConfig:
    max_tilt_rad: float = 1.15
    max_cmd_delta: float = 0.50
    max_step_timeout_ms: float = 100.0
    estop_on_nonfinite: bool = True


class EnvStepOut(NamedTuple):
    """Structured vectorized transition returned by ``FlexGraphEnv.step_batch``."""

    obs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (tokens, glob, active_mask) after optional auto-reset
    priv: torch.Tensor                 # (B, 16) privileged critic state after optional auto-reset
    est_target: torch.Tensor           # (B, 7) EstNet ground-truth target at pre-reset next state
    next_glob_sens: torch.Tensor       # (B, 6) next-step IMU [omega_b, grav_b] at pre-reset next state
    reward: torch.Tensor               # (B,) FP32 reward
    terminated: torch.Tensor           # (B,) bool physical/nonfinite failure termination
    truncated: torch.Tensor            # (B,) bool time-limit truncation
    done: torch.Tensor                 # (B,) bool terminated | truncated
    terminal_tokens: torch.Tensor      # (B, H, N_act, 8) pre-reset next tokens (for GAE bootstrap)
    terminal_glob: torch.Tensor        # (B, H, 9) pre-reset next glob (for GAE bootstrap)
    terminal_priv: torch.Tensor        # (B, 16) pre-reset next priv (for GAE bootstrap)
    info: dict[str, Any]


class UniversalSafetyGuard:
    """Hardware-agnostic safety monitor enforcing tilt, finiteness, slew-rate, and E-stop latching."""

    def __init__(self, cfg: SafetyConfig | None = None):
        self.cfg = cfg or SafetyConfig()
        self.estop_latched: bool = False
        self.estop_reason: str = ''

    def reset(self) -> None:
        self.estop_latched = False
        self.estop_reason = ''

    def filter_action(
        self,
        action: torch.Tensor,
        prev_u: torch.Tensor,
        grav_body: torch.Tensor,
        elapsed_ms: float = 0.0,
    ) -> torch.Tensor:
        if self.estop_latched:
            return torch.zeros_like(prev_u)
        if self.cfg.estop_on_nonfinite and (not torch.isfinite(action).all() or not torch.isfinite(grav_body).all()):
            self.estop_latched = True
            self.estop_reason = 'nonfinite_signal'
            return torch.zeros_like(prev_u)
        if elapsed_ms > self.cfg.max_step_timeout_ms:
            self.estop_latched = True
            self.estop_reason = f'timeout_{elapsed_ms:.1f}ms'
            return torch.zeros_like(prev_u)
        # Body gravity z component in [-1, 1]; upright is -1.0 -> tilt = acos(-gz)
        gz = (-grav_body[..., 2]).clamp(-1.0, 1.0)
        tilt = torch.acos(gz)
        if float(tilt.max().item()) > self.cfg.max_tilt_rad:
            self.estop_latched = True
            self.estop_reason = 'tilt_exceeded'
            return torch.zeros_like(prev_u)
        delta = (action - prev_u).clamp(-self.cfg.max_cmd_delta, self.cfg.max_cmd_delta)
        return (prev_u + delta).clamp(-1.0, 1.0)


class FlexGraphEnv:
    """Morphology-agnostic batched proprioceptive RL environment wrapping ANY RigidFlexGraph."""

    PRIV_DIM: int = 16
    EST_DIM: int = 7

    def __init__(
        self,
        graph: RigidFlexGraph,
        world: ComposableSDFWorld | None = None,
        *,
        batch: int = 1,
        dt: float = 0.001,
        substeps_per_policy: int = 20,
        history_len: int = 16,
        device: str | torch.device = 'cpu',
        dtype: torch.dtype = torch.float64,
        implicit_contact: bool = True,
        enable_self_contact: bool = False,
        backend: str = 'python',
        backend_lib_path: str | Path | None = None,
        max_episode_steps: int = 200,
        max_tilt_rad: float = 1.35,
        auto_reset: bool = False,
        return_done: bool = False,
    ):
        self.graph = graph
        self.solver = create_solver(
            graph,
            world=world,
            backend=backend,
            lib_path=backend_lib_path,
            batch=batch,
            dt=dt,
            device=device,
            dtype=dtype,
            implicit_contact=implicit_contact,
            enable_self_contact=enable_self_contact,
        )
        self.batch = int(batch)
        self.substeps_per_policy = int(substeps_per_policy)
        self.policy_dt = float(dt) * self.substeps_per_policy
        self.H = int(history_len)
        self.N_act = len(graph.actuators)
        self.device = self.solver.device
        self.dtype = dtype
        self.max_episode_steps = int(max_episode_steps)
        self.max_tilt_rad = float(max_tilt_rad)
        self.auto_reset = bool(auto_reset)
        self.return_done = bool(return_done)
        self.active_backend = getattr(self.solver, 'active_backend', 'python')

        # Per-actuator static morphology metadata: [seg_norm, side, is_tendon, is_joint]
        meta = []
        lo, hi = [], []
        for a in graph.actuators:
            meta.append(
                [a.segment_norm, a.side, 1.0 if a.act_type == 'tendon' else 0.0, 1.0 if a.act_type == 'joint' else 0.0]
            )
            lo.append(a.min_cmd)
            hi.append(a.max_cmd)
        self._meta = torch.tensor(meta, device=self.device, dtype=torch.float32)  # (N_act, 4)
        self._cmd_lo = torch.tensor(lo, device=self.device, dtype=dtype)
        self._cmd_hi = torch.tensor(hi, device=self.device, dtype=dtype)
        self._cmd_mid = 0.5 * (self._cmd_lo + self._cmd_hi)
        self._cmd_half = (0.5 * (self._cmd_hi - self._cmd_lo)).clamp_min(1e-6)

        # Precompute site side signs (+1 left y>0, -1 right y<0) for wall_side EstNet target
        if graph.contact_sites:
            s_sides = [
                1.0 if s.local_pos[1] > 1e-4 else (-1.0 if s.local_pos[1] < -1e-4 else 0.0)
                for s in graph.contact_sites
            ]
            self._site_side_sign = torch.tensor(s_sides, device=self.device, dtype=dtype)
        else:
            self._site_side_sign = torch.zeros((0,), device=self.device, dtype=dtype)

        self._total_mass_kg = max(sum(float(b.mass_kg) for b in graph.bodies if not b.fixed), 1e-3)
        self._has_fixed_base = any(b.fixed for b in graph.bodies)
        self._g_world_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=dtype)

        # Rolling observation histories (FP32 for fast neural policy consumption)
        self.tok_hist = torch.zeros((self.batch, self.H, self.N_act, 8), device=self.device, dtype=torch.float32)
        self.glob_hist = torch.zeros((self.batch, self.H, 9), device=self.device, dtype=torch.float32)
        self.u_prev = torch.zeros((self.batch, self.N_act), device=self.device, dtype=dtype)
        self.a_prev = torch.zeros((self.batch, self.N_act), device=self.device, dtype=dtype)
        self.q_prev = torch.zeros((self.batch, self.N_act), device=self.device, dtype=dtype)
        self.dq_ema = torch.zeros((self.batch, self.N_act), device=self.device, dtype=dtype)
        self.cmd = torch.zeros((self.batch, 3), device=self.device, dtype=dtype)
        self.cmd[:, 0] = 1.0  # default forward command
        self.active_mask = torch.ones((self.batch, self.N_act), device=self.device, dtype=torch.bool)
        self.step_count = torch.zeros((self.batch,), device=self.device, dtype=torch.long)
        self._prev_arc_s = torch.zeros((self.batch,), device=self.device, dtype=dtype)
        self._last_v_arc = torch.zeros((self.batch,), device=self.device, dtype=dtype)
        self._imu_body_idx = 0 if len(graph.bodies) == 1 else (1 if graph.bodies[0].fixed else 0)
        self._tip_body_idx = self.solver.P - 1

    def set_failed_actuators(self, active_mask: torch.Tensor) -> None:
        """Disable arbitrary actuators (False = jammed/unpowered) for live fault-tolerance tests."""
        mask = active_mask.to(device=self.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask[None].expand(self.batch, -1).clone()
        self.active_mask = mask

    def sample_actuator_faults(
        self,
        fault_prob: float,
        max_faults: int = 1,
        env_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample random actuator fault masks (keeping >= 2 healthy actuators per env)."""
        if env_mask is None:
            env_mask = torch.ones((self.batch,), device=self.device, dtype=torch.bool)
        else:
            env_mask = env_mask.to(device=self.device, dtype=torch.bool)
        if not bool(env_mask.any().item()):
            return self.active_mask.clone()

        self.active_mask[env_mask] = True
        max_allowed = min(int(max_faults), max(0, self.N_act - 2))
        if fault_prob <= 0.0 or max_allowed <= 0:
            return self.active_mask.clone()

        ids = torch.where(env_mask)[0].tolist()
        for b in ids:
            r = float(torch.rand((), generator=generator).item())
            if r < float(fault_prob):
                k = 1 if max_allowed == 1 else int(torch.randint(1, max_allowed + 1, (), generator=generator).item())
                perm = torch.randperm(self.N_act, generator=generator)[:k]
                self.active_mask[b, perm] = False
        return self.active_mask.clone()

    def _query_com_arc(self) -> torch.Tensor:
        free_idx = self.solver.free_body_ids if self.solver.n_free > 0 else torch.arange(self.solver.P, device=self.device)
        com = self.solver.x[:, free_idx, :3].mean(dim=1, keepdim=True)  # (B, 1, 3)
        arc = self.solver.world.query(com).arc_s[:, 0]
        return torch.nan_to_num(arc, nan=0.0, posinf=0.0, neginf=0.0)

    def _imu_kinematics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = self.batch
        ib = self._imu_body_idx
        q_imu = self.solver.x[:, ib, 3:]
        v_imu = self.solver.v[:, ib, :3]
        w_imu = self.solver.v[:, ib, 3:]
        q_inv = quat_conj(q_imu)
        vel_b = quat_rotate(q_inv, v_imu)
        omega_b = quat_rotate(q_inv, w_imu)
        grav_b = quat_rotate(q_inv, self._g_world_unit[None].expand(B, 3))
        return (
            torch.nan_to_num(vel_b, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(omega_b, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(grav_b, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _build_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = self.batch, self.N_act
        q_norm = ((self.solver.act_state - self._cmd_mid) / self._cmd_half).clamp(-2.0, 2.0)
        q_err_norm = ((self.solver.act_target - self.solver.act_state) / self._cmd_half).clamp(-2.0, 2.0)
        dq_norm = (self.dq_ema / 6.0).clamp(-2.0, 2.0)
        dyn = torch.stack((q_norm, dq_norm, q_err_norm, self.u_prev), dim=-1).float()  # (B, N, 4)
        dyn = torch.nan_to_num(dyn, nan=0.0, posinf=2.0, neginf=-2.0)
        meta_exp = self._meta[None].expand(B, N, 4)
        tok_frame = torch.cat((meta_exp, dyn), dim=-1)  # (B, N, 8)

        _, omega_b, grav_b = self._imu_kinematics()
        glob_frame = torch.cat((omega_b.clamp(-20.0, 20.0), grav_b.clamp(-1.0, 1.0), self.cmd), dim=-1).float()  # (B, 9)
        glob_frame = torch.nan_to_num(glob_frame, nan=0.0, posinf=0.0, neginf=0.0)
        return tok_frame, glob_frame

    def get_est_target(self) -> torch.Tensor:
        """Compute 7D morphology-agnostic EstNet ground-truth targets ``[vx, vy, wz, wall_conf, wall_side, drag, slip]``."""
        rust_summary = getattr(self.solver, 'last_contact_summary', None)
        if getattr(self.solver, 'active_backend', 'python') == 'rust' and rust_summary is not None:
            return torch.nan_to_num(rust_summary.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        vel_b, omega_b, _ = self._imu_kinematics()
        vx = vel_b[:, 0].clamp(-3.0, 3.0)
        vy = vel_b[:, 1].clamp(-3.0, 3.0)
        wz = omega_b[:, 2].clamp(-6.0, 6.0)

        weight_n = self._total_mass_kg * 9.81
        if self.solver.n_sites > 0:
            normals = torch.nan_to_num(self.solver.last_site_normal, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            n_sum = normals.sum(dim=-1)
            wall_conf = torch.tanh(n_sum / max(weight_n, 1e-3))
            side_net = (normals * self._site_side_sign[None, :]).sum(dim=-1)
            wall_side = (side_net / n_sum.clamp_min(1e-3)).clamp(-1.0, 1.0)
        else:
            wall_conf = torch.zeros((self.batch,), device=self.device, dtype=self.dtype)
            wall_side = torch.zeros((self.batch,), device=self.device, dtype=self.dtype)

        ten_load = (
            self.solver.tendon_tensions.abs().mean(dim=-1) / 40.0
            if self.solver.tendon_tensions.numel() > 0
            else torch.zeros((self.batch,), device=self.device, dtype=self.dtype)
        )
        jnt_load = (
            self.solver.joint_torques.abs().mean(dim=-1) / 2.5
            if self.solver.joint_torques.numel() > 0
            else torch.zeros((self.batch,), device=self.device, dtype=self.dtype)
        )
        drag = torch.tanh(ten_load + jnt_load).clamp(0.0, 1.0)

        q_err = ((self.solver.act_target - self.solver.act_state) / self._cmd_half).abs().mean(dim=-1)
        slip = torch.tanh(0.5 * q_err).clamp(0.0, 1.0)
        out = torch.stack((vx, vy, wz, wall_conf, wall_side, drag, slip), dim=-1).float()
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

    def get_privileged_obs(self) -> torch.Tensor:
        """Compute fixed-width 16D morphology-invariant privileged critic observation."""
        vel_b, omega_b, grav_b = self._imu_kinematics()
        free_idx = self.solver.free_body_ids if self.solver.n_free > 0 else torch.arange(self.solver.P, device=self.device)
        z_com = self.solver.x[:, free_idx, 2].mean(dim=1).clamp(-2.0, 5.0)
        upright = (-grav_b[:, 2]).clamp(-1.0, 1.0)

        weight_n = self._total_mass_kg * 9.81
        if self.solver.n_sites > 0:
            normals = torch.nan_to_num(self.solver.last_site_normal, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
            n_norm = (normals.sum(dim=-1) / max(weight_n, 1e-3)).clamp(0.0, 10.0)
            contact_frac = (normals > 0.05).to(dtype=self.dtype).mean(dim=-1)
        else:
            n_norm = torch.zeros((self.batch,), device=self.device, dtype=self.dtype)
            contact_frac = torch.zeros((self.batch,), device=self.device, dtype=self.dtype)

        ten_mean = (
            (self.solver.tendon_tensions.abs().mean(dim=-1) / 50.0).clamp(0.0, 10.0)
            if self.solver.tendon_tensions.numel() > 0
            else torch.zeros((self.batch,), device=self.device, dtype=self.dtype)
        )
        jnt_mean = (
            (self.solver.joint_torques.abs().mean(dim=-1) / 3.0).clamp(0.0, 10.0)
            if self.solver.joint_torques.numel() > 0
            else torch.zeros((self.batch,), device=self.device, dtype=self.dtype)
        )
        v_arc = self._last_v_arc.clamp(-5.0, 5.0)
        act_frac = self.active_mask.to(dtype=self.dtype).mean(dim=-1)
        fixed_flag = torch.full((self.batch,), 1.0 if self._has_fixed_base else 0.0, device=self.device, dtype=self.dtype)
        sc_e = torch.nan_to_num(self.solver.last_self_contact_energy, nan=0.0).clamp(0.0, 10.0)

        priv = torch.cat(
            (
                vel_b.clamp(-5.0, 5.0),
                omega_b.clamp(-15.0, 15.0),
                torch.stack(
                    (z_com, upright, n_norm, contact_frac, ten_mean, jnt_mean, v_arc, act_frac, fixed_flag, sc_e),
                    dim=-1,
                ),
            ),
            dim=-1,
        ).float()
        return torch.nan_to_num(priv, nan=0.0, posinf=0.0, neginf=0.0)

    def _reset_solver_subset(self, mask: torch.Tensor) -> None:
        if hasattr(self.solver, 'reset_envs'):
            self.solver.reset_envs(mask)
            return
        if bool(mask.all().item()):
            self.solver.reset()
            return
        s = self.solver
        s.x[mask] = s.rest[mask]
        s.v[mask] = 0.0
        s.act_state[mask] = 0.0
        s.act_target[mask] = 0.0
        s.tendon_tensions[mask] = 0.0
        s.joint_torques[mask] = 0.0
        s.last_site_normal[mask] = 0.0
        s.last_self_contact_energy[mask] = 0.0
        s.contact_state.rut[mask] = 0.0
        s.contact_state.memory[mask] = 0.0
        s.contact_state.spin[mask] = 0.0
        s.contact_state.angle[mask] = 0.0
        s.contact_state.ratchet_play[mask] = 0.0
        if bool(torch.isfinite(s.x).all().item()) and bool(torch.isfinite(s.v).all().item()):
            s.failed.zero_()

    def reset(self, cmd: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.solver.reset()
        self.u_prev.zero_()
        self.a_prev.zero_()
        self.q_prev.copy_(self.solver.act_state)
        self.dq_ema.zero_()
        self.step_count.zero_()
        self._last_v_arc.zero_()
        if cmd is not None:
            c = torch.as_tensor(cmd, device=self.device, dtype=self.dtype)
            if c.ndim == 1:
                c = c[None].expand(self.batch, -1)
            self.cmd.copy_(c)
        self._prev_arc_s = self._query_com_arc()
        tf, gf = self._build_frame()
        self.tok_hist.copy_(tf[:, None].expand(-1, self.H, -1, -1))
        self.glob_hist.copy_(gf[:, None].expand(-1, self.H, -1))
        return self.tok_hist.clone(), self.glob_hist.clone(), self.active_mask.clone()

    def reset_done(
        self,
        done_mask: torch.Tensor,
        cmd: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized partial reset for only the environments where ``done_mask`` is True."""
        mask = done_mask.to(device=self.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            return self.tok_hist.clone(), self.glob_hist.clone(), self.active_mask.clone()
        if bool(mask.all().item()):
            return self.reset(cmd=cmd)

        self._reset_solver_subset(mask)
        self.u_prev[mask] = 0.0
        self.a_prev[mask] = 0.0
        self.q_prev[mask] = self.solver.act_state[mask]
        self.dq_ema[mask] = 0.0
        self.step_count[mask] = 0
        self._last_v_arc[mask] = 0.0
        if cmd is not None:
            c = torch.as_tensor(cmd, device=self.device, dtype=self.dtype)
            if c.ndim == 1:
                self.cmd[mask] = c[None]
            elif c.shape[0] == self.batch:
                self.cmd[mask] = c[mask]
            else:
                self.cmd[mask] = c

        arc_all = self._query_com_arc()
        self._prev_arc_s[mask] = arc_all[mask]
        tf, gf = self._build_frame()
        self.tok_hist[mask] = tf[mask, None].expand(-1, self.H, -1, -1)
        self.glob_hist[mask] = gf[mask, None].expand(-1, self.H, -1)
        return self.tok_hist.clone(), self.glob_hist.clone(), self.active_mask.clone()

    def _compute_step_core(
        self,
        action_delta: torch.Tensor,
        *,
        delta_scale: float = 0.50,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        a = torch.as_tensor(action_delta, device=self.device, dtype=self.dtype)
        a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        if a.ndim == 1:
            a = a[None].expand(self.batch, -1)
        u_next = (self.u_prev + float(delta_scale) * a).clamp(-1.0, 1.0)
        # Jammed/failed actuators hold previous state
        u_eff = torch.where(self.active_mask, u_next, self.u_prev)
        raw_tgt = self._cmd_mid + u_eff * self._cmd_half
        action_delta_diff = (a - self.a_prev).square().mean(dim=-1)
        self.a_prev.copy_(a)
        self.u_prev.copy_(u_eff)

        self.solver.step(raw_tgt, steps=self.substeps_per_policy)
        self.step_count.add_(1)

        state_finite = (
            torch.isfinite(self.solver.x).all(dim=(1, 2))
            & torch.isfinite(self.solver.v).all(dim=(1, 2))
            & torch.isfinite(self.solver.act_state).all(dim=1)
        )
        solver_failed_flag = bool(self.solver.failed.item())

        dq_inst = (self.solver.act_state - self.q_prev) / max(self.policy_dt, 1e-6)
        dq_inst = torch.nan_to_num(dq_inst, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
        self.dq_ema.mul_(0.65).add_(0.35 * dq_inst)
        self.q_prev.copy_(self.solver.act_state)

        arc_now = self._query_com_arc()
        v_arc = ((arc_now - self._prev_arc_s) / max(self.policy_dt, 1e-6)).clamp(-5.0, 5.0)
        self._prev_arc_s = arc_now
        self._last_v_arc = v_arc

        vel_b, omega_b, grav_b = self._imu_kinematics()
        tilt = torch.acos((-grav_b[:, 2]).clamp(-1.0, 1.0))

        # Termination & truncation guards (batch-safe per-env detection)
        all_finite = bool(state_finite.all().item())
        solver_fail_mask = getattr(self.solver, 'failed_mask', None)
        if solver_fail_mask is not None and solver_fail_mask.shape == (self.batch,):
            per_env_fail = solver_fail_mask.to(device=self.device, dtype=torch.bool)
        else:
            per_env_fail = torch.full(
                (self.batch,),
                bool(solver_failed_flag and all_finite),
                device=self.device,
                dtype=torch.bool,
            )
        if self._has_fixed_base:
            terminated = (~state_finite) | per_env_fail
        else:
            flipped = tilt > self.max_tilt_rad
            terminated = (~state_finite) | flipped | per_env_fail
        truncated = (self.step_count >= self.max_episode_steps) & (~terminated)
        done = terminated | truncated

        tf, gf = self._build_frame()
        self.tok_hist = torch.roll(self.tok_hist, shifts=-1, dims=1)
        self.glob_hist = torch.roll(self.glob_hist, shifts=-1, dims=1)
        self.tok_hist[:, -1] = tf
        self.glob_hist[:, -1] = gf

        next_glob_sens = gf[:, :6].clone()
        est_target = self.get_est_target()
        priv = self.get_privileged_obs()

        # Morphology-aware dense reward (finite, bounded, encouraging progress + stability + smoothness)
        if self._has_fixed_base:
            # Fixed-base continuum arm/endoscope: track 3D tip deflection target driven by cmd
            tip_pos = self.solver.x[:, self._tip_body_idx, :3]
            tip_rest = self.solver.rest[:, self._tip_body_idx, :3]
            tip_disp = tip_pos - tip_rest
            # Target tip offset in meters derived cleanly from cmd [x_fwd, y_lat, z_vert]
            tgt_offset = torch.stack(
                (
                    torch.zeros_like(self.cmd[:, 0]),
                    0.025 * self.cmd[:, 1].clamp(-1.0, 1.0),
                    0.025 * self.cmd[:, 2].clamp(-1.0, 1.0),
                ),
                dim=-1,
            )
            tip_err = torch.linalg.vector_norm(tip_disp - tgt_offset, dim=-1)
            progress_term = torch.exp(-20.0 * tip_err) + 0.5 * v_arc.clamp(-1.0, 1.0)
            reward = (
                progress_term.float()
                - 0.01 * (a.float() ** 2).mean(dim=-1)
                - 0.005 * action_delta_diff.float()
            )
        else:
            upright_bonus = 0.05 * (-grav_b[:, 2]).clamp(0.0, 1.0)
            reward = (
                v_arc.float()
                + upright_bonus.float()
                - 0.01 * (a.float() ** 2).mean(dim=-1)
                - 0.005 * action_delta_diff.float()
            )
        reward = torch.where(terminated, reward - 1.0, reward)
        reward = torch.nan_to_num(reward, nan=-1.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        info: dict[str, Any] = {
            'arc_s_m': arc_now,
            'v_arc_m_s': v_arc,
            'body_vel_m_s': vel_b,
            'tilt_rad': tilt,
            'failed': bool(solver_failed_flag or (not bool(state_finite.all().item()))),
            'self_contact_energy': self.solver.last_self_contact_energy,
            'terminated': terminated,
            'truncated': truncated,
            'time_outs': truncated,
            'done': done,
            'priv': priv,
            'est_target': est_target,
            'next_glob_sens': next_glob_sens,
            'step_count': self.step_count.clone(),
            'backend': self.active_backend,
        }
        return (
            reward,
            terminated,
            truncated,
            done,
            priv,
            est_target,
            next_glob_sens,
            v_arc,
            info,
        )

    def step_batch(
        self,
        action_delta: torch.Tensor,
        *,
        delta_scale: float = 0.50,
        auto_reset: bool = True,
    ) -> EnvStepOut:
        """Structured vectorized RL step returning pre-reset terminal obs and post-reset next obs."""
        reward, terminated, truncated, done, priv, est_target, next_glob_sens, _, info = self._compute_step_core(
            action_delta, delta_scale=delta_scale
        )
        term_tok = self.tok_hist.clone()
        term_glob = self.glob_hist.clone()
        term_priv = priv.clone()

        if auto_reset and bool(done.any().item()):
            obs = self.reset_done(done)
            next_priv = self.get_privileged_obs()
            info['terminal_obs'] = (term_tok, term_glob, self.active_mask.clone())
            info['terminal_priv'] = term_priv
        else:
            obs = (self.tok_hist.clone(), self.glob_hist.clone(), self.active_mask.clone())
            next_priv = priv

        return EnvStepOut(
            obs=obs,
            priv=next_priv,
            est_target=est_target,
            next_glob_sens=next_glob_sens,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            done=done,
            terminal_tokens=term_tok,
            terminal_glob=term_glob,
            terminal_priv=term_priv,
            info=info,
        )

    def step(
        self,
        action_delta: torch.Tensor,
        *,
        delta_scale: float = 0.50,
        return_done: bool | None = None,
    ) -> tuple[Any, ...]:
        """Apply incremental normalized action in [-1, 1], advance physics, and return updated histories.

        Backward compatible by default (returns ``(obs, reward, info)``). Set ``return_done=True``
        on ``__init__`` or ``step(...)`` for Gymnasium-like 4-tuple ``(obs, reward, done, info)``.
        """
        out = self.step_batch(action_delta, delta_scale=delta_scale, auto_reset=self.auto_reset)
        use_4tuple = self.return_done if return_done is None else bool(return_done)
        if use_4tuple:
            return out.obs, out.reward, out.done, out.info
        return out.obs, out.reward, out.info

    def export_manifest(self, safety: SafetyConfig | None = None) -> dict[str, Any]:
        """Export hardware-agnostic RobotManifest dict for 1-hour real-robot onboarding."""
        return {
            'robot_name': self.graph.name,
            'n_actuators': self.N_act,
            'history_len': self.H,
            'policy_hz': round(1.0 / max(self.policy_dt, 1e-6), 2),
            'symmetry_group': self.graph.symmetry.name if self.graph.symmetry else None,
            'sparsity_route': self.solver.route,
            'physics_backend': self.active_backend,
            'actuators': [asdict(a) for a in self.graph.actuators],
            'safety': asdict(safety or SafetyConfig()),
        }
