"""Unit tests for v6 mode-free pure-proprioceptive autonomous phase-transition RL."""
from __future__ import annotations

import torch

from serpworm.config import DomainRandConfig, HardwareConfig
from serpworm.envs.autonomous_env import (
    AutonomousPhaseEnv,
    AutonomousTaskConfig,
    TERRAIN_PROFILES,
    mirror_autonomous_batch,
)
from serpworm.physics.gpu_core import GPUCore
from serpworm.rl.models import ProprioceptiveEstimatorActorCritic


def test_per_env_terrain_and_wall_contact_cpu():
    hw = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
    c = GPUCore(5, 2, dt=0.001, device='cpu', plate_solver='segment_lu', hardware=hw, elements_per_strip=8)
    # Env 0: open hard floor; Env 1: soft soil + narrow pipe wall pushing back lateral velocity
    c.env_terrain[1] = torch.tensor(TERRAIN_PROFILES['soft_soil'][:5], dtype=torch.float64)
    c.wheel_mu[1] = TERRAIN_PROFILES['soft_soil'][5]
    c.wall_cfg[1] = torch.tensor([0.0, 0.015, 10.0, -10.0, 3200.0], dtype=torch.float64)
    # Push Env 1 laterally past y_allow = 0.015 m
    c.x[1, :, 1] += 0.025
    c.v[1, :, 1] = 0.05
    for _ in range(5):
        c.physics()
    assert torch.isfinite(c.x).all() and torch.isfinite(c.v).all()
    assert float(c.last_wall_force[0].sum()) == 0.0
    assert float(c.last_wall_force[1].sum()) > 10.0
    # Implicit wall force should decelerate outward lateral motion smoothly
    assert float(c.v[1, :10, 1].mean()) < 0.05


def test_autonomous_env_51d_and_65d_and_symmetry():
    for include_q_err, expected_dim in ((False, 51), (True, 65)):
        hw = HardwareConfig.v5_realistic(tendon_pairing='independent_10d', terrain='hard')
        task = AutonomousTaskConfig(elements_per_strip=6, physics_dt_s=0.002, include_q_err=include_q_err, history_len=8)
        env = AutonomousPhaseEnv(
            n=2, hardware=hw, task=task, dr=DomainRandConfig(enabled=False),
            device='cpu', plate_solver='segment_lu', seed=11, noise=False, settle_ticks=6,
        )
        assert env.frame_size == expected_dim
        assert env.obs.shape == (2, 8, expected_dim)
        assert env.critic_obs.shape == (2, env.privileged_size) and env.privileged_size == 108
        assert env.est_target.shape == (2, 7)

        model = ProprioceptiveEstimatorActorCritic(
            frame_size=env.frame_size,
            history_len=env.history_len,
            privileged_size=env.privileged_size,
            est_size=env.est_size,
            action_size=14,
        )
        act, lat, lp, val = model.sample(env.obs, env.critic_obs)
        assert act.shape == (2, 14) and lp.shape == (2,) and val.shape == (2,)
        nobs, npriv, nest, rew, term, trunc, tobs, tpriv, info = env.step(act)
        assert torch.isfinite(nobs).all() and torch.isfinite(npriv).all() and torch.isfinite(nest).all()
        assert torch.isfinite(rew).all()

        # Verify exact double-reflection involution for mirror_autonomous_batch
        m1 = mirror_autonomous_batch(nobs, npriv, nest, lat, include_q_err=include_q_err)
        m2 = mirror_autonomous_batch(*m1, include_q_err=include_q_err)
        for orig, rt in zip((nobs, npriv, nest, lat), m2):
            assert torch.allclose(orig, rt, atol=1e-6)

        # Verify full evaluate_full + backward pass with EstNet loss
        lp2, ent, v2, mu, est_pred = model.evaluate_full(nobs, npriv, lat)
        assert est_pred.shape == (2, 7) and mu.shape == (2, 14)
        loss = -lp2.mean() + 0.5 * v2.square().mean() + (est_pred - nest).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
