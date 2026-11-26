"""Verification tests for SOFA-free parametric topology, frozen-tangent reuse,
wheel clearance indexing, reflection symmetry equivariance, and GPU-resident evaluation.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig, UnifiedTrainConfig
from serpworm.envs.unified_env import UnifiedSerpWormEnv, mirror_rollout_batch
from serpworm.paths import INPUTS_DIR
from serpworm.physics.gpu_core import GPUCore, pack_beam_matrix
from serpworm.physics.topology import build_topology
from serpworm.physics.wheel_envelope import active_pair_indices, certified_clearance
from serpworm.rl.eval_unified import evaluate_unified
from serpworm.rl.train_unified import train_unified


class TopologyAndAccelerationTests(unittest.TestCase):
    def test_sofa_free_topology_matches_frozen_npz_all_counts(self):
        for count in (1, 2, 5):
            gen = build_topology(count=count, elements_per_strip=16)
            ref = np.load(INPUTS_DIR / f'topology_{count}.npz')
            np.testing.assert_array_equal(gen['edges'], ref['edges'])
            np.testing.assert_array_equal(gen['strip_nodes'], ref['strip_nodes'])
            np.testing.assert_allclose(gen['lengths'], ref['lengths'], atol=1e-15, rtol=0.0)
            np.testing.assert_allclose(gen['rest'][:, :3], ref['rest'][:, :3], atol=1e-15, rtol=0.0)
            np.testing.assert_allclose(gen['transforms'][..., :3], ref['transforms'][..., :3], atol=1e-15, rtol=0.0)
            np.testing.assert_allclose(gen['wheel_mounts'], ref['wheel_mounts'], atol=1e-15, rtol=0.0)
            np.testing.assert_allclose(gen['anchors'], ref['anchors'], atol=1e-15, rtol=0.0)
            np.testing.assert_allclose(gen['guides'], ref['guides'], atol=1e-15, rtol=0.0)
            # Compare quaternions row-by-row up to double-cover sign (+q ~ -q)
            q_diff = np.minimum(
                np.linalg.norm(gen['rest'][:, 3:] - ref['rest'][:, 3:], axis=-1),
                np.linalg.norm(gen['rest'][:, 3:] + ref['rest'][:, 3:], axis=-1),
            ).max()
            self.assertLess(q_diff, 1e-14)
            tr_diff = np.minimum(
                np.linalg.norm(gen['transforms'][..., 3:] - ref['transforms'][..., 3:], axis=-1),
                np.linalg.norm(gen['transforms'][..., 3:] + ref['transforms'][..., 3:], axis=-1),
            ).max()
            self.assertLess(tr_diff, 1e-14)

    def test_coarsened_mesh_e8_and_packed_beam_constants(self):
        hw = HardwareConfig.ds3218_default()
        core8 = GPUCore(5, 2, dt=0.0005, device='cpu', plate_solver='segment_lu', hardware=hw, elements_per_strip=8)
        self.assertEqual(core8.E, 8)
        self.assertEqual(core8.D, 10 + 40 * 7)
        packed = pack_beam_matrix(core8.K)
        self.assertEqual(tuple(packed.shape), (40, 8, 16))
        core8.target.fill_(1.1)
        core8.yaw.copy_(torch.tensor([[0.25, -0.20, 0.15, -0.10]] * 2, dtype=torch.float64))
        core8.advance_steps(20)
        self.assertFalse(bool(core8.failed))
        self.assertTrue(bool(torch.isfinite(core8.x).all() and torch.isfinite(core8.v).all()))

    def test_frozen_tangent_interval_tracks_exact_trajectory(self):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        hw = HardwareConfig.ds3218_default()
        ref = GPUCore(5, 2, dt=0.0005, device=dev, plate_solver='segment_lu', hardware=hw, elements_per_strip=8, tangent_interval=1)
        fast = GPUCore(5, 2, dt=0.0005, device=dev, plate_solver='segment_lu', hardware=hw, elements_per_strip=8, tangent_interval=4)
        if dev == 'cuda':
            ref.capture(4)
            fast.capture(4)
        for c in (ref, fast):
            c.target.fill_(1.15)
            c.yaw.copy_(torch.tensor([[0.30, -0.25, 0.20, -0.15], [-0.20, 0.25, -0.15, 0.10]], device=dev, dtype=torch.float64))
        ref.advance_steps(40)
        fast.advance_steps(40)
        self.assertFalse(bool(ref.failed or fast.failed))
        pos_err = (ref.x[:, :10, :3] - fast.x[:, :10, :3]).abs().max().item()
        self.assertLess(pos_err, 2.5e-4, f'Expected < 0.25 mm 20 ms frozen-tangent drift, got {pos_err:.3e} m')

    def test_wheel_clearance_indexed_pair_bit_parity(self):
        rng = torch.Generator().manual_seed(99)
        centers = torch.randn(16, 20, 3, generator=rng, dtype=torch.float64) * 0.08
        raw_axes = torch.randn(16, 20, 3, generator=rng, dtype=torch.float64)
        axes = raw_axes / torch.linalg.vector_norm(raw_axes, dim=-1, keepdim=True)
        owners = torch.arange(10).repeat_interleave(2)
        mask = owners[:, None] != owners[None, :]
        idx = active_pair_indices(mask)
        self.assertEqual(int(idx[0].numel()), 180)
        c1, s1 = certified_clearance(centers, axes, mask)
        c2, s2 = certified_clearance(centers, axes, mask, pair_indices=idx)
        torch.testing.assert_close(c1, c2, atol=0.0, rtol=0.0)
        torch.testing.assert_close(s1, s2, atol=0.0, rtol=0.0)

    def test_mirror_symmetry_involution_and_physical_equivariance(self):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        hw = HardwareConfig.ds3218_default()
        task = TaskConfig(physics_dt_s=0.001, policy_dt_s=0.10, sensor_dt_s=0.02, elements_per_strip=8, episode_seconds=4.0, stage=2)
        env = UnifiedSerpWormEnv(n=2, hardware=hw, task=task, device=dev, plate_solver='segment_lu', noise=False, forced_mode=2)
        # Lane 0 and Lane 1 start in identical settled states; give mirrored commands & actions
        env.cmd[0] = torch.tensor([0.04, +0.08, 0.0], device=dev, dtype=torch.float64)
        env.cmd[1] = torch.tensor([0.04, -0.08, 0.0], device=dev, dtype=torch.float64)
        act = torch.tensor([
            [0.3, -0.2, 0.4, 0.1, -0.3, +0.6, -0.5, +0.4, -0.3],
            [0.3, -0.2, 0.4, 0.1, -0.3, -0.6, +0.5, -0.4, +0.3],
        ], device=dev, dtype=torch.float32)
        obs, priv, rew, _, _, _, _, _ = env.step(act)
        # Reward must be identical for mirror-image trajectories!
        torch.testing.assert_close(rew[0], rew[1], atol=1e-6, rtol=1e-5)
        m_obs, m_priv, m_lat = mirror_rollout_batch(obs[0:1], priv[0:1], act[0:1], hw.tendon_pairing)
        torch.testing.assert_close(m_obs[0, -1], obs[1, -1], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(m_priv[0], priv[1], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(m_lat[0], act[1], atol=0.0, rtol=0.0)
        # Involution: mirror twice restores exact original
        obs2, priv2, lat2 = mirror_rollout_batch(m_obs, m_priv, m_lat, hw.tendon_pairing)
        torch.testing.assert_close(obs2, obs[0:1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(priv2, priv[0:1], atol=0.0, rtol=0.0)
        torch.testing.assert_close(lat2, act[0:1], atol=0.0, rtol=0.0)

    def test_gpu_resident_eval_and_symmetry_train_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            hw = HardwareConfig.ds3218_default()
            task = TaskConfig(physics_dt_s=0.001, policy_dt_s=0.10, sensor_dt_s=0.02, elements_per_strip=8, tangent_interval=2, episode_seconds=4.0, stage=2)
            cfg = UnifiedTrainConfig(
                hardware=hw, task=task, dr=DomainRandConfig(enabled=False),
                envs=2, horizon=4, rollouts=1, epochs=2, minibatches=2,
                symmetry_aug=True, amp_bf16=True,
                stage2_rollout=0, stage3_rollout=0, eval_every=0, plate_solver='segment_lu',
            )
            train_unified(cfg, tmp / 'train')
            summary = evaluate_unified(tmp / 'train' / 'latest.pt', tmp / 'eval', seconds=0.4, plate_solver='segment_lu', save_traj=True)
            self.assertEqual(len(summary['results']), 9)
            self.assertTrue((tmp / 'eval' / 'eval_trajectories.npz').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
