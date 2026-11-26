"""Package layout, manifest, physics core, and CPG smoke tests."""
import unittest
import torch

import serpworm
from serpworm.config import UnifiedTrainConfig
from serpworm.envs.matsuoka import Matsuoka
from serpworm.envs.rewards import locomotion_reward, velocity_reward
from serpworm.envs.unified_env import UnifiedSerpWormEnv
from serpworm.paths import (
    EVAL_SOURCES,
    INPUTS_DIR,
    PHYSICS_SOURCES,
    PKG_ROOT,
    UNIFIED_SOURCES,
    collect_input_hashes,
    collect_source_hashes,
)
from serpworm.physics.gpu_core import GPUCore
from serpworm.rl.models import TemporalActorCritic


class PackageSmokeTests(unittest.TestCase):
    def test_manifests_cover_required_sources_and_inputs(self):
        sources = collect_source_hashes(PKG_ROOT)
        for rel in (*PHYSICS_SOURCES, *EVAL_SOURCES, *UNIFIED_SOURCES):
            self.assertIn(rel, sources)
        inputs = collect_input_hashes(PKG_ROOT)
        self.assertIn('parameters.json', inputs)
        self.assertIn('topology_5.npz', inputs)
        self.assertTrue((INPUTS_DIR / 'topology_5.npz').is_file())

    def test_lazy_exports_match_submodules(self):
        self.assertIs(serpworm.GPUCore, GPUCore)
        self.assertIs(serpworm.Matsuoka, Matsuoka)
        self.assertIs(serpworm.UnifiedSerpWormEnv, UnifiedSerpWormEnv)
        self.assertIs(serpworm.TemporalActorCritic, TemporalActorCritic)
        self.assertIs(serpworm.UnifiedTrainConfig, UnifiedTrainConfig)
        self.assertIs(serpworm.locomotion_reward, locomotion_reward)
        self.assertIs(serpworm.velocity_reward, velocity_reward)

    def test_gpu_core_legacy_and_segment_lu_match_on_cpu(self):
        torch.set_num_threads(1)
        legacy = GPUCore(count=5, batch=1, dt=.00025, device='cpu', strain_jacobian=True, plate_solver='legacy')
        seg_lu = GPUCore(count=5, batch=1, dt=.00025, device='cpu', strain_jacobian=True, plate_solver='segment_lu')
        for core in (legacy, seg_lu):
            core.yaw.copy_(core.tensor([.05, -.05, .05, -.05]))
            core.advance_steps(4)
        self.assertFalse(bool(legacy.failed or seg_lu.failed))
        torch.testing.assert_close(legacy.x[..., :3], seg_lu.x[..., :3], atol=1e-7, rtol=0.)
        torch.testing.assert_close(legacy.x[..., 3:], seg_lu.x[..., 3:], atol=1e-6, rtol=0.)
        torch.testing.assert_close(legacy.v[..., :3], seg_lu.v[..., :3], atol=1e-5, rtol=0.)
        torch.testing.assert_close(legacy.v[..., 3:], seg_lu.v[..., 3:], atol=1e-4, rtol=0.)

    def test_matsuoka_oscillator_steps_cleanly(self):
        cpg = Matsuoka(2, device='cpu', c=.75)
        out = cpg.step(torch.zeros(2, 4, dtype=torch.float64))
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
