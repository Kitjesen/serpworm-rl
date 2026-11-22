"""Numerical certificate tests; CPU geometry is not CUDA/physical validation."""
import math
import unittest
import numpy as np
import torch

from audit_wheel_geometry import ROOT, geometry, cylinder_certificate
from wheel_envelope import certified_clearance, SPHERE_RADIUS


def one_pair(delta, dtype=torch.float64):
    centers = torch.zeros(1, 20, 3, dtype=dtype)
    centers[0, 1] = torch.tensor(delta, dtype=dtype)
    axes = torch.zeros_like(centers); axes[..., 1] = 1.
    mask = torch.zeros(20, 20, dtype=torch.bool); mask[0, 1] = mask[1, 0] = True
    return centers, axes, mask


class WheelCertificateTests(unittest.TestCase):
    def test_exact_parallel_cylinder_radial_and_axial_gaps(self):
        for delta, expected in (([.04, 0., 0.], .004), ([0., .014, 0.], .002), ([0., 0., .04], .004)):
            gap, sphere = certified_clearance(*one_pair(delta))
            self.assertAlmostEqual(float(gap), expected, delta=2e-10)
            self.assertAlmostEqual(float(sphere), math.dist([0., 0., 0.], delta)-2*SPHERE_RADIUS, delta=1e-14)
        boundary, _ = certified_clearance(*one_pair([0., .014, 0.]))
        self.assertLess(float(boundary), .002)  # Exact boundary cannot bypass the strict safety margin.

    def test_coincident_parallel_axes_and_zero_directions_do_not_pass(self):
        gap, sphere = certified_clearance(*one_pair([0., 0., 0.]))
        self.assertTrue(torch.isfinite(gap).all() and torch.isfinite(sphere).all())
        self.assertLess(float(gap), 0.)
        self.assertLess(float(sphere), 0.)

    def test_rigid_geometry_matches_independent_numpy_certificate(self):
        with np.load(ROOT/'inputs'/'topology_5.npz') as data:
            rest, mounts = data['rest'], data['wheel_mounts']
        q = np.radians([[v, 0., 0., 0.] for v in (25., 29.25, 35., 45.)])
        _, frames, centers = geometry(q, .1175, rest, mounts)
        axes = frames[..., 1].repeat(2, axis=1)
        owners = np.arange(20)//2
        mask = owners[:, None] != owners[None]
        actual, raw_sphere = certified_clearance(torch.from_numpy(centers), torch.from_numpy(axes), torch.from_numpy(mask))
        delta = centers[:, None]-centers[:, :, None]
        a = np.broadcast_to(axes[:, :, None], delta.shape)
        b = np.broadcast_to(axes[:, None], delta.shape)
        projected = cylinder_certificate(centers[:, :, None], a, centers[:, None], b)[0]
        sphere = np.linalg.norm(delta, axis=-1)-2*SPHERE_RADIUS
        expected = np.where(mask[None], np.maximum(sphere, projected), np.inf).min(axis=(1, 2))
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-10, rtol=0.)
        np.testing.assert_allclose(raw_sphere.numpy(), np.where(mask[None], sphere, np.inf).min(axis=(1, 2)), atol=1e-14)
        self.assertLess(float(raw_sphere[1]), .002)
        self.assertGreater(float(actual[1]), .002)  # The sphere falsely rejects a provably separated pair.
        self.assertLess(float(actual[2]), .002)
        self.assertLess(float(actual[3]), .002)

    def test_parallel_radial_gap_is_preserved_by_random_rigid_transforms(self):
        rng = np.random.default_rng(11)
        centers, axes, mask = one_pair([.04, 0., 0.])
        rows, directions = [], []
        for _ in range(30):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            translation = rng.uniform(-10., 10., size=3)
            rows.append(centers[0].numpy()@rotation.T+translation)
            directions.append(axes[0].numpy()@rotation.T)
        gap, sphere = certified_clearance(torch.tensor(np.asarray(rows)), torch.tensor(np.asarray(directions)), mask)
        torch.testing.assert_close(gap, torch.full_like(gap, .004), atol=3e-10, rtol=0.)
        self.assertTrue(torch.isfinite(sphere).all())

    def test_random_common_point_overlap_never_gives_positive_certificate(self):
        rng = np.random.default_rng(21)
        centers, axes, mask = one_pair([0., 0., 0.])
        centers = centers.expand(100, -1, -1).clone()
        axes = axes.expand(100, -1, -1).clone()
        # Each cylinder contains a radius-6 mm ball around its center. Both
        # centers within 2 mm of the origin therefore share an interior point.
        values = rng.normal(size=(100, 2, 3)); values /= np.linalg.norm(values, axis=-1, keepdims=True)
        centers[:, :2] = torch.tensor(values*.002)
        values = rng.normal(size=(100, 2, 3)); values /= np.linalg.norm(values, axis=-1, keepdims=True)
        axes[:, :2] = torch.tensor(values)
        gap, sphere = certified_clearance(centers, axes, mask)
        self.assertTrue((gap < 0.).all())
        self.assertTrue(torch.isfinite(gap).all() and torch.isfinite(sphere).all())

    def test_invalid_axes_fail_closed_and_empty_mask_has_no_nan(self):
        centers, axes, mask = one_pair([1., 0., 0.])
        axes[0, 1].zero_()
        gap, _ = certified_clearance(centers, axes, mask)
        self.assertTrue(torch.isfinite(gap).all())
        self.assertLess(float(gap), 0.)
        gap, sphere = certified_clearance(centers, axes, torch.zeros_like(mask))
        self.assertTrue(torch.isinf(gap).all() and torch.isinf(sphere).all())
        self.assertFalse(torch.isnan(gap).any() or torch.isnan(sphere).any())

    def test_float32_is_finite_and_preserves_clear_cases(self):
        for delta, passed in (([.04, 0., 0.], True), ([0., 0., 0.], False)):
            gap, sphere = certified_clearance(*one_pair(delta, torch.float32))
            self.assertTrue(torch.isfinite(gap).all() and torch.isfinite(sphere).all())
            self.assertEqual(bool(gap >= .002), passed)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; no GPU parity claim')
    def test_cuda_cpu_parity(self):
        values = one_pair([.04, 0., 0.])
        reference = certified_clearance(*values)
        actual = certified_clearance(*(value.cuda() for value in values))
        for a, b in zip(actual, reference):
            torch.testing.assert_close(a.cpu(), b, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
