"""Rigid five-module kinematics and the existing spherical wheel guard; no dynamics/CAD collision claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from serpworm.paths import INPUTS_DIR, PKG_ROOT, REPO_ROOT, sha256_file

ROOT = REPO_ROOT
ARM = np.array([-.058, 0., -.0010452742534])
RADIUS = math.hypot(.018, .006)
GUARD = .002
PLANE_NAMES = ('axis_a', 'axis_b', 'cross_axes', 'center_delta', 'delta_perpendicular_a', 'delta_perpendicular_b', 'world_up')


def rz(angle):
    result = np.zeros((*np.shape(angle), 3, 3))
    c, s = np.cos(angle), np.sin(angle)
    result[..., 0, 0] = result[..., 1, 1] = c
    result[..., 0, 1], result[..., 1, 0], result[..., 2, 2] = -s, s, 1.
    return result


def geometry(q, length, rest, mounts, connector=ARM):
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if q.shape[1] != 4 or not np.isfinite(q).all() or length <= 0:
        raise ValueError('Expected four finite relative yaw angles and positive segment length')
    front = np.zeros((len(q), 5, 3))
    front[:, 0] = rest[0, :3]
    heading = np.column_stack((np.zeros(len(q)), np.cumsum(q, axis=1)))
    rotation = rz(heading)
    inside = np.array([-length, 0., rest[1, 2]-rest[0, 2]])
    back = np.zeros_like(front)
    for segment in range(5):
        if segment:
            front[:, segment] = back[:, segment-1]+np.einsum('bij,j->bi', rotation[:, segment-1], connector)
        back[:, segment] = front[:, segment]+np.einsum('bij,j->bi', rotation[:, segment], inside)
    position = np.stack((front, back), axis=2).reshape(len(q), 10, 3)
    frames = rotation.repeat(2, axis=1)
    centers = position[:, :, None]+np.einsum('bpij,psj->bpsi', frames, mounts[np.arange(10) % 2])
    return position, frames, centers.reshape(len(q), 20, 3)


def clearance(centers):
    distances = np.linalg.norm(centers[:, :, None]-centers[:, None], axis=-1)
    owners = np.arange(20)//2
    distances[:, owners[:, None] == owners[None]] = np.inf
    flat = distances.reshape(len(centers), -1).argmin(axis=1)
    ids = np.column_stack(np.unravel_index(flat, (20, 20)))
    return distances.reshape(len(centers), -1)[np.arange(len(centers)), flat]-2*RADIUS, ids


def cylinder_certificate(center_a, axis_a, center_b, axis_b):
    """A positive projected gap certifies separation; a negative result proves nothing."""
    delta, a, b = np.asarray(center_b)-center_a, np.asarray(axis_a), np.asarray(axis_b)
    if not all(np.isfinite(value).all() for value in (delta, a, b)):
        raise ValueError('Finite wheel centers and axes are required')
    if not all(np.allclose(np.linalg.norm(axis, axis=-1), 1., atol=1e-10) for axis in (a, b)):
        raise ValueError('Wheel axes must be unit length')
    directions = np.stack((a, b, np.cross(a, b), delta,
        delta-(delta*a).sum(-1)[..., None]*a, delta-(delta*b).sum(-1)[..., None]*b,
        np.broadcast_to(np.array([0., 0., 1.]), delta.shape)), axis=-2)
    length = np.linalg.norm(directions, axis=-1)
    normals = directions/np.maximum(length[..., None], 1e-15)
    dot_a, dot_b = (normals*a[..., None, :]).sum(-1), (normals*b[..., None, :]).sum(-1)
    support = lambda value: .006*np.abs(value)+.018*np.sqrt(np.maximum(0., 1.-value**2))
    gaps = np.abs((normals*delta[..., None, :]).sum(-1))-support(dot_a)-support(dot_b)
    gaps = np.where(length > 1e-12, gaps, -np.inf)
    best = np.argmax(gaps, axis=-1)
    return np.max(gaps, axis=-1), best, normals, gaps


def pair_certificate(centers, frames, ids):
    batch = np.arange(len(centers))
    axes = frames[..., 1]
    gap, best, normals, _ = cylinder_certificate(centers[batch, ids[:, 0]], axes[batch, ids[:, 0]//2],
        centers[batch, ids[:, 1]], axes[batch, ids[:, 1]//2])
    return [dict(separating_plane_gap_lower_bound_mm=float(gap[i]*1000),
        certified_2mm_separation=bool(gap[i] >= GUARD+1e-12),
        plane=PLANE_NAMES[int(best[i])], normal=normals[i, best[i]].tolist()) for i in range(len(ids))]


def refined_guard(centers, frames):
    """All different-plate pairs must have a sphere or cylinder-plane certificate."""
    left, right = np.triu_indices(20, k=1)
    valid = left//2 != right//2
    left, right = left[valid], right[valid]
    a, b = centers[:, left], centers[:, right]
    sphere = np.linalg.norm(b-a, axis=-1)-2*RADIUS
    axes = frames[..., 1]
    projected, _, _, _ = cylinder_certificate(a, axes[:, left//2], b, axes[:, right//2])
    certified = (sphere >= GUARD+1e-12) | (projected >= GUARD+1e-12)
    return certified.all(-1)


def pair_description(ids):
    return [dict(wheel_index=int(i), plate_index=int(i//2), segment_index=int(i//4),
                 plate_end='front' if (i//2) % 2 == 0 else 'back',
                 local_side='positive_y' if i % 2 == 0 else 'negative_y') for i in ids]


def self_check(rest, mounts):
    position, frames, centers = geometry(np.zeros(4), .1175, rest, mounts)
    np.testing.assert_allclose(position[0], rest[:10, :3], atol=1e-12)
    np.testing.assert_allclose(frames[0], np.broadcast_to(np.eye(3), (10, 3, 3)), atol=1e-12)
    expected = (rest[:10, None, :3]+mounts[np.arange(10) % 2]).reshape(20, 3)
    np.testing.assert_allclose(centers[0], expected, atol=1e-12)
    direct = min(np.linalg.norm(expected[i]-expected[j])-2*RADIUS for i in range(20) for j in range(i+1, 20) if i//2 != j//2)
    np.testing.assert_allclose(clearance(centers)[0][0], direct, atol=1e-12)
    q = np.radians([35., -21., 14., 7.])
    p, r, _ = geometry(q, .105, rest, mounts)
    for j in range(4):
        np.testing.assert_allclose(p[:, 2*j+2]-p[:, 2*j+1], np.einsum('bij,j->bi', r[:, 2*j+1], ARM), atol=1e-12)
        relative = r[:, 2*j+1].transpose(0, 2, 1)@r[:, 2*j+2]
        np.testing.assert_allclose(np.arctan2(relative[:, 1, 0], relative[:, 0, 0]), q[j], atol=1e-12)
    axis = np.array([0., 1., 0.])
    # Parallel solid cylinders have exact radial/axial gaps of 4 mm and 2 mm.
    for delta, expected_gap in (([.04, 0., 0.], .004), ([0., .014, 0.], .002)):
        gap, _, _, _ = cylinder_certificate(np.zeros(3), axis, np.array(delta), axis)
        np.testing.assert_allclose(gap, expected_gap, atol=1e-12)
    p, r, c = geometry(np.radians([29.25, 0., 0., 0.]), .1175, rest, mounts)
    sphere, ids = clearance(c)
    assert sphere[0] < GUARD and pair_certificate(c, r, ids)[0]['certified_2mm_separation']
    assert refined_guard(c, r).all()
    return dict(rest_pose_matches=True, rest_direct_clearance_matches=True,
                exact_connector_constraint=True, measured_relative_yaw_matches=True,
                exact_parallel_cylinder_projection_cases=True,
                sphere_false_rejection_29_25deg_strictly_separated=True,
                rest_clearance_mm=float(direct*1000))


def audit(phase_samples=720):
    if phase_samples < 16:
        raise ValueError('Use at least sixteen wave phases')
    with np.load(INPUTS_DIR/'topology_5.npz') as data:
        rest, mounts = data['rest'], data['wheel_mounts']
    proof = self_check(rest, mounts)
    angles = np.linspace(0., 45., 181)
    phases = np.arange(phase_samples)*2*math.pi/phase_samples
    reports = []
    for length in (.1175, .105, .086, .075):
        row = dict(segment_plate_spacing_m=length, single_joint=[], spatial_waves=[])
        for joint in range(4):
            q = np.zeros((len(angles), 4)); q[:, joint] = np.radians(angles)
            values, ids = clearance(geometry(q, length, rest, mounts)[2])
            lo, hi = 0., 45.
            for _ in range(45):
                middle = .5*(lo+hi)
                trial = np.zeros(4); trial[joint] = math.radians(middle)
                if clearance(geometry(trial, length, rest, mounts)[2])[0][0] < GUARD: hi = middle
                else: lo = middle
            curve = {str(a): float(values[np.argmin(abs(angles-a))]*1000) for a in (0., 15., 25., 29.25, 35., 45.)}
            check_angles = (25., 29.25, 35., 45.)
            checks = np.zeros((len(check_angles), 4)); checks[:, joint] = np.radians(check_angles)
            _, check_frames, check_centers = geometry(checks, length, rest, mounts)
            check_clearance, check_ids = clearance(check_centers)
            certificates = pair_certificate(check_centers, check_frames, check_ids)
            passed = refined_guard(check_centers, check_frames)
            for index, certificate in enumerate(certificates):
                certificate.update(joint_angle_deg=check_angles[index], sphere_clearance_mm=float(check_clearance[index]*1000),
                    pair=pair_description(check_ids[index]), all_wheel_pairs_certified=bool(passed[index]))
            row['single_joint'].append(dict(joint_index=joint, guard_2mm_threshold_deg=.5*(lo+hi),
                example_clearance_mm=curve, at_35deg_pair=pair_description(ids[np.argmin(abs(angles-35.))]),
                finite_cylinder_plane_certificates=certificates,
                scan_angle_deg=angles.tolist(), scan_clearance_mm=(values*1000).tolist()))
        for amplitude in (29.25, 35., 45.):
            q = np.radians(amplitude)*np.sin(phases[:, None]+np.arange(4)*math.pi/2)
            _, frames, centers = geometry(q, length, rest, mounts)
            values, ids = clearance(centers)
            certified = refined_guard(centers, frames)
            worst = int(values.argmin())
            row['spatial_waves'].append(dict(amplitude_deg=amplitude, adjacent_phase_deg=90., phase_samples=phase_samples,
                guard_failure_fraction=float((values < GUARD).mean()), minimum_clearance_mm=float(values.min()*1000),
                cylinder_certificate_unresolved_fraction=float((~certified).mean()),
                cylinder_certificate_all_phases_clear=bool(certified.all()),
                maximum_clearance_mm=float(values.max()*1000), worst_phase_deg=float(np.degrees(phases[worst])),
                worst_joint_angles_deg=np.degrees(q[worst]).tolist(), worst_pair=pair_description(ids[worst])))
        reports.append(row)
    return dict(scope='Rigid modules and current wheel sphere-envelope guard only; no force balance, deformation or CAD collision validation',
        wheel_radius_m=.018, wheel_half_width_m=.006, sphere_radius_m=RADIUS, clearance_guard_m=GUARD,
        finite_cylinder_certificate=dict(planes=list(PLANE_NAMES), support='0.006*abs(n dot axis)+0.018*sqrt(max(0,1-(n dot axis)^2))',
            interpretation='gap >=2 mm certifies at least 2 mm solid-cylinder separation; failure of finitely many planes is inconclusive, never proof of collision',
            numerical_acceptance_margin_m=1e-12),
        connector_arm_m=ARM.tolist(), source_sha256={name: sha256_file(PKG_ROOT/name)
            for name in ('probes/wheel_geometry.py', 'physics/gpu_core.py', 'envs/tensor_env.py')},
        topology_sha256=sha256_file(INPUTS_DIR/'topology_5.npz'),
        self_check=proof, results=reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO_ROOT/'runs'/'wheel_geometry.json')
    parser.add_argument('--phase-samples', type=int, default=720)
    args = parser.parse_args(argv)
    report = audit(args.phase_samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
    print(json.dumps(dict(path=str(args.out), self_check=report['self_check'],
        threshold_deg=report['results'][0]['single_joint'][0]['guard_2mm_threshold_deg'],
        waves=report['results'][0]['spatial_waves']), indent=2))


if __name__ == '__main__':
    main()
