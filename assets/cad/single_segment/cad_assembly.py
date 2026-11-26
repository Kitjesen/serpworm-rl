"""CAD-confirmed bore geometry and explicitly provisional four-cord routing.

This module does not alter simulation forces or turn old recordings into a new
physical experiment. Bore IDs are extracted CAD labels, not measured hardware.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load_assembly():
    return json.loads((HERE / 'cad_assembly.json').read_text(encoding='utf-8'))


def strip_endpoints(params):
    """Eight front/back crease midpoints, shape (8, 2, 3), in plate-local frames.

    Front inward normal is -X; back inward normal is +X. The elastic span
    starts at the outer edge of the bolted tab, not at its bolt-pair centre.
    This is the same one-crease construction used by assembly_visuals.
    """
    count = int(params['strip_count'])
    thickness = float(params['strip_thickness_m'])
    fold_offset = float(params['tab_bolt_to_fold_m'])
    if count != 8 or thickness <= 0 or fold_offset <= 0:
        raise ValueError('Expected eight strips with positive thickness and bolt-to-fold offset')
    result = np.empty((count, 2, 3))
    for end, kind in enumerate(('front', 'back')):
        pairs = np.asarray(params[kind+'_clamp_hole_pairs_m'], dtype=float)
        center = np.asarray(params[kind+'_plate_center_m'], dtype=float)
        if pairs.shape != (count, 2, 3) or center.shape != (3,):
            raise ValueError('Clamp hole pairs or plate center have incorrect shapes')
        if not np.isfinite(pairs).all() or not np.isfinite(center).all():
            raise ValueError('Nonfinite CAD clamp coordinates')
        if not np.allclose(pairs[:, :, 0], center[0], rtol=0, atol=1e-7):
            raise ValueError('Clamp holes must lie on the CAD inside face')
        midpoint = pairs.mean(axis=1)
        radial = midpoint-center
        radial[:, 0] = 0.
        norms = np.linalg.norm(radial, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError('Clamp centre cannot coincide with plate centre')
        radial /= norms[:, None]
        normal = np.array([-1., 0., 0.]) if kind == 'front' else np.array([1., 0., 0.])
        result[:, end] = midpoint + radial*fold_offset + normal*thickness/2
    return result


def horn_tie(angle, route, assembly=None):
    """Surface-equivalent tie on the recorded-angle CAD horn; radians, plate frame."""
    assembly = load_assembly() if assembly is None else assembly
    side = route['servo_slot']
    pivot = np.array(assembly['horn_pivots_m'][side])
    phase = np.pi / 2 + (1 if side == 0 else -1) * angle
    c, s = np.cos(phase), np.sin(phase)
    rotor = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    offset = np.array([assembly['horn_surface_tie_offset_x_m'],
                       route['arm_sign'] * assembly['horn_tie_radius_m'], 0.])
    return pivot + rotor @ offset


def back_route(angle, route, assembly=None):
    """Back-plate points in span-to-servo order, including both bore faces."""
    points = [route['back_entry_m'], route['back_exit_m']]
    if route.get('use_external_guide', True):
        points.append(route['external_guide_m'])
    return np.array(points+[horn_tie(angle, route, assembly)])


def segment_box_distance(a, b, lower, upper):
    """Exact minimum distance between a finite line segment and a solid AABB."""
    a, b, lower, upper = map(np.asarray, (a, b, lower, upper))
    delta = b - a
    knots = [0., 1.]
    for axis in range(3):
        if abs(delta[axis]) > 1e-15:
            knots += [float(t) for face in (lower[axis], upper[axis])
                      if 0 < (t := (face-a[axis])/delta[axis]) < 1]
    knots = sorted(set(knots))
    candidates = knots.copy()
    for start, end in zip(knots, knots[1:]):
        midpoint = a + (start+end)/2 * delta
        active = (midpoint < lower) | (midpoint > upper)
        boundary = np.where(midpoint < lower, lower, upper)
        denominator = np.dot(delta[active], delta[active])
        if denominator:
            stationary = -np.dot(delta[active], (a-boundary)[active])/denominator
            candidates.append(float(np.clip(stationary, start, end)))
    points = a + np.array(candidates)[:, None]*delta
    distances = np.linalg.norm(np.maximum(np.maximum(lower-points, points-upper), 0.), axis=1)
    return float(distances.min())


def verify(assembly=None):
    """Conservative motor clearance, exact bore alignment, and occupied-hole checks."""
    assembly = load_assembly() if assembly is None else assembly
    radius = assembly['rope_radius_m']
    clamped = {name for pair in assembly['steel_clamps']
               for key in ('front_hole_ids', 'back_hole_ids') for name in pair[key]}
    assert not clamped.intersection(assembly['reserved_rope_hole_ids'])
    # Meaningful sanity checks for the distance calculation, including segment interiors.
    box = (np.zeros(3), np.ones(3))
    assert segment_box_distance([-1, .5, .5], [2, .5, .5], *box) == 0.
    assert abs(segment_box_distance([-1, 2, .5], [2, 2, .5], *box)-1.) < 1e-12
    rows = []
    for route in assembly['routes']:
        for side in ('front', 'back'):
            assert route[side+'_bore_confirmed'] and route[side+'_bore_axis_clear']
            assert route[side+'_bore_type'] == 'straight'
        entry, exit = map(np.array, (route['back_entry_m'], route['back_exit_m']))
        assert np.allclose(entry[1:], exit[1:], atol=1e-12)
        assert entry[0] == 0. and exit[0] == -.003
        assert route['front_anchor_m'][0] == -.0165
        assert route['front_hole_id'] not in clamped and route['back_hole_id'] not in clamped
        assert np.allclose(route['external_guide_m'][1:], entry[1:])
        assert route['external_guide_m'][0] == -.025
        minimum, worst = float('inf'), None
        direct_minimum = float('inf')
        for deg in np.arange(0., 90.0001, .5):
            points = back_route(np.deg2rad(deg), route, assembly)
            assert len(points) == (4 if route.get('use_external_guide', True) else 3)
            for motor, box in enumerate(assembly['motor_aabbs_m']):
                for leg, (a, b) in enumerate(zip(points, points[1:])):
                    gap = segment_box_distance(a, b, *box) - radius
                    if gap < minimum:
                        minimum, worst = gap, {'angle_deg': float(deg), 'motor': motor, 'leg': leg}
                direct_minimum = min(direct_minimum,
                    segment_box_distance(points[1], points[-1], *box)-radius)
        rows.append({'route': route['name'], 'bore_radius_mm': route['back_hole_radius_m']*1000,
                     'external_guide_used': route.get('use_external_guide', True),
                     'bore_radial_margin_mm': (route['back_hole_radius_m']-radius)*1000,
                     'minimum_motor_aabb_surface_gap_mm': minimum*1000,
                     'worst_sample': worst,
                     'omitting_external_guide_minimum_aabb_gap_mm': direct_minimum*1000})
    assert all(row['minimum_motor_aabb_surface_gap_mm'] > 0 for row in rows)
    return {'passed': True, 'angles_deg': [0, 90], 'sample_increment_deg': .5,
            'rope_radius_mm': radius*1000, 'routes': rows,
            'method': 'Exact capsule-centerline distance to conservative solid motor AABBs; positive distance certifies separation from the enclosed original mesh. Nonpositive AABB distance alone would not prove mesh penetration.',
            'continuous_angle_note': 'Upper candidate external-guide-to-tie segments remain at x <= -16.75 mm while motors satisfy x >= -16.25 mm, giving >=0.1 mm motor clearance. Lower direct exit-to-tie spans are checked over the stated sampled angles; the upper X-bound does not apply to them.',
            'limits': 'Candidate x=-25 mm external guides and front anchor usage are unconfirmed hardware; no rim-wrap contact, guide friction, guide support/bracket collision, folded steel contact, nor evolving inter-plate span clearance is certified. This is geometry only, not a new SOFA force/trajectory validation.'}


if __name__ == '__main__':
    result = verify()
    path = HERE/'cad_assembly_validation.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
