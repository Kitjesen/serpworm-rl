"""Declarative Rigid-Flexible Graph IR (``RigidFlexGraph``), Sparsity Analyzer, and 4 Archetype Builders.

Represents any rigid-flexible hybrid or continuum robot as a typed bipartite multigraph:
    G = (V_rigid, E_beam, E_tendon, E_joint, C_site, G_sym)

Supports JSON/dict serialization, automatic rigid-graph sparsity routing
(``block_diagonal_pairs`` | ``tree_chain`` | ``sparse_general``), fixed bases,
free cantilever beams (``body_b = -1``), and four built-in heterogeneous robot archetypes:
  1. ``build_serpworm_graph(n_segments=...)``       -- exact 1:1 parity with ``serpworm.GPUCore``
  2. ``build_bristle_worm_graph(n_segments=...)``   -- wheelless anisotropic-scale limbless crawler
  3. ``build_tendon_endoscope_graph(n_sections=...)`` -- fixed-base 3D 4-tendon surgical catheter
  4. ``build_flexspine_crawler_graph(n_segments=...)`` -- leaf-spring compliant-spine articulated robot
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from flexgraph.math3d import compute_section_properties, mat_to_quat_np


@dataclass
class RigidBodySpec:
    name: str
    mass_kg: float
    inertia_diag: tuple[float, float, float]
    rest_pose_7d: tuple[float, float, float, float, float, float, float]
    fixed: bool = False
    collision_radius_m: float = 0.04


@dataclass
class BeamEdgeSpec:
    name: str
    body_a: int
    body_b: int  # -1 indicates free cantilever endpoint
    clamp_a_7d: tuple[float, float, float, float, float, float, float]
    clamp_b_7d: tuple[float, float, float, float, float, float, float]
    n_elem: int
    node_poses_7d: list[tuple[float, float, float, float, float, float, float]]  # length = n_elem + 1
    element_lengths_m: list[float]  # length = n_elem
    section_type: str = 'rect'  # 'rect' | 'circle' | 'custom'
    width_m: float = 0.016
    thickness_m: float = 0.00022
    radius_m: float = 0.0012
    area_m2: float | None = None
    iy_m4: float | None = None
    iz_m4: float | None = None
    j_m4: float | None = None
    youngs_modulus_pa: float = 200e9
    poisson_ratio: float = 0.3
    density_kg_m3: float = 7850.0
    rayleigh_alpha: float = 0.05
    rayleigh_beta: float = 0.002
    prestress_frac: float = 0.0
    thin_shell_compat: bool = False
    enable_node_contact: bool = False
    contact_node_radius_m: float = 0.002


@dataclass
class TendonEdgeSpec:
    name: str
    waypoints: list[tuple[int, tuple[float, float, float]]]  # [(body_idx, local_xyz), ...] length >= 2
    actuator_idx: int
    drive_mode: str = 'crank'  # 'crank' | 'legacy_wrap' | 'capstan' | 'linear'
    rest_length_m: float = 0.101
    horn_radius_m: float = 0.025
    guide_dist_m: float = 0.036
    slack_m: float = 0.0114
    stiffness_n_m: float = 8000.0
    damping_ns_m: float = 2.5
    capstan_mu_theta: float = 0.0
    torque_limit_nm: float = 2.45
    speed_limit_rad_s: float = 6.0
    deadband_rad: float = 0.0


@dataclass
class JointEdgeSpec:
    name: str
    body_a: int
    body_b: int
    joint_type: str = 'revolute'  # 'revolute' | 'stop_gap' | 'connector_revolute'
    anchor_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    actuator_idx: int = -1  # -1 = passive constraint
    kp: float = 12.0
    kd: float = 0.25
    k_pos: float = 10000.0
    c_pos: float = 10.0
    k_align: float = 20.0
    c_align: float = 0.1
    tau_max: float = 2.45
    omega_max: float = 6.0
    tau_static_fric: float = 0.0
    stop_gap_m: float = 0.075
    k_stop: float = 50000.0
    implicit_coupling: bool = False


@dataclass
class ContactSiteSpec:
    name: str
    owner_body: int  # rigid body index (beam interior node contacts are configured on BeamEdgeSpec)
    local_pos: tuple[float, float, float]
    contact_mode: str = 'ratchet_wheel'  # 'ratchet_wheel' | 'anisotropic_scale' | 'isotropic_pad'
    radius_m: float = 0.018
    patch_radius_m: float = 0.0
    mu_fwd: float = 0.80
    mu_rev: float = 0.80
    mu_lat: float = 0.80
    rolling: float = 0.06
    wheel_mass_kg: float = 0.0179
    backlash_rad: float = 0.0
    static_mu_ratio: float = 1.0
    pivot_mu: float = 0.65
    axle_local: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fwd_local: tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass
class ActuatorMeta:
    name: str
    act_type: str  # 'tendon' | 'joint'
    segment_norm: float  # normalized position along body [0, 1]
    side: float  # -1 right, 0 center, +1 left
    min_cmd: float
    max_cmd: float


@dataclass
class SymmetryGroupSpec:
    name: str
    act_perms: list[list[int]]
    act_signs: list[list[float]]
    est_perms: list[list[int]] = field(default_factory=list)
    est_signs: list[list[float]] = field(default_factory=list)
    obs_perms: list[list[int]] = field(default_factory=list)
    obs_signs: list[list[float]] = field(default_factory=list)

    @property
    def order(self) -> int:
        return len(self.act_perms)


@dataclass
class SparsityAnalysis:
    n_bodies: int
    n_free_bodies: int
    n_fixed_bodies: int
    n_beams: int
    n_beam_interior_nodes: int
    n_cantilever_beams: int
    total_dofs: int
    components: list[list[int]]
    is_acyclic: bool
    recommended_route: str  # 'block_diagonal_pairs' | 'tree_chain' | 'sparse_general'
    beam_elem_buckets: dict[int, list[int]]


@dataclass
class RigidFlexGraph:
    name: str
    bodies: list[RigidBodySpec]
    beams: list[BeamEdgeSpec]
    tendons: list[TendonEdgeSpec] = field(default_factory=list)
    joints: list[JointEdgeSpec] = field(default_factory=list)
    contact_sites: list[ContactSiteSpec] = field(default_factory=list)
    actuators: list[ActuatorMeta] = field(default_factory=list)
    symmetry: SymmetryGroupSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def analyze_sparsity(self) -> SparsityAnalysis:
        """Analyze bipartite rigid coupling graph G_R and determine optimal Schur routing."""
        n_b = len(self.bodies)
        adj: list[set[int]] = [set() for _ in range(n_b)]
        edge_set: set[tuple[int, int]] = set()
        buckets: dict[int, list[int]] = {}
        n_cant = 0
        n_int = 0

        for sid, b in enumerate(self.beams):
            buckets.setdefault(int(b.n_elem), []).append(sid)
            n_int += int(b.n_elem) - 1 if b.body_b >= 0 else int(b.n_elem)
            if b.body_b < 0:
                n_cant += 1
            else:
                u, v = int(b.body_a), int(b.body_b)
                if u != v and (not self.bodies[u].fixed) and (not self.bodies[v].fixed):
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((min(u, v), max(u, v)))

        for jt in self.joints:
            if jt.implicit_coupling:
                u, v = int(jt.body_a), int(jt.body_b)
                if u != v and (not self.bodies[u].fixed) and (not self.bodies[v].fixed):
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((min(u, v), max(u, v)))

        visited = [False] * n_b
        components: list[list[int]] = []
        free_bodies = [i for i, bd in enumerate(self.bodies) if not bd.fixed]
        for i in free_bodies:
            if visited[i]:
                continue
            comp = []
            q = deque([i])
            visited[i] = True
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nxt in sorted(adj[cur]):
                    if not visited[nxt]:
                        visited[nxt] = True
                        q.append(nxt)
            components.append(sorted(comp))

        # A simple undirected graph with V vertices, E edges, C components is a forest iff E == V - C
        is_acyclic = len(edge_set) == (len(free_bodies) - len(components))
        all_pairs = bool(components) and all(len(c) == 2 and c[1] == c[0] + 1 for c in components)
        if all_pairs and n_cant == 0 and len(free_bodies) == n_b:
            route = 'block_diagonal_pairs'
        elif is_acyclic:
            route = 'tree_chain'
        else:
            route = 'sparse_general'

        return SparsityAnalysis(
            n_bodies=n_b,
            n_free_bodies=len(free_bodies),
            n_fixed_bodies=n_b - len(free_bodies),
            n_beams=len(self.beams),
            n_beam_interior_nodes=n_int,
            n_cantilever_beams=n_cant,
            total_dofs=6 * (len(free_bodies) + n_int),
            components=components,
            is_acyclic=is_acyclic,
            recommended_route=route,
            beam_elem_buckets=buckets,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'RigidFlexGraph':
        sym = SymmetryGroupSpec(**d['symmetry']) if d.get('symmetry') else None
        return cls(
            name=d['name'],
            bodies=[RigidBodySpec(**x) for x in d['bodies']],
            beams=[
                BeamEdgeSpec(
                    **{
                        **x,
                        'clamp_a_7d': tuple(x['clamp_a_7d']),
                        'clamp_b_7d': tuple(x['clamp_b_7d']),
                        'node_poses_7d': [tuple(p) for p in x['node_poses_7d']],
                        'element_lengths_m': list(x['element_lengths_m']),
                    }
                )
                for x in d['beams']
            ],
            tendons=[
                TendonEdgeSpec(
                    **{**x, 'waypoints': [(int(w[0]), tuple(w[1])) for w in x['waypoints']]}
                )
                for x in d.get('tendons', [])
            ],
            joints=[
                JointEdgeSpec(
                    **{**x, 'anchor_a': tuple(x['anchor_a']), 'axis_local': tuple(x['axis_local'])}
                )
                for x in d.get('joints', [])
            ],
            contact_sites=[
                ContactSiteSpec(
                    **{
                        **x,
                        'local_pos': tuple(x['local_pos']),
                        'axle_local': tuple(x['axle_local']),
                        'fwd_local': tuple(x['fwd_local']),
                    }
                )
                for x in d.get('contact_sites', [])
            ],
            actuators=[ActuatorMeta(**x) for x in d.get('actuators', [])],
            symmetry=sym,
            metadata=dict(d.get('metadata', {})),
        )

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
        return p

    @classmethod
    def load_json(cls, path: str | Path) -> 'RigidFlexGraph':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


# ---------------------------------------------------------------------------
# Symmetry Group Generators (Z2, Z2xZ2, C4, D4)
# ---------------------------------------------------------------------------
def make_z2_bilateral_symmetry(n_segments: int) -> SymmetryGroupSpec:
    """Left-right reflection Z2 for N-segment modular snake-worm (2N tendons + (N-1) yaw)."""
    n_t = 2 * n_segments
    n_y = max(0, n_segments - 1)
    n_a = n_t + n_y
    id_p = list(range(n_a))
    id_s = [1.0] * n_a
    # Swap left/right tendon pairs (0<->1, 2<->3, ...); negate yaw commands
    ref_p = []
    ref_s = []
    for seg in range(n_segments):
        ref_p.extend([2 * seg + 1, 2 * seg])
        ref_s.extend([1.0, 1.0])
    for j in range(n_y):
        ref_p.append(n_t + j)
        ref_s.append(-1.0)
    # 7D EstNet [vx, vy, wz, wall_conf, wall_side, drag, slip]
    est_id_p = list(range(7))
    est_id_s = [1.0] * 7
    est_ref_s = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0]
    return SymmetryGroupSpec(
        name=f'Z2_bilateral_N{n_segments}',
        act_perms=[id_p, ref_p],
        act_signs=[id_s, ref_s],
        est_perms=[est_id_p, est_id_p],
        est_signs=[est_id_s, est_ref_s],
    )


def make_d4_radial_symmetry(n_actuators: int = 4) -> SymmetryGroupSpec:
    """Dihedral D4 symmetry (8 elements: 4 rotations x 2 reflections) for 4-tendon radial continuum arm.

    Assumes actuator ordering [0: +Y, 1: +Z, 2: -Y, 3: -Z] (90 deg CCW around +X backbone).
    """
    if n_actuators != 4:
        raise ValueError('D4 radial symmetry expects 4 actuators [+Y, +Z, -Y, -Z]')
    perms = [
        [0, 1, 2, 3],  # e
        [1, 2, 3, 0],  # r90
        [2, 3, 0, 1],  # r180
        [3, 0, 1, 2],  # r270
        [2, 1, 0, 3],  # reflect Y
        [0, 3, 2, 1],  # reflect Z
        [1, 0, 3, 2],  # reflect Y=Z
        [3, 2, 1, 0],  # reflect Y=-Z
    ]
    signs = [[1.0, 1.0, 1.0, 1.0] for _ in perms]
    return SymmetryGroupSpec(name='D4_radial_4tendon', act_perms=perms, act_signs=signs)


def make_z2xz2_bidirectional_symmetry(n_segments: int) -> SymmetryGroupSpec:
    """Klein four-group Z2 x Z2 (4 elements: identity, L-R mirror, F-B reverse, both) for reversible worms."""
    n_t = 2 * n_segments
    n_y = max(0, n_segments - 1)
    n_a = n_t + n_y
    g_id = (list(range(n_a)), [1.0] * n_a)
    # g_lr: swap L/R tendon pairs, negate yaw
    p_lr = [2 * s + (1 - k) for s in range(n_segments) for k in range(2)] + [n_t + j for j in range(n_y)]
    s_lr = [1.0] * n_t + [-1.0] * n_y
    # g_fb: reverse segment order (seg s -> N-1-s), swap L/R orientation & reverse yaw order
    p_fb = [2 * (n_segments - 1 - s) + (1 - k) for s in range(n_segments) for k in range(2)] + [
        n_t + (n_y - 1 - j) for j in range(n_y)
    ]
    s_fb = [1.0] * n_t + [1.0] * n_y
    # g_both = g_lr o g_fb
    p_both = [p_lr[i] for i in p_fb]
    s_both = [s_lr[p_fb[i]] * s_fb[i] for i in range(n_a)]
    return SymmetryGroupSpec(
        name=f'Z2xZ2_bidirectional_N{n_segments}',
        act_perms=[g_id[0], p_lr, p_fb, p_both],
        act_signs=[g_id[1], s_lr, s_fb, s_both],
    )


# ---------------------------------------------------------------------------
# Archetype 1: Exact 1:1 SerpWorm-N Builder (Parity-Exact with serpworm.GPUCore)
# ---------------------------------------------------------------------------
def build_serpworm_graph(
    n_segments: int = 5,
    elements_per_strip: int = 16,
    hardware: Any = None,
    parameters_path: Path | None = None,
) -> RigidFlexGraph:
    """Build exact RigidFlexGraph for N-segment pre-curved steel-strip SerpWorm (N >= 1).

    Matches ``serpworm.physics.topology.build_topology`` & ``GPUCore`` geometry, inertia,
    tendon anchors/guides, connector arms, and wheel mounts bit-for-bit.
    """
    from serpworm.paths import INPUTS_DIR
    from serpworm.physics.topology import WHEEL_DROP_M, WHEEL_HOLES_M, WHEEL_RADIUS_M, build_topology

    p_path = parameters_path or (INPUTS_DIR / 'parameters.json')
    params = json.loads(Path(p_path).read_text(encoding='utf-8'))
    if hardware is not None:
        if getattr(hardware, 'strip_width_m', None) is not None:
            params['strip_width_m'] = float(hardware.strip_width_m)
        if getattr(hardware, 'strip_thickness_m', None) is not None:
            params['strip_thickness_m'] = float(hardware.strip_thickness_m)
        if getattr(hardware, 'stress_free_bow_m', None) is not None:
            params['stress_free_bow_m'] = float(hardware.strip_free_bow_m if hasattr(hardware, 'strip_free_bow_m') else hardware.stress_free_bow_m)

    raw = build_topology(count=n_segments, elements_per_strip=elements_per_strip, parameters=params)
    rest = raw['rest']
    strip_nodes = raw['strip_nodes']
    transforms = raw['transforms']
    lengths = raw['lengths'].reshape(8 * n_segments, elements_per_strip)
    anchors = raw['anchors'].copy()
    guides = raw['guides'].copy()
    wheel_mounts = raw['wheel_mounts']

    # Hardware overrides (exact match to GPUCore.__init__)
    plate_masses_cfg = getattr(hardware, 'plate_masses_kg', None) if hardware is not None else None
    if plate_masses_cfg is not None and len(plate_masses_cfg) >= 2 * n_segments:
        plate_masses = [float(plate_masses_cfg[i]) for i in range(2 * n_segments)]
    else:
        plate_masses = [0.28] * (2 * n_segments)

    w_mu = float(getattr(hardware, 'wheel_mu', 0.8)) if hardware is not None else 0.8
    w_roll = float(getattr(hardware, 'wheel_rolling', 0.06)) if hardware is not None else 0.06
    w_mass = float(getattr(hardware, 'wheel_mass_kg', 0.009)) if hardware is not None else 0.009
    w_mu_lat = w_mu * (float(getattr(hardware, 'wheel_mu_lat_ratio', 1.0)) if hardware is not None else 1.0)
    w_patch_r = float(getattr(hardware, 'wheel_patch_radius_m', 0.0)) if hardware is not None else 0.0
    w_piv_mu = float(getattr(hardware, 'wheel_pivot_mu', 0.65)) if hardware is not None else 0.65
    w_backlash = float(getattr(hardware, 'ratchet_backlash_rad', 0.0)) if hardware is not None else 0.0
    w_stat_r = float(getattr(hardware, 'wheel_static_mu_ratio', 1.0)) if hardware is not None else 1.0

    r_alpha = float(getattr(hardware, 'rayleigh_alpha', 0.05)) if hardware is not None else 0.05
    r_beta = float(getattr(hardware, 'rayleigh_beta', 0.002)) if hardware is not None else 0.002
    thin_shell = bool(getattr(hardware, 'strip_thin_shell_compat', False)) if hardware is not None else False
    prestress = float(getattr(hardware, 'strip_prestress_frac', 0.0)) if hardware is not None else 0.0
    strip_con = bool(getattr(hardware, 'strip_node_contact', False)) if hardware is not None else False

    crank_mode = 'crank' if (hardware is not None and getattr(hardware, 'tendon_model', 'legacy_wrap') == 'crank_calibrated') else 'legacy_wrap'
    horn_r = float(getattr(hardware, 'tendon_horn_radius_m', 0.0235)) if hardware is not None else 0.0235
    guide_d = float(getattr(hardware, 'tendon_guide_dist_m', 0.035)) if hardware is not None else 0.035
    slack = float(getattr(hardware, 'tendon_slack_m', 0.0)) if hardware is not None else 0.0
    t_k = float(getattr(hardware, 'tendon_stiffness_n_m', 2000.0)) if hardware is not None else 2000.0
    t_c = float(getattr(hardware, 'tendon_damping_ns_m', 0.5)) if hardware is not None else 0.5
    t_y = float(getattr(hardware, 'tendon_y_m', 0.0285)) if hardware is not None else 0.0285
    cap_mt = float(getattr(hardware, 'tendon_capstan_mu_theta', 0.0)) if hardware is not None else 0.0
    sv_tlim = float(getattr(hardware, 'servo_torque_limit_nm', 0.5)) if hardware is not None else 0.5
    sv_slim = float(getattr(hardware, 'servo_speed_limit_rad_s', 2.0)) if hardware is not None else 2.0
    sv_db = float(getattr(hardware, 'servo_deadband_rad', 0.0)) if hardware is not None else 0.0
    stop_gap = float(getattr(hardware, 'plate_stop_gap_m', 0.075)) if hardware is not None else 0.075

    y_kp = float(getattr(hardware, 'yaw_kp_nm_rad', 2.0)) if hardware is not None else 2.0
    y_kd = float(getattr(hardware, 'yaw_kd_nms_rad', 0.03)) if hardware is not None else 0.03
    y_tlim = float(getattr(hardware, 'yaw_torque_limit_nm', 0.5)) if hardware is not None else 0.5
    y_slim = float(getattr(hardware, 'yaw_speed_limit_rad_s', -1.0)) if hardware is not None else -1.0
    y_sf = float(getattr(hardware, 'yaw_static_friction_nm', 0.0)) if hardware is not None else 0.0

    if abs(t_y - 0.0285) > 1e-9:
        anchors[:, 1] = np.array([t_y, t_y, -t_y, -t_y], dtype=np.float64)
        guides[:, 1] = anchors[:, 1]

    bodies: list[RigidBodySpec] = []
    base_inert = np.array([0.0016, 0.0009, 0.0009], dtype=np.float64) * 0.28
    for p in range(2 * n_segments):
        m_kg = plate_masses[p]
        inert = tuple((base_inert * (m_kg / 0.28)).tolist())
        bodies.append(
            RigidBodySpec(
                name=f'plate_{p}',
                mass_kg=m_kg,
                inertia_diag=inert,
                rest_pose_7d=tuple(rest[p].tolist()),
                fixed=False,
                collision_radius_m=0.055,
            )
        )

    beams: list[BeamEdgeSpec] = []
    for sid in range(8 * n_segments):
        seg = sid // 8
        nds = strip_nodes[sid]
        poses_7d = [tuple(rest[int(nid)].tolist()) for nid in nds]
        beams.append(
            BeamEdgeSpec(
                name=f'seg{seg}_strip{sid % 8}',
                body_a=2 * seg,
                body_b=2 * seg + 1,
                clamp_a_7d=tuple(transforms[sid, 0, 0].tolist()),
                clamp_b_7d=tuple(transforms[sid, -1, 1].tolist()),
                n_elem=elements_per_strip,
                node_poses_7d=poses_7d,
                element_lengths_m=lengths[sid].tolist(),
                section_type='rect',
                width_m=float(params['strip_width_m']),
                thickness_m=float(params['strip_thickness_m']),
                youngs_modulus_pa=float(params['youngs_modulus_pa']),
                poisson_ratio=float(params['poisson_ratio']),
                density_kg_m3=float(params['steel_density_kg_m3']),
                rayleigh_alpha=r_alpha,
                rayleigh_beta=r_beta,
                prestress_frac=prestress,
                thin_shell_compat=thin_shell,
                enable_node_contact=strip_con,
                contact_node_radius_m=0.002,
            )
        )

    tendons: list[TendonEdgeSpec] = []
    actuators: list[ActuatorMeta] = []
    for seg in range(n_segments):
        seg_norm = seg / max(n_segments - 1, 1)
        for side_i, side_lbl in enumerate(('left', 'right')):
            act_id = 2 * seg + side_i
            actuators.append(
                ActuatorMeta(
                    name=f'seg{seg}_tendon_{side_lbl}',
                    act_type='tendon',
                    segment_norm=seg_norm,
                    side=+1.0 if side_i == 0 else -1.0,
                    min_cmd=0.0,
                    max_cmd=math.pi,
                )
            )
            for k_pair in range(2):
                k = 2 * side_i + k_pair
                tendons.append(
                    TendonEdgeSpec(
                        name=f'seg{seg}_tendon_{side_lbl}_{k_pair}',
                        waypoints=[(2 * seg, tuple(anchors[k].tolist())), (2 * seg + 1, tuple(guides[k].tolist()))],
                        actuator_idx=act_id,
                        drive_mode=crank_mode,
                        rest_length_m=0.101,
                        horn_radius_m=horn_r,
                        guide_dist_m=guide_d,
                        slack_m=slack,
                        stiffness_n_m=t_k,
                        damping_ns_m=t_c,
                        capstan_mu_theta=cap_mt,
                        torque_limit_nm=sv_tlim,
                        speed_limit_rad_s=sv_slim,
                        deadband_rad=sv_db,
                    )
                )

    joints: list[JointEdgeSpec] = []
    for seg in range(n_segments):
        joints.append(
            JointEdgeSpec(
                name=f'seg{seg}_stop_gap',
                body_a=2 * seg,
                body_b=2 * seg + 1,
                joint_type='stop_gap',
                stop_gap_m=stop_gap,
                k_stop=50000.0,
            )
        )
    for j in range(max(0, n_segments - 1)):
        act_id = 2 * n_segments + j
        actuators.append(
            ActuatorMeta(
                name=f'yaw_{j}',
                act_type='joint',
                segment_norm=(j + 0.5) / max(n_segments - 1, 1),
                side=0.0,
                min_cmd=-0.75,
                max_cmd=+0.75,
            )
        )
        joints.append(
            JointEdgeSpec(
                name=f'inter_seg_yaw_{j}',
                body_a=2 * j + 1,
                body_b=2 * j + 2,
                joint_type='connector_revolute',
                anchor_a=(-0.058, 0.0, -0.0010452742534),
                axis_local=(0.0, 0.0, 1.0),
                actuator_idx=act_id,
                kp=y_kp,
                kd=y_kd,
                k_pos=10000.0,
                c_pos=10.0,
                k_align=20.0,
                c_align=0.1,
                tau_max=y_tlim,
                omega_max=y_slim,
                tau_static_fric=y_sf,
            )
        )

    contact_sites: list[ContactSiteSpec] = []
    for p in range(2 * n_segments):
        for s in range(2):
            pos = tuple(wheel_mounts[p % 2, s].tolist())
            contact_sites.append(
                ContactSiteSpec(
                    name=f'plate_{p}_wheel_{s}',
                    owner_body=p,
                    local_pos=pos,
                    contact_mode='ratchet_wheel',
                    radius_m=float(WHEEL_RADIUS_M),
                    patch_radius_m=w_patch_r,
                    mu_fwd=w_mu,
                    mu_rev=w_mu,
                    mu_lat=w_mu_lat,
                    rolling=w_roll,
                    wheel_mass_kg=w_mass,
                    backlash_rad=w_backlash,
                    static_mu_ratio=w_stat_r,
                    pivot_mu=w_piv_mu,
                )
            )

    return RigidFlexGraph(
        name=f'serpworm_N{n_segments}',
        bodies=bodies,
        beams=beams,
        tendons=tendons,
        joints=joints,
        contact_sites=contact_sites,
        actuators=actuators,
        symmetry=make_z2_bilateral_symmetry(n_segments),
        metadata={'archetype': 'serpworm', 'n_segments': n_segments, 'elements_per_strip': elements_per_strip},
    )


# ---------------------------------------------------------------------------
# Archetype 2: Wheelless Anisotropic-Scale / Bristle Earthworm-Snake Crawler
# ---------------------------------------------------------------------------
def build_bristle_worm_graph(
    n_segments: int = 5,
    elements_per_strip: int = 10,
    mu_fwd: float = 0.14,
    mu_rev: float = 0.88,
    mu_lat: float = 0.92,
) -> RigidFlexGraph:
    """Wheelless limbless crawler replacing passive wheels with anisotropic ventral scales / bristles."""
    g = build_serpworm_graph(n_segments=n_segments, elements_per_strip=elements_per_strip)
    g.name = f'bristle_worm_N{n_segments}'
    g.symmetry = make_z2xz2_bidirectional_symmetry(n_segments)
    new_sites: list[ContactSiteSpec] = []
    for idx, site in enumerate(g.contact_sites):
        new_sites.append(
            ContactSiteSpec(
                name=f'scale_pad_{idx}',
                owner_body=site.owner_body,
                local_pos=(site.local_pos[0], site.local_pos[1], site.local_pos[2] - 0.008),
                contact_mode='anisotropic_scale',
                radius_m=0.010,
                patch_radius_m=0.014,
                mu_fwd=mu_fwd,
                mu_rev=mu_rev,
                mu_lat=mu_lat,
                rolling=0.0,
                wheel_mass_kg=0.0,
                backlash_rad=0.0,
                static_mu_ratio=1.15,
                pivot_mu=0.50,
                # Robot crawls along -X (matching SerpWorm wheel fwd = cross(+Z, +Y) = -X and HeightFieldSDF arc_s = -x)
                fwd_local=(-1.0, 0.0, 0.0),
            )
        )
    g.contact_sites = new_sites
    g.metadata['archetype'] = 'bristle_worm'
    return g


# ---------------------------------------------------------------------------
# Archetype 3: Fixed-Base 3D 4-Tendon Continuum Surgical Endoscope / Arm
# ---------------------------------------------------------------------------
def build_tendon_endoscope_graph(
    n_sections: int = 4,
    elements_per_section: int = 8,
    section_length_m: float = 0.060,
    disc_radius_m: float = 0.012,
    backbone_radius_m: float = 0.0014,
    free_cantilever_tip: bool = False,
) -> RigidFlexGraph:
    """3D fixed-base tendon-driven continuum manipulator / endoscope with nitinol rod & 4 routing tendons."""
    bodies: list[RigidBodySpec] = []
    n_discs = n_sections if free_cantilever_tip else (n_sections + 1)
    for i in range(n_discs):
        x_pos = i * section_length_m
        m_disc = 0.012 if i > 0 else 0.100
        I_disc = 0.5 * m_disc * disc_radius_m**2
        bodies.append(
            RigidBodySpec(
                name='base_fixed' if i == 0 else f'spacer_disc_{i}',
                mass_kg=m_disc,
                inertia_diag=(I_disc, 0.5 * I_disc, 0.5 * I_disc),
                rest_pose_7d=(x_pos, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0),
                fixed=(i == 0),
                collision_radius_m=disc_radius_m,
            )
        )

    beams: list[BeamEdgeSpec] = []
    for s in range(n_sections):
        x0 = s * section_length_m
        x1 = (s + 1) * section_length_m
        us = np.linspace(0.0, 1.0, elements_per_section + 1)
        poses = [(float(x0 + u * (x1 - x0)), 0.0, 0.12, 0.0, 0.0, 0.0, 1.0) for u in us]
        lens = [section_length_m / elements_per_section] * elements_per_section
        is_cant = free_cantilever_tip and (s == n_sections - 1)
        beams.append(
            BeamEdgeSpec(
                name=f'nitinol_backbone_{s}',
                body_a=s,
                body_b=-1 if is_cant else (s + 1),
                clamp_a_7d=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                clamp_b_7d=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                n_elem=elements_per_section,
                node_poses_7d=poses,
                element_lengths_m=lens,
                section_type='circle',
                radius_m=backbone_radius_m,
                youngs_modulus_pa=65e9,  # Nitinol superelastic modulus
                poisson_ratio=0.33,
                density_kg_m3=6450.0,
                rayleigh_alpha=0.10,
                rayleigh_beta=0.001,
                enable_node_contact=True,
                contact_node_radius_m=disc_radius_m * 0.5,
            )
        )

    # 4 full-length multi-waypoint routing tendons (+Y, +Z, -Y, -Z) with D4 symmetry
    tendon_offsets = [
        ('t_pos_y', (0.0, +0.009, 0.0), +1.0),
        ('t_pos_z', (0.0, 0.0, +0.009), 0.0),
        ('t_neg_y', (0.0, -0.009, 0.0), -1.0),
        ('t_neg_z', (0.0, 0.0, -0.009), 0.0),
    ]
    tendons: list[TendonEdgeSpec] = []
    actuators: list[ActuatorMeta] = []
    total_len = (n_discs - 1) * section_length_m
    for idx, (tname, off, side_v) in enumerate(tendon_offsets):
        actuators.append(
            ActuatorMeta(name=tname, act_type='tendon', segment_norm=1.0, side=side_v, min_cmd=0.0, max_cmd=0.025)
        )
        tendons.append(
            TendonEdgeSpec(
                name=tname,
                waypoints=[(b_i, off) for b_i in range(n_discs)],
                actuator_idx=idx,
                drive_mode='linear',
                rest_length_m=total_len,
                slack_m=0.0,
                stiffness_n_m=3000.0,
                damping_ns_m=1.2,
                capstan_mu_theta=0.12,
                torque_limit_nm=2.0,
                speed_limit_rad_s=0.15,
            )
        )

    contact_sites = [
        ContactSiteSpec(
            name=f'disc_{i}_rim',
            owner_body=i,
            local_pos=(0.0, 0.0, 0.0),
            contact_mode='isotropic_pad',
            radius_m=disc_radius_m,
            patch_radius_m=0.004,
            mu_fwd=0.22,
            mu_rev=0.22,
            mu_lat=0.22,
            pivot_mu=0.25,
        )
        for i in range(1, n_discs)
    ]

    return RigidFlexGraph(
        name=f'tendon_endoscope_3d_S{n_sections}',
        bodies=bodies,
        beams=beams,
        tendons=tendons,
        joints=[],
        contact_sites=contact_sites,
        actuators=actuators,
        symmetry=make_d4_radial_symmetry(4),
        metadata={'archetype': 'tendon_endoscope_3d', 'n_sections': n_sections},
    )


# ---------------------------------------------------------------------------
# Archetype 4: Compliant-Spine Multi-Segment Articulated Crawler (Leaf-Spring Spine + Legs)
# ---------------------------------------------------------------------------
def build_flexspine_crawler_graph(
    n_segments: int = 3,
    elements_per_spine: int = 8,
    seg_pitch_m: float = 0.14,
) -> RigidFlexGraph:
    """Rigid torso segments joined by dual parallel steel leaf-spring spine beams + revolute leg feet."""
    bodies: list[RigidBodySpec] = []
    contact_sites: list[ContactSiteSpec] = []
    joints: list[JointEdgeSpec] = []
    actuators: list[ActuatorMeta] = []

    z_body = 0.075
    for s in range(n_segments):
        x_s = -s * seg_pitch_m
        b_idx = len(bodies)
        bodies.append(
            RigidBodySpec(
                name=f'torso_{s}',
                mass_kg=0.35,
                inertia_diag=(0.0005, 0.0006, 0.0008),
                rest_pose_7d=(x_s, 0.0, z_body, 0.0, 0.0, 0.0, 1.0),
                fixed=False,
                collision_radius_m=0.04,
            )
        )
        # Left & right rigid leg links attached via revolute hip joints
        for side_i, y_sgn in enumerate((+1.0, -1.0)):
            leg_idx = len(bodies)
            bodies.append(
                RigidBodySpec(
                    name=f'torso_{s}_leg_{"L" if side_i == 0 else "R"}',
                    mass_kg=0.06,
                    inertia_diag=(0.00008, 0.00008, 0.00005),
                    rest_pose_7d=(x_s, y_sgn * 0.065, z_body - 0.030, 0.0, 0.0, 0.0, 1.0),
                    fixed=False,
                    collision_radius_m=0.015,
                )
            )
            act_id = len(actuators)
            actuators.append(
                ActuatorMeta(
                    name=f'hip_{s}_{"L" if side_i == 0 else "R"}',
                    act_type='joint',
                    segment_norm=s / max(n_segments - 1, 1),
                    side=y_sgn,
                    min_cmd=-0.6,
                    max_cmd=+0.6,
                )
            )
            joints.append(
                JointEdgeSpec(
                    name=f'hip_joint_{s}_{side_i}',
                    body_a=b_idx,
                    body_b=leg_idx,
                    joint_type='connector_revolute',
                    anchor_a=(0.0, y_sgn * 0.065, -0.030),
                    axis_local=(0.0, 1.0, 0.0),
                    actuator_idx=act_id,
                    kp=8.0,
                    kd=0.15,
                    k_pos=12000.0,
                    c_pos=15.0,
                    k_align=25.0,
                    c_align=0.15,
                    tau_max=1.8,
                    omega_max=6.0,
                    tau_static_fric=0.02,
                    implicit_coupling=True,
                )
            )
            contact_sites.append(
                ContactSiteSpec(
                    name=f'foot_{s}_{side_i}',
                    owner_body=leg_idx,
                    local_pos=(0.0, 0.0, -0.032),
                    contact_mode='isotropic_pad',
                    radius_m=0.012,
                    patch_radius_m=0.012,
                    mu_fwd=0.85,
                    mu_rev=0.85,
                    mu_lat=0.85,
                    static_mu_ratio=1.12,
                    pivot_mu=0.55,
                )
            )

    # Dual parallel compliant steel leaf-spring beams connecting consecutive torso bodies (indices 0, 3, 6, ...)
    beams: list[BeamEdgeSpec] = []
    torso_ids = [3 * s for s in range(n_segments)]
    for s in range(n_segments - 1):
        ta, tb = torso_ids[s], torso_ids[s + 1]
        xa, xb = bodies[ta].rest_pose_7d[0], bodies[tb].rest_pose_7d[0]
        for y_off, lbl in ((+0.020, 'L'), (-0.020, 'R')):
            a_pt = np.array([xa - 0.025, y_off, z_body], dtype=np.float64)
            b_pt = np.array([xb + 0.025, y_off, z_body], dtype=np.float64)
            us = np.linspace(0.0, 1.0, elements_per_spine + 1)
            poses = []
            for u in us:
                pt = a_pt + u * (b_pt - a_pt)
                # Beam local +X points from body_a toward body_b (-world X)
                q = mat_to_quat_np(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64))
                poses.append((*pt.tolist(), *q.tolist()))
            seg_len = float(np.linalg.norm(b_pt - a_pt)) / elements_per_spine
            beams.append(
                BeamEdgeSpec(
                    name=f'spine_{s}_{lbl}',
                    body_a=ta,
                    body_b=tb,
                    clamp_a_7d=(-0.025, y_off, 0.0, *poses[0][3:]),
                    clamp_b_7d=(+0.025, y_off, 0.0, *poses[-1][3:]),
                    n_elem=elements_per_spine,
                    node_poses_7d=poses,
                    element_lengths_m=[seg_len] * elements_per_spine,
                    section_type='rect',
                    width_m=0.018,
                    thickness_m=0.0006,
                    youngs_modulus_pa=200e9,
                    poisson_ratio=0.3,
                    density_kg_m3=7850.0,
                    rayleigh_alpha=0.08,
                    rayleigh_beta=0.001,
                )
            )

    n_a = len(actuators)
    id_p = list(range(n_a))
    id_s = [1.0] * n_a
    ref_p = [2 * s + (1 - k) for s in range(n_segments) for k in range(2)]
    ref_s = [1.0] * n_a
    est_id_p = list(range(7))
    sym = SymmetryGroupSpec(
        name=f'Z2_bilateral_flexspine_N{n_segments}',
        act_perms=[id_p, ref_p],
        act_signs=[id_s, ref_s],
        est_perms=[est_id_p, est_id_p],
        est_signs=[[1.0] * 7, [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0]],
    )

    return RigidFlexGraph(
        name=f'flexspine_crawler_N{n_segments}',
        bodies=bodies,
        beams=beams,
        tendons=[],
        joints=joints,
        contact_sites=contact_sites,
        actuators=actuators,
        symmetry=sym,
        metadata={'archetype': 'flexspine_crawler', 'n_segments': n_segments},
    )
