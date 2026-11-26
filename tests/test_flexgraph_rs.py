"""End-to-end verification of `flexgraph-rs` native Rust `cdylib` (`fg-ffi`),
zero-dependency Python `ctypes` wrapper (`src/flexgraph_rs/`), all 7 robot archetypes
across all 3 Schur routes, ROS URDF & MuJoCo MJCF loading, Parry3D corrugated pipe mesh
with Tight Inclusion CCD, Rust IFT reverse-mode adjoints, vectorized Rayon RL batching,
and `< 1e-10` FP64 parity against `serpworm.physics.gpu_core.GPUCore`.
"""
from __future__ import annotations

import numpy as np
import pytest

from flexgraph_rs import (
    FlexGraphRsBatchSolver,
    FlexGraphRsSolver,
    build_serpworm_envelope_json,
    check_serpworm_gpucore_parity,
)


def test_python_ctypes_zero_copy_views_and_all_5_archetypes():
    cases = [
        ("serpworm_1", "block_diagonal_pairs", 0, 0.0005),
        ("bristle_worm", "block_diagonal_pairs", 0, 0.001),
        ("tendon_endoscope_3d", "tree_chain", 0, 0.001),
        ("flexspine_centipede", "tree_chain", 0, 0.001),
        ("tensegrity_6bar", "sparse_general", 7, 0.001),
    ]
    for name, expected_route, expected_beta1, dt in cases:
        with FlexGraphRsSolver.from_archetype(name) as sim:
            info = sim.info
            assert info.schur_route == expected_route
            assert info.cyclomatic_number == expected_beta1

            targets = np.full(info.num_total_actuators, 0.02, dtype=np.float64)
            sim.set_targets(targets)
            diag = sim.step(dt=dt, steps=6)
            assert not diag.failed
            assert diag.step_index == 6
            assert diag.max_quat_norm_err < 1e-13

            # Zero-copy NumPy views are read-only and bit-exact with copy-out get_state()
            poses_view = sim.poses
            vels_view = sim.vels
            assert not poses_view.flags.writeable
            assert not vels_view.flags.writeable
            poses_copy, vels_copy = sim.get_state()
            np.testing.assert_array_equal(poses_view, poses_copy)
            np.testing.assert_array_equal(vels_view, vels_copy)
            assert np.all(np.isfinite(poses_view))
            assert np.all(np.isfinite(vels_view))

            # Energy/momentum and JSON round-trip
            em = sim.energy_momentum()
            assert np.isfinite(em.total_energy)
            raw_json = sim.export_graph_json()
            with FlexGraphRsSolver.from_json(raw_json) as sim2:
                assert sim2.info.total_nodes == info.total_nodes


def test_python_ctypes_urdf_flex_and_yaml_loading():
    urdf_xml = """<?xml version="1.0"?>
    <robot name="ctypes_cantilever">
      <fixed_base link="base"/>
      <link name="base" xyz="0 0 0.12">
        <inertial><mass value="0.5"/><inertia ixx="1e-3" iyy="1e-3" izz="1e-3"/></inertial>
      </link>
      <beam name="rod" body_a="base" n_elem="4" length="0.08" dir="1 0 0"
            section="circle" radius="0.0012" E="65e9" nu="0.33" rho="6450"/>
    </robot>"""
    with FlexGraphRsSolver.from_urdf_flex(urdf_xml, config={"world_sdf": "empty"}) as sim:
        z_start = float(sim.poses[-1, 2])
        diag = sim.step(dt=0.001, steps=15)
        assert not diag.failed
        # Cantilever tip droops under gravity while fixed base stays strictly pinned
        assert float(sim.poses[0, 2]) == pytest.approx(0.12, abs=0.0)
        assert float(sim.poses[-1, 2]) < z_start

    with FlexGraphRsSolver.from_yaml("archetype: tensegrity6bar\n") as sim_y:
        assert sim_y.info.schur_route == "sparse_general"
        assert sim_y.info.cyclomatic_number == 7


def test_vectorized_rayon_batch_rl_api_and_selective_reset():
    env_json = build_serpworm_envelope_json(n_segments=2, elements_per_strip=6, dt=0.0005)
    with FlexGraphRsBatchSolver.from_json(
        env_json,
        batch_size=4,
        dt=0.0005,
        num_threads=2,
        config={"solver_preset": "serpworm_kelvin_parity", "world_sdf": "kelvin_floor"},
    ) as batch:
        assert batch.batch_size == 4
        assert batch.num_bodies == 4
        assert batch.num_actuators == 5
        assert batch.num_beams == 16

        actions = np.array(
            [
                [0.6, 0.3, 0.4, 0.7, 0.15],
                [0.2, 0.8, 0.5, 0.5, -0.10],
                [0.9, 0.2, 0.3, 0.6, 0.20],
                [0.4, 0.4, 0.8, 0.2, -0.20],
            ],
            dtype=np.float64,
        )
        poses, vels, act_state, summary, failed = batch.step(substeps=8, actions=actions)
        assert poses.shape == (4, 4, 7)
        assert vels.shape == (4, 4, 6)
        assert act_state.shape == (4, 5, 3)
        assert summary.shape == (4, 7)
        assert failed.shape == (4,)
        assert np.all(failed == 0)
        assert np.all(np.isfinite(poses))
        assert np.all(np.isfinite(act_state))
        assert np.all(np.isfinite(summary))

        # Selective auto-reset on envs [0, 2] only
        batch.reset_mask([True, False, True, False])
        _, vels_after, _, _, _ = batch.step(substeps=0)
        assert float(np.max(np.abs(vels_after[0]))) == 0.0
        assert float(np.max(np.abs(vels_after[1]))) > 1e-4
        assert float(np.max(np.abs(vels_after[2]))) == 0.0
        assert float(np.max(np.abs(vels_after[3]))) > 1e-4


def test_fp64_exact_parity_free_flight_vs_serpworm_gpucore():
    from serpworm.config import HardwareConfig

    rep = check_serpworm_gpucore_parity(
        case_name="SerpWorm-3 free-flight bench_exp128",
        n_segments=3,
        elements_per_strip=8,
        steps=15,
        dt=0.0005,
        ground_contact=False,
        hardware=HardwareConfig.bench_exp128_calibrated(),
        tendon_targets=[0.9, 0.3, 0.5, 1.1, 0.7, 0.2],
        yaw_targets=[0.25, -0.30],
    )
    assert rep.max_pos_err_m < 1e-10, f"pos_err={rep.max_pos_err_m}"
    assert rep.max_quat_err < 1e-10, f"quat_err={rep.max_quat_err}"
    assert rep.max_tension_err_n < 1e-10, f"ten_err={rep.max_tension_err_n}"


def test_fp64_exact_parity_kelvin_ground_contact_vs_serpworm_gpucore():
    from serpworm.config import HardwareConfig

    rep = check_serpworm_gpucore_parity(
        case_name="SerpWorm-2 Kelvin active ratchet wheels",
        n_segments=2,
        elements_per_strip=6,
        steps=40,
        dt=0.0005,
        ground_contact=True,
        hardware=HardwareConfig.ds3218_default(),
        tendon_targets=[0.6, 0.6, 0.4, 0.8],
        yaw_targets=[0.15],
    )
    assert rep.max_support_n > 0.5, f"expected active wheel contact, got {rep.max_support_n}"
    assert rep.max_pos_err_m < 1e-10, f"pos_err={rep.max_pos_err_m}"
    assert rep.max_quat_err < 1e-10, f"quat_err={rep.max_quat_err}"
    assert rep.max_tension_err_n < 1e-8, f"ten_err={rep.max_tension_err_n}"
    assert rep.max_support_err_n < 1e-8, f"sup_err={rep.max_support_err_n}"


def test_fp64_exact_parity_arpc_hard_implicit_vs_serpworm_gpucore():
    from serpworm.config import HardwareConfig

    hw = HardwareConfig.ds3218_default().with_terrain("hard", implicit_contact=True)
    rep = check_serpworm_gpucore_parity(
        case_name="SerpWorm-2 ARPC hard implicit active",
        n_segments=2,
        elements_per_strip=6,
        steps=40,
        dt=0.0005,
        ground_contact=True,
        hardware=hw,
        tendon_targets=[0.6, 0.6, 0.4, 0.8],
        yaw_targets=[0.15],
    )
    assert rep.max_support_n > 0.5, f"expected active wheel contact, got {rep.max_support_n}"
    assert rep.max_pos_err_m < 1e-10, f"pos_err={rep.max_pos_err_m}"
    assert rep.max_quat_err < 1e-10, f"quat_err={rep.max_quat_err}"
    assert rep.max_tension_err_n < 1e-8, f"ten_err={rep.max_tension_err_n}"
    assert rep.max_support_err_n < 1e-8, f"sup_err={rep.max_support_err_n}"


def test_v2_new_archetypes_faer_and_clarabel_socp():
    with FlexGraphRsSolver.from_archetype("soft_octopus_arm_der") as octo:
        info = octo.info
        assert info.schur_route == "tree_chain"
        assert info.num_tendons == 4
        octo.set_targets([0.015, 0.0, 0.0, 0.0])
        diag = octo.step(dt=0.001, steps=8)
        assert not diag.failed
        assert np.all(np.isfinite(octo.poses))

    with FlexGraphRsSolver.from_archetype(
        "rigid_flex_grid_4x4", config={"sparse_backend": "faer"}
    ) as grid_faer, FlexGraphRsSolver.from_archetype(
        "rigid_flex_grid_4x4", config={"sparse_backend": "custom_block_csc"}
    ) as grid_csc:
        assert grid_faer.info.schur_route == "sparse_general"
        assert grid_faer.info.cyclomatic_number == 9
        grid_faer.set_targets([0.004, 0.002, 0.001, 0.003])
        grid_csc.set_targets([0.004, 0.002, 0.001, 0.003])
        d_f = grid_faer.step(dt=0.001, steps=5)
        d_c = grid_csc.step(dt=0.001, steps=5)
        assert not d_f.failed and not d_c.failed
        max_diff = float(np.max(np.abs(grid_faer.poses - grid_csc.poses)))
        assert max_diff < 1e-11, f"faer vs custom_block_csc pose mismatch: {max_diff}"


def test_v2_ros_urdf_mjcf_and_corrugated_pipe_mesh():
    ros_urdf = """<?xml version="1.0"?>
    <robot name="py_ros_pendulum">
      <link name="world"/>
      <link name="arm">
        <inertial><origin xyz="0.1 0 0"/><mass value="0.5"/>
          <inertia ixx="1e-3" ixy="0" ixz="0" iyy="2e-3" iyz="0" izz="2e-3"/>
        </inertial>
        <collision><geometry><capsule radius="0.02" length="0.20"/></geometry></collision>
      </link>
      <joint name="base_hinge" type="revolute">
        <parent link="world"/><child link="arm"/>
        <origin xyz="0 0 0.30" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.5" upper="1.5" effort="5.0" velocity="4.0"/>
      </joint>
    </robot>"""
    with FlexGraphRsSolver.from_ros_urdf(ros_urdf, config={"lcp_solver": "clarabel_socp"}) as sim:
        assert sim.info.num_bodies == 2
        sim.set_targets([0.4])
        diag = sim.step(dt=0.001, steps=10)
        assert not diag.failed
        assert np.all(np.isfinite(sim.poses))

    mjcf_xml = """<mujoco model="py_mjcf_hopper">
      <compiler angle="degree"/>
      <option timestep="0.001" gravity="0 0 -9.81"/>
      <worldbody>
        <body name="torso" pos="0 0 0.25">
          <freejoint name="root"/>
          <geom name="torso_s" type="sphere" size="0.05" solref="0.02 1.0"/>
          <body name="foot" pos="0 0 -0.10">
            <joint name="knee" type="hinge" axis="0 1 0" range="-45 45"/>
            <geom name="shin" type="capsule" fromto="0 0 0 0 0 -0.10" size="0.02"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="knee" gear="20"/></actuator>
    </mujoco>"""
    with FlexGraphRsSolver.from_mjcf(mjcf_xml) as sim_m:
        assert sim_m.info.num_bodies == 2
        sim_m.set_targets([0.2])
        diag = sim_m.step(dt=0.001, steps=10)
        assert not diag.failed
        assert np.all(np.isfinite(sim_m.poses))

    with FlexGraphRsSolver.from_archetype("bristle_worm") as bw_tmp:
        bw_json = bw_tmp.export_graph_json()
    with FlexGraphRsSolver.with_corrugated_pipe_mesh(
        bw_json,
        config={"use_tight_inclusion_ccd": True},
        length=1.2,
        r_mean=0.14,
        corrugation_amp=0.006,
        n_waves=6.0,
        origin=(0.2, 0.0, 0.08),
    ) as pipe_sim:
        diag = pipe_sim.step(dt=0.001, steps=10)
        assert not diag.failed
        assert np.all(np.isfinite(pipe_sim.poses))
        # All rigid bodies remain strictly inside the corrugated pipe cavity
        z_min = float(np.min(pipe_sim.poses[: pipe_sim.info.num_bodies, 2]))
        assert z_min > -0.06, f"crawler tunneled through corrugated pipe wall: z_min={z_min}"


def test_v2_ift_adjoint_grad_matches_finite_differences():
    # Fixed-base SoftOctopusArm-DER (tapered 3D DER arm + 4 tendons): tests Rust IFT reverse-mode
    # adjoint through C-ABI on BOTH actuator controls dL/du and physical Young's modulus dL/dE!
    cfg = {
        "use_strain_jacobian": True,
        "world_fixed_inertia": False,
        "contact": {"backend": "none"},
        "world_sdf": "empty",
    }
    with FlexGraphRsSolver.from_archetype("soft_octopus_arm_der", config=cfg) as sim:
        n_steps = 8
        n_act = sim.info.num_total_actuators
        dt = 0.001
        controls = np.zeros((n_steps, n_act), dtype=np.float64)
        controls[:, 0] = np.linspace(0.001, 0.006, n_steps)
        controls[:, 1] = np.linspace(0.0005, 0.003, n_steps)

        targets = np.zeros((n_steps + 1, sim.info.num_bodies, 7), dtype=np.float64)
        targets[0] = sim.poses[: sim.info.num_bodies]
        for k in range(n_steps):
            sim.set_targets(controls[k])
            sim.step(dt=dt, steps=1)
            targets[k + 1] = sim.poses[: sim.info.num_bodies]
        # Shift distal rings so both tendon actuation and elastic bending modulus carry strong signal
        targets[:, -1, 1] += 0.015
        targets[:, -1, 2] -= 0.012
        sim.reset()

        diff_params = [
            {"name": "beam_E", "kind": "beam_youngs_modulus", "target_ids": [], "scale": 1e6},
            {"name": "ten_k", "kind": "tendon_stiffness", "target_ids": [], "scale": 1e3},
        ]
        adj = sim.adjoint_grad(
            controls=controls,
            dt=dt,
            target_poses=targets,
            weight_pos=500.0,
            weight_rot=2.0,
            weight_ctrl=0.01,
            diff_params=diff_params,
        )
        assert adj.loss > 0.0 and np.isfinite(adj.loss)
        assert adj.grad_controls.shape == (n_steps, n_act)
        assert adj.grad_params.shape == (2,)

        def eval_loss(u_mat: np.ndarray) -> float:
            sim.reset()
            return sim.adjoint_grad(
                controls=u_mat,
                dt=dt,
                target_poses=targets,
                weight_pos=500.0,
                weight_rot=2.0,
                weight_ctrl=0.01,
                diff_params=[],
            ).loss

        eps_u = 1e-6
        u_p = controls.copy()
        u_m = controls.copy()
        u_p[0, 0] += eps_u
        u_m[0, 0] -= eps_u
        fd_u00 = (eval_loss(u_p) - eval_loss(u_m)) / (2.0 * eps_u)
        rel_u = abs(adj.grad_controls[0, 0] - fd_u00) / max(abs(fd_u00), 1e-8)
        assert (
            abs(fd_u00) > 1e-6 and rel_u < 1e-4
        ), f"dL/du[0,0] mismatch: adj={adj.grad_controls[0,0]}, fd={fd_u00}, rel={rel_u}"

        e0 = sim.get_diff_param(diff_params[0])
        eps_e = e0 * 1e-5
        sim.set_diff_param(diff_params[0], e0 + eps_e)
        lp = eval_loss(controls)
        sim.set_diff_param(diff_params[0], e0 - eps_e)
        lm = eval_loss(controls)
        sim.set_diff_param(diff_params[0], e0)
        fd_e = ((lp - lm) / (2.0 * eps_e)) * diff_params[0]["scale"]
        rel_e = abs(adj.grad_params[0] - fd_e) / max(abs(fd_e), 1e-8)
        assert (
            abs(fd_e) > 1e-6 and rel_e < 1e-4
        ), f"dL/dE mismatch: adj={adj.grad_params[0]}, fd={fd_e}, rel={rel_e}"
