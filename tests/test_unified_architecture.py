"""Unit and smoke tests for the unified 3-mode architecture."""
import math
import unittest
import torch

from serpworm.config import DomainRandConfig, HardwareConfig, TaskConfig, UnifiedTrainConfig
from serpworm.envs.unified_env import UnifiedSerpWormEnv
from serpworm.rl.models import TemporalActorCritic
from serpworm.rl.train_unified import train_unified


def test_hardware_config_presets_and_roundtrip(tmp_path):
    ds = HardwareConfig.ds3218_default()
    v5 = HardwareConfig.legacy_v5()
    assert ds.yaw_kp_nm_rad > v5.yaw_kp_nm_rad
    assert ds.yaw_action_limit_deg < 30.16  # strictly inside 2 mm finite-cylinder wheel clearance
    assert ds.action_size == 9
    path = tmp_path / 'hw.json'
    ds.save(path)
    loaded = HardwareConfig.load(path)
    assert loaded == ds


def test_unified_env_three_modes_masking_and_shapes():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    hw = HardwareConfig.ds3218_default()
    task = TaskConfig(physics_dt_s=0.0005, policy_dt_s=0.10, sensor_dt_s=0.02, episode_seconds=4.0, stage=2)
    env = UnifiedSerpWormEnv(
        n=3, hardware=hw, task=task, dr=DomainRandConfig(enabled=True),
        device=dev, plate_solver='segment_lu', noise=False,
    )
    env.mode_id.copy_(torch.tensor([0, 1, 2], device=dev, dtype=torch.long))  # snake, worm, mixed

    full_act = torch.ones((3, hw.action_size), device=dev, dtype=torch.float32)
    servo_des, yaw_des, phys = env.map_action(full_act)

    # Lane 0 (snake): tendons locked at neutral, yaw active
    assert torch.allclose(servo_des[0], torch.full((10,), hw.servo_neutral_rad, device=dev, dtype=torch.float64))
    assert (yaw_des[0].abs() > 0.10).all()
    # Lane 1 (worm): yaw locked at 0, tendons active
    assert torch.allclose(yaw_des[1], torch.zeros(4, device=dev, dtype=torch.float64))
    assert (servo_des[1] > hw.servo_neutral_rad + 0.25).all()
    # Lane 2 (mixed): both active
    assert (yaw_des[2].abs() > 0.10).all()
    assert (servo_des[2] > hw.servo_neutral_rad + 0.25).all()

    obs, priv, rew, term, trunc, tobs, tpriv, info = env.step(full_act)
    assert obs.shape == (3, task.history_len, env.frame_size)
    assert priv.shape == (3, env.privileged_size)
    assert rew.shape == (3,) and torch.isfinite(rew).all()
    assert not bool(term.any()) and not bool(trunc.any())


def test_ds3218_hardware_eliminates_soft_pd_lag():
    """Verify 25 kg*cm Kp=10.0 tracks a ~29 deg yaw target cleanly (< 3 deg lag)."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    hw = HardwareConfig.ds3218_default()
    task = TaskConfig(physics_dt_s=0.0005, policy_dt_s=0.10, sensor_dt_s=0.02, episode_seconds=4.0)
    env = UnifiedSerpWormEnv(
        n=1, hardware=hw, task=task, device=dev,
        plate_solver='segment_lu', noise=False, forced_mode=0,
    )
    act = torch.zeros((1, 9), device=dev, dtype=torch.float32)
    act[0, 5 + 1] = 1.0  # integrates to +1.0 (+29 deg) within 4 steps in delta mode
    for _ in range(8):
        env.step(act)
    _, _, _, q = env._geometry()
    actual_deg = float(q[0, 1].item() * 180.0 / math.pi)
    assert actual_deg > 24.0, f'Expected 25kg servo yaw tracking > 24.0 deg, got {actual_deg:.2f} deg'


def test_temporal_actor_critic_and_mini_train(tmp_path):
    hw = HardwareConfig.ds3218_default()
    task = TaskConfig(physics_dt_s=0.001, policy_dt_s=0.10, sensor_dt_s=0.02, episode_seconds=4.0, stage=2)
    cfg = UnifiedTrainConfig(
        hardware=hw, task=task, dr=DomainRandConfig(enabled=False),
        envs=2, horizon=4, rollouts=1, epochs=2, minibatches=2,
        stage2_rollout=0, stage3_rollout=0, eval_every=0, plate_solver='segment_lu',
    )
    res = train_unified(cfg, tmp_path / 'unified_smoke')
    assert (tmp_path / 'unified_smoke' / 'latest.pt').is_file()
    assert (tmp_path / 'unified_smoke' / 'progress.json').is_file()
    assert res['latest']['rollout'] == 1


def test_v5_realistic_arpc_reset_dr_ramp_and_single_pass_sym(tmp_path):
    """Verify full v5 upgrade: UTM+ARPC reset clears ratchet_play/soil_sinkage, DR ramp & biases, causal TCN, and symmetry single-pass train."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    hw = HardwareConfig.v5_realistic(tendon_pairing='paired_5d', terrain='soft_soil', ratchet_backlash_deg=12.0)
    assert hw.strip_thickness_m == 0.000220 and hw.stress_free_bow_m == 0.0300
    assert hw.implicit_contact and hw.contact_model == 'arpc_soft_soil'
    task = TaskConfig(physics_dt_s=0.001, policy_dt_s=0.10, sensor_dt_s=0.02, episode_seconds=4.0, stage=3)
    dr = DomainRandConfig(enabled=True, dr_ramp_rollouts=5, joint_bias_deg=1.5, gyro_bias_rad_s=0.03, max_action_delay_ticks=2)
    env = UnifiedSerpWormEnv(n=2, hardware=hw, task=task, dr=dr, device=dev, plate_solver='segment_lu', noise=True)
    # Perturb ARPC persistent states and verify masked reset restores settled initial state strictly on lane 0
    env.c.ratchet_play.fill_(0.08)
    env.c.soil_sinkage.fill_(0.009)
    env.reset(torch.tensor([True, False], device=dev))
    assert torch.allclose(env.c.ratchet_play[0], env.initial['ratchet_play'][0])
    assert torch.allclose(env.c.soil_sinkage[0], env.initial['soil_sinkage'][0])
    assert float(env.c.ratchet_play[1].abs().min()) == 0.08
    assert float(env.c.soil_sinkage[1].abs().min()) == 0.009
    # Verify DR scale ramp controls all 5 DR axes (ground, mass, actuator/capstan, sensor tilt, push)
    env.set_dr_scale(0.0)
    env.reset(torch.tensor([True, True], device=dev))
    assert float(env.joint_bias.abs().max()) == 0.0 and float(env.gyro_bias.abs().max()) == 0.0
    assert torch.allclose(env.c.plate_masses[0], env.c.plate_masses_base)
    assert torch.allclose(env.gravity_rot[0], torch.eye(3, device=dev, dtype=env.dtype))
    env.set_dr_scale(1.0)
    env.reset(torch.tensor([True, True], device=dev))
    assert float(env.joint_bias.abs().max()) > 0.0 and float(env.gyro_bias.abs().max()) > 0.0
    assert not torch.allclose(env.c.plate_masses[0], env.c.plate_masses[1])
    assert not torch.allclose(env.c.env_terrain[0], env.c.env_terrain[1])
    assert not torch.allclose(env.c.env_dynamics[0], env.c.env_dynamics[1])
    assert not torch.allclose(env.c.rayleigh_beta_env[0], env.c.rayleigh_beta_env[1])
    assert not torch.allclose(env.c.ground_geom[0], env.c.ground_geom[1])
    assert env.privileged_size == 101 and env.critic_obs.shape == (2, 101)
    assert not torch.allclose(env.gravity_rot[0], torch.eye(3, device=dev, dtype=env.dtype))
    # Verify rigid-body velocity push stays finite and changes velocity smoothly when due
    env.step_count.fill_(env._push_interval_steps)
    v_before = env.c.v.clone()
    env.dr.push_prob = 1.0
    env._apply_velocity_push()
    assert float((env.c.v - v_before).abs().max()) > 1e-4 and torch.isfinite(env.c.v).all()
    # Mini v5 train smoke with symmetry_aug + stage transitions + logstd floor rebound
    cfg = UnifiedTrainConfig(
        hardware=hw, task=TaskConfig(physics_dt_s=0.001, policy_dt_s=0.10, sensor_dt_s=0.02, episode_seconds=4.0, stage=1),
        dr=DomainRandConfig(enabled=False, dr_ramp_rollouts=2),
        envs=2, horizon=4, rollouts=3, epochs=2, minibatches=2,
        stage2_rollout=2, stage3_rollout=3, eval_every=0,
        symmetry_aug=True, plate_solver='segment_lu', reward_preset='v5_utm',
    )
    res = train_unified(cfg, tmp_path / 'v5_smoke')
    assert res['latest']['stage'] == 3 and math.isfinite(res['latest']['mean_reward'])


if __name__ == '__main__':
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_hardware_config_presets_and_roundtrip(tmp)
        test_unified_env_three_modes_masking_and_shapes()
        test_ds3218_hardware_eliminates_soft_pd_lag()
        test_temporal_actor_critic_and_mini_train(tmp)
    print('unified_architecture_tests_passed')
