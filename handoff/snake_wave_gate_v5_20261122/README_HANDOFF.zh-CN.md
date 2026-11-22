# Steel Snake v5 交接包

这个目录是 `experiments/snake_wave_gate_v5_20261105` 的可交接副本，包含当前 v5 的动力学代码、五体节输入模型、固定波验证、行为克隆、PPO、评估和测试。

## 运行环境

- Windows 或 Linux
- Python 3.10 或更高版本
- PyTorch 2.8，GPU 运行需要 CUDA 12.8 兼容环境
- 推荐 GPU 为 NVIDIA RTX 5090 或同等级 CUDA 设备
- 物理步长为 `0.00025 s`
- 传感器频率为 50 Hz
- 策略频率为 5 Hz，每次策略动作保持 0.2 s

本目录中的 `validation.json` 是 CPU 数值检查记录。当前记录显示本地 CPU 检查通过，CUDA 固定波验证、教师行为克隆和新 PPO 仍需在可用 CUDA 机器上执行。

## 文件组织

```text
gpu_core.py                梁、隔板、绳索、关节和轮地动力学
tensor_env.py              五体节环境、观测、动作映射和重置
wheel_stiction.py          轮地黏着和滑移
wheel_envelope.py          轮间几何分离保护
rewards.py                 sustain_v4 奖励
train_gpu.py               直接 PPO
evaluate_gpu.py            确定性和随机策略评估
pretrain_actor.py          固定波行为克隆初始化
run_pipeline.py            固定波到 PPO 的总入口
probe_fixed_waves.py       固定蛇形波和混合波扫描
probe_policy_distribution.py 旧策略均值/随机动作对照
inputs/                    parameters.json 和 topology_5.npz
diagnostics/               轻量诊断结果和拓扑图
test_*.py                  合同测试
```

## 第一步，运行本地合同测试

```powershell
cd D:\doso\robot\steel-snake-gpu\handoff\snake_wave_gate_v5_20261122
python validate_local.py
```

也可以单独运行：

```powershell
python test_ppo.py --device cpu
python test_rewards.py --device cpu
python test_evaluation.py
python test_wave_probe.py
python test_pretrain.py
python test_pipeline.py
python test_policy_distribution.py
python test_wheel_envelope.py
```

## 第二步，运行固定波验证

固定波验证必须在 CUDA 环境运行。它使用完整的五体节动力学，默认持续 40 秒，首个物理 episode 不跨 reset 拼接。

```powershell
python probe_fixed_waves.py `
  --device cuda `
  --dt 0.00025 `
  --seconds 40 `
  --out diagnostics/cuda_wave_matrix
```

查看可用案例：

```powershell
python probe_fixed_waves.py --list-cases
```

固定波通过后，输出目录中应存在 `teacher_data.pt`。它必须包含完整 episode、尾段持续推进、实际关节摆幅、同中位角静止对照和源代码/输入哈希。

## 第三步，运行行为克隆

行为克隆只接受通过 CUDA 固定波门槛的教师数据。示例命令如下：

```powershell
python pretrain_actor.py `
  --data diagnostics/cuda_wave_matrix/teacher_data.pt `
  --case <passed-snake-case> `
  --mode snake `
  --out runs/wave_gate_v5_trial1/bc_snake `
  --device cpu `
  --epochs 400
```

混合模式使用同一命令，将 `--mode snake` 改为 `--mode mixed`，并选择通过 mixed 门槛的案例。

## 第四步，运行完整流程

总入口会按顺序运行固定波、教师筛选、行为克隆、确定性闭环检查和两组 PPO。

```powershell
python run_pipeline.py `
  --out runs/wave_gate_v5_trial1 `
  --probe-seconds 40 `
  --bc-epochs 400 `
  --rollouts 128
```

流程顺序如下：

```text
固定蛇形波/混合波验证
→ 同速度静止奖励对照
→ 行为克隆 actor
→ 20 秒确定性闭环检查
→ 纯蛇形 PPO
→ 混合模式 PPO
→ 独立评估和检查点保存
```

## 单独运行 PPO

PPO 只允许加载通过验证的行为克隆 actor。纯蛇形和混合模式分别运行，输出目录必须为空或不存在。

```powershell
python train_gpu.py `
  --mode snake `
  --warmstart-actor runs/wave_gate_v5_trial1/bc_snake/actor_checkpoint.pt `
  --eval-initial `
  --initial-std 0.65 `
  --out runs/wave_gate_v5_trial1/ppo_snake `
  --envs 8 `
  --horizon 64 `
  --rollouts 128 `
  --epochs 4 `
  --batch-size 128 `
  --save-every 8 `
  --eval-every 8
```

```powershell
python train_gpu.py `
  --mode mixed `
  --warmstart-actor runs/wave_gate_v5_trial1/bc_mixed/actor_checkpoint.pt `
  --eval-initial `
  --initial-std 0.65 `
  --out runs/wave_gate_v5_trial1/ppo_mixed `
  --envs 8 `
  --horizon 64 `
  --rollouts 128 `
  --epochs 4 `
  --batch-size 128 `
  --save-every 8 `
  --eval-every 8
```

速度课程默认是连续均匀采样 `5–200 mm/s`，并混入 10% 停车命令，每 5 秒切换一次。纯蛇形动作是 4 维，混合动作是 9 维。

## 评估检查点

```powershell
python evaluate_gpu.py `
  runs/wave_gate_v5_trial1/ppo_snake/checkpoint.pt `
  --out runs/wave_gate_v5_trial1/eval_snake `
  --seconds 20
```

随机策略诊断需要显式加入 `--stochastic`。随机评估不能作为确定性持续运动验收。

## 交接边界

当前包包含完整代码和输入模型。它不包含大体积原始轨迹、缓存、视频或训练检查点。当前 v5 的 GPU 固定波、行为克隆和新 PPO 需要在可用 CUDA 环境重新运行，不能把旧实验检查点当作 v5 最终权重。

