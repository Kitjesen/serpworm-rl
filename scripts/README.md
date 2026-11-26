# `scripts/` 目录规范与索引

按职责分层管理，顶层保留核心训练、评测、部署与验证 CLI 入口；物理演示与论文出图分别收敛到 `demos/` 与 `paper/` 子目录，所有条目与磁盘文件一一对应。

## 1. 顶层 CLI 入口 (`scripts/`)

| 脚本 | 职责 |
|---|---|
| `train_autonomous.py` / `eval_autonomous.py` | **主线 v6–v8** 纯本体自主多管道/多地形训练与确定性评测入口 |
| `eval_tracking_sweep.py` | 多速度、左右转弯与动态阶跃指令跟踪扫频评测（含 Raw / 鞍点安全 $\mathbb{Z}_2$ / Naive $\mathbb{Z}_2$ 对照） |
| `train_flexgraph.py` | **FlexGraph** 多形态非对称 PPO + HIM-DreamWaQ 训练入口（支持 `auto`/`python`/`rust` 后端与 5 类构型） |
| `export_deploy_policy.py` | 导出冻结 TorchScript (`.ts`) 与实机部署元数据 (`deploy_meta.json`) |
| `run_hil_dryrun.py` | 200 步硬件在环（HIL）干跑基准与 TorchScript 零误差校验 |
| `train_unified.py` / `eval_unified.py` | v5 三模态统一训练与确定性评测入口 |
| `train_ppo.py` / `evaluate.py` / `pretrain_actor.py` / `run_pipeline.py` | v5 单模式训练、行为克隆热启动与流水线入口（`tests/` 与 `pipeline` 依赖） |
| `benchmark_cuda_kernel.py` / `benchmark_gpu_migration.py` | FP64 `fused_cuda` 融合核双精度等价性与多批量吞吐基准 |
| `bench_dual_laser.py` / `bench_utm_compression.py` | 单体节双激光台架（Exp 126–128）与 UTM 5 周期压缩 1:1 物理标定 |
| `validate_local.py` | 本地轻量数值契约回归测试汇总 |
| `probe_fixed_waves.py` / `probe_policy_distribution.py` | 固定波扫参门控与策略分布诊断 |
| `audit_wheel_geometry.py` / `audit_saved_wave.py` | 轮距防干涉几何审计与轨迹奖励复算 |
| `render_mujoco.py` / `render_topology_3d.py` | MuJoCo 3D CAD+钢片轨迹视频与拓扑静态渲染入口 |

## 2. 物理展示与 3D 渲染演示 (`scripts/demos/`)

- `demo_capstan_1seg_formal.py` / `demo_capstan_1seg_pipe.py`：导孔 Capstan 摩擦与单节管内挤压对比展示
- `demo_contact_comparison.py` / `demo_pipe_showcase.py`：新旧接触模型对比与零穿透管内力学展示
- `render_training_arena_3d.py` / `render_v8_new_scenes_preview.py`：并行训练场全景与 v8 管道/地形 3D 预览
- `render_bench_126_128_real_vs_sim.py`：实机双激光试验与仿真同频对齐合成
- `demo_flexgraph_summary.py`：`flexgraph` 异构构型拓扑与稀疏性摘要

## 3. 论文图表与学术汇报 (`scripts/paper/`)

- `generate_paper_figures.py` / `generate_nc_arch_figure.py` / `analyze_phase_diagram_cot.py`：论文正交/相图/架构主图（300 DPI + PDF）
- `build_academic_ppt.py`：学术汇报 PPTX 构建器（输出已忽略的 `docs/SerpWorm_RL_Academic_Talk.pptx`）
