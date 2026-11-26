# SerpWorm-RL & FlexGraph (`v0.8.0`)

面向刚柔耦合连续体与模块化仿生机器人的全栈 **FP64 物理仿真、纯本体强化学习、形态无关通用控制与硬件在环部署** 代码库，包含三个协同演进的软件层：

1. **SerpWorm-RL (`src/serpworm/`)**：五体节预弯弹簧钢片蠕虫／蛇形机器人（5 体节、40 根预弯钢片、10 路拉绳舵机、4 个偏航关节、20 个单向被动轮）的无损 FP64 `fused_cuda` 仿真引擎、实机标定数字孪生、v6–v8 纯本体自主多管道／多地形强化学习、HIM-DreamWaQ + 鞍点安全精确 $\mathbb{Z}_2$ 等变封装与 TorchScript / HIL 部署栈。
2. **FlexGraph Python (`src/flexgraph/`)**：声明式图编译刚柔耦合框架（`RigidFlexGraph` IR、可组合 3D SDF 环境、形态无关 `TokenizedHIMDreamWaQ`、有限群精确 Reynolds 等变封装、多形态 PPO 训练器与零崩溃 Rust FFI 桥接）。
3. **FlexGraph-RS v2 (`rust/flexgraph-rs/` + `src/flexgraph_rs/`)**：8-Crate 纯 Rust FP64 刚柔耦合与连续体物理引擎，支持 3D 协同旋转梁与 DER/Cosserat 杆双求解路径、C-IPC / Tight-Inclusion CCD、Compliance LCP + Clarabel SOCP、`faer` 稀疏分解、ROS URDF / MJCF 导入、伴随可微与 36 个 C-ABI 符号。

---

## 核心架构与力学模型

### 连续体梁／杆离散模型
- **SerpWorm-RL & FlexGraph Python**：采用 **3D 协同旋转欧拉-伯努利梁（3D Corotational Euler-Bernoulli Beam）**，基于 cubic-Hermite 中点参考架、Saint-Venant 矩形截面扭转刚度与可选 Kirchhoff-Love / Vlasov 薄壳刚化及装配预应力模型，结合 6×6 Block-Thomas / Block-Cholesky 实时 Schur 消元完成刚柔耦合隐式积分。
- **FlexGraph-RS v2**：在保留与 `GPUCore` 机器精度等价的 3D 协同旋转梁基础上，原生扩展 **离散弹性杆（DER / Cosserat Rod）**（时间平行 Bishop 参考架、全解析 11×11 Hessian 与带状 $LDL^\top$ 凝聚），统一支撑板条、细杆、张拉整体与导管连续体。

### 三层能力概览

#### (a) SerpWorm-RL (`src/serpworm/`)
- **无损 FP64 `fused_cuda` 融合核**：与 PyTorch 参考实现 `segment_lu` 200 步双精度位置偏差 **`≤ 4.35×10⁻¹⁰ m`**，支持最多 **`4,096` 并行环境**；RTX 5090 上 `B=1024, E=16, dt=1.0 ms` 隐式 ARPC 吞吐达 **`4,745.9 sps`（`153.18×` 提速，显存 `3.39 GB`）**，`E=8` 达 **`9,383.3 sps`（`302.86×` 提速，显存 `1.72 GB`）**。
- **实机重量与双台架 1:1 物理标定**：
  - 5 周期万能试验机（UTM）0–45 mm 轴向压缩：峰值力 **`42.681 vs 42.707 N`（-0.06%）**，**`RMSE = 0.1384 N`**，**`R² = 0.999857`**。
  - Exp 126–128 单体节双激光偏心曲柄拉绳标定：双侧伸缩与左右偏转中心位移误差 **`≤ 0.42 mm`**。
  - 集成导孔 **Capstan 摩擦** 与隐式 **ARPC 各向异性面接触**（硬地 / 地毯 / 软土 Bekker 车辙），静止 4 s 漂移速率降至 `0.013 mm/s`（较显式 Kelvin 降低 `348×`）。
- **v6–v8 纯本体自主强化学习、HIM-DreamWaQ 与鞍点安全精确 $\mathbb{Z}_2$ 等变**：
  - 无需外部视觉或定位，仅凭关节编码器、软件 `q_err` 与机身 IMU，在多管径弯管、S 弯、直角 L 管及砂砾／草地等复杂工况下自主切换运动模式。
  - **`v8_diverse_4096`（12 类管道与自然地形工况）**：直行速度跟踪 **`MAE = 4.03 mm/s, R² = 0.9878`**，转弯跟踪 **`R² = 0.9810`**；鞍点安全精确 $\mathbb{Z}_2$ 封装（`GroupEquivariantPolicyWrapper`）将左右转弯不对称度降低 **`2.6×–18×`**。
  - **`v7.1`（`v71_Lduct_4096`）**：直行速度跟踪 **`MAE = 2.74 mm/s, R² = 0.9946`**，24 组扫频工况 **`0/24` 失败**。
  - 提供 **`HIMDreamWaQ`**（显式 7D 状态估计 + $\beta$-VAE 隐式地形上下文 + InfoNCE/SwAV 对比动力学）、**TorchScript (`.ts`) 冻结导出**、**200 步 HIL 干跑基准** 与 **`sysid_toolkit.py`** 辨识工具。

#### (b) FlexGraph Python (`src/flexgraph/`)
- **图编译 IR 与自适应稀疏路由 (`graph.py`, `solver.py`)**：将任意刚柔混合机器人表达为二分多图 `RigidFlexGraph`（支持 JSON 序列化），自动路由至 `block_diagonal_pairs` / `tree_chain` / `sparse_general` 稀疏 Schur 求解器；内置 `serpworm`、`bristle_worm`、`tendon_endoscope`、`flexspine_crawler` 构型生成器与 $\mathbb{Z}_2, \mathbb{Z}_2\times\mathbb{Z}_2, D_4$ 对称群定义。
- **可组合 3D SDF 世界与统一接触 (`sdf_world.py`, `contact.py`)**：支持高度场、变截面 3D 样条管腔、包围盒与体素 SDF 组合，以及 `AnisotropicPatchContact` 和 $C^2$ 光滑自接触。
- **形态无关 RL 策略与多形态训练器 (`policy.py`, `env.py`, `trainer.py`, `rust_backend.py`)**：
  - `TokenizedHIMDreamWaQ`（含掩码感知 `TokenRunningNorm`、变长执行器 Token 注意力、非对称特权 Critic 与故障掩码容错）+ 任意有限群精确 Reynolds 等变封装 `FiniteGroupEquivariantWrapper`。
  - `FlexGraphPPOTrainer`（`scripts/train_flexgraph.py`）支持 5 类内置构型（`serpworm_N3`, `serpworm_N5`, `bristle_worm`, `tendon_endoscope`, `flexspine_crawler`）轮询／混合训练、自适应 KL 学习率、GAE `time_outs` 截断引导、对称正则与故障注入课程，并经由 `rust_backend.py` 无缝对接 `fg_ffi` 原生动态库（缺失时零崩溃回退至 `GraphSchurSolver`）。

#### (c) FlexGraph-RS v2 (`rust/flexgraph-rs/` + `src/flexgraph_rs/`)
- **8-Crate 纯 Rust FP64 工作空间**：`fg-math`（李群/空间代数/协同旋转梁/DER）、`fg-ir`（图 IR 与稀疏消元树分析）、`fg-sdf`（BVH/CSG/样条管/体素 SDF）、`fg-contact`（C-IPC / Tight-Inclusion CCD / Compliance LCP + Clarabel SOCP / ARPC）、`fg-robots`（内建构型 + ROS URDF / MJCF 导入）、`fg-solver`（并行 Block-Thomas、树型/$LDL^\top$、`faer` 稀疏求解器、伴随可微）、`fg-ffi`（36 个 C-ABI 符号 + Rayon 并行批量环境池）、`fg-bench`。
- **实测基准与回归验证**：
  - 与 `GPUCore` 双精度位置误差 **`≤ 3.425×10⁻¹² m`**。
  - SerpWorm-5 上 Rayon 多核 CPU 提速 **`46.82×`（`1,064 env-steps/s`）**；TendonEndoscope-3D 吞吐达 **`22,192 env-steps/s`**。
  - 全工作空间 **`83/83` Cargo 测试** 与 **`9/9` Python FFI 测试** 全部通过。

---

## 目录结构

```text
serpworm-rl/
├── pyproject.toml                   # 包配置 (v0.8.0)、依赖与 CLI 入口
├── requirements.txt                 # 基础 Python 依赖清单
├── src/
│   ├── serpworm/                    # SerpWorm-RL 主线包
│   │   ├── config.py                # HardwareConfig / DomainRandConfig / TaskConfig / UnifiedTrainConfig
│   │   ├── paths.py                 # 路径解析与资产清单
│   │   ├── pipeline.py              # v5 固定波 → BC → PPO 流水线
│   │   ├── physics/                 # gpu_core.py, fused_cuda.py, csrc/fused_physics.cu, pipe_geometry.py,
│   │   │                            # topology.py, wheel_stiction.py, wheel_envelope.py
│   │   ├── envs/                    # autonomous_env.py (v6–v8), unified_env.py (v5), tensor_env.py,
│   │   │                            # rewards.py, matsuoka.py
│   │   ├── rl/                      # models.py, him_dreamwaq.py, ppo.py, train_autonomous.py,
│   │   │                            # eval_autonomous.py, train_unified.py, eval_unified.py,
│   │   │                            # ablation_bench.py, train_ppo.py, evaluate.py, pretrain_bc.py
│   │   ├── deploy/                  # policy_runtime.py (TorchScript), hardware_io.py (HIL), sysid_toolkit.py
│   │   └── probes/                  # bench_dual_laser.py, bench_utm_compression.py, render_mujoco.py,
│   │                                # fixed_waves.py, wheel_geometry.py, saved_wave.py 等
│   ├── flexgraph/                   # FlexGraph Python 通用刚柔耦合 RL 框架 (见 src/flexgraph/README.md)
│   │   ├── graph.py                 # RigidFlexGraph IR、稀疏路由与内置构型生成器
│   │   ├── math3d.py                # FP64 SO(3)/SE(3) 与 3D 协同旋转欧拉-伯努利梁矩阵
│   │   ├── sdf_world.py             # 可组合 3D SDF 环境与材料库
│   │   ├── contact.py               # 统一 AnisotropicPatchContact 与 C² 自接触
│   │   ├── solver.py                # 无损 FP64 GraphSchurSolver
│   │   ├── policy.py                # TokenizedHIMDreamWaQ、TokenRunningNorm 与 FiniteGroupEquivariantWrapper
│   │   ├── env.py                   # 批量向量化 FlexGraphEnv 与 UniversalSafetyGuard
│   │   ├── trainer.py               # 多形态非对称 PPO + HIM-DreamWaQ 训练器
│   │   └── rust_backend.py          # fg_ffi C-ABI 桥接与自动回退
│   └── flexgraph_rs/                # FlexGraph-RS v2 Python 绑定与一致性校验 (_ffi.py, solver.py, parity.py, bench_parity.py)
├── rust/flexgraph-rs/               # FlexGraph-RS v2 8-Crate 纯 Rust 工作空间 (Cargo.toml)
│   └── crates/                      # fg-math, fg-ir, fg-sdf, fg-contact, fg-robots, fg-solver, fg-ffi, fg-bench
├── scripts/                         # 训练、评测、标定、部署、演示与论文出图入口 (详见 scripts/README.md)
│   ├── demos/                       # 物理展示与 3D 渲染演示脚本
│   └── paper/                       # 论文主图、相图与汇报构建脚本
├── tests/                           # pytest 单元测试与数值等价回归套件 (21 个测试模块)
├── docs/                            # 架构、GPU 迁移、奖励消融与论文技术文档
│   └── paper/                       # 基准表、理论设计、架构 RFC、生态集成与方法综述
├── assets/                          # 规范输入 (inputs/)、CAD/URDF STL (cad/) 与演示视频 (media/)
├── legacy/                          # 归档的早期 v1–v3 脚本与基准 (见 legacy/README.md)
└── archive/                         # 历史实验快照、日志与 BeamAdapter 许可参考头
```

---

## 环境与安装

- **Python / GPU 训练**：Python ≥ 3.10、PyTorch ≥ 2.8（GPU 训练建议配置 CUDA Toolkit `nvcc` + `ninja` 以自动 JIT 编译 `fused_physics.cu`）。
- **3D 渲染（可选）**：`mujoco>=3.2`、`pillow`、`ffmpeg`。
- **Rust 引擎（可选）**：Rust 2021 工具链（`cargo` / `rustc`），用于编译 `rust/flexgraph-rs/` 与 `fg-ffi` 动态库。

```bash
# 安装 Python 包与测试/可视化依赖
python -m pip install -e ".[dev,viz]"

# 可选：启用 fused_cuda JIT 编译与 MuJoCo 3D 渲染
python -m pip install ninja mujoco pillow

# 可选：编译 FlexGraph-RS v2 发布版动态库 (fg-ffi)
cargo build --release --manifest-path rust/flexgraph-rs/Cargo.toml -p fg-ffi
```

---

## 快速开始

### 1. 单元测试与 Rust 工作空间验证

```bash
# 快速冒烟与 FlexGraph / FlexGraph-RS 核心回归
pytest -q tests/test_package_smoke.py tests/test_flexgraph.py tests/test_flexgraph_rs.py

# 运行完整 Python 回归套件
pytest -q

# 运行 FlexGraph-RS v2 全部 Cargo 测试 (83 tests)
cargo test --workspace --manifest-path rust/flexgraph-rs/Cargo.toml
```

### 2. FP64 `fused_cuda` 双精度一致性与吞吐基准

```bash
python scripts/benchmark_cuda_kernel.py --batch 2048 --dt 0.0005 --substeps 200 --iters 5
```

### 3. SerpWorm-RL v6–v8 纯本体自主训练与扫频评测

```bash
# 启动 v8 多管道/多地形 65D 纯本体自主强化学习 (GPU, fused_cuda)
python scripts/train_autonomous.py \
  --out runs/v8_diverse_4096 \
  --diverse-v8 --include-q-err \
  --envs 4096 --horizon 48 --rollouts 240 \
  --plate-solver fused_cuda --symmetry-aug --amp

# 多速度、左右转弯与动态阶跃指令跟踪扫频评测 (Raw vs 鞍点安全 Z2 vs Naive Z2)
python scripts/eval_tracking_sweep.py \
  --checkpoints runs/v8_diverse_4096/latest.pt runs/v71_Lduct_4096/best.pt \
  --modes raw,z2,z2_naive \
  --device cuda --plate-solver fused_cuda \
  --out-dir runs/tracking_eval
```

### 4. FlexGraph 多形态通用强化学习训练

```bash
python scripts/train_flexgraph.py \
  --morphologies all \
  --schedule round_robin \
  --device cuda --backend auto \
  --envs 128 --horizon 24 --rollouts 120 \
  --epochs 4 --minibatches 4 \
  --substeps 10 --elements-per-beam 8 \
  --symmetry-reg 0.15 --fault-prob-end 0.15 \
  --amp-bf16 --eval-every 15 --save-every 10 \
  --out-dir runs/flexgraph_multi_v1
```

### 5. TorchScript 部署导出与 200 步 HIL 干跑验证

```bash
# 导出冻结 TorchScript 策略 (.ts) 与 deploy_meta.json 并做随机输入零误差比对
python scripts/export_deploy_policy.py \
  --checkpoint runs/v8_diverse_4096/latest.pt \
  --out-dir runs/v8_diverse_4096/deploy_export

# 运行 200 步硬件在环 (HIL) 干跑基准与 TorchScript 数值一致性审计
python scripts/run_hil_dryrun.py \
  --checkpoint runs/v8_diverse_4096/latest.pt \
  --out-dir runs/deploy_hil_dryrun --steps 200
```

---

## 关键实测汇总

| 维度 | 配置 / 对象 | 关键指标 |
| :--- | :--- | :--- |
| **UTM 压缩标定** | 0–45 mm 5 周期实机 vs FP64 3D 协同旋转梁 | 峰值 `42.681 vs 42.707 N`（`-0.06%`），`RMSE = 0.1384 N`，`R² = 0.999857` |
| **双激光台架标定** | Exp 126–128 单体节偏心曲柄拉绳 | 双侧伸缩与单侧偏转中心位移误差 `≤ 0.42 mm` |
| **GPU 融合物理吞吐** | RTX 5090, `B=1024`, `dt=1.0 ms`, 隐式 ARPC | `E=16`: `4,745.9 sps`（`153.18×`, `3.39 GB`）；`E=8`: `9,383.3 sps`（`302.86×`, `1.72 GB`）；FP64 误差 `≤ 4.35×10⁻¹⁰ m` |
| **v8 纯本体自主跟踪** | `v8_diverse_4096`（12 类管道/地形工况） | 直行 `MAE = 4.03 mm/s, R² = 0.9878`；转弯 `R² = 0.9810`；鞍点安全 $\mathbb{Z}_2$ 降不对称 `2.6×–18×` |
| **v7.1 管道自主跟踪** | `v71_Lduct_4096`（24 组扫频工况） | 直行 `MAE = 2.74 mm/s, R² = 0.9946`，`0/24` 失败 |
| **FlexGraph-RS v2** | 纯 Rust FP64 + Rayon 批量引擎 | 对 `GPUCore` 误差 `≤ 3.425×10⁻¹² m`；SerpWorm-5 `46.82×`（`1,064 env-steps/s`）；TendonEndoscope-3D `22,192 env-steps/s`；`83/83` Cargo + `9/9` FFI 测试通过 |

---

## 文档与索引

### 核心技术与论文文档 (`docs/paper/`)
- [已验证基准数据与 LaTeX 表格](docs/paper/BENCHMARK_TABLES.md)
- [FlexGraph-RS v2 理论与架构设计](docs/paper/FLEXGRAPH_RS_THEORY_AND_DESIGN.md)
- [通用刚柔耦合架构 RFC](docs/paper/GENERIC_RIGIDFLEX_ARCHITECTURE_RFC.md)
- [GitHub 开源生态集成审计](docs/paper/GITHUB_ECOSYSTEM_INTEGRATION.md)
- [文献洞察与演进路线图](docs/paper/LITERATURE_INSIGHTS_AND_ROADMAP.md)
- [方法论与相关工作](docs/paper/METHODOLOGY_AND_RELATED_WORK.md)

### 工程文档与子目录索引
- [脚本目录规范与完整索引](scripts/README.md)
- [FlexGraph Python 模块说明](src/flexgraph/README.md)
- [Rust FlexGraph-RS 工作空间配置](rust/flexgraph-rs/Cargo.toml)
- [架构、方程与验证缺口](docs/ARCHITECTURE.zh-CN.md)
- [GPU 迁移与 SOFA 的关系](docs/GPU_MIGRATION.zh-CN.md)
- [钢片动力学与采样加速文献调研](docs/STEEL_DYNAMICS_ACCELERATION_REVIEW.zh-CN.md)
- [奖励消融与采样吞吐记录](docs/REWARD_ABLATION.zh-CN.md)
- [v5 大幅固定波门槛与直接 PPO 设计](docs/WAVE_GATE_V5.zh-CN.md)
- [CAD / URDF 参考资源](assets/cad/README.zh-CN.md)
- [旧版 v1–v3 脚本说明](legacy/README.md)

---

## 来源与致谢

- 3D 协同旋转梁参考 [SOFA BeamAdapter](https://github.com/sofa-framework/BeamAdapter)（LGPL-2.1 许可保留于 [archive/reference/BeamAdapter/LICENSE.md](archive/reference/BeamAdapter/LICENSE.md)）。
- CPG 对照参考 [Liu, Onal, Fu (2022)](https://arxiv.org/pdf/2207.04899)；软体蛇耦合仿真参考 [Gasoto et al. (2019)](https://arxiv.org/pdf/1904.02833)。
