# Steel Worm–Snake GPU — 五体节蠕虫／蛇形钢片机器人仿真与学习

**研究候选实现。尚未经过实机参数标定、完整能量验收或策略收敛验证。**

仓库现名 `steel-worm-snake-gpu`，覆盖蠕虫、蛇形和融合模态的研究目标。当前RL阶段仅训练纯蛇形四关节控制；不能把三个控制器分支理解为三种运动模态。已完成三组各1024样本的短训练，最终检查点的独立评估已启动，详见[运行记录](results/GPU_RUN.zh-CN.md)。

新一轮已启动[奖励对照长训练](docs/REWARD_ABLATION.zh-CN.md)：旧、新奖励各256轮、8环境、131072样本，每32轮独立评估。新奖励先运行；这个预算和旧8轮短试验不同。

从原项目 [Kitjesen/worm-sim](https://github.com/Kitjesen/worm-sim) 独立整理的专用 GPU 仿真与训练仓库。机器人有五个钢片体节、四十片预弯钢片、十个收绳驱动和四个节间摆动驱动。当前训练只控制四个摆动驱动，十个收绳命令保持固定，钢片仍被动参与动力学。

## 当前版本做什么

- `gpu_core.py`：随形梁、刚性隔板、只承拉绳索、有限力矩关节、被动单向轮和地面接触。每片16个梁单元；五体节610个截面自由度块。运行时不导入 SOFA。
- `tensor_env.py`：CUDA 上的传感器延迟与噪声、观测历史、奖励、失效判断、并行环境重置。
- `matsuoka_torch.py`：四个 Matsuoka 振荡器及 FOC 的张量实现，固定频率阶段。
- `train_gpu.py`：CUDA 上的策略、价值网络、rollout、GAE 和 PPO 更新，不经过 SB3 的 NumPy 环境接口。
- `reference/pilot_cpu_interface/`：迁移前的 CPU 控制／学习接口，供对照，不是当前 GPU 训练入口。

“GPU”指在线数值计算驻留 CUDA。CPU 仍负责读取拓扑、初始化常量、Python 调度、rollout 边界诊断、存盘及视频编码。**不声称整个操作系统或所有 IO 都在 GPU 上执行。**

## 快速运行

已测试目标环境为 Linux、RTX 5090、PyTorch 2.8.0+cu128。先安装适合设备的 CUDA PyTorch，再安装其余依赖。运行命令均在仓库根目录：

```bash
python -m pip install -r requirements.txt
python check_migration.py --device cpu --out results/local_cpu_checks.json
python check_migration.py --device cuda --out results/migration_checks.json
python train_gpu.py --kind direct --envs 2 --rollouts 8 --out runs/direct_seed7
python evaluate_gpu.py runs/direct_seed7/checkpoint.pt --out runs/direct_seed7/evaluation
```

文献迁移分支：将 `--kind` 换为 `liu` 或 `liu_foc`。`run_pilots.py` 在已有 CUDA 检查通过后依次启动三组有时限的短训练。输出目录必须不存在，以免覆盖旧实验。

## 任务与公平比较

| 分支 | 任务 | 策略输入 | 输出 |
|---|---|---|---|
| direct | 速度指令，首轮 vy=0 | 编码器、差分角速度、陀螺仪、历史命令、指令，10帧历史 | 四关节目标，±20° |
| liu | 模拟视觉到点 | 目标相对状态、节间等效曲率、历史输出 | 四个 CPG 互补输入，c=0 |
| liu_foc | 与 liu 相同 | 与 liu 相同 | 同上，c=0.75 |

当前两个任务不同，不能比较不同分支奖励来宣布某方法优越。Liu 迁移分支不是原气动机器人的完整复现，不包含高层 Option-Critic 频率学习。没有原论文实机成功率的复现声明。

## 结果与演示

- [结果目录说明](results/README.md)：迁移检查、旧训练日志、检查点和历史物理验证。
- [CPG 数值检查图](results/controller_checks.png)：振荡器积分对照，不能作为整机控制有效性的证据。
- [演示说明](media/README.md)：区分手动运动学演示与历史动力学视频；视频不是新 GPU 策略成绩。
- [架构、方程与验证缺口](docs/ARCHITECTURE.zh-CN.md)。
- [GPU 迁移与 SOFA 的关系](docs/GPU_MIGRATION.zh-CN.md)。
- [复现／上传清单](MANIFEST.json)：文件哈希、尺寸和快照时间。持续训练产生的新结果需要后续同步，不会自动出现在本次快照中。

## 仍然没有解决的物理问题

1. 梁切线省略应力相关几何 Hessian；配置相关惯性及完整能量收支仍待验收。
2. 绳索、连接与接触以分步方式反馈，并非全系统 Newton 收敛求解；绳阻尼没有完整纳入变化自由长度的速度。
3. 轮地摩擦0.8、滚阻0.06、舵机0.5 N·m等是候选参数；没有实物标定。
4. 仅有轮包络、关节角和姿态等保护，尚无完整机身与钢片自碰撞响应。
5. 仅角度＋陀螺仪不能唯一确定平移滑移或全球位置；部署到点任务需要对应定位。

短训练、程序检查和数值一致性通过，只说明特定接口与计算检查通过，不等于上述问题已解决。

## 来源

梁数值实现参考 [SOFA BeamAdapter](https://github.com/sofa-framework/BeamAdapter)，保留 [原始源码头、作者与 LGPL-2.1 许可](reference/BeamAdapter/)。本仓库是专用方程迁移及控制实现，不是 SOFA 官方分支，也不是将完整 SOFA 拷贝出来。

控制参考 [Liu、Onal、Fu：Reinforcement Learning of CPG-regulated Locomotion Controller for a Soft Snake Robot](https://arxiv.org/pdf/2207.04899)。耦合仿真参考 [Gasoto 等：A Validated Physical Model For Real-Time Simulation of Soft Robotic Snakes](https://arxiv.org/pdf/1904.02833)。论文 PDF 不在本仓库重新分发。
