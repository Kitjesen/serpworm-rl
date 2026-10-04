# 结果索引

所有结果均为仿真或数值检查，未包含实机标定数据。

| 路径 | 含义 | 不可推出的结论 |
|---|---|---|
| `migration_cpu_checks.json` | 原 CPG 与 torch 数值对照、GAE终止检查 | GPU整机或实物有效 |
| `migration_checks.json`（生成后归档） | RTX5090迁移检查 | 长时稳定性、策略收敛 |
| `tensor_direct/` 等（运行后归档） | 新 CUDA 数值链训练配置、状态、检查点 | 未评估前的导航能力 |
| `pilot_cpu_interface/` | 迁移前3组短训练；包含原始日志和检查点 | 新 CUDA PPO 的结果 |
| `historical_physics/` | 旧核心版本的接触、耦合与性能对照 | 最新应变 Jacobian 版本全部通过 |
| `controller_checks.*` | 原振荡器独立积分检查 | 整机控制性能 |

迁移前首次训练：direct采样1090步、完整更新8轮；liu与liu_foc各938步、完整更新7轮。每组约30分钟，单种子；未确认收敛。评估状态以快照文件为准。

旧 `status.json` 中的 `optimizer_updates` 实际来自 SB3 `_n_updates`（epoch计数）。新训练文件分别记录 `rollouts`、`epochs` 和真正的 Adam `optimizer_steps`，避免混淆。

历史日志和合同里的绝对路径、哈希指向当时实验；保存用于追溯，不代表本仓库具有全部历史依赖。新代码运行以根目录 README 为准。
