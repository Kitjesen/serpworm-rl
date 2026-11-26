# Legacy v1–v3 脚本归档

本目录保留仓库整理前平铺在根目录的早期 v1–v3 源码、CPG（Matsuoka）迁移脚本、奖励消融启动脚本和求解器基准工具，供历史实验复现与对照：

- **仿真与环境**：`gpu_core.py`、`wheel_stiction.py`、`matsuoka_torch.py`、`rewards.py`、`tensor_env.py`（支持 `direct` / `liu` / `liu_foc`）
- **训练与评估**：`train_gpu.py`、`evaluate_gpu.py`、`run_pilots.py`、`run_reward_ablation.py`、`probe_snake_control.py`
- **回归与基准**：`check_migration.py`、`check_rewards.py`、`check_control_v3.py`、`check_plate_solver.py`、`bench_parallel.py`、`bench_plate_structure.py`
- **历史快照清单**：`make_manifest.py`、`MANIFEST.json`

当前主线开发请使用根目录的 [src/serpworm](../src/serpworm/) 与 [scripts](../scripts/)。
如需运行旧脚本，请在 `legacy/` 目录下执行（`gpu_core.py` 与 `check_migration.py` 已兼容指向 `../assets/inputs/` 与 `../archive/reference/`）。
