# 演示视频的证据范围

## 手动蛇形：运动学演示

`manual_kinematic/` 包含俯视、斜视及波形视频，配有角度 CSV 和 NPZ。该组为手动规定关节角的运动学展示，使用静态钢片形状，不计算真实推进和轮地动力学。它展示用户认可的蛇形协调外观，**不是 RL 学会行走的证据**。

- [俯视＋关节波形](manual_kinematic/manual_snake_top_wave.mp4)
- [纯俯视](manual_kinematic/manual_snake_top_clean.mp4)
- [纯斜视](manual_kinematic/manual_snake_oblique_clean.mp4)

## 历史 GPU 动力学演示

`historical_physics/` 为此前保存的真实数值积分视频，包含后退蠕动波和收缩／恢复，附原录制配置。使用历史核心和固定驱动命令，不是本轮纯蛇形 PPO 或最新力学修正版的评估。

- [后退蠕动波](historical_physics/5_worm_retrograde_h250_cad_1x.mp4)
- [收缩与恢复](historical_physics/5_h250_smooth_corrected_recovery2000_rrimplicit_1x.mp4)

本次未重新生成视频。新训练效果应由对应检查点的独立评估与新录制证明。
