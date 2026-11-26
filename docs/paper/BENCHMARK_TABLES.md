# SerpWorm-RL: Verified Benchmark Tables (Markdown + LaTeX)

> 本文档中**所有数值均直接提取自本地已验证实验产物**（JSON / NPZ / `config.py`），每张表均标注精确源文件来源，并提供可直接粘贴至 *Nature Communications* / *Science Robotics* / *IEEE RA-L* 稿件的 `booktabs` LaTeX 表格代码。

---

## Table 1: Single-Segment System Identification — UTM 5-Cycle Compression & Exp 126–128 Dual-Laser Calibration

**Verified Sources**:
- [`runs/utm_validation/utm_validation_summary.json`](../../runs/utm_validation/utm_validation_summary.json)
- [`runs/bench_exp126_128/bench_summary.json`](../../runs/bench_exp126_128/bench_summary.json)

### Markdown Table (1a: UTM 5-Cycle Axial Compression, 0–45 mm)

| Model / Measurement | Strip $t$ (mm) | Free Bow $b_0$ (mm) | Peak Force @ 45 mm (N) | RMSE vs Real Mid (N) | $R^2$ | Max Abs Error (N) | Bulge Dia. 0→45 mm (mm) | Bulge Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Real UTM 5-Cycle (Measured)** | 0.220* | 30.0* | **42.707** (load 42.709 / unload 42.705) | — (hyst. mean 0.295, max 0.438, loss 1.102%) | — | — | — | — |
| **Calibrated FP64 (`bench_exp128_calibrated`)** | **0.220** | **30.0** | **42.681** (-0.06%) | **0.1384** | **0.999857** | **0.4948** | 163.8 → 201.8 | 1.232 |
| **Legacy Soft Default (`ds3218_default`)** | 0.150 | 50.7 | 6.064 (**7.04× softer**) | 24.5553 | -3.4964 | — | 205.2 → 230.9 | 1.125 |

*\*UTM-identified effective structural parameters for the 8-strip 65Mn spring-steel segment (`w = 16.0 mm`, `E = 200 GPa`).*

### Markdown Table (1b: Exp 126–128 Dual-Laser Eccentric-Crank Calibration, 0°→180°→0°)

| Experiment | Mode | Sim Ch1 Peak (mm) | Ref Ch1 Peak (mm) | Sim Ch2 Peak (mm) | Ref Ch2 Peak (mm) | Sim Center Peak (mm) | Ref Center Peak (mm) | Center Error (mm) | Sim Yaw Peak (°) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp 128 | Bilateral | 37.22 | 37.2 | 37.22 | 36.4 | 37.22 | 36.8 | +0.42 | 0.00 |
| Exp 127 | Left Only | 11.96 | 12.8 | 35.64 | 35.1 | 23.80 | 23.9 | -0.10 | +18.21 |
| Exp 126 | Right Only | 35.64 | 35.2 | 11.96 | 13.2 | 23.80 | 24.2 | -0.40 | -18.21 |

### LaTeX Table 1

```latex
\begin{table*}[t]
\centering
\caption{Single-segment physical system identification against 5-cycle UTM axial compression (0--45\,mm) and Exp.~126--128 dual-laser eccentric-crank tendon deflection benchmarks.}
\label{tab:system_id}
\begin{tabular}{lccccccc}
\toprule
\multicolumn{8}{l}{\textit{(a) 5-Cycle UTM Axial Compression (0--45\,mm), Source: \texttt{utm\_validation\_summary.json}}} \\
\midrule
Configuration & $t$ (mm) & $b_0$ (mm) & Peak $F$ (N) & RMSE (N) & $R^2$ & Max $|e|$ (N) & Bulge Ratio \\
\midrule
Real UTM 5-Cycle (Mean) & 0.220$^\dagger$ & 30.0$^\dagger$ & 42.707 & 0.295$^\ddagger$ & --- & 0.438$^\ddagger$ & --- \\
\textbf{Calibrated FP64} & \textbf{0.220} & \textbf{30.0} & \textbf{42.681} & \textbf{0.1384} & \textbf{0.999857} & \textbf{0.4948} & \textbf{1.232} \\
Legacy Default & 0.150 & 50.7 & 6.064 (7.0$\times$ gap) & 24.5553 & $-3.4964$ & --- & 1.125 \\
\midrule
\multicolumn{8}{l}{\textit{(b) Exp.~126--128 Dual-Laser Crank-Tendon Calibration (0--180$^\circ$), Source: \texttt{bench\_summary.json}}} \\
\midrule
Experiment & Actuation & Sim Ch1 (mm) & Ref Ch1 (mm) & Sim Ch2 (mm) & Ref Ch2 (mm) & Sim / Ref Center (mm) & Peak Yaw ($^\circ$) \\
\midrule
Exp.~128 & Bilateral & 37.22 & 37.2 & 37.22 & 36.4 & 37.22 / 36.8 (+0.42) & 0.00 \\
Exp.~127 & Left Only & 11.96 & 12.8 & 35.64 & 35.1 & 23.80 / 23.9 ($-0.10$) & $+18.21$ \\
Exp.~126 & Right Only & 35.64 & 35.2 & 11.96 & 13.2 & 23.80 / 24.2 ($-0.40$) & $-18.21$ \\
\bottomrule
\end{tabular}
\vspace{2pt}
\begin{minipage}{0.95\textwidth}
\footnotesize $^\dagger$UTM-identified effective 65Mn steel strip thickness and stress-free arch bow ($w=16$\,mm, $E=200$\,GPa). $^\ddagger$Measured real mean/max loading--unloading hysteresis gap (steady-state cyclic energy loss: 1.102\%).
\end{minipage}
\end{table*}
```

---

## Table 2: Contact Formulation Comparison & RTX 5090 Fused-CUDA Throughput (`B = 1024`)

**Verified Sources**:
- [`runs/demos/contact_demo_metrics.json`](../../runs/demos/contact_demo_metrics.json) (`dt = 1.0 ms`, `E = 16`, `fused_cuda`)
- [`runs/demos/rtx5090_arpc_benchmark.json`](../../runs/demos/rtx5090_arpc_benchmark.json) (RTX 5090, 1024 parallel 5-segment environments)

### Markdown Table (2a: 4-Terrain Contact Stability & Locomotion Dynamics at `dt = 1.0 ms`)

| Contact Model / Terrain | Drop 1st-Step Jump Total / Wheel (N) | Drop Rebound Lift (ms) | Static Hold 4s Drift (mm) | Static Steady Drift Speed (mm/s) | Snake Wave (0.7 Hz, 25°) Speed (mm/s) | Snake Normal Jitter PP (N) | Snake Pivoting Torque (mN·m) | Hybrid 10s Steady Speed (mm/s) | Hybrid Mean / Max Rut (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Explicit Kelvin (`kelvin`)** | 57.89 / 2.89 | 43.0 | 16.73 | 4.575 | 42.62 | 18.57 | 0.00 | 73.25 | 0.00 / 0.00 |
| **Implicit ARPC Hard (`hard`)** | **15.18 / 0.76** | **12.0** | **-0.21** | **0.013** (**348× lower**) | **47.59** | **9.81** | 86.98 | 69.41 | 0.00 / 0.00 |
| **Implicit ARPC Carpet (`carpet`)** | 4.83 / 0.24 | 0.0 | -0.38 | 0.019 | 31.22 | 2.18 | 135.35 | 49.44 | 0.00 / 0.00 |
| **Implicit ARPC Soft Soil (`soft_soil`)** | 5.49 / 0.27 | 0.0 | -0.30 | 0.018 | 23.80 | 1.91 | 169.65 | 35.22 | **1.08 / 2.39** |

### Markdown Table (2b: RTX 5090 FP64 Physics Throughput Benchmark, 1024 Parallel Envs)

| Configuration | Elements / Strip $E$ | Solver | $\Delta t$ (ms) | Substeps / Policy Step | Contact | Time / Substep (ms) | Policy Steps / s | Sim Speed ($\times$ Real-Time) | Speedup vs `segment_lu` | VRAM (GB) |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| `E16_dt0.50_seg_lu_kelvin` | 16 | `segment_lu` | 0.50 | 200 | Explicit Kelvin | 330.52 | 31.0 | 3.1× | 1.00× | 9.85 |
| `E16_dt0.50_fused_kelvin` | 16 | `fused_cuda` | 0.50 | 200 | Explicit Kelvin | 4.26 | 2,402.7 | 240.3× | 77.55× | 3.39 |
| `E16_dt0.50_fused_arpc_hard` | 16 | `fused_cuda` | 0.50 | 200 | Implicit ARPC Hard | 4.31 | 2,378.0 | 237.8× | 76.75× | 3.39 |
| `E16_dt1.00_fused_arpc_hard` | 16 | `fused_cuda` | 1.00 | 100 | Implicit ARPC Hard | 4.32 | **4,745.9** | **474.6×** | **153.18×** | **3.39** |
| `E16_dt1.00_fused_arpc_carpet` | 16 | `fused_cuda` | 1.00 | 100 | Implicit ARPC Carpet | 4.31 | 4,746.4 | 474.6× | 153.20× | 3.39 |
| `E16_dt1.00_fused_arpc_softsoil` | 16 | `fused_cuda` | 1.00 | 100 | Implicit ARPC Soft Soil | 4.32 | 4,736.8 | 473.7× | 152.89× | 3.39 |
| `E8_dt1.00_fused_arpc_hard` | 8 | `fused_cuda` | 1.00 | 100 | Implicit ARPC Hard | 2.18 | **9,383.3** | **938.3×** | **302.86×** | **1.72** |

### LaTeX Table 2

```latex
\begin{table*}[t]
\centering
\caption{Quantitative comparison of explicit Kelvin--Voigt versus implicit ARPC contact dynamics at $\Delta t=1.0$\,ms (\texttt{contact\_demo\_metrics.json}) and RTX 5090 FP64 solver throughput at $B=1024$ parallel 5-segment environments (\texttt{rtx5090\_arpc\_benchmark.json}).}
\label{tab:contact_and_throughput}
\begin{tabular}{lccccccc}
\toprule
\multicolumn{8}{l}{\textit{(a) Contact Dynamics \& Stability across 4 Terrain Presets ($\Delta t = 1.0$\,ms, $E=16$, \texttt{fused\_cuda})}} \\
\midrule
Contact / Terrain & Drop Jump (N) & Bounce (ms) & 4\,s Drift (mm) & Drift Rate (mm/s) & Snake $v_x$ (mm/s) & Normal Jitter (N) & Hybrid $v_x$ / Rut \\
\midrule
Explicit Kelvin & 57.89 & 43.0 & 16.73 & 4.575 & 42.62 & 18.57 & 73.25\,mm/s / 0.00\,mm \\
\textbf{Implicit ARPC Hard} & \textbf{15.18} & \textbf{12.0} & \textbf{$-$0.21} & \textbf{0.013} & \textbf{47.59} & \textbf{9.81} & 69.41\,mm/s / 0.00\,mm \\
Implicit ARPC Carpet & 4.83 & 0.0 & $-$0.38 & 0.019 & 31.22 & 2.18 & 49.44\,mm/s / 0.00\,mm \\
Implicit ARPC Soft Soil & 5.49 & 0.0 & $-$0.30 & 0.018 & 23.80 & 1.91 & 35.22\,mm/s / 1.08\,mm \\
\midrule
\multicolumn{8}{l}{\textit{(b) RTX 5090 FP64 Batched Physics Throughput ($B=1024$ Envs, 10\,Hz Policy Step)}} \\
\midrule
Solver Configuration & $E$ & $\Delta t$ (ms) & Contact & ms / Substep & Policy Steps/s & Speedup & VRAM (GB) \\
\midrule
PyTorch \texttt{segment\_lu} & 16 & 0.50 & Kelvin & 330.52 & 31.0 & 1.00$\times$ & 9.85 \\
\texttt{fused\_cuda} (Kelvin) & 16 & 0.50 & Kelvin & 4.26 & 2,402.7 & 77.55$\times$ & 3.39 \\
\texttt{fused\_cuda} (ARPC Hard) & 16 & 0.50 & Implicit ARPC & 4.31 & 2,378.0 & 76.75$\times$ & 3.39 \\
\textbf{\texttt{fused\_cuda} (ARPC Hard)} & \textbf{16} & \textbf{1.00} & \textbf{Implicit ARPC} & \textbf{4.32} & \textbf{4,745.9} & \textbf{153.18$\times$} & \textbf{3.39} \\
\texttt{fused\_cuda} (ARPC Soft Soil) & 16 & 1.00 & Implicit ARPC & 4.32 & 4,736.8 & 152.89$\times$ & 3.39 \\
\texttt{fused\_cuda} (ARPC Hard, Coarse) & 8 & 1.00 & Implicit ARPC & 2.18 & 9,383.3 & 302.86$\times$ & 1.72 \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## Table 3: Multi-Track RL Policy Ablation (`v6a` 51D vs `v6b` 65D vs `v6.1` 65D+Elec & `v5` Mode-Conditioned Reference)

**Verified Sources**:
- [`runs/v6_ablation_eval.json`](../../runs/v6_ablation_eval.json) (`v6a_51d_pure/best.pt` @ r116, `v6b_65d_qerr/best.pt` @ r194, plus `latest.pt` @ r200; matches `runs/v6a_51d_pure/eval_best_summary.json` & `runs/v6b_65d_qerr/eval_best_summary.json` identically)
- [`runs/v6_1_65d_elec/eval_v6_1_all.json`](../../runs/v6_1_65d_elec/eval_v6_1_all.json) (`best.pt` @ r200, electrical Joule power `work_weight = 0.012`, `vx_overshoot_weight = 1.85`)
- [`runs/unified_v5_hispeed_14d/eval_v5_full.json`](../../runs/unified_v5_hispeed_14d/eval_v5_full.json) (`best.pt` @ r185, mode-conditioned `snake`/`worm`/`mixed` on `arpc_hard`, `carpet`, `soft_soil`)

### Markdown Table (3a: 6-Track Mode-Free Autonomous Ablation, 28 s Deterministic Evaluation, 0% Failure Across All Policies)

| Track (Command) | Policy (`best.pt`) | Obs Dim | Net Fwd (mm) | Mean $v_x$ (mm/s) | Mean $\omega_z$ (rad/s) | In-Pipe $v_x$ (mm/s) | In-Pipe Wall $F_w$ (N) | Post-Pipe $v_x$ (mm/s) | Yaw RMS Pre→In→Post (°) | Min Clr (mm) | EstNet $v_x$ MAE (mm/s) | EstNet Wall MAE | EstNet Drag MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **`open_hard_140`** (140 mm/s) | `v6a` Pure | 51D | 7115.9 | 278.1 | -0.0097 | — | — | — | 7.0 | 15.21 | 68.00 | 0.015 | 0.000 |
| | `v6b` +`q_err` | 65D | 5572.2 | 207.8 | +0.0066 | — | — | — | 6.9 | 5.04 | 39.61 | 0.017 | 0.000 |
| | `v6.1` +Elec | 65D | 5629.4 | 209.5 | -0.0054 | — | — | — | 7.2 | **12.00** | **38.61** | 0.024 | 0.000 |
| **`open_hard_turn_L`** (90, +0.20) | `v6a` Pure | 51D | 2145.0 | 169.2 | +0.0737 | — | — | — | 9.3 | 5.88 | 17.61 | 0.005 | 0.002 |
| | `v6b` +`q_err` | 65D | -573.6 | 141.6 | **+0.1941** | — | — | — | 6.8 | 4.22 | 37.64 | **0.002** | **0.001** |
| | `v6.1` +Elec | 65D | -201.5 | 63.7 | +0.0930 | — | — | — | 6.5 | 4.64 | **15.30** | 0.003 | **0.001** |
| **`pipe_3zone_soil`** (130 mm/s) | `v6a` Pure | 51D | 2229.3 | 73.2 | +0.0062 | 141.8 | 0.41 | 21.4 | 7.4→6.3→7.4 | 6.13 | 21.71 | 0.045 | 0.034 |
| | `v6b` +`q_err` | 65D | **2420.1** | 69.2 | +0.0025 | 140.6 | 0.36 | **35.6** | **15.8→6.8→10.2** | 5.37 | **14.68** | 0.038 | **0.032** |
| | `v6.1` +Elec | 65D | 2382.2 | 64.5 | -0.0013 | **165.4** | **0.09** | 28.8 | 6.1→5.6→7.1 | **7.46** | 18.24 | **0.014** | 0.038 |
| **`pipe_3zone_open`** (155 mm/s) | `v6a` Pure | 51D | 7042.0 | 289.6 | +0.0041 | 178.9 | 0.30 | 296.6 | 7.7→7.2→7.9 | **13.68** | 54.09 | 0.034 | 0.000 |
| | `v6b` +`q_err` | 65D | 5645.8 | 218.1 | -0.0006 | 177.5 | 0.39 | 219.3 | **15.8→7.6→7.3** | 5.44 | **37.74** | 0.040 | 0.000 |
| | `v6.1` +Elec | 65D | 6108.3 | 230.7 | -0.0054 | **199.6** | **0.16** | 228.4 | 5.6→7.0→7.8 | 11.13 | 42.82 | **0.030** | 0.000 |
| **`soft_soil_100`** (100 mm/s) | `v6a` Pure | 51D | 354.0 | 12.1 | +0.0031 | — | — | — | 4.6 | **7.23** | 8.62 | 0.002 | **0.025** |
| | `v6b` +`q_err` | 65D | **965.6** | **36.1** | -0.0067 | — | — | — | 10.9 | 4.18 | **7.90** | **0.001** | 0.026 |
| | `v6.1` +Elec | 65D | 616.2 | 21.0 | -0.0023 | — | — | — | 6.0 | 5.86 | 8.17 | 0.002 | 0.072 |
| **`carpet_100`** (100 mm/s) | `v6a` Pure | 51D | 940.0 | 34.0 | +0.0024 | — | — | — | 6.3 | **13.81** | 13.75 | 0.003 | **0.078** |
| | `v6b` +`q_err` | 65D | **1344.5** | **46.8** | -0.0035 | — | — | — | 7.9 | 5.01 | **9.47** | 0.004 | 0.123 |
| | `v6.1` +Elec | 65D | 1050.3 | 36.8 | -0.0015 | — | — | — | 5.5 | 8.91 | 11.62 | **0.002** | 0.237 |

> *Note: On `latest.pt` (r200), `v6b` further achieves `soft_soil_100 = 39.3 mm/s`, `carpet_100 = 50.1 mm/s`, `in_pipe_vx = 165.7 mm/s` (`in_pipe_wall_n = 0.22 N`) on `pipe_3zone_soil`, and `open_hard_turn_L wz = 0.2448 rad/s` (all `exited_pipe = True`, `failed = False`).*

### Markdown Table (3b: Mode-Conditioned `v5` Reference Across Terrains, `unified_v5_hispeed_14d/eval_v5_full.json`, `best.pt` @ r185)

| Mode & Scenario | Hard Floor `tail_vx` (mm/s) | Hard `min_clr` (mm) | Carpet `tail_vx` (mm/s) | Carpet `min_clr` (mm) | Soft Soil `tail_vx` (mm/s) | Soft Soil `min_clr` (mm) |
|---|---:|---:|---:|---:|---:|---:|
| `snake_120` (cmd 120) | 159.0 | 3.91 | 13.3 | 4.70 | 6.0 | 5.20 |
| `snake_160` (cmd 160) | **187.3** | 4.11 | 19.7 | 5.50 | 6.9 | 6.16 |
| `worm_110` (cmd 110) | 176.4 | 19.44 | 23.4 | 19.56 | 18.1 | 19.74 |
| `mixed_150` (cmd 150) | 152.3 | 3.98 | **32.8** | 4.96 | **18.3** | 5.32 |
| `snake_turn_L` (cmd +0.20 rad/s) | 148.6 ($\omega_z=0.1983$) | 4.39 | 8.9 ($\omega_z=0.0255$) | 5.05 | 3.0 ($\omega_z=0.0123$) | 6.00 |
| `snake_stop` / `worm_stop` | 0.0 / 0.0 | 3.42 / 19.40 | 0.0 / 0.0 | 2.56 / 19.41 | 0.0 / 0.0 | 2.32 / 19.27 |

> **Key Takeaway**: Without any mode ID or terrain label, `v6b` (`36.1 mm/s` on soft soil, `46.8 mm/s` on carpet) achieves **1.97×** the soft-soil speed of `v5`'s best manually selected mode (`mixed_150: 18.3 mm/s`), **5.23×** that of `v5` `snake_160` (`6.9 mm/s`), and **1.43×** `v5`'s best carpet speed (`32.8 mm/s`), while `v6.1` reduces in-pipe wall contact force to **0.09 N** (`pipe_3zone_soil`) and maintains minimum clearance $\ge 4.64\text{ mm}$ across all 6 tracks.

### LaTeX Table 3

```latex
\begin{table*}[t]
\centering
\caption{Mode-free pure-proprioceptive RL ablation across six deterministic evaluation tracks (28\,s each, 0\% failure rate across all policies) comparing 51D pure encoder+IMU (\texttt{v6a}), 65D with software servo tracking error $\mathbf{q}_{\mathrm{err}}$ (\texttt{v6b}), and 65D with DC Joule-heat regularization (\texttt{v6.1}), alongside mode-conditioned \texttt{v5} cross-terrain reference.}
\label{tab:v6_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccc}
\toprule
Track & Policy & Obs & Mean $v_x$ & In-Pipe $v_x$ & Pipe Wall $F_w$ & Post-Pipe $v_x$ & Min Clr & EstNet $v_x$ MAE & EstNet Wall / Drag \\
 & (\texttt{best.pt}) & Dim & (mm/s) & (mm/s) & (N) & (mm/s) & (mm) & (mm/s) & MAE \\
\midrule
\multirow{3}{*}{\texttt{open\_hard\_140}}
  & \texttt{v6a} Pure & 51D & 278.1 & --- & --- & --- & 15.21 & 68.00 & 0.015 / 0.000 \\
  & \texttt{v6b} $+\mathbf{q}_{\mathrm{err}}$ & 65D & 207.8 & --- & --- & --- & 5.04 & 39.61 & 0.017 / 0.000 \\
  & \texttt{v6.1} $+$Elec & 65D & 209.5 & --- & --- & --- & 12.00 & \textbf{38.61} & 0.024 / 0.000 \\
\midrule
\multirow{3}{*}{\texttt{open\_hard\_turn\_L}}
  & \texttt{v6a} Pure & 51D & 169.2 ($\omega_z{=}0.074$) & --- & --- & --- & 5.88 & 17.61 & 0.005 / 0.002 \\
  & \texttt{v6b} $+\mathbf{q}_{\mathrm{err}}$ & 65D & 141.6 (\textbf{$\omega_z{=}0.194$}) & --- & --- & --- & 4.22 & 37.64 & \textbf{0.002} / \textbf{0.001} \\
  & \texttt{v6.1} $+$Elec & 65D & 63.7 ($\omega_z{=}0.093$) & --- & --- & --- & 4.64 & \textbf{15.30} & 0.003 / \textbf{0.001} \\
\midrule
\multirow{3}{*}{\texttt{pipe\_3zone\_soil}}
  & \texttt{v6a} Pure & 51D & 73.2 & 141.8 & 0.41 & 21.4 & 6.13 & 21.71 & 0.045 / 0.034 \\
  & \texttt{v6b} $+\mathbf{q}_{\mathrm{err}}$ & 65D & 69.2 & 140.6 & 0.36 & \textbf{35.6} & 5.37 & \textbf{14.68} & 0.038 / \textbf{0.032} \\
  & \texttt{v6.1} $+$Elec & 65D & 64.5 & \textbf{165.4} & \textbf{0.09} & 28.8 & \textbf{7.46} & 18.24 & \textbf{0.014} / 0.038 \\
\midrule
\multirow{3}{*}{\texttt{pipe\_3zone\_open}}
  & \texttt{v6a} Pure & 51D & 289.6 & 178.9 & 0.30 & 296.6 & \textbf{13.68} & 54.09 & 0.034 / 0.000 \\
  & \texttt{v6b} $+\mathbf{q}_{\mathrm{err}}$ & 65D & 218.1 & 177.5 & 0.39 & 219.3 & 5.44 & \textbf{37.74} & 0.040 / 0.000 \\
  & \texttt{v6.1} $+$Elec & 65D & 230.7 & \textbf{199.6} & \textbf{0.16} & 228.4 & 11.13 & 42.82 & \textbf{0.030} / 0.000 \\
\midrule
\multirow{4}{*}{\texttt{soft\_soil\_100}}
  & \texttt{v5} Best (\texttt{mixed}) & 54D$^*$ & 18.3 & --- & --- & --- & 5.32 & --- & --- \\
  & \texttt{v6a} Pure & 51D & 12.1 & --- & --- & --- & \textbf{7.23} & 8.62 & 0.002 / \textbf{0.025} \\
  & \texttt{v6b} $+\mathbf{q}_{\mathrm{err}}$ & 65D & \textbf{36.1} (\textbf{3.0$\times$}) & --- & --- & --- & 4.18 & \textbf{7.90} & \textbf{0.001} / 0.026 \\
  & \texttt{v6.1} $+$Elec & 65D & 21.0 & --- & --- & --- & 5.86 & 8.17 & 0.002 / 0.072 \\
\midrule
\multirow{4}{*}{\texttt{carpet\_100}}
  & \texttt{v5} Best (\texttt{mixed}) & 54D$^*$ & 32.8 & --- & --- & --- & 4.96 & --- & --- \\
  & \texttt{v6a} Pure & 51D & 34.0 & --- & --- & --- & \textbf{13.81} & 13.75 & 0.003 / \textbf{0.078} \\
  & \texttt{v6b} $+\mathbf{q}_{\mathrm{err}}$ & 65D & \textbf{46.8} & --- & --- & --- & 5.01 & \textbf{9.47} & 0.004 / 0.123 \\
  & \texttt{v6.1} $+$Elec & 65D & 36.8 & --- & --- & --- & 8.91 & 11.62 & \textbf{0.002} / 0.237 \\
\bottomrule
\end{tabular}%
}
\vspace{2pt}
\begin{minipage}{0.98\textwidth}
\footnotesize $^*$\texttt{v5} (\texttt{unified\_v5\_hispeed\_14d/best.pt}) requires explicit manual \texttt{mode\_id} one-hot conditioning (\texttt{snake}/\texttt{worm}/\texttt{mixed}) and has no EstNet head; \texttt{v6a}/\texttt{v6b}/\texttt{v6.1} are 100\% mode-free. All \texttt{pipe\_3zone\_*} evaluations achieve \texttt{exited\_pipe=True}.
\end{minipage}
\end{table*}
```

---

## Table 4: Continuous 3-Zone Autonomous Phase Transition (`v6b` Mode-Free vs `v5` Fixed Snake Baseline)

**Verified Sources**:
- [`runs/v6b_65d_qerr/v6b_autonomous_pipe_traj.json`](../../runs/v6b_65d_qerr/v6b_autonomous_pipe_traj.json)
- [`runs/v6b_65d_qerr/v6b_autonomous_pipe_traj.npz`](../../runs/v6b_65d_qerr/v6b_autonomous_pipe_traj.npz) (`duration = 25.96 s`, 650 frames @ 25 Hz; Zone I Hard Floor `0.00–2.60 s`, Zone II Narrow Pipe ID 236 mm `x ∈ [-1.35, -2.35] m`, Zone III Soft Soil `x < -2.42 m`)

### Markdown Table 4

| Metric | `v6b` Mode-Free Autonomous (65D) | `v5` Fixed-Snake Baseline | Relative Improvement |
|---|---:|---:|---:|
| **Total Net Forward Displacement (25.96 s)** | **2,346.8 mm** | 1,081.2 mm | **+117.0%** (2.17×) |
| Zone I (Open Hard, 2.6 s) Mean Speed / Yaw RMS | 142.5 mm/s / 10.2° | 125.9 mm/s / 9.8° | +13.2% speed |
| Zone II (236 mm Pipe, all `in_pipe` 19.0 s) Mean Speed / Yaw RMS | **96.4 mm/s / 8.2°** | 36.4 mm/s / 14.1° | **+165% speed**, spontaneous yaw contraction |
| Zone II Pure Pipe (pre-soil `2.64–13.04 s`, 10.4 s) Speed / Yaw RMS | **145.5 mm/s / 6.0°** | 52.9 mm/s / 14.4° | **2.75× speed**, **−40.5% yaw** vs Zone I |
| Zone II Mean Pipe Wall Contact Force | **0.49 N** | 4.72 N | **-89.6%** (9.6× lower) |
| Zone II Peak Pipe Wall Contact Force | **7.02 N** | 21.99 N | **-68.1%** (3.1× lower) |
| Zone III (Soft Soil Only, 4.4 s, `21.60–25.96 s`) Mean Speed / Yaw RMS | **32.9 mm/s / 10.0°** | 13.5 mm/s* / 14.1°* | **+144% speed**, spontaneous re-expansion |
| EstNet Proprioceptive Velocity Correlation $r$ / MAE | **0.955** / 14.35 mm/s | — | Zero accelerometer / zero vision |
| EstNet Proprioceptive Terrain Drag Correlation $r$ | **0.966** | — | Zero external terrain sensor |

*\*During `t ∈ [21.60, 25.96] s`, `v6b` has fully exited the pipe onto soft soil (`wall_n = 0.00 N`), whereas the `v5` fixed-snake baseline remains jammed inside the pipe slamming the wall (`7.28 N` mean, `21.99 N` peak).*

### LaTeX Table 4

```latex
\begin{table}[t]
\centering
\caption{Continuous 3-zone autonomous phase-transition benchmark (25.96\,s: Zone~I open hard floor $\to$ Zone~II 236\,mm narrow pipe $\to$ Zone~III soft soil; \texttt{v6b\_autonomous\_pipe\_traj.json}).}
\label{tab:phase_transition}
\begin{tabular}{lccc}
\toprule
Metric & \texttt{v6b} Autonomous & \texttt{v5} Fixed Snake & Improvement \\
\midrule
Net Displacement (mm) & \textbf{2,346.8} & 1,081.2 & \textbf{+117.0\%} \\
Zone~I Speed / Yaw RMS & 142.5\,mm/s / 10.2$^\circ$ & 125.9\,mm/s / 9.8$^\circ$ & +13.2\% \\
Zone~II Pipe Speed / Yaw (all 19.0\,s) & \textbf{96.4\,mm/s / 8.2$^\circ$} & 36.4\,mm/s / 14.1$^\circ$ & \textbf{+165\%} \\
Zone~II Pure Pipe (pre-soil, 10.4\,s) & \textbf{145.5\,mm/s / 6.0$^\circ$} & 52.9\,mm/s / 14.4$^\circ$ & \textbf{2.75$\times$} \\
Zone~II Mean Wall Force & \textbf{0.49\,N} & 4.72\,N & \textbf{$-$89.6\%} \\
Zone~II Peak Wall Force & \textbf{7.02\,N} & 21.99\,N & \textbf{$-$68.1\%} \\
Zone~III Soil Only (4.4\,s) & \textbf{32.9\,mm/s / 10.0$^\circ$} & 13.5\,mm/s / 14.1$^\circ$ & \textbf{+144\%} \\
EstNet $v_x$ / Drag Corr. $r$ & \textbf{0.955 / 0.966} & --- & Pure proprioception \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 5: 5-Axis Sim-to-Real Domain Randomization Parameter Specification

**Verified Source**: [`src/serpworm/config.py`](../../src/serpworm/config.py) (`DomainRandConfig` & `HardwareConfig.v5_realistic()`).

### LaTeX Table 5

```latex
\begin{table*}[t]
\centering
\caption{Complete 5-axis Sim-to-Real domain randomization specification (\texttt{DomainRandConfig} and nominal \texttt{v5\_realistic} preset in \texttt{src/serpworm/config.py}; 20-rollout linear ramp-in upon entering Stage~3).}
\label{tab:domain_rand}
\begin{tabular}{lllc}
\toprule
Axis & Parameter (Symbol) & Nominal (\texttt{v5\_realistic}) & Randomization Range \\
\midrule
\multirow{7}{*}{\textbf{1. Ground \& Terrain}}
  & Wheel kinetic friction ($\mu_{\mathrm{fwd}}$) & 0.80 & $[0.42,\, 1.15]$ \\
  & Rolling resistance ($\mu_{\mathrm{rr}}$) & 0.060 & $[0.025,\, 0.140]$ \\
  & Anisotropy / Stribeck ($\mu_{\mathrm{lat}}/\mu_{\mathrm{fwd}}$, $\mu_s/\mu_k$) & 1.05, 1.18 & $[0.85, 1.35]$, $[1.00, 1.35]$ \\
  & Hunt--Crossley stiffness scale ($k_{\mathrm{HC}}$) & $1.0\times$ ($1.6\times 10^5$) & $[0.15,\, 1.50]\times$ \\
  & Rubber patch radius ($r_{\mathrm{patch}}$) & 8.0\,mm & $[4.0,\, 13.0]$\,mm \\
  & Ratchet pawl backlash ($\psi_{\max}$) & 10.0$^\circ$ & $[4.0^\circ,\, 18.0^\circ]$ \\
  & Floor slope / micro-wave ($s_{x,y}$, $A$, $k_{x,y}$) & 0$^\circ$, 0\,mm & $\pm 3.5^\circ$, $[0, 2.5]$\,mm, $[6, 14]$\,rad/m \\
\midrule
\multirow{3}{*}{\textbf{2. Structure \& Mass}}
  & Global mass scale ($s_m$) & $1.0\times$ (2.76\,kg) & $[0.85,\, 1.15]\times$ \\
  & Per-plate asymmetric mass jitter & 0\% & $\pm 15\%$ (10 plates independent) \\
  & Structural Rayleigh damping ($\beta_R$) & 0.0005\,s & $[0.0002,\, 0.0018]$\,s \\
\midrule
\multirow{6}{*}{\textbf{3. Tendon \& Actuators}}
  & Capstan guide friction ($\mu_c\theta_{\mathrm{wrap}}$) & 0.18 & $[0.04,\, 0.36]$ \\
  & Hand-tied tendon slack jitter ($\Delta s_0$) & 11.4\,mm & $\pm 3.0$\,mm \\
  & Yaw gearbox static breakaway ($\tau_{\mathrm{sf}}$) & 0.06\,N$\cdot$m & $[0.00,\, 0.18]$\,N$\cdot$m \\
  & Servo angular deadband ($\delta_{\mathrm{db}}$) & 0.5$^\circ$ & $[0.0^\circ,\, 1.8^\circ]$ \\
  & Torque / speed / PD ($K_p, K_d$) / $\tau_{\mathrm{filt}}$ scales & $1.0\times$ & $[0.72,1.15]$, $[0.75,1.25]$, $[0.70,1.30]$, $[0.70,1.45]\times$ \\
  & Serial bus command delay & 0\,ms & $p=0.35$, 1--2 ticks (20--40\,ms) \\
\midrule
\multirow{3}{*}{\textbf{4. Sensors \& IMU}}
  & Joint encoder zero-offset bias & 0$^\circ$ & $\pm 2.0^\circ$ \\
  & MEMS gyroscope zero-rate bias & 0\,rad/s & $\pm 0.04$\,rad/s \\
  & IMU mounting roll/pitch tilt error & 0$^\circ$ & $\pm 2.0^\circ$ \\
\midrule
\textbf{5. Disturbances}
  & Rigid-body velocity push ($\Delta v_{xy}$, $\Delta\omega_z$) & 0 & $\pm 0.06$\,m/s, $\pm 0.25$\,rad/s ($p=0.5$ / 4.0\,s) \\
\bottomrule
\end{tabular}
\end{table*}
```
