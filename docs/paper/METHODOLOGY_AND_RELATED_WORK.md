# SerpWorm-RL: Related Work, Physics Formulation, and Proprioceptive RL Methodology

> 本文档面向 *Nature Communications* / *Science Robotics* / *IEEE RA-L* 撰写，采用中英双语技术术语与标准 LaTeX 公式。所有物理常数、网络维度与域随机化区间均直接核自源码 [`src/serpworm/config.py`](../../src/serpworm/config.py)、[`src/serpworm/physics/gpu_core.py`](../../src/serpworm/physics/gpu_core.py)、[`src/serpworm/physics/wheel_stiction.py`](../../src/serpworm/physics/wheel_stiction.py)、[`src/serpworm/physics/pipe_geometry.py`](../../src/serpworm/physics/pipe_geometry.py)、[`src/serpworm/envs/autonomous_env.py`](../../src/serpworm/envs/autonomous_env.py)、[`src/serpworm/envs/rewards.py`](../../src/serpworm/envs/rewards.py)、[`src/serpworm/rl/models.py`](../../src/serpworm/rl/models.py) 与 [`src/serpworm/rl/train_autonomous.py`](../../src/serpworm/rl/train_autonomous.py)；全部 26 篇核心参考文献均已通过本地 PDF 完整性校验（见 [`ref/README.md`](../../../ref/README.md)）。

---

## 1. Related Work（相关工作）

仿生蛇形与蠕动连续体机器人在非结构化狭小空间（如工业管道、灾后废墟与颗粒软土）中的自主多模态运动，横跨**地面接触与土力学**、**连续体弹性杆/薄壳高效求解**、**拉绳摩擦滞回建模**以及**执行器标定与纯本体 Sim-to-Real 强化学习**四个研究领域。下述综述按这四个核心维度系统梳理本研究与 26 篇前沿文献及经典力学原典的对应关系与本质区别。

### 1.1 Ground Contact Mechanics, Anisotropic Friction, and Granular Terramechanics（地面接触力学、各向异性摩擦与颗粒土力学）

蛇形与蠕虫式无足机器人的推进本质上依赖于**方向各向异性摩擦**（anisotropic friction）将内部周期性形变整流为净前向位移。在仿蚯蚓/尺蠖蠕动机理方面，Ge 等 [arXiv:1707.04084, 2017] 建立了摩擦驱动软体蠕虫的动力学与控制模型；Gidoni [arXiv:1710.08340, 2017] 给出了率无关（rate-independent）各向异性摩擦爬行的严格数学理论，揭示了停滞域与步态效率边界；近期 Javadi 等 [arXiv:2510.03660, 2025] 进一步利用各向异性刚毛实现了水陆两栖无系绳尺蠖机器人的快速爬行。在蛇形连续体与轮地接触方面，Liu 等 [arXiv:2105.04608, 2021] 提出了面向软体蛇的接触感知中枢模式发生器（CPG）强化学习方法；Arachchige 等 [arXiv:2303.02291, 2023] 验证了分布式地面接触下的软体蛇动力学模型；Zhang 等 [arXiv:2605.24985, 2026] 将非对称摩擦与 3D 连续体接触推广至三维异质环境导航。

然而，大多数现有 RL 仿真仍采用显式线性 Kelvin–Voigt 罚函数弹簧–阻尼法向接触与理想无隙点接触摩擦，这在 $1\text{ ms}$ 大步长与真实地形上会引发触地力脉冲跳变、静置数值漂移与缺乏车辙沉陷等问题。经典接触力学中，Hunt & Crossley（ASME JAM, 1975）提出零脉冲非线性粘弹性接触律 $F_n = k \delta^p (1 + \alpha \dot{\delta})$，而 Bekker（1960）与 Wong（2008）奠定了颗粒土压力–沉陷与推土阻力的经典地面力学框架。在机器人接触隐式求解方向，Xie 等 [arXiv:1809.05555, 2018] 通过多凸接触面片（convex contact patches）刻画了面片旋转摩擦力矩；Crotti 等 [arXiv:2412.03191, 2024] 在 MuJoCo 中验证了柔性足底在粗糙地形上的接触孪生标定；Salagame 等 [arXiv:2508.02953, 2025] 采用 Moreau 时间步进法实现了蛇形机器人的接触隐式优化；Hublikar [arXiv:2512.05008, 2025] 针对颗粒与柔顺地面（Project Chrono SCM）构建了蛇形机器人接触隐式仿真。

与上述工作相比，本文针对 **20 个被动单向棘轮橡胶轮**（机械单向离合 + 棘爪齿隙 + 宽面橡胶印迹）提出了**隐式各向异性棘轮面片接触模型（Implicit ARPC）**：在同一可捕获 CUDA Graph 的 FP64 内核中统一融合了 Hunt–Crossley 零脉冲法向接触、弹塑性车辙松弛与侧壁推土阻力、Stribeck 静动摩擦椭圆、棘轮单向齿隙死区、圆形接触面片垂直枢转阻尼力矩，以及显式耦合至隔板速度求解的对称正定（SPD）Contact-Schur 隐式切线矩阵，在大步长 $\Delta t = 1.0\text{ ms}$ 下将静态漂移抑制 348 倍并支持 1024 并行环境跨地形实时切换。

### 1.2 Continuum Rod/Thin-Shell Solvers and Implicit Contact Algorithms（连续体杆/薄壳求解器与隐式接触算法）

我们的五体节机器人主体由 **40 片大变形预弯 65Mn 弹簧钢片**（宽 $16\text{ mm}$、厚 $0.22\text{ mm}$、无应力拱高 $30\text{ mm}$）构成 5 个闭环并联弹性骨架，其弯曲刚度各向异性比达 $(w/t)^2 \approx 5.29 \times 10^3$，对求解器数值稳定性与吞吐率提出极高要求。

在通用连续体与薄壳力学求解领域，Li 等 [arXiv:2012.04457, 2020] 提出的 C-IPC（Codimensional Incremental Potential Contact）为薄片/杆件等余维结构提供了无穿透隐式屏障接触求解；Du 等 [arXiv:2101.05917, 2021] 提出的 DiffPD 借助可微投影动力学与 Cholesky 预分解显著加速了刚性软体的隐式积分与参数辨识；Wang 等 [arXiv:2402.12315, 2024] 验证了可变曲率软体连续体的 Cosserat 杆建模精度；Lahoti 等 [arXiv:2512.09911, 2025] 推出的 Py-DiSMech 基于离散微分几何（DDG）与隐式罚能量接触实现了可扩展软体机器人仿真；Parthasarathy 等 [arXiv:2605.13766, 2026] 推出的 Elastica++ 为大规模耦合 Cosserat 杆集群提供了高性能多物理场求解框架。

尽管上述框架在通用连续体力学上表现优异，但针对**闭环多钢片并联体节 + 刚性隔板 + 拉绳 + 轮地/管壁接触**的大规模 RL 采样，通用 CPU/稀疏组装求解器难以兼顾 FP64 精度与数千步/秒的策略吞吐。本文利用 40 片钢片与 10 块刚性隔板的二分拓扑稀疏结构，研发了融合 **块 Thomas 链式消元（每片 15 个内部 6-DOF 节点）+ 体节级 $12\times 12$ 隔板 Schur 补求解** 的专用 FP64 CUDA 融合内核（`fused_cuda`），配合全图 CUDA Graph 捕获，在单张 RTX 5090 上实现 1024 环境 $4746\text{ policy steps/s}$（$474.6\times$ 实时速率，相比 PyTorch `segment_lu` 加速 $153\times$），且保留与 UTM 5 循环实测压载曲线 $R^2 = 0.999857$、$\text{RMSE} = 0.138\text{ N}$ 的物理保真度。

### 1.3 Tendon Guide-Hole Capstan Friction and Hysteresis Modeling（拉绳导孔 Capstan 摩擦与滞回建模）

拉绳驱动（tendon-driven）连续体机器人在经过穿孔隔板与导向槽时，绳缆包角引起的 **Capstan 指数摩擦**与拉/放方向切换滞回会显著扭曲驱动刚度与运动学映射。Kim 等 [arXiv:2109.06907, 2021] 提出了针对拉绳连续体机械臂的形状自适应滞回补偿方法；Cho 等 [arXiv:2404.03816, 2024] 通过学习型深度解码器补偿了非线性走线拉绳机器人的前向运动学摩擦滞回；Zhang 等 [arXiv:2608.13396, 2026] 系统研究了绞盘（Capstan）驱动连续体手术机器人的张力摩擦建模与感知；Shen 等在 IEEE RA-L 奠基工作与最新 IJRR 扩展 [arXiv:2611.18546, 2026] 中将连续体机器人导孔 Capstan 摩擦与环境接触统一表述为线性互补问题（LCP）以精确捕获粘滑切换。

受上述理论启发，我们将传统恒半径滑轮绕绳近似升级为经双激光台架（实验 126–128）1:1 标定的**偏心舵盘曲柄几何（`crank_calibrated`）+ 被动导孔 Capstan 摩擦滞回模型**：拉绳收紧时按 $e^{-\mu_c \theta_{\mathrm{wrap}} \tanh(\dot{\theta}_+/v_0)}$ 衰减有效传递张力并按 $e^{+\mu_c \theta_{\mathrm{wrap}}}$ 放大舵机负载力矩，同时附加与 $\mu_c \theta_{\mathrm{wrap}}$ 成正比的导孔阻尼耗散，并在域随机化中覆盖 $\mu_c \theta_{\mathrm{wrap}} \in [0.04, 0.36]$ 与手工系绳余量 $\pm 3.0\text{ mm}$ 的实物装配不确定性。

### 1.4 Actuator Calibration and Proprioceptive Sim-to-Real RL（执行器标定与纯本体 Sim-to-Real 强化学习）

从仿真向真实硬件迁移（Sim-to-Real）面临执行器非理想动态、静摩擦死区与无外接传感器条件下的部分可观测性三重挑战。Hwangbo 等 [arXiv:1901.08652, 2019]（ActuatorNet, *Science Robotics*）证明了高保真执行器动态建模对足式机器人敏捷迁移的决定性意义；Hu 等 [arXiv:2503.01255, 2025] 最新定量分析表明，忽视**静摩擦（static friction）**会导致 RL 策略在实物部署时出现严重性能退化，直接支撑了我们在偏航舵机齿轮箱中显式引入库仑起动静摩擦 $\tau_{\mathrm{sf}}$、在收绳舵机中引入角度死区 $\delta_{\mathrm{db}}$ 以及在轮地接触中引入 Stribeck 静摩擦增益的设计。

在本体状态估计与隐式环境自适应架构方面，Kumar 等 [arXiv:2107.04034, 2021]（RMA, *RSS*）提出利用特权环境编码与本体历史自适应实现快速地形迁移；Nahrendra 等 [arXiv:2301.10602, 2023]（DreamWaQ, *ICRA*）与 Long 等 [arXiv:2312.11460, 2023]（HIMLoco, *ICLR 2024*）进一步在单阶段非对称 Actor–Critic 训练中引入显式速度与隐式上下文估计器；Mittal 等 [arXiv:2403.04359, 2024]（*ICRA*）系统确立了机器人策略学习中的任务镜像对称性理论；Hou 等 [arXiv:2511.02192, 2025] 则基于 PyElastica 定量比较了软体连续体机器人的集中式与分布式 RL 控制。

区别于足式机器人依赖足端接触周期或软体蛇依赖外接视觉/预设模式 ID 的方案，本文提出**零模式 ID、零步态模板、零外接传感器的纯本体自主相变架构（v6/v6.1）**：仅凭 14 路串口舵机编码器（及纯软件伺服跟踪误差 $\mathbf{q}_{\mathrm{err}} = \mathbf{q}_{\mathrm{tgt}} - \mathbf{q}_{\mathrm{meas}}$）与单颗 6 轴 IMU（不含线加速度计），通过因果 1D-TCN + 梯度截断的 7 维物理状态估计器 `EstNet` 实时反演机体速度、管壁约束强度/侧向与地形滚动阻尼，使单一 14D 全开放策略在“开阔硬地 $\to$ 236 mm 窄管 $\to$ 松软沙土”间自发涌现蛇行 $\leftrightarrow$ 蠕动 $\leftrightarrow$ 混合步态相变。

---

## 2. Physics Engine and Contact Formulation（物理引擎与接触力学建模）

### 2.1 40-Strip Corotational Euler–Bernoulli Beam & Block-Thomas–Schur Solver（40 钢片共旋欧拉–伯努利梁与块消元–Schur 求解器）

五体节蛇蠕机器人由 $N_{\mathrm{seg}} = 5$ 个相同体节串联而成，包含 $P = 2N_{\mathrm{seg}} = 10$ 块刚性隔板与 $S = 8N_{\mathrm{seg}} = 40$ 片周向均布的预弯 65Mn 弹簧钢片。每片钢片离散为 $E = 16$ 个共旋欧拉–伯努利（corotational Euler–Bernoulli）梁单元，共含 $D = P + S(E - 1) = 610$ 个 6 自由度空间节点（$3660$ 广义自由度）。

#### 2.1.1 Parabolic Stress-Free Arch Geometry（抛物预弯无应力拓扑）

设体节内某片钢片在前、后隔板螺栓夹持孔对的中点分别为 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^3$，单位径向外法向为 $\hat{\mathbf{r}}$，无应力拱高为 $b_0$（UTM 标定值 $b_0 = 30.0\text{ mm}$，旧默认值 $50.72\text{ mm}$）。沿无量纲弧长参数 $u \in [0, 1]$，钢片无应力中心线 $\mathbf{p}_0(u)$ 及其切向为：

$$
\mathbf{p}_0(u) = \mathbf{a} + u(\mathbf{b} - \mathbf{a}) + 4 b_0 u(1 - u)\hat{\mathbf{r}}, \qquad
\mathbf{t}_0(u) = \frac{(\mathbf{b} - \mathbf{a}) + 4 b_0 (1 - 2u)\hat{\mathbf{r}}}{\|(\mathbf{b} - \mathbf{a}) + 4 b_0 (1 - 2u)\hat{\mathbf{r}}\|_2}.
$$

矩形截面尺寸为宽度 $w = 16.0\text{ mm}$、厚度 $t = 0.220\text{ mm}$（旧默认 $0.150\text{ mm}$），弹性模量 $E_s = 200\text{ GPa}$、泊松比 $\nu = 0.30$、剪切模量 $G = E_s / [2(1+\nu)]$、密度 $\rho = 7850\text{ kg/m}^3$。截面面积与主惯性矩、Saint-Venant 扭转常数为：

$$
A = wt, \qquad I_y = \frac{wt^3}{12}, \qquad I_z = \frac{tw^3}{12}, \qquad
J = wt^3\left[\frac{1}{3} - 0.21\frac{t}{w}\left(1 - \frac{t^4}{12w^4}\right)\right].
$$

当启用宽薄带壳修正（`strip_thin_shell_compat=True`）时，面外弯曲模量取 Kirchhoff–Love 板刚度 $E_y = E_s / (1 - \nu^2)$，扭转常数加入 Vlasov 约束翘曲修正 $J_{\mathrm{eff}} = J\left[1 + \frac{w^2}{12 L_c^2(1-\nu)}\right]$（$L_c = 0.145\text{ m}$）。

#### 2.1.2 Corotational Frame & Linearized Implicit Integration with Per-Env Rayleigh Damping（共旋架与逐环境瑞利阻尼隐式积分）

对任意梁单元 $e \in \{1, \dots, E\}$，由端点世界位姿 $(\mathbf{p}_0, \mathbf{q}_0), (\mathbf{p}_3, \mathbf{q}_1)$ 构造三次 Hermite 样条控制点 $\mathbf{p}_1 = \mathbf{p}_0 + \frac{L_e}{3}\mathbf{t}_0$、$\mathbf{p}_2 = \mathbf{p}_3 - \frac{L_e}{3}\mathbf{t}_1$，定义单元中点与共旋姿态四元数 $(\mathbf{c}_e, \mathbf{r}_e)$：

$$
\mathbf{c}_e = \frac{\mathbf{p}_0 + 3\mathbf{p}_1 + 3\mathbf{p}_2 + \mathbf{p}_3}{8}, \qquad
\mathbf{t}_m = \mathrm{unit}\!\left(-\tfrac{3}{4}\mathbf{p}_0 - \tfrac{3}{4}\mathbf{p}_1 + \tfrac{3}{4}\mathbf{p}_2 + \tfrac{3}{4}\mathbf{p}_3\right).
$$

将两端点变换至单元局部共旋架得到 12 维局部广义变形向量 $\mathbf{u}_e \in \mathbb{R}^{12}$ 及其相对无应力参考构型 $\mathbf{u}_{e,0}$ 的增量 $\Delta\mathbf{u}_e = \mathbf{u}_e - \mathbf{u}_{e,0}$（当装配预应力比例 $p_f \in [0, 1]$ 非零时，$\mathbf{u}_{e,0} = (1 - p_f)\mathbf{u}_{e,\mathrm{arch}} + p_f \mathbf{u}_{e,\mathrm{flat}}$）。设 $\mathbf{T}_e \in \mathbb{R}^{12\times 12}$ 为世界速度至共旋局部速度的坐标映射矩阵，局部刚度与一致质量矩阵为 $\mathbf{K}_e, \mathbf{M}_e \in \mathbb{R}^{12\times 12}$，则弹性恢复力与世界系矩阵为：

$$
\mathbf{f}_{\mathrm{el},e} = -\mathbf{T}_e^\top \mathbf{K}_e (\mathbf{u}_e - \mathbf{u}_{e,0}), \qquad
\mathbf{K}_e^w = \mathbf{T}_e^\top \mathbf{K}_e \mathbf{T}_e, \qquad
\mathbf{M}_e^w = \mathbf{T}_e^\top \mathbf{M}_e \mathbf{T}_e.
$$

引入按环境独立随机化的**结构瑞利阻尼（per-env Rayleigh damping）** $\mathbf{D}^{(b)} = \alpha_R \mathbf{M} + \beta_R^{(b)} \mathbf{K}$（其中 $\alpha_R = 0.05\text{ s}^{-1}$，名义装配刚度阻尼 $\beta_R = 0.0005\text{ s}$，域随机化覆盖 $\beta_R^{(b)} \in [0.0002, 0.0018]\text{ s}$），以及按隔板分布的非对称质量矩阵 $\mathbf{M}_p^{(b)} = s_m^{(b)} \mathrm{diag}(m_{p}^{(b)}\mathbf{I}_3, \frac{m_p^{(b)}}{0.28}\mathbf{I}_{\mathrm{plate}})$（`v5_realistic` 名义 10 块隔板质量为 $(0.27, 0.33, 0.23, 0.33, 0.23, 0.33, 0.23, 0.33, 0.23, 0.25)\text{ kg}$，总重 $2.76\text{ kg}$）。在时间步长 $h = \Delta t$（默认 $1.0\text{ ms}$）下的单步线性化隐式欧拉速度方程为：

$$
\left[\,(1 + h\alpha_R)\mathbf{M} + (h\beta_R^{(b)} + h^2)\mathbf{K} + \mathbf{K}_{\mathrm{contact}}^{(b)}\,\right]\mathbf{v}^{k+1}
= \mathbf{M}(\mathbf{v}^k + h\mathbf{g}) + h\left(\mathbf{f}_{\mathrm{el}}(\mathbf{x}^k) + \mathbf{f}_{\mathrm{ext}}(\mathbf{x}^k, \mathbf{v}^k)\right).
$$

记局部单元系统矩阵 $\mathbf{A}_e^{(b)} = (1 + h\alpha_R)\mathbf{M}_e^w + (h\beta_R^{(b)} + h^2)\mathbf{K}_e^w$。由于每片钢片内部的 $E - 1 = 15$ 个节点构成严格块三对角链，仅通过两端点与相邻两块刚性隔板相连，系统通过**两级精确直接求解**完成：
1. **钢片内部块 Thomas 消元**：对全部 $B \times 40$ 条钢片并行执行 15 步 $6\times 6$ 块三对角前代/回代，同步求出内部节点受右端载荷项 $\mathbf{z}_{s,i}^{(0)} \in \mathbb{R}^6$ 及端点单位扰动响应 $\mathbf{Z}_{s,i}^{(\mathrm{end})} \in \mathbb{R}^{6\times 12}$（若启用 600 节点钢片接触，则在对角块直接叠加 $3\times 3$ 隐式接触刚度 $\mathbf{K}_{\mathrm{strip},i}$）；
2. **体节级 Schur 补隔板求解**：将每个体节 8 片钢片凝聚后的 $12\times 12$ 界面刚度 $\mathbf{S}_s$ 与等效载荷 $\mathbf{r}_s$ 累加至所属两块隔板，连同隔板惯量 $(1 + h\alpha_R)\mathbf{M}_p^{(b)}$ 及隐式接触 Schur 矩阵 $\mathbf{K}_{\mathrm{contact},p}^{(b)}$ 构成 5 个独立 $12\times 12$ 块线性系统求解 $\mathbf{v}_p^{k+1}$，随后 $O(1)$ 回代全体内部节点速度 $\mathbf{v}_{\mathrm{int}}^{k+1}$。

最终位置与姿态通过 $\mathrm{SO}(3)$ 指数映射更新（`sinc` 形式确保小角度精度）：

$$
\mathbf{x}_i^{k+1} = \mathbf{x}_i^k + h\,\mathbf{v}_{i,\mathrm{lin}}^{k+1}, \qquad
\mathbf{q}_i^{k+1} = \mathrm{unit}\!\left(\exp\!\left(\tfrac{h}{2}\boldsymbol{\omega}_i^{k+1}\right) \otimes \mathbf{q}_i^k\right).
$$

---

### 2.2 Implicit ARPC: Anisotropic Ratchet-Patch Contact（隐式各向异性棘轮面片接触模型）

每块刚性隔板底部对称安装左右 2 个被动单向棘轮橡胶轮（整机共 20 轮，轮半径 $r_w = 18\text{ mm}$，轮转子惯量 $I_w = \frac{1}{2} m_w r_w^2$，$m_w = 0.009\text{ kg}$）。给定第 $w$ 个轮缘接触点 $\mathbf{p}_c = \mathbf{x}_p + \mathbf{r}_{\mathrm{arm}}$（其中 $\mathbf{r}_{\mathrm{arm}} = \mathbf{r}_{\mathrm{mount}} + \mathbf{r}_{\mathrm{rim}}$）与地形局部曲面高度/单位法向 $(z_g, \hat{\mathbf{n}}_g)$，定义法向穿透深度、法向速度与局部切向基 $\{\hat{\mathbf{e}}_{\mathrm{fwd}}, \hat{\mathbf{e}}_{\mathrm{lat}}\}$：

$$
\delta = (z_g(\mathbf{p}_{c,xy}) - p_{c,z})\, n_{g,z}, \qquad
v_n = \mathbf{v}_c \cdot \hat{\mathbf{n}}_g, \qquad
\hat{\mathbf{e}}_{\mathrm{fwd}} = \mathrm{unit}(\hat{\mathbf{n}}_g \times \hat{\mathbf{a}}_{\mathrm{axle}}), \qquad
\hat{\mathbf{e}}_{\mathrm{lat}} = \hat{\mathbf{n}}_g \times \hat{\mathbf{e}}_{\mathrm{fwd}}.
$$

#### 2.2.1 Hunt–Crossley Normal Contact & Elasto-Plastic Soil Rut Hysteresis（法向 Hunt–Crossley 接触与弹塑性车辙滞回）

在柔顺与松软土质上，轮下维护持久塑性车辙状态 $r \ge 0$（`soil_sinkage`）。当卸载刚度倍比 $\gamma_u > 1$ 时，随轮水平移速 $v_h = \sqrt{v_0^2 + v_1^2}$ 按松弛长度 $\lambda_{\mathrm{rut}}$ 指数衰减并记录最大塑性残余沉陷：

$$
r_{\mathrm{rel}} = r \exp\!\left(-\frac{v_h h}{\lambda_{\mathrm{rut}}}\right), \qquad
r_{\mathrm{new}} =
\begin{cases}
\max\!\left(r_{\mathrm{rel}},\;\left(1 - \gamma_u^{-1}\right)\max(\delta, 0)\right), & \gamma_u > 1, \\
0, & \gamma_u \le 1.
\end{cases}
$$

当满足接触条件 $\delta > r_{\mathrm{rel}}$ 且 $\delta > 0$ 时，准静态弹性承载力取初载分支（Bekker 幂律）与较陡卸载分支的下包络（在 $\delta = r_{\mathrm{rel}} / (1 - \gamma_u^{-1})$ 处 $C^0$ 连续），再乘以 Hunt–Crossley 零脉冲非负耗散因子：

$$
F_{\mathrm{el}}(\delta, r_{\mathrm{rel}}) = \min\!\Big(k_{\mathrm{HC}}\,\delta^p,\;\; \gamma_u k_{\mathrm{HC}}\,(\delta - r_{\mathrm{rel}})\,\delta^{p-1}\Big),
$$

$$
F_n = F_{\mathrm{el}}(\delta, r_{\mathrm{rel}})\,\max\!\left(0,\; 1 - \alpha_{\mathrm{HC}} v_n\right).
$$

在触地瞬间（$\delta \to 0^+$），由于 $F_{\mathrm{el}} \propto \delta^p \to 0$，法向阻尼项 $-\alpha_{\mathrm{HC}} F_{\mathrm{el}} v_n$ 严格趋近于零，彻底消除了传统 Kelvin–Voigt 模型 $F_n = k_n \delta - c_n v_n$ 在零穿透瞬间由 $-c_n v_n$ 引起的非物理力阶跃脉冲。对应隐式 Contact-Schur 的法向正定刚度与阻尼切线为：

$$
k_{n,\mathrm{tan}} = \frac{\partial F_{\mathrm{el}}}{\partial \delta}\,(1 - \alpha_{\mathrm{HC}} v_n), \qquad
c_{n,\mathrm{tan}} = \alpha_{\mathrm{HC}} F_{\mathrm{el}} \ge 0.
$$

#### 2.2.2 Anisotropic Coulomb–Stribeck, Ratchet Backlash, Rut Bulldozing & Patch Pivoting（各向异性库仑–Stribeck 摩擦、棘轮齿隙、车辙推土与面片枢转）

切向采用刷毛粘滑弹簧–阻尼记忆模型（$k_s = 2000\text{ N/m}, c_s = 8\text{ N}\cdot\text{s/m}, c_v = 2\times 10^{-6}\text{ N}\cdot\text{m}\cdot\text{s/rad}$），令有效增益 $g_s = k_s h + c_s$、基力 $\mathbf{b} = -k_s \mathbf{m} - g_s \mathbf{v}_t$、滚阻力矩阈值 $\tau_{\mathrm{rr}} = \mu_{\mathrm{rr}} F_n r_w$。

- **棘轮齿隙约束（Ratchet Pawl Backlash）**：维护每轮内部空程角状态 $\psi \in [0, \psi_{\max}]$（`ratchet_play`，名义 $\psi_{\max} = 10^\circ$），允许轮子在空程内微幅倒转，最小角速度下界为 $\omega_{\min} = -\psi / h$（$\psi_{\max}=0$ 时退化为理想单向离合 $\omega_{\min}=0$）。轮角速度与试探切向力联合闭式求解：

$$
\omega_{\mathrm{unc}} = \frac{\mathrm{sgn}(n_{\mathrm{drive}})\max(|n_{\mathrm{drive}}| - \tau_{\mathrm{rr}}, 0)}{I_w/h + r_w^2 g_s + c_v}, \qquad
n_{\mathrm{drive}} = \frac{I_w}{h}\omega^k - r_w b_0, \qquad
\omega^* = \max(\omega_{\mathrm{unc}}, \omega_{\min}).
$$

- **Stribeck 静动摩擦椭圆投影**：基于当前滑移速度估计 $v_{\mathrm{slip}} = \sqrt{(v_0 - r_w \omega^*)^2 + v_1^2}$，定义 Stribeck 静摩擦放大因子与各向异性摩擦上限：

$$
s(v_{\mathrm{slip}}) = 1 + \max\!\left(\frac{\mu_s}{\mu_k} - 1,\, 0\right)\exp\!\left(-\left(\frac{v_{\mathrm{slip}}}{0.03}\right)^2\right), \qquad
F_{\mathrm{cap},0} = \mu_{\mathrm{fwd}} s F_n, \quad F_{\mathrm{cap},1} = \mu_{\mathrm{lat}} s F_n,
$$

其中 $\mu_{\mathrm{lat}} = \rho_{\mathrm{lat}} \mu_{\mathrm{fwd}}$。将试探切向力 $\tilde{\mathbf{f}} = (b_0 + g_s r_w \omega^*,\, b_1)$ 投影至各向异性摩擦椭圆 $\ell = \sqrt{(\tilde{f}_0 / F_{\mathrm{cap},0})^2 + (\tilde{f}_1 / F_{\mathrm{cap},1})^2} \le 1$，缩放因子 $\eta = \min(1, \ell^{-1})$，随后回代饱和后轮速 $\omega^{k+1}$、轮轴反力矩 $\tau_{\mathrm{axle}} = \frac{I_w}{h}(\omega^{k+1} - \omega^k) + r_w f_0$ 及空程角 $\psi^{k+1} = \mathrm{clip}(\psi^k + \omega^{k+1} h,\, 0,\, \psi_{\max})$。

- **侧向车辙推土阻力（Lateral Rut Bulldozing）** 与 **圆形橡胶面片垂直枢转力矩（Contact-Patch Pivoting Torque）**：

$$
f_{\mathrm{rut,lat}} = -k_{\mathrm{rut,lat}}\, r_{\mathrm{new}}\, \tanh\!\left(\frac{v_1}{0.02}\right), \qquad
\tau_z = -\frac{2}{3}\mu_{\mathrm{piv}}\, r_{\mathrm{patch}}\, F_n\, \tanh\!\left(\frac{\omega_{z,p}}{0.15}\right).
$$

- **Contact-Schur 隐式刚度矩阵装配**：记接触点沿三正交方向 $\hat{\mathbf{u}}_0 = \hat{\mathbf{e}}_{\mathrm{fwd}}, \hat{\mathbf{u}}_1 = \hat{\mathbf{e}}_{\mathrm{lat}}, \hat{\mathbf{u}}_2 = \hat{\mathbf{n}}_g$ 的空间雅可比行 $\mathbf{j}_m = [\hat{\mathbf{u}}_m^\top,\; (\mathbf{r}_{\mathrm{arm}} \times \hat{\mathbf{u}}_m)^\top]^\top \in \mathbb{R}^6$，将对应显式阻尼在右端项中补偿后，向隔板 $6\times 6$ 左端矩阵叠加严格半正定的隐式接触刚度矩阵：

$$
\mathbf{K}_{\mathrm{contact},p} = \sum_{w \in \{L, R\}} \left[
h c_{0,w}\,\mathbf{j}_{0,w}\mathbf{j}_{0,w}^\top
+ h c_{1,w}\,\mathbf{j}_{1,w}\mathbf{j}_{1,w}^\top
+ (h c_{n,w} + h^2 k_{n,w})\,\mathbf{j}_{2,w}\mathbf{j}_{2,w}^\top
+ h c_{\mathrm{piv},w}\,\mathbf{e}_6\mathbf{e}_6^\top
\right] \succeq 0.
$$

| 地形预设 (`HardwareConfig.with_terrain`) | `contact_model` | $k_{\mathrm{HC}}$ ($\text{N/m}^p$) | $\alpha_{\mathrm{HC}}$ ($\text{s/m}$) | 幂指数 $p$ | 卸载倍比 $\gamma_u$ | 侧向推土 $k_{\mathrm{rut,lat}}$ (N/m) | $\mu_{\mathrm{fwd}}$ | $\mu_{\mathrm{lat}}/\mu_{\mathrm{fwd}}$ | 滚阻 $\mu_{\mathrm{rr}}$ | 面片半径 $r_{\mathrm{patch}}$ (mm) | 枢转 $\mu_{\mathrm{piv}}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧线性 (`kelvin`) | `kelvin` | ($k_n=5000$ N/m) | ($c_n=8.0$ N·s/m) | 1.0 | 1.0 | 0.0 | 0.80 | 1.00 | 0.060 | 0.0 | — |
| ARPC 硬地 (`hard`) | `arpc_hard` | $1.6 \times 10^5$ | 3.5 | 1.50 | 1.0 | 0.0 | 0.80 | 1.05 | 0.060* | 8.0 | 0.65 |
| ARPC 地毯 (`carpet`) | `arpc_carpet` | $1.8 \times 10^4$ | 5.0 | 1.40 | 1.0 | 0.0 | 0.90 | 1.20 | 0.140 | 11.0 | 0.75 |
| ARPC 软土 (`soft_soil`) | `arpc_soft_soil` | $1.2 \times 10^4$ | 6.0 | 1.35 | 3.5 | 650.0 | 0.70 | 1.35 | 0.220 | 13.0 | 0.80 |

> *注：在 `v5_realistic` 与 `v7` 中硬地名义滚阻为 $\mu_{\mathrm{rr}} = 0.060$（匹配 3D 打印机械棘轮实测），在早期 `v6` 训练配置中硬地名义滚阻为 $0.025$（见 `TERRAIN_PROFILES['hard']`）。*

---

### 2.3 Eccentric Crank-Arm Tendon & Passive Capstan Friction Hysteresis（偏心曲柄拉绳与导孔 Capstan 摩擦滞回）

每个体节后隔板左右各装有 1 颗 $25\text{ kg}\cdot\text{cm}$（$2.45\text{ N}\cdot\text{m}$）总线舵机，单舵机驱动上下 2 根编织 PE 拉绳（整机 10 颗收绳舵机、20 根拉绳）。经双激光伸缩偏转实验（Exp 126–128）标定，偏心舵盘半径 $r_h = 25.0\text{ mm}$、导孔径向距 $g = 36.0\text{ mm}$、初始系绳松弛死区 $s_0 = 11.4\text{ mm}$、拉绳刚度 $k_t = 8000\text{ N/m}$、阻尼 $c_t = 2.5\text{ N}\cdot\text{s/m}$。在舵机角 $\theta \in [0, \pi]$ 处，曲柄弦长、等效收绳量与力臂为：

$$
c(\theta) = \sqrt{g^2 + r_h^2 - 2 g r_h \cos\theta}, \qquad
\Delta L(\theta) = c(\theta) - (g - r_h), \qquad
l(\theta) = \min\!\left(r_h,\;\frac{r_h g |\sin\theta|}{c(\theta)}\right).
$$

引入导孔包角 **Capstan 摩擦乘积** $\eta_c = \mu_c \theta_{\mathrm{wrap}} \in [0, 0.6]$（名义值 $0.18$，DR 区间 $[0.04, 0.36]$）及舵机角度死区 $\delta_{\mathrm{db}}$（名义 $0.5^\circ$）：
1. **电机端负载放大与力矩-转速下垂**：收绳方向受导孔摩擦放大负载 $\tau_{\mathrm{load}} = \left(\sum_{j=1}^2 T_j\right) l(\theta)\, e^{\eta_c}$，有效位置误差 $\tilde{e}_\theta = \mathrm{sgn}(\theta_{\mathrm{tgt}} - \theta)\max(|\theta_{\mathrm{tgt}} - \theta| - \delta_{\mathrm{db}}, 0)$，单步角增量为：

$$
\Delta\theta = \mathrm{clip}(\tilde{e}_\theta,\, -\omega_{\max} h,\, \omega_{\max} h) \cdot
\begin{cases}
\mathrm{clip}\!\left(1 - \dfrac{\tau_{\mathrm{load}}}{\tau_{\max}},\, 0,\, 1\right), & \tilde{e}_\theta > 0, \\
1, & \tilde{e}_\theta \le 0.
\end{cases}
$$

2. **拉绳有效张力与方向滞回衰减**：记锚点–导孔实时弦长 $L_i$、名义自由长 $L_{\mathrm{rest},i} = (0.101 + s_0^{(b)}) - \Delta L(\theta)$、伸长量 $e_i = L_i - L_{\mathrm{rest},i}$ 及伸长速率 $\dot{L}_i$，则考虑导孔耗散阻尼 $c_{\mathrm{eff}} = c_t + 12\eta_c$ 与主动拉拽 Capstan 衰减后的单根拉绳张力为：

$$
T_i = \mathbb{I}(e_i \ge 0)\,\max\!\left(0,\; k_t e_i + c_{\mathrm{eff}} \dot{L}_i\right)\cdot \exp\!\left(-\eta_c \tanh\!\left(\frac{\max(\dot{\theta}, 0)}{0.35}\right)\right).
$$

相邻体节间的 4 个偏航关节由同型号 $2.45\text{ N}\cdot\text{m}$ 舵机经 PD 控制（$K_p = 10.0\text{ N}\cdot\text{m/rad}, K_d = 0.12\text{ N}\cdot\text{m}\cdot\text{s/rad}$）驱动，并显式计入**齿轮箱库仑起动静摩擦** $\tau_{\mathrm{sf}}$（名义 $0.06\text{ N}\cdot\text{m}$，DR $[0, 0.18]\text{ N}\cdot\text{m}$）及直流电机反电动势线性限矩包络：

$$
\tau_{\mathrm{yaw}} = \mathrm{clip}\!\left(\mathrm{sgn}(\tau_{\mathrm{raw}})\max(|\tau_{\mathrm{raw}}| - \tau_{\mathrm{sf}},\, 0),\; -\tau_{\mathrm{cap}}(\dot{\phi}),\; \tau_{\mathrm{cap}}(\dot{\phi})\right), \quad
\tau_{\mathrm{cap}}(\dot{\phi}) = \tau_{\lim}\cdot\mathrm{clip}\!\left(1 - \frac{|\dot{\phi}|}{\omega_{\lim}}, 0, 1\right)\;\;(\text{当 }\tau_{\mathrm{yaw}}\dot{\phi} > 0).
$$

---

### 2.4 Curved/Sloped Ground Geometry & Analytical Pipe Contact（非平整倾斜地面与解析管壁接触）

#### 2.4.1 Per-Environment Sloped & Undulating Terrain $z_g(x, y)$

为在无重编译 CUDA Graph 前提下覆盖真实地面的坡度与微起伏缝隙，每个并行环境 $b$ 持有 5 维地形几何向量 $\mathbf{g}^{(b)} = (s_x, s_y, A, k_x, k_y)$，定义连续光滑地表高程及其精确单位法向：

$$
z_g(x, y) = s_x x + s_y y + A \sin(k_x x)\cos(k_y y), \qquad
\hat{\mathbf{n}}_g(x, y) = \frac{(-\partial_x z_g,\; -\partial_y z_g,\; 1)^\top}{\sqrt{1 + (\partial_x z_g)^2 + (\partial_y z_g)^2}},
$$

其中 $\partial_x z_g = s_x + A k_x \cos(k_x x)\cos(k_y y)$，$\partial_y z_g = s_y - A k_y \sin(k_x x)\sin(k_y y)$。重置时通过刚体无应力姿态旋转 `align_pose_to_ground` 使整机贴合初始坡面。

#### 2.4.2 Analytical Confined Pipe Contact (`pipe_geometry.py`)

对受限管道任务，采用 12 维解析管道描述符 `pipe_cfg[B, 12]` 统一表征 4 类透明亚克力管/槽道几何（直圆管、$90^\circ$ 圆弧弯管、$C^1$ 连续双弧 S 弯管及带 $8\text{ mm}$ 内倒角的锐角 $90^\circ$ 方形 L 槽道，入口处均带有 $80\text{ mm}$ 长度、$55\text{ mm}$ 喇叭口平滑导入过渡）。对任意接触节点（10 块隔板半径 $r_p = 74\text{ mm}$，或全部 600 个钢片内部节点半径 $r_s = 2\text{ mm}$），由解析中心线投影求得穿透深度 $\delta_w$ 与指向管腔内部的单位法向 $\hat{\mathbf{n}}_w$，计算法向力与正则化库仑滑移摩擦（$c_w = 18.0\text{ N}\cdot\text{s/m}, \mu_w = 0.35, v_{\mathrm{reg}} = 0.015\text{ m/s}$）：

$$
F_{n,w} = \max(0,\; k_w \delta_w - c_w (\mathbf{v} \cdot \hat{\mathbf{n}}_w)), \qquad
\mathbf{f}_{t,w} = -\mu_w F_{n,w} \frac{\tanh(\|\mathbf{v}_t\| / v_{\mathrm{reg}})}{\|\mathbf{v}_t\|}\,\mathbf{v}_t,
$$

并向隐式矩阵累加 SPD $3\times 3$ 切线刚度 $(h c_w + h^2 k_w)\hat{\mathbf{n}}_w\hat{\mathbf{n}}_w^\top + h c_{t,w}(\mathbf{I}_3 - \hat{\mathbf{n}}_w\hat{\mathbf{n}}_w^\top)$。

---

## 3. Mode-Free Asymmetric Actor–Critic, Proprioceptive `EstNet`, and 5-Axis Domain Randomization（无模式非对称强化学习、纯本体 `EstNet` 与五轴域随机化）

### 3.1 Pure-Proprioceptive Observation Space & Hardware Deployability（纯本体观测空间与零外接传感器部署）

策略以 $f_{\pi} = 10\text{ Hz}$（$\Delta t_\pi = 0.10\text{ s}$）运行，底层舵机与 IMU 以 $50\text{ Hz}$（$\Delta t_s = 0.02\text{ s}$）采样，物理求解器以 $1\text{ kHz}$（$\Delta t = 0.001\text{ s}$）推进（即每策略步含 5 个传感步、100 个 FP64 物理子步）。**Actor 观测严格剔除所有全局位置、机体速度、线加速度计、轮地滑移、管壁接触力、地形标签与步态模式 ID**，保留 $H = 16$ 帧（$1.6\text{ s}$ 因果时序窗口，覆盖完整蛇行与蠕动周期）：

- **51D 纯编码器+IMU 观测（`v6a_51d_pure`, `include_q_err=False`）**：
  $$\mathbf{o}_t^{(51)} = \Big[\underbrace{\tilde{\mathbf{q}}_t}_{14\text{D}},\; \underbrace{\tilde{\dot{\mathbf{q}}}_t}_{14\text{D}},\; \underbrace{\boldsymbol{\omega}_{\mathrm{imu},t}}_{3\text{D}},\; \underbrace{\mathbf{g}_{\mathrm{proj},t}}_{3\text{D}},\; \underbrace{\mathbf{u}_{t-1}}_{14\text{D}},\; \underbrace{\tilde{\mathbf{c}}_t}_{3\text{D}}\Big] \in \mathbb{R}^{51},$$
  其中 $\tilde{\mathbf{q}}_t$ 为归一化 10 路收绳角 + 4 路偏航角，$\tilde{\dot{\mathbf{q}}}_t$ 为一阶 EMA 滤波角速度（平滑系数 $0.35$，归一化除数 $3.0\text{ rad/s}$），$\boldsymbol{\omega}_{\mathrm{imu},t}$ 与 $\mathbf{g}_{\mathrm{proj},t}$ 来自中段隔板单颗 6 轴 IMU，$\mathbf{u}_{t-1} \in [-1,1]^{14}$ 为上一时刻物理动作状态，$\tilde{\mathbf{c}}_t = [v_{x,\mathrm{cmd}}/v_{x,\max},\; \omega_{z,\mathrm{cmd}}/\omega_{z,\max},\; \mathbb{I}_{\mathrm{stop}}]$ 为任务指令。
- **65D 纯软件伺服跟踪误差增强观测（`v6b_65d_qerr` / `v6_1_65d_elec`, `include_q_err=True`）**：无需新增任何物理传感器，在软件端直接拼入指令与编码器之差：
  $$\mathbf{q}_{\mathrm{err},t} = \mathrm{clip}\!\left(\frac{\mathbf{q}_{\mathrm{tgt},t} - \mathbf{q}_{\mathrm{meas},t}}{\mathbf{s}_q},\; -2,\; 2\right) \in \mathbb{R}^{14}, \qquad \mathbf{o}_t^{(65)} \in \mathbb{R}^{65}.$$
  当机器人进入窄管受壁面阻碍或驶入松软沙土遭遇高滚动阻尼时，受载舵机因力矩–转速下垂特性产生的微小跟随滞后 $\mathbf{q}_{\mathrm{err},t}$ 成为零成本的**隐式本体力矩/接触感知通道**。

### 3.2 Asymmetric Privileged Critic & Causal TCN + `EstNet` Architecture（非对称特权 Critic 与因果 TCN + `EstNet` 架构）

网络采用 `ProprioceptiveEstimatorActorCritic`（[`src/serpworm/rl/models.py`](../../src/serpworm/rl/models.py)）：
1. **共享在线 Welford 归一化与三路独立因果 1D-TCN 编码器**：每路 TCN 由两层左填充因果卷积 `_CausalConv1d` 组成（通道 $D_{\mathrm{frame}} \to 64 \to 48$，卷积核 $k = 3$、步幅 $s = 2$、激活 $\mathrm{SiLU}$），将 $H = 16$ 帧压缩为严格保序的 $T_{\mathrm{out}} = 4$ 个时间槽（$48 \times 4 = 192\text{D}$），并与最新帧 $\mathbf{o}_t$ 跳跃拼接形成 $192 + D_{\mathrm{frame}}$ 维时序表征 $\mathbf{h}_t$。
2. **纯本体状态与接触估计器 `EstNet`（实机在线部署）**：通过 3 层 MLP（$192 \to 128 \to 7$，带 `LayerNorm` 与 `SiLU`）仅从本体历史序列 $\mathbf{o}_{t-15:t}$ 实时预测 7 维物理状态：
   $$\hat{\mathbf{e}}_t = \Big[\underbrace{\hat{v}_x,\; \hat{v}_y,\; \hat{\omega}_z}_{\text{Linear (3D)}},\;\; \underbrace{\sigma(\hat{z}_{\mathrm{wall}})}_{\text{wall\_conf}\in[0,1]},\;\; \underbrace{\tanh(\hat{z}_{\mathrm{side}})}_{\text{wall\_side}\in[-1,1]},\;\; \underbrace{\sigma(\hat{z}_{\mathrm{drag}})}_{\text{terrain\_drag}\in[0,1]},\;\; \underbrace{\sigma(\hat{z}_{\mathrm{slip}})}_{\text{slip}\in[0,1]}\Big] \in \mathbb{R}^7.$$
3. **无模式 14D Actor（梯度截断耦合）**：将 Actor-TCN 特征与**截断梯度（`stop_grad`）**的估计输出 $\mathrm{sg}(\hat{\mathbf{e}}_t)$ 拼接，经 `256 -> 256 -> 14`（`LayerNorm + SiLU`）输出有界均值 $\boldsymbol{\mu}_t = 4.0\tanh(\tilde{\boldsymbol{\mu}}_t / 4.0)$，配合可学习对角标准差 $\boldsymbol{\sigma}$ 与存储潜变量 $\tanh$-squashed 高斯分布（在 FP32 中精确计算雅可比对数行列式，防止 bf16 混精度 KL 跳变）。采用增量动作积分映射：
   $$\mathbf{u}_t = \mathrm{clip}\!\left(\mathbf{u}_{t-1} + 0.50\,\mathbf{a}_t,\; -1,\; 1\right), \qquad \mathbf{a}_t = \tanh(\mathbf{z}_t) \in [-1, 1]^{14},$$
   同步开放全部 10 路独立拉绳舵机与 4 路节间偏航舵机，彻底消除离散模式切换与 10 Hz 棘轮抖动。
4. **非对称特权 Critic（仅训练期使用）**：接收 Critic-TCN 特征与 $102\text{D}$（v6）/$108\text{D}$（v7 含扩展 DR 物理参数）仿真真值特权向量 $\mathbf{s}_{\mathrm{priv},t}$（涵盖实时/短窗/长窗机体速度与速度误差、10 块隔板离地高与直立度、5 段间距、20 根拉绳张力、20 轮支撑力与滑移、最小轮隙、偏航力矩 RMS、管壁峰值/均值/有符号接触力、管腔居中偏移、滚动阻力与推土刚度、棘轮齿隙、Capstan 摩擦、偏航静摩擦、瑞利阻尼 $\beta_R$ 与坡度 $s_x, s_y$），经 `512 -> 256 -> 128 -> 1` 估计状态价值 $V(\mathbf{o}_{t-15:t}, \mathbf{s}_{\mathrm{priv},t})$。

### 3.3 Pure-Physics Reward, Electrical Power, and Joint PPO + EstNet + Symmetry Loss（纯物理奖励、电热功耗与联合优化目标）

奖励函数 `autonomous_phase_reward`（[`src/serpworm/envs/rewards.py`](../../src/serpworm/envs/rewards.py)）不含任何步态模板或模式引导项，仅由物理目标驱动：

$$
r_t = \Delta t_\pi \Big(
r_{v_x} + r_{\omega_z} + r_{v_y} + r_{\mathrm{wall}} + r_{\mathrm{slip}} + r_{\mathrm{posture}} + r_{\mathrm{barrier}} + r_{\mathrm{shock}} + r_{\mathrm{edge}} + r_{\mathrm{sym}} + r_{\mathrm{reg}} + r_{\mathrm{alive}}
\Big),
$$

其中关键物理项包括：
- **管壁非对称二次惩罚**（允许 $0.4\text{--}1.2\text{ N}$ 轻触试探以感知出管，重罚 $>2.5\text{ N}$ 撞壁）：$r_{\mathrm{wall}} = -\left[0.75\left(\frac{F_{w,\max}}{2.5}\right)^2 + 0.12\frac{\bar{F}_w}{2.5}\right]$；
- **轮间防碰撞软屏障**（严格守护 $d_{\min} = 2.0\text{ mm}$ 硬限位、软阈值 $d_{\mathrm{soft}} = 4.2\text{ mm}$）：$r_{\mathrm{barrier}} = -0.45\left[\mathrm{clip}\!\left(\frac{d_{\mathrm{soft}} - d_{\mathrm{clr}}}{d_{\mathrm{soft}} - d_{\min}}, 0, 1.5\right)\right]^2$；
- **真实直流总线舵机电热功耗（`v6.1` 强化能效分离）**：除偏航与拉绳机械做功外，显式计入 7.4 V / 3 A 堵转特性对应的铜损焦耳热 $P_{\mathrm{Joule}} = 22.0\left[\sum_{j=1}^4 (\tau_{\mathrm{yaw},j}/\tau_{\lim})^2 + \sum_{k=1}^{10}(\tau_{\mathrm{sv},k}/\tau_{\lim})^2\right]\text{ W}$，使策略在开阔硬地与狭小管道间自然形成能耗最优的步态分离。

训练采用 2× 左右镜像数据增强（`mirror_autonomous_batch`，沿 $y \to -y$ 精确反射左右拉绳、偏航角、IMU 陀螺与重力、管壁侧向与动作潜变量）及联合损失函数（[`src/serpworm/rl/train_autonomous.py`](../../src/serpworm/rl/train_autonomous.py)）：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{PPO}}^{\mathrm{clip}} + c_v \mathcal{L}_V^{\mathrm{clip}} + c_{\mathrm{est}}\,\mathrm{SmoothL1}_{\beta=0.15}(\hat{\mathbf{e}}_t, \mathbf{e}_t^*) - c_{\mathrm{ent}}\mathcal{H}(\pi) + 0.25\,\mathcal{L}_{\mathrm{sym}},
$$

其中 $c_v = 0.5, c_{\mathrm{est}} = 1.0, \gamma = 0.994, \lambda_{\mathrm{GAE}} = 0.95, \epsilon_{\mathrm{clip}} = 0.20$，KL 自适应学习率目标 $\mathrm{KL}_{\mathrm{tgt}} = 0.015$，对称一致性正则 $\mathcal{L}_{\mathrm{sym}} = \|\boldsymbol{\mu}' - \mathcal{M}_a(\boldsymbol{\mu})\|_2^2 + 0.25(V' - V)^2 + 0.25\|\hat{\mathbf{e}}' - \mathcal{M}_e(\hat{\mathbf{e}})\|_2^2$。三阶段自动课程在 1024 并行环境、$H=48$ 步、200 轮（9.83M 真实交互步，镜像增强后 19.66M 样本）上依次开启直行/急停（Stage 1: r1–29）、双向转向（Stage 2: r30–114）与 20 轮线性渐进域随机化（Stage 3: r115–200）。

### 3.4 Five-Axis Sim-to-Real Domain Randomization（五轴 Sim-to-Real 域随机化参数表）

为消除实机装配离散性、多地面摩擦变化、拉绳摩擦与舵机供电跌落引起的 Sim-to-Real 鸿沟，[`DomainRandConfig`](../../src/serpworm/config.py) 定义了完整的 **5 轴 26 项**逐回合/逐控制步域随机化参数（Stage 3 前 20 轮由名义值线性插值渐进加载）：

| 随机化维度（Axis） | 参数名称（`DomainRandConfig` 字段） | 物理含义 | 名义值（`v5_realistic`） | 随机化范围 / 分布 | 作用周期 |
|---|---|---|---|---|---|
| **Axis 1：地面与多地形接触** | `wheel_mu_range` | 轮地滑动摩擦系数 $\mu_{\mathrm{fwd}}$ | $0.80$ | $[0.42,\, 1.15]$ | Per-episode |
| | `wheel_rolling_range` | 滚动阻力系数 $\mu_{\mathrm{rr}}$ | $0.060$ | $[0.025,\, 0.140]$ | Per-episode |
| | `wheel_mu_lat_ratio_range` | 侧/纵向摩擦各向异性比 $\mu_{\mathrm{lat}}/\mu_{\mathrm{fwd}}$ | $1.05$ | $[0.85,\, 1.35]$ | Per-episode |
| | `wheel_static_mu_ratio_range` | Stribeck 静/动摩擦比 $\mu_s/\mu_k$ | $1.18$ | $[1.00,\, 1.35]$ | Per-episode |
| | `ground_hc_k_scale_range` | Hunt–Crossley 法向刚度缩放倍比 | $1.0\times$ | $[0.15,\, 1.50]\times$ | Per-episode |
| | `wheel_patch_radius_range` | 橡胶接触面片半径 $r_{\mathrm{patch}}$ | $8.0\text{ mm}$ | $[4.0,\, 13.0]\text{ mm}$ | Per-episode |
| | `ratchet_backlash_deg_range` | 单向棘轮齿隙死区 $\psi_{\max}$ | $10.0^\circ$ | $[4.0^\circ,\, 18.0^\circ]$ | Per-episode |
| | `mixed_soil_prob` / `unload` / `rut_lat_k` | 硬地回合混入弹塑性软土比例与车辙参数 | $0$ / $1.0$ / $0$ | $25\%$ / $\gamma_u\in[1.3, 3.0]$ / $[100, 550]\text{ N/m}$ | Per-episode |
| | `ground_slope_deg_max` | 平面地面 X/Y 坡度 | $0^\circ$ | $[-3.5^\circ,\, +3.5^\circ]$ | Per-episode |
| | `ground_wave_amp_max_m` / `k_range` | 地表微起伏幅值 $A$ 与空间波数 $k_x, k_y$ | $0\text{ mm}$ | $[0,\, 2.5]\text{ mm}$ / $[6.0,\, 14.0]\text{ rad/m}$ | Per-episode |
| **Axis 2：结构、阻尼与非对称质量** | `mass_scale_range` | 整机全局质量缩放 | $1.0\times$ ($2.76\text{ kg}$) | $[0.85,\, 1.15]\times$ | Per-episode |
| | `plate_mass_jitter` | 10 块隔板独立质量/质心扰动 | $0\%$ | $\pm 15\%$ | Per-episode |
| | `rayleigh_beta_range` | 钢片与夹持装配瑞利阻尼 $\beta_R$ | $0.0005\text{ s}$ | $[0.0002,\, 0.0018]\text{ s}$ | Per-episode |
| **Axis 3：拉绳传动与执行器** | `capstan_mu_theta_range` | 导孔 Capstan 摩擦乘积 $\mu_c \theta_{\mathrm{wrap}}$ | $0.18$ | $[0.04,\, 0.36]$ | Per-episode |
| | `tendon_slack_jitter_m` | 手工系绳初始松弛误差 | $11.4\text{ mm}$ | $\pm 3.0\text{ mm}$ | Per-episode |
| | `yaw_static_friction_range` | 偏航齿轮箱库仑起动静摩擦 $\tau_{\mathrm{sf}}$ | $0.06\text{ N}\cdot\text{m}$ | $[0.00,\, 0.18]\text{ N}\cdot\text{m}$ | Per-episode |
| | `servo_deadband_deg_range` | 总线舵机角度死区 $\delta_{\mathrm{db}}$ | $0.5^\circ$ | $[0.0^\circ,\, 1.8^\circ]$ | Per-episode |
| | `yaw_kp_scale_range` / `yaw_kd_scale_range` | 偏航伺服 PD 刚度/阻尼缩放 | $1.0\times$ / $1.0\times$ | $[0.75, 1.25]\times$ / $[0.70, 1.30]\times$ | Per-episode |
| | `torque_scale_range` / `speed_scale_range` | 14 舵机总线供电跌落（力矩/转速缩放） | $1.0\times$ / $1.0\times$ | $[0.72, 1.15]\times$ / $[0.72, 1.15]\times$ | Per-episode |
| | `filter_tau_scale_range` | 一阶伺服响应时间常数 $\tau$ 缩放 | $1.0\times$ ($60\text{ ms}$) | $[0.70,\, 1.45]\times$ | Per-episode |
| | `action_delay_prob` / `max_action_delay_ticks` | 总线指令随机通信延迟 | $0\text{ ms}$ | $p=35\%$ 触发 $20\text{--}40\text{ ms}$ (1–2 ticks) | Per-step |
| **Axis 4：传感器与 IMU 安装** | `obs_noise_scale` | 编码器/陀螺/重力高斯白噪声倍率 | $1.0\times$ | $[0.002\text{ rad},\, 0.01\text{ rad/s},\, 0.005]$ | Per-step |
| | `joint_bias_deg` | 14 路舵机零点标定偏置 | $0^\circ$ | $\pm 2.0^\circ$ | Per-episode |
| | `gyro_bias_rad_s` | MEMS 陀螺零偏漂移 | $0\text{ rad/s}$ | $\pm 0.04\text{ rad/s}$ | Per-episode |
| | `gravity_tilt_bias_deg` | IMU 安装滚转/俯仰倾角误差 | $0^\circ$ | $\pm 2.0^\circ$ | Per-episode |
| **Axis 5：外部冲击扰动** | `push_interval_s` / `push_prob` | 随机外推/线束拉拽检测周期与概率 | 关闭 | 每 $4.0\text{ s}$ 以 $50\%$ 概率施加 | Periodic |
| | `push_xy_vel_max` / `push_yaw_vel_max` | 刚体一致平面速度与整机偏航角速度脉冲 | $0$ | $\pm 0.06\text{ m/s}$ / $\pm 0.25\text{ rad/s}$ | Periodic |

---

## 4. Verified Reference Library Index（已校验 26 篇核心参考文献索引）

所有 26 篇 arXiv 开放获取论文均存档于 [`ref/`](../../../ref/README.md) 并通过完整性校验：

1. **Ge, J. Z. et al. (2017)**, *An Earthworm-Inspired Soft Crawling Robot Controlled by Friction*, `arXiv:1707.04084`.
2. **Gidoni, P. (2017)**, *Rate-independent soft crawlers*, `arXiv:1710.08340`.
3. **Xie, J. et al. (2018)**, *Rigid Body Dynamic Simulation with Multiple Convex Contact Patches*, `arXiv:1809.05555`.
4. **Liu, X. et al. (2021)**, *Learning Contact-aware CPG-based Locomotion in a Soft Snake Robot*, `arXiv:2105.04608`.
5. **Arachchige, D. D. K. et al. (2023)**, *Dynamic Modeling and Validation of Soft Robotic Snake Locomotion*, `arXiv:2303.02291`.
6. **Crotti, M. et al. (2024)**, *Soft Adaptive Feet for Legged Robots: An Open-Source Model for Locomotion Simulation*, `arXiv:2412.03191`.
7. **Salagame, A. et al. (2025)**, *Optimal Trajectory Planning in a Vertically Undulating Snake Locomotion using Contact-implicit Optimization*, `arXiv:2508.02953`.
8. **Javadi, M. et al. (2025)**, *An Amphibious Untethered Inchworm Soft Robot for Fast Crawling Locomotion*, `arXiv:2510.03660`.
9. **Hublikar, H. (2025)**, *Contact-Implicit Modeling and Simulation of a Snake Robot on Compliant and Granular Terrain*, `arXiv:2512.05008`.
10. **Zhang, X. et al. (2026)**, *Learning, locomotion, and navigation of soft synthetic snakes in three-dimensional, heterogeneous environments*, `arXiv:2605.24985`.
11. **Li, M. et al. (2020)**, *Codimensional Incremental Potential Contact*, `arXiv:2012.04457`.
12. **Du, T. et al. (2021)**, *DiffPD: Differentiable Projective Dynamics*, `arXiv:2101.05917`.
13. **Wang, X. et al. (2024)**, *Cosserat Rod Modeling and Validation for a Soft Continuum Robot with Self-Controllable Variable Curvature*, `arXiv:2402.12315`.
14. **Lahoti, R. et al. (2025)**, *Py-DiSMech: A Scalable and Efficient Framework for Discrete Differential Geometry-Based Modeling and Control of Soft Robots*, `arXiv:2512.09911`.
15. **Parthasarathy, T. et al. (2026)**, *Elastica++: A high-performance, multiphysics framework for large interacting assemblies of Cosserat rods*, `arXiv:2605.13766`.
16. **Kim, Y.-H. et al. (2021)**, *Shape-adaptive Hysteresis Compensation for Tendon-driven Continuum Manipulators*, `arXiv:2109.06907`.
17. **Cho, B. Y. et al. (2024)**, *Accounting for Hysteresis in the Forward Kinematics of Nonlinearly-Routed Tendon-Driven Continuum Robots via a Learned Deep Decoder Network*, `arXiv:2404.03816`.
18. **Zhang, G. et al. (2026)**, *Capstan-driven Continuum Surgical Robot: Design, Modeling, and Perception*, `arXiv:2608.13396`.
19. **Shen, J. et al. (2026)**, *Modeling the Frictional Contact of Continuum Robots with Environments through Linear Complementarity Problem*, `arXiv:2611.18546` (IJRR).
20. **Hwangbo, J. et al. (2019)**, *Learning agile and dynamic motor skills for legged robots*, `arXiv:1901.08652` (*Science Robotics*).
21. **Kumar, A. et al. (2021)**, *RMA: Rapid Motor Adaptation for Legged Robots*, `arXiv:2107.04034` (*RSS*).
22. **Nahrendra, I. M. A. et al. (2023)**, *DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning*, `arXiv:2301.10602` (*ICRA*).
23. **Long, J. et al. (2023)**, *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response*, `arXiv:2312.11460` (*ICLR 2024*).
24. **Mittal, M. et al. (2024)**, *Symmetry Considerations for Learning Task Symmetric Robot Policies*, `arXiv:2403.04359` (*ICRA*).
25. **Hu, X. et al. (2025)**, *Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning*, `arXiv:2503.01255`.
26. **Hou, L. et al. (2025)**, *A Quantitative Comparison of Centralised and Distributed Reinforcement Learning-Based Control for Soft Robotic Arms*, `arXiv:2511.02192`.

**Classical Journal References（经典期刊原典，非 arXiv）**：
- Hunt, K. H. & Crossley, F. R. E. (1975), *Coefficient of Restitution Interpreted as Damping in Vibroimpact*, ASME Journal of Applied Mechanics.
- Bekker, M. G. (1960) / Wong, J. Y. (2008), *Theory of Ground Vehicles*.
- Shen, J. et al. (IEEE RA-L), *Friction Modeling of Tendon-driven Continuum Robots through Linear Complementarity Problem*.
