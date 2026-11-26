# SerpWorm-RL: Deep Literature Synthesis, Architectural Gap Analysis, and v8 Upgrade Roadmap
# （顶会文献深度精读、现有架构差距诊断与 v8 升级技术路线图）

> 本文档基于对本地文献库 [`ref/`](../../../ref/README.md) 中 **8 篇最高杠杆顶会/顶刊论文全文 PDF**（PyMuPDF 逐页精读，提取原论文精确公式、网络结构与实测消融数据）与当前 SerpWorm-RL 源码（[`src/serpworm/rl/models.py`](../../src/serpworm/rl/models.py)、[`src/serpworm/rl/train_autonomous.py`](../../src/serpworm/rl/train_autonomous.py)、[`src/serpworm/envs/autonomous_env.py`](../../src/serpworm/envs/autonomous_env.py)、[`src/serpworm/physics/wheel_stiction.py`](../../src/serpworm/physics/wheel_stiction.py)、[`src/serpworm/physics/gpu_core.py`](../../src/serpworm/physics/gpu_core.py)、[`src/serpworm/physics/pipe_geometry.py`](../../src/serpworm/physics/pipe_geometry.py) 与 [`src/serpworm/config.py`](../../src/serpworm/config.py)）的逐项方法学对标撰写。所有 arXiv 编号、原论文公式及实验数据均严格核自本地原文与已验证实验产物。

---

## 1. Executive Summary（核心洞察总览）

SerpWorm-RL 当前 `v6/v7` 主线架构（`ProprioceptiveEstimatorActorCritic` + 40 钢片 FP64 隐式 `fused_cuda` + Implicit ARPC 接触 + 五轴域随机化）已实现零外接传感器条件下的“开阔硬地 $\to$ 236 mm 窄管 $\to$ 软土”自主相变（3-zone 总位移 $2346.8\text{ mm}$，较固定蛇行基线 `v5` 提升 $+117.0\%$，管壁均值力下降 $89.6\%$ 至 $0.49\text{ N}$）。然而，将当前实现与 2021–2026 年机器人强化学习与连续体接触力学 8 篇核心顶会/顶刊（ICLR 2024 / ICRA 2023–2024 / RSS 2021 / IJRR 2026）逐公式对比后，我们识别出 **4 个结构性算法瓶颈** 与 **4 个下一阶段物理建模高价值升级点**：

1. **纯显式监督 `EstNet` 缺失隐式环境潜变量 $\mathbf{z}_t / \mathbf{l}_t$**：RMA [`arXiv:2107.04034`]（Table II）、DreamWaQ [`arXiv:2301.10602`]（Fig. 3 & Table III）与 HIMLoco [`arXiv:2312.11460`]（Table 4）三篇独立工作一致证明——仅回归显式物理量（如速度/摩擦）在强域随机化（DR）与未见地形上存在**参数共线性不可辨识**与**表征瓶颈**，引入自监督隐式潜变量可将抗扰存活率由 $80.92\%$ 提升至 $95.23\%$（DreamWaQ），并将未见柔顺坡道成功率由 $0\%$ 提升至 $55\%$（HIMLoco）。
2. **缺失下一步本体重构与对比动力学对齐**：当前 `est_tcn` 与 `actor_tcn` 均无前向动力学监督。DreamWaQ 的 $\beta$-VAE 下一步本体观测重构可学习前后向动力学并提供后验不确定度 $\boldsymbol{\sigma}_z^2$，而 HIMLoco 的批内对比动力学对齐（SwAV / InfoNCE）则消除强 DR 下的回归噪声过拟合（表征塌缩）。**将二者统一为 v8 `HIMDreamWaQActorCritic`** 可兼得细粒度动力学重构与高可分跨地形簇边界。
3. **镜像增强 PPO 分母偏差与推理端非零不对称性**：Symmetry Considerations [`arXiv:2403.04359`]（ICRA 2024, Eq. 6 & Fig. 2）从理论上证明，镜像增强样本的 PPO 重要性比率分母必须保持原始 rollout 概率 $\pi_{\theta_k}(a_t|s_t)$，而非在镜像样本上重新求值（当前 `train_autonomous.py:346-354` 正处于这一高方差区间）；同时，仅靠软正则 `0.25 * sym_loss` 导致 `open_hard_140` 残留 $-0.0097$ 至 $+0.0066\text{ rad/s}$ 的直行偏航漂移。引入**精确 $\mathbb{Z}_2$ 群等变 Reynolds 推理封装（`GroupEquivariantPolicyWrapper`）**可在零重训条件下将左右镜像不对称误差严格归零（`0.0`），并需妥善处理论文 Sec. III 指出的中性态（$L_g[\mathbf{o}] = \mathbf{o}$）奇异性。
4. **连续体摩擦粘滑、薄带自碰撞与静摩擦比值量化**：
   - `2503.01255` 证明静摩擦扭矩比 $f/\tau_{\max} = 0.98\%$ 即可摧毁常规 DR 的 Sim-to-Real 迁移，而 SerpWorm 偏航舵机名义比值达 **$2.45\%$**（DR 上界 **$7.35\%$**），从控制理论上验证了我们以 `0.0` 为下界的“欺骗式（Deception Method）”宽幅静摩擦 DR 的必要性；
   - `1710.08340` 的率无关**停滞域（Stasis Domain）**理论 $\Delta y = \Delta L - 2\mu_+/k$ 从解析上解释了软土高滚阻导致 `v6a`（51D）速度骤降至 $12.1\text{ mm/s}$、而 `v6b`（65D $+\mathbf{q}_{\mathrm{err}}$）暴增至 $36.1\text{ mm/s}$ 的力学本质；
   - `2611.18546`（IJRR 2026）与 `2512.09911`（Py-DiSMech）为下一阶段物理引擎指明了**管壁/导孔切向粘滑记忆**与 **$C^2$ 软惩罚隐式钢片自碰撞**两条可无痛融入 `fused_cuda` 的精确升级路径。

---

## 2. Deep-Read Technical Synthesis of 8 Core Papers（8 篇核心论文精读与精确公式提取）

### 2.1 RL & Proprioceptive Adaptation Pillars（本体感知强化学习与对称性五大支柱）

#### (1) HIMLoco — Hybrid Internal Model (`arXiv:2312.11460`, Long et al., ICLR 2024)
- **核心思想（Internal Model Control 视角）**：受经典内模控制（IMC, Rivera et al. 1986）启发，将地形高程、摩擦、恢复系数等不可观测外部状态统一视为系统扰动，不再采用 RMA/Teacher-Student 的两阶段模仿学习（避免模仿信息损失与回归噪声塌缩），而是直接从本体历史 $\mathbf{o}_{t-H:t}^a$（$H=5$）估计**机器人仿真响应**——即混合内部嵌入 $(\hat{\mathbf{v}}_t, \hat{\mathbf{l}}_t)$，其中 $\hat{\mathbf{v}}_t \in \mathbb{R}^3$ 为显式机体速度，$\hat{\mathbf{l}}_t \in \mathcal{Z} \subset \mathbb{R}^{16}$ 为隐式稳定性/环境响应嵌入。
- **精确算法与损失函数（Sec. 3.3, Eq. 1–2）**：
  - 将历史观测 $\mathbf{o}_{t-H:t}^a$ 经源编码器映射并 $L_2$ 归一化至单位球面得 $\mathbf{l}_t^{\mathrm{source}}$，将下一时刻本体观测 $\mathbf{o}_{t+1}^a$ 经目标编码器并 $L_2$ 归一化得 $\mathbf{l}_t^{\mathrm{target}}$。
  - 引入 $K$ 个单位归一化原型向量矩阵 $\mathbf{E} = \{\bar{\mathbf{e}}_1, \dots, \bar{\mathbf{e}}_K\}$（默认 $K = 16$，Appendix B.3 消融显示随 $K$ 增至 $64$ 持续提升、超过 $64$ 性能下降），计算原型簇分配概率（Eq. 1）：
    $$
    p_t^{\mathrm{source}}(k) = \frac{\exp\!\left(\tfrac{1}{\tau}\, (\mathbf{l}_t^{\mathrm{source}})^\top \bar{\mathbf{e}}_k\right)}{\sum_{k'=1}^K \exp\!\left(\tfrac{1}{\tau}\, (\mathbf{l}_t^{\mathrm{source}})^\top \bar{\mathbf{e}}_{k'}\right)}, \qquad
    p_t^{\mathrm{target}}(k) = \frac{\exp\!\left(\tfrac{1}{\tau}\, (\mathbf{l}_t^{\mathrm{target}})^\top \bar{\mathbf{e}}_k\right)}{\sum_{k'=1}^K \exp\!\left(\tfrac{1}{\tau}\, (\mathbf{l}_t^{\mathrm{target}})^\top \bar{\mathbf{e}}_{k'}\right)}.
    $$
  - 为防止原型塌缩为平凡解，对批内特征执行 **Sinkhorn–Knopp 最优传输** 生成等分目标编码 $\mathbf{q}_t^{\mathrm{source}}, \mathbf{q}_t^{\mathrm{target}}$，最小化 SwAV 交换预测对比损失（Eq. 2），并与速度均方回归联合执行交替优化（HIO 冻结更新后再跑 PPO，Table 7 中对比损失权重 $1.0$、速度损失权重 $1.0$）：
    $$
    J_{\mathrm{SwAV}} = -\frac{1}{2H}\sum_{t=1}^H \left(\mathbf{q}_t^{\mathrm{source}\top}\log \mathbf{p}_t^{\mathrm{target}} + \mathbf{q}_t^{\mathrm{target}\top}\log \mathbf{p}_t^{\mathrm{source}}\right).
    $$
- **关键实测发现（Table 1, Fig. 4, Table 2 & Table 4）**：
  - **样本效率**：仅需 **200M 帧**（RTX 4090 上 1 小时），较 RMA 的 **1,280M 帧** 节省 $6.4\times$（Table 1）。
  - **潜空间可分性**：Fig. 4 的 t-SNE 明确显示，对比学习（HIO）在 Stairs / Rough Slope / Slope / Discrete 四类地形上形成清晰分离的簇，而纯回归方法（Nahrendra et al. 2023）在强 DR 噪声下出现明显簇混叠。
  - **真机消融铁证（Table 4，20 次实机试验）**：完整 HIM 在短楼梯（100%）、长楼梯（$176.5 \pm 7.81$ 级）、未见组合地形（85%）、未见可变形软坡（55%）与踏空台阶（100%）全面领先；**一旦移除隐式潜变量输入或损失（`w/o lat. inp.` / `w/o lat. loss`，即与我们当前纯 `EstNet` v6/v7 拓扑等价），可变形软坡成功率暴跌至 $0\%$、踏空恢复跌至 $0\text{--}5\%$、长楼梯跌至 $10.6\text{--}13.3$ 级**；纯回归方案在可变形软坡与踏空台阶也分别落后 $25\%$（30% vs 55%）与 $35\%$（65% vs 100%）。

#### (2) DreamWaQ — Implicit Terrain Imagination (`arXiv:2301.10602`, Nahrendra, Yu & Myung, ICRA 2023)
- **核心思想**：单阶段非对称 Actor–Critic（Critic 输入完整特权状态 $\mathbf{s}_t = [\mathbf{o}_t, \mathbf{v}_t, \mathbf{d}_t, \mathbf{h}_t]$，Actor 仅输入纯本体 $\mathbf{o}_t$ 与上下文辅助估计器 **CENet** 的输出 $(\tilde{\mathbf{v}}_t, \mathbf{z}_t)$），通过 $\beta$-VAE 下一步观测重构使编码器同时学到前向与后向动力学，实现**隐式地形想象（Implicit Terrain Imagination）**。
- **精确网络与损失公式（Sec. II-C, Fig. 2, Eq. 5–8 & Table I）**：
  - **CENet 共享编码器 + 多头解码器**：编码器 `128 -> 64 -> 19` 将 $H=5$ 帧历史 $\mathbf{o}_t^H$ 映射为 $3\text{D}$ 速度 $\tilde{\mathbf{v}}_t$ 与 $16\text{D}$ 高斯后验上下文 $\mathbf{z}_t \sim q(\mathbf{z}_t \mid \mathbf{o}_t^H)$；解码器 `64 -> 128 -> 48` 重构下一步本体观测 $\tilde{\mathbf{o}}_{t+1}$。联合损失（Eq. 5–7）：
    $$
    \mathcal{L}_{\mathrm{CE}} = \mathcal{L}_{\mathrm{est}} + \mathcal{L}_{\mathrm{VAE}}, \qquad
    \mathcal{L}_{\mathrm{est}} = \mathrm{MSE}(\tilde{\mathbf{v}}_t, \mathbf{v}_t),
    $$
    $$
    \mathcal{L}_{\mathrm{VAE}} = \mathrm{MSE}(\tilde{\mathbf{o}}_{t+1}, \mathbf{o}_{t+1}) + \beta\, D_{\mathrm{KL}}\!\left(q(\mathbf{z}_t \mid \mathbf{o}_t^H) \,\|\, p(\mathbf{z}_t)\right), \quad p(\mathbf{z}_t) = \mathcal{N}(\mathbf{0}, \mathbf{I}).
    $$
  - **自适应自举机制 AdaBoot（Eq. 8）**：早期训练估计器噪声大，直接自举会破坏策略收敛；后期需逐步切换至估计值以增强实机抗噪性。根据 $m$ 个 DR 环境回合奖励的变异系数 $\mathrm{CV}(R) = \sigma(R)/\mu(R)$ 动态调节自举概率：
    $$
    p_{\mathrm{boot}} = 1 - \tanh\!\left(\mathrm{CV}(R)\right) \in [0, 1].
    $$
  - **电机功率均衡奖励（Table I）**：除总功耗 $|\boldsymbol{\tau}||\dot{\boldsymbol{\theta}}|$ 外，新增功率方差惩罚 $-10^{-5}\mathrm{Var}(\boldsymbol{\tau} \cdot \dot{\boldsymbol{\theta}})^2$，防止长距离爬坡中个别舵机提前进入过热保护——对 SerpWorm 的 14 路串口总线舵机连续穿管具有直接工程价值。
- **关键实测发现（Fig. 3, Fig. 5, Table III）**：
  - Fig. 3 显示纯 `EstimatorNet`（无上下文潜变量）在早期上升快，但课程进入困难地形后性能骤降；而 DreamWaQ 逼近持有真值高程图的 Oracle 策略。
  - Table III 抗推挤基准：`Baseline`（20.51% 存活 / 0.511 m/s 最大推力）$\to$ `EstimatorNet`（80.92% / 0.871 m/s）$\to$ `AdaptationNet/RMA`（82.37% / 0.714 m/s）$\to$ `DreamWaQ w/o AdaBoot`（90.71% / 1.015 m/s）$\to$ **`DreamWaQ w/ AdaBoot`（95.23% / 1.121 m/s）**，且在室外 430 m 与 465 m（高差 22 m）复杂地形零样本实机走通。

#### (3) Symmetry Considerations for Task-Symmetric Policies (`arXiv:2403.04359`, Mittal, Rudin, Klemm, Allshire & Hutter, ICRA 2024)
- **核心思想**：严格区分**运动对称性（motion symmetry）**与**任务对称性（task symmetry）**。对于群对称 MDP $\mathcal{G} = \{g_0, \dots, g_{N-1}\}$（状态变换 $L_g: \mathcal{S}\to\mathcal{S}$、动作变换 $K_g: \mathcal{A}\to\mathcal{A}$），系统对比了镜像损失（Mirror Loss）、对称数据增强（Data Augmentation）与等变网络结构（Equivariant Architectures）三类范式。
- **核心数学公式（Sec. III, Eq. 2–6）**：
  - **镜像损失（Eq. 2–3）**：
    $$
    \mathcal{L}(\theta) = \mathcal{L}_{\mathrm{PPO}}(\theta) + w \sum_{g\in\mathcal{G}} \mathbb{E}_{\tau}\!\left[\sum_{t=0}^\infty \left\|K_g[\pi_\theta(\mathbf{s}_t)] - \pi_\theta(L_g[\mathbf{s}_t])\right\|_2^2\right],
    $$
    论文指出：实现中 $K_g[\pi_\theta(\mathbf{s}_t)]$ 应视为截断梯度的目标标签，且 $w$ 过大会迫使策略将低回报方向的动作反向拉向高回报方向，与 PPO 策略梯度竞争（Sec. IV-C）。
  - **On-Policy 对称数据增强的正确重要性比率（Eq. 4–6）**：定义虚拟镜像策略 $\pi_{\theta_k}^g(K_g[\mathbf{a}] \mid L_g[\mathbf{s}]) = \pi_{\theta_k}(\mathbf{a} \mid \mathbf{s})$，由对称 MDP 性质有 $A^{\pi_{\theta_k}^g}(L_g[\mathbf{s}], K_g[\mathbf{a}]) = A^{\pi_{\theta_k}}(\mathbf{s}, \mathbf{a})$ 与 $p^{\pi_{\theta_k}^g}(L_g[\mathbf{s}]) = p^{\pi_{\theta_k}}(\mathbf{s})$，由此导出精确梯度（Eq. 6）：
    $$
    \nabla_\theta J(\pi_\theta) = \sum_{g\in\mathcal{G}} \mathbb{E}_{\tau \sim p_{\pi_{\theta_k}}}\!\left[\sum_{t=0}^\infty \frac{p^{\pi_\theta}(L_g[\mathbf{s}_t])}{p^{\pi_{\theta_k}}(\mathbf{s}_t)} \cdot \frac{\pi_\theta(K_g[\mathbf{a}_t] \mid L_g[\mathbf{s}_t])}{\mathbf{\pi_{\theta_k}(\mathbf{a}_t \mid \mathbf{s}_t)}} \, A^{\pi_{\theta_k}}(\mathbf{s}_t, \mathbf{a}_t)\,\nabla_\theta \log \pi_\theta(K_g[\mathbf{a}_t] \mid L_g[\mathbf{s}_t])\right].
    $$
    **关键发现（Fig. 2）**：若错误地在分母中使用 $\pi_{\theta_k}(K_g[\mathbf{a}_t] \mid L_g[\mathbf{s}_t])$，当策略尚未完全对称时分母可能极小导致剧烈梯度爆炸与训练崩溃；**保持分母为原始样本对数概率 $\pi_{\theta_k}(\mathbf{a}_t \mid \mathbf{s}_t)$** 且配合小权重初始化（Sec. IV-D, Fig. 5）可保证稳定收敛。
  - **严格等变结构的中性态奇异性警示（Sec. III & Sec. V）**：严格等变策略在中性不动点（$\mathbf{s} = L_g[\mathbf{s}]$，例如蛇形机器人完全拉直静止起动时）满足 $\pi(\mathbf{s}) = K_g[\pi(\mathbf{s})]$，迫使反对称分量（全部 4 路偏航角指令）严格为零，无法自主打破对称性迈出第一步，除非引入微小环境/传感噪声或扰动打破中性态。同时论文在 Sec. V 显式提出开放问题：“*当状态经过自编码器映射到隐空间后，如何实施对称性操作？*”——我们的 v8 **Reynolds 推理封装**直接对原始观测历史执行 $L_g$ 并共享编码器前向计算，干净绕开潜空间未知群表示难题。
- **关键实测发现（Table I）**：数据增强将等价目标间的回报差异降低一个数量级——CartPole `0.353 -> 0.003`、ANYmal-Climb `1.022 -> 0.124`、ANYmal-Push `2.255 -> 0.424`、Trifinger `75.752 -> 7.884`，且在存在真实载荷与磨损不对称的 ANYmal-D 实机上成功泛化。

#### (4) Impact of Static Friction on Sim2Real in Robotic RL (`arXiv:2503.01255`, Hu et al., Mar 2025)
- **核心问题与控制理论关节建模（Sec. V, Eq. 12–13）**：在 32 kg 六足机器人 Saturn Lite 上发现传统 DR（仅随机化惯量、粘滞阻尼、$K_p, K_d$）实机前进即摔倒，ActuatorNet（Eq. 10–11，仅平地数据采集）虽可在平地行走但在楼梯 OOD 场景失败且无法跨个体/磨损泛化。建立单关节二阶动力学模型（Eq. 12）：
  $$
  I_j \ddot{\theta}_j(t) + B_j \dot{\theta}_j(t) = \tau_j(t) + f_j(t), \qquad
  f_j(t) = \begin{cases} -b_j^c, & \dot{\theta}_j > 0 \\ +b_j^c, & \dot{\theta}_j < 0 \end{cases}, \qquad
  \tau_j(t) = k_{\mathrm{motor}}\big(k_p(\theta_j - a_t) - k_d \dot{\theta}_j\big),
  $$
  指出 $k_d$ 与粘滞阻尼 $B_j$ 存在线性抵消，而正弦扫频最小二乘辨识（$\theta_j^*(t) = A\sin(\omega t)$，Eq. 13）暴露出核心瓶颈：**静摩擦与峰值扭矩比 $f/\tau_{\max}$**（Table III）：Unitree Go1 仅为 **$0.13\%$**（$0.0481\text{ N}\cdot\text{m}$），而带 IP67 防水密封圈的 Saturn Lite 达 **$0.98\%$**（$0.442\text{ N}\cdot\text{m}$，拆除密封圈可降 70%）。
- **对 SerpWorm 的震动性量化对照**：核对 [`config.py`](../../src/serpworm/config.py) `v5_realistic` 预设，FashionStar 25 kg·cm 偏航舵机名义库仑静摩擦 $\tau_{\mathrm{sf}} = 0.06\text{ N}\cdot\text{m}$、峰值扭矩 $\tau_{\max} = 2.45\text{ N}\cdot\text{m}$，其比值 $f/\tau_{\max} = \mathbf{2.45\%}$（DR 上界 $0.18\text{ N}\cdot\text{m}$ 对应 **$\mathbf{7.35\%}$**）——是已导致 Sim-to-Real 崩溃的 Saturn Lite 的 **$2.5\times\text{--}7.5\times$**！
- **训练难题与“欺骗法（Deception Method）”（Sec. V & Table II）**：直接施加固定高静摩擦会像强非线性滤波器一样锁死 RL 早期混沌探索，使机器人原地不动；渐进微调（Iterative Method）则在实机引起严重站立抖动。作者提出**欺骗法**：将静摩擦随机化区间下界拉至 `0.0`（如 `[0.0, 1.2]×`，Table II），使部分并行环境在早期保持无静摩擦平滑梯度，同时让 RMA/自适应编码器在高摩擦环境下学会补偿起动死区与长期机械磨损。这直接印证了我们 `DomainRandConfig` 将 `yaw_static_friction_range=(0.00, 0.18)` 与 `servo_deadband_deg_range=(0.0, 1.8)` 下界设为 `0.0` 的合理性！

#### (5) RMA — Rapid Motor Adaptation (`arXiv:2107.04034`, Kumar, Fu, Pathak & Malik, RSS 2021)
- **核心架构与公式（Sec. III, Alg. 1–2）**：
  - Phase 1（1.2B 步）：环境因子编码器 $\mathbf{z}_t = \mu(\mathbf{e}_t) \in \mathbb{R}^8$（$\mathbf{e}_t \in \mathbb{R}^{17}$：质量/质心、12 电机强度、摩擦、局部地形高度）与基策略 $\mathbf{a}_t = \pi(\mathbf{x}_t, \mathbf{a}_{t-1}, \mathbf{z}_t)$ 联合 PPO 训练，并采用生物能耗惩罚指数课程 $k_{t+1} = k_t^{0.997}$（$k_0 = 0.03$，Appendix S1）避免早期惩罚过大导致“原地躺平”。
  - Phase 2（80M 步）：1D-CNN 自适应模块 $\hat{\mathbf{z}}_t = \phi(\mathbf{x}_{t-k:t-1}, \mathbf{a}_{t-k:t-1})$（$k=50$ 步即 $0.5\text{ s}$）通过 **on-policy DAgger 式回滚**最小化 $\mathrm{MSE}(\hat{\mathbf{z}}_t, \mathbf{z}_t)$。
  - **异步双频部署（Sec. III-C, Alg. 2）**：基策略运行于 $100\text{ Hz}$，自适应模块异步运行于 $10\text{ Hz}$——为 SerpWorm 未来将底层反射策略提频至 $50\text{ Hz}$、上下文编码保持 $10\text{ Hz}$ 提供了蓝本。
- **关键实测发现（Table II & Fig. 4）**：在每步 $p=0.01$ 动态重采样环境参数的严苛测试中，直接预测物理参数的 **`SysID` 成功率仅为 $56.5\%$**，甚至**低于无自适应纯 DR 的 $62.4\%$**，而预测低维隐式外参 $\hat{\mathbf{z}}_t$ 的 **RMA 达 $73.5\%$**（逼近 Expert 上界 $76.2\%$），证明了“**显式物理辨识受共线性制约，必须辅以任务导向的低维隐式潜空间**”这一第一性原理。

---

### 2.2 Continuum Contact, Soft Solver & Crawling Mechanics Pillars（连续体接触、隐式求解与爬行力学三大支柱）

#### (6) Continuum Robot Frictional Contact via Compliance-Based LCP (`arXiv:2611.18546`, Shen, Ha & Chen, IJRR 2026)
- **核心贡献**：针对连续体机器人在环境接触中同时存在**静摩擦粘滞（stick）**、**动摩擦滑移（slide）**与**反向运动滞回（hysteresis）**的难题，摒弃依赖惯量矩阵的刚体冲量 LCP，将 Cosserat 杆微分运动学与柔度矩阵（compliance matrix）直接耦合成准静态线性互补问题（LCP）。
- **精确公式（Eq. 21–35, Eq. 47）**：
  - 接触点增量位移与接触力、驱动增量的线性关系（Eq. 21, 25–26）：
    $$
    \Delta \mathbf{p}_c = \mathbf{J}_c \Delta \mathbf{q} + \mathbf{C}_c \Delta \mathbf{f}_c, \qquad
    \mathbf{v}_c = \mathbf{h} + \mathbf{C}_c \mathbf{f}_c, \quad \mathbf{h} = -\boldsymbol{\rho} + \mathbf{J}_c \Delta\mathbf{q} - \mathbf{C}_c \mathbf{f}_{0,c},
    $$
    对弹性杆有显式正定柔度矩阵 $\mathbf{C}_c = \mathbf{J}_u \mathbf{K}^{-1} \mathbf{J}_u^\top$（Eq. 47）。
  - 结合 Signorini 法向无穿透条件 $0 \le \mathbf{v}_n \perp \mathbf{f}_n \ge 0$ 与 $m$ 边多面体摩擦锥 $\mathbf{f}_c = \mathbf{n}_c \mathbf{f}_n + \mathbf{D}_c \boldsymbol{\beta}_c$，构建可证解存在（$M$ 协正，Lemma 1 & Eq. 35）的完整互补问题（Eq. 33–34）：
    $$
    \begin{bmatrix} \mathbf{v}_n \\ \boldsymbol{\sigma}_c \\ \boldsymbol{\eta}_c \end{bmatrix}
    = \begin{bmatrix}
    \mathbf{n}_c^\top \mathbf{C}_c \mathbf{n}_c & \mathbf{n}_c^\top \mathbf{C}_c \mathbf{D}_c & \mathbf{0} \\
    \mathbf{D}_c^\top \mathbf{C}_c \mathbf{n}_c & \mathbf{D}_c^\top \mathbf{C}_c \mathbf{D}_c & \mathbf{e}_c \\
    \boldsymbol{\mu}_c & -\mathbf{e}_c^\top & \mathbf{0}
    \end{bmatrix}
    \begin{bmatrix} \mathbf{f}_n \\ \boldsymbol{\beta}_c \\ \boldsymbol{\lambda}_c \end{bmatrix}
    + \begin{bmatrix} \mathbf{n}_c^\top \mathbf{h} \\ \mathbf{D}_c^\top \mathbf{h} \\ \mathbf{0} \end{bmatrix}, \qquad
    \mathbf{0} \le \begin{bmatrix}\mathbf{v}_n \\ \boldsymbol{\sigma}_c \\ \boldsymbol{\eta}_c\end{bmatrix} \perp \begin{bmatrix}\mathbf{f}_n \\ \boldsymbol{\beta}_c \\ \boldsymbol{\lambda}_c\end{bmatrix} \ge \mathbf{0}.
    $$
  - **尖角接触法向与冗余接触去重（Sec. Contact Detection, Fig. 3）**：相邻多个穿透节点会导致柔度矩阵近奇异，仅保留最大穿透代表点；在尖角/棱边（corner contact）处障碍物 SDF 梯度不唯一，改取机器人光滑截面的内法向 $\mathbf{n}_i = -\mathbf{R}(s_i)\mathbf{r}_i / \|\mathbf{r}_i\|$。
- **关键实测发现（Tables 3–8 & TDCR-plane 实验）**：
  - 在 287.9 mm 拉绳连续体机器人（TDCR）碰壁实验中，相比无摩擦模型，LCP 摩擦模型将末端最大位置误差由 **$45.4\text{ mm}$ 骤降至 $5.85\text{ mm}$（$-87.1\%$）**，最大弯曲角误差由 **$60.6^\circ$ 降至 $4.6^\circ$**，完整捕获“粘滞蓄能 $\to$ 突破静摩擦锥 $\to$ 弹性突跳释放”的全过程，而仅含动摩擦模型完全丢失反向过渡相位。
  - 管内多点接触实验（Tube-pipe, Fig. 15）揭示：**管口角点接触与管壁面接触相互激发会显著放大 stick-slip 颤振**；而 Stribeck 速度衰减效应是误差主因（Sec. Conclusion）。

#### (7) Py-DiSMech — DDG Implicit Contact & Thin-Shell Soft Robot Framework (`arXiv:2512.09911`, Lahoti, Chaiyakul & Jawed, Dec 2025)
- **核心思想与公式（Sec. III, Table I, Eq. 15–28）**：
  - 基于离散微分几何（DDG）统一建模离散弹性杆（DER，自由度 $3N+E$）与离散薄壳（铰链弯矩或中边法向形状算子 $\boldsymbol{\Lambda}_i$，薄壳中边刚度 $K_{\mathrm{midedge}} = \frac{E_{\mathrm{shell}} h^3}{24(1 - \nu_{\mathrm{shell}}^2)}$，Table I），采用隐式欧拉 / Newmark–Beta 牛顿迭代（Eq. 4–7）以解析雅可比实现高刚度大步长稳定求解。
  - **$C^2$ 光滑软惩罚隐式自碰撞接触能（IMC，Eq. 25–27）**：对候选接触对距离 $\Delta$（支持 Point-Point / Point-Edge / Edge-Edge / Point-Triangle 四类杆-杆、杆-壳、壳-壳自碰撞），定义分段 $C^2$ 接触势能与光滑库仑摩擦：
    $$
    E_{\mathrm{con}}(\Delta) =
    \begin{cases}
    (2h - \Delta)^2, & \Delta \le 2h - \delta, \\
    0, & \Delta \ge 2h + \delta, \\
    \left[\dfrac{1}{K_1}\log\!\left(1 + e^{K_1(2h - \Delta)}\right)\right]^2, & \text{otherwise},
    \end{cases}
    \qquad
    \mathbf{F}_{\mathrm{fr}, i} = -\mu \left(\frac{2}{1 + e^{-K_2\|\mathbf{u}\|}} - 1\right)\hat{\mathbf{u}}\,\|\mathbf{F}_{\mathrm{con}, i}\|,
    $$
    其中 $K_1 = 15/\delta,\; K_2 = 15/\nu$。该能量函数在接触阈值附近无限光滑且 Hessian 严格半正定，免除硬屏障线搜索开销。
  - **自然应变空间 PI 闭环控制（Sec. VII, Fig. 10 & Fig. 12）**：在应变/曲率残差空间设计带抗饱和与变化率限幅的 PI 反馈，使蛇形杆跟踪误差收敛至 $0.1\%$ 以内。
- **性能标杆（Table II）**：在 $E = 1\text{ GPa}$ 刚性螺旋杆基准下，Py-DiSMech 隐式欧拉（$\Delta t = 10\text{ ms}$）相比显式 Verlet 的 PyElastica（$\Delta t = 1\,\mu\text{s}$）实现 **$32\times$ 加速**（`7.16 s` vs `229.52 s`）；整体 6 组算例加速 $2\text{--}30\times$。

#### (8) Rate-Independent Soft Crawlers & Stasis Domains (`arXiv:1710.08340`, Paolo Gidoni, Dec 2017)
- **核心理论（Sec. 2–4, Eq. 2.4–2.14, Remark 2.5, Lemma 3.1, Examples 3.2 & 3.5）**：
  - 将无足软体爬虫在干摩擦上的准静态推进抽象为无 Dirichlet 边界条件的**率无关变分包含系统（RIS）**：
    $$
    \mathbf{0} \in \partial \mathcal{R}(\dot{\mathbf{x}}(t)) + D_{\mathbf{x}} E(t, \mathbf{x}),
    $$
    其中构型能 $E(t, \mathbf{x}) = E_{\mathrm{sh}}(t, \sigma(\mathbf{x}))$ 仅依赖于形状坐标 $\mathbf{z} = \sigma(\mathbf{x})$（对刚体平移 $y = \pi(\mathbf{x})$ 严格不变），耗散势 $\mathcal{R}$ 为正一次齐次凸函数。
  - **形状归约与净位移唯一性（Theorem 2.2, Eq. 2.8 & 2.14）**：只要满足唯一性条件 $(\ast)$（对均匀离散 $N$ 节点爬虫即 $m\mu_- - (N-m)\mu_+ \neq 0, \forall m$，Lemma 3.1），净平移速度由最小耗散原理唯一决定：$\dot{y}(t) = v_m(\dot{\mathbf{z}}(t)) = \arg\min_{v} \mathcal{R}(\chi(\dot{\mathbf{z}}, v))$。
  - **停滞域（Stasis Domain）$\mathcal{C}_{\mathrm{sh}} = \partial \mathcal{R}_{\mathrm{sh}}(\mathbf{0})$ 与最小有效行程阈值（Remark 2.5, Example 3.2 & 3.5）**：
    - 只要内部弹性张力 $-D_{\mathbf{z}} E_{\mathrm{sh}}$ 位于凸停滞域 $\mathcal{C}_{\mathrm{sh}}$ 内部，法锥 $\mathcal{N}_{\mathcal{C}_{\mathrm{sh}}} = \{\mathbf{0}\}$，机器人产生**严格零位移**；只有驱动形变将内力推至 $\partial\mathcal{C}_{\mathrm{sh}}$ 边界时才能产生净推进。
    - 对弹性刚度 $k$、前向摩擦 $\mu_+$、反向锚固摩擦 $\mu_- > \mu_+$ 的爬虫，周期驱动幅值 $\Delta L$（或应变幅值 $\Delta\varepsilon$）必须突破**死区阈值**才能前进，单周期精确净步距为：
      $$
      \Delta y_{\mathrm{discrete}} = \max\!\left(0,\; \Delta L - \frac{2\mu_+}{k}\right), \qquad
      \Delta y_{\mathrm{cont}} = l\,\max\!\left(0,\; \Delta\varepsilon - \frac{\mu_+ l}{k}\right)\frac{\mu_- - \mu_+}{\mu_- + \mu_+}.
      $$
  - **时变摩擦操控（Sec. 4–5）**：当机器人能通过体节隆起/俯仰主动调制局部法向接触力从而产生时变耗散势 $\mathcal{R}(t, \dot{\mathbf{x}})$ 时，即便摩擦各向异性比退化（如软土深陷或湿滑路面），仍可通过法向力-形变相位协同恢复高效净推进。

---

## 3. Code-Level Gap Diagnosis Against Current SerpWorm-RL v6/v7（基于源码的逐项差距诊断）

对照 [`models.py`](../../src/serpworm/rl/models.py)、[`train_autonomous.py`](../../src/serpworm/rl/train_autonomous.py)、[`autonomous_env.py`](../../src/serpworm/envs/autonomous_env.py)、[`wheel_stiction.py`](../../src/serpworm/physics/wheel_stiction.py) 与 [`gpu_core.py`](../../src/serpworm/physics/gpu_core.py)，当前 v6/v7 系统的精确短板与实证表现如下：

| 诊断维度 | 当前 v6/v7 实现（精确源码定位） | 顶会文献揭示的本质缺陷 | 当前实测数据佐证 |
|---|---|---|---|
| **1. 状态估计与上下文表示** | `models.py:337-409`：`est_tcn` + `est_mlp` 仅输出 7D 显式物理估计 $\hat{\mathbf{e}}_t$，经 `SmoothL1(beta=0.15)` 监督后 `detach()` 喂给 `actor_mlp`；**无隐式潜变量 $\mathbf{z}_t$，无下一步观测重构，无对比损失**。 | RMA Table II 证明显式 SysID（56.5%）因参数共线性弱于潜向量（73.5%）；DreamWaQ Table III 与 HIMLoco Table 4 证明缺失隐式上下文在难/未见地形上发生断崖退化（软坡 $0\%$、踏空 $0\text{--}5\%$）。 | `v6_ablation_eval.json`：`carpet_100` 上 `EstNet drag MAE` 高达 `0.123`（`v6b`）与 `0.237`（`v6.1`）；`open_hard_140` 速度估计 MAE 达 `38.6–68.0 mm/s`（无法仅凭 7D 显式标量分辨滚阻/坡度/电池电压跌落）。 |
| **2. 特征学习与抗 DR 噪声塌缩** | `models.py:329-331`：`est_tcn` / `actor_tcn` / `critic_tcn` 三路独立 2 层因果卷积（`k=3, s=2`），无多尺度感受野，无前向-后向动力学自监督。 | HIMLoco Sec. 2 & Fig. 4 指出强 DR 下纯回归易拟合随机噪声导致潜空间混叠；批内对比学习（InfoNCE / SwAV）利用并行多地形负样本显式拉开 Hard / Pipe / Carpet / Soft-Soil 边界。 | `v6b` 穿管切软土时需约 1–2 个完整波周期才完成步态重展开，且单尺度 `k=3, s=2` 对瞬时碰壁冲击（20–100 ms）与长周期蠕动波（1.2–1.6 s）难以兼顾最优时间分辨率。 |
| **3. 镜像对称训练与推理等变性** | (a) `train_autonomous.py:346-354` 对镜像增强后半批重算 `f_logp[base_b:]`；(b) `train_autonomous.py:410-418` 使用固定权重 `0.25 * sym_loss`；(c) `models.py:442-450` 推理端无精确群等变约束。 | Symmetry Considerations (`2403.04359` Eq. 6 & Fig. 2) 证明重算镜像分母 $\pi_{\theta_k}(K_g a_t \mid L_g s_t)$ 会引入离策略高方差；固定软正则无法保证严格对称，推理端仍留残差偏航。 | `v6_ablation_eval.json`：`open_hard_140` 直线指令下 `v6a` 偏航 `-0.0097 rad/s`、`v6b` `+0.0066 rad/s`、`v6.1` `-0.0054 rad/s`；实测未封装 checkpoint 动作镜像误差显著 $> 0$。 |
| **4. 估计置信度与早期自举** | `models.py:409`：从第 1 轮起硬性拼接 `est_pred.detach()`，无不确定度门控，也无 AdaBoot 调度。 | DreamWaQ Eq. 8 & Table III 证明无 AdaBoot 时早期噪声污染策略（存活率 `90.71%` vs `95.23%`）；无后验方差 $\boldsymbol{\sigma}_z^2$ 无法在 OOD 冲击瞬间抑制错误外推。 | `train_autonomous.py:406` 被迫在 Stage 3 手工把 `est_w` 减半至 `0.5` 防止信赖域漂移，属启发式补丁而非不确定度感知机制。 |
| **5. 物理接触与停滞域机理** | (a) `pipe_geometry.py:157-161` 管壁摩擦仅用粘性光滑 `tanh(\|v_t\|/0.015)`，无切向粘滑记忆；(b) `gpu_core.py:526-531` Capstan 摩擦无反向粘滑状态；(c) 40 钢片有地面/管壁接触但无钢片间自碰撞。 | `2611.18546` 证明无粘滑静态相的纯滑移模型丢失方向反转滞回与突跳；`2512.09911` 提供 $C^2$ 软惩罚隐式自碰撞势能；`1710.08340` 停滞域公式揭示软土减速机理。 | `v6a`（无 `q_err`）在 `soft_soil_100` 陷入停滞域仅 `12.1 mm/s`，引入软件负荷感知 `q_err` 后 `v6b` 突破死区升至 `36.1 mm/s`（$3.0\times$）。 |

---

## 4. Side-by-Side Architecture Comparison: Current v6/v7 vs Top-Venue SOTA vs Our v8 Upgrade（三代架构全景对比表）

| 技术维度 | Current SerpWorm v6 / v7 (`ProprioceptiveEstimatorActorCritic`) | Top-Venue SOTA（RMA / DreamWaQ / HIMLoco / Symmetry） | **Our v8 Upgrade (`HIMDreamWaQActorCritic` + `GroupEquivariantPolicyWrapper`)** |
|---|---|---|---|
| **观测与部署传感器** | 51D / 65D（含纯软件 `q_err`），14 路舵机编码器 + 单 6 轴 IMU（无加速计），$H=16$（1.6 s） | 关节编码器 + IMU，$H=5$（DreamWaQ / HIMLoco）或 $H=50$（RMA 100 Hz） | **完全继承 51D / 65D 纯本体接口**（零新增传感器），100% 兼容现有 `SerpWormPolicyRuntime` 与 TorchScript 导出 |
| **时序编码骨干** | 3 路独立单尺度 `_CausalConv1d`（`k=3, s=2` $\times 2$） | 3 层 MLP（DreamWaQ / HIMLoco）或 3 层 1D-CNN（RMA） | **共享多尺度因果 1D-TCN**（复用 `_CausalConv1d`，并行捕获短时碰壁/打滑脉冲与长周期步态波），CENet 与 HIM 共用骨干以压低 CPU 推理延迟至 `< 1.5 ms` |
| **显式状态估计头** | 7D `EstNet`：`[vx, vy, wz, wall_conf, wall_side, drag, slip]`，`SmoothL1(beta=0.15)` | 3D 机体速度 `MSE`（DreamWaQ / HIMLoco）；无显式接触/阻尼头 | **保留全部 7D 物理估计头**（连续体蛇蠕专属管壁+软土+滑移遥测）+ `SmoothL1(beta=0.15)` |
| **隐式上下文表征 $\mathbf{z}_t$** | **无**（纯显式 7D） | DreamWaQ：16D $\beta$-VAE 高斯后验；HIMLoco：16D 单位球面对比响应；RMA：8D 两阶段特权蒸馏 | **16D $\beta$-VAE 高斯后验 $(\boldsymbol{\mu}_z, \log\boldsymbol{\sigma}_z^2)$ + 后验不确定度门控 $\tilde{\mathbf{z}}_t = \mathbf{g}_t \odot \mathbf{z}_t$**，单阶段端到端训练 |
| **前向动力学自监督** | **无** | DreamWaQ：$\mathrm{MSE}(\hat{\mathbf{o}}_{t+1}, \mathbf{o}_{t+1}) + \beta D_{\mathrm{KL}}$；HIMLoco：无像素/特征重构 | **信息瓶颈下一步本体解码器** $\hat{\mathbf{o}}_{t+1}^{\mathrm{sens}} = f_{\mathrm{dec}}(\hat{\mathbf{e}}_t, \mathbf{z}_t, \mathbf{a}_t)$（严格经由 $\mathbf{z}_t$ 瓶颈消除后验塌缩，屏蔽外生指令维 `cmd[-3:]`） |
| **抗 DR 噪声对比对齐** | **无** | HIMLoco：SwAV 原型交换对比损失（Sinkhorn–Knopp，$K=16$ 原型） | **信息瓶颈 HIM 对比动力学头** $f_{\mathrm{him,src}}(\mathbf{z}_t, \mathbf{a}_t) \leftrightarrow f_{\mathrm{him,tgt}}^{\mathrm{EMA}}(\tilde{\mathbf{o}}_{t+1}^{\mathrm{sens}})$：对称 InfoNCE（$\tau=0.10$）+ 16 原型 3 步 Sinkhorn–Knopp SwAV |
| **早期自举与正则** | 固定 `est_pred.detach()`，Stage 3 手工半权 | DreamWaQ：AdaBoot $p_{\mathrm{boot}} = 1 - \tanh(\mathrm{CV}(R))$ + 电机功率方差奖励 | **内置 `adaboot_prob(R)` 调度** + 后验不确定度门控 $\mathbf{g}_t$ + 下一阶段 v8 训练集成电机功率方差均衡项 |
| **左右镜像对称性** | 2× 镜像增强（镜像分母重算）+ 软正则 `0.25 * sym_loss`，推理残差偏航 $>0$ | Symmetry (`2403.04359`)：原始分母 Eq. 6 数据增强（无硬推理等变封装） | **推理端精确 $\mathbb{Z}_2$ Reynolds 群等变封装**（$\bar{\boldsymbol{\mu}}(\mathbf{o}) = \frac{1}{2}(\boldsymbol{\mu}(\mathbf{o}) + K_g^{-1}\boldsymbol{\mu}(L_g[\mathbf{o}]))$，实测镜像误差严格 **`0.0`**）+ 训练期 Eq. 6 分母修复 |
| **部署参数量与单帧 CPU 延迟** | 65D 部署 `257,653` 参；单线程 CPU p50 `0.828 ms`（v6a 51D `1.088 ms`） | 50–100 Hz 机载 NUC 实时 | **65D 部署 `229,125` 参（`0.889×`，轻 `11.1%`）**；单线程 CPU p50 **`1.124 ms`**（51D **`1.057 ms`**，等变拼批 `2.158 ms`，较 `100 ms` 周期快 `~46×`） |

---

## 5. Exact Mathematical Specification of v8 Upgrade（v8 架构精确数学规范）

本节定义已与 `@rl-architect` 在模块接口层面完全对齐的 v8 `HIMDreamWaQActorCritic` 与 `GroupEquivariantPolicyWrapper`（落地于新文件 [`src/serpworm/rl/him_dreamwaq.py`](../../src/serpworm/rl/him_dreamwaq.py)，零修改 `models.py`）。

### 5.1 Multi-Scale Causal Proprioceptive Encoder（多尺度因果本体编码器）
给定纯本体历史窗口 $\mathbf{O}_t = \mathbf{o}_{t-H+1:t} \in \mathbb{R}^{B \times H \times F}$（$H=16, F \in \{51, 65\}$），先经设备驻留 Welford 归一化器 `RunningNorm(F)` 得 $\tilde{\mathbf{O}}_t$。将其输入共享双分支多尺度右对齐因果 1D-TCN `context_tcn`（`_RightAlignedCausalConv1d` 继承 `_CausalConv1d`，偶数长度且步幅 $s=2$ 时取左填充 $k-2$，确保最新帧 $t = H-1$ 严格落于最右输出槽且零未来泄漏）：
- **短窗高频分支**（$k=3, s=2 \times 2$: $F \to 48 \to 32$，感受野 7 帧 $\approx 0.7\text{ s}$）：捕获管壁碰撞、棘轮卡顿与打滑瞬变；
- **宽感受野低频分支**（$k=5, s=2 \times 2$: $F \to 24 \to 16$，感受野 13 帧 $\approx 1.3\text{ s}$）：捕获完整蛇行/蠕动波相位与软土沉陷累积。

两分支拼接后输出 $48 \times T_{\mathrm{out}} = 192\text{D}$ 严格保序时序槽（与 v6 `tcn_flat = 192` 精确对齐），并与最新归一化帧 $\tilde{\mathbf{o}}_t$ 跳跃拼接得 $\mathbf{h}_t \in \mathbb{R}^{B \times (192 + F)}$，再经 `cenet_trunk`（`192 -> 128`, `LayerNorm + SiLU`）提取共享上下文特征 $\mathbf{c}_t \in \mathbb{R}^{B \times 128}$。**部署侧单编码器共享设计**：`context_tcn` 经 `stop_grad` 直接复用至 `actor_mlp`（`256 -> 192 -> 14`，隔离 PPO 梯度不破坏自监督表征），使 65D 部署参数量较 v6 反降 $11.1\%$（`229,125` vs `257,653`）且单线程 CPU 推理保持 `< 1.5 ms`。

### 5.2 DreamWaQ CENet $\beta$-VAE + Uncertainty-Gated Context（$\beta$-VAE 隐式地形想象与不确定度门控）
由共享特征 $\mathbf{c}_t$ 并行引出显式物理估计头与变分潜变量后验头：
1. **7D 显式本体物理状态估计（与 v6/v7 语义完全一致）**：
   $$
   \hat{\mathbf{e}}_t = \Big[\hat{v}_x,\; \hat{v}_y,\; \hat{\omega}_z,\; \sigma(\hat{z}_{\mathrm{wall}}),\; \tanh(\hat{z}_{\mathrm{side}}),\; \sigma(\hat{z}_{\mathrm{drag}}),\; \sigma(\hat{z}_{\mathrm{slip}})\Big] \in \mathbb{R}^7,
   $$
   $$
   \mathcal{L}_{\mathrm{est}} = \mathrm{SmoothL1}_{\beta=0.15}\!\left(\hat{\mathbf{e}}_t,\; \mathbf{e}_t^*\right).
   $$
2. **$d_z = 16$ 维 $\beta$-VAE 高斯后验与重参数化采样**：
   $$
   \left(\boldsymbol{\mu}_z,\; \log\boldsymbol{\sigma}_z^2\right) = f_{\mathrm{vae}}(\mathbf{c}_t), \qquad
   \log\boldsymbol{\sigma}_z^2 \leftarrow \mathrm{clip}\!\left(\log\boldsymbol{\sigma}_z^2,\; -6.0,\; 2.0\right),
   $$
   $$
   \mathbf{z}_t =
   \begin{cases}
   \boldsymbol{\mu}_z + \exp\!\left(\tfrac{1}{2}\log\boldsymbol{\sigma}_z^2\right) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), & \text{training}, \\
   \boldsymbol{\mu}_z, & \text{deterministic inference}.
   \end{cases}
   $$
3. **后验不确定度门控上下文（Uncertainty-Gated Context）**：对 CENet 后验统计量施以梯度截断（`stop_grad`），使 PPO 端到端优化门控参数 $\mathbf{W}_g, \mathbf{b}_g$ 而不污染 CENet 表征，同时在 OOD 冲击瞬间按后验方差自动衰减高噪声维度：
   $$
   \mathbf{g}_t = \sigma\!\left(\mathbf{W}_g \big[\mathrm{sg}(\boldsymbol{\mu}_z),\; \mathrm{sg}(\log\boldsymbol{\sigma}_z^2)\big] + \mathbf{b}_g\right) \in (0, 1)^{d_z}, \qquad
   \tilde{\mathbf{z}}_t = \mathbf{g}_t \odot \mathrm{sg}(\mathbf{z}_t).
   $$
   Actor 消费 $\big[\mathrm{sg}(\mathbf{h}_t),\; \mathrm{sg}(\hat{\mathbf{e}}_t),\; \tilde{\mathbf{z}}_t\big]$，保持 PPO 信赖域稳定。
4. **信息瓶颈下一步本体解码器与 FP32 $\beta$-VAE ELBO 损失**（`2301.10602` Fig. 2 & Eq. 7）：屏蔽外生随机切换的最后 3 维速度指令 `cmd`，且**严格移除当前帧观测 `cur_sens` 跳跃捷径**（防止解码器绕开 $\mathbf{z}_t$ 引发 $\beta$-VAE 后验塌缩），强制通过信息瓶颈 $(\hat{\mathbf{e}}_t, \mathbf{z}_t, \mathbf{a}_t)$ 预测下一步传感观测子集 $\tilde{\mathbf{o}}_{t+1}^{\mathrm{sens}} = \tilde{\mathbf{o}}_{t+1, 0:F-3} \in \mathbb{R}^{F_{\mathrm{sens}}}$（$48\text{D}$ 或 $62\text{D}$）：
   $$
   \hat{\mathbf{o}}_{t+1}^{\mathrm{sens}} = f_{\mathrm{dec}}\!\left(\hat{\mathbf{e}}_t,\; \mathbf{z}_t,\; \mathbf{a}_t\right),
   $$
   $$
   \mathcal{L}_{\mathrm{rec}} = \mathrm{MSE}\!\left(\hat{\mathbf{o}}_{t+1}^{\mathrm{sens}},\; \mathrm{sg}\!\left(\tilde{\mathbf{o}}_{t+1}^{\mathrm{sens}}\right)\right), \qquad
   \mathcal{L}_{\mathrm{KL}} = -\frac{1}{2B}\sum_{i=1}^B \sum_{j=1}^{d_z} \left(1 + \log\sigma_{z,ij}^2 - \mu_{z,ij}^2 - \sigma_{z,ij}^2\right),
   $$
   $$
   \mathcal{L}_{\mathrm{VAE}} = \mathcal{L}_{\mathrm{rec}} + \beta_{\mathrm{KL}}\,\mathcal{L}_{\mathrm{KL}} \qquad (\text{默认 }\beta_{\mathrm{KL}} = 0.002\text{，FP32 计算防止 bf16 下溢}).
   $$

### 5.3 HIMLoco Contrastive Dynamics Alignment Head（HIM 对比动力学对齐头）
为避免强 DR 噪声导致纯重构表征塌缩（`2312.11460` Fig. 4），同样严格移除 `cenet_feat` 捷径，强制对比动力学预测经由 $d_z = 16$ 瓶颈潜变量与动作 $[\mathbf{z}_t, \mathbf{a}_t] \in \mathbb{R}^{16 + 14}$ 映射至 $d_c = 32$ 维单位球面（目标侧在线分支 `him_tgt_proj` 辅以动量 $m = 0.99$ 的 Polyak EMA 影子分支 `him_tgt_ema` 稳定目标表征）：
$$
\mathbf{l}_t^{\mathrm{src}} = \frac{f_{\mathrm{him,src}}(\mathbf{z}_t, \mathbf{a}_t)}{\|f_{\mathrm{him,src}}(\mathbf{z}_t, \mathbf{a}_t)\|_2 + \varepsilon} \in \mathbb{S}^{31}, \qquad
\mathbf{l}_t^{\mathrm{tgt,on}} = \frac{f_{\mathrm{him,tgt}}(\tilde{\mathbf{o}}_{t+1}^{\mathrm{sens}})}{\|f_{\mathrm{him,tgt}}(\tilde{\mathbf{o}}_{t+1}^{\mathrm{sens}})\|_2 + \varepsilon}, \qquad
\mathbf{l}_t^{\mathrm{tgt,EMA}} = \frac{f_{\mathrm{him,tgt}}^{\mathrm{EMA}}(\tilde{\mathbf{o}}_{t+1}^{\mathrm{sens}})}{\|f_{\mathrm{him,tgt}}^{\mathrm{EMA}}(\tilde{\mathbf{o}}_{t+1}^{\mathrm{sens}})\|_2 + \varepsilon}.
$$
给定批内 $B$ 个跨地形转移对与温度 $\tau = 0.10$，定义前向与反向余弦逻辑矩阵 $S_{ij}^{\mathrm{fwd}} = \frac{1}{\tau}(\mathbf{l}_i^{\mathrm{src}})^\top \mathbf{l}_j^{\mathrm{tgt,EMA}}$、$S_{ij}^{\mathrm{bwd}} = \frac{1}{\tau}(\mathbf{l}_i^{\mathrm{tgt,on}})^\top \mathrm{sg}(\mathbf{l}_j^{\mathrm{src}})$（关闭 EMA 时退化为标准对称矩阵 $S_{ij} = \frac{1}{\tau}(\mathbf{l}_i^{\mathrm{src}})^\top \mathbf{l}_j^{\mathrm{tgt,on}}$），联合优化：
- **对称 InfoNCE 动力学对比损失**（权重 $w_{\mathrm{nce}} = 0.25$）：
  $$
  \mathcal{L}_{\mathrm{InfoNCE}} = \frac{1}{2B}\sum_{i=1}^B \left[-\log\frac{\exp(S_{ii}^{\mathrm{fwd}})}{\sum_{j=1}^B \exp(S_{ij}^{\mathrm{fwd}})} - \log\frac{\exp(S_{ii}^{\mathrm{bwd}})}{\sum_{j=1}^B \exp(S_{ij}^{\mathrm{bwd}})}\right].
  $$
- **HIMLoco SwAV 原型交换损失**（$K=16$ 单位原型 $\mathbf{E} = [\bar{\mathbf{e}}_1, \dots, \bar{\mathbf{e}}_K]$ + 3 步 Sinkhorn–Knopp 等分目标 $\mathbf{q}_i^{\mathrm{src}}, \mathbf{q}_i^{\mathrm{tgt}}$，熵正则 $\epsilon_{\mathrm{SK}} = 0.05$，`2312.11460` Eq. 1–2，权重 $w_{\mathrm{swav}} = 0.05$）：
  $$
  \mathcal{L}_{\mathrm{SwAV}} = -\frac{1}{2B}\sum_{i=1}^B \left(\mathbf{q}_i^{\mathrm{tgt}\top}\log \mathbf{p}_i^{\mathrm{src}} + \mathbf{q}_i^{\mathrm{src}\top}\log \mathbf{p}_i^{\mathrm{tgt}}\right).
  $$
- **v8 总辅助目标函数**（与 PPO 截断目标联合更新，全部在 FP32 中计算以兼容 RTX 5090 `bf16` 自动混精度）：
  $$
  \mathcal{L}_{\mathrm{aux}} = w_{\mathrm{est}}\,\mathcal{L}_{\mathrm{est}} + w_{\mathrm{rec}}\,\mathcal{L}_{\mathrm{rec}} + \beta_{\mathrm{KL}}\,\mathcal{L}_{\mathrm{KL}} + w_{\mathrm{nce}}\,\mathcal{L}_{\mathrm{InfoNCE}} + w_{\mathrm{swav}}\,\mathcal{L}_{\mathrm{SwAV}},
  $$
  默认权重（[`HIMDreamWaQConfig`](../../src/serpworm/rl/him_dreamwaq.py)）：$w_{\mathrm{est}} = 1.0,\; w_{\mathrm{rec}} = 0.5,\; \beta_{\mathrm{KL}} = 0.002,\; w_{\mathrm{nce}} = 0.25,\; w_{\mathrm{swav}} = 0.05$；同时内置 DreamWaQ Eq. 8 自举概率函数 `adaboot_prob(R)` $= 1 - \tanh\!\left(\frac{\mathrm{std}(R)}{|\mathrm{mean}(R)| + \varepsilon}\right)$ 供下一阶段 RTX 5090 完整训练开启 AdaBoot。

### 5.4 Exact $\mathbb{Z}_2$ Group-Equivariant Inference Wrapper & Training Fix（精确群等变推理封装与训练修正）
设 $y \to -y$ 镜像反射群 $\mathbb{Z}_2 = \{e, g\}$ 在观测、动作与 EstNet 空间上的精确对合算子（$L_g^2 = I, K_g^2 = I, M_g^2 = I$，全面兼容 `51D` / `65D` 自主架构以及 `54D`（14D unified）/ `49D`（9D unified）v5 架构，与 [`autonomous_env.py:91-149`](../../src/serpworm/envs/autonomous_env.py#L91-L149) 逐位一致）：
- $L_g$：按 `_PAIR_SWAP_10 = (1,0,3,2,5,4,7,6,9,8)` 交换左右收绳编码器（及 `dq`、`q_err`、`prev_action`），反转 4 路偏航角，反射陀螺 $(\omega_x, \omega_y, \omega_z) \mapsto (-\omega_x, +\omega_y, -\omega_z)$ 与重力投影 $(g_x, g_y, g_z) \mapsto (+g_x, -g_y, +g_z)$，反转偏航指令 $\omega_{z,\mathrm{cmd}} \mapsto -\omega_{z,\mathrm{cmd}}$；
- $K_g = K_g^{-1}$：前 10 维按 `_PAIR_SWAP_10` 互换，后 4 维取负（9D 模式下前 5 维配对拉绳不变、后 4 维偏航取负）；
- $M_g = M_g^{-1} = \mathrm{diag}(+1, -1, -1, +1, -1, +1, +1)$。

对任意已训练策略（`TemporalActorCritic`、`ProprioceptiveEstimatorActorCritic` 或 `HIMDreamWaQActorCritic`），**Reynolds 群等变平均算子**定义为：
$$
\bar{\boldsymbol{\mu}}(\mathbf{O}_t) = \frac{1}{2}\Big(\boldsymbol{\mu}(\mathbf{O}_t) + K_g^{-1}\boldsymbol{\mu}\!\left(L_g[\mathbf{O}_t]\right)\Big), \qquad
\bar{\mathbf{e}}(\mathbf{O}_t) = \frac{1}{2}\Big(\hat{\mathbf{e}}(\mathbf{O}_t) + M_g^{-1}\hat{\mathbf{e}}\!\left(L_g[\mathbf{O}_t]\right)\Big).
$$
- **精确等变性证明**：因为 $L_g, K_g, M_g$ 均为对合算子，故对任意输入 $\mathbf{O}_t$ 恒有
  $$
  \bar{\boldsymbol{\mu}}\!\left(L_g[\mathbf{O}_t]\right)
  = \frac{1}{2}\Big(\boldsymbol{\mu}(L_g[\mathbf{O}_t]) + K_g^{-1}\boldsymbol{\mu}(\mathbf{O}_t)\Big)
  = K_g\!\left[\frac{1}{2}\Big(K_g^{-1}\boldsymbol{\mu}(L_g[\mathbf{O}_t]) + \boldsymbol{\mu}(\mathbf{O}_t)\Big)\right]
  = K_g\!\left[\bar{\boldsymbol{\mu}}(\mathbf{O}_t)\right].
  $$
  实现上默认提供 `batched=False`（独立前向两次，规避 BLAS 批维 GEMM 舍入差异以保证 IEEE-754 位级精确 **`0.0`** 镜像误差）与 `batched=True`（拼批 `[O_t, L_g(O_t)]` GPU 极速模式）双路径。
- **中性态奇异性边界（`2403.04359` Sec. III）**：在精确零偏航直立静止状态（$L_g[\mathbf{O}_t] = \mathbf{O}_t$），等变性必然导出 $\bar{\mu}_{\mathrm{yaw}}(\mathbf{O}_t) = 0$。在闭环部署中，真实 IMU/编码器白噪声或起动首步微扰即刻离开该零测度不变子空间，随后策略全程享受严格零偏航漂移与完全对称的左右转向响应。
- **v8 训练端配套单行修复（`2403.04359` Eq. 6）**：未来在 RTX 5090 启动完整 v8 训练时，将镜像后半批的旧策略参考对数概率由重算改为直接复用原始样本概率 `f_logp[base_b:] = f_logp[:base_b]`，即可彻底消除离策略高方差风险。

---

## 6. Empirical Verification & Local v8 Ablation Results（本地实证验证与 v8 消融结果）

本节全部数据直接核自已通过 `pytest tests/test_him_dreamwaq.py -v`（**29/29 通过**）的 [`src/serpworm/rl/him_dreamwaq.py`](../../src/serpworm/rl/him_dreamwaq.py)、[`src/serpworm/rl/ablation_bench.py`](../../src/serpworm/rl/ablation_bench.py) 及本地确定性基准产物 [`runs/v8_ablation/v8_ablation_summary.json`](../../runs/v8_ablation/v8_ablation_summary.json)（`seed=42`，FP64 `dt=1.0 ms` 物理仿真，PyTorch `2.10.0+cu128`）。

### 6.1 Benchmark (a): Exact $\mathbb{Z}_2$ Group-Equivariant Inference on Real Trained Checkpoints（真实权重精确群等变验证）

#### (1) 静态镜像审计（128 帧随机本体序列 + 中性直立零偏航探测）

| 真实训练 Checkpoint | 观测布局 | 动作镜像误差 Mean / Max（封装前） | 动作镜像误差 Exact / Batched（封装后） | EstNet 镜像误差 Mean / Max（前 $\to$ 后） | 中性态偏航偏置（前 $\to$ 后） |
|---|---|---:|---:|---:|---:|
| `runs/v6_1_65d_elec/best.pt` (r200) | 65D + EstNet | `0.2769 / 1.9872` | **`0.0 / 0.0`** | `0.0841 / 0.9985` $\to$ **`0.0 / 0.0`** | `0.9985` $\to$ **`0.0`** |
| `runs/v6a_51d_pure/best.pt` (r116) | 51D + EstNet | `0.4060 / 1.9962` | **`0.0 / 0.0`** | `0.1489 / 0.9988` $\to$ **`0.0 / 0.0`** | `0.9984` $\to$ **`0.0`** |
| `runs/unified_v5_hispeed_14d/best.pt` (r185) | 54D Temporal | `0.2701 / 1.9798` | **`0.0 / 0.0`** | — | `0.9021` $\to$ **`0.0`** |

#### (2) 闭环 `AutonomousPhaseEnv` 动力学验证（`dt = 1.0 ms`, `E = 8`, `1.2 s`, 0% 失效）

| Checkpoint | 模式 | 轨迹动作不对称 Max | 直行航向漂移（°）/ $\omega_z$（rad/s） | 直行 $v_x$（mm/s） | 左/右转向 $\omega_z$（rad/s） | 左/右转向偏置（rad/s） / 航向不对称（°） |
|---|---|---:|---:|---:|---:|---:|
| `v6_1_65d_elec` | Base（原权重） | `1.9972` | `0.1832°` / `-0.0077` | `138.24` | `+0.3208 / -0.4376` | `0.1169` / `4.3031°` |
| `v6_1_65d_elec` | **Equivariant 封装** | **`0.0`** | **`0.0°` / `0.0`** | **`216.08`** | **`+0.4054 / -0.4054`** | **`0.0` / `0.0°`** |
| `v6a_51d_pure` | Base（原权重） | `1.9981` | `1.8504°` / `-0.0537` | `145.66` | `-0.0365 / -0.1594` | `0.1960` / `9.8335°` |
| `v6a_51d_pure` | **Equivariant 封装** | **`0.0`** | **`0.0°` / `0.0`** | **`189.59`** | **`+0.0071 / -0.0071`** | **`0.0` / `0.0°`** |

> **关键理论呼应（`2403.04359` Sec. III 实证）**：在确定性零噪仿真直行起步时，精确 $\mathbb{Z}_2$ 封装严格保持于对称不动子空间（`straight_yaw_rms_deg = 0.0°`），机器人自动切换到零偏航双侧对称蠕动推进并将直行速度提升至 `216.1 mm/s`（`v6.1`）与 `189.6 mm/s`（`v6a`）；而在 $\pm 0.20\text{ rad/s}$ 左右转弯指令下，左/右角速度与航向偏差均由最高 `9.83°` 精确归零至 **`0.0°`**，直接印证了 Mittal et al. Sec. III 对中性态与对称任务执行的双重启示。

### 6.2 Benchmark (b): Unseen-OOD 3-Zone Self-Supervised Representation & CPU Latency Parity（未见 OOD 表征消融与单线程 CPU 延迟）

在真实 FP64 三区域轨迹（`120` 帧名义工况训练 / `120` 帧未见 OOD 工况测试：硬地 $+3.0^\circ$ 坡与 $\mu=0.50$、窄管加严至 $15\text{ mm}$ 间隙与 $k_w=3800\text{ N/m}$、软土 $-2.5^\circ$ 坡与滚阻 $0.260$ / 推土 $750\text{ N/m}$）上对比三代估计表征架构（220 步同超参训练）：

| 架构方案 | OOD 62D 下一步传感预测 MSE $\downarrow$ | OOD 地形阻尼 MAE $\downarrow$ | OOD 3-Zone 潜特征线性探针 Acc $\uparrow$ | OOD 联合 23D 探针 Acc $\uparrow$ | OOD 欧氏 Silhouette | 65D 部署参数量 | 单线程 CPU p50 / Mean 延迟（ms） |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`EstNet_v6`（纯监督基线）** | `0.8800` | `0.1213` | `42.5%` (128D) / `65.0%` (7D) | — | `0.2181` (128D) | `257,653` | `0.8283 / 0.9561` |
| **`DreamWaQ_betaVAE`（$\beta$-VAE 重构）** | `0.7528` (`-14.4%`) | `0.0911` (`-24.9%`) | **`70.0%`** ($z_\mu$) | **`74.2%`** | `0.0423` | `229,125` | `1.1235 / 1.1824` |
| **`HIMDreamWaQ_v8`（完整 $\beta$-VAE + HIM）** | **`0.7060` (`-19.8%`)** | **`0.0898` (`-26.0%`)** | `58.3%` ($z_\mu$) | `66.7%` | `0.0477` | **`229,125` (`-11.1%`)** | **`1.1235 / 1.1824`**（51D **`1.0570`**） |

- **关键实证发现 1（消除 $\beta$-VAE 后验塌缩）**：在初版消融中，给 `obs_decoder` 与 `him_src_proj` 保留当前帧/骨干直连捷径会导致模型绕开 $\mathbf{z}_t$ 并诱发后验塌缩；严格按 DreamWaQ Fig. 2 实施信息瓶颈（`obs_decoder(est, z, a)` + `him_src_proj(z, a)`）后，OOD 下一步本体预测 MSE 下降 **`19.8%`**（且完整 `HIMDreamWaQ_v8` 较纯 `DreamWaQ_betaVAE` 再降 **`6.2%`**），OOD 地形阻尼 MAE 下降 **`26.0%`**（`0.1213 -> 0.0898`），OOD 线性可分度较 `EstNet_v6` 骨干提升 **`+15.8` 至 `+31.7` 个百分点**。
- **关键实证发现 2（小样本离线 Silhouette 权衡的诚实归因）**：在仅 120 帧的离线轨迹上，$\mathbf{z}_t$ 需同时编码连续步态相位以达成 $-19.8\%$ 的前向动力学预测精度，因此各向同性欧氏 Silhouette（`0.042–0.048`）低于仅拟合静态标签的 `EstNet_v6`（`0.218`，而在真实 650 帧 `v6b` 三区轨迹上 `EstNet` 7D Silhouette 为 `-0.240`）。这也印证了 HIMLoco（`2312.11460` Sec. 3.4）必须在千环境大规模并行在线批采样（$B \ge 1024 \times 48$）下才能充分发挥批内跨地形对比分离优势。
- **关键实证发现 3（轻量与实时性硬达标）**：通过共享多尺度右对齐因果 TCN，`v8 65D` 部署参数较 `v6 65D` 精简 **`11.1%`**（`229,125` vs `257,653`），单线程 CPU p50 延迟 **`1.124 ms`**（51D **`1.057 ms`**，甚至快于 `v6 51D` 的 `1.088 ms`），叠加精确 $\mathbb{Z}_2$ 等变封装后 `v8_equivariant_65d_batched` 为 **`2.158 ms`**（`exact` 为 `2.437 ms`），仍距 $10\text{ Hz}$ 控制周期（`100 ms`）保有 **`41×–46×`** 的充裕安全余量。

---

## 7. Prioritized Physics, Control & Sim-to-Real Roadmap（分优先级下一阶段研发路线图）

综合 8 篇核心精读文献与 SerpWorm 实机部署节奏，按 **影响-成本比（Impact / Effort）** 排定 P0–P2 三级演进路线：

| 优先级 | 升级项 | 文献理论依据 | 具体工程落点与量化预期收益 | 依赖与风险 |
|---|---|---|---|---|
| **P0（立即可用，零重训）** | **精确 $\mathbb{Z}_2$ `GroupEquivariantPolicyWrapper` 实机部署集成** | Symmetry Considerations (`2403.04359`) + Reynolds 群投影 | 直接封装已训好的 `v6a/v6b/v6.1/v7` 权重，将动作与 EstNet 左右镜像误差严格压至 `0.0`，消除直行偏航漂移与左右转弯响应偏差；批大小 2 合并前向实测单线程 CPU p50 为 `2.16–2.29 ms`（精确双次前向 `2.44 ms`，均 `< 2.5 ms`，距 `100 ms` 10 Hz 预算超 `40×` 余量；单次非等变 `v8` 为 `1.124 ms`）。 | 零风险；仅需注意完全静态无噪仿真的首步中性态微扰。 |
| **P0（本轮已实现，v8 首发）** | **v8 `HIMDreamWaQActorCritic` 完整 RTX 5090 训练（含 `2403.04359` Eq. 6 分母修复 + AdaBoot）** | HIMLoco (`2312.11460`) + DreamWaQ (`2301.10602`) + RMA (`2107.04034`) + Symmetry (`2403.04359`) | 融合 16D $\beta$-VAE 下一步本体重构 + 不确定度门控 + 对称 InfoNCE/SwAV 对比动力学对齐；预期显著压低未见地毯/软土/弯管中的速度与阻力估计 MAE，提升跨区切换响应速度与抗推挤存活率。 | 需待云端 RTX 5090 当前 v7b 训练结束后发起 v8 全量 200-rollout 训练。 |
| **P1（奖励与实机热保护）** | **电机功率均衡正则 + 静摩擦辨识闭环** | DreamWaQ (`2301.10602` Table I) + Static Friction (`2503.01255` Eq. 12–13, Table II–III) | (1) 在 `AutonomousRewardConfig` 中加入 $-\lambda_{\mathrm{pvar}}\mathrm{Var}(P_{\mathrm{elec},j})$ 防止单颗收绳/偏航舵机在长穿管中过热限流；(2) 用 `sysid_toolkit` 对实机 FashionStar 舵机执行正弦扫频辨识实测 $f/\tau_{\max}$，保持 `0.0` 下界欺骗式静摩擦 DR 并按实测校准上界。 | 极低风险，几乎零算力开销。 |
| **P1（物理保真度升级）** | **管壁 & 拉绳导孔切向粘滑记忆（Stick-Slip Bristle）** | Continuum Frictional Contact LCP (`2611.18546`, IJRR 2026) | 在 `pipe_geometry.py` 与 `fused_physics.cu` 的管壁/L 槽角点接触中，用带切向弹簧记忆与 Stribeck 静摩擦的粘滑刷毛模型替代纯粘性 `tanh(|v_t|/0.015)`；为拉绳导孔 Capstan 增加 1D 粘滑状态捕获方向反转滞回，并在尖角处优先采用钢片光滑法向防止法向跳变。 | 需同步更新 PyTorch 与 `fused_physics.cu` 并重跑 FP64 `< 1e-9 m` 一致性测试。 |
| **P2（极端大变形安全）** | **$C^2$ 软惩罚隐式钢片自碰撞 + 薄壳中边弯曲交叉校验** | Py-DiSMech (`2512.09911` Eq. 25–27, Table I) | 针对锐角 90° L 槽与极限 S 弯中相邻 65Mn 钢片可能发生的互碰，采用 Py-DiSMech 的软惩罚平方势能 $E_{\mathrm{con}}(\Delta) = [\frac{1}{K_1}\log(1+e^{K_1(2h-\Delta)})]^2$ 为相邻钢片固定候选对叠加 SPD 隐式对角块；并用中边法向薄壳刚度 $Eh^3/(24(1-\nu^2))$ 独立验证 `strip_thin_shell_compat` 的大扭转精度。 | 中等开发量；需控制候选对数以保持 `> 4000 pol/s` 吞吐。 |
| **P2（软土与低摩擦机理优化）** | **停滞域感知的主动法向力调制与异步双频部署** | Rate-Independent Soft Crawlers (`1710.08340` Remark 2.5, Sec. 4–5) + RMA (`2107.04034` Alg. 2) | 基于停滞域步距律 $\Delta y = \Delta L - 2\mu_+/k$，在软土/低摩擦湿滑地面通过俯仰/拱起调制前后节法向载荷分布（时变耗散 $\mathcal{R}(t, \dot{\mathbf{x}})$）打破锚固失效；若机载嵌入式算力受限，可按 RMA Alg. 2 将轻量化策略头以 $50\text{ Hz}$ 运行、HIM-DreamWaQ 上下文编码异步以 $10\text{ Hz}$ 刷新。 | 属下一代硬件/控制架构探索方向。 |

---

## 8. Verified Citations Index（精读与关联参考文献索引）

所有引文 arXiv 编号与本地 PDF 路径均与 [`ref/README.md`](../../../ref/README.md) 严格一致：

### 8.1 Deep-Read Core Papers（全文逐页精读 8 篇）
1. **Long, J., Wang, Z., Li, Q., Gao, J., Cao, L., & Pang, J. (2024)**, *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response*, **ICLR 2024**, `arXiv:2312.11460` ([PDF](../../../ref/04_actuator_and_sim2real_rl/2312.11460_HIMLoco_Hybrid_Internal_Model_ICLR2024.pdf)).
2. **Nahrendra, I. M. A., Yu, B., & Myung, H. (2023)**, *DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning*, **ICRA 2023**, `arXiv:2301.10602` ([PDF](../../../ref/04_actuator_and_sim2real_rl/2301.10602_DreamWaQ_Implicit_Terrain_Imagination_ICRA2023.pdf)).
3. **Mittal, M., Rudin, N., Klemm, V., Allshire, A., & Hutter, M. (2024)**, *Symmetry Considerations for Learning Task Symmetric Robot Policies*, **ICRA 2024**, `arXiv:2403.04359` ([PDF](../../../ref/04_actuator_and_sim2real_rl/2403.04359_Symmetry_Considerations_Task_Symmetric_Policies_ICRA2024.pdf)).
4. **Hu, X., Sun, Q., He, B., Liu, H., Zhang, X., Lu, C., & Zhong, J. (2025)**, *Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning*, `arXiv:2503.01255` ([PDF](../../../ref/04_actuator_and_sim2real_rl/2503.01255_Static_Friction_Impact_Sim2Real_RL.pdf)).
5. **Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021)**, *RMA: Rapid Motor Adaptation for Legged Robots*, **RSS 2021**, `arXiv:2107.04034` ([PDF](../../../ref/04_actuator_and_sim2real_rl/2107.04034_RMA_Rapid_Motor_Adaptation_RSS2021.pdf)).
6. **Shen, J., Ha, J., & Chen, Y. (2026)**, *Modeling the Frictional Contact of Continuum Robots with Environments through Linear Complementarity Problem*, **IJRR 2026**, `arXiv:2611.18546` ([PDF](../../../ref/03_tendon_capstan_friction_and_hysteresis/2611.18546_Continuum_Frictional_Contact_LCP_IJRR.pdf)).
7. **Lahoti, R., Chaiyakul, R., & Jawed, M. K. (2025)**, *Py-DiSMech: A Scalable and Efficient Framework for Discrete Differential Geometry-Based Modeling and Control of Soft Robots*, `arXiv:2512.09911` ([PDF](../../../ref/02_continuum_solvers_and_beams/2512.09911_PyDiSMech_DDG_Implicit_Contact_Soft_Robots.pdf)).
8. **Gidoni, P. (2017)**, *Rate-independent soft crawlers*, `arXiv:1710.08340` ([PDF](../../../ref/01_ground_contact_and_terramechanics/1710.08340_Rate_Independent_Soft_Crawlers_Anisotropic_Friction.pdf)).

### 8.2 Cross-Referenced Library Context（其余关联库内文献）
- **Actuator / Continuum RL**: Hwangbo et al. (`1901.08652`, *Science Robotics* 2019, ActuatorNet), Hou et al. (`2511.02192`, PyElastica soft RL).
- **Tendon Capstan & Hysteresis**: Kim et al. (`2109.06907`), Cho et al. (`2404.03816`), Zhang et al. (`2608.13396`).
- **Continuum Solvers & Contact**: Li et al. (`2012.04457`, C-IPC), Du et al. (`2101.05917`, DiffPD), Wang et al. (`2402.12315`), Parthasarathy et al. (`2605.13766`, Elastica++), Xie et al. (`1809.05555`), Salagame et al. (`2508.02953`), Hublikar (`2512.05008`), Zhang et al. (`2605.24985`), Ge et al. (`1707.04084`), Liu et al. (`2105.04608`), Arachchige et al. (`2303.02291`), Crotti et al. (`2412.03191`), Javadi et al. (`2510.03660`).
