# `flexgraph-rs`: Theoretical Foundations, SOTA Literature Synthesis, and Pure-Rust Architecture Specification

> **Repository Path**: `rust/flexgraph-rs/`  
> **Precision & Paradigm**: Pure-Rust FP64 (`f64`) Bipartite Rigid-Flexible Continuum Physics Engine with Adaptive Sparse Block-$LDL^\top$, Dual Continuum Formulations (Corotational Euler-Bernoulli/Timoshenko + Discrete Elastic Rods), Composable 3D SDF World, and Unified Contact Solvers (C-IPC Log-Barrier + Py-DiSMech $C^2$ Self-Contact + Compliance-Based Frictional LCP + Implicit ARPC).  
> **Backward Parity Contract**: Strict `< 1e-10` FP64 position, tension, and contact-force parity with `serpworm.physics.gpu_core.GPUCore` (`segment_lu` CPU / `fused_cuda` GPU reference) on `SerpWorm-N` (`N=1,3,5,7`), without modifying a single file under `src/serpworm/`.

---

## 1. SOTA Literature Positioning & Architectural Motivation

Modern rigid-flexible and soft continuum robotics spans high-aspect-ratio spring-steel ribbons ($w/t \approx 70\text{--}110$, Young's modulus $E = 200\text{ GPa}$), superelastic Nitinol catheters, wheelless anisotropic bristle crawlers, compliant-spine legged robots, and closed-loop tensegrity metamaterials. Existing simulation platforms split across five disconnected lineages, none of which simultaneously provides **FP64 stiff-continuum exactness**, **cyclic closed-loop rigid-flexible coupling**, **intersection-free codimensional contact**, **anisotropic terramechanics**, and **zero-overhead native RL bindings**:

| Engine / Lineage | Continuum / Structural Model | Rigid-Soft Coupling & Linear Solver | Contact & Friction Formulation | Primary Gap Addressed by `flexgraph-rs` |
| :--- | :--- | :--- | :--- | :--- |
| **SOFA / BeamAdapter** (Duriez & Cotin, 2006–2024) | Corotational cubic-Hermite Euler-Bernoulli 1D FEM | Generic CPU sparse assembly (`CG` / `LDL^T`) | Penalty / compliance LCP | Heavy C++ runtime, low multi-env RL throughput (`0.1–1.5 sps`), no built-in ARPC/DER/C-IPC |
| **C-IPC / Rigid-IPC / ABD** (Li et al. SIGGRAPH 2021; Lan et al. SIGGRAPH 2022) | Volumetric FEM + codimensional rods/shells + affine stiff bodies | Projected Newton + barrier line search + CCD | $C^2$ clamped log-barrier + smoothed mollifier friction (guaranteed zero penetration) | Graphics-oriented; lacks articulated motor/tendon loops, Bekker soil rutting, and ratchet wheel stiction |
| **DisMech / Py-DiSMech** (Bergou et al. 2008/2010; Jawed 2018; Lahoti et al. `arXiv:2512.09911`, 2025) | Discrete Differential Geometry (DER rods + mid-edge shells) | Implicit Euler / Newmark-$\beta$ Newton (dense/banded NumPy/C++) | $C^2$ soft-penalty implicit rod–rod / rod–shell contact (`IMC`) | Rod/shell focused; lacks general bipartite rigid-body Schur routing, Capstan LCP tendons, and anisotropic patch wheels |
| **Elastica++ / Cosserat-ABA** (Renda et al. T-RO 2018; Parthasarathy et al. `arXiv:2605.13766`, 2026) | Continuous/discrete Cosserat rods (PCS / finite-diff) | Explicit Verlet or serial $O(N)$ recursive ABA | Penalty / hash-grid rod contact | Explicit schemes require $\Delta t \sim 1\,\mu\text{s}$ at $E=200\text{ GPa}$; serial ABA cannot natively close cyclic rigid-flexible loops (e.g., tensegrity) |
| **Compliance-Based Frictional LCP & Moreau NSCD** (Shen et al. `arXiv:2611.18546`, IJRR 2026; Salagame `2508.02953`; Hublikar `2512.05008`) | Cosserat rod + compliance operator $C_c = J_u K^{-1} J_u^\top$ | Quasi-static / Moreau time-stepping complementarity | Exact Signorini–Coulomb stick-slip hysteresis (`-87.1%` tip error vs frictionless) | Standalone continuum solver without unified 3D SDF world, rigid-body multi-loop dynamics, or C-IPC barrier |
| **Genesis / NVIDIA Newton (Warp) / Dojo / Rapier** (Howell et al. 2022; NVIDIA/DeepMind 2025; Dimforge) | MPM / FEM / PBD / rigid bodies (FP32 GPU or Rust rigid-only) | GPU kernels / primal-dual SOCP interior-point (Dojo) / PGS (Rapier) | Hard SOCP (Dojo), penalty/PBD (Genesis/Newton), rigid impulses (Rapier) | FP32 suffers ill-conditioning on $0.15\text{--}0.22\text{ mm}$ steel strips ($E=200\text{ GPa}$, condition number $\kappa > 10^{11}$); Rapier has no continuum beams |
| **`flexgraph-rs` (This Work)** | **Dual FP64 Continuum**: Corotational Euler-Bernoulli/Timoshenko (+ Kirchhoff-Love/Vlasov) **&** DER/Cosserat Bishop-Frame Rod | **Bipartite Direct Solver**: Parallel 6×6 Block-Thomas + Adaptive Rigid Schur (`BlockDiagonalPairs` / `TreeChain` $O(N_b)$ / `SparseGeneral` Symbolic Block-$LDL^\top$) | **Tri-Engine Unified Contact**: (A) C-IPC Log-Barrier + CCD + Py-DiSMech $C^2$, (B) Delassus-Schur Frictional LCP, (C) Implicit ARPC (Ratchet/Scale/Pad + Bekker Rutting) | **Unifies all five lineages in memory-safe, zero-GC, Rayon-parallel pure Rust FP64 with C-ABI Python parity `< 1e-10`** |

---

## 2. Exact Mathematical Formulation

### 2.1 Conventions, Lie Groups $SO(3)/SE(3)$, and Spatial Plücker Algebra (`fg-math`)

To maintain exact algebraic parity with `serpworm.physics.gpu_core` while supporting full Featherstone spatial dynamics, **`flexgraph-rs` enforces two global conventions across all 8 crates**:
1. **6D Spatial Ordering**: Every spatial velocity (twist) $\mathbf{v} = [\mathbf{v}_{\mathrm{lin}};\,\boldsymbol{\omega}] \in \mathbb{R}^6$ and spatial force (wrench) $\mathbf{f} = [\mathbf{f}_{\mathrm{lin}};\,\boldsymbol{\tau}] \in \mathbb{R}^6$ places **linear components in `0..3`** and **angular components in `3..6`**.
2. **Quaternion Layout**: Unit quaternions are stored in `[x, y, z, w]` (`[vector; scalar]`) with canonical hemisphere $w \ge 0$ upon logarithmic extraction.

#### $SO(3)$ Exponential, Smooth Logarithm, and Inverse Left Jacobian
For $\boldsymbol{\phi} \in \mathfrak{so}(3)$ with $\theta = \|\boldsymbol{\phi}\|_2$:
$$
\exp_{\mathrm{SO}(3)}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\phi}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times^2, \qquad
\mathbf{q}(\boldsymbol{\phi}) = \begin{bmatrix} \frac{\sin(\theta/2)}{\theta/2}\,\frac{\boldsymbol{\phi}}{2} \\ \cos(\theta/2) \end{bmatrix}.
$$
Near $\theta \to 0$ ($\|\mathbf{q}_v\|^2 < 10^{-12}$), `Quat::log` uses the exact 5th-order Taylor expansion so the derivative at identity is strictly $2\mathbf{I}$ (matching `stable_logq` in `gpu_core.py`):
$$
\log_{\mathrm{SO}(3)}(\mathbf{q}) =
\begin{cases}
\left(\dfrac{2}{w} - \dfrac{2n^2}{3w^3} + \dfrac{2n^4}{5w^5}\right)\mathbf{q}_v, & n^2 < 10^{-12}, \\[8pt]
\dfrac{2\operatorname{atan2}(n, w)}{n}\,\mathbf{q}_v, & \text{otherwise},
\end{cases}
\qquad n = \|\mathbf{q}_v\|_2.
$$
The left Jacobian $\mathbf{J}_l(\boldsymbol{\phi}) = d\exp_{\boldsymbol{\phi}}$ and its exact inverse $d\exp^{-1}_{\boldsymbol{\phi}}$ (satisfying $\log(\exp(\boldsymbol{\epsilon})\exp(\boldsymbol{\phi})) = \boldsymbol{\phi} + d\exp^{-1}_{\boldsymbol{\phi}}\boldsymbol{\epsilon} + O(\|\boldsymbol{\epsilon}\|^2)$) are:
$$
\mathbf{J}_l(\boldsymbol{\phi}) = \mathbf{I} + \frac{1-\cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times + \frac{\theta-\sin\theta}{\theta^3}[\boldsymbol{\phi}]_\times^2, \qquad
d\exp^{-1}_{\boldsymbol{\phi}} = \mathbf{I} - \frac{1}{2}[\boldsymbol{\phi}]_\times + c(\theta)\,[\boldsymbol{\phi}]_\times^2,
$$
where $c(\theta) = \frac{1 - \frac{\theta}{2}\cot(\frac{\theta}{2})}{\theta^2} = \frac{1}{12} + \frac{\theta^2}{720} + \frac{\theta^4}{30240} + O(\theta^6)$ for $\theta^2 < 10^{-8}$.

#### $SE(3)$ Maps and Spatial Plücker Algebra
For twist $\boldsymbol{\xi} = [\mathbf{v};\,\boldsymbol{\omega}] \in \mathfrak{se}(3)$:
$$
\exp_{\mathrm{SE}(3)}(\boldsymbol{\xi}) = \big(\mathbf{J}_l(\boldsymbol{\omega})\mathbf{v},\; \exp_{\mathrm{SO}(3)}(\boldsymbol{\omega})\big), \qquad
\log_{\mathrm{SE}(3)}(\mathbf{p}, \mathbf{q}) = \begin{bmatrix} d\exp^{-1}_{\boldsymbol{\omega}}\mathbf{p} \\ \boldsymbol{\omega} \end{bmatrix}, \quad \boldsymbol{\omega} = \log_{\mathrm{SO}(3)}(\mathbf{q}),
$$
with exact $6\times 6$ inverse left Jacobian $d\exp^{-1}_{\boldsymbol{\xi}} = \begin{bmatrix} \mathbf{J}_l^{-1}(\boldsymbol{\omega}) & -\mathbf{J}_l^{-1}(\boldsymbol{\omega})\mathbf{Q}(\mathbf{v},\boldsymbol{\omega})\mathbf{J}_l^{-1}(\boldsymbol{\omega}) \\ \mathbf{0} & \mathbf{J}_l^{-1}(\boldsymbol{\omega}) \end{bmatrix}$ (Barfoot, Eq. 7.86).  
The spatial adjoint, Plücker motion/force transforms, spatial cross products, and body spatial inertia in `[linear; angular]` ordering are:
$$
\operatorname{Ad}_{(\mathbf{R},\mathbf{p})} = \begin{bmatrix} \mathbf{R} & [\mathbf{p}]_\times \mathbf{R} \\ \mathbf{0} & \mathbf{R} \end{bmatrix}, \quad
\mathbf{X}_M(\mathbf{R}_r^\top, \mathbf{o}) = \begin{bmatrix} \mathbf{R}_r^\top & -\mathbf{R}_r^\top[\mathbf{o}]_\times \\ \mathbf{0} & \mathbf{R}_r^\top \end{bmatrix}, \quad
\mathbf{X}_F = \mathbf{X}_M^\top,
$$
$$
\mathbf{v}\times = \begin{bmatrix} [\boldsymbol{\omega}]_\times & [\mathbf{v}_{\mathrm{lin}}]_\times \\ \mathbf{0} & [\boldsymbol{\omega}]_\times \end{bmatrix}, \qquad
\mathbf{v}\times^* = -(\mathbf{v}\times)^\top = \begin{bmatrix} [\boldsymbol{\omega}]_\times & \mathbf{0} \\ [\mathbf{v}_{\mathrm{lin}}]_\times & [\boldsymbol{\omega}]_\times \end{bmatrix}, \qquad
\mathbf{I}_b = \begin{bmatrix} m\mathbf{I}_3 & -m[\mathbf{c}]_\times \\ m[\mathbf{c}]_\times & \mathbf{I}_c - m[\mathbf{c}]_\times^2 \end{bmatrix}.
$$

---

### 2.2 Dual Continuum Formulations

#### Formulation A: Cubic-Hermite Corotational 3D Beam (`fg_math::beam`)
Given an element of rest length $L$ connecting world-frame clamped endpoints $(\mathbf{p}_0, \mathbf{q}_0)$ and $(\mathbf{p}_3, \mathbf{q}_1)$ with body-origin clamp offsets $(\mathbf{o}_0, \mathbf{o}_1)$, we form the cubic Bézier/Hermite control polygon $\mathbf{p}_1 = \mathbf{p}_0 + \frac{L}{3}\mathbf{t}_0$, $\mathbf{p}_2 = \mathbf{p}_3 - \frac{L}{3}\mathbf{t}_1$ ($\mathbf{t}_k = \mathbf{q}_k \mathbf{e}_x$), midpoint center $\mathbf{c} = \frac{1}{8}(\mathbf{p}_0 + 3\mathbf{p}_1 + 3\mathbf{p}_2 + \mathbf{p}_3)$, and midpoint tangent $\mathbf{t} = \operatorname{unit}(-\frac{3}{4}\mathbf{p}_0 - \frac{3}{4}\mathbf{p}_1 + \frac{3}{4}\mathbf{p}_2 + \frac{3}{4}\mathbf{p}_3)$. The midpoint orientation $\mathbf{r}$ aligns the average of $\operatorname{align}(\mathbf{q}_0, \mathbf{t})$ and $\operatorname{align}(\mathbf{q}_1, \mathbf{t})$ to $\mathbf{t}$ (with compressed-buckle fallback $(\frac{\mathbf{p}_0+\mathbf{p}_3}{2}, \mathbf{q}_0)$ when $(\mathbf{p}_1-\mathbf{p}_0)\cdot(\mathbf{p}_2-\mathbf{p}_1) < 0$ and $\|\mathbf{p}_3-\mathbf{p}_0\| < 0.4L$, matching `gpu_core.py:59-70`).

The local 12D deformation vector $\mathbf{u} \in \mathbb{R}^{12}$ and elastic potential energy are:
$$
\mathbf{u} = \begin{bmatrix} \mathbf{R}_r^\top(\mathbf{p}_0 - \mathbf{c}) \\ \log_{\mathrm{SO}(3)}(\mathbf{r}^{-1}\mathbf{q}_0) \\ \mathbf{R}_r^\top(\mathbf{p}_3 - \mathbf{c}) \\ \log_{\mathrm{SO}(3)}(\mathbf{r}^{-1}\mathbf{q}_1) \end{bmatrix}, \qquad
U_{\mathrm{beam}} = \frac{1}{2}(\mathbf{u} - \mathbf{u}_0)^\top \mathbf{K}_{\mathrm{loc}}(\mathbf{u} - \mathbf{u}_0).
$$
Both the fast frozen-frame Plücker transform $\mathbf{T} = \operatorname{blkdiag}(\mathbf{X}_M(\mathbf{R}_r^\top, \mathbf{o}_0), \mathbf{X}_M(\mathbf{R}_r^\top, \mathbf{o}_1))$ (for `< 1e-10` parity with `gpu_core.py`) **and** the **exact analytic strain Jacobian** $\mathbf{J}_{\mathrm{strain}} = \frac{\partial \mathbf{u}}{\partial \mathbf{q}_{\mathrm{world}}} \in \mathbb{R}^{12\times 12}$ (differentiating through $\mathbf{c}$, $\mathbf{r}$, and $d\exp^{-1}(\boldsymbol{\phi}_k)$) are implemented in `fg_math::beam::beam_strain_jacobian`:
$$
\mathbf{J}_{p,k} = \mathbf{R}_r^\top\big(\mathbf{P}_k - \mathbf{C} + [\mathbf{p}_k - \mathbf{c}]_\times \mathbf{B}\big), \qquad
\mathbf{J}_{r,k} = d\exp^{-1}_{\boldsymbol{\phi}_k}\,\mathbf{R}_r^\top(\mathbf{W}_k - \mathbf{B}),
$$
where $\mathbf{B} = \frac{\partial \boldsymbol{\omega}_r}{\partial \mathbf{q}_{\mathrm{world}}}$ and $\mathbf{C} = \frac{\partial \mathbf{c}}{\partial \mathbf{q}_{\mathrm{world}}}$.

**Heterogeneous Cross-Sections, Kirchhoff-Love/Vlasov Thin-Shell & Timoshenko Shear Corrections**:  
`local_beam_matrices` supports `Rectangular`, `Circular`, `HollowCircular`, and `General` sections with:
- Exact Saint-Venant rectangular torsion $J = w t^3\big(\frac{1}{3} - 0.21\frac{t}{w}(1 - \frac{t^4}{12 w^4})\big)$;
- Optional **Kirchhoff-Love plate flexural stiffening** $E_y = \frac{E}{1-\nu^2}$ on out-of-plane bending $I_y$ and **Vlasov restrained-warping torsional factor** $J_{\mathrm{eff}} = J\big(1 + \frac{w^2}{12\cdot 0.145^2(1-\nu)}\big)$ for high-$w/t$ spring-steel strips (`thin_shell_compat = true`);
- Optional **Przemieniecki / Timoshenko shear-deformation** factors $\Phi_y = \frac{12 E I_z}{G A k_y L^2}$, $\Phi_z = \frac{12 E_y I_y}{G A k_z L^2}$ in $\mathbf{K}_{\mathrm{loc}}$;
- Exact consistent mass matrix $\mathbf{M}_{\mathrm{loc}}$ with rotary inertia terms $\frac{6 I_{y,z}}{5 A L^2}, \frac{2 I_{y,z}}{15 A}$.

#### Formulation B: Discrete Elastic Rods / Cosserat Rod (`fg_math::cosserat`)
Following Bergou et al. (SIGGRAPH 2008 / 2010), Jawed et al. (2018), and Py-DiSMech (`arXiv:2512.09911`, 2025), a slender rod with vertices $\{\mathbf{x}_i\}_{i=0}^E$, edges $\mathbf{e}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$, unit tangents $\mathbf{t}_k = \mathbf{e}_k / \|\mathbf{e}_k\|$, and edge twist angles $\{\theta_k\}_{k=0}^{E-1}$ maintains a twist-free **Bishop frame** $\{\mathbf{t}_k, \mathbf{u}_k, \mathbf{v}_k\}$ updated via closed-form minimal-rotation parallel transport:
$$
\mathcal{P}_{\mathbf{t}_a \to \mathbf{t}_b}(\mathbf{u}) = \mathbf{u} - \frac{\mathbf{u}\cdot \mathbf{t}_b}{1 + \mathbf{t}_a\cdot \mathbf{t}_b}(\mathbf{t}_a + \mathbf{t}_b), \qquad
\begin{bmatrix}\mathbf{m}_{1,k} \\ \mathbf{m}_{2,k}\end{bmatrix} = \begin{bmatrix}\cos\theta_k & \sin\theta_k \\ -\sin\theta_k & \cos\theta_k\end{bmatrix}\begin{bmatrix}\mathbf{u}_k \\ \mathbf{v}_k\end{bmatrix}.
$$
At each interior hinge $i$ (Voronoi length $\bar{l}_i = \frac{1}{2}(\bar{l}_{i-1}+\bar{l}_i)$, denominator $\chi_i = 1 + \mathbf{t}_{i-1}\cdot\mathbf{t}_i$), the discrete curvature binormal, material curvatures, and holonomy-aware twist are:
$$
(\kappa\mathbf{b})_i = \frac{2\,\mathbf{t}_{i-1}\times \mathbf{t}_i}{\chi_i}, \qquad
\kappa_i^{(1)} = \frac{1}{2}(\mathbf{m}_{2,i-1}+\mathbf{m}_{2,i})\cdot(\kappa\mathbf{b})_i, \qquad
\kappa_i^{(2)} = -\frac{1}{2}(\mathbf{m}_{1,i-1}+\mathbf{m}_{1,i})\cdot(\kappa\mathbf{b})_i,
$$
$$
m_i = \theta_i - \theta_{i-1} + m_i^{\mathrm{ref}}, \qquad m_i^{\mathrm{ref}} = \angle_{\mathbf{t}_i}\big(\mathcal{P}_{\mathbf{t}_{i-1}\to\mathbf{t}_i}(\mathbf{u}_{i-1}),\; \mathbf{u}_i\big).
$$
The total elastic energy decomposes into stretching, anisotropic bending, and twisting potentials:
$$
E_{\mathrm{DER}} = \sum_{k=0}^{E-1} \frac{1}{2} E A \bar{l}_k \left(\frac{\|\mathbf{e}_k\|}{\bar{l}_k} - 1\right)^2
+ \sum_{i=1}^{E-1} \frac{1}{2\bar{l}_i}(\boldsymbol{\kappa}_i - \bar{\boldsymbol{\kappa}}_i)^\top \operatorname{diag}(EI_1, EI_2)(\boldsymbol{\kappa}_i - \bar{\boldsymbol{\kappa}}_i)
+ \sum_{i=1}^{E-1} \frac{GJ}{2\bar{l}_i}(m_i - \bar{m}_i)^2.
$$
`fg_math::cosserat` provides the **exact closed-form 6D stretching gradient/Hessian** and **exact closed-form 11-DOF bending and twisting gradients and analytical 11×11 Hessians** under time-parallel frame transport, verified against central finite differences to $< 10^{-6}$.

---

### 2.3 Bipartite Direct Solver, Implicit Lie-Group Integration & Adaptive Rigid Schur Routing (`fg-ir` + `fg-solver`)

Partitioning generalized velocities into free rigid-body twists $\mathbf{v}_R \in \mathbb{R}^{6 N_{\mathrm{free}}}$ and free beam-node twists $\mathbf{v}_F \in \mathbb{R}^{6 N_{\mathrm{beam}}}$ yields an exact bipartite arrow-block linear system at every linearly implicit Euler or damped Newton-Raphson / Newmark-$\beta$ / BDF2 iteration:
$$
\begin{bmatrix}
\mathbf{H}_{RR} & \mathbf{H}_{RF} \\
\mathbf{H}_{FR} & \mathbf{H}_{FF}
\end{bmatrix}
\begin{bmatrix} \mathbf{v}_R \\ \mathbf{v}_F \end{bmatrix}
=
\begin{bmatrix} \mathbf{b}_R \\ \mathbf{b}_F \end{bmatrix},
\qquad
\mathbf{H}_{FF} = \operatorname{blkdiag}(\mathbf{T}_1, \dots, \mathbf{T}_{N_s}),
$$
where for each beam $s$, $\mathbf{A}_{s,e} = (1 + h\alpha_R)\mathbf{M}_{s,e} + (h\beta_R + h^2)\mathbf{K}_{s,e}$ (linearly implicit Euler) or $\mathbf{M} + \gamma h \mathbf{C} + \beta h^2 \mathbf{K}$ (Newmark-$\beta$ / Newton tangent), and diagonal interior-node contact Hessians $\mathbf{K}_{\mathrm{con},i} \in \mathbb{R}^{3\times 3}_{\succeq 0}$ (ground, pipe SDF, or Py-DiSMech $C^2$ self-contact) enter strictly into the diagonal $6\times 6$ blocks of $\mathbf{T}_s$, preserving exact block-tridiagonality!

1. **Stage 1 — Parallel Per-Beam 6×6 Block-Thomas Condensation**:  
   Each beam $s$ eliminates its interior nodes in $O(E_s)$ operations:
   - **Clamped-clamped** (`body_b = Some(b)`): solves $1 + 12$ RHS columns ($[\mathbf{b}_F, \mathbf{H}_{FR}]$) to produce exact condensed $12\times 12$ Schur matrix $\mathbf{S}_s$ and $12\times 1$ RHS $\mathbf{r}_s$ on $(a_s, b_s)$ (Dirichlet fixed bases simply omit their rows/columns).
   - **Cantilever free tip** (`body_b = None`): eliminates all $E_s$ free nodes including the free tip against $1 + 6$ RHS columns, returning a single exact $6\times 6$ Schur contribution $\mathbf{S}_{aa}$ and $6\times 1$ RHS $\mathbf{r}_a$ on `body_a`.
2. **Stage 2 — Compile-Time Adaptive Rigid Schur Router (`fg_ir::analysis`)**:  
   Assembling $\mathbf{H}_{RR}^{\mathrm{eff}} = \mathbf{H}_{RR} + \sum_s \mathbf{S}_s + \sum_c \mathbf{K}_{\mathrm{contact},c} + \sum_j \mathbf{K}_{\mathrm{joint},j}$, `fg-ir` analyzes $\mathcal{G}_R$ once at load time and selects:
   - **`SchurRoute::BlockDiagonalPairs`**: all connected components have size $\le 2$ (e.g., `SerpWorm-N` with explicit inter-segment yaw) $\implies$ batched $12\times 12$ / $6\times 6$ Cholesky/LU, exact parity with `segment_lu`.
   - **`SchurRoute::TreeChain`**: acyclic ($\beta_1 = E - V + C = 0$) with max component size $\ge 3$ (e.g., `TendonEndoscope-3D`, `FlexSpine-Centipede`) $\implies$ $O(N_b)$ post-order Block-Tree $LDL^\top$ with zero fill-in.
   - **`SchurRoute::SparseGeneral`**: cyclic ($\beta_1 > 0$, e.g., `Tensegrity-6Bar` icosahedral 6-strut closed loop with $\beta_1 = 7$) $\implies$ Minimum-Degree fill-reducing permutation $\mathbf{P}$, Liu's elimination tree (`etree`) post-ordering, and Tim Davis `ereach` exact symbolic lower-triangular block-CSC factorization $\mathbf{P}\mathbf{H}_{RR}^{\mathrm{eff}}\mathbf{P}^\top = \mathbf{L}\mathbf{D}\mathbf{L}^\top$ over $6\times 6$ blocks.
3. **Stage 3 — Parallel $O(E)$ Back-Substitution & Lie-Group Pose Update**:  
   Back-substitutes $\mathbf{v}_F$ per beam in parallel and updates poses via exact unit-quaternion exponential integration (`Se3::integrate_decoupled` for `<1e-10` `GPUCore` parity; `Se3::integrate_lie_body` + Armijo damped Newton line search for large-step $\Delta t = 1\text{--}5\text{ ms}$ nonlinear stability). Multi-environment batch stepping executes across CPU cores via `rayon`.

---

### 2.4 Composable 3D SDF World (`fg-sdf`) & Unified Tri-Engine Contact Mechanics (`fg-contact`)

#### Composable 3D SDF Query Contract (`fg-sdf`)
Every environment primitive (`HeightFieldSdf`, `SplineTubeSdf` with 3D cubic B-spline / Catmull-Rom centerline and variable $r(s)$, `BoxSdf`, `CylinderSdf`, `SphereSdf`, trilinear `VoxelGridSdf`, and CSG `Union` / `SmoothUnion(k)` / `Difference` / `Intersection`) implements one FP64 trait returning:
$$
\operatorname{query}(\mathbf{x}) \;\longmapsto\; \texttt{SdfQuery}\{\;\phi(\mathbf{x}),\; \mathbf{n}(\mathbf{x}),\; s_{\mathrm{arc}}(\mathbf{x}),\; \mathbf{t}_{\mathrm{arc}}(\mathbf{x}),\; \mathrm{mat\_id}\;\},
$$
where $\phi > 0$ outside the solid, $\phi < 0$ inside (penetration $\delta = \max(0, -\phi)$), and $\mathbf{n} = \nabla\phi / \|\nabla\phi\|$ points outward into free space.

#### Tri-Engine Contact Architecture (`fg-contact`)
1. **Engine A — Codimensional IPC Log-Barrier + Py-DiSMech $C^2$ Self-Contact + Conservative CCD**:
   - $O(N)$ spatial hash grid broadphase detecting point–point and **segment–segment (edge–edge)** proximity pairs.
   - **C-IPC $C^2$ Clamped Log-Barrier** (Li et al. SIGGRAPH 2020/2021, `arXiv:2012.04457`) for distance $d \in (0, \hat{d})$:
     $$
     b(d; \hat{d}) = -(d - \hat{d})^2 \ln\!\left(\frac{d}{\hat{d}}\right), \qquad
     b'(d) = -2(d - \hat{d})\ln\!\left(\frac{d}{\hat{d}}\right) - \frac{(d - \hat{d})^2}{d}, \qquad
     b''(d) = -2\ln\!\left(\frac{d}{\hat{d}}\right) - \frac{3(d-\hat{d})}{d} + \frac{(d-\hat{d})^2}{d^2} > 0,
     $$
     paired with conservative **Continuous Collision Detection (CCD) Time-of-Impact (TOI)** step clamping ($t_{\max} \in (0, 1]$) guaranteeing strictly intersection-free trajectories, **SPD eigenvalue projection** (`Mat3::project_spd` / `Mat6::project_spd`) of non-convex curvature terms, and IPC $C^1$ mollified Coulomb friction $f_1(\|\mathbf{u}_t\|; \epsilon_v)$.
   - **Py-DiSMech $C^2$ Soft-Penalty Self-Contact** (`arXiv:2512.09911`, Eq. 25–27):
     $$
     E_{\mathrm{con}}(\Delta) =
     \begin{cases}
     (2h - \Delta)^2, & \Delta \le 2h - \delta, \\
     0, & \Delta \ge 2h + \delta, \\
     \left[\frac{1}{K_1}\ln\!\left(1 + e^{K_1(2h-\Delta)}\right)\right]^2, & \text{otherwise},
     \end{cases}
     \quad K_1 = \frac{15}{\delta},
     $$
     allowing line-search-free implicit assembly into $\mathbf{T}_s$ diagonal blocks.
2. **Engine B — Compliance-Based Frictional Contact LCP / Delassus-Schur PGS (Shen et al. `arXiv:2611.18546`, IJRR 2026 & Moreau NSCD)**:
   Given the contact compliance (Delassus) matrix $\mathbf{C}_c = \mathbf{J}_c \mathbf{H}^{-1} \mathbf{J}_c^\top$ and free contact velocity $\mathbf{v}_c^{\mathrm{free}}$, solves the exact Signorini–Coulomb complementarity problem:
   $$
   0 \le \mathbf{v}_{n,i} \perp \lambda_{n,i} \ge 0, \qquad
   \boldsymbol{\lambda}_{t,i} \in \arg\min_{\|\boldsymbol{\lambda}_t\| \le \mu_i \lambda_{n,i}} \mathbf{v}_{t,i}^\top \boldsymbol{\lambda}_t
   $$
   via Projected Gauss-Seidel / Alart-Curnier proximal projection onto the second-order Coulomb friction cone $\mathcal{K}_{\mu_i}$, capturing multi-point stick-slip hysteresis and elastic snap-through.
3. **Engine C — Unified Implicit Anisotropic Patch Contact (ARPC)**:
   Unifies `RatchetWheel` (exact parity with `serpworm.physics.wheel_stiction`: wheel spin inertia $I_w = \frac{1}{2}m_w r_w^2$, rolling resistance, pawl backlash $\psi \in [0, \psi_{\max}]$, tangential spring-damper memory), `AnisotropicScale` ($\mu_{\mathrm{fwd}} \ll \mu_{\mathrm{rev}}, \mu_{\mathrm{lat}}$, grounded in Gidoni `arXiv:1710.08340` stasis-domain theory), and `IsotropicPad` under Hunt-Crossley + Bekker elasto-plastic soil rutting:
   $$
   F_{\mathrm{el}} = \begin{cases} k_{\mathrm{hc}}\delta^p & (\text{loading}) \\ \gamma_{\mathrm{un}} k_{\mathrm{hc}}(\delta - \delta_{\mathrm{rut}})\delta^{p-1} & (\text{unloading}) \end{cases}, \qquad
   F_n = F_{\mathrm{el}}\max(0, 1 - \alpha_{\mathrm{hc}} v_n),
   $$
   with Stribeck friction ellipse $\mu(\|\mathbf{v}_{\mathrm{slip}}\|) = \mu_k\big(1 + (r_s - 1)e^{-(\|\mathbf{v}_{\mathrm{slip}}\|/0.03)^2}\big)$, circular patch pivot torque $\tau_z = -\frac{2}{3}\mu_{\mathrm{piv}} r_p F_n \tanh(\omega_z/0.15)$, and exact SPD $6\times 6$ Contact-Schur tangent $\mathbf{K}_{c,6} = \sum_{d\in\{0,1,2\}} w_d \mathbf{j}_d \mathbf{j}_d^\top + h c_{\mathrm{piv}} \mathbf{e}_5\mathbf{e}_5^\top$.

---

## 3. Workspace Architecture & Crate-Boundary Contracts

```
rust/flexgraph-rs/
├── Cargo.toml                  # 8-crate workspace root (edition 2021, FP64)
└── crates/
    ├── fg-math/    [@lit-architect]    Pure-Rust FP64 Vec/Mat, SO(3)/SE(3), Plücker, Corotational & DER math
    ├── fg-ir/      [@lit-architect]    RigidFlexGraph IR, JSON/YAML serde, Symmetry groups, Sparsity/etree analyzer
    ├── fg-sdf/     [@contact-ipc]      Composable 3D SDF world, HeightField, SplineTube, Primitives, VoxelGrid, CSG
    ├── fg-contact/ [@contact-ipc]      C-IPC log-barrier + CCD, Py-DiSMech C2, Compliance LCP/PGS, Unified ARPC
    ├── fg-solver/  [@continuum-solver] Implicit Lie-Group Euler & Damped Newton/Newmark/BDF2, Dual Beam/DER, 3-route Schur
    ├── fg-robots/  [@bindings-bench]   JSON/YAML/URDF-Flex loader + 5 calibrated robot archetypes
    ├── fg-ffi/     [@bindings-bench]   C-ABI cdylib + zero-dependency Python ctypes wrapper (`src/flexgraph_rs/`)
    └── fg-bench/   [@bindings-bench]   <5s Rust single-thread & Rayon batch + Python-vs-Rust FP64 parity benchmark
```

### Key Public Contracts Provided by `fg-math` & `fg-ir`

- **`fg_math`**:
  - Linear algebra: `Vec3`, `Vec6`, `Vec12`, `Mat3` (incl. `sym_eigen`, `project_spd`), `Mat6` (incl. `solve_lu`, `inv_lu`, `cholesky`, `solve_cholesky`, `sym_eigen`, `project_spd`), `Mat12` (incl. `congruent_transform`, `solve_lu`, `pack_beam_16`).
  - Lie groups: `Quat` (`[x,y,z,w]`, `exp`, `log`, `log_legacy`, `align_x_to`, `from_two_unit_vectors`), `Se3` (`exp`, `log`, `integrate_decoupled`, `integrate_lie_body`, `integrate_lie_spatial`), `so3_exp/log/dexp/dexp_inv`, `se3_exp/log/dexp_inv`.
  - Spatial algebra: `SpatialInertia`, `adjoint_se3`, `coadjoint_se3`, `plucker_motion_transform`, `plucker_force_transform`, `spatial_cross_motion`, `spatial_cross_force`, `point_force_to_wrench`, `point_wrench_jacobian`.
  - Continuum beams: `BeamSection`, `BeamMaterial`, `CorotationalFrame`, `cubic_hermite_midpoint`, `beam_local_strain`, `beam_frame_transform`, `beam_strain_jacobian`, `local_beam_matrices`.
  - Cosserat/DER: `parallel_transport`, `compute_bishop_frames`, `material_frame`, `reference_twist`, `curvature_binormal`, `material_curvature`, `der_stretching_energy_grad_hess`, `der_bending_energy_grad_hess`, `der_twisting_energy_grad_hess`.
  - MuJoCo soft-constraint math (v2): `SolRefImp` (`from_mujoco`, `direct`, `impedance`, `impedance_and_deriv`, `kbip`, `to_stiffness_damping`, `to_stiffness_damping_dt`, `to_force_stiffness_damping`, `constraint_regularizer`, `reference_accel`).
- **`fg_ir`**:
  - Schema: `RigidFlexGraph` (`Default`, `validate`, `analyze`, `to_json_string`, `from_json_str`, `to_yaml_string`, `from_yaml_str`, plus v2 `diff_params: Vec<DiffParamSpec>` and `mesh_colliders: Vec<MeshColliderSpec>`), `RigidBodySpec` (`is_fixed`, `spatial_inertia`), `BeamEdgeSpec` (`body_b: Option<usize>` for cantilever tips, `BeamFormulation`), `TendonEdgeSpec` (`waypoints`, `TendonDriveMode`, `solref_imp: Option<SolRefImp>`), `JointEdgeSpec` (`JointType`, `implicit_coupling`, `solref_imp: Option<SolRefImp>`), `ContactSiteSpec` (`ContactAttachment`, `ContactMode`, `solref_imp: Option<SolRefImp>`), `DiffParamSpec`/`DiffParamKind` (7 parameter families), `MeshColliderSpec`/`ColliderGeometry` (TriMesh, ConvexHull, Sphere, Box, Cylinder, Capsule, ExternalMesh), `SymmetryGroupSpec` (`Z2Bilateral`, `Z2xZ2Bidirectional`, `C4Cyclic`, `D4Dihedral`, `from_generators`, `reynolds_average`), `SolverHints`.
  - Topology & Sparsity: `GraphTopologyAnalysis`, `SchurRoute::{BlockDiagonalPairs, TreeChain, SparseGeneral}`, `SparseBlockPattern` (`perm`, `inv_perm`, `etree`, `col_ptr`, `row_idx`, `fill_in_blocks`), `NodeIndexing`, `BeamDofMap`.

### Target Crate Boundaries for Downstream Peers

- **`fg-sdf`** (`@contact-ipc`):
  ```rust
  pub struct SdfQuery { pub phi: f64, pub normal: Vec3, pub arc_s: f64, pub tangent: Vec3, pub mat_id: usize }
  pub trait SdfField: Send + Sync { fn query(&self, pos: Vec3) -> SdfQuery; }
  ```
- **`fg-contact`** (`@contact-ipc`):
  - `ipc`: spatial-hash broadphase, point–point & edge–edge distance + CCD TOI + C-IPC $C^2$ log-barrier + Py-DiSMech $C^2$ soft-barrier + SPD-projected Hessians + smoothed mollifier friction.
  - `lcp`: compliance-based Delassus-Schur PGS / Alart-Curnier Signorini–Coulomb solver.
  - `arpc`: unified implicit ARPC evaluator returning `(wrench: Vec6, k_tan: Mat6, next_state)` for `RatchetWheel`, `AnisotropicScale`, `IsotropicPad`.
- **`fg-solver`** (`@continuum-solver`):
  - `WorldState`, `SolverConfig` (integrator: `LinearlyImplicitEuler`, `DampedNewtonEuler`, `NewmarkBeta`, `Bdf2`; continuum: `Corotational` / `Der`; contact plugins), `FlexGraphWorld::new(graph, sdf, cfg)`, `step(&mut self, dt)`, `step_batch_rayon(...)`.
- **`fg-robots`, `fg-ffi`, `fg-bench`** (`@bindings-bench`):
  - 5 archetypes proving all 3 `SchurRoute`s: `SerpWorm-N` (`BlockDiagonalPairs`), `BristleWorm` (`Z2xZ2`), `TendonEndoscope-3D` (`TreeChain`, fixed base + cantilever + `D4` + 3D `SplineTubeSdf`), `FlexSpine-Centipede` (`TreeChain` + Bekker feet), `Tensegrity-6Bar` (`SparseGeneral`, cyclic $\beta_1 = 7$).
  - C-ABI `cdylib` (`fg_world_create_from_json`, `fg_world_step`, `fg_world_get_state`, `fg_world_set_targets`, `fg_world_free`) + zero-dependency Python `ctypes` wrapper (`src/flexgraph_rs/`).

---

## 4. Literature References

1. **Li, M., Kaufman, D. M., & Jiang, C.** (2021). *Codimensional Incremental Potential Contact (C-IPC)*. ACM Trans. Graph. (SIGGRAPH), `arXiv:2012.04457`.
2. **Lan, L., Kaufman, D. M., Li, M., Jiang, C., & Yang, Y.** (2022). *Affine Body Dynamics: Fast, Stable & Intersection-free Simulation of Stiff Materials*. ACM Trans. Graph. (SIGGRAPH).
3. **Ferguson, Z., Li, M., Schneider, T., Gil-Ureta, F., Langlois, T., Jiang, C., Zorin, D., Kaufman, D. M., & Panozzo, D.** (2021). *Intersection-free Rigid Body Dynamics (Rigid-IPC)*. ACM Trans. Graph. (SIGGRAPH).
4. **Bergou, M., Wardetzky, M., Robinson, S., Audoly, B., & Grinspun, E.** (2008). *Discrete Elastic Rods*. ACM Trans. Graph. (SIGGRAPH).
5. **Bergou, M., Audoly, B., Vouga, E., Wardetzky, M., & Grinspun, E.** (2010). *Discrete Viscous Threads*. ACM Trans. Graph. (SIGGRAPH).
6. **Jawed, M. K., Novelia, A., & O'Reilly, O. M.** (2018). *A Primer on the Kinematics of Discrete Elastic Rods*. Springer.
7. **Lahoti, R., Chaiyakul, A., & Jawed, M. K.** (2025). *Py-DiSMech: A Scalable and Efficient Framework for Discrete Differential Geometry-Based Modeling and Control of Soft Robots*. `arXiv:2512.09911`.
8. **Parthasarathy, T., Kim, S. H., Cui, S., & Gazzola, M.** (2026). *Elastica++: A high-performance, multiphysics framework for large interacting assemblies of Cosserat rods*. `arXiv:2605.13766`.
9. **Renda, F., Boyer, F., Dias, J., & Seneviratne, L.** (2018). *Discrete Cosserat Approach for Multisection Soft Manipulator Dynamics*. IEEE Trans. Robotics (T-RO), 34(6), 1518–1533.
10. **Shen, J., Ha, J., & Chen, Y.** (2026). *Modeling the Frictional Contact of Continuum Robots with Environments through Linear Complementarity Problem*. Int. J. Robotics Research (IJRR), `arXiv:2611.18546`.
11. **Salagame, A., et al.** (2025). *Optimal Trajectory Planning in a Vertically Undulating Snake Locomotion using Contact-Implicit Optimization*. `arXiv:2508.02953`.
12. **Hublikar, H.** (2025). *Contact-Implicit Modeling and Simulation of a Snake Robot on Compliant and Granular Terrain*. `arXiv:2512.05008`.
13. **Gidoni, P.** (2017). *Rate-independent soft crawlers*. Quart. J. Mech. Appl. Math., `arXiv:1710.08340`.
14. **Xie, J., et al.** (2018). *Rigid Body Dynamic Simulation with Multiple Convex Contact Patches*. `arXiv:1809.05555`.
15. **Hunt, K. H., & Crossley, F. R. E.** (1975). *Coefficient of Restitution Interpreted as Damping in Vibroimpact*. ASME J. Appl. Mech., 42(2), 440–445.
16. **Bekker, M. G.** (1960) / **Wong, J. Y.** (2008). *Theory of Ground Vehicles*. Wiley.
17. **Du, T., Wu, K., Ma, P., Wah, S., Spielberg, A., Rus, D., & Matusik, W.** (2021). *DiffPD: Differentiable Projective Dynamics*. ACM Trans. Graph., `arXiv:2101.05917`.
18. **Howell, T. A., Le Cleac'h, S., Brüdigam, J., Kolter, J. Z., Schwager, M., & Manchester, Z.** (2022). *Dojo: A Differentiable Physics Engine for Robotics*. `arXiv:2203.00806`.
19. **Featherstone, R.** (2008). *Rigid Body Dynamics Algorithms*. Springer.
20. **Davis, T. A.** (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
21. **Mittal, M., Rudin, N., Klemm, V., Allshire, A., & Hutter, M.** (2024). *Symmetry Considerations for Learning Task Symmetric Robot Policies*. IEEE ICRA, `arXiv:2403.04359`.
22. **Hu, X., et al.** (2025). *Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning*. `arXiv:2503.01255`.
23. **Vlasov, V. Z.** (1961). *Thin-Walled Elastic Beams* (restrained warping torsion) & **Timoshenko, S.** (1959) *Theory of Plates and Shells* (Kirchhoff-Love flexural stiffening).
24. **Wang, B., Ferguson, Z., Schneider, T., Gao, X., Panozzo, D., & Zorin, D.** (2021). *A Large Scale Benchmark and an Inclusion-Based Algorithm for Continuous Collision Detection*. ACM Trans. Graph. (SIGGRAPH), 40(5), 1–16.
25. **Todorov, E., Erez, T., & Tassa, Y.** (2012). *MuJoCo: A physics engine for model-based control*. IEEE/RSJ IROS, 5026–5033.
26. **Goulart, P. J., & Chen, Y.** (2024). *Clarabel: An interior-point solver for conic programs with quadratic objectives*. `arXiv:2405.12762`.
27. **Snyder, J. M.** (1992). *Interval Analysis for Computer Graphics*. ACM SIGGRAPH Computer Graphics, 26(2), 121–130.

---

## 5. v2 Industrial Ecosystem & Differentiable Engine Architecture

FlexGraph-RS v2 elevates the pure-Rust FP64 core from a bespoke continuum solver into an industrial-grade, interoperable, **end-to-end differentiable** rigid-flexible simulation engine while preserving strict `< 1e-10` FP64 parity on `SerpWorm-N` and strict permissive licensing (**MIT / Apache-2.0 only**, zero GPL/copyleft dependencies).

### 5.1 Permissive Rust Ecosystem Matrix (`rust/flexgraph-rs/Cargo.toml`)

| Crate | Resolved Version | License | Target Role in `flexgraph-rs` v2 |
| :--- | :--- | :--- | :--- |
| **`faer`** | `0.24.4` (`std, rayon, sparse, sparse-linalg`) | MIT | Supernodal SIMD-accelerated sparse Cholesky / $LDL^\top$ backend for large cyclic rigid-interface Schur systems ($N_b > 24$) and dense/sparse adjoint linear solves |
| **`parry3d-f64`** | `0.31.1` (`glamx 0.3.1`) | Apache-2.0 | BVH-accelerated FP64 collision queries (`TriMesh`, `ConvexPolyhedron`, `Ball`, `Cuboid`, `Cylinder`, `Capsule`), signed distance & closest-point narrowphase for `MeshColliderSpec` |
| **`clarabel`** | `0.11.1` (pure-Rust QDLDL) | Apache-2.0 | Primal-dual interior-point **Second-Order Cone Program (SOCP)** solver for exact isotropic & anisotropic 3D Coulomb friction cones (`Delassus` contact complementarity) |
| **`urdf-rs`** | `0.10.0` | Apache-2.0 | Standard ROS URDF XML importer (`<link>`, `<joint>`, `<inertial>`, `<collision>`) directly populating `RigidFlexGraph` + `MeshColliderSpec` |
| **`rapier3d-f64`** | `0.36.1` | Apache-2.0 | **Benchmark validation ONLY** (`fg-bench` head-to-head rigid-body accuracy & throughput baseline; never imported in core physics crates) |

### 5.2 Exact Tight Inclusion CCD (Wang et al., ACM TOG 2021)

Classical floating-point cubic root finders for edge–edge (EE) and point–triangle (PT) Continuous Collision Detection suffer from **false negatives** (missed collisions due to IEEE-754 rounding near grazing/coplanar contacts), which break IPC's intersection-free invariant. FlexGraph-RS v2 upgrades `fg-contact::ccd` with the **Tight Inclusion** interval-arithmetic bisection algorithm (Wang et al. 2021):

Given linear trajectory interpolation over $t \in [0, 1]$ and barycentric/parametric coordinates $\mathbf{u} = (u_1, u_2) \in \Omega \subset [0, 1]^2$, the separation function $\mathbf{f}(t, u_1, u_2) \in \mathbb{R}^3$ is a trilinear Bernstein/monomial map on each sub-box $B = [t_a, t_b] \times [u_{1a}, u_{1b}] \times [u_{2a}, u_{2b}]$. Because trilinear maps attain their extrema at the 8 box corners $\mathcal{V}(B)$:
$$
\min_{(t,\mathbf{u}) \in B} f_c(t, \mathbf{u}) \ge \min_{v \in \mathcal{V}(B)} \operatorname{fl}\!\big(f_c(v)\big) - \delta_{\mathrm{fp},c}, \qquad
\max_{(t,\mathbf{u}) \in B} f_c(t, \mathbf{u}) \le \max_{v \in \mathcal{V}(B)} \operatorname{fl}\!\big(f_c(v)\big) + \delta_{\mathrm{fp},c},
$$
where $\delta_{\mathrm{fp},c}$ is a rigorous IEEE-754 rounding error bound (plus optional minimum separation offset $\xi / 2$ for C-IPC barrier safety). A sub-box $B$ is pruned **if and only if** the padded interval $[f_{c,\min} - \delta_{\mathrm{fp},c} - \xi,\, f_{c,\max} + \delta_{\mathrm{fp},c} + \xi]$ excludes $0$ along any axis $c \in \{x, y, z\}$. Priority-queue bisection ordered by earliest $t_a$ terminates either when all boxes are pruned (certified no impact) or when box width falls below tolerance $\epsilon_{\mathrm{toi}}$, returning the conservative lower bound $t_{\mathrm{TOI}} = t_a$ with **provably zero false negatives**.

### 5.3 MuJoCo-Compatible Soft-Constraint Impedance (`SolRefImp`) & MJCF Interoperability

To provide seamless MJCF/MuJoCo parameter portability across contacts (`ContactSiteSpec`), articulated joints (`JointEdgeSpec`), tendons (`TendonEdgeSpec`), and mesh colliders (`MeshColliderSpec`), `fg_math::SolRefImp` implements MuJoCo's exact constraint impedance and reference-acceleration law (Todorov et al. 2012, `engine_core_constraint.c`):

1. **$C^1$-Smooth Piecewise Power Impedance $d(r)$** (`solimp = [dmin, dmax, width, midpoint, power]`):  
   For constraint violation magnitude $|r|$ with normalized depth $x = |r| / \mathrm{width} \in [0, 1]$, midpoint $m \in (0, 1)$, and exponent $p \ge 1$:
   $$
   y(x) = \begin{cases}
   x, & p = 1, \\
   \dfrac{x^p}{m^{p-1}}, & 0 \le x \le m, \\
   1 - \dfrac{(1 - x)^p}{(1 - m)^{p-1}}, & m < x < 1,
   \end{cases}
   \qquad
   d(r) = \operatorname{clip}_{[10^{-4}, 0.9999]}\!\Big(d_{\min} + y(x)\,(d_{\max} - d_{\min})\Big),
   $$
   which satisfies $y(m) = m$ and $y'(m^-) = y'(m^+) = p$ identically for all $(m, p)$, with exact analytical derivative $\frac{dd}{d|r|}$ verified by finite differences.
2. **Dual `solref` Conventions** (`solref = [timeconst, dampratio]`):
   - **Positive standard mode** ($\tau > 0, \zeta > 0$, stability-clamped to $\tau_{\mathrm{eff}} = \max(\tau, 2\Delta t)$):
     $$
     B = \frac{2}{d_{\max}\tau_{\mathrm{eff}}}, \qquad
     K = \frac{1}{d_{\max}^2 \tau_{\mathrm{eff}}^2 \zeta^2}.
     $$
   - **Direct negative mode** ($\mathrm{solref} = [-k_{\mathrm{in}}, -b_{\mathrm{in}}]$):
     $$
     B = \frac{b_{\mathrm{in}}}{d_{\max}}, \qquad
     K = \frac{k_{\mathrm{in}}}{d_{\max}^2}.
     $$
3. **Closed-Loop Constraint Dynamics & SOCP Regularization**:  
   With reference acceleration $a_{\mathrm{ref}} = -B v - K\,d(r)\,r$, the constrained acceleration satisfies:
   $$
   a_1 + \underbrace{(d B)}_{b_{\mathrm{eff}}} v + \underbrace{(d^2 K)}_{k_{\mathrm{eff}}} r = (1 - d)\,a_0,
   \qquad
   R = \frac{1 - d(r)}{d(r)}\,\tilde{A}_{ii},
   $$
   so at $d(r) = d_{\max}$, positive mode recovers exact natural frequency $\omega_n = 1/(\tau\zeta)$ and damping ratio $\zeta$ (invariant to $r$!), direct negative mode recovers exact $(k_{\mathrm{in}}, b_{\mathrm{in}})$, and $R$ directly regularizes the Clarabel SOCP Delassus matrix $\mathbf{A}_{\mathrm{reg}} = \mathbf{J}_c \mathbf{H}^{-1} \mathbf{J}_c^\top + \operatorname{diag}(\mathbf{R})$.

### 5.4 Differentiable Physics via IFT Reverse-Mode Adjoints (DiffPD / Dojo)

For gradient-based system identification (UTM strip stiffness, Bekker soil/friction calibration) and trajectory/policy optimization, `fg-ir` exposes `DiffParamSpec` / `DiffParamKind` covering 7 physical parameter families (`BeamYoungsModulus`, `BeamDampingBeta`, `ContactMuFwd`, `ContactMuRev`, `ContactStiffness`, `TendonStiffness`, `BodyMass`).

At each implicit time step $k \in \{0, \dots, T-1\}$, the forward physics solves the nonlinear residual $\mathbf{F}_k(\mathbf{z}_{k+1}, \mathbf{z}_k, \boldsymbol{\theta}) = \mathbf{0}$ (where $\mathbf{z}_k = (\mathbf{x}_k, \mathbf{v}_k)$ and $\boldsymbol{\theta} \in \mathbb{R}^{n_\theta}$ are the active `DiffParamSpec` values) with converged tangent Jacobian $\mathbf{H}_k = \frac{\partial \mathbf{F}_k}{\partial \mathbf{v}_{k+1}}$ (and for Clarabel SOCP contacts, the relaxed interior-point KKT Jacobian at central-path parameter $\kappa_{\mathrm{ip}}$, following Dojo [Howell et al. 2022]). Given a scalar rollout objective $\mathcal{L}(\mathbf{z}_1, \dots, \mathbf{z}_T; \boldsymbol{\theta})$, the **Implicit Function Theorem (IFT)** yields exact reverse-mode adjoint recursion without differentiating through Newton/solver loops:
$$
\mathbf{H}_k^\top \boldsymbol{\lambda}_{v,k} = \frac{\partial \mathcal{L}}{\partial \mathbf{v}_{k+1}} + \left(\frac{\partial \mathbf{x}_{k+1}}{\partial \mathbf{v}_{k+1}}\right)^\top \boldsymbol{\lambda}_{x,k+1} - \left(\frac{\partial \mathbf{F}_{k+1}}{\partial \mathbf{v}_{k+1}}\right)^\top \boldsymbol{\lambda}_{v,k+1},
$$
$$
\frac{d\mathcal{L}}{d\boldsymbol{\theta}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} - \sum_{k=0}^{T-1} \left(\frac{\partial \mathbf{F}_k}{\partial \boldsymbol{\theta}}\right)^\top \boldsymbol{\lambda}_{v,k}.
$$
Crucially, because $\mathbf{H}_k$ is symmetric positive definite (or block-factored via the exact same Stage 1 Block-Thomas + Stage 2 Rigid Schur / `faer` $LDL^\top$ factorization already computed during the forward step), **each backward adjoint step requires ZERO new matrix factorizations** — only a single $O(E + \mathrm{nnz}(L))$ forward/backward substitution, matching DiffPD's asymptotic complexity while supporting full 3D Lie-group $SE(3)$ kinematics and frictional contact.

