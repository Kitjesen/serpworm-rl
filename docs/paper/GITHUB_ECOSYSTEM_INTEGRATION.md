# GitHub Open-Source Ecosystem Integration Report — SerpWorm-RL / FlexGraph

**Date:** 2026-11-26  
**Scope:** Engineering audit of verified, actively maintained open-source repositories across Rust physics/FFI crates, soft/continuum/GPU robotics simulators, morphology-agnostic/equivariant locomotion RL, and vectorized RL environment bridges — mapped directly onto Python `src/flexgraph/` and Rust `rust/flexgraph-rs/`.  
**License Policy:** Workspace license is `MIT OR Apache-2.0` (`rust/flexgraph-rs/Cargo.toml`). Direct dependencies are strictly restricted to permissive licenses (`MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`). Copyleft (`GPL-3.0`), non-commercial (`CC BY-NC-SA 4.0`), and unlicensed research repositories are explicitly flagged for **clean-room algorithmic/benchmark reference only** (zero code copying).

---

## 1. Current Baseline & Capability Gaps (Local Codebase Audit)

### Python `src/flexgraph/` (Layers 1–5)
- **Already strong:** FP64 bipartite `GraphSchurSolver` (`solver.py`), declarative `RigidFlexGraph` IR + 4 robot archetypes (`serpworm`, `bristle_worm`, `tendon_endoscope`, `flexspine_crawler`) + finite symmetry group specs $\mathbb{Z}_2, \mathbb{Z}_2\times\mathbb{Z}_2, D_4$ (`graph.py`), `ComposableSDFWorld` (`sdf_world.py`), `AnisotropicPatchContact` + $C^2$ self-contact (`contact.py`), `TokenizedHIMDreamWaQ` (multi-scale entity TCN + `TransformerEncoder` + 7D EstNet + $\beta$-VAE + InfoNCE/Sinkhorn-SwAV) + exact Reynolds `FiniteGroupEquivariantWrapper` (`policy.py`), and `FlexGraphEnv` + `UniversalSafetyGuard` (`env.py`).
- **Concrete gaps blocking SOTA multi-morphology RL:**
  1. **No PPO trainer, asymmetric critic, or running observation normalization:** `env.py` uses hardcoded scaling (`dq_ema / 6.0`, raw `omega_b`), which mismatches across heterogeneous morphologies (e.g., `tendon_endoscope` `speed_limit=0.15` vs. `serpworm` `speed_limit=6.0`). `TokenizedHIMDreamWaQ` has no value head (`critic`).
  2. **Rollout-time symmetry overhead & un-averaged latents:** `FiniteGroupEquivariantWrapper` multiplies rollout forward cost by $|G|$ ($4\times$ for $\mathbb{Z}_2\times\mathbb{Z}_2$, $8\times$ for $D_4$) and currently returns un-averaged `z_mu[:B]` (`policy.py:420–423`) rather than group-invariant latents.
  3. **Minimal 4D actuator metadata & untrained fault masking:** `env.py:112` encodes only `[seg_norm, side, is_tendon, is_joint]`, limiting zero-shot cross-embodiment transfer; `active_mask` is supported structurally but has no training-time fault-dropout curriculum.
  4. **Incomplete vectorized env contract:** `FlexGraphEnv.step()` lacks per-env `terminated`/`truncated` (`time_outs`) outputs and selective partial `reset_done(mask)` / `reset_idx(env_ids)`, and `env.py:210` calls `bool(self.solver.failed.item())` which breaks for batched failure tensors (`batch > 1`).

### Rust `rust/flexgraph-rs/` (8-Crate Workspace)
- **Already strong:** Clean pure-Rust FP64 stack (`fg-math`, `fg-ir`, `fg-sdf`, `fg-contact`, `fg-solver`, `fg-robots`) with only `serde`, `serde_json`, `serde_yaml`, `thiserror`, and `rayon` (`1.10`). Already includes corotational + DER beams, `BlockDiagonalPairs` / `TreeChain` / `SparseGeneral` Schur routing, `SpatialHashGrid`, ACCD, IPC log-barrier, Delassus-LCP (PGS/Alart-Curnier/Fischer-Burmeister), ARPC, and BVH/CSG/SplineTube/Voxel SDFs.
- **Concrete gaps blocking high-throughput RL & real-world adoption:**
  1. **`fg-ffi` is currently a 1-line placeholder (`crates/fg-ffi/src/lib.rs`):** No PyO3 Python module, no `maturin` wheel build, no zero-copy `rust-numpy` or DLPack buffer handoff, and no C-ABI batch stepping symbols yet.
  2. **Scalar `SparseGeneral` LDLT & dense Delassus scaling:** Hand-written scalar `SparseBlockLdltFactor` lacks SIMD supernodal factorization and fill-reducing ordering for large cyclic/mesh rigid-flexible graphs.
  3. **No triangle-mesh/CAD collider BVH, full ROS URDF mesh tags, or MJCF importer:** `fg-sdf` supports analytic primitives, spline tubes, and voxel grids, but lacks general FP64 triangle-mesh/convex-hull collision/projection. Likewise, while `fg-robots/src/loader.rs:453–1123` already implements a custom `URDF-Flex XML` parser (`load_graph_from_urdf_flex_str` / `load_graph_from_urdf_flex_file` for `<link>`, `<joint>`, `<beam>`, `<tendon>`, `<contact_site>`), full standard ROS URDF tags (`<visual>`/`<collision>`/`<mesh>`/`<mimic>`) and **MJCF** XML import are currently missing.

---

## 2. Verified Open-Source Ecosystem Comparison Matrix

*All URLs, licenses, and approximate star counts below were verified via live web search on 2026-11-26. Crate versions reflect `crates.io` / `docs.rs` listings verified on the same date.*

### (a) Rust Physics, Linear Algebra, Contact & FFI Crates

| Repository | Verified URL | Stars (approx.) | License | Core Capability | Direct Integration Target | Effort vs. Gain |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **`faer-rs`** (`faer` `0.24.4`) | https://github.com/sarah-quinones/faer-rs | ~2.6k+ | **MIT** | Pure-Rust SIMD dense & sparse linear algebra (`faer-sparse`, `faer-cholesky`), sparse $LDL^\top$ & Bunch-Kaufman $LBL^\top$ | **Rust** `fg-solver` (`SparseGeneral` Schur), `fg-contact` (Delassus solves) | Low–Med / **Very High** |
| **`parry`** (`parry3d-f64` `0.31.1`) | https://github.com/dimforge/parry | ~880–900+ | **Apache-2.0** | FP64 3D geometry & collision: SIMD dynamic BVH, TOI CCD, triangle meshes, convex hulls, sparse voxels, point projection | **Rust** `fg-sdf` & `fg-contact` (CAD/mesh obstacles & broadphase) | Med / **High** |
| **`rapier`** (`rapier3d-f64`) | https://github.com/dimforge/rapier | ~5.9k–6.1k+ | **Apache-2.0** | Deterministic 2D/3D Rust rigid-body engine with CCD, joints, island sleeping | **Rust** `fg-bench` (rigid-body speed/accuracy reference baseline) | Low / Med |
| **`nalgebra`** (`0.35.0`) | https://github.com/dimforge/nalgebra | ~4.8k+ | **Apache-2.0** | Static/dynamic FP64 matrices, geometric transforms, SVD/QR/Cholesky/eigen decompositions | **Rust** `fg-math` (optional feature-gated interop bridge to `parry3d-f64`) | Low / Med |
| **`Clarabel.rs`** (`clarabel` `0.11.1`) | https://github.com/oxfordcontrol/Clarabel.rs | ~630+ | **Apache-2.0** | Pure-Rust interior-point QP/SOCP conic solver (exact 3D second-order Coulomb friction cones with native sparse `qdldl` KKT solver) | **Rust** `fg-contact/src/lcp.rs` (optional exact SOCP fallback for stiff ill-conditioned contacts) | Med / High |
| **`pyo3`** (`0.29–0.30`) | https://github.com/PyO3/pyo3 | ~16.4k | **Apache-2.0 OR MIT** | Native Rust-to-Python extension bindings, GIL release (`py.allow_threads`) for Rayon parallelism | **Rust** `fg-ffi` $\leftrightarrow$ **Python** `rust_backend.py` | Low / **Very High** |
| **`maturin`** | https://github.com/PyO3/maturin | ~5.9k | **Apache-2.0 OR MIT** | Zero-config Cargo-to-pip wheel builder for PyO3/CFFI extensions | **Rust** `fg-ffi` CI & remote RTX 5090 Linux wheel builds | Low / **High** |
| **`rust-numpy`** (`numpy` `0.30`) | https://github.com/PyO3/rust-numpy | ~1.4k | **BSD-2-Clause** | PyO3 bindings to NumPy C-API for zero-copy `ndarray` slice views | **Rust** `fg-ffi` (`step_batch_into` zero-copy obs/act arrays) | Low / **Very High** |
| **`rayon`** (`1.10`, *already in workspace*) | https://github.com/rayon-rs/rayon | ~13.5k | **Apache-2.0 OR MIT** | Work-stealing CPU data parallelism (`par_iter`) | **Rust** `fg-solver` + `fg-ffi` (GIL-free parallel multi-env pool) | Already wired / **High** |
| **`urdf-rs`** (`0.10.0`) | https://github.com/openrr/urdf-rs | ~50 | **Apache-2.0** | Pure-Rust URDF XML parser covering full standard ROS URDF `<link>`/`<joint>`/`<visual>`/`<collision>`/`<mesh>`/`<mimic>` tags | **Rust** `fg-ir` & `fg-robots/src/loader.rs` (complement existing custom `URDF-Flex` parser with full ROS URDF mesh/mimic support) | Low / Med–High |

### (b) Soft / Continuum / Rigid-Flexible Simulators & GPU/Differentiable Engines

| Repository | Verified URL | Stars (approx.) | License | Core Capability | Direct Integration Target | Effort vs. Gain |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **`PyElastica`** | https://github.com/GazzolaLab/PyElastica | ~360+ | **MIT** | Cosserat rod assemblies, large bending/twisting/shearing, muscular actuation, contact & fluid interaction | **Rust** `fg-bench` (Cosserat/DER validation suite) & **Python** continuum benchmark parity | Low / High |
| **`ipc-toolkit`** | https://github.com/ipc-sim/ipc-toolkit | ~340 | **MIT** | Reference C++ Incremental Potential Contact: codimensional distances, edge mollifiers, CCD, $C^2$ clamped log-barrier, $C^1$ friction, SPD projection | **Rust** `fg-contact` (verification oracle & Tight-Inclusion CCD / mesh self-contact reference) | Low–Med / High |
| **`Genesis`** (`genesis-world`) | https://github.com/Genesis-Embodied-AI/genesis-world | ~30.3k+ | **Apache-2.0** | Universal GPU multi-physics engine (rigid, FEM, MPM, fluids), URDF/MJCF/mesh loaders, differentiable kernels | **Both** (paper throughput/accuracy benchmark & multi-physics scene ergonomics) | Med / Med–High |
| **`NVIDIA Warp`** | https://github.com/NVIDIA/warp | ~7.1k | **Apache-2.0** | Python JIT-compiled CUDA/CPU spatial kernels with reverse-mode autodiff | **Python** `src/flexgraph/` (future GPU kernel backend if scaling rigid-flexible sims past 4,096 envs) | High / Very High |
| **`Newton Physics`** | https://github.com/newton-physics/newton | ~6.0k+ | **Apache-2.0** | GPU-accelerated differentiable robotics engine built on Warp + MuJoCo Warp + OpenUSD | **Both** (OpenUSD / differentiable multi-body interop & paper comparison) | High / Med–High |
| **`MuJoCo`** (incl. MJX) | https://github.com/google-deepmind/mujoco | ~18.5k | **Apache-2.0** | Canonical MJCF XML schema (`<tendon><spatial>`, `<flexcomp>`, `<site>`), smooth contact dynamics, JAX GPU MJX | **Both** `fg-ir` / `graph.py` (MJCF subset importer/exporter & rigid-tendon baseline) | Med / High |
| **`MuJoCo Warp`** | https://github.com/google-deepmind/mujoco_warp | ~1.7k | **Apache-2.0** | Warp-accelerated GPU MuJoCo backend for massively parallel RL | **Both** (GPU RL throughput baseline for paper tables) | Low / Med |
| **`dismech` / `dismech-python`** | https://github.com/StructuresComp/dismech <br> https://github.com/StructuresComp/dismech-python | ~51+ / ~8 | **GPL-3.0** *(FLAG: Copyleft)* | Discrete Elastic Rods (DER) / shells, implicit Euler & Newmark-beta, $C^2$ smooth soft-contact | **Reference Only** (published equations only; never copy GPL code — `fg-solver`/`fg-contact` already use clean-room implementations) | N/A (Clean-Room) |
| **`SoRoSim`** | https://github.com/GVSRobotics/SoRoSim | ~91–93 | **No explicit license** *(FLAG: Restricted)* | MATLAB toolbox for Geometric Variable Strain (GVS) hybrid soft-rigid Cosserat robots | **Reference Only** (published GVS benchmark problems & paper related-work comparison; do not copy code) | N/A (Clean-Room) |

### (c) Morphology-Agnostic & Symmetry-Equivariant Locomotion RL

| Repository | Verified URL | Stars (approx.) | License | Core Capability | Direct Integration Target | Effort vs. Gain |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **`rsl_rl`** | https://github.com/leggedrobotics/rsl_rl | ~3.2k+ | **BSD-3-Clause** | GPU PPO with **Adaptive KL LR (`desired_kl=0.01`)**, `EmpiricalNormalization` (`torch.compile`-safe), clipped value loss, `symmetry.py` data augmentation + mirror loss | **Python** `src/flexgraph/policy.py`, `trainer.py` (**Adopt immediately**) | Low / **Very High** |
| **`MorphoSymm`** | https://github.com/Danfoa/MorphoSymm | ~114 | **MIT** | Finite-group morphological symmetry representations ($L_g, K_g$), symmetry data augmentation for RL, equivariant NN design (`escnn`) | **Python** `src/flexgraph/policy.py`, `trainer.py` (group-invariant critic + batch symmetry augmentation) | Low / **Very High** |
| **`URMA`** (`one_policy_to_run_them_all`) | https://github.com/nico-bohlinger/one_policy_to_run_them_all | ~230+ | **MIT** *(code; excl. 3rd-party XMLs)* | Unified Robot Morphology Architecture: rich per-joint description vectors as attention keys + morphology-conditioned per-actuator decoder for zero-shot cross-embodiment transfer | **Python** `src/flexgraph/env.py`, `policy.py`, `trainer.py` (rich actuator descriptors & multi-morphology sampling) | Low–Med / **Very High** |
| **`go2_dreamwaq`** / **`dreamwaq-isaaclab`** | https://github.com/curieuxjy/go2_dreamwaq <br> https://github.com/romankalyna/dreamwaq-isaaclab | Community reproductions | **MIT** (`curieuxjy`) <br> **No explicit license** *(FLAG: Restricted)* (`romankalyna`) | DreamWaQ CENet ($\beta$-VAE + explicit velocity estimator) + asymmetric critic & AdaBoot reward-CV bootstrapping | **Python** `src/flexgraph/trainer.py`, `policy.py` (asymmetric privileged critic + EstNet warm-up; adapt only from MIT `curieuxjy` / clean-room paper) | Low / **High** |
| **`BodyTransformer`** (BoT) | https://github.com/carlosferrazza/BodyTransformer | ~200+ | **No explicit license** *(FLAG: Restricted)* | Graph-structured robot embodiment Transformer using kinematic adjacency / hop-distance attention masks | **Python** `src/flexgraph/policy.py` (**Clean-room paper concept only**: graph hop bias from `RigidFlexGraph`) | Low / High |
| **`MetaMorph`** (ICLR 2022) | https://github.com/agrimgupta92/metamorph | ~134 | **No explicit license** *(FLAG: Restricted)* | Transformer universal controller over modular robot design spaces; held-out morphology generalization protocol | **Python** `src/flexgraph/trainer.py` (**Clean-room protocol only**: multi-length curriculum & zero-shot held-out eval) | Low / Med–High |
| **`HIMLoco`** | https://github.com/Sage-PhysicalAI/HIMLoco | Active research repo | **CC BY-NC-SA 4.0** *(FLAG: Non-Commercial)* | Hybrid Internal Model (contrastive dynamics + explicit velocity estimation from proprioception) | **Reference Only** (already cleanly implemented from scratch in `policy.py`; never vendor repo code) | N/A (Clean-Room) |

### (d) Vectorized GPU / Rust RL Environment Bridges

| Repository | Verified URL | Stars (approx.) | License | Core Capability | Direct Integration Target | Effort vs. Gain |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **`IsaacLab`** | https://github.com/isaac-sim/IsaacLab | ~8.7k+ | **BSD-3-Clause** *(core)* | GPU vectorized RL conventions: selective `reset_idx(env_ids)`, `terminated` vs. `truncated` (`time_outs`), separated `policy` vs. `critic` obs dicts | **Python** `src/flexgraph/env.py`, `trainer.py` (**Adopt immediately**) | Low / **Very High** |
| **`Gymnasium`** | https://github.com/Farama-Foundation/Gymnasium | ~13k | **MIT** | Standard `VectorEnv` 5-tuple contract (`obs, rew, terminated, truncated, info`), `AutoresetMode`, seed/options | **Python** `src/flexgraph/env.py` (optional thin wrapper for ecosystem compatibility) | Low / High |
| **`PufferLib`** | https://github.com/PufferAI/PufferLib | ~6.9k+ | **MIT** | Zero-copy shared-buffer `VectorEnv`, heterogeneous observation packing, high-throughput native env orchestration | **Python** `rust_backend.py` & `trainer.py` (pre-allocated flat batch buffers) | Low–Med / High |
| **`EnvPool`** | https://github.com/sail-sg/envpool | ~2k+ | **Apache-2.0** | Native thread-pool vectorized env architecture returning contiguous batched buffers in a single FFI call | **Rust** `fg-ffi` (`BatchWorldPool` design pattern over Rayon) | Low–Med / **Very High** |
| **`DLPack`** & **`dlpark`** / **`dlpack-rs`** | https://github.com/dmlc/dlpack <br> https://github.com/SunDoge/dlpark <br> https://github.com/kornia/dlpack-rs | ~1.3k / ~66 / ~7 | **Apache-2.0** | Standard `DLTensor` / `__dlpack__` capsule for zero-copy Rust $\leftrightarrow$ PyTorch (`torch.from_dlpack`) tensor sharing | **Rust** `fg-ffi` $\leftrightarrow$ **Python** `rust_backend.py` | Low–Med / **High** |

---

## 3. Top 5 Concrete, Immediately Adoptable Upgrades

### Part A — Python FlexGraph RL / Control Upgrades (Immediate for `@flexgraph-rl`)

#### Upgrade #1: `rsl_rl`-Grade PPO Stability Core + Mask-Aware `TokenEmpiricalNorm` + Asymmetric Critic
- **Sources:** `leggedrobotics/rsl_rl` (`BSD-3-Clause`), DreamWaQ CENet (`MIT` community reproductions).
- **Targets:** `src/flexgraph/policy.py`, `src/flexgraph/trainer.py`.
- **Why now:** Right now `env.py:144–158` hardcodes `dq_ema / 6.0` and passes unscaled `omega_b`, which spans 1–2 orders of magnitude across `tendon_endoscope`, `bristle_worm`, `serpworm_N3/N5`, and `flexspine_crawler`. Without running normalization, adaptive KL scheduling, and a privileged asymmetric critic, multi-morphology PPO gradients oscillate or collapse.
- **Concrete changes:**
  1. **Mask-Aware `TokenEmpiricalNorm(nn.Module)`** (modeled on `rsl_rl/modules/normalization.py`): Maintain Welford/Chan running `mean` and `var` buffers for both actuator token dynamic channels `(q, dq, q_err, u_prev)` (updating statistics **only over `active_mask == True` elements** so jammed/padded tokens never skew statistics; leave static morphology descriptors un-normalized or range-normalized) and global IMU/cmd channels `(B, H, d_glob)`. Freeze buffer updates during `.eval()` and keep pure tensor ops for `torch.compile` compatibility.
  2. **Asymmetric Critic Head in `TokenizedHIMDreamWaQ` (backward-compatible):** Keep `forward()` returning `TokenizedForwardOut` unchanged for `tests/test_flexgraph.py`, and add an optional `critic_head` + `evaluate_value(tokens, glob, active_mask, priv_state=None)` consuming pooled token embeddings + global context + privileged 7D ground-truth physics state (`est_target`: `[vx, vy, wz, wall_conf, wall_side, drag, slip]`, plus morphology scalar summary / contact indicators).
  3. **PPO Update Stability Triad (`src/flexgraph/trainer.py`):**
     - **Adaptive KL LR schedule** (`rsl_rl/algorithms/ppo.py`): target `desired_kl = 0.01`; if `kl > 2.0 * desired_kl` scale `lr = max(1e-5, lr / 1.5)`; if `kl < 0.5 * desired_kl` scale `lr = min(1e-2, lr * 1.5)`.
     - **Clipped / Smooth-L1 (Huber) Value Loss** + **GAE with `time_outs` bootstrapping** (`r += gamma * V(s_next) * truncated`) + per-minibatch advantage normalization + `clip_grad_norm_(..., 1.0)`.
     - **Coupled HIM-DreamWaQ Auxiliary Optimization:** Run `compute_aux_losses()` jointly on minibatches with explicit loss weight ramp / FP32 clamping and finite-loss guard.

#### Upgrade #2: Dual-Mode Morphological Symmetry (`MorphoSymm` + `rsl_rl`) & URMA Multi-Morphology / Fault-Masking Curriculum
- **Sources:** `Danfoa/MorphoSymm` (`MIT`), `leggedrobotics/rsl_rl` (`BSD-3-Clause`), `nico-bohlinger/one_policy_to_run_them_all` (`MIT`), plus clean-room graph-hop bias concept from BodyTransformer.
- **Targets:** `src/flexgraph/policy.py`, `src/flexgraph/env.py`, `src/flexgraph/trainer.py`.
- **Why now:**
  - `FiniteGroupEquivariantWrapper` in `policy.py` executes $|G|$ batched passes every single step ($4\times$ on `bristle_worm`, $8\times$ on `tendon_endoscope`) and currently returns `z_mu[:B]` (`policy.py:420–423`) without group-invariant averaging.
  - `TokenizedHIMDreamWaQ` already supports variable $N$ and `active_mask`, but without multi-morphology sampling and random fault masking during training, cross-morphology and fault resilience remain untested at convergence.
- **Concrete changes:**
  1. **Symmetry Augmentation + Mirror Loss (`rsl_rl/extensions/symmetry.py` + `MorphoSymm`):** During PPO updates, offer two composable modes on top of `SymmetryGroupSpec`:
     - *Fast Training Mode:* Augment PPO minibatches with group-transformed `(L_g o, K_g a)` transitions (or add mirror penalty $\lambda_{\text{sym}} \|\pi(L_g o) - K_g \pi(o)\|_2^2$ with invariant value target $V(L_g o) = V(o)$), keeping rollout collection $1\times$ speed while multiplying effective sample efficiency by $|G|$.
     - *Exact Deployment/Fine-Tune Mode:* Use `FiniteGroupEquivariantWrapper` (and average `z_mu`/`z_gate` across the group orbit so latents are strictly $G$-invariant, not identity-slot-only).
  2. **Richer URMA-Style Actuator Descriptors & Optional Graph-Hop Attention Bias:** Keep `d_tok=8` default for 100% backward compatibility, with an optional expanded descriptor mode (adding normalized actuator speed/torque limits, range, or graph-hop distance bias derived from `RigidFlexGraph` topology) so the shared `TransformerEncoder` distinguishes `tendon_endoscope` vs. `flexspine_crawler` vs. `serpworm_N3/N5` cleanly.
  3. **Multi-Morphology Round-Robin / Mixture Trainer + Fault-Masking Curriculum:** Sample rollouts across built-in archetypes (`serpworm_N3`, `serpworm_N5`, `bristle_worm`, `tendon_endoscope`, `flexspine_crawler`) with per-morphology advantage normalization and a progressive actuator-dropout curriculum (randomly zeroing 1–2 `active_mask` bits on 10–25% of episodes after warmup) to bake in zero-shot fault tolerance.

#### Upgrade #3: `IsaacLab` / `Gymnasium` Vectorized Auto-Reset & Pre-Allocated Flat Batch Bridge Contract
- **Sources:** `isaac-sim/IsaacLab` (`BSD-3-Clause`), `Farama-Foundation/Gymnasium` (`MIT`), `PufferAI/PufferLib` (`MIT`).
- **Targets:** `src/flexgraph/env.py`, `src/flexgraph/rust_backend.py`.
- **Why now:**
  - `env.py:210` currently evaluates `bool(self.solver.failed.item())`, which raises `RuntimeError` if `self.solver.failed` is a 1D batch tensor (`batch > 1`) and prevents per-env selective resets.
  - GPU/Rust RL rollouts lose massive throughput if one failed/timed-out env forces resetting all `B` environments simultaneously.
- **Concrete changes:**
  1. **Batch-Safe Per-Env `terminated` / `truncated` & `reset_done(mask)`:** Track per-env `episode_length` (`(B,)`), compute per-env `terminated` (solver divergence / tilt limit exceeded) vs. `truncated` (`episode_length >= max_episode_steps`), add selective `reset_done(done_mask)` (resetting only finished lanes in-place while preserving active lanes), and expose `time_outs` + privileged 7D `est_target` inside `info` while preserving the existing 3-tuple `step()` return signature (or adding a clean `step_vec()` / Gymnasium adapter) so `tests/test_flexgraph.py` stays 100% green.
  2. **Pre-Allocated Flat Batch Contract for `rust_backend.py`:** Design `RustGraphSolverBackend` around contiguous FP64/FP32 batch buffers (`step_into(raw_tgt_f64, substeps)` $\to$ in-place state/obs/failed arrays) with zero-crash automatic fallback to `GraphSchurSolver`, matching both ctypes C-ABI and future PyO3/DLPack buffers seamlessly.

---

### Part B — Rust `flexgraph-rs` Engine & FFI Upgrades (Relay to Partner Session `af2331aa`)

#### Upgrade #4: Zero-Copy `fg-ffi` Batched Env Pool via `pyo3` + `rust-numpy` + `maturin` + `dlpack` + GIL-Free `rayon`
- **Sources:** `PyO3/pyo3` (`Apache-2.0 OR MIT`), `PyO3/maturin` (`Apache-2.0 OR MIT`), `PyO3/rust-numpy` (`BSD-2-Clause`), `dmlc/dlpack` / `SunDoge/dlpark` (`Apache-2.0`), `sail-sg/envpool` (`Apache-2.0` pattern).
- **Targets:** `rust/flexgraph-rs/crates/fg-ffi/Cargo.toml`, `crates/fg-ffi/src/lib.rs`, `pyproject.toml`.
- **Why now:** `crates/fg-ffi/src/lib.rs` is currently a 1-line placeholder. Python RL training needs a **single-call batched step** across `B` environments without per-env Python loop overhead or tensor copying.
- **Concrete design (dual C-ABI + PyO3):**
  1. **Keep C-ABI (`cdylib`) exports** (`fg_world_create_json`, `fg_world_reset`, `fg_world_step_batch`, `fg_world_get_state`, `fg_world_destroy`) for zero-Python-version-coupling ctypes loading and C++/embedded hardware deployment.
  2. **Add feature-gated PyO3 + `rust-numpy` native module (`--features python`, built via `maturin develop --release`):**
     - Expose `PyBatchFlexGraphWorld(graph_json, world_json, batch_size, dt)` owning `Vec<FlexGraphWorld>` (EnvPool pattern).
     - `step_batch_into(py, actions: PyReadonlyArray2<f64>, substeps: usize, out_x: &Bound<PyArray3<f64>>, out_v: &Bound<PyArray3<f64>>, out_act: &Bound<PyArray2<f64>>, out_failed: &Bound<PyArray1<u8>>)`: borrow raw mutable slices, release the Python GIL via `py.allow_threads(|| worlds.par_iter_mut().enumerate().for_each(...))` using `rayon`, and write directly into pre-allocated NumPy/PyTorch-backed host memory with **zero heap allocation and zero data copies** per step.
     - Optional Phase 2 (`dlpark` / `dmlc/dlpack` C-ABI): expose `__dlpack__` capsule on internal state buffers for direct `torch.from_dlpack()` zero-copy tensor views.

#### Upgrade #5: Feature-Gated High-Performance Rust Crates — `faer` (SIMD Sparse LDLT), `parry3d-f64` (FP64 Mesh BVH/CCD), `Clarabel.rs` (SOCP Contact), & `urdf-rs`
- **Sources:** `sarah-quinones/faer-rs` (`MIT`), `dimforge/parry` (`Apache-2.0`), `oxfordcontrol/Clarabel.rs` (`Apache-2.0`), `openrr/urdf-rs` (`Apache-2.0`).
- **Targets:** `rust/flexgraph-rs/Cargo.toml`, `crates/fg-solver/src/schur.rs`, `crates/fg-contact/src/lcp.rs`, `crates/fg-sdf/`, `crates/fg-robots/src/loader.rs`.
- **Why now:** `fg-math` already has clean, zero-overhead fixed-size `Vec3`/`Vec6`/`Mat6`/`Mat12` types (keep those untouched for fast Block-Thomas & `BlockDiagonalPairs`!). External crates deliver the highest ROI where hand-written scalar algorithms hit asymptotic/SIMD walls:
  1. **`faer = "0.24"` (MIT) behind feature `faer-sparse` in `fg-solver`:** Replace/accelerate `SparseGeneral` (`SparseBlockLdltFactor`) with `faer`'s SIMD-vectorized sparse supernodal/simplicial $LDL^\top$ (with AMD fill-reducing ordering) and Bunch-Kaufman $LBL^\top$ for cyclic tensegrity/mesh topologies and dense Delassus compliance solves (`C_c = J_c H^{-1} J_c^T`).
  2. **`parry3d-f64 = "0.31"` (Apache-2.0) behind feature `parry-mesh` in `fg-sdf` / `fg-contact`:** Add general FP64 triangle-mesh (`TriMesh`), convex-hull (`ConvexPolyhedron`), and compound-shape `SdfField` adapters using Parry's SIMD BVH + exact point projection + TOI CCD — unlocking real scanned pipes, cluttered rubble meshes, and CAD obstacles without writing a custom mesh BVH from scratch.
  3. **`clarabel` (`Clarabel.rs`, Apache-2.0) behind optional feature `clarabel-socp` in `fg-contact`:** Add an exact second-order friction-cone (SOCP) fallback in `lcp.rs` alongside PGS/Alart-Curnier for extreme mass/stiffness ratios (e.g., 200 GPa steel beams coupled to rigid plates and soft soil) and as an exact solver oracle in `fg-bench`.
  4. **`urdf-rs = "0.10"` (Apache-2.0) in `fg-robots/src/loader.rs`:** Complement the existing custom `URDF-Flex XML` parser (`loader.rs:453–1123`, which already handles `<link>`, `<joint>`, `<beam>`, `<tendon>`, `<contact_site>`) with full standard ROS URDF tag coverage (`<visual>`, `<collision>`, `<mesh>`, `<mimic>`) so unmodified third-party ROS URDFs load natively (with **MJCF** XML import remaining the primary unimplemented format gap, tracked in §5).

```toml
# Recommended optional workspace additions for rust/flexgraph-rs/Cargo.toml (all MIT / Apache-2.0 / BSD-2)
[workspace.dependencies]
faer = { version = "0.24", default-features = false, features = ["std", "rayon", "sparse"] }
parry3d-f64 = { version = "0.31", features = ["serde-serialize"] }
pyo3 = { version = "0.30", features = ["extension-module"] }
numpy = "0.30"
urdf-rs = "0.10"
clarabel = "0.11"
```
*(Note: pin exact minor versions of `pyo3`/`numpy` to match the installed `rustc` MSRV and `Cargo.lock` when integrating in session `af2331aa`; all crate versions and features above were verified against `crates.io`/`docs.rs` on 2026-11-26: `faer 0.24.4`, `parry3d-f64 0.31.1`, `nalgebra 0.35.0`, `pyo3 0.29.3–0.30.x`, `numpy 0.30.0`, `urdf-rs 0.10.0`, `clarabel 0.11.1`.)*

---

## 4. License Compliance & Clean-Room Guardrails

| Tier | Repositories | Permitted Action |
| :--- | :--- | :--- |
| **Green (Permissive — Direct Dependency / Code Adaptation Safe)** | `faer-rs` (MIT), `parry` / `rapier` / `nalgebra` (Apache-2.0), `Clarabel.rs` (Apache-2.0), `pyo3` / `maturin` / `rayon` (Apache-2.0 OR MIT), `rust-numpy` (BSD-2), `urdf-rs` (Apache-2.0), `dlpack` / `dlpark` / `dlpack-rs` (Apache-2.0), `rsl_rl` (BSD-3), `MorphoSymm` (MIT), `URMA` code (MIT), `IsaacLab` core (BSD-3), `Gymnasium` (MIT), `PufferLib` (MIT), `EnvPool` (Apache-2.0), `PyElastica` (MIT), `ipc-toolkit` (MIT), `Genesis` (Apache-2.0), `Warp` / `Newton` / `MuJoCo` / `MuJoCo Warp` (Apache-2.0), `go2_dreamwaq` (MIT) | Add to `Cargo.toml` / `pyproject.toml`, adapt patterns with attribution, or use as benchmark/verification dependencies. |
| **Red (Copyleft / Non-Commercial / Unlicensed — Clean-Room ONLY)** | `StructuresComp/dismech` & `dismech-python` (**GPL-3.0**), `Sage-PhysicalAI/HIMLoco` (**CC BY-NC-SA 4.0**), `GVSRobotics/SoRoSim` (**No license**), `carlosferrazza/BodyTransformer` (**No license**), `agrimgupta92/metamorph` (**No license**), `romankalyna/dreamwaq-isaaclab` (**No license**) | **NEVER copy, vendor, or link source code.** Only cite published papers, implement mathematical formulations cleanly from scratch, or run external benchmarks outside distributed package artifacts. |

---

## 5. Medium-Term Ecosystem Extensions (Post-v1 Training)

1. **MJCF Subset Bridge (`google-deepmind/mujoco`):** Import/export `<body/>`, `<joint/>`, `<tendon><spatial>`, and `<site/>` between MJCF XML and `RigidFlexGraph` JSON so community locomotion benchmarks run unmodified in FlexGraph.
2. **Cosserat & IPC Verification Suite (`PyElastica` + `ipc-toolkit`):** Add standardized benchmark cases (cantilever buckling, helical tendon bending, rod-on-rod self-contact energy conservation) into `fg-bench` for paper validation tables.
3. **GPU Kernel Escalation Path (`NVIDIA/warp` & `newton-physics/newton`):** Once `fg-ffi` saturates multi-core CPU throughput and remote RTX 5090 RL requires >4,096 batched rigid-flexible environments with general contact, author fused FP64/mixed-precision Block-Thomas + ARPC kernels in `NVIDIA/warp` (Apache-2.0) to eliminate PyTorch eager-mode kernel-launch overhead while preserving the identical `RigidFlexGraph` IR.
