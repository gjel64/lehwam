# HWAM: Hierarchical World-Action Modeling for Long-Horizon Planning with Latent-Variable JEPAs

## 1. Summary

LeWAM learns a world model and a policy in one transformer. It predicts one dynamics step ahead (one chunk of `frameskip` actions). Its planner rolls this one-step model forward in the latent space. Over long horizons, the prediction errors accumulate, and the policy sees horizons that it did not see in training.

HWAM adds a temporal hierarchy of JEPAs above LeWAM:

1. Each level $l$ computes a slow latent $z^l$ from a time window of the level below. The module that does this is the **elevator** $E_l$.
2. Each level has a JEPA **predictor** $P_l$ and its own SIGReg loss.
3. The predictor of each level above level 0 uses a **latent variable** $u^l$, the macro-action. No network computes $u^l$. The model **infers** $u^l$ by minimization of the prediction energy.
4. SIGReg on the inferred latents makes the distribution of $u^l$ close to $\mathcal{N}(0, I)$.
5. **Training and planning solve the same problem.** In training, the model finds the $u$ that explains the observed future. In planning, the model finds the $u$ that reaches the goal.
6. The levels communicate by **consistency through the elevators**. Level $l-1$ plans a trajectory whose elevated latent matches the plan of level $l$. HWAM has no subgoal decoder.

Each level contains only an encoder (the elevator), a predictor and a regularizer. These are the three parts of a JEPA.

HWAM can train in three ways (Section 5): on top of a frozen LeWAM, with a subgoal-tuned LeWAM, or jointly from scratch.

Main claim: HWAM improves success on long-horizon tasks and decreases the planning cost. It does not decrease success on short-horizon tasks.

## 2. Motivation

LeWAM is almost saturated on the standard short-horizon protocol. Its success drops on long horizons and on multi-step tasks.

Results of the released LeWAM checkpoints (gradient planner, mean ± std over seeds 42, 0, 1):

| Setting | Result | Observation |
|---|---|---|
| Cube, TwoRoom, Reacher, PointMaze | 100.0 | Saturated |
| Push-T, H = 25 | 98.7 ± 1.2 | Almost saturated |
| Push-T random start, H = 50 | 92.0 ± 4.0 (reactive: 64.7) | Planning is necessary |
| Push-T long horizon, H = 100 | 67.3 ± 7.6 | Large drop |
| Push-T long horizon, H = 200 | 58.7 ± 4.2 | Large drop |
| Puzzle 3x3 | 78.7 ± 9.5 (reactive: 76.7) | Planning adds almost nothing |
| Drawer Cleanup | 60.0 ± 7.2 (reactive: 64.0) | Planning decreases success |

We think that three causes explain these drops:

1. **Compounding error.** The dynamics head trains on one step only (`num_states_pred = 1`). A 200-step plan needs 40 autoregressive steps.
2. **Horizon clamp.** The policy trains with goals up to `H_max = 10` chunks (50 env steps). Beyond that, the horizon input stays at its maximum value.
3. **Weak cost signal.** The planner cost is the squared distance in $z$ at one time step. On Puzzle, this distance does not measure progress well.

HWAM addresses causes 1 and 2 directly. Each upper level predicts further with fewer autoregressive steps. Level 0 always receives a target inside its training horizon. Section 9 lists cause 3 as an extension.

## 3. Background

### 3.1 LeWAM

- **Encoder.** A ResNet18 (GroupNorm, random crop) with a 32-keypoint spatial softmax and an MLP projector gives $z_t \in \mathbb{R}^{192}$ for each image.
- **Anti-collapse.** LeWAM uses no EMA target encoder. SIGReg (Sketch Isotropic Gaussian Regularizer) pushes each group of latents toward an isotropic Gaussian.
- **MoT transformer.** One transformer has two weight streams (state and action) and one shared attention. A fixed attention mask separates two tasks:
  - The **policy** predicts an action chunk from the history, the goal and the horizon. It uses flow matching.
  - The **dynamics** predicts the next latent from the history and the clean actions. It does not see the goal. It uses an MSE head.
- **Planner.** The planner imagines 32 policy rollouts in the latent space. It refines them with 5 Adam steps on the cost $\lVert \hat z_{t_g} - z_g \rVert^2$. It executes the first chunk of the best plan.

### 3.2 Latent-variable JEPA

In a latent-variable JEPA (LeCun, 2022), the predictor gets a latent variable $u$ in addition to the context. The variable $u$ carries the information about the target that the context does not contain. The model infers $u$ by minimization of an energy:

$$u^* = \arg\min_u \; \lVert P(s_x, u) - s_y \rVert^2 + \beta R(u)$$

The regularizer $R(u)$ limits the information in $u$. Without this limit, $u$ can encode the full target, and the predictor ignores the context.

GLO (Bojanowski et al., 2018) and the auto-decoder of DeepSDF (Park et al., 2019) also find latent codes by optimization instead of with an encoder. They show that this training method is stable in practice.

## 4. Method

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| $L$ | Number of levels above level 0 |
| $z^0_t$ | Level-0 latent. This is the LeWAM encoder output. |
| $K_l$ | Temporal stride of level $l$, in steps of level $l-1$ |
| $z^l_t$ | Level-$l$ latent at level-$l$ step $t$ |
| $u^l_t$ | Level-$l$ latent variable (macro-action) between steps $t$ and $t+1$ |
| $g^l$ | Goal in the latent space of level $l$ |
| $\mathrm{sg}(\cdot)$ | Stop-gradient |

Time units for Push-T: a level-0 step is one chunk (5 env steps). With $K_1 = 5$, a level-1 step is 25 env steps. A goal 200 env steps ahead needs 40 level-0 steps or 8 level-1 steps.

### 4.2 Elevator: from $z^{l-1}$ to $z^l$

The elevator $E_l$ reads a window of $K_l$ consecutive latents of the level below:

$$z^l_t = E_l\big(z^{l-1}_{tK_l - K_l + 1}, \dots, z^{l-1}_{tK_l}\big)$$

The windows do not overlap. One elevator, with one set of weights, computes all the latents of its level: the history, the current latent and the prediction target.

The window is the important part. A per-frame network only compresses one frame. It does not change the time scale, so it does not decrease the number of planning steps.

HWAM supports three elevators:

| Elevator | Description |
|---|---|
| `transformer` (default) | A small causal transformer (2 layers, $d = 192$) over the $K_l$ tokens, with a read-out token |
| `mlp_window` | An MLP on the concatenated window |
| `mlp_frame` | An MLP on the last frame of the window. This elevator has no temporal abstraction. It is a control. |

The goal latent of level $l$ is the elevator output for a window of $K_l$ copies of the goal latent of level $l-1$:

$$g^0 = z_g, \qquad g^l = E_l(g^{l-1}, \dots, g^{l-1})$$

### 4.3 Predictor with a latent variable

The predictor of level $l \ge 1$ uses the same MoT block as LeWAM. Its state stream reads the history $z^l_{\le t}$. Its action stream reads one token, the latent variable $u^l_t \in \mathbb{R}^{d_u}$, in the position of the clean actions. It predicts the next latent:

$$\hat z^l_{t+1} = P_l(z^l_{\le t}, u^l_t)$$

The energy of a transition is:

$$\mathcal{E}_l(u) = \lVert P_l(z^l_{\le t}, u) - z^l_{t+1} \rVert^2$$

The predictor does not see the goal. The future $z^l_{t+1}$ is never an input of a network. It is only the target of the energy, as in all JEPAs.

### 4.4 Inference of the latent variable

For each training batch, HWAM infers $u^*$ with a short inner optimization:

1. Set $u = 0$ for each sample.
2. Freeze the parameters of $P_l$ for the inner loop.
3. Do $k$ steps of gradient descent (or Adam) on $u$ for the inner objective below.
4. Use $u^*$, the result of step 3, in the outer loss.

Inner objective, over a batch of $B$ samples:

$$\min_{u_1, \dots, u_B} \; \frac{1}{B}\sum_{i=1}^{B} \mathcal{E}_l(u_i) + \beta\, R(u_1, \dots, u_B)$$

HWAM supports two regularizers:

| Regularizer | $R$ | Effect |
|---|---|---|
| `sigreg` (default) | $\mathrm{SIGReg}(u_1, \dots, u_B)$ | The inferred latents of the batch become isotropic Gaussian |
| `l2` | $\frac{1}{B}\sum_i \lVert u_i \rVert^2$ | MAP inference with a Gaussian prior. One sample does not affect the others. |

The outer loss trains $P_l$ and $E_l$ with $u^*$. HWAM supports three gradient modes:

| Gradient mode | Gradient | Cost |
|---|---|---|
| `detach` (default) | $u^*$ is a constant. Same as EM, GLO and DeepSDF. | Low |
| `last_step` | The last inner step is differentiable. The gradient goes through $\nabla_u \mathcal{E}$ into $P_l$. | Medium |
| `unroll` | All inner steps are differentiable | High |

### 4.5 Training objective

Losses of level $l \ge 1$:

$$\mathcal{L}^l_{\text{dyn}} = \lVert P_l(z^l_{\le t}, u^*) - z^l_{t+1} \rVert^2$$

$$\mathcal{L}^l_{\text{reg}} = \frac{1}{|G_l|} \sum_{G \in G_l} \mathrm{SIGReg}(G)$$

$G_l$ is the set of latent groups of level $l$: each history latent, the current latent and the target latent. This is the grouped SIGReg of LeWAM, applied at level $l$.

The SIGReg term on $z^l$ is necessary. Without it, nothing prevents the elevator from collapsing to a constant, because a constant $z^l$ gives a zero energy. As in LeWAM, the target $z^l_{t+1}$ has no stop-gradient. The gradient of the outer loss goes into $P_l$ and into $E_l$ on both sides, the context and the target. SIGReg prevents the collapse.

Total loss:

$$\mathcal{L} = \mathcal{L}_{\text{LeWAM}} + w_{\text{hwam}}(p) \sum_{l=1}^{L} \Big( \mathcal{L}^l_{\text{dyn}} + w_{\text{reg}} \mathcal{L}^l_{\text{reg}} \Big)$$

$\mathcal{L}_{\text{LeWAM}}$ is the action, state and SIGReg loss of LeWAM. Its use depends on the training regime (Section 5). $w_{\text{hwam}}(p)$ is a warmup weight of the training progress $p$.

### 4.6 Isotropic macro-actions

With the `sigreg` regularizer, the inferred macro-actions have a distribution close to $\mathcal{N}(0, I)$. This has a direct use in planning. A sampling planner (CEM, MPPI) can draw candidates from $\mathcal{N}(0, I)$, and these candidates stay in the distribution that $P_l$ saw in training.

Random raw actions do not have this property. In LeWAM, random candidates on raw actions (`plan_random_candidates`) are a weak baseline, because random raw action sequences are far from the demonstrations.

### 4.7 Hierarchical planning

The planner uses the same energy as the training. The decision variables are the macro-actions $u_{1:T}$. The target is the goal, not an observed future.

**Level $L$ (the top level).** This is the only level that sees the goal.

$$\min_{u_{1:T}} \; \lVert \hat z^L_{T} - g^L \rVert^2 + \beta' \sum_{\tau=1}^{T} \lVert u_\tau \rVert^2, \qquad \hat z^L_{\tau+1} = P_L(\hat z^L_{\le \tau}, u_\tau)$$

$T$ is the number of level-$L$ steps to the goal. The planner draws the first candidates from $\mathcal{N}(0, I)$ (CEM). Then it refines the best candidates with Adam, as in the inner loop of Section 4.4. The result is an imagined trajectory $\hat z^L_{t+1}, \hat z^L_{t+2}, \dots$

**Level $l-1$ (below level $l$).** The target is the next imagined latent of level $l$. The cost is a **consistency cost** through the elevator:

$$\mathcal{C}_{l-1} = \big\lVert E_l\big(\hat z^{l-1}_{t+1}, \dots, \hat z^{l-1}_{t+K_l}\big) - \hat z^l_{t+1} \big\rVert^2$$

The elevator is the interface between the levels. HWAM needs no subgoal decoder.

**Level 0.** The LeWAM gradient planner minimizes $\mathcal{C}_0$ instead of the goal cost. The LeWAM policy gives the first proposals. Its goal input depends on the training regime:

- Without a subgoal token, the policy receives the final goal $z_g$, with the horizon clamped at $H_{\max}$.
- With a subgoal token (Section 4.8), the policy receives the imagined subgoal $\hat z^1_{t+1}$.

When the goal is less than $K_1$ chunks away, level 0 uses the direct goal cost, as in LeWAM.

**Replanning.** Level 0 replans after each executed chunk. Level $l$ replans every $K_l$ steps of level $l-1$, or at each level-0 replan with a warm start from the previous plan.

### 4.8 Subgoal token and scheduled sampling

A subgoal token lets the LeWAM policy take a level-1 latent as its goal. Without this token, only the consistency cost of the planner guides level 0.

At inference, the subgoal is an imagined latent $\hat z^1_{t+1}$. At training, the real latent is available. This is a train/test mismatch. Scheduled sampling (Bengio et al., 2015) reduces it. For each sample, the subgoal comes from one of two sources:

- With probability $1 - \lambda$: the **hindsight** subgoal $E_1(z^0_{t+1..t+K_1})$, from the real future of the same demonstration.
- With probability $\lambda$: the **generated** subgoal $\mathrm{sg}(\hat z^1_{t+1})$, from the planner of level 1.

$\lambda$ follows a renormalized sigmoid of the training progress $p \in [0, 1]$:

$$\lambda(p) = \lambda_{\max} \cdot \frac{\sigma(s(p - c)) - \sigma(-sc)}{\sigma(s(1 - c)) - \sigma(-sc)}$$

Parameters: $c$ is the center of the transition, $s$ is the sharpness and $\lambda_{\max}$ is the final value. The renormalization gives $\lambda(0) = 0$ and $\lambda(1) = \lambda_{\max}$ exactly.

Two properties of the schedule:

1. $\lambda$ changes the **source of the subgoals**. It does not change the weights of the losses. All losses stay active for the full training. If the loss on real data goes to zero, nothing keeps the model close to the data, and level 0 can drift.
2. We expect $\lambda_{\max} \approx 0.7$ to $0.8$ to be better than $\lambda_{\max} = 1$. Some real targets act as a regularizer.

### 4.9 Training diagnostics

HWAM logs these values for each level and each epoch:

1. **Latent std** of $z^l$ and of $u^*$. A value near zero shows a collapse.
2. **Inference gain** $\mathcal{E}_l(u^*) / \mathcal{E}_l(0)$. A value near 1 shows that $u$ carries no information. A value near 0 can show that $u$ carries all the information.
3. **History gap**: $\mathcal{E}_l(u^*)$ with the real history, compared with $\mathcal{E}_l(u^*)$ with a history from another sample. If the two values are equal, the predictor ignores the history, and $u$ encodes the full target.
4. **SIGReg statistic of $u^*$**. It measures the distance of the inferred macro-actions to $\mathcal{N}(0, I)$.

## 5. Three ways to train HWAM

| | Frozen base | Subgoal-tuned base | Joint |
|---|---|---|---|
| Start | Released LeWAM checkpoint | Released LeWAM checkpoint | Random initialization |
| Encoder | Frozen | Frozen | Trained |
| LeWAM (level 0) | Frozen | Policy fine-tuned with a subgoal token | Trained |
| Upper levels | Trained | Trained | Trained |
| $\mathcal{L}_{\text{LeWAM}}$ | Not used | Used, with scheduled subgoals (Section 4.8) | Used |
| Input data | Cached latents $z^0$ | Cached latents $z^0$ | Images |
| Cost | Low | Medium | High |

### 5.1 Frozen base

HWAM trains only the elevators and the predictors of the levels $l \ge 1$. LeWAM and its encoder do not change. Because the encoder is frozen, HWAM computes $z^0$ once for the full dataset and caches it. The training reads latents, not images, so it is fast.

This regime adds a hierarchy to any trained LeWAM. It cannot decrease the quality of level 0.

### 5.2 Subgoal-tuned base

HWAM adds the subgoal token to the LeWAM policy and fine-tunes this policy with scheduled subgoals (Section 4.8). The encoder stays frozen. Level 0 learns to reach the subgoals of level 1 directly, in addition to the guidance of the consistency cost.

### 5.3 Joint

HWAM trains all modules together from a random initialization. The gradients of the upper levels go into the encoder. The latents can capture more information that is useful at long horizons. But the upper losses can also decrease the quality of level 0.

## 6. Configuration

### 6.1 Rules

Each component of HWAM is a configuration option. Each option can change or disable its component from the command line:

1. Each component has its own config key.
2. With `hwam.enabled=false`, HWAM is exactly LeWAM. The released checkpoints load with `strict=True` and give the same results.
3. Each ablation in Section 7.3 is one command-line override, for example `hwam.elevator=mlp_window`.
4. The training script writes all the keys to `lewam_config.json`. The eval script reads them back and builds the same model.
5. Configs and checkpoints without the HWAM keys use the default values.

### 6.2 Options

Model (`hwam:` group):

| Key | Values | Default | Ablation |
|---|---|---|---|
| `hwam.enabled` | `true`, `false` | `false` | HWAM on or off |
| `hwam.num_levels` | int ≥ 1 | `1` | 1 |
| `hwam.strides` | list of $K_l$, in steps of the level below | `[5]` | 2 |
| `hwam.dims` | list of $z^l$ dimensions | `[128]` | Tuning |
| `hwam.history_len` | int | `2` | Tuning |
| `hwam.elevator` | `transformer`, `mlp_window`, `mlp_frame` | `transformer` | 3 |
| `hwam.elevator_depth` | int | `2` | Tuning |
| `hwam.predictor_depth` | int | `2` | Tuning |
| `hwam.action_cond` | `latent`, `actions` | `latent` | 4 |
| `hwam.u_dim` | int | `16` | 6 |
| `hwam.u_reg` | `sigreg`, `l2`, `none` | `sigreg` | 5 |
| `hwam.u_reg_weight` | float ($\beta$) | `0.1` | Tuning |
| `hwam.infer_steps` | int ($k$) | `5` | 7 |
| `hwam.infer_lr` | float | `0.1` | Tuning |
| `hwam.infer_optimizer` | `sgd`, `adam` | `adam` | Tuning |
| `hwam.infer_grad` | `detach`, `last_step`, `unroll` | `detach` | 7 |
| `hwam.subgoal_token` | `true`, `false` | `false` | 10 |
| `model.action_head_init` | `noise`, `mse` | `noise` | 12 |

Training (`train:` group):

| Key | Values | Default | Ablation |
|---|---|---|---|
| `train.init_from` | run name or `null` | `null` | Start from a released checkpoint |
| `train.freeze_encoder` | `true`, `false` | `true` | 11 |
| `train.freeze_lewam` | `true`, `false` | `true` | 10 |
| `train.cache_latents` | `true`, `false` | `true` | Speed only. It needs `freeze_encoder=true`. |
| `train.w_hwam` | float | `1.0` | Weight of the HWAM losses |
| `train.hwam_warmup_epochs` | int | `0` | Warmup of $w_{\text{hwam}}$ |
| `train.subgoal_source` | `hindsight`, `generated`, `scheduled` | `hindsight` | 10 |
| `train.hwam_schedule` | `constant`, `linear`, `sigmoid` | `sigmoid` | 10 |
| `train.lam_max` | float in [0, 1] | `0.8` | 10 |
| `train.lam_center` | float in [0, 1] | `0.5` | Sigmoid only |
| `train.lam_sharpness` | float | `10.0` | Sigmoid only |

The three training regimes of Section 5 are three combinations of these keys:

| Regime | `init_from` | `freeze_encoder` | `freeze_lewam` | `subgoal_token` |
|---|---|---|---|---|
| Frozen base | checkpoint | `true` | `true` | `false` |
| Subgoal-tuned base | checkpoint | `true` | `false` | `true` |
| Joint | `null` | `false` | `false` | `false` or `true` |

Evaluation (`eval:` group):

| Key | Values | Default | Ablation |
|---|---|---|---|
| `eval.mode` | adds `lewam_hwam_plan` | `lewam_plan` | HWAM planner on or off |
| `eval.hwam_plan_mode` | `cem`, `grad`, `cem_grad` | `cem_grad` | 8 |
| `eval.hwam_num_proposals` | int | `64` | Compute curve |
| `eval.hwam_cem_iters` | int | `3` | Compute curve |
| `eval.hwam_grad_steps` | int | `5` | Compute curve |
| `eval.hwam_u_reg_weight` | float ($\beta'$) | `0.1` | Tuning |
| `eval.hwam_warm_start` | `true`, `false` | `true` | 8 |
| `eval.hwam_replan` | `stride`, `every_chunk` | `every_chunk` | Replanning rule (Section 4.7) |
| `eval.hwam_level0_cost` | `consistency`, `consistency_goal`, `goal` | `consistency` | 9 |

## 7. Experimental protocol

### 7.1 Benchmarks and metrics

| Setting | Why | Protocol |
|---|---|---|
| Push-T long horizon, H ∈ {50, 75, 100, 200} | Main result | LeWAM long-horizon protocol (`docs/horizons/PUSHT_HORIZONS.md`) |
| Push-T random start, H up to 100 | Horizons beyond `H_max` | LeWAM protocol |
| Puzzle 3x3 | Multi-step task with weak latent cost | LeWAM protocol |
| Drawer, Transport | The LeWAM planner is weak on these tasks | LeWAM protocol |
| Standard short-horizon table | No regression | LeWAM protocol |
| Longer OGBench tasks (puzzle 4x4, cube-double) | Stronger long-horizon evidence | OGBench protocol |

Metrics:

1. Success rate, with 50 episodes for each of the seeds 42, 0 and 1, as in the LeWAM paper.
2. Success as a function of the number of model forward passes per replan.
3. Wall-clock time per replan.

### 7.2 Baselines

- LeWAM (released checkpoints, all planners)
- LeWM
- DINO-WM
- HIQL (hierarchical subgoals, offline goal-conditioned RL)
- Diffusion Policy (reactive)

The goal is to measure baselines ourself but if we cannot, we report the numbers from the papers.

### 7.3 Hypotheses and ablations

Hypotheses:

1. HWAM increases success at long horizons (H ≥ 100) and keeps the short-horizon success of LeWAM.
2. Random macro-actions from $\mathcal{N}(0, I)$ are good planning proposals with the `sigreg` regularizer, and weak proposals with `l2` or without a regularizer.
3. A macro-action has a stable meaning across states. Thus a warm start of the plan from the previous replan helps.
4. The temporal window of the elevator is necessary. The `mlp_frame` elevator gives no long-horizon gain.

Ablations:

1. Number of levels: $L \in \{1, 2\}$.
2. Stride: $K_1 \in \{2, 5, 10\}$ chunks.
3. Elevator: `transformer`, `mlp_window`, `mlp_frame`.
4. Condition of the upper predictor: latent variable $u$, or the $K_l \cdot$ `frameskip` raw actions as clean action tokens.
5. Regularizer of $u$: `sigreg`, `l2`, none. CEM with $\mathcal{N}(0, I)$ proposals in each case.
6. Dimension of $u$: $d_u \in \{4, 8, 16, 32\}$.
7. Inference: $k \in \{1, 3, 10\}$ inner steps. Gradient mode `detach`, `last_step`, `unroll`.
8. Top-level planner: CEM, Adam, CEM then Adam. Warm start or not.
9. Level-0 cost: consistency only, consistency and goal, goal only (LeWAM).
10. Training regime: frozen base or subgoal-tuned base. For the subgoal-tuned base: subgoal source (hindsight, generated, scheduled), schedule shape (constant, linear, sigmoid) and $\lambda_{\max} \in \{0.5, 0.8, 1.0\}$.
11. Encoder: frozen (frozen base) or trained (joint).
12. Level-0 action head: flow, or flow that starts from an MSE prediction. The MSE start can decrease the number of ODE steps from 8 to 2 or 3.

## 8. Failure modes

| Failure mode | Sign | Mechanism that prevents it |
|---|---|---|
| $u$ encodes the full target | History gap near zero (Section 4.9) | Small $d_u$. Regularizer $R(u)$. This is the known failure of latent-variable JEPAs. |
| Collapse of upper levels | Std of $z^l$ near zero | Grouped SIGReg on each level. A `zstd_floor` check on each level stops the run. |
| Bad inner optimization | Inference gain stays near 1 | More inner steps, Adam, higher `infer_lr` |
| High training cost | Slow epochs | $k$ inner steps cost about $k$ forward and backward passes of $P_l$. Cached latents and a small $P_l$ keep this cost low. |
| Plans outside the training distribution | Low top-level cost, but level 0 cannot follow the plan | $\beta' \lVert u \rVert^2$ in the planning cost. Trust region (`grad_tr`). |
| Level 0 cannot follow the consistency cost | High $\mathcal{C}_0$ after planning | Subgoal-tuned base (subgoal token, scheduled sampling) |
| Not enough long windows | Few training samples at level 1 on tasks with short episodes | A level-1 sample needs $(H_1 + 1) \cdot K_1$ chunks. The stride $K_1$ must match the episode lengths of the dataset. |

The gain of HWAM is expected at long horizons only. On the saturated short-horizon tasks, the expected result is no regression.

## 9. Extensions

- Learned temporal-distance head (a quasimetric value) as the planning cost, for Puzzle.
- Multi-step dynamics loss at level 0 (unrolled training with teacher forcing).
- Ensemble of predictors, to penalize uncertain plans.
- A goal-conditioned prior on $u$ at the top level, trained on the inferred $u^*$, to give better first proposals to the planner.

## 10. Glossary

| Term | Definition |
|---|---|
| Chunk | `frameskip` consecutive env actions. One level-0 dynamics step. |
| Consistency cost | The distance between the elevated plan of level $l-1$ and the plan of level $l$ |
| Elevator | The module $E_l$ that computes $z^l$ from a window of $z^{l-1}$. It is the encoder of level $l$. |
| Energy | The prediction error of a JEPA for a given latent variable |
| Hindsight subgoal | A real future latent from the same demonstration, used as a goal at training time |
| JEPA | Joint-Embedding Predictive Architecture. The model predicts future latents, not pixels. |
| Latent variable | An input of the predictor that the model infers by minimization of the energy, not with an encoder |
| Macro-action | The latent variable $u^l$ of an upper level. One macro-action covers $K_l$ steps of the level below. |
| MoT | Mixture of Transformers. One transformer with separate weights for the state and action tokens. |
| Scheduled sampling | A training method that slowly replaces real inputs with the model's own predictions |
| SIGReg | Sketch Isotropic Gaussian Regularizer. A loss that pushes a set of vectors toward $\mathcal{N}(0, I)$. |
| Subgoal token | An input token of the LeWAM policy that receives a level-1 latent as the goal |
