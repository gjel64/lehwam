"""
HWAM: a temporal hierarchy of latent-variable JEPAs above LeWAM (IDEA.md)

Each level l >= 1 has an elevator E_l (its latent z^l from a window of K_l latents of level l - 1), a predictor P_l
(LeWAM's MoT block, conditioned on a macro-action u inferred by energy minimization) and a SIGReg regularizer.
The level-0 latent is the LeWAM latent of the goal views, concatenated. Module lists are indexed by level - 1.
"""

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from lewam.models.lewam import MoTBlock
from lewam.models.module import MLP, SIGReg


def sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared distance over the last dim"""
    return ((a - b) ** 2).mean(-1)


class Elevator(nn.Module):
    """E_l: one latent of level l from a window of K_l latents of level l - 1"""

    def __init__(self, kind: str, in_dim: int, out_dim: int, window: int, depth: int, n_heads: int):
        """
        Args:
            kind (str): "transformer" (causal, read out by a last token), "mlp_window" (MLP on the concatenated
                window) or "mlp_frame" (MLP on the last latent of the window, a control without temporal abstraction)
            in_dim (int): the latent dim of level l - 1
            out_dim (int): the latent dim of level l
            window (int): K_l
            depth (int): transformer layers
            n_heads (int): transformer heads
        """
        super().__init__()
        assert kind in ("transformer", "mlp_window", "mlp_frame"), kind
        self.kind = kind
        if kind == "transformer":
            self.readout = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
            self.pos_embed = nn.Parameter(torch.zeros(1, window + 1, in_dim))
            layer = nn.TransformerEncoderLayer(in_dim, n_heads, 4 * in_dim, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.register_buffer("causal_mask", nn.Transformer.generate_square_subsequent_mask(window + 1), persistent=False)
        self.out = MLP(in_dim * window if kind == "mlp_window" else in_dim, 4 * out_dim, out_dim, norm_fn=nn.LayerNorm)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            window (torch.Tensor): (..., K_l, in_dim) latents of level l - 1, oldest first

        Returns:
            z (torch.Tensor): (..., out_dim) latent of level l
        """
        x = window.flatten(0, -3)
        if self.kind == "transformer":
            x = torch.cat([x, self.readout.expand(x.shape[0], -1, -1)], dim=1) + self.pos_embed
            x = self.transformer(x, mask=self.causal_mask, is_causal=True)[:, -1]
        elif self.kind == "mlp_window":
            x = x.flatten(1)
        else:
            x = x[:, -1]
        return self.out(x).view(*window.shape[:-2], -1)


class LevelPredictor(nn.Module):
    """
    P_l: the next latent of level l from its history and the condition (a macro-action u, or raw actions)
    Token order: [history | next-state query | condition], the condition in the action stream of the MoT block
    """

    def __init__(self, dim: int, history_len: int, n_cond: int, cond_dim: int, depth: int, n_heads: int):
        """
        Args:
            dim (int): the latent dim of level l, also the embedding dim
            history_len (int): history latents
            n_cond (int): condition tokens: 1 macro-action, or the raw actions of one step of level l
            cond_dim (int): dim of a condition token
            depth (int): MoT blocks
            n_heads (int): attention heads
        """
        super().__init__()
        self.hist_proj, self.cond_proj = nn.Linear(dim, dim), nn.Linear(cond_dim, dim)
        self.hist_pos_embed = nn.Parameter(torch.zeros(1, history_len, dim))
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, n_cond, dim))
        self.next_state_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(MoTBlock(dim, n_heads, cond_state=False, cond_action=False) for _ in range(depth))
        self.state_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        # history attends to itself, the query to every token, the condition to the history and itself causally
        n_tokens = history_len + 1 + n_cond
        attends = torch.zeros(n_tokens, n_tokens, dtype=torch.bool)
        attends[:history_len, :history_len] = True
        attends[history_len] = True
        attends[history_len + 1:, :history_len] = True
        attends[history_len + 1:, history_len + 1:] = torch.ones(n_cond, n_cond, dtype=torch.bool).tril()
        self.register_buffer("attn_mask", torch.zeros(n_tokens, n_tokens).masked_fill(~attends, float("-inf")), persistent=False)
        self.layout = (history_len + 1, slice(0, 0), slice(0, 0))   # no AdaLN conditioning

    def forward(self, history: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history (torch.Tensor): (B, history_len, dim) latents, oldest first
            cond (torch.Tensor): (B, n_cond, cond_dim) condition tokens

        Returns:
            z_next (torch.Tensor): (B, dim) predicted next latent
        """
        B, history_len = history.shape[:2]
        tokens = torch.cat([
            self.hist_proj(history) + self.hist_pos_embed,
            self.next_state_query.expand(B, -1, -1),
            self.cond_proj(cond) + self.cond_pos_embed
        ], dim=1)
        for block in self.blocks:
            tokens = block(tokens, self.attn_mask, self.layout, None, None)
        return self.state_out(tokens[:, history_len])


class HWAM(nn.Module):
    def __init__(self, cfg: dict):
        """
        Config: the LeWAM checkpoint config, whose `hwam` entry is the `hwam` group of the train config:
            num_levels (int): levels above level 0
            strides (list[int]): K_l, in steps of level l - 1
            dims (list[int]): latent dim of each level
            history_len (int): history latents of each predictor
            elevator (str): "transformer", "mlp_window" or "mlp_frame"
            elevator_depth (int): transformer layers of each elevator
            predictor_depth (int): MoT blocks of each predictor
            action_cond (str): "latent" conditions P_l on an inferred macro-action, "actions" on the raw actions
            u_dim (int): macro-action dim
            u_reg (str): regularizer R of the macro-actions: "sigreg", "l2" or "none"
            u_reg_weight (float): beta, the weight of R in the inference
            infer_steps (int): k, gradient steps of the inference
            infer_lr (float): step size of the inference
            infer_optimizer (str): "sgd" or "adam"
            infer_grad (str): gradient of the outer loss through u*: "detach", "last_step" or "unroll"
        """
        super().__init__()
        hwam_cfg = cfg["hwam"]
        self.num_levels = int(hwam_cfg["num_levels"])
        self.strides, self.dims = list(hwam_cfg["strides"]), list(hwam_cfg["dims"])
        assert len(self.strides) == len(self.dims) == self.num_levels, "hwam.strides and hwam.dims need one entry per level"
        self.history_len = int(hwam_cfg["history_len"])
        self.action_cond, self.u_reg = hwam_cfg["action_cond"], hwam_cfg["u_reg"]
        self.u_reg_weight = float(hwam_cfg["u_reg_weight"])
        self.infer_steps, self.infer_lr = int(hwam_cfg["infer_steps"]), float(hwam_cfg["infer_lr"])
        self.infer_optimizer, self.infer_grad = hwam_cfg["infer_optimizer"], hwam_cfg["infer_grad"]
        assert self.action_cond in ("latent", "actions"), self.action_cond
        assert self.u_reg in ("sigreg", "l2", "none"), self.u_reg
        assert self.infer_optimizer in ("sgd", "adam"), self.infer_optimizer
        assert self.infer_grad in ("detach", "last_step", "unroll"), self.infer_grad

        # level-0 steps per step of each level
        self.spans = [math.prod(self.strides[:level]) for level in range(1, self.num_levels + 1)]
        # shape of the condition of one transition of each level
        self.cond_shapes = [
            (1, int(hwam_cfg["u_dim"])) if self.action_cond == "latent" else (span * cfg["fs"], cfg["action_raw_dim"])
            for span in self.spans
        ]

        z0_dim = len(cfg.get("goal_view_indices", [0])) * cfg["z_dim"]
        self.elevators = nn.ModuleList(
            Elevator(hwam_cfg["elevator"], in_dim, out_dim, stride, hwam_cfg["elevator_depth"], cfg["n_heads"])
            for in_dim, out_dim, stride in zip([z0_dim] + self.dims[:-1], self.dims, self.strides)
        )
        self.predictors = nn.ModuleList(
            LevelPredictor(dim, self.history_len, *cond_shape, hwam_cfg["predictor_depth"], cfg["n_heads"])
            for dim, cond_shape in zip(self.dims, self.cond_shapes)
        )
        self.sigreg = SIGReg()

    # -- Latents --
    @property
    def strip_past(self) -> int:
        """Level-0 latents up to the current one that cover the history of the top level"""
        return self.history_len * self.spans[-1]

    @property
    def strip_ahead(self) -> int:
        """Level-0 latents after the current one that cover the next latent of the top level"""
        return self.spans[-1]

    def elevate(self, z0: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            z0 (torch.Tensor): (B, n, z0_dim) level-0 latents, oldest first, n a multiple of the top span

        Returns:
            levels (list[torch.Tensor]): (B, n / span_l, dim_l) latents of each level, whose windows end at the
                last level-0 latent
        """
        levels, z = [], z0
        for elevator, stride in zip(self.elevators, self.strides):
            z = elevator(z.unflatten(1, (z.shape[1] // stride, stride)))
            levels.append(z)
        return levels

    def current_index(self, level: int) -> int:
        """Index of the current latent of `level` in elevate() of a strip with strip_past latents up to the current one"""
        return self.strip_past // self.spans[level - 1] - 1

    def goals(self, g0: torch.Tensor) -> list[torch.Tensor]:
        """The goal of each level, g^l = E_l(K_l copies of g^(l-1)), from (B, z0_dim) level-0 goal latents"""
        goals = []
        for elevator, stride in zip(self.elevators, self.strides):
            g0 = elevator(g0[:, None].expand(-1, stride, -1))
            goals.append(g0)
        return goals

    # -- Energy and inference --
    def energy(self, level: int, history: torch.Tensor, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(B,) prediction error of P_level for the (B, history_len, dim) histories, (B, *cond_shape) conditions
        and (B, dim) targets"""
        return sq_dist(self.predictors[level - 1](history, cond), target)

    def regularizer(self, u: torch.Tensor) -> torch.Tensor:
        """R of a batch of (B, *cond_shape) macro-actions"""
        if self.u_reg == "none":
            return u.new_zeros(())
        with torch.autocast(u.device.type, enabled=False):
            u = u.float().flatten(1)
            return self.sigreg(u[None]) if self.u_reg == "sigreg" else u.square().sum(-1).mean()

    def infer(self, level: int, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Infer the macro-actions u* = argmin_u sum_i E(u_i) + beta B R(u_1..u_B) with infer_steps gradient steps
        from u = 0, P_level fixed. The objective is summed over the batch, so the step of each u_i does not depend
        on the batch size.

        Args:
            level (int): the level
            history (torch.Tensor): (B, history_len, dim) histories
            target (torch.Tensor): (B, dim) observed next latents

        Returns:
            u (torch.Tensor): (B, *cond_shape) the macro-actions: a constant for infer_grad=detach, else
                differentiable through the last step (last_step) or every step (unroll)
        """
        B = history.shape[0]
        n_differentiable = {"detach": 0, "last_step": 1, "unroll": self.infer_steps}[self.infer_grad]
        u = history.new_zeros(B, *self.cond_shapes[level - 1], dtype=torch.float32)
        adam_m, adam_v = torch.zeros_like(u), torch.zeros_like(u)
        with torch.enable_grad():
            for step in range(self.infer_steps):
                differentiable = step >= self.infer_steps - n_differentiable
                if not differentiable or step == 0:
                    u = u.detach().requires_grad_()
                step_history, step_target = (history, target) if differentiable else (history.detach(), target.detach())
                # the fused attention kernels have no double backward
                with sdpa_kernel(SDPBackend.MATH) if differentiable else nullcontext():
                    objective = self.energy(level, step_history, u, step_target).sum() + self.u_reg_weight * B * self.regularizer(u)
                    grad = torch.autograd.grad(objective, u, create_graph=differentiable)[0]
                if self.infer_optimizer == "adam":
                    adam_m, adam_v = 0.9 * adam_m + 0.1 * grad, 0.999 * adam_v + 0.001 * grad ** 2
                    grad = adam_m / (1 - 0.9 ** (step + 1)) / ((adam_v / (1 - 0.999 ** (step + 1))).sqrt() + 1e-8)
                u = u - self.infer_lr * grad
        return u if n_differentiable else u.detach()

    # -- Training --
    def loss(self, levels: list[torch.Tensor], actions: torch.Tensor, w_reg: float) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Dynamics loss of every level with its inferred macro-actions, and its grouped SIGReg (one group per history
        latent and the target)

        Args:
            levels (list[torch.Tensor]): elevate() of a strip of strip_past + strip_ahead level-0 latents
            actions (torch.Tensor): (B, span_L * fs, action_raw_dim) z-scored actions from the current step
            w_reg (float): SIGReg weight

        Returns:
            loss (torch.Tensor): scalar loss
            loss_terms (dict[str, float]): per level l: h{l}_dyn, h{l}_reg, and the diagnostics h{l}_zstd (std of z^l),
                h{l}_ustd (std of u*), h{l}_gain (E(u*) / E(0)), h{l}_gap (E(u*) with the history of another sample /
                E(u*)) and h{l}_usig (SIGReg statistic of u*)
        """
        loss, loss_terms = 0.0, {}
        for level, z in enumerate(levels, 1):
            current = self.current_index(level)
            history, target = z[:, current - self.history_len + 1:current + 1], z[:, current + 1]
            if self.action_cond == "latent":
                cond = self.infer(level, history, target)
            else:
                cond = actions[:, :self.cond_shapes[level - 1][0]]
            energy = self.energy(level, history, cond, target).mean()
            with torch.autocast(z.device.type, enabled=False):
                loss_reg = self.sigreg(torch.cat([history, target[:, None]], dim=1).float().transpose(0, 1))
            loss = loss + energy + w_reg * loss_reg

            with torch.no_grad():
                loss_terms.update({
                    f"h{level}_dyn": energy.item(), f"h{level}_reg": loss_reg.item(),
                    f"h{level}_zstd": z.float().flatten(0, 1).std(0).mean().item()
                })
                if self.action_cond == "latent":
                    u = cond.float().flatten(1)
                    energy_zero = self.energy(level, history, torch.zeros_like(cond), target).mean()
                    energy_other = self.energy(level, history.roll(1, 0), cond, target).mean()
                    loss_terms.update({
                        f"h{level}_ustd": u.std(0).mean().item(),
                        f"h{level}_gain": (energy / energy_zero.clamp_min(1e-12)).item(),
                        f"h{level}_gap": (energy_other / energy.clamp_min(1e-12)).item(),
                        f"h{level}_usig": self.sigreg(u[None]).item()
                    })
        return loss, loss_terms

    # -- Planning --
    def rollout(self, level: int, history: torch.Tensor, conds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            level (int): the level
            history (torch.Tensor): (B, history_len, dim) histories
            conds (torch.Tensor): (B, T, *cond_shape) one condition per step

        Returns:
            latents (torch.Tensor): (B, T, dim) imagined latents
        """
        latents = []
        for step in range(conds.shape[1]):
            z_next = self.predictors[level - 1](history, conds[:, step])
            latents.append(z_next)
            history = torch.cat([history[:, 1:], z_next[:, None]], dim=1)
        return torch.stack(latents, dim=1)

    def plan(
        self, level: int, history: torch.Tensor, cost_fn, init: torch.Tensor, mode: str = "cem_grad",
        num_proposals: int = 64, cem_iters: int = 3, cem_elites: int = 6, grad_steps: int = 5,
        u_reg_weight: float = 0.1, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Macro-actions that minimize cost_fn(imagined latents) + u_reg_weight * sum_t |u_t|^2: CEM with candidates
        drawn around init with std 1, then Adam (step size infer_lr) on the CEM elites, or on init for mode=grad.
        Returns the cheapest plan seen.

        Args:
            level (int): the level
            history (torch.Tensor): (B, history_len, dim) histories
            cost_fn: (B, n, T, dim) imagined latents of n plans per row -> (B, n) cost
            init (torch.Tensor): (B, T, *cond_shape) warm start, zeros for none
            mode (str): "cem", "grad" or "cem_grad"
            num_proposals (int): CEM candidates per row
            cem_iters (int): CEM rounds
            cem_elites (int): candidates refitting the mean and std each round
            grad_steps (int): Adam steps
            u_reg_weight (float): beta', the weight of |u|^2
            generator (torch.Generator | None): RNG of the CEM draws

        Returns:
            u (torch.Tensor): (B, T, *cond_shape) the cheapest plan of each row
            latents (torch.Tensor): (B, T, dim) its imagined latents
        """
        assert mode in ("cem", "grad", "cem_grad"), mode
        B, device = init.shape[0], init.device
        rows = torch.arange(B, device=device)

        def plan_cost(plans: torch.Tensor) -> torch.Tensor:
            """(B, n, T, *cond_shape) -> (B, n)"""
            n = plans.shape[1]
            latents = self.rollout(level, history.repeat_interleave(n, 0), plans.flatten(0, 1)).unflatten(0, (B, n))
            return cost_fn(latents) + u_reg_weight * plans.flatten(3).square().mean(-1).sum(-1)

        best_plan, best_cost = init.clone(), torch.full((B,), float("inf"), device=device)

        def keep_cheapest(plans: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
            """Store each row's cheapest plan in best_plan; returns the new best cost"""
            round_cost, cheapest = cost.detach().min(1)
            better = round_cost < best_cost
            best_plan[better] = plans.detach()[rows, cheapest][better]
            return torch.minimum(round_cost, best_cost)

        candidates = init[:, None]
        if mode != "grad":
            with torch.no_grad():
                mean, std = init, torch.ones_like(init)
                for _ in range(cem_iters):
                    candidates = mean[:, None] + std[:, None] * torch.randn(
                        B, num_proposals, *init.shape[1:], device=device, generator=generator)
                    candidates[:, 0] = mean
                    cost = plan_cost(candidates)
                    best_cost = keep_cheapest(candidates, cost)
                    elite_indices = cost.topk(min(cem_elites, num_proposals), dim=1, largest=False).indices
                    candidates = candidates[rows[:, None], elite_indices]
                    mean, std = candidates.mean(1), candidates.std(1, correction=0).clamp_min(0.02)

        if mode != "cem":
            plans = candidates.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([plans], lr=self.infer_lr)
            with torch.enable_grad():
                for iteration in range(grad_steps + 1):
                    cost = plan_cost(plans)
                    best_cost = keep_cheapest(plans, cost)
                    if iteration == grad_steps:
                        break
                    # the gradient of the plans only: P_level may be training
                    plans.grad = torch.autograd.grad(cost.sum(), plans)[0]
                    optimizer.step()

        with torch.no_grad():
            return best_plan, self.rollout(level, history, best_plan)

    @staticmethod
    def goal_cost(goal: torch.Tensor, steps_to_goal: torch.Tensor):
        """cost_fn of plan(): the distance of each plan's latent at the goal step to the (B, dim) goal, the goal
        step being steps_to_goal (B,) steps ahead"""
        rows = torch.arange(goal.shape[0], device=goal.device)
        return lambda latents: sq_dist(latents[rows, :, steps_to_goal - 1], goal[:, None])

    def consistency(
        self, level: int, imagined: torch.Tensor, target: torch.Tensor, real: torch.Tensor, elapsed: torch.Tensor
    ) -> torch.Tensor:
        """
        Consistency of a plan of level - 1 with the plan of `level`: |E_level(window) - target|^2, the window being
        the `elapsed` real latents of level - 1 since `level` planned, then the first imagined ones

        Args:
            level (int): the level above the plan
            imagined (torch.Tensor): (B, T >= K, dim) imagined latents of level - 1
            target (torch.Tensor): (B, dim_level) the next imagined latent of `level`
            real (torch.Tensor): (B, K, dim) the last real latents of level - 1, the current one last
            elapsed (torch.Tensor): (B,) steps of level - 1 since `level` planned, in 0..K-1

        Returns:
            cost (torch.Tensor): (B,)
        """
        B, _, dim = imagined.shape
        stride = self.strides[level - 1]
        sequence = torch.cat([real, imagined[:, :stride]], dim=1)
        window_steps = (stride - elapsed)[:, None] + torch.arange(stride, device=imagined.device)
        window = sequence.gather(1, window_steps[:, :, None].expand(B, stride, dim))
        return sq_dist(self.elevators[level - 1](window), target)
