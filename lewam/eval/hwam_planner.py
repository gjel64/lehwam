"""Hierarchical planning with HWAM above LeWAM"""

import math

import numpy as np
import torch

from lewam.eval.planners import LeWAMPlanner
from lewam.models.hwam import HWAM
from lewam.models.lewam import LeWAM


class HwamPlanner(LeWAMPlanner):
    """
    Hierarchical planner (mode=lewam_hwam_plan, IDEA.md 4.7): the top level of each env plans macro-actions to the goal,
    each level below plans the consistency of its plan with the next imagined latent of the level above, through the
    elevator, and level 0 is the LeWAM planner with this consistency as its cost. The top level of an env is the highest
    level whose step is not longer than the steps left to the goal; an env without one is planned as by LeWAMPlanner.

    Replanning:
    - `every_chunk`: every level replans at every replan of level 0, warm-started from its previous plan
    - `stride`: level l replans every K_l steps of level l - 1, and when the level above replans
    """

    def __init__(
        self, model: LeWAM, cfg: dict, *args, hwam: HWAM, hwam_plan_mode: str = "cem_grad", hwam_num_proposals: int = 64,
        hwam_cem_iters: int = 3, hwam_grad_steps: int = 5, hwam_u_reg_weight: float = 0.1, hwam_warm_start: bool = True,
        hwam_replan: str = "every_chunk", hwam_level0_cost: str = "consistency", **kwargs
    ):
        """
        Args:
            model (LeWAM): the trained model
            cfg (dict): LeWAM checkpoint config
            hwam (HWAM): the trained levels above LeWAM
            hwam_plan_mode (str): planner of the levels above: "cem", "grad" or "cem_grad"
            hwam_num_proposals (int): CEM candidates per env
            hwam_cem_iters (int): CEM rounds
            hwam_grad_steps (int): Adam steps
            hwam_u_reg_weight (float): beta', the weight of |u|^2 in the planning cost
            hwam_warm_start (bool): start each plan from the previous plan of the level, shifted by the elapsed steps
            hwam_replan (str): "every_chunk" or "stride"
            hwam_level0_cost (str): "consistency", "consistency_goal" (their sum) or "goal" (the LeWAM cost)
            *args, **kwargs: LeWAMPlanner's arguments; cem_elites also applies to the levels above
        """
        super().__init__(model, cfg, *args, **kwargs)
        assert self.plan_mode != "steer", "steer conditions the policy on a biased goal, not on the HWAM plan"
        assert hwam_replan in ("every_chunk", "stride"), hwam_replan
        assert hwam_level0_cost in ("consistency", "consistency_goal", "goal"), hwam_level0_cost
        assert self.exec_actions_per_plan == self.frameskip, "HWAM replans level 0 after every chunk"
        self.hwam = hwam.eval()
        self.hwam_plan_kwargs = dict(
            mode=hwam_plan_mode, num_proposals=hwam_num_proposals, cem_iters=hwam_cem_iters, cem_elites=self.cem_elites,
            grad_steps=hwam_grad_steps, u_reg_weight=hwam_u_reg_weight
        )
        self.hwam_warm_start, self.hwam_replan, self.hwam_level0_cost = hwam_warm_start, hwam_replan, hwam_level0_cost
        self.type = f"lewam_hwam_plan_{self.plan_mode}"
        # level 0 imagines at least the window of one level-1 latent; the strip of real latents covers the top history
        self.rollout_steps = max(self.rollout_steps, hwam.strides[0])
        self.latent_window = max(self.latent_window, (hwam.strip_past - 1) * self.frameskip)

    def set_env(self, env) -> None:
        super().set_env(env)
        self._level_plans = [{} for _ in range(getattr(env, "num_envs", 1))]   # per env: {level: its last plan}

    def _flush_env(self, i: int) -> None:
        super()._flush_env(i)
        self._level_plans[i].clear()

    def _propose(self, info_dict: dict, replan: list[int], history: torch.Tensor, history_pad: torch.Tensor) -> torch.Tensor:
        """Plan the levels above, then level 0 with LeWAMPlanner and the costs of _plan_cost."""
        # the strip of real level-0 latents, one every frameskip steps, the current one last, clamped to the episode start
        n_strip = self.hwam.strip_past
        env_steps = [(i, max(int(self._step[i]) - k * self.frameskip, 0)) for i in replan for k in range(n_strip - 1, -1, -1)]
        self._encode_missing(env_steps)
        strip = torch.stack([self._latents[i][s] for i, s in env_steps]).view(len(replan), n_strip, len(self.view_keys), -1)
        strip = strip[:, :, self.goal_view_indices].flatten(2)
        self._plan_levels(replan, strip, self._goal_latents(info_dict, replan).flatten(1))
        return super()._propose(info_dict, replan, history, history_pad)

    def _plan_levels(self, replan: list[int], strip: torch.Tensor, g0: torch.Tensor) -> None:
        """
        Plan the levels above level 0, top-down, and set the level-0 cost inputs of each replanning env

        Args:
            replan (list[int]): the replanning envs
            strip (torch.Tensor): (R, strip_past, z0_dim) real level-0 latents, the current one last
            g0 (torch.Tensor): (R, z0_dim) level-0 goal latents
        """
        hwam, fs = self.hwam, self.frameskip
        levels, goals = hwam.elevate(strip), hwam.goals(g0)
        steps_left = np.maximum(self._steps_left[replan], 1.0)
        top = [max([level for level, span in enumerate(hwam.spans, 1) if span <= left], default=0) for left in steps_left]
        replanned = [False] * len(replan)
        for level in range(hwam.num_levels, 0, -1):
            span = hwam.spans[level - 1]
            rows = [r for r, i in enumerate(replan) if top[r] >= level and self._needs_replan(i, level, top[r], replanned[r])]
            replanned = [r in rows for r in range(len(replan))]

            goal_rows = [r for r in rows if top[r] == level]
            if goal_rows:
                steps_to_goal = torch.tensor([math.ceil(steps_left[r] / span) for r in goal_rows], device=self.device)
                self._plan_level(level, [replan[r] for r in goal_rows], [top[r] for r in goal_rows],
                                 levels[level - 1][goal_rows], hwam.goal_cost(goals[level - 1][goal_rows], steps_to_goal),
                                 int(steps_to_goal.max()))

            consistency_rows = [r for r in rows if top[r] > level]
            if consistency_rows:
                envs = [replan[r] for r in consistency_rows]
                above = [self._level_plans[i][level + 1] for i in envs]
                target = torch.stack([plan["z_next"] for plan in above])
                elapsed = torch.tensor([(self._step[i] - plan["step"]) // (span * fs) for i, plan in zip(envs, above)], device=self.device)
                window = hwam.strides[level]
                real = levels[level - 1][consistency_rows, -window:]

                def consistency(latents: torch.Tensor) -> torch.Tensor:
                    n = latents.shape[1]
                    return hwam.consistency(level + 1, latents.flatten(0, 1), target.repeat_interleave(n, 0),
                                            real.repeat_interleave(n, 0), elapsed.repeat_interleave(n, 0)).view(-1, n)
                self._plan_level(level, envs, [top[r] for r in consistency_rows], levels[level - 1][consistency_rows],
                                 consistency, window)

        # level 0: consistency with the next imagined level-1 latent, also the subgoal; g^1 for envs without HWAM plan
        target, elapsed = goals[0].clone(), torch.zeros(len(replan), dtype=torch.long, device=self.device)
        for r, i in enumerate(replan):
            if top[r]:
                plan = self._level_plans[i][1]
                target[r], elapsed[r] = plan["z_next"], (self._step[i] - plan["step"]) // fs
        self._level0 = dict(active=torch.tensor(top, device=self.device) > 0, target=target,
                            real=strip[:, -hwam.strides[0]:], elapsed=elapsed)

    def _needs_replan(self, i: int, level: int, top: int, replanned_above: bool) -> bool:
        """Whether env i replans `level`: always for every_chunk; for stride, when its plan is missing, from another
        top level, K_l steps of level l - 1 old, or the level above replanned"""
        plan = self._level_plans[i].get(level)
        return (self.hwam_replan == "every_chunk" or plan is None or plan["top"] != top or replanned_above
                or self._step[i] - plan["step"] >= self.hwam.spans[level - 1] * self.frameskip)

    def _plan_level(self, level: int, envs: list[int], tops: list[int], history: torch.Tensor, cost_fn, horizon: int) -> None:
        """Plan `horizon` macro-actions of `level` for the envs from their (n_envs, ..., dim) latents of the level, the
        current one last, and store each plan"""
        plans, latents = self.hwam.plan(level, history[:, -self.hwam.history_len:], cost_fn,
                                        torch.stack([self._warm_start(i, level, horizon) for i in envs]),
                                        generator=self._flow_generator(self.device), **self.hwam_plan_kwargs)
        for row, (i, top) in enumerate(zip(envs, tops)):
            self._level_plans[i][level] = dict(u=plans[row], z_next=latents[row, 0], step=int(self._step[i]), top=top)

    def _warm_start(self, i: int, level: int, horizon: int) -> torch.Tensor:
        """(horizon, *cond_shape) the previous plan of env i at `level` without its elapsed steps, zero-padded; zeros
        without warm start"""
        init = torch.zeros(horizon, *self.hwam.cond_shapes[level - 1], device=self.device)
        plan = self._level_plans[i].get(level)
        if self.hwam_warm_start and plan is not None:
            elapsed = (self._step[i] - plan["step"]) // (self.hwam.spans[level - 1] * self.frameskip)
            kept = plan["u"][elapsed:][:horizon]
            init[:len(kept)] = kept
        return init

    def _plan_cost(self, latents_per_step: torch.Tensor, z_goal: torch.Tensor, steps_to_goal: torch.Tensor | None) -> torch.Tensor:
        """The level-0 cost of hwam_level0_cost for the envs with an HWAM plan, the LeWAM goal cost for the others."""
        if steps_to_goal is not None:   # hwam_level0_cost=consistency imagines K_1 steps: the goal of an env without HWAM plan
            steps_to_goal = steps_to_goal.clamp(max=latents_per_step.shape[1])
        goal_cost = super()._plan_cost(latents_per_step, z_goal, steps_to_goal)
        if self.hwam_level0_cost == "goal":
            return goal_cost
        level0 = self._level0_rows(latents_per_step.shape[0])
        z0 = latents_per_step if latents_per_step.dim() == 3 else latents_per_step[:, :, self.goal_view_indices].flatten(2)
        cost = self.hwam.consistency(1, z0, level0["target"], level0["real"], level0["elapsed"])
        if self.hwam_level0_cost == "consistency_goal":
            cost = cost + goal_cost
        return torch.where(level0["active"], cost, goal_cost)

    def _level0_rows(self, num_rows: int) -> dict:
        """The level-0 cost inputs with each env's row repeated for its num_rows / n_envs proposals"""
        repeat = num_rows // self._level0["target"].shape[0]
        return {key: value.repeat_interleave(repeat, 0) for key, value in self._level0.items()}

    def _dyn_steps_needed(self, steps_to_goal: torch.Tensor | None) -> int:
        """The consistency cost needs the window of the next level-1 latent; the goal cost needs the goal step."""
        if self.hwam_level0_cost == "consistency":
            return self.hwam.strides[0]
        needed = super()._dyn_steps_needed(steps_to_goal)
        return needed if self.hwam_level0_cost == "goal" else max(needed, self.hwam.strides[0])

    def _goal_inputs(self, z_goal: torch.Tensor, steps_to_goal: torch.Tensor | None, step: int, num_rows: int) -> dict:
        """The policy's goal inputs, and the next imagined level-1 latent as its subgoal with hwam.subgoal_token"""
        inputs = super()._goal_inputs(z_goal, steps_to_goal, step, num_rows)
        if self.model.subgoal_dim:
            inputs["z_subgoal"] = self._level0_rows(num_rows)["target"]
        return inputs
