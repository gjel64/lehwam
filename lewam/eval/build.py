"""The LeWAM eval policy for `eval.mode`: lewam_policy runs it reactively, lewam_plan plans through its dynamics,
lewam_hwam_plan plans through the HWAM levels above it."""

import numpy as np
from omegaconf import DictConfig

from lewam.eval.hwam_planner import HwamPlanner
from lewam.eval.planners import LeWAMPlanner
from lewam.eval.policy import LeWAMPolicy
from lewam.models.hwam import HWAM
from lewam.models.lewam import LeWAM


def build_policy(
    eval_cfg: DictConfig, model: LeWAM, model_cfg: dict, seed: int, transform: dict,
    goal_offsets: list[int] | None = None, hwam: HWAM | None = None
) -> LeWAMPolicy:
    """
    Args:
        eval_cfg (DictConfig): the `eval` section of the eval config
        model (LeWAM): the trained model, on its device
        model_cfg (dict): the checkpoint config
        seed (int): seed of the flow-matching draws
        transform (dict): per-key transforms applied to the info dict
        goal_offsets (list[int] | None): each episode's goal offset in env steps (full-trajectory protocol)
        hwam (HWAM | None): the trained levels above LeWAM, for lewam_hwam_plan

    Returns:
        policy (LeWAMPolicy): a LeWAMPolicy, a LeWAMPlanner for lewam_plan, or a HwamPlanner for lewam_hwam_plan
    """
    frameskip = int(model_cfg["fs"])
    h0 = eval_cfg.h0
    if h0 is None:
        h0 = (np.asarray(goal_offsets, np.float64) / frameskip if goal_offsets is not None
              else float(eval_cfg.goal_offset_steps) / frameskip)
    shared = dict(model=model, cfg=model_cfg, frameskip=frameskip, action_dim=int(model_cfg["action_raw_dim"]),
                  h0=h0, transform=transform, flow_seed=seed)

    if eval_cfg.mode == "lewam_policy":
        return LeWAMPolicy(num_exec_actions=eval_cfg.exec_actions, **shared)
    assert eval_cfg.mode in ("lewam_plan", "lewam_hwam_plan"), \
        f"unknown eval.mode={eval_cfg.mode!r}: expected lewam_policy, lewam_plan or lewam_hwam_plan"
    if eval_cfg.mode == "lewam_hwam_plan":
        shared.update(
            hwam=hwam,
            hwam_plan_mode=eval_cfg.hwam_plan_mode,
            hwam_num_proposals=eval_cfg.hwam_num_proposals,
            hwam_cem_iters=eval_cfg.hwam_cem_iters,
            hwam_grad_steps=eval_cfg.hwam_grad_steps,
            hwam_u_reg_weight=eval_cfg.hwam_u_reg_weight,
            hwam_warm_start=eval_cfg.hwam_warm_start,
            hwam_replan=eval_cfg.hwam_replan,
            hwam_level0_cost=eval_cfg.hwam_level0_cost)
    planner = HwamPlanner if eval_cfg.mode == "lewam_hwam_plan" else LeWAMPlanner
    return planner(
        plan_mode=eval_cfg.plan_mode,
        num_proposals=eval_cfg.num_proposals,
        rollout_steps=eval_cfg.rollout_steps,
        exec_actions_per_plan=eval_cfg.exec_actions,
        plan_random_candidates=eval_cfg.plan_random_candidates,
        plan_goal_time=eval_cfg.plan_goal_time,
        grad_steps=eval_cfg.grad_steps,
        grad_lr=eval_cfg.grad_lr,
        grad_clip=eval_cfg.grad_clip,
        grad_tr=eval_cfg.grad_tr,
        grad_action_clip=eval_cfg.grad_action_clip,
        grad_all_k=eval_cfg.grad_all_k,
        steer_steps=eval_cfg.steer_steps,
        steer_lr=eval_cfg.steer_lr,
        steer_max_bias=eval_cfg.steer_max_bias,
        steer_k=eval_cfg.steer_k,
        steer_env_batch=eval_cfg.steer_env_batch,
        cem_iters=eval_cfg.cem_iters,
        cem_elites=eval_cfg.cem_elites,
        cem_std=eval_cfg.cem_std,
        cem_init=eval_cfg.cem_init,
        **shared)
