import numpy as np
import torch
from collections import deque
from stable_worldmodel.policy import BasePolicy


# ------------------------------------------------ Utils ------------------------------------------------


def last_frame(x):
    """(n_envs, T, 3, H, W) -> (n_envs, 3, H, W) : the world can give a history of T frames"""
    return x[:, -1] if x.ndim == 5 else x


# ------------------------------------------------ Policy ------------------------------------------------


class LeWAMPolicy(BasePolicy):
    """
    wrap leWAM.generate as a stable_worldmodel policy (LeWAMPolicy + LeWAMPlanner in LeWAM)
    for each env we keep :
        - the frames of the last {stride} steps to build the history (t-5, t) as in data.py
        - the actions left to execute before the next replan
        - the nb of chunks left to reach the goal (-> h_norm)
    """

    def __init__(self, model, action_mean, action_std, inference_type, h0, exec_actions=5, stride=5, h_max=10, transform=None):
        """
        model          : leWAM on its device
        action_mean    : (act_dim,) saved in the ckpt by train.py, to un-normalize the actions
        action_std     : (act_dim,)
        inference_type : "direct_policy" | "best-of-K" | "gradient_based"
        h0             : nb of chunks to reach the goal at the start (goal_offset_steps / stride)
        exec_actions   : nb of actions executed before replanning
        """
        super().__init__()
        self.type = inference_type
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.mean = torch.tensor(action_mean, dtype=torch.float32)
        self.std = torch.tensor(action_std, dtype=torch.float32)
        self.inference_type = inference_type
        self.h0 = float(h0)
        self.exec_actions = exec_actions
        self.stride, self.h_max = stride, h_max
        self.transform = transform or {}

        assert exec_actions > 0, "need at least 1 action per replan"
        # direct_policy only gives the first chunk (5 actions), planners give n_steps*5
        assert inference_type != "direct_policy" or exec_actions <= 5, "direct_policy only gives 5 actions"

    def set_env(self, env):
        """called by world.set_policy : one buffer per env"""
        self.env = env
        n = env.num_envs
        self.buffers = [deque() for _ in range(n)]      # actions to execute
        self.frames = [{} for _ in range(n)]            # {step: (3, H, W)} of the last {stride} steps
        self.steps = np.zeros(n, dtype=np.int64)        # steps since the start of the episode
        self.chunks_left = np.full(n, self.h0)          # chunks to reach the goal

    def _flush(self, i):
        """env i was reset by the world"""
        self.buffers[i].clear()
        self.frames[i].clear()
        self.steps[i] = 0
        self.chunks_left[i] = self.h0

# ------------------------------- Get action -------------------------------

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        """
        called by the world at every step
        info_dict["pixels"], info_dict["goal"] : (n_envs, T, H, W, 3) uint8
        -> (n_envs, act_dim) one action per env
        """
        n = self.env.num_envs

        # only with reset_mode="auto" (not for the dataset eval), pop it so we don't flush at every step
        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in np.nonzero(np.asarray(flush, dtype=bool).reshape(n, -1)[:, -1])[0]:
                self._flush(i)

        info = self._prepare_info(info_dict) # pixels/goal -> (n_envs, T, 3, H, W) still uint8 (see transform)

        # finished envs aren't stepped anymore by the world (reset_mode="wait")
        dead = np.asarray(info.get("terminated", np.zeros(n)), dtype=bool).reshape(n, -1)[:, -1]
        alive = [i for i in range(n) if not dead[i]]
        replan = [i for i in alive if len(self.buffers[i]) == 0]

        # --- keep the current frame of each env
        pixels = last_frame(info["pixels"])                                           # (n_envs, 3, H, W)
        for i in alive:
            s = int(self.steps[i])
            self.frames[i][s] = pixels[i].clone()
            self.frames[i] = {k: f for k, f in self.frames[i].items() if k >= s - self.stride}

        # --- new actions for the envs with an empty buffer
        if replan:
            a = self._plan(info, replan)[:, :self.exec_actions]                       # (R, exec_actions, act_dim)
            a = a.float().cpu() * self.std + self.mean                                # un-normalize as in data.py
            for r, i in enumerate(replan):
                self.buffers[i].extend(a[r])
                self.chunks_left[i] = max(self.chunks_left[i] - a.shape[1] / self.stride, 1.0) # as LeWAM

        # --- one action per env
        action = torch.full((n, self.mean.shape[0]), float("nan"))                    # dead envs aren't stepped
        for i in alive:
            action[i] = self.buffers[i].popleft()
            self.steps[i] += 1
        return action.reshape(*self.env.action_space.shape).numpy()

    def _plan(self, info, replan):
        """
        build the inputs of leWAM.generate for the {R} envs that need new actions
        -> (R, n_actions, act_dim) normalized actions
        """
        R = len(replan)
        o_history = []
        history_pad = torch.zeros(R, 2, dtype=torch.bool)
        for r, i in enumerate(replan):
            s = int(self.steps[i])
            pad = s < self.stride                       # no frame at t-5 at the start of the episode
            t_prev = s if pad else s - self.stride      # as in data.py
            history_pad[r, 0] = pad
            o_history.append(torch.stack([self.frames[i][t_prev], self.frames[i][s]]))

        o_history = torch.stack(o_history).to(self.device)                            # (R, 2, 3, H, W)
        o_goal = last_frame(info["goal"])[replan].unsqueeze(1).to(self.device)        # (R, 1, 3, H, W)
        h_norm = np.minimum(self.chunks_left[replan], self.h_max) / self.h_max        # chunks remaining / 10 as in data.py
        h_norm = torch.tensor(h_norm, dtype=torch.float32, device=self.device)        # (R,)

        return self.model.generate(o_history, o_goal, h_norm, history_pad.to(self.device), self.inference_type)
