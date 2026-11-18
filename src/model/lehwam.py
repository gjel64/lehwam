import torch
import torch.nn as nn
import sys

from .hlayer import HLayer
from .lewam import leWAM
from .modules import attn_mask


class leHWAM(nn.Module):
    def __init__(self, cfg, act_dim, device, ks, h_max=10):
        super().__init__()

        self.n = len(cfg.emb_dims) - 1
        self.h_max = ks[1] // ks[0] if len(ks) > 1 else h_max # old LeWAM ckpts : Ks = [K^0] but emb_dims can have several levels
        self.lewam = leWAM(cfg, cfg.emb_dims[0], act_dim, device, self.h_max)

        # Hierarchical layers
        self.hlayers = nn.ModuleList()
        n_hlayer = len(cfg.emb_dims) - 1
        for i in range(1, n_hlayer+1):
            above_dim = cfg.emb_dims[i+1] if i < n_hlayer else cfg.emb_dims[i] # the above dim of the top layer is his own dim
            below_dim = cfg.emb_dims[i-1]
            self.hlayers.append(
                HLayer(cfg, cfg.emb_dims[i], below_dim, above_dim, device)
            )

        self.sigreg_coef = cfg.sigreg_coef
        self.Ks = ks
        self.emb_dims = list(cfg.emb_dims)

        # planning
        self.k = cfg.planning.k
        self.n_heuler_step = cfg.planning.n_heuler_step



    def forward(self, frames, actions, action_valid, hs, history_pad):
        """
        frames : (B, 4+2n, 3, H, W) pixels 
                [o_t-K0, o_t, o_target_0, o_goal_0, o_target_1, o_goal_1, ... o_target_n, o_goal_n]
        hs     : (B, n+1) h of each level
        """

        # --- Up (Encoding)
        z, l = self.lewam.forward_enc(frames)
        sigreg = [l]                                    # one SIGReg per level, to see the collapse of each level
        zs = [z[:, 0:4]]                                # [z_t-K0, z_t, z_tgt_0, z_goal_0]
        z = torch.cat([z[:, 1:2], z[:, 3:]], dim=1)     # [z_t, z_goal_0, z_tgt_1, z_goal_1, ..., z_tgt_n, z_goal_n]
        for hl in self.hlayers:
            z, l = hl.forward_enc(z)                    # level k : [z_t^k, z_goal_{k-1}^k, z_tgt_k^k, z_goal_k^k, z_tgt_{k+1}^k, ...]
            zs.append(z[:, :4])                         # [z_t^k, z_goal_{k-1}^k (goal of the layer below), z_tgt_k^k, z_goal_k^k (used only at the top)]
            z = torch.cat([z[:, 0:1], z[:, 3:]], dim=1) # [z_t^k, z_goal_k^k, z_tgt_{k+1}^k, ...] -> same layout for level k+1
            sigreg.append(l)

        # --- Down (Predictor) teacher forcing : layer k = n-i predicts z_tgt_k^k from z_t^k,
        #     conditioned on z_goal_k^{k+1} (the top is conditioned on its own z_goal_n^n)
        mse = []
        for i, hlayer in enumerate(reversed(self.hlayers)):

            goal_above = zs[-1][:, 3:4] if i == 0 else zs[-i][:, 1:2]

            _, _, l_s, l_f = hlayer.forward_pre(
                zs[-i-1][:, 0:1], zs[-i-1][:, 2:3], goal_above, hs[:, -i-1]
            )
            mse += [l_s, l_f]

        _, _, l1, l2, zstd = self.lewam.forward_pre(
            zs[0][:, :4], actions, history_pad, hs[:, 0], action_valid,
            z_goal_above=zs[1][:, 1:2] if self.n > 0 else None   # z_goal_0^1
        )


        mse = [l1, l2] + mse
        # std over the batch of the target of each level (as LeWAM for level 0) : ~0 = collapse of this level
        zstd = [zstd] + [zs[k][:, 2].detach().std(0).mean() for k in range(1, self.n + 1)]
        return {"SIGReg": sigreg, "MSE": mse, "zstd": zstd,
                "total": sum(mse) + self.sigreg_coef * sum(sigreg)}
        # MSE          : [l_wam state, l_wam action, l^n state, l^n flow, l^n-1 state, l^n-1 flow, ...] (0, top -> down)
        # SIGReg, zstd : [level 0, level 1, ..., level n] (bottom -> up)



# ------------------------------- Generation -------------------------------

    @torch.no_grad()
    def generate(self, o_history, o_goal, hs, steps_to_goal, history_pad=None, inference_type="direct_policy", subgoals=None,
                 n_predicted_steps=5):
        """
        o_history         : (B, 2, 3, 224, 224)  2 last images spaced by K^0 steps
        o_goal            : (B, 1, 3, 224, 224)  goal image
        hs                : (B, n+1)             h of each level (only the one of the top level is used)
        steps_to_goal     : (B,)                 nb of steps left to reach o_goal
        inference_type    : "direct_policy" | "persistent_direct_policy" | "best-of-K" | "gradient_based"
        subgoals          : persistent_direct_policy only, sub-goals of the previous replan, updated in place (see empty_subgoals)
        n_predicted_steps : nb of step predicted by the generate (it can be less than asked)

        -> (B, n_predicted_steps, act_dim)
        """

        was_training = self.training
        self.eval()
        B = o_history.shape[0]
        if history_pad is None:
            history_pad = torch.zeros(B, 2, dtype=torch.bool, device=o_history.device)

        ##### Encoder Up #####
        z, _ = self.lewam.forward_enc(torch.cat([o_history, o_goal], 1), need_loss=False) # (B, 3, d0)
        zs = [z] # (B, 3, d0)
        z = z[:, 1:] # (B, 2, d)
        for hl in self.hlayers:
            z, _ = hl.forward_enc(z, need_loss=False)                                      # (B, 2, d_i)
            zs.append(z)

        match inference_type:
            case "direct_policy":
                x = self._direct_policy(zs, hs, steps_to_goal, history_pad)[0][:, :n_predicted_steps]
            case "persistent_direct_policy":
                x = self._persistent_direct_policy(zs, hs, steps_to_goal, history_pad, subgoals)[:, :n_predicted_steps]
            case "best-of-K":
                x = self._best_of_k(zs, hs, steps_to_goal, history_pad)[:, :n_predicted_steps]
            case "gradient_based":
                x = self._gradient_based(zs, hs, steps_to_goal, history_pad)[:, :n_predicted_steps]
            case _:
                sys.exit('Choose one inference type')

        if was_training:
            self.train()

        return x

# ------------------------------- Generation methods -------------------------------

    def _direct_policy(self, zs, hs, steps_to_goal, history_pad):
        """
        sample z_next from noise, then to level below, and after _direct_policy from lewam
        
        -> (B, 5 act_dim), gs: n * (B, 1, d^n)
        """
        B = zs[0].shape[0]

        # for each traj how many layers needed (from Ks)
        needed_layer = torch.bucketize(steps_to_goal, torch.tensor(self.Ks[1:], device=zs[0].device)) # (B,)
        h_top = hs[torch.arange(B, device=zs[0].device), needed_layer]

        gs = [zs[self.n][:, 1:2] if self.n > 0 else zs[0][:, 2:3]]

        for k in range(self.n, 0, -1):
            g_below = zs[k][:, 1:2].clone()
            idx = torch.where(needed_layer >= k)[0] # trajs that need level k
            if idx.numel() > 0:
                h_k = torch.where(needed_layer[idx] == k, h_top[idx], torch.ones_like(h_top[idx]))
                g_below[idx] = self.hlayers[k-1].sample_subgoal(
                    zs[k][idx, 0:1], gs[-1][idx], h_k, self.n_heuler_step
                )
            gs.append(g_below)

        h0 = torch.where(needed_layer == 0, h_top, torch.ones_like(h_top))
        z_history = zs[0][:, :2]
        z_next = self.lewam.next_state_query.expand(B, 1, -1)
        image_tokens = self.lewam.predictor_adaptor(z_history, gs[-1], z_next, None, type="images")
        mask = attn_mask(history_pad, self.lewam.mask)

        return self.lewam._direct_policy(B, image_tokens, mask, h0), gs

        
    def empty_subgoals(self, B, device):
        return {"z": [torch.zeros(B, 1, d, device=device) for d in self.emb_dims[1:self.n+1]],
                "dist": torch.zeros(B, self.n, dtype=torch.long, device=device),
                "top": torch.full((B,), -1, dtype=torch.long, device=device)}


    def _persistent_direct_policy(self, zs, hs, steps_to_goal, history_pad, subgoals):
        """
        as _direct_policy but don't reconstruct all the goals at each step and has a cache
        """

        B = zs[0].shape[0]
        dist = subgoals["dist"]

        needed_layer = torch.bucketize(steps_to_goal, torch.tensor(self.Ks[1:], device=zs[0].device)) # (B,)
        h_top = hs[torch.arange(B, device=zs[0].device), needed_layer]

        changed = subgoals["top"] != needed_layer   # (B,) goal of the current level changed -> resample below
        subgoals["top"] = needed_layer

        g = zs[self.n][:, 1:2] if self.n > 0 else zs[0][:, 2:3]

        for k in range(self.n, 0, -1):
            g_below = zs[k][:, 1:2].clone()                                 # final goal (if k-1 is the top level)
            used = needed_layer >= k                                        # trajs that need level k
            resample = used & (changed | (dist[:, k-1] < self.Ks[k-1]))
            idx = torch.where(resample)[0]
            if idx.numel() > 0:
                h_k = h_top[idx]
                if k < self.n: # below the top : goal^k is a kept sub-goal
                    h_k = torch.where(needed_layer[idx] == k, h_k, dist[idx, k].float() / self.Ks[k+1])
                subgoals["z"][k-1][idx] = self.hlayers[k-1].sample_subgoal(
                    zs[k][idx, 0:1], g[idx], h_k, self.n_heuler_step
                )
                dist[idx, k-1] = self.Ks[k]
            g_below[used] = subgoals["z"][k-1][used]
            changed = changed | resample
            g = g_below

        h0 = h_top if self.n == 0 else torch.where(needed_layer == 0, h_top, dist[:, 0].float() / self.Ks[1])
        z_history = zs[0][:, :2]
        z_next = self.lewam.next_state_query.expand(B, 1, -1)
        image_tokens = self.lewam.predictor_adaptor(z_history, g, z_next, None, type="images")
        mask = attn_mask(history_pad, self.lewam.mask)

        return self.lewam._direct_policy(B, image_tokens, mask, h0)


    def _best_of_k(self, zs, hs, steps_to_goal, history_pad):
        """
        give the goal from _direct_policy, constrcut K trajectories, select the best
        """

        B, K = zs[0].shape[0], self.k
        dev = zs[0].device
        _, gs = self._direct_policy(zs, hs, steps_to_goal, history_pad)

        needed = torch.bucketize(steps_to_goal, torch.tensor(self.Ks[1:], device=dev))
        h_top = hs[torch.arange(B, device=dev), needed]
        h0 = torch.where(needed == 0, h_top, torch.ones_like(h_top))

        rep = lambda t: t.repeat_interleave(K, 0)
        z_goal, h0 = rep(gs[-1]), rep(h0)
        mask = rep(attn_mask(history_pad, self.lewam.mask))
        z_next = self.lewam.next_state_query.expand(B*K, 1, -1)
        steps = (h0 * self.h_max).round().clamp(min=1).long()

        preds, actions = self.lewam._rollout(B*K, rep(zs[0][:, :2]), z_goal, z_next, mask, h0, int(steps.max()))
        z_end = torch.stack(preds, 1)[torch.arange(B*K, device=dev), steps - 1]   # (B*K, d0)
        if self.n > 0:
            z_end = self.hlayers[0].forward_enc(z_end[:, None], need_loss=False)[0][:, 0]
        cost = ((z_end - z_goal[:, 0]) ** 2).mean(-1)

        best = cost.view(B, K).argmin(1)
        x = torch.cat([a[:, :5] for a in actions], 1).view(B, K, -1, self.lewam.act_dim)
        return x[torch.arange(B, device=dev), best]


    def _gradient_based(self, zs, hs, steps_to_goal, history_pad):
        """
        give the goal from _direct_policy, then LeWAM gradient based planning toward it
        (the z^0 predicted by the rollout are encoded by E^1 to be compared with the goal of level 1)

        -> (B, 5*n_steps, act_dim)
        """

        B, dev = zs[0].shape[0], zs[0].device
        _, gs = self._direct_policy(zs, hs, steps_to_goal, history_pad)

        needed = torch.bucketize(steps_to_goal, torch.tensor(self.Ks[1:], device=dev))
        h_top = hs[torch.arange(B, device=dev), needed]
        h0 = torch.where(needed == 0, h_top, torch.ones_like(h_top))

        z_next = self.lewam.next_state_query.expand(B, 1, -1)
        mask = attn_mask(history_pad, self.lewam.mask)

        encode_up = None
        if self.n > 0: # encoder called directly : forward_enc would detach z_below (z_below_detach) and cut the gradient
            encode_up = lambda z: self.hlayers[0].encoder(z[:, None])[:, 0] # (N, d0) -> (N, d1)

        return self.lewam._gradient_based(B, zs[0][:, :2], gs[-1], z_next, mask, h0, encode_up)


