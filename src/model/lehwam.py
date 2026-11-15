import torch
import torch.nn as nn
import sys

from .hlayer import HLayer
from .lewam import leWAM
from .modules import attn_mask


class leHWAM(nn.Module):
    def __init__(self, cfg, act_dim, device, h_max=10):
        super().__init__()

        self.lewam = leWAM(cfg, cfg.emb_dims[0], act_dim, device, h_max)

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
        self.n_heuler_step = cfg.planning.n_heuler_step


    def forward(self, frames, actions, action_valid, hs, history_pad):
        """
        frames : (B, 4+2n, 3, H, W) pixels 
        hs     : (B, n+1) h of each level
        """
        n = len(self.hlayers)

        # --- Up (Encoding)
        z, loss_sigreg = self.lewam.forward_enc(frames)
        zs = [z]
        for hl in self.hlayers:
            z, l = hl.forward_enc(z)
            zs.append(z)
            loss_sigreg = loss_sigreg + l

        # --- Down (Predictor) for now teacher forcing (only from dataset and not from above goal)
        mse = []
        for i, hlayer in enumerate(self.hlayers, 1):
            nxt, goal = slice(2 + 2*i, 3 + 2*i), slice(3 + 2*i, 4 + 2*i)
            z_goal = zs[i + 1][:, goal] if i != n else zs[i][:, goal]  # on the top is himself
            _, _, [l_s, l_f] = hlayer.forward_pre(zs[i][:, 1:2], zs[i][:, nxt], z_goal, hs[:, i])
            mse += [l_s, l_f]

        _, _, l1, l2, zstd = self.lewam.forward_pre(
            zs[0][:, :4], actions, history_pad, hs[:, 0], action_valid,
            z_goal_above=zs[1][:, 3:4] if n > 0 else None
        )
        mse = [l1, l2] + mse
        return {"SIGReg": [loss_sigreg], "MSE": mse, "zstd": zstd,
                "total": sum(mse) + self.sigreg_coef * loss_sigreg}



# ------------------------------- Generation -------------------------------

    @torch.no_grad()
    def generate(self, o_history, o_goal, hs, history_pad=None, inference_type="direct_policy"):
        """
        o_history : (B, 2, 3, 224, 224)  2 last images spaced by K^0 steps
        o_goal    : (B, 1, 3, 224, 224)  goal image
        hs        : (B, n+1)             h of each level
        inference_type : "direct_policy" | "best-of-K" | "gradient_based"
        -> (B, 5, act_dim)
        """

        was_training = self.training
        self.eval()
        B = o_history.shape[0]
        if history_pad is None:
            history_pad = torch.zeros(B, 2, dtype=torch.bool, device=o_history.device)

        ##### Encoder Up #####
        z, _ = self.lewam.forward_enc(torch.cat([o_history, o_goal], 1), need_loss=False) # (B, 3, d0)
        zs = [z]
        z = z[:, 1:]                                                                       # (B, 2, d0)
        for hl in self.hlayers:
            z, _ = hl.forward_enc(z, need_loss=False)                                      # (B, 2, d_i)
            zs.append(z)

        match inference_type:
            case "direct_policy":
                x = self._direct_policy(zs, hs[:, -1], history_pad)[:, :5]
            case "persistent_direct_policy":
                x = self._persistent_direct_policy(zs, hs, history_pad)[:, :5]
            case _:
                sys.exit('Choose one inference type')

        if was_training:
            self.train()

        return x

# ------------------------------- Generation methods -------------------------------

    def _direct_policy(self, zs, h_top, history_pad):
        B = zs[0].shape[0]

        z_goal_above = zs[-1][:, 1:2] if len(self.hlayers) > 0 else zs[-1][:, 2:3]

        h = h_top

        for i, hl in enumerate(reversed(self.hlayers)):
            z = zs[-i-1]
            z_goal_above = hl.sample_subgoal(z[:, 0:1], z_goal_above, h, self.n_heuler_step)
            h = torch.ones_like(h_top)

        z_history = zs[0][:, :2]
        z_next = self.lewam.next_state_query.expand(B, 1, -1)
        image_tokens = self.lewam.predictor_adaptor(z_history, z_goal_above, z_next, None, type="images")
        mask = attn_mask(history_pad, self.lewam.mask)

        return self.lewam._direct_policy(B, image_tokens, mask, h)

        
    def _persistent_direct_policy(self, zs, hs, history_pad):
        return None