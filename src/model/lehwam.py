import torch
import torch.nn as nn

from .hlayer import HLayer
from .lewam import leWAM

class leHWAM(nn.Module):
    def __init__(self, cfg, act_dim, device):
        super().__init__()

        self.lewam = leWAM(cfg, cfg.emb_dims[0], act_dim, device)

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
            z_goal_above=zs[1][:, 3:4]
        )
        mse = [l1, l2] + mse
        return {"SIGReg": [loss_sigreg], "MSE": mse, "zstd": zstd,
                "total": sum(mse) + self.sigreg_coef * loss_sigreg}


        



        
            

