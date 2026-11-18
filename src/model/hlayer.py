import torch
import torch.nn as nn

from .encoder import HEncoder
from .predictor import MoT
from .modules import SIGReg

class HLayer(nn.Module):
    def __init__(self, cfg, emb_dim, below_dim, above_dim, device):
        """
        emb_dim        : dim of the current layer
        below_dim      : dim of the layer below
        above_dim      : dim of the layer above
        """
        super().__init__()

        self.encoder = HEncoder(emb_dim, below_dim, cfg.enc_dropout_p, cfg.enc_depth)
        self.predictor = MoT(emb_dim, emb_dim, cfg.n_head, 3, cfg.n_block, cfg.dropout_p, True) # as for h layers there is 3 state 

        self.next_state_query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)

        mask = torch.ones(4, 4, device=device) * float('-inf')
        mask[:, :2] = 0
        mask[2, 2] = 0
        mask[3, 3] = 0
        self.register_buffer("mask", mask)

        self.sigreg = SIGReg()

        self.descender_proj = nn.Linear(above_dim, emb_dim)
        self.type_emb = nn.Parameter(torch.randn(2, emb_dim) * 0.02)  # 0: z, 1: y
        self.h_emb = nn.Linear(1, emb_dim)
        self.z_below_detach = cfg.z_below_detach

# ------------------------------ Forward ------------------------------

    def forward_enc(self, z_below, need_loss=True):
        """
        z_below : (B, T, z_dim_below)

        -> z (B, T, z_dim)
        -> loss scalar
        """

        z = self.encoder(z_below.detach() if self.z_below_detach else z_below)

        if not need_loss:
            return z, None
        loss = torch.stack([self.sigreg(z[:, k].unsqueeze(0)) for k in range(z.shape[1])]).mean()

        return z, loss

    def forward_pre(self, z, z_next, z_goal_above, h_norm, need_loss=True):
        """
        z            : (B, 1, d)
        z_next       : (B, 1, d)
        z_goal_above : (B, 1, d_above)
        h_norm       : (B)

        -> v_pred, z_pred, loss_state, loss_flow
        """

        B = z.shape[0]

        # ---- Construction of token 
        # y
        if self.training:
            tau_y = (0.7 + 0.3 * torch.rand(B, device=z.device))[:, None, None]
        else:
            tau_y = 1 # on eval : no noise
        z_g = z_goal_above.detach()
        y = self.descender_proj((1 - tau_y) * torch.randn_like(z_g) + tau_y * z_g)

        # x
        tau_x = torch.rand(B, device=z.device)
        eps_x = torch.randn_like(z_next)
        x = (1 - tau_x[:, None, None]) * eps_x + tau_x[:, None, None] * z_next.detach()

        # state 
        state = self.next_state_query.expand(B, 1, -1)

        # token
        token = torch.cat([z + self.type_emb[0], y + self.type_emb[1], state + self.h_emb(h_norm[:, None, None]), x], dim=1)

        # ---- Forward 
        v_pred, z_pred = self.predictor(
            token, 
            self.mask, 
            tau_x, 
            h_norm
        ) # (B, 4, z_dim) -> (B, 1, z_dim), (B, 1, z_dim)

        # ---- Losses
        if not need_loss :
            return v_pred, z_pred, None, None
        
        loss_state = ((z_pred - z_next)**2).mean()
        loss_flow = ((v_pred - (z_next.detach() - eps_x))**2).mean()

        return v_pred, z_pred, loss_state, loss_flow


# ------------------------------ Generation ------------------------------

    def sample_subgoal(self, z, z_goal_above, h, n_step):
        """
        compute n_step euler step to get z_next that is the goal of the lower layer
        z            : (B, 1, d)
        z_goal_above : (B, 1, d_above)
        h            : (B)
        """

        B = z.shape[0]
        y = self.descender_proj(z_goal_above)
        state = self.next_state_query.expand(B, 1, -1) + self.h_emb(h[:, None, None])

        x = torch.randn_like(z) # what we opti (z_next)

        for i in range(n_step):

            tau = torch.full((B,), i / n_step, device=z.device)
            token = torch.cat([z + self.type_emb[0], y + self.type_emb[1], state, x], 1)
            v, _ = self.predictor(token, self.mask, tau, h)
            x = x + v / n_step

        return x







        

        





