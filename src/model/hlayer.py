import torch
import torch.nn as nn

from .encoder import HEncoder
from .predictor import MoT
from .modules import SIGReg

class HLayer(nn.Module):
    def __init__(self, emb_dim, below_dim, above_dim, dropout_p, enc_depth, n_head, n_block, z_below_detach):
        """
        emb_dim        : dim of this layer
        below_dim      : dim of the layer below
        above_dim      : dim of the layer above
        dropout_p      : dropout pourcentage
        enc_depth      : encoder depth
        n_head         : number of heads for the MoT
        n_block        : number of MoTBock in the MoT
        z_below_detach : if we detach z_below on encoding
        """
        super().__init__()

        self.encoder = HEncoder(emb_dim, below_dim, dropout_p, enc_depth)
        self.predictor = MoT(emb_dim, emb_dim, n_head, 3, n_block, dropout_p, True) # as for h layers there is 3 state 

        self.next_state_query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)

        mask = torch.ones(4, 4) * float('-inf')
        mask[:, :2] = 0
        mask[2, 2] = 0
        mask[3, 3] = 0
        self.register_buffer("mask", mask)

        self.sigreg = SIGReg()

        self.descender_proj = nn.Linear(above_dim, emb_dim)
        self.type_emb = nn.Parameter(torch.randn(2, emb_dim) * 0.02)  # 0: z, 1: y
        self.h_emb = nn.Linear(1, emb_dim)
        self.z_below_detach = z_below_detach

    def forward_enc(self, z_below):
        """
        z_below : (B, T, z_dim_below)

        -> z (B, T, z_dim)
        -> loss scalar
        """

        z = self.encoder(z_below.detach() if self.z_below_detach else z_below)
        loss = torch.stack([self.sigreg(z[:, k].unsqueeze(0)) for k in range(z.shape[1])]).mean()

        return z, loss

    def forward_pre(self, z, z_next, z_goal_above, h_norm):
        """
        z            : (B, 1, d)
        z_next       : (B, 1, d)
        z_goal_above : (B, 1, d_above)
        h_norm       : (B)
        """

        B = z.shape[0]

        # ---- Construction of token 
        # y
        tau_y = (0.7 + 0.3 * torch.rand(B, device=z.device))[:, None, None]
        g = self.descender_proj(z_goal_above.detach())
        eps_y = torch.randn_like(g)
        y = (1 - tau_y) * eps_y + tau_y * g

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
        loss_state = ((z_pred - z_next)**2).mean()
        loss_flow = ((v_pred - (z_next.detach() - eps_x))**2).mean()

        return v_pred, z_pred, [loss_state, loss_flow]









        

        





