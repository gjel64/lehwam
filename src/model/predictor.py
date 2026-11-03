import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# ------------------------------------------------ Utils ------------------------------------------------


def on_slots(x, neutral, n_tokens=19, slots=slice(9, 19)):
    """ 
    create tensor with `x` values for the slice and neutral for the rest 
    x : (B, z_dim) -> (B, n_tokens, z_dim)
    """
    full = x.new_full((x.shape[0], n_tokens, x.shape[1]), neutral) # (B, n_tokens, z_dim) of neutral value
    full[:, slots] = x.unsqueeze(1)                                # (B, n_tokens, z_dim)
    return full

def per_stream(x, n_state, f_state, f_action):
    """ f_action and f_state : nn.Linear(z_dim, 3*z_dim) """
    gauche = f_state(x[:, :n_state])
    droite = f_action(x[:, n_state:])
    return torch.cat([gauche, droite], dim=1)

def sinusoid(x, dim):
    """ creates cat(96 sin, 96 cos) exponencially from 1 to 1/10000 """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
    a = x[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


# ------------------------------------------------ Predictor ------------------------------------------------


class MoTBlock(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(emb_dim, elementwise_affine=False)

        [self.qkv_state, self.qkv_action] = [nn.Linear(emb_dim, 3*emb_dim) for i in range(2)]
        [self.out_state, self.out_action] = [nn.Linear(emb_dim, emb_dim) for i in range(2)]

        [self.mlp_state, self.mlp_action] = [nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim*4, emb_dim)
        ) for _ in range(2)]

        self.ada_fns = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 6 * emb_dim))

        nn.init.zeros_(self.ada_fns[-1].weight)
        nn.init.zeros_(self.ada_fns[-1].bias)

        self.dropout_p = dropout
        
        
    def forward(self, x, mask, cond):
        """
        x    : (B, n_token, z_dim)    n_token tokens = h0 h1 S G | a0...a4 | x0...x9
        mask : (B, 1, n_token, n_token)
        """
        B, n_tokens, z_dim = x.shape 

        h = self.norm1(x)                                                                  # (B, n_token, z_dim)
        # calc the vectors for adaLN-Zero
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_fns(cond).chunk(6, dim=-1) # (B, z_dim) each
        # +1 to become identity at init
        h = (1 + on_slots(scale1, 0)) * h + on_slots(shift1, 0)                            # (B, n_token, z_dim)
    
        qkv = per_stream(h, 4, self.qkv_state, self.qkv_action)                            # (B, n_token, 3*z_dim) 3 because q k v
    
        qkv = qkv.reshape(B, n_tokens, 3, 4, 48)                                           # (B, n_token, 3, n_head, 48) 
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                                               # 3 * (B, n_head, n_token, 48)

        dp = self.dropout_p if self.training else 0.0
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)        # (B, n_head, n_token, 48)
        att = att.transpose(1, 2)                                                          # (B, n_token, n_head, 48)
        att = att.reshape(B, n_tokens, z_dim)                                              # (B, n_token, z_dim)
    
        x = x + on_slots(gate1, 1) * per_stream(att, 4, self.out_state, self.out_action)   # (B, n_token, z_dim)
    
        h = self.norm2(x)     
        h = (1 + on_slots(scale2, 0)) * h + on_slots(shift2, 0)                            # (B, n_token, z_dim)
        x = x + on_slots(gate2, 1) * per_stream(h, 4, self.mlp_state, self.mlp_action)     # (B, n_token, z_dim)
    
        return x                                                                           # (B, n_token, z_dim)



class MoT(nn.Module):
    def __init__(self, emb_dim, act_dim):
        super().__init__()
        self.blocks = nn.ModuleList([MoTBlock(emb_dim) for i in range(4)])
        self.action_out = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, act_dim))
        self.state_out = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, emb_dim))

        self.emb_dim = emb_dim
        self.tau_proj = nn.Linear(emb_dim, emb_dim)
        self.h_proj = nn.Linear(emb_dim, emb_dim)

        # to test at init as DiT
        # nn.init.zeros_(self.action_out[-1].weight)
        # nn.init.zeros_(self.action_out[-1].bias)

    def forward(self, token, mask, tau, h_norm):

        cond = self.tau_proj(sinusoid(tau, self.emb_dim)) + self.h_proj(sinusoid(h_norm, self.emb_dim))

        x = token
        for block in self.blocks:
            x = block(x, mask, cond)

        v_pred = self.action_out(x[:, 9:19])    # flow speed
        z_pred = self.state_out(x[:, 2:3])      # ẑ_{t+5}
        return v_pred, z_pred


