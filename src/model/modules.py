import torch
import torch.nn as nn

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        dt = 3 / (knots - 1) # size of one step
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32) # a vector full of "2 * dt" value
        # 2 * dt because f(-t) = f(t) so 2 * 0->3 = -3->3
        weights[[0, -1]] = dt # trapeziums rules (intergral -> ∫≈dt​[(f0​+f1​)+(f1​+f2​)+(f2​+f3​)+(f3​+f4​)]) -> need 2dt on all excpet on 0 and -1

        t = torch.linspace(0, 3, knots, dtype=torch.float32) # tensor of "knots" value from 0 to 3 linearly
        # choice of 3 :
        # - after 3 useless : w(3) = e−4.5 = 0.011 -> 1%
        window = torch.exp(-t.square() / 2.0) # values of CF of standard normal
        self.register_buffer("t", t)  # (knots)
        self.register_buffer("phi", window)  # (knots)
        self.register_buffer("weights", weights * window) # simplify the code of the forward by precomputing it    # (knots)


    def forward(self, proj):
        """
        proj: (T, B, D)
        D -> emb dimmension
        B -> batch of image
        T -> augmented view (type of changes of images)
        """
        ####### sample random projections #######
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device) # A = random tensor (D, num_proj)
        A = A / (A.norm(dim=0)) # -> var = 1 -> in circle
        # 1024 view of the tensor
        # Cramer-Wold : z follow N(0, I_D) and ||u|| = 1 <=> u_T @ z follow N(0, 1)
        # so we norm


        ####### compute the epps-pulley statistic #######
        x_t = (proj @ A).unsqueeze(-1) * self.t # project proj into num_proj random dimension   # (T, B, num_proj, 1) -> (T, B, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        # x_t.cos().mean(-3)     # (T, num_proj, knots)
        # err                    # (T, num_proj, knots)
        statistic = (err @ self.weights) * proj.size(-2)
        # (err @ self.weights)   # (T, num_proj)
        # statistic              # (T, num_proj)
        return statistic.mean()       # (1)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, depth=1, dropout_p=0.0, norm_fn=None):
        """
        l -> nb of layer (GELU + dp + Linear)
        """
        assert depth >= 1
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim != None else 4*in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
        )

        for i in range(depth-1):
            self.mlp.append(nn.GELU())
            self.mlp.append(nn.Dropout(dropout_p))
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
        # as depth >= 1
        self.mlp.append(norm_fn(hidden_dim) if norm_fn is not None else nn.Identity())
        self.mlp.append(nn.GELU())
        self.mlp.append(nn.Dropout(dropout_p))
        self.mlp.append(nn.Linear(hidden_dim, out_dim))
        
    def forward(self, x):
        """
        x: (B*T, z_dim)
        """
        return self.mlp(x)

def attn_mask(history_pad, mask):
    """(x, x) -> (B, 1, x, x)"""
    x = mask.shape[0]
    mask = mask.expand(history_pad.shape[0], 1, x, x).clone()
    mask[history_pad[:, 0], :, :, 0] = float("-inf")
    return mask