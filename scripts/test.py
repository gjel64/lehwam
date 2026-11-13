import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.encoder import HEncoder
from model.predictor import MoT
from model.hlayer import HLayer
import torch

# --- Data
B = 32
z_dim_below = 192
z_dim = 128
z_dim_above = 96
T = 3
token = torch.rand((B, T, z_dim_below))

mask = torch.ones(4, 4) * float('-inf')
mask[:, :2] = 0
mask[2, 2] = 0
mask[3, 3] = 0
print(mask)
tau_x = torch.rand(B)
h_norm = torch.rand(B)
y = torch.rand(B, 1, z_dim)
x = torch.rand(B, 1, z_dim)
state = torch.rand(B, 1, z_dim)

"""
# --- Encoder
enc = HEncoder(z_dim, z_dim_below, 0.0, depth=2)
z = enc(token)

# --- Predictor
predictor = MoT(z_dim, z_dim, 4, 3, 4, 0.0, True)
z = torch.cat([z, y, state, x], dim=1)
v_pred, z_pred = predictor(z, mask, tau_x, h_norm)
print(v_pred.shape, z_pred.shape)
"""

l = HLayer(z_dim, z_dim_below, z_dim_above, 0.0, 2, 4, 3, True)
z, loss = l.forward_enc(token)
print(z.shape, loss)

v_pred, z_pred, [l1, l2] = l.forward_pre(z[:, :1], z[:, 1:2], torch.randn(B, 1, z_dim_above), h_norm )
print(z_pred.shape, v_pred.shape, l1.shape, l2.shape)