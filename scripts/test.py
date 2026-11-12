import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.encoder import HEncoder
from model.predictor import MoT
import torch

# --- Data
B = 32
z_dim_below = 192
z_dim = 128
T = 1
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

# --- Encoder
enc = HEncoder(z_dim, z_dim_below, 0.0, depth=2)
z = enc(token)

# --- Predictor
predictor = MoT(z_dim, z_dim, 4, 3, 4, 0.0, True)
z = torch.cat([z, y, state, x], dim=1)
v_pred, z_pred = predictor(z, mask, tau_x, h_norm)
print(v_pred.shape, z_pred.shape)