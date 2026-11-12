import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.encoder import HEncoder
import torch

B = 32
z_dim_below = 192
z_dim = 128

enc = HEncoder(z_dim, z_dim_below, 0.0, depth=2)

token = torch.rand((B, 1, z_dim_below))

z = enc(token)

print(z.shape)