import sys
import torch
torch.manual_seed(42)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydra import initialize, compose

with initialize(config_path="../config"):
    cfg = compose(config_name="train")

from model.lehwam import leHWAM

B = 32
act_dim = 2
device = torch.device("cpu")
model = leHWAM(cfg.model, 2, ks=[5, 25, 125], device=device).to(device)
n = len(model.hlayers)

o_history = torch.rand(B, 2, 3, 224, 224, device=device)
o_goal = torch.rand(B, 2, 3, 224, 224, device=device)
hs = torch.rand(B, n+1, device=device)
goal_offset_steps = torch.randint(0, 200, (B,))

print(model.generate(o_history, o_goal, hs, goal_offset_steps, inference_type="direct_policy").shape)

