import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # project root, to import src

from src.lewam import leWAM
import torch


model = leWAM(192, 2, torch.device("cpu"))


# ------------------------------------------------ Fake data ------------------------------------------------


B = 32
o_history = torch.randint(0, 256, (B, 2, 3, 224, 224), dtype=torch.uint8)
o_target = torch.randint(0, 256, (B, 1, 3, 224, 224), dtype=torch.uint8)
o_goal = torch.randint(0, 256, (B, 1, 3, 224, 224), dtype=torch.uint8)
actions = torch.randn(B, 10, 2)  # (B, 10, act_dim)
h_norm = torch.rand(B)           # chunks remaining / 10


losses = model(o_history, o_target, o_goal, actions, h_norm)
print(losses)

a_pred = model.generate(o_history, o_goal, h_norm)
print(a_pred.shape)  # (B, 10, act_dim)
