import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.lewam import leWAM
import torch


model = leWAM(192, 2, torch.device("cpu"))


# ------------------------------------------------ Fake data ------------------------------------------------


B = 32
o_history = torch.randint(0, 256, (B, 2, 3, 224, 224), dtype=torch.uint8)
o_target = torch.randint(0, 256, (B, 1, 3, 224, 224), dtype=torch.uint8)
o_goal = torch.randint(0, 256, (B, 1, 3, 224, 224), dtype=torch.uint8)
actions = torch.randn(B, 10, 2)  # (B, 10, act_dim)
h_norm = torch.rand(B)           # chunks remaining / 10
action_valid = torch.ones(B, 10)
action_valid[:4, 7:] = 0 # episode ends
history_pad = torch.zeros(B, 2, dtype=torch.bool)
history_pad[:4, 0] = True # episode starts

losses = model(o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad)
print(losses)

model.eval()

o_history = torch.randint(0, 256, (1, 2, 3, 224, 224), dtype=torch.uint8)
o_goal = torch.randint(0, 256, (1, 1, 3, 224, 224), dtype=torch.uint8)
h_norm = torch.rand(1)
history_pad = torch.zeros(1, 2, dtype=torch.bool)
a_pred = model.generate(o_history, o_goal, h_norm, history_pad, inference_type="direct_policy")
print(a_pred.shape)  # (10, act_dim)
a_pred = model.generate(o_history, o_goal, h_norm, history_pad, inference_type="best-of-K")
print(a_pred.shape)  # (10, act_dim)
a_pred = model.generate(o_history, o_goal, h_norm, history_pad, inference_type="gradient_based")
print(a_pred.shape)  # (10, act_dim)