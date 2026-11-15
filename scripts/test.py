import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydra import initialize, compose

with initialize(config_path="../config"):
    cfg = compose(config_name="train")

from model.lehwam import leHWAM

model = leHWAM(cfg.model, 2, torch.device("cpu"))

B = 32
o_history = torch.rand(B, 2, 3, 224, 224)
o_goal = torch.rand(B, 1, 3, 224, 224)
hs = torch.rand(B, 3)

print(model.generate(o_history, o_goal, hs).shape)

