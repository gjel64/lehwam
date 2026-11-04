import torch
from torch.utils.data import DataLoader, random_split
from functools import partial
from contextlib import nullcontext
import hydra
from omegaconf import DictConfig


from src.data.data import DatasetLeHWAM
from src.model.lewam import leWAM

@hydra.main(config_path="../config", config_name="train")
def train(cfg : DictConfig):
    model_cfg, train_cfg, data_cfg = cfg["model"], cfg["train"], cfg["data"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    ds = DatasetLeHWAM(data_cfg.path_h5p, data_cfg.path_npy, stride=5)
    val_ds, train_ds = random_split(ds, [0.1, 0.9], generator=torch.Generator().manual_seed(42)) # first eval 10 then train 90 as in LeWAM

    kw = dict(batch_size=train_cfg.batch_size, num_workers=data_cfg.num_workers, pin_memory=(device =="cuda"), 
              persistent_workers=(data_cfg.num_workers != 0), prefetch_factor=4 if data_cfg.num_workers != 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    torch.backends.cudnn.benchmark = True
    autocast = partial(torch.autocast, device_type=device.type, dtype=torch.bfloat16) if train_cfg.use_bf16 else nullcontext

    model = leWAM(model_cfg.z_dim, 2, device).to(device)

    # Rest of training - not working just to note

    print(f"[LOG]: Training starts ------------")
    for i in range(train_cfg.epochs):
        print(f"[LOG]: epoch {i}")
        for batch in train_loader:

            model.train()
            o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad = (
                x.to(device, non_blocking=True) for x in batch)
            with autocast():
                losses = model(o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad)

            # the rest...
            torch.mps.synchronize() # temporary



    # End
    torch.save({"model": model.state_dict(), "action_mean": ds.mean, "action_std": ds.std}, data_cfg.ckpt_path)

if __name__ == "__main__":
    train()