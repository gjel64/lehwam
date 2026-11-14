import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import math
from functools import partial
from contextlib import nullcontext
import hydra
from omegaconf import DictConfig
import os

from src.data.data import DatasetLeHWAM
from src.model.lewam import leWAM

def lr_scale(e, W, E):
    # linear warmup and cosine decay to 0
    if e < W: return (e + 1) / W
    return 0.5 * (1 + math.cos(math.pi * (e - W) / max(1, E - W)))

@hydra.main(config_path="../config", config_name="train")
def train(cfg : DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # --- Load config
    model_cfg, train_cfg, data_cfg = cfg["model"], cfg["train"], cfg["data"]
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    # --- Load dataset
    ds = DatasetLeHWAM(data_cfg.path_h5p, data_cfg.path_npy, stride=5)
    val_ds, train_ds = random_split(ds, [data_cfg.eval_part, 1-data_cfg.eval_part], generator=torch.Generator().manual_seed(train_cfg.seed)) # first eval 10 then train 90 as in LeWAM
    kw = dict(batch_size=train_cfg.batch_size, num_workers=data_cfg.num_workers, pin_memory=(device.type=="cuda"), 
              persistent_workers=(data_cfg.num_workers != 0), prefetch_factor=4 if data_cfg.num_workers != 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    # --- Create model
    model = leWAM(model_cfg, model_cfg.emb_dims[0], 2, device).to(device)

    # --- params
    torch.backends.cudnn.benchmark = True
    autocast = partial(torch.autocast, device_type=device.type, dtype=torch.bfloat16) if train_cfg.use_bf16 else nullcontext
    logs = {
        "t_loss" : [],
        "t_mse" : [],
        "t_sigreg": [],
        "e_loss" : [],
        "e_mse" : [],
        "e_sigreg": [],
        "lr" : [],
        "t_n": [],   # size of train batch
        "e_n": [],   # size of eval batch
        "t_zstd" : [],
        "e_zstd" : []
    }
    optimizer = optim.AdamW(model.parameters(), train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, partial(lr_scale, W=train_cfg.warmup_epochs, E=train_cfg.epochs))

    best_loss = float('inf')
    current_loss = None

    # if we start with a checkpoint
    start_epoch, path = 0, data_cfg.ckpt_path + "latest.pt"
    if train_cfg.resume and os.path.exists(path):
        s = torch.load(path, map_location=device)
        model.load_state_dict(s["model"])
        optimizer.load_state_dict(s["optimizer"])
        scheduler.load_state_dict(s["scheduler"])
        start_epoch, best_loss, logs = s["epoch"] + 1, s["best_loss"], s["logs"]

    print(f"[LOG]: Training starts ------------")
    for i in range(start_epoch, train_cfg.epochs):
        for v in logs.values(): 
            v.append([])
        # --- Training
        model.train()
        for batch in train_loader:

            optimizer.zero_grad()
            o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad = (
                x.to(device, non_blocking=True) for x in batch)
            with autocast():
                losses = model(o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Logs
            logs["t_loss"][i].append(losses["total"].item())
            logs["t_sigreg"][i].append(losses["SIGReg"][0].item())
            logs["t_mse"][i].append([losses["MSE"][0].item(), losses["MSE"][1].item()])
            logs["t_zstd"][i].append(losses["zstd"].item())
            B = o_history.shape[0]
            logs["t_n"][i].append(B)

        # --- Evaluation
        model.eval()
        with torch.no_grad():
            for e_batch in val_loader:
                # eval dataset
                o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad = (
                                x.to(device, non_blocking=True) for x in e_batch)
                with autocast():
                    losses = model(o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad)
                logs["e_loss"][i].append(losses["total"].item())
                logs["e_sigreg"][i].append(losses["SIGReg"][0].item())
                logs["e_mse"][i].append([losses["MSE"][0].item(), losses["MSE"][1].item()])
                logs["e_zstd"][i].append(losses["zstd"].item())
                B = o_history.shape[0]
                logs["e_n"][i].append(B)

        # Check if collapse
        e_zstd = float(np.average(logs["e_zstd"][i], weights=logs["e_n"][i]))
        if i + 1 >= 5 and e_zstd < 0.005:
            print(f"[LOG]: COLLAPSE zstd={e_zstd:.6f} at epoch {i}")
            break

        logs["lr"][i] = [optimizer.param_groups[0]["lr"]] # as there is only 1 lr

        # save if best
        e_mse = np.average(logs["e_mse"][i], axis=0, weights=logs["e_n"][i]) # weigth on batch sizes as leWAM
        current_loss = float(e_mse[1]) # MSE action as leWAM
        if current_loss < best_loss:
            torch.save({"model": model.state_dict(), "eval_loss":current_loss, "action_mean": ds.mean, "action_std": ds.std, "epoch":i}, data_cfg.ckpt_path + "best.pt")
            best_loss = current_loss

        scheduler.step()

        train_loss = np.average(logs["t_loss"][i], weights=logs["t_n"][i])
        eval_loss  = np.average(logs["e_loss"][i], weights=logs["e_n"][i])

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": i, "best_loss": best_loss,
                    "logs": logs, "action_mean": ds.mean, "action_std": ds.std},
                    f"{data_cfg.ckpt_path}latest.pt")
        print(f"[LOG]: Epoch {i} | train_loss: {train_loss} | eval_loss: {eval_loss} | lr {logs['lr'][i][0]}")

    # --- End


if __name__ == "__main__":
    train()