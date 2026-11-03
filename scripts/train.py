import torch
from torch.utils.data import DataLoader, random_split
from functools import partial
from contextlib import nullcontext

from src.data.data import DatasetLeHWAM
from src.model.lewam import leWAM


def train(path_h5p, path_npy, batch_size, num_workers, epochs, use_bf16, ckpt_path="checkpoints/lehwam.pt"):
    ds = DatasetLeHWAM(path_h5p, path_npy, stride=5)
    val_ds, train_ds = random_split(ds, [0.1, 0.9], generator=torch.Generator().manual_seed(42)) # first eval 10 then train 90 as in LeWAM

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
              persistent_workers=(num_workers != 0), prefetch_factor=4 if num_workers != 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = True
    autocast = partial(torch.autocast, device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext

    model = leWAM(192, 2, device).to(device)

    # Rest of training - not working just to note

    for i in range(epochs):
        for batch in train_loader:
            o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad = (
                x.to(device, non_blocking=True) for x in batch)
            with autocast():
                losses = model(o_history, o_target, o_goal, actions, action_valid, h_norm, history_pad)

            # the rest...




    # End
    torch.save({"model": model.state_dict(), "action_mean": ds.mean, "action_std": ds.std}, ckpt_path)

if __name__ == "__main__":
    train()