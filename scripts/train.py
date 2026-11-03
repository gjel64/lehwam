from torch.utils.data import DataLoader, random_split
from src.data.data import DatasetLeHWAM
import torch
from src.model.lewam import leWAM


def train(path, batch_size, num_workers, ckpt_path="checkpoints/lehwam.pt"):
    ds = DatasetLeHWAM(path, stride=5)
    val_ds, train_ds = random_split(ds, [0.1, 0.9], generator=torch.Generator().manual_seed(42))

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    

    model = leWAM(192, 2, torch.device("cpu"))

    # Rest of training



    # End
    torch.save({"model": model.state_dict(), "action_mean": ds.mean, "action_std": ds.std}, ckpt_path)

if __name__ == "__main__":
    train()