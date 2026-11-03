import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import hdf5plugin


class DatasetLeHWAM(Dataset):
    def __init__(self, path, stride, h_max=10):
        self.path, self.stride, self.h_max = path, stride, h_max
        self.f = None
        with h5py.File(path, "r") as f:
            ep_len = f["ep_len"][:].reshape(-1)
            ep_off = f["ep_offset"][:].reshape(-1)
            raw_act = torch.from_numpy(f["action"][:])
        no_nan = raw_act[~torch.isnan(raw_act).any(1)]
        mean, std = no_nan.mean(0), no_nan.std(0)
        std[std == 0] = 1.0
        self.mean, self.std = mean.tolist(), std.tolist()
        self.actions = ((raw_act.float() - torch.tensor(self.mean)) / torch.tensor(self.std)).half()

        # valids idx
        self.starts = [(o + t, o, o + n - 1) for o, n in zip(ep_off, ep_len) for t in range(0, n - stride)]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        """
        index -> o_history, o_target, o_goal, o_actions, actions_valid, h_norm, history_pad 
        """
        if self.f is None:
            self.f = h5py.File(self.path, "r") # for multiple workers
        t, ep_start, end = self.starts[i]
        ftt = end - t
        pad = t - self.stride < ep_start
        t_prev = t if pad else t - self.stride
        history_pad = torch.tensor([pad, False])
        h = min(int(torch.randint(1, self.h_max + 1, (1,))), max(1, ftt // self.stride))

        n_act = 2 * self.stride
        n_valid = min(n_act, ftt)
        actions = torch.zeros(n_act, self.actions.shape[1])
        actions[:n_valid] = self.actions[t:t + n_valid].float()  # actions déjà normalisées, en RAM (voir c)
        action_valid = torch.zeros(n_act)
        action_valid[:n_valid] = 1

        frames = np.stack([self.f["pixels"][t_prev], self.f["pixels"][t], self.f["pixels"][t + self.stride], self.f["pixels"][t + h * self.stride]])
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)           # (4, 3, H, W)
        return frames[:2], frames[2:3], frames[3:4], actions, action_valid, torch.tensor(h / self.h_max), history_pad