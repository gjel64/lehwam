import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import hdf5plugin


class DatasetLeHWAM(Dataset):
    def __init__(self, path_h5p, path_npy, stride, h_max=10):
        self.stride, self.h_max = stride, h_max
        self.frames = torch.from_numpy(np.load(path_npy, mmap_mode=None))
        with h5py.File(path_h5p, "r") as f:
            ep_len = f["ep_len"][:].reshape(-1)
            ep_off = f["ep_offset"][:].reshape(-1)
            raw_act = torch.from_numpy(f["action"][:])
        no_nan = raw_act[~torch.isnan(raw_act).any(1)]
        mean, std = no_nan.mean(0), no_nan.std(0)
        std[std == 0] = 1.0
        self.mean, self.std = mean.tolist(), std.tolist()
        self.actions = ((raw_act.float() - torch.tensor(self.mean)) / torch.tensor(self.std)).half()

        # valids idx
        frame_starts, ep_starts, frames_to_terminal = [], [], []
        for o, n in zip(ep_off, ep_len):
            t = np.arange(max(0, n - stride))
            frame_starts.append(o + t)
            ep_starts.append(np.full(t.size, o))
            frames_to_terminal.append(n - 1 - t)
        self.frame_starts, self.ep_starts, self.frames_to_terminal = (
            torch.from_numpy(np.concatenate(a).astype(np.int64)) for a in (frame_starts, ep_starts, frames_to_terminal))

    def __len__(self):
        return self.frame_starts.numel()

    def __getitem__(self, i):
        """
        index -> o_history, o_target, o_goal, o_actions, actions_valid, h_norm, history_pad 
        """
        t, ep_start = int(self.frame_starts[i]), int(self.ep_starts[i])
        ftt = int(self.frames_to_terminal[i])
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

        frames = self.frames[torch.tensor([t_prev, t, t + self.stride, t + h * self.stride])]
        return frames[:2], frames[2:3], frames[3:4], actions, action_valid, torch.tensor(h / self.h_max), history_pad