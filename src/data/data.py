import time
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import hdf5plugin


def warm_page_cache(path, chunk=1 << 26):
    """
    read the file once, sequentially : its pages stay in the page cache of the OS, so the random reads of the
    training come from RAM (as a full np.load) but the pages are shared by every process that maps the file
    (2 runs on 2 GPUs -> 1 copy in RAM instead of 2). Already cached (the other run) : a few seconds.
    """
    t, buf, n = time.time(), bytearray(chunk), 0
    with open(path, "rb", buffering=0) as f:
        while (k := f.readinto(buf)):
            n += k
    print(f"[LOG]: {path} in the page cache ({n / 1e9:.1f} GB, {time.time() - t:.0f}s)", flush=True)


class DatasetLeHWAM(Dataset):
    def __init__(self, path_h5p, path_npy, stride, h_max=10, Ks=None):
        """
        stride : K^0, step between two frames of LeWAM
        h_max  : max nb of chunks to the goal of LeWAM (overwritten by K^1 / K^0 if Ks has more than 1 level)
        Ks     : [K^0, ..., K^n] step of each level -> hierarchical samples (None -> LeWAM samples)
        """
        if Ks is not None:
            Ks = [int(k) for k in Ks]
            assert Ks[0] == stride, "K^0 must be the LeWAM stride"
            assert all(a < b for a, b in zip(Ks, Ks[1:])), "Ks must be increasing"
            if len(Ks) > 1:
                assert Ks[1] % Ks[0] == 0, "K^1 must be a multiple of K^0 (the goal of level 0 is sampled in chunks)"
                h_max = Ks[1] // Ks[0] # t_g^0 = t + h*K^0, h in {1..K^1/K^0} -> t_g^0 in [t+K^0, t+K^1] -> h^0 = (t_g^0 - t) / K^1
        self.stride, self.h_max, self.Ks = stride, h_max, Ks
        # mapped, not loaded : one copy in RAM (page cache) for all the runs of the machine, and if the frames don't
        # fit in RAM the training reads the disk (slower) instead of crashing
        warm_page_cache(path_npy)
        self.path_npy = path_npy
        self.frames = np.load(path_npy, mmap_mode="r") # (N, 3, H, W) uint8
        with h5py.File(path_h5p, "r") as f:
            ep_len = f["ep_len"][:].reshape(-1)
            ep_off = f["ep_offset"][:].reshape(-1)
            raw_act = torch.from_numpy(f["action"][:])
        self.h_top = int(ep_len.max()) # normalization of the horizon of the top level (its goal is the random between [t+K^n,end of the episode])
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

    def __getstate__(self):
        """dataloader workers started with spawn (macOS) get a pickled dataset : the memmap is reopened, not copied"""
        state = self.__dict__.copy()
        state["frames"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.frames = np.load(self.path_npy, mmap_mode="r")

    def split_episodes(self, eval_part, seed):
        """
        eval on whole episodes (LeWAM splits random frames : the eval frames have neighbors 1 step away in the
        training set, so the eval loss is a training loss and can't show overfitting)
        -> train_idx, eval_idx : indices of samples, {eval_part} of the episodes in eval (at least 1)
        """
        eps = self.ep_starts.unique()
        n_eval = max(1, round(eval_part * len(eps)))
        eval_eps = eps[torch.randperm(len(eps), generator=torch.Generator().manual_seed(seed))[:n_eval]]
        is_eval = torch.isin(self.ep_starts, eval_eps)
        return (~is_eval).nonzero()[:, 0].tolist(), is_eval.nonzero()[:, 0].tolist()

    def __getitem__(self, i):
        """
        Ks is None : index -> o_history, o_target, o_goal, o_actions, actions_valid, h_norm, history_pad
        Ks given   : index -> frames, o_actions, actions_valid, hs, history_pad
            frames : (4+2n, 3, H, W) [t-K^0, t, (t+K^i, goal^i) for i = 0..n]
                     -> t+K^i at 2+2i, goal^i at 3+2i (t_g^i, in [t+K^n, end of the episode] for the top level)
            hs     : (n+1,) h^i of each level
        """
        t, ep_start = int(self.frame_starts[i]), int(self.ep_starts[i])
        ftt = int(self.frames_to_terminal[i]) # nb of frames until the end of the episode
        pad = t - self.stride < ep_start
        t_prev = t if pad else t - self.stride
        history_pad = torch.tensor([pad, False])
        h = min(int(torch.randint(1, self.h_max + 1, (1,))), max(1, ftt // self.stride))

        n_act = 2 * self.stride
        n_valid = min(n_act, ftt)
        actions = torch.zeros(n_act, self.actions.shape[1])
        actions[:n_valid] = self.actions[t:t + n_valid].float()
        action_valid = torch.zeros(n_act)
        action_valid[:n_valid] = 1

        # level 0 : as LeWAM
        idx = [t_prev, t, t + self.stride, t + h * self.stride]

        if self.Ks is None:
            frames = torch.from_numpy(self.frames[idx]) # fancy indexing of the memmap : a copy in RAM
            return frames[:2], frames[2:3], frames[3:4], actions, action_valid, torch.tensor(h / self.h_max), history_pad

        # level i >= 1 : every frame is clamped to the end of the episode
        hs = [h / self.h_max]
        n = len(self.Ks) - 1
        for k in range(1, n + 1):
            K = self.Ks[k]
            if k < n: # t_g^k in [t+K^k, t+K^{k+1}] (closed : at inference the last replan before the level above refreshes is at distance K^k)
                d = min(int(torch.randint(K, self.Ks[k + 1] + 1, (1,))), ftt)
                hs.append(d / self.Ks[k + 1])
            else:   # top level : t_g^n in [t+K^n, end of the episode] (hindsight goal), t_g^n = end if the episode ends before t+K^n
                d = int(torch.randint(min(K, ftt), ftt + 1, (1,)))
                hs.append(d / self.h_top)
            idx += [t + min(K, ftt), t + d]

        frames = torch.from_numpy(self.frames[idx])
        return frames, actions, action_valid, torch.tensor(hs, dtype=torch.float32), history_pad