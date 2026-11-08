"""
Goal-reaching benchmark of a leWAM checkpoint, same protocol as scripts/eval_lewam.py of LeWAM :
    - start at a random step of an episode of the dataset, the goal is the frame {goal_offset_steps} later
    - {eval_budget} actions to reach it, replan every {exec_actions} actions
    - success = the env says the goal is reached

    python scripts/benchmark.py                                             # pusht, gradient_based, best.pt
    python scripts/benchmark.py eval.inference_type=direct_policy
    python scripts/benchmark.py task=tworoom eval.inference_type=best-of-K eval.ckpt=latest
    python scripts/benchmark.py model.planning.k=64 model.planning.n_gd_step=10
    for s in 42 0 1; do python scripts/benchmark.py seed=$s; done           # 50 episodes x 3 seeds as the paper

long horizon (as LeWAM) : eval.random_start=False eval.min_episode_len=101 eval.exec_actions=25 task.goal_offset_steps=H task.eval_budget=2H
results (txt + videos) in outputs/eval/<task>/<inference_type>/
"""

import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import hdf5plugin # noqa: F401  the h5 frames are compressed with plugin filters
import hydra
import numpy as np
import torch
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa: F401  registers the HDF5 format
from stable_worldmodel.data import HDF5Dataset
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as transforms

from src.model.lewam import leWAM
from src.eval.policy import LeWAMPolicy

STRIDE = 5 # frameskip, as in train.py


# ------------------------------------------------ Utils ------------------------------------------------


def get_dataset(path, keys_to_cache):
    """
    swm dataset of the h5 with only the columns that have one row per step (as LeWAM)
    a per-episode column (ep_len, ep_offset, ...) would raise an IndexError in get_row_data
    """
    with h5py.File(path, "r") as f:
        n_rows = len(f["step_idx"])
        keys = [k for k in f.keys() if k not in ("ep_len", "ep_offset") and f[k].shape[:1] == (n_rows,)]
    return HDF5Dataset(path=path, keys_to_load=keys, keys_to_cache=keys_to_cache)


def sample_eval_episodes(dataset, num_eval, seed, goal_offset_steps, random_start=True, min_episode_len=0):
    """
    draw {num_eval} (episode, start step) with the seed, the goal is {goal_offset_steps} after the start
    episodes too short to reach the goal are removed
    same draw as LeWAM (and LeWM) so the same seed gives the same episodes -> fair comparison
    -> episodes, starts
    """
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    row_ep = np.asarray(dataset.get_col_data(col)).reshape(-1)              # episode of each row
    row_step = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)     # step of each row in its episode
    ep_ids = np.unique(row_ep)
    lengths = np.array([row_step[row_ep == e].max() + 1 for e in ep_ids])
    row_len = lengths[np.searchsorted(ep_ids, row_ep)]                      # length of the episode of each row

    # a start is valid if the goal is still in the episode
    valid = (row_step <= row_len - goal_offset_steps - 1) & (row_len >= min_episode_len)
    if not random_start:
        valid &= row_step == 0 # only the first frames (long horizon protocol)
    valid = np.nonzero(valid)[0]
    assert len(valid) > num_eval, f"only {len(valid)} valid starts for {num_eval} episodes"

    rng = np.random.default_rng(seed)
    # len(valid) - 1 to have the same draw as LeWAM/LeWM
    picks = rng.choice(len(valid) - 1 if random_start else len(valid), size=num_eval, replace=False)
    picks = np.sort(valid[picks]) # sorted for the h5 reads
    rows = dataset.get_row_data(picks)
    return rows[col].tolist(), rows["step_idx"].tolist()


# ------------------------------------------------ Benchmark ------------------------------------------------


@hydra.main(config_path="../config", config_name="eval")
def benchmark(cfg : DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # --- Load config
    eval_cfg, task_cfg = cfg["eval"], cfg["task"]
    torch.manual_seed(cfg.seed) # noise of the flow matching
    np.random.seed(cfg.seed)

    # --- Load model
    s = torch.load(ROOT / cfg.data.ckpt_path / f"{eval_cfg.ckpt}.pt", map_location=device)
    model = leWAM(cfg.model, len(s["action_mean"]), device).to(device)
    model.load_state_dict(s["model"])
    model.eval().requires_grad_(False) # gradient_based only needs grads on the actions
    print(f"[LOG]: {eval_cfg.ckpt}.pt loaded (epoch {s['epoch']})")

    # --- Episodes (from the training data as LeWAM)
    dataset = get_dataset(ROOT / eval_cfg.dataset_path, list(task_cfg.keys_to_cache))
    episodes, starts = sample_eval_episodes(dataset, eval_cfg.num_eval, cfg.seed, task_cfg.goal_offset_steps,
                                            eval_cfg.random_start, eval_cfg.min_episode_len)

    # --- World
    world = swm.World(**task_cfg.world, num_envs=eval_cfg.num_eval, image_shape=(224, 224),
                      max_episode_steps=2 * task_cfg.eval_budget) # 2x budget as LeWAM

    # --- Policy
    # no ToDtype/Normalize as LeWAM : our encoder takes uint8 and normalizes itself
    transform = {"pixels": transforms.Resize(eval_cfg.img_size), "goal": transforms.Resize(eval_cfg.img_size)}
    policy = LeWAMPolicy(model, s["action_mean"], s["action_std"], eval_cfg.inference_type,
                         h0=task_cfg.goal_offset_steps / STRIDE, exec_actions=eval_cfg.exec_actions,
                         stride=STRIDE, transform=transform)
    world.set_policy(policy)

    # --- Evaluation
    results_path = ROOT / "outputs" / "eval" / task_cfg.name / eval_cfg.inference_type
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"[LOG]: Benchmark starts ------------ {task_cfg.name} | {eval_cfg.inference_type} | {eval_cfg.num_eval} episodes")
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=episodes,
        start_steps=starts,
        goal_offset=task_cfg.goal_offset_steps,
        eval_budget=task_cfg.eval_budget,
        callables=OmegaConf.to_container(task_cfg.callables, resolve=True),
        video=results_path if eval_cfg.video else None,
    )
    wall = time.time() - start_time
    print(f"[LOG]: success rate {metrics['success_rate']:.1f}% | {wall:.1f}s")

    # --- Save (appended as LeWAM)
    with (results_path / "results.txt").open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\n==== RESULTS ({eval_cfg.inference_type}) ====\nmetrics: {metrics}\ntime: {wall:.1f}s\n")
    print(f"[LOG]: results in {results_path}")


if __name__ == "__main__":
    benchmark()
