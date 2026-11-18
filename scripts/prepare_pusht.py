"""
Download LeWAM's Push-T, keep only the episodes long enough for the hierarchy, and build the files the training reads.

    python scripts/prepare_pusht.py                                 # auto : PushT 125 if it fits in RAM, else PushT 200
    python scripts/prepare_pusht.py --horizon 200                   # PushT 200 : episodes > 200 steps
    python scripts/prepare_pusht.py --full_h5 /path/to/pusht.h5     # already downloaded + decompressed somewhere
    python scripts/prepare_pusht.py --cleanup                       # delete pusht.h5.zst / pusht.h5 once used

Steps (each one is skipped if its output exists, so the script can be relaunched after a crash) :
    1. download pusht.h5.zst (13.8 GB) from HF LeWAM/lewam-pusht   -> <data_dir>/raw/pusht.h5.zst
    2. decompress it with zstd (46.3 GB)                           -> <data_dir>/raw/pusht.h5
    3. keep the episodes with ep_len >= H + 1                      -> <data_dir>/pusht<H>.h5
       same columns, chunks and compression as the source, written with libver="latest" (swm opens it in SWMR)
       episode_idx is renumbered from 0 because swm indexes the episodes by their position in ep_offset
    4. uncompressed frames with scripts/decompress_h5.py           -> <data_dir>/decomp/pusht<H>/pixels.npy
    5. stats : size, RAM, and how often each level of the hierarchy (Ks of config/train.yaml) gets an unclamped target

Why H + 1 : the long-horizon protocol of LeWAM starts at frame 0 and puts the goal H steps later,
so an episode is usable at H only if it has at least H + 1 frames (505 episodes for H = 200).
It also gives the upper levels real targets : with K^n = 125, t + K^n stays in the episode for most starts.

--horizon auto (default) : the training only needs episodes of K^n + 1 = 126 frames (a real target for the top level),
and the long200 benchmarks of scripts/train_all.sh keep by themselves the episodes >= 201 frames of any subset.
So the first H of AUTO_HORIZONS whose pixels.npy fits in RAM_FRACTION of the RAM is prepared (more episodes for the
training), the size being computed from the full h5 before anything is written. train_all.sh takes pusht125 when it
is prepared, else pusht200. The frames are mapped by DatasetLeHWAM : one copy in RAM for all the runs of the machine.

Needs : h5py, hdf5plugin, numpy, omegaconf (requirements.txt) + the zstd CLI (or `pip install zstandard`)
        + huggingface_hub for the download (otherwise curl is used).
Peak disk without --cleanup : 13.8 (zst) + 46.3 (h5) + subset h5 + pixels.npy (~19 GB for H = 200).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  the frames are compressed with blosc (plugin filter), for reading and writing
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

REPO = "LeWAM/lewam-pusht"
ZST_NAME = "pusht.h5.zst"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{ZST_NAME}"
ZST_BYTES = 13_804_466_582  # sizes of the HF file and of its decompressed h5 (zstd frame header), for the disk checks
H5_BYTES = 46_300_921_856
SMALL_COLUMN_BYTES = 2 << 30  # columns smaller than this are read at once, the others (pixels) episode by episode
EPISODE_COLUMNS = ("episode_idx", "ep_idx")  # per-step episode id, renumbered in the subset
AUTO_HORIZONS = (125, 200)  # --horizon auto : the first one whose pixels.npy fits in RAM_FRACTION of the RAM
RAM_FRACTION = 0.7          # the rest : batches in flight of the dataloaders (~5 GB per run), CUDA, the OS
LONG_BENCH_H = 200          # task.goal_offset_steps of the long200 benchmarks of scripts/train_all.sh


# ------------------------------------------------ Utils ------------------------------------------------


def log(msg: str, end: str = "\n") -> None:
    print(f"[prepare] {msg}", end=end, flush=True)


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def show(path: Path) -> str:
    """Path relative to the repo when inside it (the configs are relative to the repo root)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def check_disk(directory: Path, needed: int, what: str) -> None:
    free = shutil.disk_usage(directory).free
    log(f"{what}: {gb(needed)} needed in {directory} ({gb(free)} free)")
    if needed > free:
        sys.exit(f"[prepare] not enough disk for {what}: need {gb(needed)}, {gb(free)} free")


def total_ram() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def load_ks(config: Path) -> list[int]:
    from omegaconf import OmegaConf
    return [int(k) for k in OmegaConf.load(config).data.Ks]


# ------------------------------------------------ 1. Download ------------------------------------------------


def download(dst: Path) -> None:
    if dst.exists():
        log(f"{dst.name} already there, skip download")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".partial")
    done = partial.stat().st_size if partial.exists() else 0  # curl resumes it
    check_disk(dst.parent, ZST_BYTES - done, "download")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None

    if hf_hub_download is not None:
        # resumable, the file only appears at local_dir/ZST_NAME once complete
        hf_hub_download(repo_id=REPO, filename=ZST_NAME, repo_type="dataset", local_dir=dst.parent)
    else:
        log("huggingface_hub not installed, downloading with curl")
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(partial), URL], check=True)
        os.replace(partial, dst)
    log(f"downloaded {dst} ({gb(dst.stat().st_size)})")


# ------------------------------------------------ 2. Decompress the archive ------------------------------------------------


def decompress_zst(src: Path, dst: Path, cleanup: bool) -> None:
    if dst.exists():
        log(f"{dst.name} already there, skip zstd")
        return
    partial = dst.with_name(dst.name + ".partial")
    partial.unlink(missing_ok=True)
    check_disk(dst.parent, H5_BYTES, "zstd -d")
    log(f"decompressing {src.name} -> {dst.name} (a few minutes)")
    if shutil.which("zstd"):
        subprocess.run(["zstd", "-d", "--long=27", "-f", str(src), "-o", str(partial)], check=True)
    else:
        import zstandard  # pip install zstandard
        with open(src, "rb") as fi, open(partial, "wb") as fo:
            # window of 2^27 (--long=27) > default limit of some builds
            zstandard.ZstdDecompressor(max_window_size=1 << 31).copy_stream(fi, fo, read_size=1 << 24, write_size=1 << 24)
    os.replace(partial, dst)
    if cleanup:
        src.unlink()
        log(f"removed {src.name}")


def get_full(full: Path, data_dir: Path, args) -> None:
    """steps 1-2 : the full decompressed h5 (downloaded + zstd -d, unless --full_h5 is given)"""
    if full.exists():
        return
    if args.full_h5 is not None:
        sys.exit(f"[prepare] {full} not found")
    zst = data_dir / "raw" / ZST_NAME
    download(zst)
    decompress_zst(zst, full, args.cleanup)


# ------------------------------------------------ 3. Subset of long episodes ------------------------------------------------


def choose_horizon(full: Path) -> int:
    """--horizon auto : the smallest H of AUTO_HORIZONS (= the most episodes) whose pixels.npy fits in RAM_FRACTION of the RAM"""
    with h5py.File(full, "r") as f:
        ep_len = f["ep_len"][:].reshape(-1).astype(np.int64)
        frame_bytes = int(np.prod(f["pixels"].shape[1:]))  # uint8
    ram = total_ram()
    budget = RAM_FRACTION * ram if ram else None
    log("--horizon auto : " + (f"{gb(budget)} for pixels.npy ({RAM_FRACTION:.0%} of the {gb(ram)} of RAM)" if ram
                               else "RAM unknown, the first one is taken"))
    for H in AUTO_HORIZONS:
        lens = ep_len[ep_len >= H + 1]
        npy = int(lens.sum()) * frame_bytes
        fits = budget is None or npy <= budget
        log(f"  H = {H}: {len(lens)}/{len(ep_len)} episodes, {lens.sum()} frames -> pixels.npy {gb(npy)}"
            + ("" if fits else " : too big"))
        if fits:
            return H
    log(f"WARNING: even H = {AUTO_HORIZONS[-1]} doesn't fit, the training will read the frames from the disk (slower)")
    return AUTO_HORIZONS[-1]



def create_like(dst: h5py.File, name: str, src: h5py.Dataset, n: int) -> h5py.Dataset:
    """Empty dataset of n rows with the dtype, chunks and filters (blosc, ...) of src."""
    shape = (n, *src.shape[1:])
    dcpl = src.id.get_create_plist()
    chunked = dcpl.get_layout() == h5py.h5d.CHUNKED
    maxshape = (h5py.h5s.UNLIMITED, *src.shape[1:]) if chunked else shape  # a chunk may be longer than n rows
    space = h5py.h5s.create_simple(shape, maxshape)
    ds = h5py.Dataset(h5py.h5d.create(dst.id, name.encode(), src.id.get_type(), space, dcpl=dcpl))
    for k, v in src.attrs.items():
        ds.attrs[k] = v
    return ds


def copy_rows(src: h5py.Dataset, dst: h5py.Dataset, offs: np.ndarray, lens: np.ndarray, rows: np.ndarray) -> None:
    """dst = rows of the kept episodes of src, in order."""
    if src.nbytes <= SMALL_COLUMN_BYTES:
        dst[...] = src[:][rows]
        return
    ptr = 0
    for k, (o, n) in enumerate(zip(offs, lens)):
        dst[ptr:ptr + n] = src[o:o + n]
        ptr += n
        if k % 10 == 0 or k == len(lens) - 1:
            log(f"{src.name[1:]}: episode {k + 1}/{len(lens)}", end="\r")
    print(flush=True)


def make_subset(full: Path, out: Path, min_len: int) -> None:
    if out.exists():
        with h5py.File(out, "r") as f:
            if int(f.attrs.get("min_len", -1)) != min_len:
                sys.exit(f"[prepare] {out} was built with min_len={f.attrs.get('min_len')}, delete it to rebuild")
        log(f"{out.name} already there, skip subset")
        return
    partial = out.with_name(out.name + ".partial")
    partial.unlink(missing_ok=True)

    with h5py.File(full, "r", rdcc_nbytes=512 << 20) as f:
        ep_len = f["ep_len"][:].reshape(-1).astype(np.int64)
        ep_off = f["ep_offset"][:].reshape(-1).astype(np.int64)
        n_eps = len(ep_len)
        n_rows = f["step_idx"].shape[0] if "step_idx" in f else int(ep_len.sum())

        keep = np.nonzero(ep_len >= min_len)[0]
        if len(keep) == 0:
            sys.exit(f"[prepare] no episode with at least {min_len} frames (longest: {ep_len.max()})")
        lens, offs = ep_len[keep], ep_off[keep]
        new_off = np.concatenate([[0], np.cumsum(lens)[:-1]])
        n_new = int(lens.sum())
        rows = np.concatenate([np.arange(o, o + n) for o, n in zip(offs, lens)])
        log(f"keep {len(keep)}/{n_eps} episodes (ep_len >= {min_len}) -> {n_new}/{n_rows} frames")

        if "pixels" in f:
            ratio = n_new / n_rows
            check_disk(out.parent, int(full.stat().st_size * ratio * 1.2), "subset h5 (estimate)")

        with h5py.File(partial, "w", libver="latest") as g:
            for k, v in f.attrs.items():
                g.attrs[k] = v
            for name, src in f.items():
                if not isinstance(src, h5py.Dataset) or src.shape == ():
                    f.copy(src, g, name=name)
                    log(f"{name}: copied as is (not a column)")
                elif name in ("ep_len", "ep_offset"):
                    values = lens if name == "ep_len" else new_off
                    create_like(g, name, src, len(keep))[...] = values.reshape(-1, *src.shape[1:]).astype(src.dtype)
                elif src.shape[0] == n_rows:  # one row per step
                    dst = create_like(g, name, src, n_new)
                    if name in EPISODE_COLUMNS:
                        ids = np.repeat(np.arange(len(keep)), lens)
                        dst[...] = ids.reshape(-1, *src.shape[1:]).astype(src.dtype)
                    else:
                        copy_rows(src, dst, offs, lens, rows)
                elif src.shape[0] == n_eps:  # one row per episode
                    create_like(g, name, src, len(keep))[...] = src[:][keep]
                else:
                    f.copy(src, g, name=name)
                    log(f"{name}: shape {src.shape} is neither per step nor per episode, copied as is")
            g.attrs["source"] = str(full.resolve())
            g.attrs["min_len"] = min_len
            g.attrs["src_episodes"] = keep  # index of each kept episode in the full dataset
    os.replace(partial, out)
    log(f"wrote {out} ({gb(out.stat().st_size)})")


# ------------------------------------------------ 4. Uncompressed frames ------------------------------------------------


def decompress_frames(h5: Path, decomp_dir: Path) -> Path:
    npy_dir = decomp_dir / h5.stem
    meta = npy_dir / "meta.json"
    with h5py.File(h5, "r") as f:
        n_frames = f["pixels"].shape[0]
    if meta.exists() and (npy_dir / "pixels.npy").exists():
        m = json.load(open(meta))
        if m.get("source") == str(h5.resolve()) and m["views"].get("pixels", [None])[0] == n_frames:
            log(f"{show(npy_dir / 'pixels.npy')} already there, skip")
            return npy_dir / "pixels.npy"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "decompress_h5.py"),
                    "--dataset_path", str(h5), "--decomp_dir", str(decomp_dir)], check=True)
    return npy_dir / "pixels.npy"


# ------------------------------------------------ 5. Stats ------------------------------------------------


def coverage(ep_len: np.ndarray, Ks: list[int]) -> list[tuple[float, float]]:
    """
    over the training starts of DatasetLeHWAM (t in [0, n - K^0) of each episode), for each level i :
        - % of starts with t + K^i inside the episode (next-state target not clamped)
        - % of starts whose goal range is not clamped :
              i < n : t_g in [t + K^i, t + K^{i+1}]  -> needs K^{i+1} frames left
              top   : t_g in [t + K^n, end]          -> needs K^n frames left
    """
    ftt = np.concatenate([n - 1 - np.arange(max(0, n - Ks[0])) for n in ep_len])  # frames left after t
    out = []
    for i, K in enumerate(Ks):
        goal_K = Ks[i + 1] if i + 1 < len(Ks) else K
        out.append((100 * (ftt >= K).mean(), 100 * (ftt >= goal_K).mean()))
    return out


def report(full: Path | None, out: Path, npy: Path, Ks: list[int], horizon: int) -> None:
    with h5py.File(out, "r") as f:
        ep_len = f["ep_len"][:].reshape(-1).astype(np.int64)
    full_len = None
    if full is not None and full.exists():
        with h5py.File(full, "r") as f:
            full_len = f["ep_len"][:].reshape(-1).astype(np.int64)

    log(f"{out.name}: {len(ep_len)} episodes, {ep_len.sum()} frames, "
        f"ep_len min {ep_len.min()} / mean {ep_len.mean():.0f} / max {ep_len.max()} (= h_top)")
    npy_bytes = npy.stat().st_size
    ram = total_ram()
    log(f"pixels.npy: {gb(npy_bytes)}, mapped by DatasetLeHWAM : one copy in the page cache for all the runs"
        + (f" ({gb(ram)} of RAM here)" if ram else ""))
    if ram and npy_bytes > RAM_FRACTION * ram:
        log("WARNING: pixels.npy doesn't fit in RAM next to the training, the frames will be read from the disk (slower)")
    n_long = int((ep_len >= LONG_BENCH_H + 1).sum())
    log(f"long{LONG_BENCH_H} benchmarks (start at frame 0, goal {LONG_BENCH_H} steps later) : {n_long} episodes "
        f"(eval.num_eval must be < {n_long})")

    sub_cov = coverage(ep_len, Ks)
    full_cov = coverage(full_len, Ks) if full_len is not None else None
    log(f"hierarchy coverage over the training starts (Ks = {Ks})" + (", full dataset in ()" if full_cov else ""))
    log(f"  {'level':>5} {'K':>5} {'t+K in episode':>22} {'goal range not clamped':>24}")
    for i, (K, (a, b)) in enumerate(zip(Ks, sub_cov)):
        fa = f" ({full_cov[i][0]:5.1f}%)" if full_cov else ""
        fb = f" ({full_cov[i][1]:5.1f}%)" if full_cov else ""
        log(f"  {i:>5} {K:>5} {f'{a:5.1f}%{fa}':>22} {f'{b:5.1f}%{fb}':>24}")
    if horizon < Ks[-1]:
        log(f"WARNING: H = {horizon} < K^n = {Ks[-1]} : the top level is never fully used at this horizon")

    h5_cfg, npy_cfg = show(out), show(npy)
    log("done. Train with (from the repo root):")
    print(f"    python -m scripts.train data.path_h5p={h5_cfg} data.path_npy={npy_cfg}")
    H = max(horizon, LONG_BENCH_H)
    print(f"  long-horizon benchmark at H = {H} (as LeWAM) :")
    print(f"    python scripts/benchmark.py data.path_h5p={h5_cfg} eval.random_start=False "
          f"eval.exec_actions=25 task.goal_offset_steps={H} task.eval_budget={2 * H}")


# ------------------------------------------------ Main ------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", type=lambda s: s if s == "auto" else int(s), default="auto",
                   help=f"H : keep the episodes with at least H + 1 frames, "
                        f"auto : the first of {AUTO_HORIZONS} whose frames fit in RAM (default)")
    p.add_argument("--data_dir", type=Path, default=ROOT / "data", help="raw/, pusht<H>.h5 and decomp/ go here")
    p.add_argument("--full_h5", type=Path, default=None,
                   help="an already decompressed pusht.h5 (skips the download and zstd, never deleted)")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "train.yaml", help="to read Ks for the stats")
    p.add_argument("--cleanup", action="store_true",
                   help="delete pusht.h5.zst after zstd and the downloaded pusht.h5 after the subset")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    Ks = load_ks(args.config)
    full = args.full_h5.resolve() if args.full_h5 else data_dir / "raw" / "pusht.h5"

    horizon = args.horizon
    if horizon == "auto":
        if (data_dir / f"pusht{AUTO_HORIZONS[0]}.h5").exists(): # chosen by a previous launch (its full h5 may be deleted)
            horizon = AUTO_HORIZONS[0]
            log(f"--horizon auto : pusht{horizon}.h5 already there")
        else:
            get_full(full, data_dir, args)
            horizon = choose_horizon(full)
    min_len = horizon + 1
    out = data_dir / f"pusht{horizon}.h5"
    log(f"PushT {horizon}: episodes with >= {min_len} frames | Ks {Ks} | -> {show(out)}")

    if not out.exists():
        get_full(full, data_dir, args)
    make_subset(full, out, min_len)
    if args.cleanup and args.full_h5 is None and full.exists():
        full.unlink()
        log(f"removed {full}")

    npy = decompress_frames(out, data_dir / "decomp")
    report(full, out, npy, Ks, horizon)


if __name__ == "__main__":
    main()
