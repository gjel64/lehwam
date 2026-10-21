"""Write a copy of an h5 dataset with its image columns resized, e.g. 224 -> 112 px to train at lower resolution.

Frames are resized like the eval transform (torchvision Resize: bilinear, antialiased) and stored uncompressed and
contiguous, so data.source=h5 reads them without decompression. The other columns are copied unchanged.
  python scripts/resize_h5.py --dataset_path $STABLEWM_HOME/datasets/pusht.h5 --size 112
writes $STABLEWM_HOME/datasets/pusht_112.h5; train with data.img_size=112 model.crop_size=101.
"""

import argparse
import os
from multiprocessing import Pool

import h5py
import hdf5plugin  # noqa: F401  the swm h5 files compress frames with plugin filters
import numpy as np
import torch
import torch.nn.functional as F

ROWS_PER_READ = 1000


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_path", required=True, help="the h5 dataset to resize")
    argparser.add_argument("--size", type=int, required=True, help="output height and width in pixels")
    argparser.add_argument("--out_path", default=None, help="default: <dataset>_<size>.h5 next to the input")
    argparser.add_argument("--views", default="pixels", help="comma separated list of the h5 image columns to resize")
    argparser.add_argument("--workers", type=int, default=16, help="processes decoding and resizing frames")
    return argparser.parse_args()


def resize_rows(job: tuple[str, str, int, int, int]) -> tuple[int, np.ndarray]:
    """Read rows [start, stop) of one image column and return them resized, channels-last uint8."""
    dataset_path, column, start, stop, size = job
    torch.set_num_threads(1)
    with h5py.File(dataset_path, "r") as f:
        rows = torch.from_numpy(f[column][start:stop]).permute(0, 3, 1, 2).float()
    rows = F.interpolate(rows, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return start, rows.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def main():
    args = parse_args()
    views = [v for v in args.views.split(",") if v]
    out_path = args.out_path or f"{os.path.splitext(args.dataset_path)[0]}_{args.size}.h5"
    partial = out_path + ".partial"

    with h5py.File(args.dataset_path, "r") as src, h5py.File(partial, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name, column in src.items():
            if name in views:
                continue
            src.copy(column, dst, name=name)
        for view in views:
            n, h, w, c = src[view].shape
            assert c == 3 and src[view].dtype == np.uint8, f"{view}: expected (N, H, W, 3) uint8, got {src[view].shape}"
            out = dst.create_dataset(view, shape=(n, args.size, args.size, 3), dtype=np.uint8)
            jobs = [(args.dataset_path, view, i, min(i + ROWS_PER_READ, n), args.size) for i in range(0, n, ROWS_PER_READ)]
            with Pool(args.workers) as pool:
                for done, (start, rows) in enumerate(pool.imap_unordered(resize_rows, jobs), 1):
                    out[start:start + len(rows)] = rows
                    print(f"[resize] {view}: {done}/{len(jobs)} chunks", end="\r", flush=True)
            print(f"\n[resize] {view}: ({n}, {h}, {w}) -> ({n}, {args.size}, {args.size})", flush=True)

    os.replace(partial, out_path)
    print(f"[resize] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
