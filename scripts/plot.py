"""
Plots of the runs (png) :
    <run>/plots/train.png    losses of every level, SIGReg, z std, lr per epoch     (train.py redraws it after each epoch)
    <run>/plots/bench.png    success rate of each benchmark of the run              (benchmark.py redraws it after each benchmark)
    <root>/plots/bench.png   success rate of every run x every benchmark            (from <root>/results.csv, idem)

    python scripts/plot.py                              # every run of checkpoints/ + checkpoints/results.csv
    python scripts/plot.py checkpoints/lehwam_3lvl      # one run (+ the benchmarks of results.csv next to it)

Colors : validated palette of the dataviz reference (categorical slots 1-2, blue sequential ramp, chrome).
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on the GPU machines
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]

SLOT1, SLOT2 = "#2a78d6", "#eb6834"   # train / eval (and the bars : one series -> slot 1)
SURFACE, INK, INK2, MUTED, GRID, AXIS, EMPTY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential, light -> dark
COLLAPSE_ZSTD = 0.005  # same threshold as train.py


class PlainLogFormatter(LogFormatter):
    """labels of a log axis as 0.6 instead of 6e-01 (LogFormatter still chooses which ticks get a label)"""
    def _num_to_string(self, x, vmin, vmax):
        return f"{x:g}"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "font.size": 9, "font.family": "sans-serif", "lines.linewidth": 1.5, "legend.frameon": False,
})


# ------------------------------------------------ Utils ------------------------------------------------


def _save(fig, out):
    """write then rename : a png being read (or a run killed while plotting) is never half written"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.png")
    fig.savefig(tmp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, out)


def _per_epoch(values, sizes):
    """logs[key] of train.py : epochs x batches x (scalar | list) -> (n_epochs, k), mean over the batches weighted by their size"""
    rows = []
    for v, n in zip(values, sizes):
        rows.append(np.average(np.asarray(v, dtype=float).reshape(len(v), -1), axis=0, weights=n[:len(v)]) if len(v) else None)
    k = max((len(r) for r in rows if r is not None), default=1)
    out = np.full((len(rows), k), np.nan)
    for i, r in enumerate(rows):
        if r is not None:
            out[i, :len(r)] = r
    return out


# ------------------------------------------------ Training ------------------------------------------------


def plot_train(logs, out, title="", Ks=None):
    """
    logs : logs of train.py (latest.pt["logs"]), train vs eval, x = epoch
    top row : total loss, criterion of best.pt (sum of the eval MSE of every level, as train.py), lr
    one row per level : next state MSE, flow MSE (actions at level 0, sub-goals above), SIGReg, z std
    MSE order of leHWAM.forward : [l0 state, l0 action, top state, top flow, top-1 state, top-1 flow, ...]
    SIGReg, zstd : [level 0, ..., level n] (older logs : the sum of the SIGReg and the z std of level 0 only)
    """
    t_mse, e_mse = _per_epoch(logs["t_mse"], logs["t_n"]), _per_epoch(logs["e_mse"], logs["e_n"])
    t_sig, e_sig = _per_epoch(logs["t_sigreg"], logs["t_n"]), _per_epoch(logs["e_sigreg"], logs["e_n"])
    t_z, e_z = _per_epoch(logs["t_zstd"], logs["t_n"]), _per_epoch(logs["e_zstd"], logs["e_n"])
    n = (t_mse.shape[1] - 2) // 2   # nb of levels above LeWAM
    Ks = list(Ks) if Ks is not None else []
    k_of = lambda L: f" (K={Ks[L]})" if L < len(Ks) else ""
    lr = np.array([v[0] if len(v) else np.nan for v in logs["lr"]], dtype=float)

    # name -> (title, train, eval, log scale)
    panels = {
        "total": ("total loss", _per_epoch(logs["t_loss"], logs["t_n"])[:, 0], _per_epoch(logs["e_loss"], logs["e_n"])[:, 0], True),
        "crit": ("sum of the MSE of every level (eval selects best.pt)", t_mse.sum(1), e_mse.sum(1), True),
        "lr": ("learning rate", lr, None, False),
    }
    for L in range(n + 1):
        j = 0 if L == 0 else 2 + 2 * (n - L)
        panels[f"state{L}"] = (f"level {L} · next state MSE{k_of(L)}", t_mse[:, j], e_mse[:, j], True)
        panels[f"flow{L}"] = (f"level {L} · {'action' if L == 0 else 'sub-goal'} flow MSE", t_mse[:, j + 1], e_mse[:, j + 1], True)
        if t_sig.shape[1] == n + 1:
            panels[f"sig{L}"] = (f"level {L} · SIGReg", t_sig[:, L], e_sig[:, L], True)
        elif L == 0:
            panels["sig0"] = ("SIGReg, sum over the levels", t_sig.sum(1), e_sig.sum(1), True)
        if L < t_z.shape[1]:
            panels[f"z{L}"] = (f"level {L} · z std (collapse < {COLLAPSE_ZSTD})", t_z[:, L], e_z[:, L], False)

    mosaic = [["total"] * 4 + ["crit"] * 4 + ["lr"] * 4]
    mosaic += [[f"state{L}"] * 3 + [f"flow{L}"] * 3 + [f"sig{L}"] * 3 + [f"z{L}"] * 3 for L in range(n + 1)]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(15, 2.3 * (n + 2) + 0.6), sharex=True, layout="constrained")

    epochs = np.arange(len(t_mse))
    marker = "o" if len(epochs) < 3 else None # a line needs 2 points
    for key, ax in axes.items():
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key not in panels: # older logs
            ax.text(0.5, 0.5, "not logged in this run", transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_yticks([])
            ax.grid(False)
            continue
        name, tr, ev, log = panels[key]
        ax.plot(epochs, tr, color=SLOT1, label="train", marker=marker, ms=3)
        if ev is not None:
            ax.plot(epochs, ev, color=SLOT2, label="eval", marker=marker, ms=3)
        vals = np.concatenate([tr, ev if ev is not None else []])
        vals = vals[np.isfinite(vals)]
        if log and len(vals) and vals.min() > 0:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(PlainLogFormatter())
            ax.yaxis.set_minor_formatter(PlainLogFormatter(labelOnlyBase=False))
        if key.startswith("z"):
            ax.axhline(COLLAPSE_ZSTD, color=MUTED, linewidth=0.8, linestyle="--") # threshold of the collapse check
        ax.set_title(name)

    # lr in 1e-4 units, best.pt (lowest eval criterion, as train.py)
    axes["lr"].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    crit = panels["crit"][2]
    if np.isfinite(crit).any():
        b = int(np.nanargmin(crit))
        left = b <= (len(epochs) - 1) / 2 # the label goes toward the middle of the panel
        axes["crit"].plot(b, crit[b], "o", color=SLOT2, ms=6, mec=SURFACE, mew=1.5, zorder=3)
        axes["crit"].annotate(f"best.pt, epoch {b}", (b, crit[b]), xytext=(6 if left else -6, 8), textcoords="offset points",
                              ha="left" if left else "right", va="bottom", color=INK2, fontsize=8)
    for key in mosaic[-1][::3]:
        axes[key].set_xlabel("epoch")
    handles, labels = axes["total"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=2)
    fig.suptitle(f"{title} · {len(epochs)} epochs".strip(" ·"), x=0.01, ha="left", color=INK, fontsize=12)
    _save(fig, out)


def plot_run(run_dir):
    """redraw <run>/plots/train.png from <run>/latest.pt"""
    import torch
    run_dir = Path(run_dir)
    s = torch.load(run_dir / "latest.pt", map_location="cpu")
    Ks = s.get("Ks")
    plot_train(s["logs"], run_dir / "plots" / "train.png", f"{run_dir.name} · Ks {Ks}", Ks)


# ------------------------------------------------ Benchmarks ------------------------------------------------


def read_results(csv_path):
    """rows of results.csv (benchmark.py), the last one of each (run, bench) : a redone benchmark replaces the old one"""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if None not in r.values()] # a line being written by another process
    last = {}
    for r in rows:
        last[(r["run"], r["bench"])] = r # the key keeps the position of its first line
    return list(last.values())


def _ci95(rows):
    """95% interval of a success rate measured on num_eval episodes (normal approximation of the binomial), in %"""
    p = np.array([float(r["success_rate"]) for r in rows]) / 100
    n = np.array([int(r["num_eval"]) for r in rows])
    return 100 * 1.96 * np.sqrt(p * (1 - p) / n)


def plot_bench_run(rows, out, title=""):
    """success rate of each benchmark of one run, one panel per protocol (goal distance, start)"""
    groups = {}
    for r in rows:
        groups.setdefault((int(r["goal_offset_steps"]), r["random_start"] == "True"), []).append(r)
    h = max(len(g) for g in groups.values())
    fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 0.42 * h + 1.3), squeeze=False, layout="constrained")
    for ax, ((goal, random_start), g) in zip(axes[0], groups.items()):
        y = -np.arange(len(g))
        sr, ci = np.array([float(r["success_rate"]) for r in g]), _ci95(g)
        xerr = [np.minimum(ci, sr), np.minimum(ci, 100 - sr)] # interval kept inside [0, 100] %
        ax.barh(y, sr, height=0.6, color=SLOT1, xerr=xerr, error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
        for yi, s, c in zip(y, sr, ci):
            ax.text(min(s + c, 100) + 2, yi, f"{s:.0f}%", va="center", color=INK, fontsize=9)
        ax.set_yticks(y, [r["bench"] for r in g])
        ax.set_ylim(y.min() - 0.6, 0.6)
        ax.set_xlim(0, 112)
        ax.set_xticks(range(0, 101, 20))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"goal {goal} steps later, {'random start' if random_start else 'start of the episode'}")
        ax.set_xlabel(f"success rate (%), bar = 95% interval ({g[0]['num_eval']} episodes)")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    _save(fig, out)


def plot_bench_all(rows, out):
    """success rate of every run x every benchmark (runs sorted by name, benchmarks in the order of results.csv)"""
    runs = sorted({r["run"] for r in rows})
    benches = list(dict.fromkeys(r["bench"] for r in rows))
    M = np.full((len(runs), len(benches)), np.nan)
    for r in rows:
        M[runs.index(r["run"]), benches.index(r["bench"])] = float(r["success_rate"])

    fig, ax = plt.subplots(figsize=(1.1 * len(benches) + 3.2, 0.42 * len(runs) + 1.6), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(EMPTY)
    mesh = ax.pcolormesh(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=100, edgecolors=SURFACE, linewidth=2)
    for i in range(len(runs)):
        for j in range(len(benches)):
            v = M[i, j]
            ax.text(j + 0.5, i + 0.5, "–" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color=MUTED if np.isnan(v) else ("#ffffff" if v > 60 else INK))
    ax.set_xticks(np.arange(len(benches)) + 0.5, benches, rotation=30, ha="left")
    ax.set_yticks(np.arange(len(runs)) + 0.5, runs)
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("success rate (%)", color=INK2)
    cbar.outline.set_visible(False)
    ax.set_title("success rate (%), last result of each benchmark", pad=12)
    _save(fig, out)


def plot_benches(csv_path):
    """<root>/plots/bench.png + <root>/<run>/plots/bench.png of every run of <root>/results.csv"""
    csv_path = Path(csv_path)
    rows = read_results(csv_path)
    if not rows:
        return
    plot_bench_all(rows, csv_path.parent / "plots" / "bench.png")
    for run in dict.fromkeys(r["run"] for r in rows):
        if (csv_path.parent / run).is_dir():
            plot_bench_run([r for r in rows if r["run"] == run], csv_path.parent / run / "plots" / "bench.png", run)


# ------------------------------------------------ Main ------------------------------------------------


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "checkpoints"
    one_run = (target / "latest.pt").exists()
    runs = [target] if one_run else sorted(d for d in target.iterdir() if (d / "latest.pt").exists())
    for d in runs:
        plot_run(d)
        print(f"[plot] {d / 'plots' / 'train.png'}")
    csv_path = (target.parent if one_run else target) / "results.csv"
    if csv_path.exists():
        plot_benches(csv_path)
        print(f"[plot] benchmarks of {csv_path}")
