"""Plot Push-T goal-reaching success against the goal horizon, from the eval results in $STABLEWM_HOME/eval.

Reads every 50-episode run of a policy (default lewam-pusht) and groups it by protocol (random start, or long
horizon = eval.random_start=false), goal offset and mode. Writes a figure and a Markdown page with the tables.
Usage: python scripts/plot_horizons.py [--policy lewam-pusht] [--out docs/horizons]
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODES = {"reactive": "reactive policy", "best_of_k": "best-of-32", "grad": "gradient planner"}
COLORS = {"reactive": "#2a78d6", "best_of_k": "#eb6834", "grad": "#1baf7a"}  # categorical slots 1-3, in order
PROTOCOLS = {"random": "Random start", "long": "Long horizon (start at frame 0)"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(policy: str) -> dict:
    """(protocol, goal offset, mode) -> {seed: success rate}; a rerun of a seed replaces the earlier one."""
    sr = defaultdict(dict)
    root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel")) / "eval"
    for f in sorted(root.glob(f"*/{policy}/*_results.txt"), key=lambda p: p.stat().st_mtime):
        for block in f.read_text().split("==== CONFIG ====")[1:]:
            if "num_eval: 50" not in block or "success_rate" not in block:
                continue
            get = lambda pat: re.search(pat, block, re.M).group(1)  # noqa: E731
            mode = "reactive" if get(r"^  mode: (\S+)") == "lewam_policy" else get(r"^  plan_mode: (\S+)")
            proto = "random" if get(r"^  random_start: (\S+)") == "true" else "long"
            sr[(proto, int(get(r"^  goal_offset_steps: (\d+)")), mode)][get(r"^seed: (\S+)")] = float(
                get(r"'success_rate': ([\d.]+)"))
    return sr


def stats(seeds: dict) -> tuple[float, float, int]:
    v = list(seeds.values())
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)


def plot(sr: dict, path: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (proto, title) in zip(axes, PROTOCOLS.items()):
        hs = sorted({h for p, h, _ in sr if p == proto})
        for mode in MODES:
            pts = [(h, *stats(sr[(proto, h, mode)])) for h in hs if sr.get((proto, h, mode))]
            if not pts:
                continue
            x, m, s, _ = map(np.array, zip(*pts))
            ax.fill_between(x, np.clip(m - s, 0, 100), np.clip(m + s, 0, 100), color=COLORS[mode], alpha=0.1, lw=0)
            ax.plot(x, m, color=COLORS[mode], lw=2, marker="o", ms=6, mec="white", mew=1.5,
                    solid_capstyle="round", solid_joinstyle="round", label=MODES[mode])
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("goal horizon H (env steps, budget 2H)")
        if hs:
            ax.set_xticks(hs)
            ax.set_xlim(hs[0] - 0.04 * (hs[-1] - hs[0] or 10), hs[-1] + 0.04 * (hs[-1] - hs[0] or 10))
        else:
            ax.text(0.5, 0.5, "running…", transform=ax.transAxes, ha="center", va="center", color=INK_2)
            ax.set_xticks([])
        ax.grid(axis="y", color=GRID, lw=1)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_ylim(0, 102)
    axes[0].set_yticks(range(0, 101, 20))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.9, 1.0))
    fig.suptitle("LeWAM on Push-T: success vs. goal horizon", x=0.07, ha="left", fontsize=12, color=INK)
    fig.text(0.07, 0.885, "mean ± std over seeds 42, 0, 1 (50 episodes each); shaded band = ±1 std",
             fontsize=9, color=INK_2)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.07, right=0.98)
    fig.savefig(path, dpi=160, facecolor="white")


def table(sr: dict, proto: str) -> str:
    rows = ["| H (env steps) | budget | " + " | ".join(MODES.values()) + " |", "|---:|---:|---|---|---|"]
    for h in sorted({h for p, h, _ in sr if p == proto}):
        cells = []
        for mode in MODES:
            if not sr.get((proto, h, mode)):
                cells.append("n/a" if (proto, mode) == ("long", "reactive") else "running")
                continue
            m, s, n = stats(sr[(proto, h, mode)])
            cells.append(f"{m:.1f} ± {s:.1f}" + ("" if n == 3 else f" ({n}/3 seeds)"))
        rows.append(f"| {h} | {2 * h} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="lewam-pusht")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "horizons"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr = load(args.policy)
    plot(sr, out / "pusht_horizons.png")
    (out / "PUSHT_HORIZONS.md").write_text(f"""# Push-T: success vs. goal horizon

Goal-reaching success (%) of the released [Push-T checkpoint](https://huggingface.co/LeWAM/lewam-pusht), mean ± std
over seeds 42, 0 and 1 with 50 episodes each. The checkpoint was trained with goals up to `H_max = 10` chunks of
`fs = 5` steps (50 env steps); beyond that the policy's horizon input is clamped at 50. Generated by
`python scripts/plot_horizons.py` from the results in `$STABLEWM_HOME/eval/`.

![Push-T success vs goal horizon](pusht_horizons.png)

## Random start

The default protocol ([REPRODUCE.md](../REPRODUCE.md#evaluation)): start at a random step of a dataset episode and
reach the frame H steps later within 2H actions.

```bash
python scripts/eval_lewam.py --config-name pusht policy=lewam-pusht seed=<seed> \\
    eval.goal_offset_steps=<H> eval.eval_budget=<2H> <mode overrides>
```

{table(sr, "random")}

## Long horizon

Start at frame 0 of episodes at least 101 steps long (and longer than H), replan every 25 actions, goal H steps
ahead, budget 2H. At H = 200 only the 505 episodes longer than 200 steps qualify, so that point is drawn from the
longest demonstrations. Planners only: the reactive policy predicts 10-action chunks, so it cannot execute 25
actions per replan.

```bash
python scripts/eval_lewam.py --config-name pusht policy=lewam-pusht seed=<seed> <mode overrides> \\
    eval.random_start=false eval.min_episode_len=101 eval.exec_actions=25 \\
    eval.goal_offset_steps=<H> eval.eval_budget=<2H>
```

{table(sr, "long")}

Mode overrides: reactive policy `eval.mode=lewam_policy`; best-of-32 `eval.mode=lewam_plan eval.plan_mode=best_of_k`;
gradient planner `eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01 eval.grad_steps=5`
(plus `eval.exec_actions=5` under random start).
""")
    print("wrote", out)


if __name__ == "__main__":
    main()
