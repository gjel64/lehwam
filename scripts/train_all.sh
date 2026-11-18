#!/usr/bin/env bash
# Train a list of configs (one queue per GPU) and resume them until they are all finished,
# then benchmark each finished run on every line of BENCHS (on the GPU that is free, before training the next run).
#
#   bash scripts/train_all.sh              # trains now, non-stop
#   bash scripts/train_all.sh --night      # weeknights NIGHT_START -> NIGHT_END + the whole weekend
#   nohup bash scripts/train_all.sh --night > checkpoints/train_all.log 2>&1 &     # keeps running after logout
#
# Each run lives in checkpoints/<name>/ : best.pt, latest.pt, train.log, hydra/<launch>/ (full config of each launch),
# DATA (its dataset : pusht125 if scripts/prepare_pusht.py prepared it, else pusht200, kept for the resumes).
# Runs are always launched with train.resume=True : from scratch the first time, from latest.pt after a pause.
# --night : at NIGHT_END the run finishes its current epoch (latest.pt saved), then waits for NIGHT_START.
# Markers in checkpoints/<name>/ : DONE or FAILED (crash) -> the run is skipped, rm the marker to retry it.
# Changed the overrides of a run that already started ? rm -r checkpoints/<name>, otherwise it resumes the old one.
# Benchmarks : checkpoints/<name>/bench/<bench>/ (bench.log, results.txt, result.json, DONE or FAILED marker)
#              + one line per benchmark in checkpoints/results.csv (column -s, -t < checkpoints/results.csv)
#              rm -r checkpoints/<name>/bench/<bench> to redo one, a new line in BENCHS is run on every finished run.
#              --night : no benchmark starts after NIGHT_END, but the one running is finished (a few minutes).
# Plots : checkpoints/<name>/plots/train.png (redrawn after each epoch), checkpoints/<name>/plots/bench.png and
#         checkpoints/plots/bench.png (all the runs, redrawn after each benchmark). Redraw all : python scripts/plot.py
# Stop : Ctrl-C or kill <pid of this script>, the runs resume from latest.pt at the next launch.
# Only one launch of this script at a time (it frees the GPU claims left by the previous one).

set -f   # no glob : Hydra lists like data.Ks=[5,25] stay as written

# ================================ EDIT HERE ================================

GPUS=(0)          # one queue per GPU : each GPU takes the next run that nobody has done or taken
PYTHON=python

# dataset : the first one prepared (data/decomp/<d>/pixels.npy) by scripts/prepare_pusht.py, which prepares pusht125
# when its frames fit in RAM. A run keeps the dataset it started with (checkpoints/<name>/DATA), its benchmarks too.
DATASETS=(pusht125 pusht200)

# overrides of config/train.yaml common to every run (and to their benchmarks)
COMMON=(
)

# one run per line : "name | overrides" (no space inside a value)
RUNS=(
  "lewam_1lvl          | data.Ks=[5]          model.emb_dims=[192]"
  "lehwam_2lvl         | data.Ks=[5,25]       model.emb_dims=[192,128]"
  "lehwam_3lvl         | data.Ks=[5,25,125]   model.emb_dims=[192,128,96]"
  "lehwam_3lvl_detach  | data.Ks=[5,25,125]   model.emb_dims=[192,128,96] model.z_below_detach=True"
  "lehwam_2lvl_k50     | data.Ks=[5,50]       model.emb_dims=[192,128]"
  "lehwam_3lvl_x4      | data.Ks=[5,20,80]    model.emb_dims=[192,128,96]"
# "my_run       | model.sigreg_coef=0.1 model.z_below_detach=True train.lr=3e-4 train.epochs=100"
)

# benchmarks of every finished run : "name | overrides of config/eval.yaml" (the overrides of the run are passed too)
# short : protocol of LeWAM (goal 25 steps later, 50 actions) | long200 : start of the episode, goal 200 steps later
BENCH_COMMON=(
  eval.video=False    # swm keeps all the frames of all the envs in RAM (~6 GB for 200 envs x 400 steps)
)
LONG="eval.random_start=False task.goal_offset_steps=200 task.eval_budget=400"
BENCHS=(
  "short_dp     | eval.inference_type=direct_policy"
  "short_pdp    | eval.inference_type=persistent_direct_policy"
  "short_gb     | eval.inference_type=gradient_based"
  "long200_dp   | eval.inference_type=direct_policy            $LONG"
  "long200_pdp  | eval.inference_type=persistent_direct_policy $LONG"
  "long200_gb   | eval.inference_type=gradient_based           $LONG eval.exec_actions=25"
# "short_gb_latest | eval.inference_type=gradient_based eval.ckpt=latest"
)

# --night
NIGHT_START=18:30   # NIGHT_START > NIGHT_END : the window crosses midnight
NIGHT_END=07:00     # the current epoch is finished, then the run waits for NIGHT_START
WEEKEND=1           # 1 : also train the whole weekend (Saturday 00:00 -> Monday NIGHT_END)
export TZ=Europe/Paris
CHECK_EVERY=30      # seconds between two checks of the clock / of the runs

# ===========================================================================

cd "$(dirname "$0")/.." || exit 1   # repo root : python -m scripts.train and the relative paths of the configs
CKPT=checkpoints
NIGHT=0; [[ ${1:-} == --night ]] && NIGHT=1

say() { echo "[$(date '+%a %d/%m %H:%M')] $*"; }
minutes() { echo $(( 10#${1%:*} * 60 + 10#${1#*:} )); }      # HH:MM -> minutes since 00:00
n_epochs() { grep -c '^\[LOG\]: Epoch' "$1"; }              # train.py prints this line right after saving latest.pt

can_train() {   # true if training is allowed right now
  (( NIGHT )) || return 0
  (( WEEKEND && $(date +%u) >= 6 )) && return 0             # Saturday, Sunday
  local now; now=$(minutes "$(date +%H:%M)")
  (( now >= $(minutes $NIGHT_START) || now < $(minutes $NIGHT_END) ))
}

data_of() {   # name : dataset of the run, chosen at its first launch then kept (a resume never changes of data)
  local f=$CKPT/$1/DATA d
  if [[ ! -s $f ]]; then
    for d in "${DATASETS[@]}"; do [[ -e data/decomp/$d/pixels.npy ]] && { echo "$d" > "$f"; break; }; done
  fi
  [[ -s $f ]] && cat "$f"
}

data_args() {   # dataset -> its overrides
  echo "data.path_h5p=data/$1.h5 data.path_npy=data/decomp/$1/pixels.npy"
}

train() {   # gpu name overrides... : launch (or resume) one run, pause it at the end of the night
  local gpu=$1 name=$2; shift 2
  local dir=$CKPT/$name log=$CKPT/$name/train.log stop_at=-1 data
  touch "$log"
  data=$(data_of $name)
  if [[ -z $data ]]; then   # nothing prepared : FAILED, like a crash
    echo "no dataset in data/decomp/{${DATASETS[*]}}, run scripts/prepare_pusht.py" >> "$log"
    touch "$dir/FAILED"; say "GPU $gpu | $name : FAILED, no dataset prepared (scripts/prepare_pusht.py)"
    return
  fi
  say "GPU $gpu | $name : start/resume on $data ($*)"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PYTHON -m scripts.train "${COMMON[@]}" $(data_args $data) "$@" \
    data.ckpt_path="$dir/" train.resume=True hydra.run.dir="$dir/hydra/$(date +%Y-%m-%d_%H-%M-%S)" >> "$log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if (( stop_at < 0 )) && ! can_train; then
      stop_at=$(n_epochs "$log")
      say "GPU $gpu | $name : end of the night, pause after the current epoch"
    elif (( stop_at >= 0 && $(n_epochs "$log") > stop_at )); then
      kill $pid; wait $pid 2>/dev/null
      say "GPU $gpu | $name : paused (latest.pt saved)"
      return
    fi
    sleep $CHECK_EVERY
  done
  if wait $pid; then
    touch "$dir/DONE"; say "GPU $gpu | $name : DONE$(grep -q COLLAPSE "$log" && echo ' (collapse)')"
  else
    touch "$dir/FAILED"; say "GPU $gpu | $name : FAILED, see $log"
  fi
}

bench_todo() {   # name : true if a benchmark of this run is neither done nor failed
  local b bname
  for b in "${BENCHS[@]}"; do
    IFS='|' read -r bname _ <<< "$b"; bname=${bname// /}
    [[ -e $CKPT/$1/bench/$bname/DONE || -e $CKPT/$1/bench/$bname/FAILED ]] || return 0
  done
  return 1
}

bench() {   # gpu name overrides... : benchmark a finished run on every line of BENCHS not done yet
  local gpu=$1 name=$2; shift 2
  local b bname boverrides out
  for b in "${BENCHS[@]}"; do
    IFS='|' read -r bname boverrides <<< "$b"; bname=${bname// /}
    out=$CKPT/$name/bench/$bname
    [[ -e $out/DONE || -e $out/FAILED ]] && continue
    can_train || return 0   # end of the night : the other benchmarks wait for the next one
    mkdir -p "$out"
    say "GPU $gpu | $name : benchmark $bname"
    if CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PYTHON scripts/benchmark.py "${COMMON[@]}" $(data_args "$(data_of $name)") "$@" "${BENCH_COMMON[@]}" $boverrides \
         data.ckpt_path="$CKPT/$name/" eval.out_dir="$out" eval.tag="$bname" eval.summary_csv="$CKPT/results.csv" \
         hydra.run.dir="$out/hydra" > "$out/bench.log" 2>&1; then
      touch "$out/DONE"; say "GPU $gpu | $name : $bname -> $(grep -o 'success rate [0-9.]*%' "$out/bench.log" | tail -1)"
    else
      touch "$out/FAILED"; say "GPU $gpu | $name : benchmark $bname FAILED, see $out/bench.log"
    fi
  done
}

worker() {  # gpu : trains the runs not done yet and benchmarks the finished ones, one at a time, until there is nothing left
  local gpu=$1 run name overrides todo got
  while true; do
    can_train || say "GPU $gpu : waiting for $NIGHT_START"
    until can_train; do sleep $CHECK_EVERY; done
    todo=0; got=0
    for run in "${RUNS[@]}"; do   # in order : the benchmarks of a finished run go before the training of the next runs
      IFS='|' read -r name overrides <<< "$run"; name=${name// /}
      [[ -e $CKPT/$name/FAILED ]] && continue
      [[ -e $CKPT/$name/DONE ]] && ! bench_todo $name && continue
      todo=1
      mkdir -p "$CKPT/$name"
      mkdir "$CKPT/$name/.running" 2>/dev/null || continue   # already taken by another GPU
      got=1
      if [[ -e $CKPT/$name/DONE ]]; then bench $gpu $name $overrides; else train $gpu $name $overrides; fi
      rmdir "$CKPT/$name/.running"
      break   # back to the top : check the clock, then take the next run
    done
    (( todo )) || return 0
    (( got )) || sleep $CHECK_EVERY   # the remaining runs are on other GPUs
  done
}

mkdir -p "$CKPT"
find "$CKPT" -mindepth 2 -maxdepth 2 -name .running -type d -exec rmdir {} +   # claims left by a killed launch
trap 'trap - INT TERM; say "stop : killing the runs (they will resume from latest.pt)"; kill 0' INT TERM
say "${#RUNS[@]} runs on GPUs ${GPUS[*]}$( (( NIGHT )) && echo " | --night : $NIGHT_START -> $NIGHT_END, weekend=$WEEKEND")"
for gpu in "${GPUS[@]}"; do worker $gpu & done
wait
for run in "${RUNS[@]}"; do
  IFS='|' read -r name _ <<< "$run"; name=${name// /}
  failed=$(( $(find "$CKPT/$name/bench" -mindepth 2 -maxdepth 2 -name FAILED 2>/dev/null | wc -l) ))
  say "$name : $( [[ -e $CKPT/$name/FAILED ]] && echo FAILED || echo DONE)$( (( failed )) && echo " ($failed benchmarks FAILED)")"
done
[[ -s $CKPT/results.csv ]] && column -s, -t < "$CKPT/results.csv"
