#!/usr/bin/env bash
# AdaGAM experiment matrix. Run from classification_ModelNet40/.
#
#   bash run_experiments.sh train      # launch training runs in the background (logs in logs/)
#   bash run_experiments.sh eval       # ModelNet-C eval of every finished run + results.md
#   bash run_experiments.sh analyze    # the two explainability analyses (saliency, CKA drift)
#
# Launch a subset with ONLY, e.g. the two headline runs first:
#   GPU=0 ONLY="base adagam" bash run_experiments.sh train
# A killed run resumes from its last_checkpoint.pth when launched again with the same settings.
#
# Edit the variables below before launching.
set -euo pipefail

GPU=${GPU:-0}                 # default: GPU 0, override with GPU=N
MODEL=${MODEL:-pointMLP}      # pointMLP (13.2M) or pointMLPElite (0.7M, much faster)
EPOCHS=${EPOCHS:-300}         # same for every run; lower it if short on time
R=${R:-8}                     # AdaGAM reduction; r=8 keeps param overhead <2% on pointMLP (r=4 is +2.65%)
SEED=${SEED:-42}
EVAL_EVERY=${EVAL_EVERY:-1}   # test on ModelNet40 every N epochs (always on the last); 5 saves ~20% time
LAMBDA=${LAMBDA:-1.0}         # consistency weight
# Corruptions seen during training. The rest are "unseen" at test time.
TRAIN_CORR=${TRAIN_CORR:-jitter,drop_global,add_global,scale}   # held out: drop_local, add_local, rotate

COMMON="--model $MODEL --epoch $EPOCHS --seed $SEED --workers 4 --eval_every $EVAL_EVERY"
mkdir -p logs

declare -A RUNS=(
  # name                       extra args
  [base]="--gam static"
  [adagam]="--gam adagam --reduction $R"
  [base_aug]="--gam static --aug corrupt --train_corruptions $TRAIN_CORR --consistency $LAMBDA"
  [adagam_aug]="--gam adagam --reduction $R --aug corrupt --train_corruptions $TRAIN_CORR --consistency $LAMBDA"
  # ---- optional ablations (uncomment if there is GPU time) ----
  # [adagam_aug_nocons]="--gam adagam --reduction $R --aug corrupt --train_corruptions $TRAIN_CORR"
  # [adagam_aug_std]="--gam adagam --reduction $R --use_std --aug corrupt --train_corruptions $TRAIN_CORR --consistency $LAMBDA"
  # [adagam_patch]="--gam adagam --reduction $R --gam_norm patch"
)

case "${1:-}" in
  train)
    # Download ModelNet40 once here; otherwise the parallel runs race on the same zip file.
    python -c "from data import download; download()"
    [ -d data/modelnet40_ply_hdf5_2048 ] || { echo "ModelNet40 download failed"; exit 1; }
    for name in ${ONLY:-${!RUNS[@]}}; do
      [ -n "${RUNS[$name]+x}" ] || { echo "unknown run: $name"; exit 1; }
      echo "launching $name"
      CUDA_VISIBLE_DEVICES=$GPU nohup python main.py $COMMON ${RUNS[$name]} --msg "$name" \
        > "logs/$name.log" 2>&1 &
      sleep 5
    done
    echo "tail -f logs/*.log   |   nvidia-smi"
    ;;
  eval)
    ref=""
    base_dir="checkpoints/${MODEL}-base-$SEED"
    if [ -f "$base_dir/last_checkpoint.pth" ]; then
      CUDA_VISIBLE_DEVICES=$GPU python eval_modelnet_c.py --ckpt "$base_dir" | tee "logs/eval_base.log"
      ref="--ref $base_dir/modelnet_c.json"
    fi
    for name in "${!RUNS[@]}"; do
      d="checkpoints/${MODEL}-$name-$SEED"
      [ "$name" = base ] && continue
      [ -f "$d/last_checkpoint.pth" ] || { echo "skip $name (no checkpoint)"; continue; }
      CUDA_VISIBLE_DEVICES=$GPU python eval_modelnet_c.py --ckpt "$d" $ref | tee "logs/eval_$name.log"
    done
    python summarize.py
    ;;
  analyze)
    base_dir="checkpoints/${MODEL}-base-$SEED"
    ada_dir="checkpoints/${MODEL}-adagam-$SEED"
    for d in "$base_dir" "$ada_dir"; do
      [ -f "$d/last_checkpoint.pth" ] || { echo "missing $d/last_checkpoint.pth"; exit 1; }
    done
    CUDA_VISIBLE_DEVICES=$GPU python xai_saliency.py --base "$base_dir" --adagam "$ada_dir"       --data data/modelnet_c --out xai_out/saliency | tee logs/xai_saliency.log
    CUDA_VISIBLE_DEVICES=$GPU python cka_drift.py --base "$base_dir" --adagam "$ada_dir"       --data data/modelnet_c --out xai_out/cka | tee logs/cka_drift.log
    ;;
  *)
    echo "usage: bash run_experiments.sh {train|eval|analyze}"; exit 1 ;;
esac
