#!/usr/bin/env bash
# AdaGAM experiment matrix. Run from classification_ModelNet40/.
#
#   bash run_experiments.sh train      # launch training runs in the background (logs in logs/)
#   bash run_experiments.sh eval       # ModelNet-C eval of every finished run + results.md
#
# Edit the variables below before launching.
set -euo pipefail

GPU=${GPU:-0}                 # default: GPU 0, override with GPU=N
MODEL=${MODEL:-pointMLP}      # pointMLP (13.2M) or pointMLPElite (0.7M, much faster)
EPOCHS=${EPOCHS:-300}         # same for every run; lower it if short on time
R=${R:-8}                     # AdaGAM reduction; r=8 keeps param overhead <2% on pointMLP (r=4 is +2.65%)
SEED=${SEED:-42}
LAMBDA=${LAMBDA:-1.0}         # consistency weight
# Corruptions seen during training. The rest are "unseen" at test time.
TRAIN_CORR=${TRAIN_CORR:-jitter,drop_global,add_global,scale}   # held out: drop_local, add_local, rotate

COMMON="--model $MODEL --epoch $EPOCHS --seed $SEED --workers 4"
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
    for name in "${!RUNS[@]}"; do
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
  *)
    echo "usage: bash run_experiments.sh {train|eval}"; exit 1 ;;
esac
