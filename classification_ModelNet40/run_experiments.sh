#!/usr/bin/env bash
# AdaGAM-3D experiment matrix (Linux / WSL2 / Colab / Git-Bash).
#   GPU=0 EPOCHS=100 R=8 bash run_experiments.sh train     # base, adagam, base_aug, adagam_aug (sequential)
#   GPU=0 EPOCHS=100 R=8 bash run_experiments.sh eval      # ModelNet-C (last ckpt) -> results.md
#   DRY=1 bash run_experiments.sh train                    # print the commands only
# Optional env: MODEL (pointMLPElite), SEED (42), BATCH (32), AMP (1), PARALLEL (0), WORKERS, PYTHON.
# Optional ablations (off by default) - uncomment to include them:
#   EXTRA="--ablations"     # adagam_aug_nocons (no consistency loss), adagam_std (std pooling), adagam_patch (--gam_norm patch)
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-}"
if [[ "$MODE" != "train" && "$MODE" != "eval" && "$MODE" != "all" ]]; then
  echo "usage: [GPU=0] [EPOCHS=100] [R=8] bash run_experiments.sh {train|eval|all}" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"
ARGS=("$MODE")
if [[ "${DRY:-0}" == "1" ]]; then ARGS+=("--dry_run"); fi
# shellcheck disable=SC2206
EXTRA_ARR=(${EXTRA:-})
export GPU="${GPU:-0}" EPOCHS="${EPOCHS:-100}" R="${R:-8}" MODEL="${MODEL:-pointMLPElite}"
"$PYTHON" run_experiments.py "${ARGS[@]}" ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}
