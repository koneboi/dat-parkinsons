#!/usr/bin/env bash
# Train multiple backbones/seeds sequentially on the shared GPU.
set -e
PY=".venv/bin/python"
cd "$(dirname "$0")"

run() {
  echo "===== $* ====="
  PYTHONPATH=. $PY -m src.train_resnet "$@"
}

# 1. resnet18 @128 seed 123, rotation aug
run --out-dir weights_resnet18_s123 --folds 5 --epochs 60 --batch-size 32 \
    --size 128 --backbone resnet18 --seed 123 --rot 15 --workers 2

# 2. resnet18 @128 seed 42, rotation aug + EMA
run --out-dir weights_resnet18_rot --folds 5 --epochs 60 --batch-size 32 \
    --size 128 --backbone resnet18 --seed 42 --rot 15 --ema 0.99 --workers 2

# 3. resnet34 @128 seed 42, rotation aug
run --out-dir weights_resnet34 --folds 5 --epochs 60 --batch-size 16 \
    --size 128 --backbone resnet34 --seed 42 --rot 15 --workers 2

# 4. resnet50 @128 seed 42, rotation aug, grad accum
run --out-dir weights_resnet50 --folds 5 --epochs 60 --batch-size 8 --accum 2 \
    --size 128 --backbone resnet50 --seed 42 --rot 15 --workers 2

# 5. efficientnet_b0 @128 seed 42, rotation aug
run --out-dir weights_effb0 --folds 5 --epochs 60 --batch-size 32 \
    --size 128 --backbone efficientnet_b0 --seed 42 --rot 15 --workers 2

# 6. densenet121 @128 seed 42, rotation aug
run --out-dir weights_densenet --folds 5 --epochs 60 --batch-size 16 \
    --size 128 --backbone densenet121 --seed 42 --rot 15 --workers 2

echo "ALL DONE"
