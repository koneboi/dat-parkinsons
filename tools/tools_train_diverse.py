"""Train diverse-architecture candidates and evaluate OOF with correct runtime flips.
Trains efficientnet_b0/b1 (different feature extractors from resnet/densenet) and
optionally convnext_tiny. All use proven recipe: size=128, rot=15, 60 epochs."""
import subprocess, sys, json, time
from pathlib import Path

PY = ".venv/bin/python"
RUNS = [
    # (name, extra args as flat list)
    ("weights_effb0_s42",   ["--backbone", "efficientnet_b0", "--seed", "42"]),
    ("weights_effb0_s1234", ["--backbone", "efficientnet_b0", "--seed", "1234"]),
    ("weights_effb1_s42",   ["--backbone", "efficientnet_b1", "--seed", "42"]),
]

def run_one(name, extra_args):
    if Path(name).exists() and len(list(Path(name).glob("fold_*.pt"))) >= 5:
        print(f"[SKIP] {name} already has 5 folds", flush=True)
        return True
    cmd = [PY, "-m", "src.train_resnet",
           "--out-dir", name, "--folds", "5", "--epochs", "60",
           "--batch-size", "16", "--accum", "2", "--size", "128",
           "--rot", "15", "--ema", "0.99", "--workers", "2"]
    cmd.extend(extra_args)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=False, timeout=3600)
    dt = time.time() - t0
    ok = p.returncode == 0
    if ok:
        n = len(list(Path(name).glob("fold_*.pt")))
        print(f"[DONE] {name}: {n} folds in {dt/60:.1f}min", flush=True)
    else:
        print(f"[FAIL] {name}: returncode={p.returncode} in {dt/60:.1f}min", flush=True)
    return ok

if __name__ == "__main__":
    for name, args in RUNS:
        print(f"\n===== {name} {args} =====", flush=True)
        run_one(name, args)
