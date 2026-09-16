"""Package the submission for the DaT Parkinson's Challenge.

Creates a deterministic submission.zip with main.py at root, matching the
official runtime layout:
    submission.zip
    ├── main.py
    ├── src/{__init__,align,model}.py
    └── model/{fold_*.pt, calib.json}

Usage:
    python make_submission.py                 # uses weights/
    python make_submission.py weights_resnet  # or any weights dir
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_src"
OUT = ROOT / "submission.zip"


def main():
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "weights"
    fold_weights = sorted(weights.glob("*_fold_*.pt"))
    calib = weights / "calib.json"
    assert fold_weights, f"No fold_*.pt weights found in {weights}. Run train.py first."
    assert calib.exists(), f"calib.json not found in {weights}. Run train.py first."

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(SRC / "main.py", "main.py")
        for py in sorted((SRC / "src").glob("*.py")):
            zf.write(py, f"src/{py.name}")
        for w in fold_weights:
            zf.write(w, f"model/{w.name}")
        zf.write(calib, "model/calib.json")

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT} ({size_mb:.1f} MB) with {len(fold_weights)} models")


if __name__ == "__main__":
    main()

