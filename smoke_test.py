"""Synthetic-data smoke test: exercises the full pipeline without real data.

Generates small fake NIfTI volumes + labels, runs preprocessing, a 1-epoch
training run with 2 folds, and the submission inference path.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

N_UIDS = 24


def make_synthetic_data(tmp: Path, n=24, size=32):
    import nibabel as nib

    niftis = tmp / "niftis"
    niftis.mkdir(exist_ok=True)
    labels = {}
    rng = np.random.RandomState(0)
    for i in range(n):
        uid = f"syn{i:04d}"
        vol = rng.rand(size, size, size) * 200
        # put a "hot" blob off-center for some to simulate pathology
        labels[uid] = 1.0 if i % 2 == 0 else 0.0
        img = nib.Nifti1Image(vol.astype(np.uint16), np.eye(4) * 3.0)
        nib.save(img, niftis / f"{uid}.nii.gz")
    labels_df = f"uid,is_pathologic\n" + "\n".join(f"{u},{labels[u]}" for u in labels)
    (tmp / "train_labels.csv").write_text(labels_df)
    fmt = "uid,is_pathologic\n" + "\n".join(f"{u}," for u in list(labels)[:8])
    (tmp / "submission_format.csv").write_text(fmt)
    return niftis, labels


def test_preprocess():
    from src.preprocess import load_volume

    with tempfile.TemporaryDirectory() as td:
        niftis, _ = make_synthetic_data(Path(td), n=2, size=32)
        vol = load_volume(next(niftis.glob("*.nii.gz")))
        assert vol.shape == (64, 64, 64)
        assert vol.dtype == np.float32
        assert 0.0 <= vol.min() and vol.max() <= 1.0
    print("preprocess: OK")


def test_train():
    from src.train import fit_sigmoid_calibration

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        niftis, labels = make_synthetic_data(tmp, n=N_UIDS, size=32)
        out = tmp / "weights"
        sys.argv = [
            "train", "--data-dir", str(tmp), "--out-dir", str(out),
            "--folds", "2", "--epochs", "1", "--batch-size", "4",
            "--workers", "0", "--cpu", "--no-amp",
        ]
        from src.train import main as train_main
        train_main()
        assert len(list(out.glob("fold_*.pt"))) == 2
        assert (out / "calib.json").exists()
        calib = json.loads((out / "calib.json").read_text())
        assert "a" in calib and "b" in calib
        # calibration sanity: output is a valid probability and monotone in logit
        a, b = calib["a"], calib["b"]
        p0 = 1 / (1 + np.exp(-(a * -3 + b)))
        p1 = 1 / (1 + np.exp(-(a * 3 + b)))
        assert 0 < p0 < 1 and 0 < p1 < 1
        assert (p1 - p0) * np.sign(a) > 0
        print("train: OK")


def test_submission_zip():
    import make_submission
    import zipfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        niftis, labels = make_synthetic_data(tmp, n=N_UIDS, size=32)
        out = tmp / "weights"
        sys.argv = [
            "train", "--data-dir", str(tmp), "--out-dir", str(out),
            "--folds", "2", "--epochs", "1", "--batch-size", "4",
            "--workers", "0", "--cpu", "--no-amp",
        ]
        from src.train import main as train_main
        train_main()

        import os
        make_submission.main(weights=out, out=tmp / "submission.zip")

        with zipfile.ZipFile(tmp / "submission.zip") as zf:
            names = zf.namelist()
            assert "main.py" in names
            assert "model/calib.json" in names
            assert any(n.startswith("model/fold_") for n in names)
            assert "src/preprocess.py" in names and "src/model.py" in names
    print("submission zip: OK")


def test_inference():
    import os
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        niftis, labels = make_synthetic_data(tmp, n=N_UIDS, size=32)
        out = tmp / "weights"
        sys.argv = [
            "train", "--data-dir", str(tmp), "--out-dir", str(out),
            "--folds", "2", "--epochs", "1", "--batch-size", "4",
            "--workers", "0", "--cpu", "--no-amp",
        ]
        from src.train import main as train_main
        train_main()

        # run the submission entrypoint against the synthetic data
        env = dict(os.environ, DAT_DATA_ROOT=str(tmp), DAT_MODEL_DIR=str(out))
        r = subprocess.run(
            [sys.executable, "submission_src/main.py"], cwd=ROOT, env=env,
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        import pandas as pd
        sub = pd.read_csv(ROOT / "submission.csv")
        fmt = pd.read_csv(tmp / "submission_format.csv")
        assert len(sub) == len(fmt) == 8
        assert sub["is_pathologic"].astype(float).between(0, 1).all()
        print("inference: OK")


if __name__ == "__main__":
    test_preprocess()
    test_train()
    test_submission_zip()
    test_inference()
    print("ALL SMOKE TESTS PASSED")
