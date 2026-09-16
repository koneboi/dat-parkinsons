"""Local end-to-end test of the submission code without Docker.

Runs submission_src/main.py against a local data dir and checks that the
produced submission.csv is well-formed (uids match, probabilities in [0,1]).
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
    assert (data_dir / "submission_format.csv").exists(), (
        f"{data_dir}/submission_format.csv missing. "
        "Download smoke test data and place its submission_format.csv + niftis/ here."
    )
    env = {**os.environ, "DAT_DATA_ROOT": str(data_dir)}
    subprocess.run([sys.executable, "submission_src/main.py"], cwd=ROOT, env=env, check=True)

    out = ROOT / "submission.csv"
    import pandas as pd

    sub = pd.read_csv(out)
    fmt = pd.read_csv(data_dir / "submission_format.csv")
    assert list(sub.columns) == list(fmt.columns), "Column mismatch"
    assert len(sub) == len(fmt), "Row count mismatch"
    assert set(sub.iloc[:, 0]) == set(fmt.iloc[:, 0]), "uid mismatch"
    probs = sub["is_pathologic"].astype(float)
    assert probs.between(0, 1).all(), "Probabilities out of [0,1]"
    print(f"OK: {len(sub)} rows, mean pred {probs.mean():.4f}, "
          f"min {probs.min():.4f}, max {probs.max():.4f}")


if __name__ == "__main__":
    main()
