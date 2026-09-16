# Recompute per-member honest single-fold OOF with EXACTLY main.py member_logits
# arithmetic (flip dims [3],[2]) for 2D MIP members. Saves tta and no-tta variants
# so we can pick the per-member best and rebuild the calibration on the runtime scale.
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.ensemble import _parse_preproc
from src.train_resnet import MIPDataset, build_model


def oof_for(member, tta_on, device):
    from src.train_resnet import MIPDataset as DS
    meta = json.load(open(f"{member}/calib.json"))
    seed = int(meta["seed"]); size = int(meta["size"]); bb = meta["backbone"]
    stack = meta.get("stack")
    preproc = _parse_preproc(meta.get("preproc"))
    proj = meta.get("proj", "mip")
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    model = build_model(backbone=bb, dropout=0.5, in_channels=6 if stack == "aip" else 3)
    oof = np.zeros(len(uids), dtype=np.float32)
    folds = sorted(Path(member).glob("fold_*.pt"))
    for f, (_, va) in enumerate(skf.split(uids, y)):
        model.load_state_dict(torch.load(folds[f], map_location="cpu", weights_only=True))
        model.eval().to(device)
        ds = DS([uids[i] for i in va], nifti_dir, labels, augment=False, use_cache=True,
                size=size, rot=0.0, preproc=preproc, proj=proj, stack=stack)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        logits = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(device)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    if tta_on:
                        # EXACTLY main.py member_logits 2D: dims [3],[2]
                        v0 = model(x)
                        v1 = model(torch.flip(x, dims=[3]))
                        v2 = model(torch.flip(x, dims=[2]))
                        v3 = model(torch.flip(torch.flip(x, dims=[2]), dims=[3]))
                        lg = (v0 + v1 + v2 + v3) / 4.0
                    else:
                        lg = model(x)
                logits.append(lg.float().cpu().numpy().ravel())
        oof[va] = np.concatenate(logits)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default="oof_v9_runtime.npz")
    ap.add_argument("--members", nargs="*", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    global uids, y, labels, nifti_dir
    labels_df = pd.read_csv("data/train_labels.csv")
    labels = dict(zip(labels_df["uid"], labels_df["is_pathologic"].astype(float)))
    uids = labels_df["uid"].tolist()
    y = labels_df["is_pathologic"].astype(int).to_numpy()
    nifti_dir = Path("data/niftis")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    members = args.members or ["weights_r50_s7", "weights_dense_s999", "weights_str120_dense",
                               "weights_r50_s123", "weights_r50_str160_128", "weights_r50_mipaip",
                               "weights_r50_ema", "weights_dense_ema", "weights_str120_r50",
                               "weights_r50_s999", "weights_dense_str160_128", "weights_r18_s999"]
    out = {}
    for mm in members:
        k = mm.replace("/", "_")
        print(f"[{device}] {mm}: tta-on ...", flush=True)
        out[f"{k}_tta"] = oof_for(mm, True, device)
        print(f"    std {out[f'{k}_tta'].std():.4f} AUC {roc_auc_score(y, out[f'{k}_tta']):.4f}", flush=True)
        print(f"[{device}] {mm}: no-tta ...", flush=True)
        out[f"{k}_notta"] = oof_for(mm, False, device)
        print(f"    std {out[f'{k}_notta'].std():.4f} AUC {roc_auc_score(y, out[f'{k}_notta']):.4f}", flush=True)
    out["y"] = y
    np.savez(args.o, **out)
    print(f"saved -> {args.o}")


if __name__ == "__main__":
    main()