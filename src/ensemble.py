"""Blend fold ensembles from multiple weight dirs into a single submission model.

Each weights dir contains a full 5-fold CV run (fold_*.pt + calib.json). For each
dir we regenerate out-of-fold (OOF) logits with the same TTA used in training,
standardize each member's OOF logits (z-score), average them, and re-fit the Platt
calibration on the blended OOF. All fold weights plus the blended calibration and
member config are copied into an output dir ready for make_submission.py.

Usage:
    python -m src.ensemble weights_resnet weights_resnet18_s123 -o weights_final
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .model import DATScan3DCNN
from .train_resnet import (MIPDataset, build_model, evaluate, evaluate_tta,
                           fit_sigmoid_calibration, _swap_conv1)
from .train_axial import AxialDataset, evaluate as evaluate_axial
from .train_vol3d import VolDataset, evaluate as evaluate_vol3d


def _parse_preproc(name):
    if name in (None, "brain200", ""):
        return None
    if name.startswith("str"):
        return (int(name[3:]), "striatum")
    if name.startswith("brain"):
        return (int(name[5:]), "brain")
    raise ValueError(f"unknown preproc variant: {name}")


@torch.no_grad()
def oof_logits_for_dir(weights_dir, uids, labels, nifti_dir, cfg):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    y = np.array([labels[u] for u in uids])
    with open(weights_dir / "calib.json") as f:
        meta = json.load(f)
    seed = int(meta.get("seed", 42))
    size = int(meta.get("size", 128))
    backbone = str(meta.get("backbone", "resnet18"))
    kind = str(meta.get("kind", "vol3d" if backbone == "vol3d" else "mip"))
    is_vol3d = kind == "vol3d" or backbone == "vol3d"
    is_axial = kind == "axial"
    preproc = None if (is_vol3d or is_axial) else _parse_preproc(meta.get("preproc"))
    proj = str(meta.get("proj", "mip"))
    stack = meta.get("stack")
    skf = StratifiedKFold(cfg.folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(uids), dtype=np.float32)
    folds = sorted(weights_dir.glob("fold_*.pt"))
    assert len(folds) == cfg.folds, f"{weights_dir}: expected {cfg.folds} folds, got {len(folds)}"
    for fold, (_, va) in enumerate(skf.split(uids, y)):
        if is_vol3d:
            model = DATScan3DCNN(base_channels=int(meta.get("base_channels", 32)))
        else:
            cnn_backbone = "resnet18" if is_axial else backbone
            in_channels = 6 if stack == "aip" else 3
            model = build_model(backbone=cnn_backbone, dropout=cfg.dropout,
                                in_channels=in_channels)
            if is_axial:
                model = _swap_conv1(model, int(meta.get("slices", 8)))
        model.load_state_dict(torch.load(folds[fold], map_location="cpu", weights_only=True))
        model.eval().to(cfg.device)
        val_uids = [uids[i] for i in va]
        if is_vol3d:
            ds = VolDataset(val_uids, nifti_dir, labels, augment=False,
                            use_cache=True, rot=0.0)
            evl = evaluate_vol3d
        elif is_axial:
            ds = AxialDataset(val_uids, nifti_dir, labels,
                              n_slices=int(meta.get("slices", 8)),
                              augment=False, use_cache=True, rot=0.0,
                              size=int(meta.get("size", 192)))
            evl = evaluate_axial
        else:
            ds = MIPDataset(val_uids, nifti_dir, labels, augment=False,
                            use_cache=True, size=size, rot=0.0, preproc=preproc,
                            proj=proj, stack=stack)
            angles = getattr(cfg, "tta_angles", None)
            evl = evaluate_tta if angles else evaluate
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        _, _, (logits, _) = evl(model, dl, cfg)
        oof[va] = logits
    return oof, y, {"seed": seed, "size": size, "backbone": backbone,
                    "kind": kind, "proj": proj, "stack": stack,
                    "slices": int(meta.get("slices", 8)) if is_axial else None,
                    "preproc": meta.get("preproc", "vol3d" if is_vol3d else
                                        ("axial" if is_axial else "brain200"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weight_dirs", nargs="+")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("-o", "--out-dir", default="weights_final")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tta-rot", type=float, default=0.0,
                    help="add rotation TTA at +/-this angle (deg) on top of flip TTA")
    ap.add_argument("--opt-weights", action="store_true",
                    help="optimize per-member blend weights on OOF log loss after selection")
    ap.add_argument("--opt-l2", type=float, default=1e-3,
                    help="L2 penalty on optimized weights")
    ap.add_argument("--cpu", action="store_true")
    cfg = ap.parse_args()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    cfg.amp = True
    cfg.tta = True
    if cfg.tta_rot > 0:
        cfg.tta_angles = [-cfg.tta_rot, 0.0, cfg.tta_rot]
    else:
        cfg.tta_angles = None

    data_dir = Path(cfg.data_dir)
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    labels = dict(zip(labels_df["uid"], labels_df["is_pathologic"].astype(float)))
    uids = labels_df["uid"].tolist()
    y = labels_df["is_pathologic"].astype(int).to_numpy()
    nifti_dir = data_dir / "niftis"

    dirs = [Path(d) for d in cfg.weight_dirs]
    oofs, configs = [], []
    for d in dirs:
        print(f"computing OOF for {d} ...", flush=True)
        oof, _, conf = oof_logits_for_dir(d, uids, labels, nifti_dir, cfg)
        oofs.append(oof)
        configs.append({"name": d.name, **conf})
        p = np.clip(1 / (1 + np.exp(-oof)), 1e-7, 1 - 1e-7)
        print(f"  {d.name}: AUC {roc_auc_score(y, oof):.4f} "
              f"logloss {log_loss(y, p):.4f}", flush=True)
    oofs_all = list(oofs)
    configs_all = list(configs)

    # standardize each member's logits
    zscores = [(oof - oof.mean()) / (oof.std() + 1e-9) for oof in oofs]

    # greedy forward selection on rank-AUC (monotone, robust to calibration)
    def rank_auc(blend):
        return roc_auc_score(y, blend)

    def select(zs):
        zs = list(zs)
        chosen, pool, best, best_auc = [], list(range(len(zs))), None, -1.0
        while pool:
            cands = []
            for i in pool:
                trial = np.mean([zs[j] for j in chosen + [i]], axis=0)
                cands.append((rank_auc(trial), i, trial))
            cands.sort(key=lambda t: -t[0])
            auc, i, trial = cands[0]
            if len(chosen) > 0 and auc <= best_auc + 1e-5:
                break
            chosen.append(i)
            pool.remove(i)
            best, best_auc = trial, auc
            print(f"  selected {configs[i]['name']:22s} -> blend AUC {auc:.4f}", flush=True)
        return chosen, best

    chosen, avg = select(zscores)
    print(f"BLEND (z-score avg of {len(chosen)} selected): "
          f"AUC {rank_auc(avg):.4f}")

    a, b = fit_sigmoid_calibration(avg, y)
    p = np.clip(1 / (1 + np.exp(-(a * avg + b))), 1e-7, 1 - 1e-7)
    print(f"BLEND calibrated: AUC {rank_auc(avg):.4f} "
          f"logloss {log_loss(y, p):.4f}")

    # keep only selected members
    oofs = [oofs[i] for i in chosen]
    configs = [configs[i] for i in chosen]
    dirs = [dirs[i] for i in chosen]
    zs = [zscores[i] for i in chosen]

    weights = [1.0] * len(zs)
    zs_mat = np.stack(zs, axis=1)  # (n_samples, n_members)

    if cfg.opt_weights:
        def fit(w, a, b):
            blend = zs_mat @ np.asarray(w, dtype=np.float64)
            p = np.clip(1 / (1 + np.exp(-(a * blend + b))), 1e-7, 1 - 1e-7)
            return log_loss(y, p)

        def obj(params):
            logw, a, b = params[:len(zs)], params[-2], params[-1]
            w = np.exp(logw - np.max(logw))
            w = w / w.sum()
            return fit(w, a, b) + cfg.opt_l2 * float(np.sum(np.square(w)))

        from scipy.optimize import minimize as scipy_minimize
        best = None
        for init in (np.zeros(len(zs) + 2),
                     np.zeros(len(zs) + 2) + np.array([0.0] * len(zs) + [1.0, 0.0]),
                     np.array([-0.5] * len(zs) + [1.2, -0.2])):
            res = scipy_minimize(obj, init, method="Nelder-Mead",
                                 options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-6})
            if best is None or res.fun < best.fun:
                best = res
        logw = best.x[:len(zs)]
        weights = np.exp(logw - np.max(logw))
        weights = weights / weights.sum()
        a, b = float(best.x[-2]), float(best.x[-1])
        avg_w = zs_mat @ weights
        p_w = np.clip(1 / (1 + np.exp(-(a * avg_w + b))), 1e-7, 1 - 1e-7)
        print("OPT-WEIGHTS blend: " + " ".join(
            f"{c['name']}={w:.3f}" for c, w in zip(configs, weights)))
        print(f"  opt-weighted calibrated: AUC {roc_auc_score(y, avg_w):.4f} "
              f"logloss {log_loss(y, p_w):.4f}")
        avg = avg_w
        p = p_w

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for conf, d in zip(configs, dirs):
        for fold in sorted(d.glob("fold_*.pt")):
            shutil.copy(fold, out / f"{conf['name']}_{fold.name}")
    np.savez(out / "members_oof.npz",
             oofs_all=np.stack(oofs_all, axis=0),
             oofs_selected=np.stack(oofs, axis=0),
             names_all=[c["name"] for c in configs_all],
             names_selected=[c["name"] for c in configs],
             selected=np.array(chosen, dtype=int),
             y=y)
    meta = {"a": a, "b": b,
            "members": [{k: v for k, v in c.items() if k != "name"} | {"name": c["name"]}
                        for c in configs],
            "zmean": [float(o.mean()) for o in oofs],
            "zstd": [float(o.std()) for o in oofs],
            "weights": [float(w) for w in weights],
            "blend_auc": float(roc_auc_score(y, avg)),
            "blend_logloss": float(log_loss(y, p))}
    with open(out / "calib.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Ensemble written to {out} ({len(list(out.glob('*_fold_*.pt')))} models)")


if __name__ == "__main__":
    main()
