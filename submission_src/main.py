"""Inference code for the DaT Parkinson's Challenge runtime.

Loads the fine-tuned ensemble fold models (multiple backbones, possibly 2D MIP
CNNs and 3D volume CNNs) + Platt calibration from `model/`, runs the same
anatomy-aligned preprocessing as training (per-member variant: brain200 MIP,
tight-crop MIP, or aligned volume), averages flip-TTA logits per member,
z-score-normalizes each member's contribution, averages across members and
calibrates. Writes `submission.csv`.

Preprocessing is parallelized across CPUs; model inference runs on the GPU in
batches with flip TTA.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.align import (aligned_grid, aligned_mips, aligned_mips_centered,
                       normalize_volume, read_nifti)
from src.model import DATScan3DCNN, build_model

DATA_ROOT = Path(os.environ.get("DAT_DATA_ROOT", "/code_execution/data"))
NIFTI_DIR = DATA_ROOT / "niftis"
SUBMISSION_FORMAT_PATH = DATA_ROOT / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")
MODEL_DIR = Path(os.environ.get("DAT_MODEL_DIR", Path(__file__).parent.resolve() / "model"))
BATCH_SIZE = 64
N_PREPROCESS_WORKERS = max(1, min(16, os.cpu_count() or 1))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VARIANT_CUBE = {"str160": (160, "striatum"), "str120": (120, "striatum"),
                "brain160": (160, "brain")}


def load_ensemble(model_dir):
    """Return (members, calib). Each member: name, backbone, size, preproc, folds."""
    with open(model_dir / "calib.json") as f:
        calib = json.load(f)
    members_raw = calib["members"]
    members = []
    for m in members_raw:
        if isinstance(m, str):
            name = m
            cfg_m = {}
        else:
            name = m["name"]
            cfg_m = m
        folds = sorted(model_dir.glob(f"{name}_fold_*.pt"))
        if not folds:
            continue
        members.append({
            "name": name,
            "backbone": cfg_m.get("backbone", "resnet18"),
            "size": int(cfg_m.get("size", 128)),
            "preproc": cfg_m.get("preproc", "brain200"),
            "proj": cfg_m.get("proj", "mip"),
            "stack": cfg_m.get("stack"),
            "tta": bool(cfg_m.get("tta", True)),
            "folds": folds,
        })
    calib["_zmean"] = calib.get("zmean", [0.0] * len(members))
    calib["_zstd"] = calib.get("zstd", [1.0] * len(members))
    calib["_weights"] = calib.get("weights", [1.0] * len(members))
    return members, calib


def preprocess_one(args):
    uid, combos = args
    vol, spacing = read_nifti(NIFTI_DIR / f"{uid}.nii.gz")
    out = {}
    for v, proj in combos:
        key = f"{v}:{proj}"
        if v == "vol3d":
            out[key] = normalize_volume(aligned_grid(vol, spacing))  # (64,64,64)
        elif v == "brain200":
            out[key] = aligned_mips(vol, spacing, proj=proj)  # (3,224,224)
        else:
            cube_mm, center = VARIANT_CUBE[v]
            out[key] = aligned_mips_centered(vol, spacing, cube_mm=cube_mm,
                                             center=center, proj=proj)
    return uid, out


@torch.no_grad()
def member_logits(model, x, device, is_vol3d=False, tta_on=True):
    """Flip-TTA logits (batch-major). x is (B,3,224,224) or (B,1,64,64,64)."""
    if tta_on:
        if is_vol3d:
            variants = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]
        else:
            variants = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]),
                        torch.flip(torch.flip(x, dims=[2]), dims=[3])]
    else:
        variants = [x]
    n = len(x)
    out = torch.zeros(n, device=device)
    for v in variants:
        for i in range(0, n, BATCH_SIZE):
            chunk = v[i:i + BATCH_SIZE].to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out[i:i + BATCH_SIZE] += model(chunk).squeeze(-1)
    return (out / len(variants)).float().cpu().numpy()


def _build_member_tensor(preproc_map, ids, member):
    """Build one member's input tensor for the given uid slice (memory-chunked)."""
    is_vol3d = member["backbone"] == "vol3d"
    variant = member.get("preproc", "brain200")
    proj = member.get("proj", "mip")
    stack = member.get("stack")
    key = "vol3d:mip" if is_vol3d else f"{variant}:{proj}"
    if is_vol3d:
        xs = np.stack([normalize_volume(preproc_map[u][key]) for u in ids])[:, None, ...]
    else:
        xs = np.stack([preproc_map[u][key] for u in ids])
        if stack == "aip":
            aip_key = f"{variant}:aip"
            xs = np.concatenate(
                [xs, np.stack([preproc_map[u][aip_key] for u in ids])], axis=1)
    x = torch.from_numpy(xs)
    if not is_vol3d:
        nc = x.shape[1]
        mean = torch.from_numpy(MEAN).view(1, 3, 1, 1).repeat(1, nc // 3, 1, 1)
        std = torch.from_numpy(STD).view(1, 3, 1, 1).repeat(1, nc // 3, 1, 1)
        x = (x - mean) / std
        size = member["size"]
        if size != 224:
            x = torch.nn.functional.interpolate(x, size=size, mode="bilinear",
                                                align_corners=False)
    return x, is_vol3d


@torch.no_grad()
def predict_probs(members, calib, preproc_map, device):
    """Return calibrated probabilities; memory-chunked to bound RAM usage.

    Processes scans and each member in chunks so we never hold (N, C, H, W)
    tensors for the whole set at once, avoiding OOM on low-RAM runtimes.
    """
    uids = list(preproc_map.keys())
    n = len(uids)
    weights = calib["_weights"]
    wsum = sum(weights) + 1e-9
    total = np.zeros(n, dtype=np.float64)
    # accumulate raw per-member logit sums in a RAM-light persistent accumulator
    member_sum = np.zeros(n, dtype=np.float64)
    chunk_t = max(64, BATCH_SIZE)
    for mi, member in enumerate(members):
        zmean, zstd = calib["_zmean"][mi], calib["_zstd"][mi]
        w = weights[mi]
        nfolds = len(member["folds"])
        is_vol3d = member["backbone"] == "vol3d"
        member_sum[:] = 0.0
        for path in member["folds"]:
            if is_vol3d:
                model = DATScan3DCNN(base_channels=32).to(device)
            else:
                model = build_model(backbone=member["backbone"],
                                    in_channels=3 if not member.get("stack") else 6).to(device)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            for start in range(0, n, chunk_t):
                stop = min(start + chunk_t, n)
                ids = uids[start:stop]
                x, _ = _build_member_tensor(preproc_map, ids, member)
                ch = member_logits(model, x, device, is_vol3d=is_vol3d,
                                   tta_on=member.get("tta", True))
                member_sum[start:stop] += ch
                del x, ch
            del model
        member_avg = member_sum / nfolds
        total += w * (member_avg - zmean) / (zstd + 1e-9)
        torch.cuda.empty_cache()
    mean_logit = total / wsum
    p = 1.0 / (1.0 + np.exp(-(calib["a"] * mean_logit + calib["b"])))
    return np.clip(p, 1e-7, 1 - 1e-7)


def main():
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    uid_col = "uid" if "uid" in submission_format.columns else submission_format.columns[0]
    uids = submission_format[uid_col].tolist()
    logger.info(f"Loaded submission format: {len(uids)} rows")

    members, calib = load_ensemble(MODEL_DIR)
    logger.info(f"Loaded {sum(len(m['folds']) for m in members)} fold models "
                f"across {len(members)} members")

    combos = set()
    for m in members:
        if m["backbone"] == "vol3d":
            combos.add(("vol3d", "mip"))
        else:
            combos.add((m.get("preproc", "brain200"), m.get("proj", "mip")))
            if m.get("stack") == "aip":
                combos.add((m.get("preproc", "brain200"), "aip"))
    logger.info(f"Needed preprocessing variants: {sorted(combos)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    args = [(u, combos) for u in uids]
    with ProcessPoolExecutor(max_workers=N_PREPROCESS_WORKERS) as ex:
        results = list(ex.map(preprocess_one, args))
    preproc_map = {uid: out for uid, out in results}
    logger.info(f"Preprocessed {len(preproc_map)} scans")

    probs = predict_probs(members, calib, preproc_map, device)
    submission_format["is_pathologic"] = probs
    submission_format.to_csv(WRITE_SUBMISSION_PATH, index=False)
    logger.success(f"Wrote predictions to {WRITE_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
