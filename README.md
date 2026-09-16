# DaT Parkinson's Challenge — Bayesian Ensemble

![Competition](https://img.shields.io/badge/DrivenData-DaT%20Parkinson's%20Challenge-blue?style=flat-square)
![Rank](https://img.shields.io/badge/Rank-%2328%2F361-6600cc?style=flat-square)
![Log loss](https://img.shields.io/badge/Test%20log%20loss-0.2472-e11d48?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)

**Top-28** solution for the **DaT Parkinson's Challenge** — a DrivenData
competition hosted by the **French Society of Nuclear Medicine (SFMN)** with the
**Health Data Hub** and **GaelO**, part of the *Health Data Challenges* call for
projects funded by the **France 2030** plan.

- **Competition:** https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/
- **Best score (private test log loss):** **0.2472**
- **Current rank:** **#28 out of 361 participants**
- **Metric:** log loss (error — lower is better); AUROC shown for reference only
- **Page:** https://koneboi.github.io/dat-parkinsons/

---

## 1. Problem

Parkinsonian syndromes affect millions of people worldwide. **Dopamine
transporter (DaT) imaging** helps distinguish neurodegenerative parkinsonian
syndromes from other conditions, but interpreting scans reliably requires
specialised expertise that is scarce. In France alone >20,000 DaT scans are
performed each year; ~1 in 5 is hard to interpret, especially early-stage or
atypical presentations.

The task is **binary classification of a DaT SPECT scan as normal vs. abnormal
(pathologic)**, predicting `P(is_pathologic)` per scan. The metric is **log
loss**, which rewards well-calibrated probabilities — not just rank
discrimination. This shapes the whole solution: calibration is not an
afterthought, it *is* the metric.

Data: a unique **multicentre dataset** of DaT scans from **10 French hospital
centres** (CHU Toulouse, Nancy, Tours, Brest, Grenoble, Lyon, Marseille, and
others), annotated by expert nuclear-medicine readers. Each scan is a single
**3D NIfTI reconstruction** of 16-bit intensities. Imaging parameters vary
across centres: volume dimensions (e.g. 142×142×112 or 128×128×128) and voxel
spacing (2.46–3.895 mm isotropic) differ scan-to-scan — the pipeline must not
assume a fixed input shape.

## 2. Results

| Metric | Value |
|---|---|
| **Private test log loss** | **0.2472** |
| **Final rank** | **#28 / 361 participants** |
| OOF log loss (honest single-fold) | 0.2267 |
| OOF ROC AUC | 0.9684 |
| Training set | 1,362 scans (747 pathologic / 615 normal) |
| Ensemble members | 13 |
| Fold models (total) | 65 (5 per member) |
| Platt slope *a* / intercept *b* | 4.6564 / 1.2699 |

Calibration was the difference-maker: fixing an OOF/runtime flip-dimension
mismatch recovered **~0.095 of log loss** (0.34 → 0.2472). Details in §6.

## 3. Preprocessing — anatomy-aligned DaT scans

The single most impactful preprocessing insight: the original pipeline
center-cropped the **field of view**, but FOV varies **135–630 mm** across
scanners, so the brain (and the striatum within it) lands in inconsistent
locations. Instead, every scan is:

1. read from NIfTI (`src/preprocess.py`) — orientation-normalised, intensity
   `uint16` array + voxel spacing from the header;
2. **intensity-normalised** per volume: percentile clipping + min-max to
   [0, 1] to reduce scanner-to-scanner intensity differences;
3. **anatomy-aligned** (`src/align.py`): centred on the **brain centre of mass**
   and cropped to a fixed *physical* cube — a cheap registration that lands the
   striatum in a consistent place regardless of scanner;
4. rendered to one of two representations:
   - **MIP** (maximum-intensity projection) of the aligned brain at **native
     in-plane resolution**, target **224×224** (`MIP_TARGET`), optionally of a
     tighter **striatum** crop (120 or 160 mm physical cube);
   - **3D volume** resampled to a fixed isotropic grid (3 mm → ~64³) for the 3D
     CNN.

Normalisation pro tip: `normalize → resize` and `resize → normalize` commute for
a per-channel constant affine + bilinear resize, which we exploited to guarantee
training/runtime preprocessing equivalence.

## 4. Models & training

### Backbones (`src/model.py`)
- **2D families** on MIP/AIP input:
  - ResNet-18 / **ResNet-50** (ImageNet backbone, `in_channels=3`, or 6 when
    stacking MIP + AIP channels);
  - **DenseNet-121**;
  - **EfficientNet-B0** (added late as a diversity-expanding backbone).
- **3D family**: custom small **3D ResNet** (`DATScan3DCNN`, ~32 base channels)
  on the aligned 3D volume grid.

### Training (`src/train_resnet.py`, `train/train_all*.sh`)
- **5-fold** stratified CV per member; honest **single-fold OOF** scores per
  sample (each sample scored only by the fold that never saw it).
- Defaults: **60 epochs**, Adam `lr=1e-3`, weight decay `1e-4`, BCE loss,
  **random rotation up to 15–25°** augmentation, batch sizes chosen per backbone
  (8–32) with **gradient accumulation** for the bigger models on a low-VRAM GPU.
- Variants used per member: **EMA** (exponential moving average of weights,
  decay 0.99), different **seeds** (42 / 123 / 999 / 1234), different **crop
  variants** (brain 200 / striatum 120 / striatum 160), **AIP stacking**
  (average-intensity projection concat), and **TTA on/off**.
- Exploration that was trained and **rejected**: rich-label smoothing, mixup,
  OLS+SAM regularisation, EfficientNet-B1 — none improved the OOF blend.

## 5. Ensemble — z-score blend + Platt calibration (`src/ensemble.py`)

Because the metric is log loss, the ensemble is designed to produce **proper
probabilities**, not just good AUC:

1. **Per member:** run inference with **flip-TTA** — spatial flips along
   `dims=[3]` and `dims=[2]` of the (B, 3, H, W) tensor (the *only* correct
   spatial flips for a channel-last tensor; see §6), averaged with the
   non-flipped logit.
2. **Fold-average:** for each member, the 5 fold models' logits are averaged
   into one `member_logit` per scan (runtime behaviour).
3. **Z-score normalise** each member with its **honest single-fold OOF
   mean/std** (computed on the *exact runtime arithmetic*):
   `z_m = (x_m − μ_m) / σ_m`. This puts all 13 members on a common scale.
4. **Blend:** average the per-member z-scores → `blend_z`.
5. **Platt:** `P(abnormal) = σ(a · blend_z + b)`, with `a`, `b` fit by
   logistic regression on the OOF `blend_z` vs the labels. Values:
   `a = 4.6564`, `b = 1.2699`.

This is the heart of the submission: it converts 13 differently-scaled,
differently-correlated logits into one calibrated probability, which is exactly
what log loss rewards.

### Final blend members (calib_v10.json)

| # | Member | Backbone | Input | TTA | OOF LL | OOF AUC |
|---|---|---|---|---|---|---|
| 1 | weights_r50_s7 | ResNet50 | Brain 200 MIP | yes | 0.2589 | 0.9570 |
| 2 | weights_dense_s999 | DenseNet121 | Brain 200 MIP | yes | 0.2528 | 0.9590 |
| 3 | weights_str120_dense | DenseNet121 | Striatum 120 MIP | yes | 0.2827 | 0.9503 |
| 4 | weights_r50_s123 | ResNet50 | Brain 200 MIP | yes | 0.2616 | 0.9559 |
| 5 | weights_r50_str160_128 | ResNet50 | Striatum 160 MIP | yes | 0.2795 | 0.9531 |
| 6 | weights_r50_mipaip | ResNet50 | Brain 200 MIP + AIP (6-ch) | yes | 0.2845 | 0.9484 |
| 7 | weights_r50_ema | ResNet50 | Brain 200 MIP | yes | 0.2578 | 0.9576 |
| 8 | weights_dense_ema | DenseNet121 | Brain 200 MIP | yes | 0.2718 | 0.9537 |
| 9 | weights_str120_r50 | ResNet50 | Striatum 120 MIP | yes | 0.2880 | 0.9481 |
| 10 | weights_r50_s999 | ResNet50 | Brain 200 MIP | **no** | 0.2916 | 0.9460 |
| 11 | weights_dense_str160_128 | DenseNet121 | Striatum 160 MIP | yes | 0.2999 | 0.9457 |
| 12 | weights_r18_s999 | ResNet18 | Brain 200 MIP | yes | 0.2729 | 0.9532 |
| 13 | weights_effb0_s1234 | EfficientNet-B0 | Brain 200 MIP | yes | 0.3070 | 0.9400 |

Member #13's inclusion improved the *blend* OOF log loss 0.2269 → 0.2267 (−0.0002)
through diversity, even though its single-model OOF is weaker. Per-member z-stats
(μ, σ) used at runtime are embedded in `model/calib.json`.

## 6. The calibration bug that cost 0.095 log loss (post-mortem)

Early submissions (v6_fixed 0.3052, v7/v8 0.3427) did far worse than the OOF
promised (0.2269). The test set was **not** harder — top leaderboard entries
were at ~0.215–0.226. A deterministic code bug was the cause:

```
runtime (main.py)  TTA flips dims = [3], [2]  → spatial W, H  → logit std ≈ 4.82
OOF calibration    TTA flips dims = [2], [1]  → H + CHANNEL   → logit std ≈ 3.14
```

Z-scores were computed with the OOF σ (3.14) but at runtime the actual logits
had σ ≈ 4.82 — so runtime z-scores were ~1.53× larger than the Platt fit
expected, making predictions **severely overconfident** → inflated log loss.

**Fix:** recompute OOF (and thus every μ, σ, and the Platt fit) using the
*exact* runtime flip dims `[3],[2]`. This recovered ~0.095 of log loss.

**Guaranteeing no recurrence** (the verification gauntlet that caught this, and
which would have caught the bug before submission):
1. **Runtime E2E run** — execute the *packaged* `main.py` on scans and compare
   against OOF: Pearson `r = 0.981`, Spearman `ρ = 0.992`, std ratio `1.016`.
   v7 would have shown r ≈ 0.5–0.7 and a std ratio ≈ 1.5.
2. **Calibration reconstruction** — re-derive z-stats + Platt from OOF and check
   they reproduce the stored `calib.json` numbers exactly.
3. **Member-config audit** — every member's preproc/proj/stack/size/backbone in
   `calib.json` matched its own training `calib.json`.
4. **Preprocessing equivalence** — training and runtime align.py differ only in
   loader/caching wrappers; core `aligned_mips` / `crop_physical` identical.

## 7. Runtime & submission (`submission_src/main.py`)

The submitted artifact is a zip run in the DrivenData container (Python 3.12,
1×A100 80 GB, 24 vCPU, no network, 3 h limit):

```
submission.zip
├── main.py                      # entrypoint (required at zip root)
├── src/{__init__,align,model,preprocess}.py
└── model/{<member>_fold_<k>.pt, calib.json}   # 65 fold models + calibration
```

`main.py` reads scans + `submission_format.csv` from `/code_execution/data`,
runs per-member inference (CPU-parallel preprocessing, GPU batched inference,
flip-TTA), applies the z-score blend + Platt, and writes `submission.csv` with
`uid, is_pathologic`. `make_submission.py` builds the zip deterministically;
`test_submission.py` and `smoke_test.py` validate locally.

## 8. Reproducing

```bash
# 1. Train members (5-fold, per-backbone batch sizes)
./train/train_all.sh            # wave 1: base backbones
./train/train_all2.sh           # wave 2: 224 res, more seeds
./train/train_all3.sh           # wave 3: EMA, striatum crops

# 2. Compute honest OOF for every candidate (exact runtime flips dims=[3],[2])
python tools/tools_v9_oof.py -o oof_runtime.npz --members weights_*_s7

# 3. Select blend that minimises OOF log loss
python tools/tools_v10_select2.py

# 4. Fit z-stats + Platt, write calib.json
python tools/tools_v10_calib.py

# 5. Package submission.zip
python make_submission.py

# 6. Verify (the gauntlet of §6)
python tools/final_gate.py          # weights load, arch match, calib sanity
python test_submission.py           # E2E packaged run vs OOF scale
python tools/verify_v9_calib.py     # calibration reconstruction
```

## 9. Lessons learned

- **Never calibrate on a distribution the runtime can't produce.** The v7/v8
  disaster was a one-line flip-axis typo in OOF-only code. Log loss lives or
  dies on calibration consistency, not on raw accuracy.
- **Verify the packaged artifact, not the training code.** Only running the
  exact `main.py` from the zip on real data, and checking scale vs OOF
  (Pearson + std ratio), could have caught the bug.
- **Optimise directly for the metric.** Here that meant honest OOF, log-loss
  blend selection, and Platt calibration — all on the same single-fold
  distribution that test-time inference actually produces.
- **Diversity beats strength in ensembles for log loss.** The weakest single
  member (EfficientNet-B0, OOF LL 0.3070) still improved the final blend
  (0.2267) via decorrelation.

## 10. Credit

Annotated imaging data and expert labels from 10 French hospital centres,
assembled by the French Society of Nuclear Medicine (SFMN) with the Health Data
Hub and GaelO. Competition run on DrivenData. This repository contains the
methodology and verified reproduction code.

## License

MIT License. **Competition data and 3.9 GB of trained weights are not
included** (redistribution-restricted data; weights available on request).