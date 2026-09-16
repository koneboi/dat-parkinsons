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
- **Registry:** www.drivendata.org/competitions/311/dat-parkinsons-challenge/

---

## 1. Problem

Parkinsonian syndromes affect millions of people worldwide. **Dopamine
transporter (DaT) imaging** helps distinguish neurodegenerative parkinsonian
syndromes from other conditions, but interpreting scans reliably requires
specialised expertise that is scarce. In France alone >20,000 DaT scans are
performed each year; ~1 in 5 is hard to interpret, especially early-stage or
atypical presentations. A model that reliably distinguishes **normal** from
**abnormal (pathologic)** DaT scans could expand access to expert-level
diagnostic support and accelerate research into neurodegenerative disorders.

The task is **binary classification**: `P(is_pathologic)` per DaT SPECT scan.
The evaluation metric is **log loss**, which rewards well-calibrated
probabilities — not just rank discrimination. This shapes the whole solution:
calibration is not an afterthought, it *is* the metric.

Data: a unique **multicentre dataset** of DaT scans from **10 French hospital
centres**, annotated by expert nuclear-medicine readers. Each scan is a single
**3D NIfTI reconstruction** of 16-bit intensity voxels. Imaging parameters vary
across centres: volume dimensions (e.g. 142×142×112 or 128×128×128) and voxel
spacing (2.46–3.895 mm isotropic) differ scan-to-scan — the pipeline must not
assume a fixed input shape. Participant institutions include CHU Toulouse,
CHU Nancy, CHU Tours, CHU Brest, CHU Grenoble-Alpes, Hospices Civils de Lyon,
CHU de la Timone Marseille, Centre Jean Perrin, Centre Henri Becquerel and
Institut Godinot. Reads were performed by a panel of expert physicians and
physicians-in-training led by members of the SFMN.

**Prize context.** €25,000 total (1st €12,500 / 2nd €7,500 / 3rd €5,000).
Winners are invited to an in-person ceremony hosted by SFMN and France's Health
Data Hub, and top teams may be contacted about a further scientific
collaboration on age and sex prediction from DaT scans. Winning solutions are
released as open source. This competition was closed to public disclosure of the
winning pipeline during the competition; this write-up is published after the
deadline for reference and learning.

---

## 2. Final results

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
| Runtime E2E Pearson *r* (vs OOF) | 0.981 |
| Runtime E2E probs std ratio | 1.016 |

### 2.1 What is calibrated here, and why it matters

The **single most important** result of this project is that log-loss
optimisation is a *calibration* problem, not an *accuracy* problem. Every
submission is scored as
`LL = −1/N Σ [y·log(p) + (1−y)·log(1−p)]`,
which penalises confident-but-wrong predictions **and** rewards probabilities
that match the true long-run base rate. A model with AUROC 0.97 can score
worse than a model with AUROC 0.90 if its probabilities are poorly calibrated.
This is why the solution is structured around **honest out-of-fold statistics,
per-member z-score normalisation, and a single global Platt recalibration** (see
sections 5 and 6).

---

## 3. Preprocessing — anatomy-aligned DaT scans

The single most impactful preprocessing insight: the original pipeline
center-cropped the **field of view**, but FOV varies **135–630 mm** across
scanners, so the brain (and the striatum within it) lands in inconsistent
locations. Instead, every scan is:

1. **read from NIfTI** (`src/preprocess.py`) — orientation-normalised, 16-bit
   intensity array + voxel spacing extracted from the NIfTI header;
2. **intensity-normalised** per volume: percentile clipping + min-max scaling to
   [0, 1] to reduce scanner-to-scanner intensity differences;
3. **anatomy-aligned** (`src/align.py`): centred on the **brain centre of mass
   (COM)** and cropped to a fixed *physical* cube — a cheap registration that
   lands the striatum in a consistent place regardless of scanner;
4. **projected / resampled** to one of:
   - **MIP** (maximum-intensity projection) of the aligned brain at **native
     in-plane resolution**, target **224×224** (`MIP_TARGET`), fixed 200 mm
     physical cube (`BRAIN_CUBE_MM = 200.0`);
   - **tight striatum MIP** crop at **120 mm** or **160 mm** physical cube
     (`VARIANT_CUBE = {"str160": (160, "striatum"), "str120": (120, "striatum"),
     "brain160": (160, "brain")}`) — these capture the dopamine-transporter
     signal in the striatum, directly where abnormal uptake shows;
   - **3D volume** resampled to a fixed isotropic grid (`GRID_MM = 3.0` → ~64³)
     for the 3D CNN;
   - **AIP stacking** — average-intensity projection concatenated channel-wise
     with the MIP (`stack="aip"`, 6-channel input).

**Why it works.** The striatum is small and sits centrally; a scanner-agnostic,
brain-COM-aligned crop turns a wildly variable multicentre problem into a
pseudo-registered one. Scanning-plane variation is absorbed by the crop, and the
intensity normalisation handles raw-count differences between centres.

**Equivalence guarantee.** `normalize → resize` and `resize → normalize` commute
for a per-channel constant affine + bilinear resize. We verified this formally
and empirically, so the *training* pipeline and the *runtime* `main.py` produce
bit-comparable inputs — a silent preprocessing drift (a classic source of
compile-time-good / runtime-bad submissions) is structurally excluded.

---

## 4. Models & training

### Backbones (`src/model.py`)
- **2D families** on MIP/AIP input:
  - ResNet-18 / **ResNet-50** (ImageNet backbone, `in_channels=3`, or 6 when
    stacking MIP+AIP);
  - **DenseNet-121**;
  - **EfficientNet-B0** (added late as a diversity-expanding backbone).
- **3D family**: custom small **3D ResNet** (`DATScan3DCNN`, 32→64→128 base
  channels, ~dropout 0.3) on the aligned 3D volume grid.

### Training (`src/train_resnet.py`, `train/train_all*.sh`)
- **5-fold stratified CV** per member; honest **single-fold OOF** scores per
  sample (each sample scored only by the fold that never saw it). This is the
  distribution used for all calibration.
- Defaults: **60 epochs**, Adam `lr=1e-3`, weight decay `1e-4`, BCE loss,
  **random rotation up to 15–25°** augmentation, batch sizes chosen per backbone
  (8–32) with **gradient accumulation** for the bigger models on a low-VRAM
  (4 GB) GPU.
- **Variants** used per member: **EMA** (exponential moving average of weights,
  decay 0.99), different **seeds** (42 / 123 / 999 / 1234), different **crop
  variants** (brain 200 / striatum 120 / striatum 160), **MIP vs AIP** stacks,
  **TTA on/off**.
- Hardware: consumer RTX-class GPU, 4 GB VRAM, CPU-parallel preprocessing +
  mixed-precision (AMP) training.

---

## 5. Ensemble — z-score blend + Platt calibration (`src/ensemble.py`)

Because the metric is log loss, the ensemble is designed to produce **proper
probabilities**, not just good AUC:

1. **Per member:** run inference with **flip-TTA** — spatial flips along
   `dims=[3]` and `dims=[2]` of a `(B, 3, H, W)` tensor (the *only* correct
   spatial flips for a channel-last tensor; see §6), averaged with the
   non-flipped logit.
2. **Fold-average:** each member aggregates its 5 fold models' logits into one
   `member_logit` per scan.
3. **Z-score normalise** each member with its **honest single-fold OOF
   mean/std**, computed on the *exact runtime arithmetic*:
   `z_m = (x_m − μ_m) / σ_m`. This puts all 13 members on a common scale.
4. **Blend:** average the per-member z-scores → `blend_z`.
5. **Platt:** `P(abnormal) = σ(a · blend_z + b)`, with `a`, `b` fit by
   logistic regression on the OOF `blend_z` vs the labels (`a = 4.6564`,
   `b = 1.2699`).

Optional greedy **forward selection** on rank-AUC is available in
`src/ensemble.py` and was used during development to prune candidate members;
the final 13 were kept because each one added log-loss value through
decorrelation.

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

Member #13's inclusion improved the *blend* OOF log loss 0.2269 → 0.2267
(−0.0002) through diversity, even though its single-model OOF is weaker —
the ensemble optimum is *not* the max-AUC-member list. Per-member z-stats
(μ, σ) used at runtime are embedded in `model/calib.json`.

---

## 6. The calibration bug that cost ~0.095 log loss (post-mortem)

Early submissions (v6_fixed 0.3052, v7/v8 0.3427) did far worse than OOF
promised (0.2269). The test set was **not** harder — top leaderboard entries
were at ~0.215–0.226. A deterministic code bug was the cause:

```
runtime (main.py)  TTA flips dims = [3], [2]  → spatial W, H   → logit std ≈ 4.82
OOF calibration    TTA flips dims = [2], [1]  → H + CHANNEL    → logit std ≈ 3.14
```

Z-scores were computed with the OOF σ (3.14) but at runtime the actual logits
had σ ≈ 4.82 — so runtime z-scores were ~1.53× larger than the Platt fit
expected, making predictions **severely overconfident** and inflating log loss.

**Fix:** recompute OOF (and thus every μ, σ and the Platt fit) using the *exact*
runtime flip dims `[3],[2]`. This recovered ~0.095 of log loss, and is the
reason v10 scored 0.2472 rather than ~0.34.

**Guaranteeing no recurrence** — the verification gauntlet that caught it:
1. **Runtime E2E run** — execute the *packaged* `main.py` on scans and compare
   against OOF: `Pearson r = 0.981`, `Spearman ρ = 0.992`, probs std ratio
   `1.016`. v7 would have shown r ≈ 0.5–0.7 and std ratio ≈ 1.5.
2. **Calibration reconstruction** — re-derive z-stats + Platt from OOF and check
   they reproduce the stored `calib.json` numbers exactly.
3. **Member-config audit** — every member's preproc/proj/stack/size/backbone in
   `calib.json` matched its own training `calib.json`.
4. **Preprocessing equivalence** — training and runtime `align.py` differ only
   in loader/caching wrappers; core `aligned_mips` / `crop_physical` identical.

---

## 7. Runtime & submission (`submission_src/main.py`)

The submitted artifact is a zip run in the DrivenData container (Python 3.12,
1×A100 80 GB, 24 vCPU, 220 GB RAM, no network, 3 h limit). Each submission is
self-contained — the platform provides no model registry, so **all 65 fold
weights + code + calibration travel inside the zip**:

```
submission.zip
├── main.py                      # entrypoint (required at zip root)
├── src/{__init__,align,model,preprocess}.py
└── model/{<member>_fold_<k>.pt, calib.json}   # 65 fold models + calibration
```

`main.py` reads scans + `submission_format.csv` from `/code_execution/data`,
runs per-member inference (CPU-parallel preprocessing, GPU batched inference,
flip-TTA), applies the z-score blend + Platt, and writes `submission.csv` with
`uid,is_pathologic`. `make_submission.py` builds the zip deterministically;
`test_submission.py` and `smoke_test.py` validate locally (the smoke-test
container mirrors the real runtime on a small sample, no submission spent).

---

## 8. The full development journey (all experiments)

Development ran in **waves**, each driven by a hypothesis and terminated by a
measurement — this is the actual log of what was tried, kept, or dropped.

### Phase 0 — baseline (2026-08-12)
- **Small 3D ResNet @ 64³, 30 epochs, 5-fold.** CV log loss **0.646**, AUC
  **0.669**; OOF calibrated log loss 0.643; 20-scan smoke test 0.547/0.824.
- *Baseline reference*: predicting the base rate (0.55) gives LL ≈ 0.685. The
  first real model added ~0.04 LL vs a constant prediction — a very weak
  starting point that motivated the anatomy-aligned rewrite.

### Phase 1 — anatomy-aligned MIP rewrite
- Switched from FOV-crop 3D to **brain-COM-aligned 2D MIP CNNs**. This is where
  the project jumped from ~0.64 LL / 0.67 AUC to ~0.95 AUC — the alignment
  insight was worth more than any model change.
- Registered alignments tested and **rejected** after measurement: striatum
  roll refinement, template FFT, SimpleITK rigid registration — all worse than
  plain COM alignment (`src/proto_align*.py`, `src/proto_sitk.py`). Simple is
  better when the preprocessing must generalise across 10 scanners.

### Phase 2 — Wave 1 backbones (rot=15°, 60 ep, seed 42)
| weights dir | backbone | CV AUC |
|---|---|---|
| weights_resnet50 | resnet50 | **0.9530** |
| weights_densenet | densenet121 | 0.9526 |
| weights_resnet18_rot | resnet18 + EMA | 0.9465 |
| weights_resnet18_s123 | resnet18 s123 | 0.9461 |
| weights_resnet34 | resnet34 | 0.9453 |
| weights_effb0 | efficientnet_b0 | 0.9412 |

Older single models: resnet 0.9393, resnet224 0.9393, resnet_s7 0.9446.

### Phase 3 — tight-crop striatum study
Fold-0 no-aug controlled experiments (same split):
| variant | AUC |
|---|---|
| brain200 (baseline) | 0.9020 |
| str160 | 0.9147 |
| **str120** | **0.9228** (+0.021) |

str120 tight crop was the fold-0 winner, but the full 5-fold production pilot
(str120_pilot, rot15/60ep) got **0.9450** vs brain200 resnet18 s42 EMA
**0.9465** — the gain did **not** reproduce under strong augmentation.
Decision: keep brain200 as default; use **str120 only for ensemble diversity**
(str120_pilot became the second greedy pick).

### Phase 4 — Wave 3 (EMA and seeds)
| weights dir | config | CV AUC |
|---|---|---|
| weights_r50_ema | resnet50 s42 + EMA | **0.9547** |
| weights_dense_ema | densenet121 s42 + EMA | **0.9549** |
| weights_r18_s999 | resnet18 s999 rot25 | 0.9509 |
| weights_r18_224 | resnet18 @224 | 0.9496 |
| weights_effb1 | efficientnet_b1 | 0.9384 |
| weights_str120_r50 | resnet50 str120 | 0.9386 |
| weights_r50_224 | resnet50 @224 | 0.9000 |

Takeaways: **EMA reliably helps** (r50 0.9530→0.9547, dense 0.9526→0.9549);
resnet18@224 helps (0.9496 vs 0.9461); effb1 is worse than effb0; str120 hurts
resnet50 (0.9386); r50@224 badly overfits (0.9000).

### Phase 5 — rich-label experiments (Wave 4)
| weights dir | config | CV AUC |
|---|---|---|
| weights_dense_mix | densenet121 + mixup 0.2 | 0.9528 |
| weights_r50_mix | resnet50 + mixup 0.2 | 0.9483 |
| (r50_smooth | label smoothing | not better) |
| (vol3d | 3D ResNet | kept for diversity) |

Mixup helped densenet but hurt resnet50 → not used in final blend (except the
3D member). Other tested-and-rejected: label smoothing, OLS+SAM regularisation,
EfficientNet-B1, large image sizes at 224.

### Phase 6 — ensemble assembly (the key lesson)
- First z-blend of 3 members gave 0.9487 / LL 0.285.
- Greedy forward selection on OOF log loss expanded the blend to 12 → 13.
- **weights_effb0_s1234 (EfficientNet-B0)** was the final diversity pick: single
  OOF LL 0.3070, yet improved the blend 0.2269 → 0.2267.
- v9 (12 members) and v10 (13 members) are the two verified, packaged
  submissions; v10 was the final.

### Phase 7 — the calibration fix (v7 → v10)
- v6_fixed scored 0.3052, v7/v8 0.3427 — the OOF/runtime flip-dim mismatch (§6).
- Diagnostic: E2E runtime-vs-OOF scale check. Found Pearson ≈0.5 and std ratio
  ≈1.5 expected for the buggy path, vs r=0.981 and ratio 1.016 after the fix.
- v9/v10 recompute OOF with the exact runtime flips, producing z-stats that
  match runtime logit distribution. Final private test: **0.2472**.

---

## 9. Expertise demonstrated

- **Medical-imaging ML**: 3D NIfTI handling, cross-scanner generalisation with
  a scanner-agnostic anatomical alignment, MIP/AIP projections, SPECT-specific
  anatomy (striatum-targeted crops).
- **Competition engineering**: 5-fold honest OOF, code-execution runtime
  packaging, local E2E container-equivalent testing, deterministic zip builds,
  smoke-test discipline (no wasted submissions), deadline risk management
  (verified primary + fallback artefacts).
- **Probabilistic modelling**: Platt calibration, per-member z-score
  standardisation, log-loss-optimal ensemble design, calibration-vs-single-model
  trade-off analysis.
- **Pragmatic experimentation**: a documented hypothesis→measure→decide loop
  including ~20 trained variants, rejected-registration audit
  (COM > FFT/template/SITK), and diversity-aware member selection.
- **Diagnostic rigour**: the bug was found by testing the *packaged artifact*
  against the OOF distribution, not by reading code — the data told the story
  (correlation and std-ratio breaks), then the code confirmed the dims.

---

## 10. Reproduction

```bash
# 1. Train members (5-fold, per-backbone batch sizes)
./train/train_all.sh            # wave 1: base backbones
./train/train_all2.sh           # wave 2: residual runs
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

## 11. Lessons learned

- **Never calibrate on a distribution the runtime can't produce.** The v7/v8
  disaster was a one-line flip-axis typo in OOF-only code. Log loss lives or
  dies on calibration consistency, not on raw accuracy.
- **Verify the packaged artifact, not the training code.** Only running the
  exact `main.py` from the zip on real data — and checking scale vs OOF
  (Pearson + std ratio) — could have caught the bug.
- **Optimise directly for the metric.** Here that meant honest OOF, log-loss
  blend selection, and Platt calibration — all on the same single-fold
  distribution that test-time inference actually produces.
- **Diversity beats strength in ensembles for log loss.** The weakest single
  member (EfficientNet-B0, OOF LL 0.3070) still improved the final blend
  (0.2267) via decorrelation.
- **Register cheaply, generalise widely.** FOV-crop fails across scanners; a
  physical-cube COM alignment beats heavier image registration on 10-centre
  data measured end-to-end.
- **Budget your submissions.** Code-execution platforms charge expensive
  submissions; a local E2E harness plus a free smoke test is the difference
  between "works locally" and "works in the container".

## 12. Credit

Annotated imaging data and expert labels from 10 French hospital centres,
assembled by the French Society of Nuclear Medicine (SFMN) with the Health Data
Hub and GaelO, supported by the France 2030 plan. Competition hosted by
DrivenData. This repository contains the methodology, experiments log and
verified reproduction code.

## License

MIT License. **Competition data and 3.9 GB of trained weights are not
included** (redistribution-restricted data; weights available on request).