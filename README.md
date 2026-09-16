# DaT Parkinson's Challenge — Bayesian Ensemble

Top-28 solution for the DrivenData **DaT Parkinson's Challenge** (Sept 2026),
ranking #28 / hundreds of teams on private test log loss (**0.2472**).

**Key result:** a 13-member deep learning ensemble over 2D MIP-CNN and 3D
volume-CNN backbones (ResNet18/50, DenseNet, EfficientNet), with anatomy-aligned
preprocessing and **Platt-scaled z-score model averaging**, driven entirely by
log loss.

---

## Performance (private test)

| Metric | Value |
|---|---|
| Private test log loss | **0.2472** |
| Private leaderboard | **#28** |
| OOF (single-fold honest) log loss | 0.2267 |
| OOF ROC AUC | 0.9684 |
| Blend members | 13 |
| Fold models | 65 (5 per member) |

Calibration was the difference-maker: an early flip-TTA dimension mismatch in
the OOF calibration path (flipping the channel / height axes instead of the
spatial axes) produced z-score statistics on a distribution the runtime never
generates, pushing private log loss to 0.34. Fixing OOF to use the exact
runtime flip dims recovered ~0.095 of log loss.

## How it works

```
nifti ──► anatomy-aligned MIPs / 3D cubes (per-member variant)
              │
              ▼
   ┌─ 2D MIP-CNN (ResNet18/50, DenseNet, EfficientNet) ─┐
   └─ 3D volume CNN (ResNet18) ──────────────────────────┘
              │  flip-TTA logits, 5-fold average per member
              ▼
   per-member z-score:  (x - mean) / std   ◄── OOF-computed stats
              │
              ▼
   average across members  →  Platt calibration  →  P(Parkinsonian)
```

- **Anatomy-aligned preprocessing** (`src/align.py`): instead of center-cropping
  the field of view (which varies from 135 mm to 630 mm across scanners), every
  scan is centered on the brain center of mass and cropped to a fixed *physical*
  cube — a cheap registration that lands the striatum in a consistent location.
  Produces aligned 3D volumes and MIP projections at native in-plane resolution.
- **Per-member variants:** brain200 MIP (full brain), striatum-tight 120/160 mm
  MIP crops, and aligned 3D volume grids — each member sees a different crop /
  representation, so the ensemble decorrelates.
- **Diversity:** multiple backbones + seeds + EMA vs fresh training curves.
- **Calibration** is the core trick: member logits are z-normalized with
  *honest single-fold OOF* statistics, averaged, then passed through a Platt
  fit on the same OOF distributions — making the ensemble output a proper
  probability (log-loss optimal).

## Repository layout

```
src/                    core training/inference library
  align.py              anatomy-aligned (center-of-mass) preprocessing
  dataset.py            Dataset + augmentation
  model.py              build_model(backbone, in_channels) — 2D/3D CNN factory
  preprocess.py         NIfTI loading, resample, percentile intensity norm
  ensemble.py           fold averaging + z-score blending + Platt calibration
  features.py           OOF/calibration feature assembly
train/                  training entry points (one script per run)
tools/                  OOF computation, blend search, calibration, zip build,
                        end-to-end verification
submission_src/         the exact code packaged into each submission
  main.py               runtime inference (loads model/ + calib.json)
make_submission.py      deterministic submission.zip packager
test_submission.py      local submit-and-evaluate harness
smoke_test.py           cheap 1-model smoke test
```

## Reproducing

1. **Train** members:
   `./train/train_all.sh` (uses `src/train_resnet.py`,
   5-fold, rotation augmentation, per-backbone batch sizes).

2. **Build runtime kernel:**
   `python tools_build_v9_zip.py` packages `main.py` + fold weights +
   `model/calib.json` into `submission.zip`.

3. **Verify end-to-end** (the step that caught the v7 bug):
   `python test_submission.py` runs the *packaged* `main.py` on local scans and
   checks output scale against OOF (expected Pearson r ≈ 0.98, std ratio ≈ 1.0).
   `python tools/e2e_scale_check.py` quantifies the correlation/scale match.

4. **Calibrate:** `tools/tools_v9_calib.py` re-derives z-stats + Platt on honest
   OOF; `tools/tools_v10_calib.py` builds the final 13-member blend.

## Lessons learned

- **Never calibrate on a distribution the runtime can't produce.** The v7
  failure (0.34 log loss vs 0.23 OOF) came from a one-line flip-axis typo in
  *OOF-only* code — the calibration stats were computed on wrong-shape TTA
  outputs, so real test logits were ~1.5× too extreme for the Platt fit.
- **Verify the packaged artifact, not the training code.** `test_submission.py`
  must decode and run `submission.zip` exactly as the competition runtime does;
  a training-time rerun would have passed while the real submission failed.
- **Ensemble for log loss**: member z-score averaging + Platt beats a single
  strong model even when each member's raw logit scale differs.

## License

Code released for reference / educational use under the **MIT License**.

**Data and trained weights are not included** (competition data is
redistribution-restricted; fold weights are ~3.9 GB and only available on
request).