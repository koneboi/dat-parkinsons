"""Clinical semi-quantitative features from DaT SPECT volumes.

These mirror the features nuclear medicine physicians use: left/right striatal
uptake (caudate/putamen region), asymmetry, and striatal-to-background binding
ratios. Computed on anatomy-aligned volumes so hemispheres are comparable.
"""

import numpy as np
from scipy.ndimage import center_of_mass, zoom


def striatal_band(vol, frac=0.5):
    """Axial slices where bright (striatal) voxels concentrate."""
    vmax = float(vol.max())
    bright = vol >= frac * vmax
    counts = bright.sum(axis=(0, 1))
    if counts.max() == 0:
        return slice(0, vol.shape[2])
    peak = int(np.argmax(counts))
    # window around the peak that still has >40% of peak count
    half = 0
    for step in (1, 2, 3, 4):
        ok = counts[max(peak - step, 0):peak + step + 1] > 0.4 * counts[peak]
        if ok.any():
            half = step
        else:
            break
    lo = max(peak - half, 0)
    hi = min(peak + half + 1, vol.shape[2])
    return slice(lo, hi)


def hemisphere_split(vol, vmax=None):
    """Split an axial-aligned volume into left/right hemispheres at mid-sagittal."""
    ys, xs, _ = np.nonzero(vol > (0.1 * vmax if vmax else vol.max()))
    if len(xs) == 0:
        return slice(0, vol.shape[1] // 2), slice(vol.shape[1] // 2, vol.shape[1])
    mid = int(round(xs.mean()))
    return slice(0, mid), slice(mid + 1, vol.shape[1])


def extract_features(vol):
    """vol: aligned (64,64,64) float32 (any intensity scale). Returns ndarray."""
    vmax = float(vol.max())
    if vmax <= 0:
        return np.zeros(20, dtype=np.float32)
    vol = vol.astype(np.float64)
    band = striatal_band(vol)
    vb = vol[:, :, band]

    # background = brain tissue below striatal threshold
    thresh_low = np.percentile(vol[vol > 0.1 * vmax], 20)
    thresh_high = np.percentile(vol[vol > 0.1 * vmax], 60)
    bg_mask = (vol > 0.1 * vmax) & (vol <= 0.5 * vmax)
    background = vol[bg_mask].mean() if bg_mask.any() else 0.0

    feats = []
    for frac in (0.4, 0.5, 0.6, 0.7):
        left, right = hemisphere_split(vb >= frac * vmax, vmax)
        L = vb[:, left, :]
        R = vb[:, right, :]
        sL = L.sum()
        sR = R.sum()
        feats += [
            sL, sR,
            (sL - sR) / (sL + sR + 1e-9),          # asymmetry index
            sL / (sR + 1e-9),                       # L/R ratio
            sL / (sL + sR + 1e-9),                  # L fraction
            (sL + sR) / (vol > 0.1 * vmax).sum(),  # striatal fraction of brain
        ]
    # binding-ratio style features (counts vs background)
    for frac in (0.4, 0.5, 0.6):
        left, right = hemisphere_split(vb >= frac * vmax, vmax)
        sL = vb[:, left, :].sum()
        sR = vb[:, right, :].sum()
        brL = sL / (background * max((vb[:, left, :] > 0).sum(), 1) + 1e-9)
        brR = sR / (background * max((vb[:, right, :] > 0).sum(), 1) + 1e-9)
        feats += [brL, brR, (brL - brR) / (brL + brR + 1e-9)]
    # overall max and band height
    feats += [vmax, float(band.stop - band.start)]
    return np.asarray(feats, dtype=np.float32)


def load_aligned(uid, nifti_dir, cache_dir):
    import hashlib
    from pathlib import Path

    path = Path(nifti_dir) / f"{uid}.nii.gz"
    st = path.stat()
    key = hashlib.md5(f"{path}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:12]
    cache_file = Path(cache_dir) / f"{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    from .align import aligned_grid, read_nifti

    vol, spacing = read_nifti(path)
    arr = aligned_grid(vol, spacing)
    np.save(cache_file, arr)
    return arr
