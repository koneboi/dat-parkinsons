"""Load NIfTI DaT scans and normalize them for model input.

Design notes
------------
- Competition images vary in shape and voxel spacing across scanners.
- We resample every volume to a fixed isotropic voxel size (preserving
  physical anatomy), then center-crop/pad to a fixed grid.
- Intensities are per-volume percentile-clipped and scaled to [0, 1] to
  reduce scanner-to-scanner intensity differences.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

TARGET_VOXEL_MM = 4.0
TARGET_SHAPE = (64, 64, 64)
LOWER_PCT = 0.5
UPPER_PCT = 99.5

CACHE_DIR = Path("/tmp/cache_volumes")


def load_volume(nifti_path, clip_percentiles=(LOWER_PCT, UPPER_PCT)):
    """Load a NIfTI volume, resample to isotropic voxels and normalize.

    Returns a float32 array of shape TARGET_SHAPE with values in [0, 1].
    """
    vol, spacing = read_nifti(nifti_path)
    vol = resample_isotropic(vol, spacing)
    vol = crop_or_pad_to_shape(vol, TARGET_SHAPE)
    vol = normalize_intensities(vol, clip_percentiles)
    return vol


def read_nifti(nifti_path):
    """Return (volume float32, voxel spacing in mm)."""
    img = nib.load(str(nifti_path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    return vol, spacing


def resample_isotropic(vol, spacing, target_voxel_mm=TARGET_VOXEL_MM, order=1):
    """Resample so voxels are approximately `target_voxel_mm` cubic."""
    zoom_factors = spacing / target_voxel_mm
    # Clamp so pathological spacings cannot blow up memory.
    zoom_factors = np.clip(zoom_factors, 0.2, 4.0)
    out = zoom(vol, zoom_factors, order=order, mode="nearest")
    return np.asarray(out, dtype=np.float32)


def crop_or_pad_to_shape(vol, shape):
    """Center crop/pad a volume to an exact shape."""
    out = np.zeros(shape, dtype=np.float32)
    in_shape = np.array(vol.shape)
    target = np.array(shape)
    in_slices = []
    out_slices = []
    for d, (in_len, tgt) in enumerate(zip(in_shape, target)):
        if in_len >= tgt:
            start = (in_len - tgt) // 2
            in_slices.append(slice(start, start + tgt))
            out_slices.append(slice(0, tgt))
        else:
            in_slices.append(slice(0, in_len))
            pad = (tgt - in_len) // 2
            out_slices.append(slice(pad, pad + in_len))
    out[tuple(out_slices)] = vol[tuple(in_slices)]
    return out


def normalize_intensities(vol, clip_percentiles=(LOWER_PCT, UPPER_PCT)):
    """Clip to percentiles and scale to [0, 1]."""
    if clip_percentiles is not None:
        lo, hi = np.percentile(vol, clip_percentiles)
        vol = np.clip(vol, lo, hi)
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return (vol - vmin) / (vmax - vmin)


def volume_cache_key(nifti_path):
    """Deterministic cache filename so preprocessing is only done once."""
    import hashlib

    st = Path(nifti_path).stat()
    h = hashlib.md5(f"{nifti_path}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()[:12]
    return h


def load_volume_cached(nifti_path, use_cache=True):
    """Like load_volume but caches result on disk for fast re-runs."""
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{volume_cache_key(nifti_path)}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        vol = load_volume(nifti_path)
        np.save(cache_file, vol)
        return vol
    return load_volume(nifti_path)
