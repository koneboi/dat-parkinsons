"""Anatomy-aligned preprocessing for DaT scans.

The original pipeline center-cropped the *field of view*, but the FOV varies
widely across scanners (some are 630 mm, some 135 mm), so the brain and striatum
land in inconsistent locations. Here we center every scan on the brain center of
mass and crop a fixed *physical* cube, which acts as a cheap registration.

Two representations are produced:
  - aligned 3D volume resampled to a fixed grid (for a 3D CNN)
  - MIP projections of the aligned brain at native in-plane resolution (for a 2D CNN)
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass, zoom

BRAIN_CUBE_MM = 200.0
GRID_MM = 3.0  # resampled voxel size for the 3D grid
MIP_TARGET = 224  # MIP projection resolution

CACHE_VOL3D = Path(__file__).resolve().parent.parent / "cache_volumes_aligned"
CACHE_MIPS = Path(__file__).resolve().parent.parent / "cache_mips_aligned224"
CACHE_MIPS_EXP = Path(__file__).resolve().parent.parent / "cache_mips_exp"


def read_nifti(nifti_path):
    img = nib.load(str(nifti_path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    return vol, spacing


def brain_com(vol):
    """Center of mass of the brain, in mm, from a coarse threshold."""
    thresh = float(np.percentile(vol, 50))
    mask = vol > max(thresh, 1e-6 * vol.max())
    if not mask.any():
        mask = vol > 0
    if not mask.any():
        return np.array(vol.shape) / 2.0
    return np.asarray(center_of_mass(mask), dtype=np.float64)


def crop_physical(vol, spacing, center_mm, half_mm):
    """Crop a physical cube (2*half_mm per side) centered at center_mm (in mm).

    Returns voxel-space slices cropped from vol plus padding offsets.
    """
    center_vox = np.array(center_mm) / np.array(spacing)
    shape = np.array(vol.shape, dtype=np.float64)
    half_vox = np.array(half_mm) / np.array(spacing)
    lo = np.floor(center_vox - half_vox).astype(int)
    hi = np.ceil(center_vox + half_vox).astype(int)
    lo = np.clip(lo, 0, shape.astype(int) - 1)
    hi = np.clip(hi, 0, shape.astype(int))
    pad_lo = (center_vox - half_vox).astype(int) - lo
    return lo, hi, pad_lo


def aligned_grid(vol, spacing, grid_mm=GRID_MM, grid_n=64, cube_mm=BRAIN_CUBE_MM):
    """Center on brain, crop physical cube, resample to grid_n**3 at grid_mm."""
    com = brain_com(vol) * spacing  # in mm
    lo, hi, _ = crop_physical(vol, spacing, com, cube_mm / 2)
    crop = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if crop.size == 0:
        crop = vol
    # make a cube in native voxels (side = max physical extent in voxels)
    n = max(crop.shape)
    cube = np.zeros((n, n, n), dtype=np.float32)
    s = crop.shape
    off = [(n - s[d]) // 2 for d in range(3)]
    cube[off[0]:off[0] + s[0], off[1]:off[1] + s[1], off[2]:off[2] + s[2]] = crop
    # resample uniformly to the target grid
    f = grid_n / n
    out = zoom(cube, (f, f, f), order=1, mode="nearest")
    return np.asarray(out, dtype=np.float32)


def _project_axes(crop, proj="mip"):
    """Project a 3D crop onto the three axes (mip = max, aip = mean)."""
    if proj == "aip":
        return [np.asarray(crop.mean(axis=d), dtype=np.float32) for d in range(3)]
    return [np.asarray(crop.max(axis=d), dtype=np.float32) for d in range(3)]


def aligned_mips(vol, spacing, target=MIP_TARGET, cube_mm=BRAIN_CUBE_MM, proj="mip"):
    """MIP/AIP projections of the brain-centered physical cube at native resolution."""
    com = brain_com(vol) * spacing
    half = cube_mm / 2
    lo, hi, pad_lo = crop_physical(vol, spacing, com, half)
    crop = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    views = _project_axes(crop, proj)
    # resize each projection so the LONG side fits the physical cube aspect
    out = []
    for m in views:
        m = np.ascontiguousarray(m, dtype=np.float32)
        f = target / max(m.shape)
        m = zoom(m, (f, f), order=1)
        t = np.zeros((target, target), dtype=np.float32)
        h, w = m.shape
        t[(target - h) // 2:(target - h) // 2 + h, (target - w) // 2:(target - w) // 2 + w] = m
        out.append(t)
    views = np.stack(out)
    vmin, vmax = views.min(), views.max()
    if vmax - vmin > 1e-8:
        views = (views - vmin) / (vmax - vmin)
    return np.ascontiguousarray(views, dtype=np.float32)


def striatum_com(vol, spacing, frac=0.6):
    """Centroid of the bright (striatal) voxels, in mm."""
    vmax = float(vol.max())
    mask = vol >= frac * vmax
    if not mask.any():
        return brain_com(vol) * spacing
    return np.asarray(center_of_mass(mask), dtype=np.float64) * np.array(spacing)


def aligned_mips_centered(vol, spacing, target=MIP_TARGET, cube_mm=160.0,
                          center="striatum", proj="mip"):
    """MIP/AIP projections of a physical cube centered on the striatum/brain.

    A tighter crop around the striatum gives the CNN more resolution on the
    diagnostically relevant region than the full 200 mm brain cube.
    """
    if center == "striatum":
        center_mm = striatum_com(vol, spacing)
    else:
        center_mm = brain_com(vol) * spacing
    half = cube_mm / 2
    lo, hi, _ = crop_physical(vol, spacing, center_mm, half)
    crop = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    views = _project_axes(crop, proj)
    out = []
    for m in views:
        m = np.ascontiguousarray(m, dtype=np.float32)
        f = target / max(m.shape)
        m = zoom(m, (f, f), order=1)
        t = np.zeros((target, target), dtype=np.float32)
        h, w = m.shape
        t[(target - h) // 2:(target - h) // 2 + h, (target - w) // 2:(target - w) // 2 + w] = m
        out.append(t)
    views = np.stack(out)
    vmin, vmax = views.min(), views.max()
    if vmax - vmin > 1e-8:
        views = (views - vmin) / (vmax - vmin)
    return np.ascontiguousarray(views, dtype=np.float32)


def _cache_load(nifti_path, cache_dir, builder, suffix=""):
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    st = Path(nifti_path).stat()
    key = hashlib.md5(f"{nifti_path}:{st.st_mtime_ns}:{st.st_size}:{suffix}".encode()).hexdigest()[:12]
    cache_file = cache_dir / f"{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    arr = builder(nifti_path)
    np.save(cache_file, arr)
    return arr


def load_aligned_grid(nifti_path, use_cache=True):
    if not use_cache:
        v, s = read_nifti(nifti_path)
        return aligned_grid(v, s)
    return _cache_load(nifti_path, CACHE_VOL3D,
                       lambda p: aligned_grid(*read_nifti(p)))


def load_aligned_mips(nifti_path, use_cache=True, proj="mip"):
    if not use_cache:
        v, s = read_nifti(nifti_path)
        return aligned_mips(v, s, proj=proj)
    cache_dir = CACHE_MIPS if proj == "mip" else CACHE_MIPS_EXP / "brain200_aip"
    return _cache_load(nifti_path, cache_dir,
                       lambda p: aligned_mips(*read_nifti(p), proj=proj),
                       suffix=f"t{MIP_TARGET}{proj}")


def centered_mips_cache_dir(cube_mm, center="striatum", target=MIP_TARGET,
                            proj="mip"):
    """Cache dir for a tight-crop MIP variant (uid-keyed), e.g. cache_mips_exp/str120."""
    name = f"str{cube_mm}" if center == "striatum" else f"brain{cube_mm}"
    if target != MIP_TARGET:
        name = f"{name}t{target}"
    if proj != "mip":
        name = f"{name}_{proj}"
    return CACHE_MIPS_EXP / name


def load_centered_mips(nifti_path, cube_mm=120, center="striatum",
                       target=MIP_TARGET, use_cache=True, proj="mip"):
    """MIP/AIPs for a tight-crop variant; reads uid-keyed cache, builds on miss."""
    if not use_cache:
        v, s = read_nifti(nifti_path)
        return aligned_mips_centered(v, s, target=target, cube_mm=cube_mm,
                                     center=center, proj=proj)
    cache_dir = centered_mips_cache_dir(cube_mm, center, target, proj)
    cache_file = cache_dir / f"{Path(nifti_path).name[:-7]}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    v, s = read_nifti(nifti_path)
    arr = aligned_mips_centered(v, s, target=target, cube_mm=cube_mm,
                                center=center, proj=proj)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, arr)
    return arr