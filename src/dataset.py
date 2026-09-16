"""PyTorch dataset for DaT scan volumes with light 3D augmentation."""

import numpy as np
import torch
from torch.utils.data import Dataset


class DATScanDataset(Dataset):
    def __init__(self, uids, nifti_dir, labels=None, augment=False, use_cache=True):
        self.uids = list(uids)
        self.nifti_dir = nifti_dir
        self.labels = labels
        self.augment = augment
        self.use_cache = use_cache

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        from .preprocess import load_volume_cached

        uid = self.uids[idx]
        vol = load_volume_cached(self.nifti_dir / f"{uid}.nii.gz", use_cache=self.use_cache)
        vol = np.ascontiguousarray(vol, dtype=np.float32)
        vol = vol[np.newaxis, ...]  # add channel dim
        if self.augment:
            vol = self._augment(vol)
        x = torch.from_numpy(np.ascontiguousarray(vol))
        if self.labels is None:
            return x
        y = torch.tensor(float(self.labels[uid]), dtype=torch.float32)
        return x, y

    def _augment(self, vol):
        # Random flips along axial axes (anatomy is roughly symmetric in-plane).
        for axis in (2, 3):
            if np.random.rand() < 0.5:
                vol = np.flip(vol, axis=axis)
        # Small intensity jitter (keep float32; numpy 2 promotes to float64).
        scale = 1.0 + 0.05 * np.random.randn()
        vol = (vol * scale).astype(np.float32)
        return np.ascontiguousarray(vol)
