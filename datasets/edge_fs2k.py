"""RGB photo plus a matching Canny edge channel for paired FS2K training."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.fs2k import FS2KDataset


def photo_edges(photo):
    """Return a [1,H,W] edge map in [0,1] from an RGB [3,H,W] tensor."""
    if photo.ndim != 3 or photo.shape[0] != 3:
        raise ValueError("Expected one RGB photo with shape [3,H,W]")
    rgb = (photo.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=5)
    edges = cv2.Canny(smooth, threshold1=45, threshold2=110, L2gradient=True)
    return torch.from_numpy(edges.copy()).unsqueeze(0).float().div(255)


class EdgeFS2KDataset(Dataset):
    def __init__(self, pairs, **kwargs):
        self.base = FS2KDataset(pairs, **kwargs)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        photo, sketch, name = self.base[index]
        return torch.cat((photo, photo_edges(photo)), dim=0), sketch, name
