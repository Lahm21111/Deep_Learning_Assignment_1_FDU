import math

import torch

from losses.losses import ssim


@torch.no_grad()
def batch_metrics(prediction, target):
    prediction, target = prediction.float(), target.float()
    mae = (prediction - target).abs().mean(dim=(1, 2, 3))
    mse = (prediction - target).square().mean(dim=(1, 2, 3))
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    return {"mae": mae, "psnr": psnr, "ssim": ssim(prediction, target)}
