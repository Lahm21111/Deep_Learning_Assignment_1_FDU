import torch
from torch import nn
from torch.nn import functional as F


def ssim(prediction, target, window=11):
    # Local SSIM for one-channel images in [0, 1].
    kernel = torch.arange(window, device=prediction.device, dtype=prediction.dtype) - window // 2
    kernel = torch.exp(-(kernel ** 2) / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    kernel = torch.outer(kernel, kernel).view(1, 1, window, window)
    mu_x = F.conv2d(prediction, kernel, padding=window // 2)
    mu_y = F.conv2d(target, kernel, padding=window // 2)
    sigma_x = F.conv2d(prediction * prediction, kernel, padding=window // 2) - mu_x.square()
    sigma_y = F.conv2d(target * target, kernel, padding=window // 2) - mu_y.square()
    sigma_xy = F.conv2d(prediction * target, kernel, padding=window // 2) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) /
            ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))).mean(dim=(1, 2, 3))


class SketchLoss(nn.Module):
    def __init__(self, kind="l1", ssim_weight=0.2):
        super().__init__()
        if kind not in ("l1", "l1_ssim"):
            raise ValueError(f"Unknown loss: {kind}")
        self.kind = kind
        self.ssim_weight = ssim_weight

    def forward(self, prediction, target):
        loss = F.l1_loss(prediction, target)
        if self.kind == "l1_ssim":
            loss = loss + self.ssim_weight * (1 - ssim(prediction.float(), target.float()).mean())
        return loss


class Pix2PixLoss(nn.Module):
    def __init__(self, lambda_l1=100.0):
        super().__init__()
        self.adversarial = nn.BCEWithLogitsLoss()
        self.reconstruction = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def discriminator(self, pred_real, pred_fake):
        real = self.adversarial(pred_real, torch.ones_like(pred_real))
        fake = self.adversarial(pred_fake, torch.zeros_like(pred_fake))
        return {"loss_D_real": real, "loss_D_fake": fake,
                "loss_D": 0.5 * (real + fake)}

    def generator(self, pred_fake, fake_sketch, real_sketch):
        gan = self.adversarial(pred_fake, torch.ones_like(pred_fake))
        l1 = self.reconstruction(fake_sketch, real_sketch)
        return {"loss_G_GAN": gan, "loss_G_L1": l1,
                "loss_G": gan + self.lambda_l1 * l1}
