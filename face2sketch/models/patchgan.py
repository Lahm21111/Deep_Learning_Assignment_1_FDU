from torch import nn
import torch


class PatchGAN(nn.Module):
    """Conditional 70x70 PatchGAN; returns 30x30 logits for 256x256 inputs."""

    def __init__(self, photo_channels=3, sketch_channels=1, base_channels=64):
        super().__init__()

        def block(in_channels, out_channels, stride, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        c = base_channels
        self.net = nn.Sequential(
            *block(photo_channels + sketch_channels, c, 2, normalize=False),
            *block(c, c * 2, 2),
            *block(c * 2, c * 4, 2),
            *block(c * 4, c * 8, 1),
            nn.Conv2d(c * 8, 1, 4, 1, 1),
        )

    def forward(self, photo, sketch):
        if photo.shape[0] != sketch.shape[0] or photo.shape[-2:] != sketch.shape[-2:]:
            raise ValueError("Photo and sketch batch/spatial dimensions must match")
        return self.net(torch.cat((photo, sketch), dim=1))
