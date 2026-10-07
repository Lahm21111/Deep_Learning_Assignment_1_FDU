import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.encoder = nn.ModuleList()
        previous = in_channels
        for width in channels:
            self.encoder.append(DoubleConv(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.decoder = nn.ModuleList()
        for width, skip in zip(reversed(channels[1:]), reversed(channels[:-1])):
            self.decoder.append(DoubleConv(width + skip, skip))
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i < len(self.encoder) - 1:
                skips.append(x)
                x = self.pool(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((skip, x), dim=1))
        return torch.sigmoid(self.head(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Pix2Pix uses the unchanged baseline U-Net as its generator.
Generator = UNet
