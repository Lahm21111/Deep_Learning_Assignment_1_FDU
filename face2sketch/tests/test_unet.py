import torch

from losses.losses import SketchLoss
from models.unet import UNet, count_parameters


def test_shape_and_backward():
    model = UNet(channels=(32, 64, 128, 256, 512))
    x = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    output = model(x)
    assert output.shape == target.shape
    assert output.min() >= 0 and output.max() <= 1
    SketchLoss()(output, target).backward()
    assert model.head.weight.grad is not None
    assert count_parameters(model) > 0


def test_small_model_ssim():
    model = UNet(channels=(4, 8, 16, 32, 64))
    output = model(torch.rand(1, 3, 32, 32))
    loss = SketchLoss("l1_ssim")(output, torch.rand_like(output))
    assert torch.isfinite(loss)
