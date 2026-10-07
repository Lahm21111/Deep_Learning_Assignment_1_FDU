import torch

from losses.losses import Pix2PixLoss
from models.patchgan import PatchGAN
from models.unet import Generator
from train_pix2pix import discriminator_step, generator_step, load_generator_weights


def test_patchgan_shape():
    discriminator = PatchGAN()
    prediction = discriminator(torch.randn(2, 3, 256, 256), torch.randn(2, 1, 256, 256))
    assert prediction.shape == (2, 1, 30, 30)


def test_gradient_isolation_and_losses():
    torch.manual_seed(42)
    generator = Generator(channels=(4, 8, 16, 32, 64))
    discriminator = PatchGAN(base_channels=8)
    criterion = Pix2PixLoss(100)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    photo = torch.rand(2, 3, 64, 64)
    sketch = torch.rand(2, 1, 64, 64)
    g_before = generator.head.weight.detach().clone()
    d_before = discriminator.net[0].weight.detach().clone()
    fake, d_losses = discriminator_step(generator, discriminator, photo, sketch, criterion,
                                        optimizer_D, scaler, False)
    assert fake.grad_fn is not None
    assert torch.equal(g_before, generator.head.weight)
    assert not torch.equal(d_before, discriminator.net[0].weight)
    assert torch.isfinite(d_losses["loss_D"])
    fake.retain_grad()
    d_after = discriminator.net[0].weight.detach().clone()
    g_losses = generator_step(discriminator, photo, sketch, fake, criterion,
                              optimizer_G, scaler, False)
    assert torch.equal(d_after, discriminator.net[0].weight)
    assert not torch.equal(g_before, generator.head.weight)
    assert fake.grad is not None and fake.grad.abs().sum() > 0
    assert torch.isfinite(g_losses["loss_G"])


def test_load_unet_checkpoint_as_generator(tmp_path):
    baseline = Generator(channels=(4, 8, 16, 32, 64))
    path = tmp_path / "unet.pth"
    torch.save({"model": baseline.state_dict()}, path)
    generator = Generator(channels=(4, 8, 16, 32, 64))
    load_generator_weights(generator, path, "cpu")
    assert torch.equal(generator.head.weight, baseline.head.weight)
