import torch

from datasets.edge_fs2k import photo_edges
from losses.losses import Pix2PixLoss
from models.patchgan import PatchGAN
from models.unet import Generator
from train_pix2pix import discriminator_step, generator_step


def test_edge_channel_and_conditional_shapes():
    photo = torch.zeros(2, 3, 256, 256)
    photo[:, :, 32:224, 32:224] = 1
    edges = torch.stack([photo_edges(item) for item in photo])
    assert edges.shape == (2, 1, 256, 256)
    assert edges.min() == 0 and edges.max() == 1
    condition = torch.cat((photo, edges), dim=1)
    generator = Generator(in_channels=4, channels=(4, 8, 16, 32, 64))
    discriminator = PatchGAN(photo_channels=4, base_channels=8)
    sketch = generator(condition)
    assert sketch.shape == (2, 1, 256, 256)
    assert discriminator(condition, sketch).shape == (2, 1, 30, 30)


def test_edge_conditioned_gradient_flow():
    torch.manual_seed(42)
    generator = Generator(in_channels=4, channels=(4, 8, 16, 32, 64))
    discriminator = PatchGAN(photo_channels=4, base_channels=8)
    condition = torch.rand(2, 4, 64, 64)
    sketch = torch.rand(2, 1, 64, 64)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    g_before = generator.head.weight.detach().clone()
    d_before = discriminator.net[0].weight.detach().clone()
    fake, _ = discriminator_step(generator, discriminator, condition, sketch, Pix2PixLoss(100),
                                 optimizer_d, scaler, False)
    assert torch.equal(g_before, generator.head.weight)
    assert not torch.equal(d_before, discriminator.net[0].weight)
    d_after = discriminator.net[0].weight.detach().clone()
    generator_step(discriminator, condition, sketch, fake, Pix2PixLoss(100), optimizer_g, scaler, False)
    assert torch.equal(d_after, discriminator.net[0].weight)
    assert not torch.equal(g_before, generator.head.weight)
