"""Pix2Pix training on the existing FS2K pipeline and U-Net generator."""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.fs2k import FS2KDataset, pairs_from_annotations, split_train_val
from datasets.edge_fs2k import EdgeFS2KDataset
from losses.losses import Pix2PixLoss
from models.patchgan import PatchGAN
from models.unet import UNet, count_parameters
from utils.visualization import save_comparison


def set_requires_grad(model, enabled):
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def discriminator_step(generator, discriminator, photo, sketch, criterion, optimizer, scaler, amp):
    set_requires_grad(discriminator, True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=photo.device.type, enabled=amp and photo.device.type == "cuda"):
        fake_sketch = generator(photo)
        pred_real = discriminator(photo, sketch)
        pred_fake = discriminator(photo, fake_sketch.detach())
        losses = criterion.discriminator(pred_real, pred_fake)
    scaler.scale(losses["loss_D"]).backward()
    scaler.step(optimizer)
    scaler.update()
    return fake_sketch, losses


def generator_step(discriminator, photo, sketch, fake_sketch, criterion, optimizer, scaler, amp):
    set_requires_grad(discriminator, False)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=photo.device.type, enabled=amp and photo.device.type == "cuda"):
        pred_fake = discriminator(photo, fake_sketch)
        losses = criterion.generator(pred_fake, fake_sketch, sketch)
    scaler.scale(losses["loss_G"]).backward()
    scaler.step(optimizer)
    scaler.update()
    return losses


def make_scheduler(optimizer, config):
    if config["scheduler"] == "constant":
        return None
    if config["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    raise ValueError(f"Unknown scheduler: {config['scheduler']}")


def load_generator_weights(generator, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    weights = state.get("generator", state.get("model"))
    if weights is None:
        raise ValueError(f"Checkpoint has no generator/model weights: {path}")
    generator.load_state_dict(weights, strict=True)


def train_pix2pix(args, config, device):
    if args.resume and args.generator_checkpoint:
        raise ValueError("Use either --resume or --generator_checkpoint")
    if config["gan_loss"] != "bce_logits" or config["reconstruction_loss"] != "l1":
        raise ValueError("First-stage Pix2Pix supports bce_logits + l1")
    pairs = pairs_from_annotations(config["data_root"], "train", config["style"])
    train_pairs, val_pairs = split_train_val(pairs, config["val_fraction"], config["seed"])
    if args.overfit:
        train_pairs = train_pairs[:args.overfit]
        val_pairs = train_pairs
    kwargs = dict(size=config["input_size"])
    dataset_type = EdgeFS2KDataset if config["model"] == "edge_pix2pix" else FS2KDataset
    train_set = dataset_type(train_pairs, augment=not args.overfit and config["horizontal_flip"],
                            brightness=config["brightness"] if not args.overfit else 0,
                            contrast=config["contrast"] if not args.overfit else 0, **kwargs)
    val_set = dataset_type(val_pairs, **kwargs)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], num_workers=config["num_workers"])
    generator = UNet(config["in_channels"], config["out_channels"], config["channels"]).to(device)
    discriminator = PatchGAN(config["in_channels"], config["out_channels"],
                             config["discriminator_channels"]).to(device)
    if args.generator_checkpoint:
        load_generator_weights(generator, args.generator_checkpoint, device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["lr_G"],
                                   betas=(config["beta1"], config["beta2"]),
                                   weight_decay=config["weight_decay"])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["lr_D"],
                                   betas=(config["beta1"], config["beta2"]),
                                   weight_decay=config["weight_decay"])
    scheduler_G = make_scheduler(optimizer_G, config)
    scheduler_D = make_scheduler(optimizer_D, config)
    criterion = Pix2PixLoss(config["lambda_L1"])
    scaler_G = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    scaler_D = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    start, best = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optimizer_G.load_state_dict(state["optimizer_G"])
        optimizer_D.load_state_dict(state["optimizer_D"])
        if scheduler_G and state.get("scheduler_G"):
            scheduler_G.load_state_dict(state["scheduler_G"])
        if scheduler_D and state.get("scheduler_D"):
            scheduler_D.load_state_dict(state["scheduler_D"])
        if state.get("scaler_G"):
            scaler_G.load_state_dict(state["scaler_G"])
            scaler_D.load_state_dict(state["scaler_D"])
        start, best = state["epoch"] + 1, state["best_val_loss"]
    if start >= config["epochs"]:
        print(f"Checkpoint is already at epoch {start}; configured epochs={config['epochs']}")
        return
    suffix = "_overfit" if args.overfit else ""
    checkpoint_stem = config["model"]
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(config["run_dir"] + suffix)
    writer = SummaryWriter(run_dir)
    print(f"device={device} G_params={count_parameters(generator):,} D_params={count_parameters(discriminator):,} "
          f"train={len(train_set)} val={len(val_set)}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step_count, step_seconds = 0, 0.0
    keys = ("loss_D", "loss_D_real", "loss_D_fake", "loss_G", "loss_G_GAN", "loss_G_L1")
    for epoch in range(start, config["epochs"]):
        generator.train()
        discriminator.train()
        totals = {key: 0.0 for key in keys}
        for photo, sketch, _ in train_loader:
            start_step = time.perf_counter()
            photo, sketch = photo.to(device), sketch.to(device)
            fake, d_losses = discriminator_step(generator, discriminator, photo, sketch, criterion,
                                                optimizer_D, scaler_D, config["amp"])
            g_losses = generator_step(discriminator, photo, sketch, fake, criterion,
                                      optimizer_G, scaler_G, config["amp"])
            losses = {**d_losses, **g_losses}
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise FloatingPointError(f"Nonfinite Pix2Pix loss in epoch {epoch+1}")
            for key in keys:
                totals[key] += losses[key].detach().item() * len(photo)
            step_seconds += time.perf_counter() - start_step
            step_count += 1
        generator.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for photo, sketch, _ in val_loader:
                photo, sketch = photo.to(device), sketch.to(device)
                val_l1 += torch.nn.functional.l1_loss(generator(photo), sketch).item() * len(photo)
        val_l1 /= len(val_set)
        if scheduler_G:
            scheduler_G.step()
            scheduler_D.step()
        averages = {key: total / len(train_set) for key, total in totals.items()}
        for key, value in averages.items():
            writer.add_scalar(f"train/{key}", value, epoch + 1)
        writer.add_scalar("validation/L1", val_l1, epoch + 1)
        print(f"epoch={epoch+1} D={averages['loss_D']:.5f} G={averages['loss_G']:.5f} "
              f"G_L1={averages['loss_G_L1']:.5f} val_L1={val_l1:.5f}", flush=True)
        if (epoch + 1) % config["preview_every"] == 0 or epoch == 0:
            photo, sketch, _ = val_set[0]
            with torch.no_grad():
                prediction = generator(photo.unsqueeze(0).to(device))[0].cpu()
            save_comparison(photo[:3], prediction, sketch, run_dir / f"epoch_{epoch+1:04d}.png")
            panel = torch.cat((photo[:3], prediction.repeat(3, 1, 1), sketch.repeat(3, 1, 1)), dim=2)
            writer.add_image("comparison", panel, epoch + 1)
        improved = val_l1 < best
        best = min(best, val_l1)
        state = {"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                 "optimizer_G": optimizer_G.state_dict(), "optimizer_D": optimizer_D.state_dict(),
                 "scheduler_G": scheduler_G.state_dict() if scheduler_G else None,
                 "scheduler_D": scheduler_D.state_dict() if scheduler_D else None,
                 "scaler_G": scaler_G.state_dict(), "scaler_D": scaler_D.state_dict(),
                 "epoch": epoch, "best_val_loss": best, "config": config}
        torch.save(state, checkpoint_dir / f"{checkpoint_stem}{suffix}_last.pth")
        if improved:
            torch.save(state, checkpoint_dir / f"{checkpoint_stem}{suffix}_best.pth")
    writer.close()
    usage = {"average_iteration_seconds": step_seconds / step_count,
             "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
             "best_validation_l1": best, "epochs": config["epochs"]}
    (run_dir / "summary.json").write_text(__import__("json").dumps(usage, indent=2))
    print(usage)


if __name__ == "__main__":
    from train import main
    main(default_config="configs/pix2pix.yaml", allowed_models=("pix2pix", "edge_pix2pix"))
