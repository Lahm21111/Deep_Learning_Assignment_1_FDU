import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.fs2k import FS2KDataset, pairs_from_annotations, split_train_val
from losses.losses import SketchLoss
from models.unet import UNet, count_parameters
from utils.visualization import save_comparison


def device_for(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(config):
    return UNet(config["in_channels"], config["out_channels"], config["channels"])


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    for photo, sketch, _ in loader:
        photo, sketch = photo.to(device), sketch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type in ("cuda", "cpu")):
                prediction = model(photo)
                loss = criterion(prediction, sketch)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total += loss.item() * len(photo)
    return total / len(loader.dataset)


def main(default_config="configs/unet.yaml", allowed_models=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overfit", type=int, default=0, help="Use N training pairs for a smoke/overfit run")
    if allowed_models == ("unet",):
        parser.set_defaults(generator_checkpoint=None)
    else:
        parser.add_argument("--generator_checkpoint", help="Initialize Pix2Pix Generator from a U-Net checkpoint")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if allowed_models is not None and config["model"] not in allowed_models:
        raise ValueError(f"This training entry point accepts {allowed_models}, got {config['model']}")
    set_seed(config["seed"])
    device = device_for(args.device)
    if config["model"] in ("pix2pix", "edge_pix2pix"):
        from train_pix2pix import train_pix2pix
        train_pix2pix(args, config, device)
        return
    if config["model"] != "unet":
        raise ValueError(f"Unknown model: {config['model']}")
    pairs = pairs_from_annotations(config["data_root"], "train", config["style"])
    train_pairs, val_pairs = split_train_val(pairs, config["val_fraction"], config["seed"])
    if args.overfit:
        train_pairs = train_pairs[:args.overfit]
        val_pairs = train_pairs
    kwargs = dict(size=config["input_size"])
    train_set = FS2KDataset(train_pairs, augment=not args.overfit and config["horizontal_flip"],
                            brightness=config["brightness"] if not args.overfit else 0,
                            contrast=config["contrast"] if not args.overfit else 0, **kwargs)
    val_set = FS2KDataset(val_pairs, **kwargs)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, batch_size=config["batch_size"],
                            num_workers=config["num_workers"])
    model = make_model(config).to(device)
    print(f"device={device} parameters={count_parameters(model):,} train={len(train_set)} val={len(val_set)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = SketchLoss(config["loss"], config["ssim_weight"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    start, best = 0, float("inf")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start, best = state["epoch"] + 1, state["best_val_loss"]
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(config["run_dir"])
    for epoch in range(start, config["epochs"]):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scaler, config["amp"])
        val_loss = run_epoch(model, val_loader, criterion, device, amp=config["amp"])
        scheduler.step()
        writer.add_scalars("loss", {"train": train_loss, "validation": val_loss}, epoch + 1)
        print(f"epoch={epoch+1} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        improved = val_loss < best
        best = min(best, val_loss)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_val_loss": best, "config": config}
        torch.save(state, checkpoint_dir / "last.pth")
        if improved:
            torch.save(state, checkpoint_dir / "best.pth")
        if (epoch + 1) % config["preview_every"] == 0 or epoch == 0:
            model.eval()
            photo, sketch, _ = val_set[0]
            with torch.no_grad():
                prediction = model(photo.unsqueeze(0).to(device))[0]
            save_comparison(photo, prediction, sketch, Path(config["run_dir"]) / f"epoch_{epoch+1:04d}.png")
    writer.close()


if __name__ == "__main__":
    main()
