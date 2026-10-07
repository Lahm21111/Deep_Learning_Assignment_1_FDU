import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.fs2k import FS2KDataset, pairs_from_annotations
from datasets.edge_fs2k import EdgeFS2KDataset
from train import device_for, load_checkpoint, make_model
from utils.metrics import batch_metrics
from utils.visualization import save_comparison, save_gray, save_model_comparison


def main(default_config="configs/unet.yaml", allowed_models=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    if allowed_models == ("unet",):
        parser.set_defaults(baseline_checkpoint=None)
    else:
        parser.add_argument("--baseline-checkpoint", help="Optional U-Net checkpoint for four-panel comparisons")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if allowed_models is not None and config["model"] not in allowed_models:
        raise ValueError(f"This evaluation entry point accepts {allowed_models}, got {config['model']}")
    pix2pix = config["model"] in ("pix2pix", "edge_pix2pix")
    checkpoint = args.checkpoint or (str(Path(config["checkpoint_dir"]) / f"{config['model']}_best.pth") if pix2pix else "checkpoints/best.pth")
    output = Path(args.output or config.get("results_dir", "results/pix2pix" if pix2pix else "results"))
    device = device_for(args.device)
    dataset_type = EdgeFS2KDataset if config["model"] == "edge_pix2pix" else FS2KDataset
    dataset = dataset_type(pairs_from_annotations(config["data_root"], "test", config["style"]), size=config["input_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])
    model = make_model(config).to(device)
    if pix2pix:
        from train_pix2pix import load_generator_weights
        load_generator_weights(model, checkpoint, device)
    else:
        load_checkpoint(checkpoint, model, device=device)
    model.eval()
    baseline = None
    if args.baseline_checkpoint:
        baseline_state = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
        baseline = make_model(baseline_state.get("config", {**config, "in_channels": 3})).to(device)
        load_checkpoint(args.baseline_checkpoint, baseline, device=device)
        baseline.eval()
    totals = {key: 0.0 for key in ("mae", "psnr", "ssim")}
    with torch.no_grad():
        for photos, sketches, names in loader:
            predictions = model(photos.to(device)).cpu()
            baseline_predictions = baseline(photos[:, :3].to(device)).cpu() if baseline else None
            metrics = batch_metrics(predictions, sketches)
            for key, values in metrics.items():
                totals[key] += values.sum().item()
            for i, name in enumerate(names):
                basename = name.replace("/", "_")
                save_gray(predictions[i], output / "predictions" / f"{basename}.png")
                save_comparison(photos[i, :3], predictions[i], sketches[i], output / "comparisons" / f"{basename}.png")
                if baseline_predictions is not None:
                    save_model_comparison(photos[i, :3], baseline_predictions[i], predictions[i], sketches[i],
                                          output / "model_comparisons" / f"{basename}.png",
                                          candidate_label="RGB + Edges Pix2Pix" if config["model"] == "edge_pix2pix" else "Pix2Pix")
    result = {key: value / len(dataset) for key, value in totals.items()}
    result.update({"count": len(dataset), "style": config["style"]})
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
