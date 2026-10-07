import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from train import device_for, load_checkpoint, make_model
from utils.visualization import save_gray
from datasets.edge_fs2k import photo_edges


def load_generator(checkpoint, device, config_override=None):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = config_override or state["config"]
    model = make_model(config).to(device)
    model.load_state_dict(state["generator"] if "generator" in state else state["model"])
    model.eval()
    return model, config


def prepare_photo(image, size, resize_mode="stretch"):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if resize_mode == "center_crop":
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC,
                             centering=(0.5, 0.5))
    elif resize_mode == "letterbox":
        image = ImageOps.pad(image, (size, size), method=Image.Resampling.BICUBIC,
                             color=(255, 255, 255), centering=(0.5, 0.5))
    elif resize_mode == "stretch":
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    else:
        raise ValueError(f"Unknown resize mode: {resize_mode}")
    photo = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)
    return image, photo


def model_input(photo, config):
    if config["model"] == "edge_pix2pix":
        photo = torch.cat((photo, photo_edges(photo)), dim=0)
    return photo.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional YAML config; checkpoint config is used by default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resize-mode", choices=("stretch", "center_crop", "letterbox"), default="stretch",
                        help="How to adapt nonsquare photos to the model's square input")
    args = parser.parse_args()
    device = device_for(args.device)
    config_override = None
    if args.config:
        import yaml
        config_override = yaml.safe_load(Path(args.config).read_text())
    model, config = load_generator(args.checkpoint, device, config_override)
    output = Path(args.output)
    if len(args.input) > 1:
        output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for filename in args.input:
            with Image.open(filename) as image:
                _, photo = prepare_photo(image, config["input_size"], args.resize_mode)
                photo = model_input(photo, config).to(device)
            target = output / (Path(filename).stem + "_sketch.png") if len(args.input) > 1 else output
            save_gray(model(photo)[0], target)
            print(target)


if __name__ == "__main__":
    main()
