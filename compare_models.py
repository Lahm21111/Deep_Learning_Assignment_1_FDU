"""Generate one comparison panel using the three retained project checkpoints."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from inference import load_generator, model_input, prepare_photo
from train import device_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unet", default="checkpoints/unet_all_styles_best.pth")
    parser.add_argument("--pix2pix", default="checkpoints/pix2pix_all_styles_best.pth")
    parser.add_argument("--edge-pix2pix", default="checkpoints/edge_pix2pix_all_styles_best.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resize-mode", choices=("stretch", "center_crop", "letterbox"), default="stretch")
    args = parser.parse_args()
    device = device_for(args.device)
    checkpoints = (args.unet, args.pix2pix, args.edge_pix2pix)
    models = [load_generator(path, device) for path in checkpoints]
    sizes = {config["input_size"] for _, config in models}
    if len(sizes) != 1:
        raise ValueError(f"Models require different input sizes: {sizes}")
    with Image.open(args.input) as source:
        photo_image, photo = prepare_photo(source, sizes.pop(), args.resize_mode)
    panels = [photo_image]
    with torch.no_grad():
        for model, config in models:
            sketch = model(model_input(photo, config).to(device))[0, 0].cpu().clamp(0, 1)
            panels.append(Image.fromarray((sketch.numpy() * 255).round().astype(np.uint8), "L").convert("RGB"))
    width, height = photo_image.size
    canvas = Image.new("RGB", (4 * width, height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ("Input photo", "U-Net L1", "U-Net L1 GAN", "U-Net Edge")
    for index, (panel, title) in enumerate(zip(panels, titles)):
        canvas.paste(panel, (index * width, 24))
        draw.text((index * width + 5, 5), title, fill="black")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
