from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_comparison(photo, prediction, target, path):
    def picture(tensor, mode):
        array = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        if mode == "RGB":
            array = array.transpose(1, 2, 0)
        else:
            array = array[0]
        return Image.fromarray(array, mode=mode)
    panels = [picture(photo, "RGB"), picture(prediction, "L").convert("RGB"),
              picture(target, "L").convert("RGB")]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * 3, height + 24), "white")
    for i, (panel, title) in enumerate(zip(panels, ("Input Photo", "Generated Sketch", "Ground Truth Sketch"))):
        canvas.paste(panel, (i * width, 24))
        ImageDraw.Draw(canvas).text((i * width + 4, 4), title, fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_gray(tensor, path):
    array = (tensor.detach().cpu().clamp(0, 1)[0].numpy() * 255).round().astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def save_model_comparison(photo, baseline, pix2pix, target, path, candidate_label="Pix2Pix"):
    images = []
    for tensor, mode in ((photo, "RGB"), (baseline, "L"), (pix2pix, "L"), (target, "L")):
        array = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        image = Image.fromarray(array.transpose(1, 2, 0) if mode == "RGB" else array[0], mode=mode)
        images.append(image.convert("RGB"))
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 4, height + 24), "white")
    for i, (image, title) in enumerate(zip(images, ("Input", "U-Net + L1", candidate_label, "Ground Truth"))):
        canvas.paste(image, (i * width, 24))
        ImageDraw.Draw(canvas).text((i * width + 4, 4), title, fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
