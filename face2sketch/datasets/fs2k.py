import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def find_image(stem):
    for suffix in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        path = stem.with_suffix(suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing image: {stem}.[jpg|png|jpeg]")


def pairs_from_annotations(root, split, style):
    root = Path(root)
    annotations = json.loads((root / f"anno_{split}.json").read_text())
    pairs = []
    for item in annotations:
        if int(item["style"]) != int(style):
            continue
        name = Path(item["image_name"])
        if len(name.parts) != 2 or not name.parts[0].startswith("photo") or not name.parts[1].startswith("image"):
            raise ValueError(f"Unexpected FS2K image_name: {name}")
        photo = find_image(root / "photo" / name)
        sketch_name = name.parts[1].replace("image", "sketch", 1)
        sketch_dir = name.parts[0].replace("photo", "sketch", 1)
        sketch = find_image(root / "sketch" / sketch_dir / sketch_name)
        pairs.append((photo, sketch, name.as_posix()))
    if not pairs:
        raise ValueError(f"No style {style} pairs in {split}")
    return pairs


def split_train_val(pairs, fraction=0.1, seed=42):
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    count = max(1, round(len(indices) * fraction))
    val = set(indices[:count])
    return ([pair for i, pair in enumerate(pairs) if i not in val],
            [pair for i, pair in enumerate(pairs) if i in val])


class FS2KDataset(Dataset):
    def __init__(self, pairs, size=256, augment=False, brightness=0.1, contrast=0.1):
        self.pairs = list(pairs)
        self.size = size
        self.augment = augment
        self.brightness = brightness
        self.contrast = contrast

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        photo_path, sketch_path, name = self.pairs[index]
        with Image.open(photo_path) as image:
            photo = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC)
        with Image.open(sketch_path) as image:
            sketch = image.convert("L").resize((self.size, self.size), Image.Resampling.BICUBIC)
        if self.augment:
            if random.random() < 0.5:
                photo = photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                sketch = sketch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            photo = ImageEnhance.Brightness(photo).enhance(random.uniform(1-self.brightness, 1+self.brightness))
            photo = ImageEnhance.Contrast(photo).enhance(random.uniform(1-self.contrast, 1+self.contrast))
        photo = torch.from_numpy(np.asarray(photo).copy()).permute(2, 0, 1).float() / 255
        sketch = torch.from_numpy(np.asarray(sketch).copy()).unsqueeze(0).float() / 255
        return photo, sketch, name
