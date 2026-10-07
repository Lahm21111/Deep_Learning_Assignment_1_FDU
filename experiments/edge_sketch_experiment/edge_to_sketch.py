"""Generate an edge map and a simple ink sketch from a face photo."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def make_sketch(image: Image.Image, max_size: int = 1024):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Smooth small sensor noise while preserving glasses, eyes, and hair boundaries.
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=7)
    edges = cv2.Canny(smooth, threshold1=25, threshold2=70, L2gradient=True)
    # Join short broken lines and remove isolated one-pixel specks.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    cleaned = np.zeros_like(edges)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 5:
            cleaned[labels == index] = 255
    ink = 255 - cleaned
    # A small gray halo makes the hard Canny lines look more like pencil marks.
    soft = cv2.GaussianBlur(cleaned, (3, 3), 0.8)
    sketch = np.clip(255 - 0.85 * cleaned - 0.15 * soft, 0, 255).astype(np.uint8)
    return image, Image.fromarray(ink, "L"), Image.fromarray(sketch, "L")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()
    with Image.open(args.input) as image:
        photo, edges, sketch = make_sketch(image, args.max_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges.save(args.output_dir / "edges.png")
    sketch.save(args.output_dir / "edge_sketch.png")
    width, height = photo.size
    canvas = Image.new("RGB", (width * 3, height + 28), "white")
    for i, (image, title) in enumerate(zip((photo, edges, sketch),
                                           ("Input Photo", "Detected Edges", "Edge Sketch"))):
        canvas.paste(image.convert("RGB"), (width * i, 28))
        ImageDraw.Draw(canvas).text((width * i + 5, 6), title, fill="black")
    canvas.save(args.output_dir / "comparison.png")
    print(args.output_dir / "comparison.png")


if __name__ == "__main__":
    main()
