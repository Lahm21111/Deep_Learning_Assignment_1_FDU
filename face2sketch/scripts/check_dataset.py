import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.fs2k import pairs_from_annotations, split_train_val
from datasets.fs2k import FS2KDataset
from utils.visualization import save_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/FS2K")
    parser.add_argument("--style", type=int, default=0)
    parser.add_argument("--preview", default="results/dataset_preview.png")
    args = parser.parse_args()
    root = Path(args.data_root)
    train = pairs_from_annotations(root, "train", args.style)
    test = pairs_from_annotations(root, "test", args.style)
    fitting, validation = split_train_val(train)
    names = [set(p[2] for p in group) for group in (fitting, validation, test)]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
    sizes = {}
    for photo, sketch, _ in train + test:
        with Image.open(photo) as image:
            image.verify()
        with Image.open(sketch) as image:
            image.verify()
        with Image.open(photo) as image:
            sizes[str(image.size)] = sizes.get(str(image.size), 0) + 1
    sample, target, _ = FS2KDataset(fitting)[0]
    save_comparison(sample, target, target, args.preview)
    print(json.dumps({"train": len(fitting), "validation": len(validation), "test": len(test),
                      "original_photo_sizes": sizes, "preview": args.preview}, indent=2))


if __name__ == "__main__":
    main()
