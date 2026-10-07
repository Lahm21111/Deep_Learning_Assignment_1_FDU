import json

import numpy as np
import pytest
from PIL import Image

from datasets.fs2k import FS2KDataset, pairs_from_annotations, split_train_val


def test_pairing_all_styles_without_validation(tmp_path):
    records = []
    for i in range(12):
        group = "photo1"
        name = f"image{i:04d}"
        photo = tmp_path / "photo" / group / f"{name}.jpg"
        sketch = tmp_path / "sketch" / "sketch1" / f"sketch{i:04d}.png"
        photo.parent.mkdir(parents=True, exist_ok=True)
        sketch.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((20, 30, 3), i * 10, dtype=np.uint8)).save(photo)
        Image.fromarray(np.full((20, 30), i * 10, dtype=np.uint8)).save(sketch)
        records.append({"image_name": f"{group}/{name}", "style": i % 2})
    (tmp_path / "anno_train.json").write_text(json.dumps(records[:10]))
    (tmp_path / "anno_test.json").write_text(json.dumps(records[10:]))
    pairs = pairs_from_annotations(tmp_path, "train", "all")
    train, val = split_train_val(pairs, 0.0, 42)
    assert len(train) == 10 and len(val) == 0
    assert len(pairs_from_annotations(tmp_path, "test", "all")) == 2
    photo, sketch, _ = FS2KDataset(train, augment=True)[0]
    assert photo.shape == (3, 256, 256) and sketch.shape == (1, 256, 256)
    assert 0 <= photo.min() <= photo.max() <= 1
    assert 0 <= sketch.min() <= sketch.max() <= 1
    pairs[0][1].unlink()
    with pytest.raises(FileNotFoundError):
        pairs_from_annotations(tmp_path, "train", "all")
