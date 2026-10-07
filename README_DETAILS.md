# Face2Sketch

One PyTorch project with three trained methods on the same FS2K style-0 split: **U-Net L1**, **U-Net L1 GAN**, and **U-Net Edge**.

| Method | Config | Generator input | Objective | Best checkpoint |
| --- | --- | --- | --- | --- |
| U-Net L1 | `configs/unet.yaml` | RGB (3 channels) | L1 | `checkpoints/best.pth` |
| U-Net L1 GAN | `configs/pix2pix.yaml` | RGB (3 channels) | GAN + 100 × L1 | `checkpoints/pix2pix_best.pth` |
| U-Net Edge | `configs/edge_pix2pix.yaml` | RGB + Canny 45/110 (4 channels) | GAN + 100 × L1 | `checkpoints/edge_pix2pix_best.pth` |

U-Net Edge creates its fourth channel after the paired geometric transform. It applies a bilateral filter (`d=5`, `sigmaColor=35`, `sigmaSpace=5`) and Canny (`45/110`, `L2gradient=True`). All methods produce a `[0,1]` grayscale tensor and save it as a 256×256 grayscale PNG. Checkpoints for all three methods are retained.

## Data and setup

Run all commands below from the repository root. Download `FS2K.zip` from the [official FS2K repository](https://github.com/DengPingFan/FS2K) ([direct Google Drive page](https://drive.google.com/file/d/1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp/view?usp=sharing)), place it at `data/FS2K.zip`, then extract it:

```bash
mkdir -p data
unzip data/FS2K.zip -d data
```

The archive contains its own `FS2K/` folder. Confirm that `data/FS2K/` contains `photo/`, `sketch/`, `anno_train.json`, and `anno_test.json`. The dataset is excluded from Git. Skip extraction if those files are already present. Training uses a seeded 10% validation split from the selected training style.

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
python scripts/check_dataset.py --data-root data/FS2K --style 0
```

`check_dataset.py` verifies pairs, missing files, image decoding, and split overlap; it writes `results/dataset_preview.png`. No offline resize step is needed. At read time, photos become RGB and sketches become grayscale, both are resized to 256×256 and normalized to `[0,1]`. Training applies the same random horizontal flip to each photo/sketch pair; brightness and contrast augmentation affect only photos. Validation and single-photo inference use deterministic transforms. For the edge config, the Canny map is calculated from the transformed photo and concatenated as a fourth channel; it does not replace the true sketch label.

## Training

`train_unet.py` trains U-Net L1. `train_pix2pix.py` trains U-Net L1 GAN or U-Net Edge; choose the YAML file with `--config`.

```bash
python train_unet.py --config configs/unet.yaml

python train_pix2pix.py --config configs/pix2pix.yaml

python train_pix2pix.py --config configs/edge_pix2pix.yaml
```

Training supports `--overfit 16`, `--resume CHECKPOINT`, and, for U-Net L1 GAN or U-Net Edge, `--generator_checkpoint CHECKPOINT`. U-Net L1 writes `best.pth` / `last.pth`; U-Net L1 GAN writes `pix2pix_best.pth` / `pix2pix_last.pth`; U-Net Edge writes `edge_pix2pix_best.pth` / `edge_pix2pix_last.pth`, all under `checkpoints/`. No retraining is needed to use the included weights. TensorBoard logs are under `runs/`.

## Inference and one-image comparison

```bash
python inference.py --checkpoint checkpoints/edge_pix2pix_best.pth --input image.jpg --output results/image_edge_sketch.png
python compare_models.py --input image.jpg --output results/image_comparison.png
```

`compare_models.py` produces one panel containing the input photo and outputs from U-Net L1, U-Net L1 GAN, and U-Net Edge. It uses the same inference preprocessing as `inference.py`. Both commands support `--resize-mode stretch|center_crop|letterbox`; `stretch` is the training-compatible default. Use `center_crop` for nonsquare portraits when you want to preserve facial proportions.

Use your own photo with `inference.py` for one sketch, or `compare_models.py` for one panel showing all three model outputs. `image.jpg` is the latest example input.
