# Face2Sketch implementation details

The repository keeps three models trained on **all 1,058 pairs in the official FS2K training split**. They share the same 256×256 paired photo/sketch preprocessing and `[0,1]` output range. No validation subset is held out in these runs; `best` means lowest **training L1**, not best test quality. The official test split remains separate.

| Method | Generator input | Generator parameters | Objective | Checkpoint |
| --- | --- | ---: | --- | --- |
| U-Net L1 | RGB, 3 channels | 7,849,601 | L1 | `checkpoints/unet_all_styles_best.pth` |
| U-Net L1 GAN | RGB, 3 channels | 7,849,601 | GAN + 100 × L1 | `checkpoints/pix2pix_all_styles_best.pth` |
| U-Net Edge | RGB + Canny, 4 channels | 7,849,889 | GAN + 100 × L1 | `checkpoints/edge_pix2pix_all_styles_best.pth` |

Both GAN models use a conditional 70×70 PatchGAN discriminator. Edge detection uses bilateral filtering (`d=5`, `sigmaColor=35`, `sigmaSpace=5`) and Canny thresholds `45/110`, then concatenates the edge map with the RGB photo. The ground-truth sketch is unchanged. Inference loads only the generator.

## Run

Follow [README.md](README.md) to download FS2K and install dependencies. Commands below run from the repository root:

```bash
python scripts/check_dataset.py --data-root data/FS2K
python -m pytest tests -q

python train_unet.py --config configs/unet_all_styles.yaml
python train_pix2pix.py --config configs/pix2pix_all_styles.yaml
python train_pix2pix.py --config configs/edge_pix2pix_all_styles.yaml

python inference.py --checkpoint checkpoints/edge_pix2pix_all_styles_best.pth --input image_2.jpg --output results/edge_sketch.png
python compare_models.py --input image_2.jpg --output results/comparison.png
```

The `--resume` option restores an experiment's `last` checkpoint, including optimizer state. `--generator_checkpoint` initializes either GAN generator from a compatible U-Net checkpoint. TensorBoard logs and periodic previews are under `runs/`. The default comparison order is input photo, U-Net L1, U-Net L1 GAN, U-Net Edge.
