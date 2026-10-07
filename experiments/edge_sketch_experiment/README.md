# Archived edge-preview experiment (no training)

This separate prototype first applies bilateral denoising and Canny edge detection, removes isolated specks, then renders the resulting edges as black pencil-like lines on white. It keeps the input aspect ratio. The trained U-Net L1 GAN model is unchanged.

```bash
python experiments/edge_sketch_experiment/edge_to_sketch.py --input image.jpg --output-dir experiments/edge_sketch_experiment/results
```

Outputs: `edges.png`, `edge_sketch.png`, and `comparison.png`. Feeding `edges.png` to the existing U-Net L1 GAN generator is an experiment only: that generator was trained on RGB photos, so edge maps are outside its training distribution.

On the supplied square selfie, the direct edge sketch keeps the eyeglass frames well. Feeding the edge map into the existing U-Net L1 GAN checkpoint produced `results/edges_through_pix2pix.png`, which lost detail. A learned edge-to-sketch model would need to be trained on edge maps paired with sketches; the current checkpoint cannot be expected to perform that translation reliably.
