# Archived edge-preview experiment (no training)

This separate prototype first applies bilateral denoising and Canny edge detection, removes isolated specks, then renders the resulting edges as black pencil-like lines on white. It keeps the input aspect ratio. The Pix2Pix model is unchanged.

```bash
cd face2sketch
python experiments/edge_sketch_experiment/edge_to_sketch.py --input ../81032e406b612e9e03de581f89e6d8a7.jpg --output-dir experiments/edge_sketch_experiment/results
```

Outputs: `edges.png`, `edge_sketch.png`, and `comparison.png`. Feeding `edges.png` to the existing Pix2Pix generator is an experiment only: that generator was trained on RGB photos, so edge maps are outside its training distribution.

On the supplied square selfie, the direct edge sketch keeps the eyeglass frames well. Feeding the edge map into the existing Pix2Pix checkpoint produced `results/edges_through_pix2pix.png`, which lost detail. A learned edge-to-sketch model would need to be trained on edge maps paired with sketches; the current checkpoint cannot be expected to perform that translation reliably.
