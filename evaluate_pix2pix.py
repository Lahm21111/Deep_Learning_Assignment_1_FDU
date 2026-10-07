"""Evaluate RGB or RGB-plus-edge Pix2Pix on the official FS2K test split."""
from evaluate import main


if __name__ == "__main__":
    main(default_config="configs/pix2pix.yaml", allowed_models=("pix2pix", "edge_pix2pix"))
