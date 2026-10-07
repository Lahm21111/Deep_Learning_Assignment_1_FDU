"""Evaluate the U-Net + L1 baseline on the official FS2K test split."""
from evaluate import main


if __name__ == "__main__":
    main(default_config="configs/unet_all_styles.yaml", allowed_models=("unet",))
