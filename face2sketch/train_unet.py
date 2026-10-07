"""Train the U-Net + L1 baseline using the shared project pipeline."""
from train import main


if __name__ == "__main__":
    main(default_config="configs/unet.yaml", allowed_models=("unet",))
