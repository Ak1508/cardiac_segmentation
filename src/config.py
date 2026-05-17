# src/config.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SegmentationConfig:
    """Configuration for independent cardiac substructure segmentation."""
    # Data paths relative to the project root
    data_dir: str = "./data/processed/"
    dataset_json: str = "dataset_split.json"
    num_classes: int = 5  # 0:background, 1:LV, 2:RV, 3:LA, 4:RA

    # Model parameters
    model_name: str = "swin_unetr"
    pretrained: bool = False  # Set to True if you have local weights
    pretrained_model_path: str = "./models/model_swinvit.pt"

    # Training Hyperparameters
    batch_size: int = 1
    num_epochs: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Target spacing and crop size matching the tutorial workflow
    spatial_size: Tuple[int, int, int] = (96, 96, 96)
    spacing: Tuple[float, float, float] = (1.5, 1.5, 2.0)
    intensity_range: Tuple[int, int] = (-175, 250)

    # Validation & Runtime
    val_interval: int = 5
    device: str = "cuda:0"
    num_workers: int = 4
    pin_memory: bool = True

    # Output paths
    model_save_dir: str = "./models/"
    log_dir: str = "./logs/"