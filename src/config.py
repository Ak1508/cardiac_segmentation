# src/config.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SegmentationConfig:
    """Configuration for independent cardiac substructure segmentation."""
    # Data paths relative to the project root
    data_dir: str = "./data/processed/"
    dataset_json: str = "dataset.json"  # FIXED: Matches our generated manifest name
    num_classes: int = 20  # 0:background, 1:LV, 2:RV, 3:LA, 4:RA

    # Model parameters
    model_name: str = "swin_unetr"
    pretrained: bool = False  
    pretrained_model_path: str = "./models/model_swinvit.pt"

    # Training Hyperparameters
    batch_size: int = 1
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Target spacing and crop size matching the tutorial workflow
    spatial_size: Tuple[int, int, int] = (96, 96, 96)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Reflects true isotropic data
    intensity_range: Tuple[int, int] = (-175, 250)  # CT window for soft tissue / blood pools

    # Validation & Runtime
    val_interval: int = 1
    device: str = "cuda:0"
    num_workers: int = 0
    pin_memory: bool = True

    # Output paths
    model_save_dir: str = "./models/"
    log_dir: str = "./logs/"