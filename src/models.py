# src/models.py
import torch
from monai.networks.nets import SwinUNETR

class CardiacSegmentationModel:
    @staticmethod
    def build_model(config):
        if config.model_name == "swin_unetr":
            model = SwinUNETR(
                in_channels=1,
                out_channels=config.num_classes,
                feature_size=48,
                use_checkpoint=True
            )
            if config.pretrained:
                try:
                    weights = torch.load(config.pretrained_model_path, weights_only=True)
                    model.load_from(weights=weights)
                    print("✓ Loaded pre-trained Swin UNETR weights.")
                except FileNotFoundError:
                    print("⚠ Pre-trained weight file not found. Initializing from scratch.")
            return model.to(config.device)
        else:
            raise ValueError(f"Unsupported model: {config.model_name}")