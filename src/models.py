# src/models.py
import torch
from monai.networks.nets import SwinUNETR

class CardiacSegmentationModel:
    @staticmethod
    def build_model(config):
        if config.model_name == "swin_unetr":
            # Explicitly adding img_size avoids positional embedding errors 
            # if loading pretrained Swin ViT weights later.
            model = SwinUNETR(
                # roi_size=config.spatial_size,
                # img_size=config.spatial_size, 
                in_channels=1,
                out_channels=config.num_classes,
                feature_size=48,
                use_checkpoint=True
            )
            
            if config.pretrained:
                try:
                    weights = torch.load(config.pretrained_model_path, weights_only=True)
                    # MONAI's SwinUNETR has a specialized loader method for SSL weights
                    model.load_from(weights=weights)
                    print(f" Loaded pre-trained Swin UNETR weights from {config.pretrained_model_path}")
                except FileNotFoundError:
                    print(f" Pre-trained weight file not found at {config.pretrained_model_path}. Initializing from scratch.")
                except Exception as e:
                    print(f" Error loading weights: {e}. Falling back to random initialization.")
                    
            return model.to(config.device)
        else:
            raise ValueError(f"Unsupported model: {config.model_name}")