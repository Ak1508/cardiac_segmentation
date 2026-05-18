# src/transforms.py
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped
)

def get_train_transforms(config):
    return Compose([
        # 1. Load the pre-processed isotropic volumes
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # --- Spacingd REMOVED: Data is already 1.0mm isotropic ---
        
        # 2. Windowing: Clip CT Hounsfield Units to focus on soft tissue / cardiac structures
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=config.intensity_range[0], a_max=config.intensity_range[1], 
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        
        # 3. Patch Extraction: Cut into smaller sub-volumes (e.g., 96x96x96) centered on labels
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=config.spatial_size,
            pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0
        ),
        
        # 4. Data Augmentations (Happens on CPU patches)
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        
        # 5. Move final patches to GPU right before feeding to network
        # EnsureTyped(keys=["image", "label"], device=config.device, track_meta=False),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ])

def get_val_transforms(config):
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=config.intensity_range[0], a_max=config.intensity_range[1], 
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        # Moved to end to keep full-size validation volumes out of VRAM cache until inference
        # EnsureTyped(keys=["image", "label"], device=config.device, track_meta=True),
        EnsureTyped(keys=["image", "label"], track_meta=True),
    ])