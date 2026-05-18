# run_pipeline.py
import os
from src.config import SegmentationConfig
from src.data_loader import CardiacDataModule
from src.train import CardiacTrainer

def main():
    # 1. Instantiate isolated configuration
    config = SegmentationConfig()
    
    # 2. Ensure artifact directories exist before training starts
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 3. Initialize your data pipeline
    print("📦 Loading dataset and initializing MONAI CacheDataset (this may take a moment)...")
    data_module = CardiacDataModule(config)
    data_module.setup()
    
    # 4. Run training loop
    print("🏗️ Building SwinUNETR architecture and starting training pipeline...")
    trainer = CardiacTrainer(config, data_module)
    trainer.fit()

if __name__ == "__main__":
    main()