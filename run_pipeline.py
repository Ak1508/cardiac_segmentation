# run_pipeline.py
from src.config import SegmentationConfig
from src.data_loader import CardiacDataModule
from src.train import CardiacTrainer

def main():
    # Instantiate isolated configuration
    config = SegmentationConfig()
    
    # Initialize your data pipeline
    data_module = CardiacDataModule(config)
    data_module.setup()
    
    # Run training loop
    trainer = CardiacTrainer(config, data_module)
    trainer.fit()

if __name__ == "__main__":
    main()