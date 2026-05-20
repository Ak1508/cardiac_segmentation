# src/train.py
import os
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from src.models import CardiacSegmentationModel

import os
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from src.models import CardiacSegmentationModel

class CardiacTrainer:
    def __init__(self, config, data_module):
        self.config = config
        self.data_module = data_module
        
        self.model = CardiacSegmentationModel.build_model(config).to(config.device) # Ensure model is on device
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # FIX 1: Corrected GradScaler instantiation
        self.scaler = torch.amp.GradScaler() 
        
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.post_pred = AsDiscrete(argmax=True, to_onehot=config.num_classes)
        self.post_label = AsDiscrete(to_onehot=config.num_classes)
        
        self.best_dice = 0.0

    def train_epoch(self):
        self.model.train()
        run_loss = 0.0
        # tqdm usage is fine, just ensure num_workers=0 in data_module
        for batch in tqdm(self.data_module.train_loader, desc="Training", leave=False):
            inputs, labels = batch["image"].to(self.config.device), batch["label"].to(self.config.device)
            self.optimizer.zero_grad()
            
            # FIX 2: Using the correct amp context manager
            with torch.amp.autocast(device_type='cuda'):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            run_loss += loss.item()
        return run_loss / len(self.data_module.train_loader)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.data_module.val_loader:
                inputs, labels = batch["image"].to(self.config.device), batch["label"].to(self.config.device)
                
                # FIX 3: Reduced overlap (sw_batch_size) from 4 to 1 to save VRAM
                # If it still crashes, reduce this number. 1 is the safest.
                with torch.amp.autocast(device_type='cuda'):
                    outputs = sliding_window_inference(inputs, self.config.spatial_size, 1, self.model)
                
                y_pred = [self.post_pred(i) for i in decollate_batch(outputs)]
                y_true = [self.post_label(i) for i in decollate_batch(labels)]
                self.dice_metric(y_pred=y_pred, y=y_true)
        
        score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        return score

    def fit(self):
        torch.backends.cudnn.benchmark = True
        # Add this line to ensure the folder exists before training starts
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            loss = self.train_epoch()
            print(f"Epoch {epoch+1:03d} | Train Loss: {loss:.4f}")
            
            if (epoch + 1) % self.config.val_interval == 0:
                val_dice = self.validate()
                print(f"Validation Dice: {val_dice:.4f}")
                
                # If val_dice is 0.0, it will still save if you use >= 
                # but usually, we want to save the first result to verify
                if val_dice >= self.best_dice:
                    self.best_dice = val_dice
                    save_path = os.path.join(self.config.model_save_dir, "cardiac_best.pth")
                    torch.save(self.model.state_dict(), save_path)
                    print(f"✓ Model saved to {save_path} with Dice: {self.best_dice:.4f}")

    def fittt(self):
        torch.backends.cudnn.benchmark = True
        for epoch in range(self.config.num_epochs):
            loss = self.train_epoch()
            print(f"Epoch {epoch+1:03d} | Train Loss: {loss:.4f}")
            
            if (epoch + 1) % self.config.val_interval == 0:
                val_dice = self.validate()
                print(f"Validation Dice: {val_dice:.4f}")
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    torch.save(self.model.state_dict(), os.path.join(self.config.model_save_dir, "cardiac_best.pth"))
                    print(f"✓ New best model saved with Dice: {self.best_dice:.4f}")