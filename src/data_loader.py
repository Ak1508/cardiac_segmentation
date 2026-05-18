# src/data_loader.py
import json
import os
from monai.data import CacheDataset, ThreadDataLoader
from src.transforms import get_train_transforms, get_val_transforms

class CardiacDataModule:
    def __init__(self, config):
        self.config = config
        
    def setup(self):
        # Using os.path.join here ensures robust path handling
        json_path = os.path.normpath(os.path.join(self.config.data_dir, self.config.dataset_json))
        
        with open(json_path) as f:
            dataset_info = json.load(f)
        
        train_files = self._add_paths(dataset_info.get("training", []))
        val_files = self._add_paths(dataset_info.get("validation", []))
        
        self.train_ds = CacheDataset(data=train_files, transform=get_train_transforms(self.config), cache_rate=1.0, num_workers=self.config.num_workers)
        self.val_ds = CacheDataset(data=val_files, transform=get_val_transforms(self.config), cache_rate=1.0, num_workers=self.config.num_workers)
        
        self.train_loader = ThreadDataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = ThreadDataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=0)
        
    def _add_paths(self, file_list):
        # FIXED: Uses os.path.normpath to cleanly eliminate the "./" prefixes from the JSON paths
        return [
            {
                "image": os.path.normpath(os.path.join(self.config.data_dir, f["image"])), 
                "label": os.path.normpath(os.path.join(self.config.data_dir, f["label"]))
            } 
            for f in file_list
        ]