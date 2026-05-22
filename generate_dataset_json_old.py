import json
import os
import glob
import random

def generate_monai_dataset_json(processed_dir, output_json_path, split_ratio=0.8):
    """
    Scans processed images and labels to generate a master dataset.json 
    manifest file, automatically splitting cases into training and validation sets.
    """
    images_dir = os.path.join(processed_dir, "images")
    labels_dir = os.path.join(processed_dir, "labels")
    
    # Find all processed mask files
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))
    
    all_pairs = []
    for label_path in label_files:
        case_name = os.path.basename(label_path).replace(".nii.gz", "")
        img_name = f"{case_name}_0000.nii.gz"
        img_path = os.path.join(images_dir, img_name)
        
        if os.path.exists(img_path):
            all_pairs.append({
                "image": f"./images/{img_name}",
                "label": f"./labels/{case_name}.nii.gz"
            })
        else:
            print(f"Warning: Missing matching image file for label {label_path}")

    # Set a random seed so the split is reproducible every time you run it
    random.seed(42)
    random.shuffle(all_pairs)
    
    # Calculate split boundary
    split_idx = int(len(all_pairs) * split_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # Construct the schema with BOTH training and validation keys
    dataset_manifest = {
        "description": "VCU Lung Cardiac Chamber Segmentation Dataset",
        "labels": {
            "0": "background",
            "1": "Ventricle Left",
            "2": "Ventricle Right",
            "3": "Attrium Left",
            "4": "Attrium Right"
        },
        "modality": {
            "0": "CT"
        },
        "numTraining": len(train_pairs),
        "numValidation": len(val_pairs),
        "training": train_pairs,
        "validation": val_pairs
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_manifest, f, indent=4, ensure_ascii=False)
        
    print(f"✓ Dataset manifest successfully generated!")
    print(f"   • Total Cases: {len(all_pairs)}")
    print(f"   • Training Set: {len(train_pairs)} cases")
    print(f"   • Validation Set: {len(val_pairs)} cases")
    print(f" Saved to: {output_json_path}")

if __name__ == "__main__":
    PROCESSED_DATA_ROOT = "/home/abishek/projects/cardiac_segmentation/data/processed"
    OUTPUT_JSON = os.path.join(PROCESSED_DATA_ROOT, "dataset.json")
    
    generate_monai_dataset_json(PROCESSED_DATA_ROOT, OUTPUT_JSON)