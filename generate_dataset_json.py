import pandas as pd
import json
import os
import glob
import random

def get_labels_from_excel(excel_path):
    # 1. Read Excel
    df = pd.read_excel(excel_path)
    
    # 2. Clean all column names: remove spaces, make lowercase
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 3. Now look for columns using the cleaned names
    # Ensure your Excel headers are lowercase 'class id' and 'standard name'
    subset_cols = ['no.', 'standard name']
    
    # Check if they exist
    if not all(col in df.columns for col in subset_cols):
        print(f"Error: Columns not found. Found columns: {list(df.columns)}")
        raise KeyError(f"Expected columns {subset_cols} not found in Excel.")
        
    df = df.dropna(subset=subset_cols)
    
    labels = {"0": "background"}
    df = df.sort_values(by='no.')
    
    for _, row in df.iterrows():
        class_id = str(int(row['no.']))
        name = str(row['standard name'])
        labels[class_id] = name
        
    return labels

def get_labels_from_excel_old(excel_path):
    """Reads the Excel file and converts it into the dictionary format MONAI expects."""
    # Assuming columns: 'Class ID', 'Standard Name'
    df = pd.read_excel(excel_path)
    
    # Start with background
    labels = {"0": "background"}
    
    # Ensure data is clean and sorted by ID
    df = df.dropna(subset=['Class ID', 'Standard Name'])
    df = df.sort_values(by='Class ID')
    
    for _, row in df.iterrows():
        class_id = str(int(row['Class ID']))
        name = str(row['Standard Name'])
        labels[class_id] = name
        
    return labels

def generate_monai_dataset_json(processed_dir, excel_path, output_json_path, split_ratio=0.8):
    # ... (Keep your existing image/label scanning logic here) ...
    images_dir = os.path.join(processed_dir, "images")
    labels_dir = os.path.join(processed_dir, "labels")
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

    random.seed(42)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * split_ratio)
    
    # DYNAMIC LABELS: Call the new function
    dynamic_labels = get_labels_from_excel(excel_path)

    dataset_manifest = {
        "description": "VCU Lung Cardiac Chamber Segmentation Dataset",
        "labels": dynamic_labels, # <--- Now dynamic!
        "modality": {"0": "CT"},
        "numTraining": split_idx,
        "numValidation": len(all_pairs) - split_idx,
        "training": all_pairs[:split_idx],
        "validation": all_pairs[split_idx:]
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_manifest, f, indent=4, ensure_ascii=False)
        
    print(f"✓ Dataset manifest generated with {len(dynamic_labels)-1} classes from Excel.")

if __name__ == "__main__":
    PROCESSED_DATA_ROOT = "/home/abishek/projects/cardiac_segmentation/data/processed"
    EXCEL_PATH = "/home/abishek/projects/cardiac_segmentation/data/roi_mapping.xlsx" # Update path
    OUTPUT_JSON = os.path.join(PROCESSED_DATA_ROOT, "dataset.json")
    
    generate_monai_dataset_json(PROCESSED_DATA_ROOT, EXCEL_PATH, OUTPUT_JSON)