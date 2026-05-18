import os
import glob
import pydicom

def audit_rtstruct_names(raw_dir):
    print("Scanning RTSTRUCT files for ROI names...\n")
    unique_names = set()
    
    # Find all RTSTRUCT files in your raw directory
    rt_files = glob.glob(os.path.join(raw_dir, "**/*RTSTRUCT*"), recursive=True) + \
               glob.glob(os.path.join(raw_dir, "**/RS.*"), recursive=True)
               
    if not rt_files:
        print("No RTSTRUCT files found. Double-check your data directory structure.")
        return

    for rt_path in rt_files:
        try:
            ds = pydicom.dcmread(rt_path, stop_before_pixels=True)
            # Check if Structure Set ROI Sequence exists
            if hasattr(ds, "StructureSetROISequence"):
                for roi in ds.StructureSetROISequence:
                    unique_names.add(roi.ROIName.strip())
        except Exception as e:
            print(f"Error reading {os.path.basename(rt_path)}: {e}")

    print("=" * 40)
    print(f"Found {len(unique_names)} Unique ROI Names Across Dataset:")
    print("=" * 40)
    for name in sorted(unique_names):
        print(f"  - '{name}'")

if __name__ == "__main__":
    # Point this to where your raw unzipped patient folders are
    RAW_DATA_DIR = "./data/raw" 
    audit_rtstruct_names(RAW_DATA_DIR)
