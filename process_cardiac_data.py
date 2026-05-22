import os, sys
import glob
import logging
import SimpleITK as sitk
import numpy as np
import pydicom

OTHER_PROJECT_PATH = "/home/abishek/projects/ProViCNet/tools/"
if OTHER_PROJECT_PATH not in sys.path:
    sys.path.append(OTHER_PROJECT_PATH)


from dicom_converter import load_dicom_series, poly2mask
from src.roi_mapping import get_class_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def robust_extract_contours(rtstruct_path):
    """
    Directly extracts contours by robustly matching ROI number tags 
    without relying on exact attribute names that might fail.
    """
    rtstruct = pydicom.dcmread(rtstruct_path)
    contours_dict = {}
    
    # 1. Build a robust fallback mapping of ROI Number -> Structure Name
    roi_to_name = {}
    if hasattr(rtstruct, 'StructureSetROISequence'):
        for roi_seq in rtstruct.StructureSetROISequence:
            # Safely grab number and name using get parameters to avoid exceptions
            roi_num = getattr(roi_seq, 'ROINumber', getattr(roi_seq, 'ReferencedROINumber', None))
            roi_name = getattr(roi_seq, 'ROIName', None)
            if roi_num is not None and roi_name:
                roi_to_name[int(roi_num)] = roi_name

    logger.info(f"Robustly mapped {len(roi_to_name)} structures inside RTSTRUCT file.")

    # 2. Extract contour sequences matching them back to names correctly
    if hasattr(rtstruct, 'ROIContourSequence'):
        for roi_contour in rtstruct.ROIContourSequence:
            roi_num = getattr(roi_contour, 'ReferencedROINumber', None)
            
            # Find matching name
            structure_name = roi_to_name.get(int(roi_num)) if roi_num is not None else None
            
            if not structure_name:
                continue
                
            contours_list = []
            if hasattr(roi_contour, 'ContourSequence'):
                for contour in roi_contour.ContourSequence:
                    if hasattr(contour, 'ContourData'):
                        data = contour.ContourData
                        coords = [(float(data[i]), float(data[i+1]), float(data[i+2])) 
                                  for i in range(0, len(data), 3)]
                        if coords:
                            contours_list.append(coords)
            
            if contours_list:
                contours_dict[structure_name] = contours_list
                
    return contours_dict

def custom_contours_to_multiclass_mask(contours_dict, dicom_image):
    """Maps coordinates safely into an integrated multi-class integer volume matrix."""
    shape = dicom_image.GetSize()  # (width, height, depth) -> (X, Y, Z)
    mask = np.zeros((shape[2], shape[1], shape[0]), dtype=np.uint8) # (Z, Y, X)
    
    processed_count = 0
    
    for raw_structure_name, contours_list in contours_dict.items():
        class_idx = get_class_label(raw_structure_name)
        if class_idx == 0:
            continue
            
        logger.info(f"Processing target structure: '{raw_structure_name}' -> Class {class_idx}")
        contour_count = 0
        
        for contour_idx, contour_coords in enumerate(contours_list):
            voxel_coords = []
            z_indices = set()
            
            # 1. Convert all real-world mm coordinates to image pixel indexes
            for point in contour_coords:
                try:
                    world_x, world_y, world_z = point
                    phys_point = (float(world_x), float(world_y), float(world_z))
                    
                    # SimpleITK returns index as (X_pixel, Y_pixel, Z_pixel)
                    voxel_idx = dicom_image.TransformPhysicalPointToContinuousIndex(phys_point)
                    
                    voxel_coords.append(voxel_idx)
                    z_indices.add(int(round(voxel_idx[2])))
                except Exception:
                    continue
            
            if not voxel_coords:
                continue
                
            # FORCE convert to explicit float64 numpy array for slicing operations
            voxel_coords = np.array(voxel_coords, dtype=np.float64)
            
            # Print a single debug trace for the first contour to audit coordinates math
            if contour_count == 0:
                logger.info(f"    [DEBUG] Sample converted voxel coordinate: {voxel_coords[0]} | Target Vol Z-Max: {shape[2]}")
            
            # 2. Rasterize the 2D contour polygon onto each intercepted slice layer
            for z_idx in z_indices:
                z_idx = int(z_idx)
                if z_idx < 0 or z_idx >= shape[2]:
                    continue  # Coordinate falls outside 3D scan volume limits
                
                try:
                    # Filter points matching this slice layer close to the Z plane
                    slice_mask = np.abs(voxel_coords[:, 2] - z_idx) < 0.5
                    coords_2d = voxel_coords[slice_mask]
                    
                    if len(coords_2d) < 3: 
                        continue
                    
                    # Extract continuous X and Y coordinates
                    x_pts = coords_2d[:, 0]
                    y_pts = coords_2d[:, 1]
                    
                    # poly2mask expects (x_coords, y_coords, mask_shape=[width, height])
                    filled_poly = poly2mask(x_pts, y_pts, [shape[0], shape[1]])
                    
                    # Burn the class index integer value into our master array slice
                    mask[z_idx, filled_poly] = class_idx
                    contour_count += 1
                except Exception as e:
                    # Temporarily log any hidden parsing failures
                    logger.debug(f"Failed to fill slice contour: {e}")
                    continue
                    
        if contour_count > 0:
            logger.info(f"  ✓ Burned {contour_count} contour layers for class {class_idx}")
            processed_count += 1

    return mask

def resample_to_isotropic(sitk_image, is_mask=False, new_spacing=(1.0, 1.0, 1.0)):
    """
    Resamples a 3D SimpleITK image to a standardized, isotropic voxel spacing.
    Uses B-Spline interpolation for structural intensity images, and Nearest Neighbor
    interpolation for integer label masks to preserve discrete boundaries.
    """
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    
    # If the image is already at the target spacing, return it unchanged
    if np.allclose(original_spacing, new_spacing):
        return sitk_image
        
    # Compute the new matrix grid size to maintain the physical field of view
    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    
    if is_mask:
        # Crucial: Nearest Neighbor prevents blending integer class IDs (e.g., creating a fake class 1.5)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetOutputPixelType(sitk.sitkUInt8)
    else:
        # B-Spline provides smooth structural interpolation for CT Hounsfield intensities
        resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetOutputPixelType(sitk_image.GetPixelID())
        
    return resample.Execute(sitk_image)

def run_single_test(dicom_dir, rtstruct_path, output_img_path, output_mask_path):
    logger.info("Starting isolated cardiac data preparation test with series filtering...")
    
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    
    # 1. Read the RTSTRUCT to find out exactly which CT series it belongs to
    rtstruct_ds = pydicom.dcmread(rtstruct_path)
    target_series_uid = None
    
    try:
        # Travel down the DICOM tree to find the referenced series UID
        if hasattr(rtstruct_ds, 'ReferencedSeriesSequence'):
            target_series_uid = rtstruct_ds.ReferencedSeriesSequence[0].SeriesInstanceUID
        elif hasattr(rtstruct_ds, 'ReferencedFrameOfReferenceSequence'):
            ref_seq = rtstruct_ds.ReferencedFrameOfReferenceSequence[0]
            if hasattr(ref_seq, 'RTReferencedStudySequence'):
                study_seq = ref_seq.RTReferencedStudySequence[0]
                if hasattr(study_seq, 'RTReferencedSeriesSequence'):
                    target_series_uid = study_seq.RTReferencedSeriesSequence[0].SeriesInstanceUID
    except Exception as e:
        logger.warning(f"Could not extract target SeriesInstanceUID from headers: {e}")

    # 2. Configure SimpleITK to selectively load ONLY the matching series ID
    reader = sitk.ImageSeriesReader()
    
    # Find all available series IDs in this directory
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise FileNotFoundError(f"No valid DICOM series found in {dicom_dir}")
        
    logger.info(f"Available Series IDs in directory: {len(series_ids)}")
    
    # If we found a target reference match, prioritize it. Otherwise default to the first one.
    selected_series = series_ids[0]
    if target_series_uid and target_series_uid in series_ids:
        selected_series = target_series_uid
        logger.info(f"✓ Successfully matched RTSTRUCT reference to Series ID: {selected_series}")
    else:
        logger.warning(f"Could not find exact reference match. Defaulting to first available series.")

    # Get file list for just this clean series group
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir, selected_series)
    reader.SetFileNames(dicom_files)
    dcm_image = reader.Execute()
    
    logger.info(f"✓ Loaded Clean DICOM Image Volume Shape: {dcm_image.GetSize()} | Spacing: {dcm_image.GetSpacing()}")
    
    # 3. Extract and Burn contours
    contours_dict = robust_extract_contours(rtstruct_path)
    multiclass_mask_array = custom_contours_to_multiclass_mask(contours_dict, dcm_image)
    
    logger.info(f"Generated Mask Total Non-Zero Voxels: {np.count_nonzero(multiclass_mask_array)}")
    
    # 4. Save outputs safely with matching orientations
    logger.info(f"Saving CT volume to: {output_img_path}")
    sitk.WriteImage(dcm_image, output_img_path)
    
    mask_image = sitk.GetImageFromArray(multiclass_mask_array)
    mask_image.CopyInformation(dcm_image)  # This will not crash anymore!

    # --- NEW: Resample both arrays to standardized 1.0mm isotropic grids ---
    logger.info("Resampling volume and label mask to 1.0mm isotropic spacing...")
    resampled_dcm = resample_to_isotropic(dcm_image, is_mask=False)
    resampled_mask = resample_to_isotropic(mask_image, is_mask=True)
    
    # Save the beautifully standardized outputs
    logger.info(f"Saving isotropic CT volume to: {output_img_path}")
    sitk.WriteImage(resampled_dcm, output_img_path)
    
    logger.info(f"Saving isotropic multi-class mask to: {output_mask_path}")
    sitk.WriteImage(resampled_mask, output_mask_path)
    


# ... [Your existing run_single_test and resample_to_isotropic functions are up here] ...

def process_all_tttpatients(base_raw_dir, output_images_dir, output_labels_dir):
    """
    Loops through all patient directories, isolates the CT series and RTSTRUCT,
    and generates pairs of isotropic images and masks.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(base_raw_dir, "*")) 
        if os.path.isdir(d)
    ])
    
    logger.info(f"Found {len(patient_dirs)} patient folders to process.")
    
    success_count = 0
    for idx, patient_dir in enumerate(patient_dirs):
        patient_name = os.path.basename(patient_dir)
        logger.info(f"\n" + "="*60)
        logger.info(f"[{idx+1}/{len(patient_dirs)}] Processing: {patient_name}")
        logger.info("="*60)
        
        rtstruct_patterns = [
            os.path.join(patient_dir, "RS*"),
            os.path.join(patient_dir, "rs*"),
            os.path.join(patient_dir, "*RS*.dcm"),
            os.path.join(patient_dir, "*rs*.dcm")
        ]
        
        rtstruct_path = None
        for pattern in rtstruct_patterns:
            matches = glob.glob(pattern)
            if matches:
                rtstruct_path = [m for m in matches if os.path.isfile(m)][0]
                break
                
        if not rtstruct_path:
            logger.error(f"❌ Skipped {patient_name}: No RTSTRUCT file found.")
            continue
            
        # case_id = f"case_{idx:04d}"  
        # out_img_path = os.path.join(output_images_dir, f"{case_id}_0000.nii.gz")
        # out_mask_path = os.path.join(output_labels_dir, f"{case_id}.nii.gz")

        # NEW TRACEABLE LOGIC:
        out_img_path = os.path.join(output_images_dir, f"{patient_name}_0000.nii.gz")
        out_mask_path = os.path.join(output_labels_dir, f"{patient_name}.nii.gz")
        
        try:
            run_single_test(
                dicom_dir=patient_dir,
                rtstruct_path=rtstruct_path,
                output_img_path=out_img_path,
                output_mask_path=out_mask_path
            )
            success_count += 1
            logger.info(f"✓ Successfully converted and saved {case_id}")
        except Exception as e:
            logger.error(f"❌ Failed processing {patient_name}: {e}")
            continue

    logger.info(f"\nPipeline complete! Successfully processed {success_count}/{len(patient_dirs)} cases.")


def process_all_patients(base_raw_dir, output_images_dir, output_labels_dir):
    """
    Loops through all patient directories, isolates the CT series and RTSTRUCT,
    and generates pairs of isotropic images and masks. Skips already processed folders.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(base_raw_dir, "*")) 
        if os.path.isdir(d)
    ])
    
    logger.info(f"Found {len(patient_dirs)} total patient folders in raw directory.")
    
    success_count = 0
    skipped_count = 0
    
    for idx, patient_dir in enumerate(patient_dirs):
        patient_name = os.path.basename(patient_dir)
        
        # 1. Define standard output filenames using the patient name
        out_img_path = os.path.join(output_images_dir, f"{patient_name}_0000.nii.gz")
        out_mask_path = os.path.join(output_labels_dir, f"{patient_name}.nii.gz")
        
        # 2. INCREMENTAL CHECK: Skip if files already exist
        if os.path.exists(out_img_path) and os.path.exists(out_mask_path):
            logger.info(f"-> [{idx+1}/{len(patient_dirs)}] Skipping {patient_name} (Already processed)")
            skipped_count += 1
            continue
            
        logger.info(f"\n" + "="*60)
        logger.info(f"[{idx+1}/{len(patient_dirs)}] Processing New Case: {patient_name}")
        logger.info("="*60)
        
        # 3. Locate the RTSTRUCT file within this specific folder
        rtstruct_patterns = [
            os.path.join(patient_dir, "RS*"),
            os.path.join(patient_dir, "rs*"),
            os.path.join(patient_dir, "*RS*.dcm"),
            os.path.join(patient_dir, "*rs*.dcm")
        ]
        
        rtstruct_path = None
        for pattern in rtstruct_patterns:
            matches = glob.glob(pattern)
            if matches:
                rtstruct_path = [m for m in matches if os.path.isfile(m)][0]
                break
                
        if not rtstruct_path:
            logger.error(f"❌ Skipped {patient_name}: No RTSTRUCT file found.")
            continue
        
        # 4. Run the isolated pipeline safely
        try:
            run_single_test(
                dicom_dir=patient_dir,
                rtstruct_path=rtstruct_path,
                output_img_path=out_img_path,
                output_mask_path=out_mask_path
            )
            success_count += 1
            logger.info(f"✓ Successfully converted and saved {patient_name}")
        except Exception as e:
            logger.error(f"❌ Failed processing {patient_name}: {e}")
            continue

    logger.info(f"\n" + "="*60)
    logger.info("PIPELINE SUMMARY:")
    logger.info(f"   • Total Folders Checked: {len(patient_dirs)}")
    logger.info(f"   • Already Processed (Skipped): {skipped_count}")
    logger.info(f"   • Newly Processed Successfully: {success_count}")
    logger.info(f"   • Failed This Run: {len(patient_dirs) - skipped_count - success_count}")
    logger.info("="*60)

if __name__ == "__main__":
    RAW_DATA_DIR = "/home/abishek/projects/cardiac_segmentation/data/raw/"
    PROCESSED_IMAGES_DIR = "./data/processed/images"
    PROCESSED_LABELS_DIR = "./data/processed/labels"
    
    process_all_patients(
        base_raw_dir=RAW_DATA_DIR,
        output_images_dir=PROCESSED_IMAGES_DIR,
        output_labels_dir=PROCESSED_LABELS_DIR
    )
# if __name__ == "__main__":
#     DICOM_DIR = "/home/abishek/projects/cardiac_segmentation/data/raw/VCU_Lung_172/"


#     # 1. Look specifically for files starting with 'RS' or containing 'RS'
#     rtstruct_patterns = [
#         os.path.join(DICOM_DIR, "RS*"),
#         os.path.join(DICOM_DIR, "rs*"),
#         os.path.join(DICOM_DIR, "*RS*.dcm"),
#         os.path.join(DICOM_DIR, "*rs*.dcm")
#     ]
    
#     RTSTRUCT_PATH = None
#     for pattern in rtstruct_patterns:
#         matches = glob.glob(pattern)
#         if matches:
#             # Pick the first valid file match
#             RTSTRUCT_PATH = [m for m in matches if os.path.isfile(m)][0]
#             logger.info(f"✓ Found RTSTRUCT file via RS filename pattern: {os.path.basename(RTSTRUCT_PATH)}")
#             break

#     # 2. Hard fallback if the pattern search completely fails
#     if not RTSTRUCT_PATH:
#         logger.warning("RS pattern match failed. Falling back to global file scan...")
#         all_files = [f for f in glob.glob(os.path.join(DICOM_DIR, "*")) if os.path.isfile(f)]
#         if all_files:
#             sizes = [os.path.getsize(f) for f in all_files]
#             RTSTRUCT_PATH = all_files[int(np.argmax([abs(s - np.median(sizes)) for s in sizes]))]
#             logger.info(f"⚠ Fallback selected file by size variance: {os.path.basename(RTSTRUCT_PATH)}")

#     if not RTSTRUCT_PATH:
#         raise FileNotFoundError(f"Could not find any files inside {DICOM_DIR}")

#     run_single_test(
#         dicom_dir=DICOM_DIR,
#         rtstruct_path=RTSTRUCT_PATH,
#         output_img_path="./data/processed/images/test_case_0002.nii.gz",
#         output_mask_path="./data/processed/labels/test_case_0002.nii.gz"
#     )
    
    
