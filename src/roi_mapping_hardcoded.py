import re

# Dictionary mapping target clean classes to regular expression patterns
ROI_PATTERNS = {
    1: r'\b(lv|ventricle[_\s]?l(eft)?)\b',                         # Matches: 'Ventricle Left', 'Ventricle_L', 'LV'
    2: r'\b(rv|ventricle[_\s]?r(ight)?)\b',                        # Matches: 'Ventricle Right', 'Ventricle_R', 'RV'
    3: r'\b(la|atrium[_\s]?l(eft)?|attrium[_\s]?left)\b',          # Matches: 'Atrium_L', 'Attrium Left', 'LA'
    4: r'\b(ra|atrium[_\s]?r(ight)?|attrium[_\s]?right)\b',        # Matches: 'Atrium_R', 'Attrium Right', 'RA'
}

# Keep track of whole heart separately if needed for cropping/bounding boxes
WHOLE_HEART_PATTERN = r'\b(heart|0\s?heart)\b'

def clean_roi_name(name: str) -> str:
    """Standardizes string by lowering case and stripping leading/trailing garbage."""
    if not name:
        return ""
    name = str(name).lower().strip()
    # Strip leading '0 ' if present (e.g., '0 heart' -> 'heart')
    name = re.sub(r'^0\s+', '', name)
    return name

def get_class_label(roi_name: str) -> int:
    """
    Evaluates an ROI name and returns its integer class (1-4).
    Returns 0 if it should be ignored or is part of background.
    """
    cleaned = clean_roi_name(roi_name)
    
    # Exclusion check: Ignore plan variants, target volumes, and non-cardiac structures
    if cleaned.startswith('z') or any(x in cleaned for x in ['ptv', 'ctv', 'gtv', 'itv', 'cord', 'spinal']):
        return 0
        
    # Match against our cardiac substructure rules
    for class_idx, pattern in ROI_PATTERNS.items():
        if re.search(pattern, cleaned):
            return class_idx
            
    return 0 # Treat as background if no match
