import pandas as pd

class ROIMapper:
    def __init__(self, excel_path: str):
        # 1. Read with header=0 (uses the first row as column names)
        df = pd.read_excel(excel_path, header=0)
        
        # 2. Clean up column names in case there are extra spaces
        df.columns = [c.strip() for c in df.columns]
        
        # 3. Drop rows where 'Standard Name' is empty
        df = df.dropna(subset=['Standard Name'])
        
        self.lookup = {}
        
        # 4. Map 'Standard Name' to unique IDs
        unique_standards = df['Standard Name'].unique()
        name_to_id = {str(name).lower().strip(): i + 1 for i, name in enumerate(unique_standards)}
        
        # 5. Build the lookup
        for _, row in df.iterrows():
            std = str(row['Standard Name']).lower().strip()
            class_id = name_to_id[std]
            
            self.lookup[std] = class_id
            
            # Handle 'Alternative Names' column
            alt_raw = row['Alternative Names']
            if pd.notna(alt_raw):
                aliases = str(alt_raw).split(',')
                for alias in aliases:
                    self.lookup[alias.lower().strip()] = class_id

    def get_class_label(self, roi_name: str) -> int:
        cleaned = str(roi_name).lower().strip().replace('0 ', '', 1)
        
        # Exclusion logic
        if cleaned.startswith('z') or any(x in cleaned for x in ['ptv', 'ctv', 'gtv', 'itv', 'cord', 'spinal']):
            return 0
            
        return self.lookup.get(cleaned, 0)

# Initialize
mapper = ROIMapper('./data/roi_mapping.xlsx')

def get_class_label(roi_name: str) -> int:
    return mapper.get_class_label(roi_name)