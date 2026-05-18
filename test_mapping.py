from src.roi_mapping import get_class_label

sample_names = [
    '0 Heart', 'Atrium_L', 'Atrium_R', 'Attrium Left', 'Attrium Right',
    'Ventricle Left', 'Ventricle Right', 'Ventricle_L', 'Ventricle_R',
    'zEsophagus_AV', 'PTV_6300', 'SpinalCord', '0 SpinalCord_PRV'
]

print("Testing ROI Class Mapping Logic:")
print("-" * 50)
for name in sample_names:
    label = get_class_label(name)
    status = f"-> Class {label}" if label > 0 else "-> IGNORED (0)"
    print(f"'{name}': {status}")
