import argparse
import logging
import json
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Cardiac4PanelViewer:
    def __init__(self, ct_path, mask_path=None, json_path=None):
        self.ct_img = sitk.ReadImage(ct_path)
        self.ct_arr = np.clip(sitk.GetArrayFromImage(self.ct_img), -175, 250)
        
        self.label_map = self.load_label_mapping(json_path)
        
        self.mask_arr = None
        self.unique_labels = []
        if mask_path:
            self.mask_img = sitk.ReadImage(mask_path)
            self.mask_arr = sitk.GetArrayFromImage(self.mask_img)
            self.unique_labels = sorted([int(l) for l in np.unique(self.mask_arr) if l > 0])
            
            base_cmap = plt.get_cmap('nipy_spectral', len(self.unique_labels) + 1)
            cmap_colors = [base_cmap(i) for i in range(len(self.unique_labels) + 1)]
            cmap_colors[0] = (0.0, 0.0, 0.0, 0.0) 
            self.cmap = ListedColormap(cmap_colors)
        
        self.z_dim, self.y_dim, self.x_dim = self.ct_arr.shape
        self.current_point = {'z': self.z_dim // 2, 'y': self.y_dim // 2, 'x': self.x_dim // 2}
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        # Explicitly assign these so on_scroll/on_click can find them
        self.ax_axial = self.axes[0, 0]
        self.ax_coronal = self.axes[0, 1]
        self.ax_sagittal = self.axes[1, 0]
        self.ax_info = self.axes[1, 1]
        
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.update_plots()
        plt.tight_layout()
        plt.show()

    def load_label_mapping(self, json_path):
        if not json_path or not os.path.exists(json_path):
            return {}
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data.get("labels", {})
        except Exception as e:
            logger.error(f"Could not load JSON: {e}")
            return {}

    def get_label_name(self, lid):
        # Fallback to "Class X" if no JSON mapping exists
        return self.label_map.get(str(lid), f"Class {lid}")

    def update_plots(self):
        z, y, x = self.current_point['z'], self.current_point['y'], self.current_point['x']
        
        views = [
            (self.ax_axial, self.ct_arr[z,:,:], (self.mask_arr[z,:,:] if self.mask_arr is not None else None), f"Axial (Z: {z})", x, y, False),
            (self.ax_coronal, self.ct_arr[:,y,:], (self.mask_arr[:,y,:] if self.mask_arr is not None else None), f"Coronal (Y: {y})", x, z, True),
            (self.ax_sagittal, self.ct_arr[:,:,x], (self.mask_arr[:,:,x] if self.mask_arr is not None else None), f"Sagittal (X: {x})", y, z, True)
        ]
        
        for ax, img, mask, title, h, v, origin_low in views:
            ax.clear()
            ax.imshow(img, cmap='gray', origin='lower' if origin_low else 'upper', aspect='equal')
            if mask is not None:
                m = np.ma.masked_where(mask == 0, mask)
                ax.imshow(m, cmap=self.cmap, vmin=0, vmax=len(self.unique_labels), 
                          origin='lower' if origin_low else 'upper', aspect='equal', interpolation='nearest')
            ax.axhline(v, color='cyan', alpha=0.5, lw=1)
            ax.axvline(h, color='cyan', alpha=0.5, lw=1)
            ax.set_title(title)
            ax.axis('off')
            
        self.ax_info.clear()
        self.ax_info.set_title("Dataset Summary")
        self.ax_info.text(0.05, 0.95, f"Coord (Z,Y,X): {z}, {y}, {x}", va='top', family='monospace')
        
        if self.mask_arr is not None:
            patches = [mpatches.Patch(color=self.cmap(i+1), label=self.get_label_name(lid)) 
                       for i, lid in enumerate(self.unique_labels)]
            self.ax_info.legend(handles=patches, loc='upper left', bbox_to_anchor=(0.05, 0.90), title="Mask Legend")
        
        self.ax_info.axis('off')
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            if event.inaxes == self.ax_axial:
                self.current_point['y'], self.current_point['x'] = int(event.ydata), int(event.xdata)
            elif event.inaxes == self.ax_coronal:
                self.current_point['z'], self.current_point['x'] = int(event.ydata), int(event.xdata)
            elif event.inaxes == self.ax_sagittal:
                self.current_point['z'], self.current_point['y'] = int(event.ydata), int(event.xdata)
            self.update_plots()

    def on_scroll(self, event):
        if event.inaxes == self.ax_axial: self.current_point['z'] += 1 if event.button == 'up' else -1
        elif event.inaxes == self.ax_coronal: self.current_point['y'] += 1 if event.button == 'up' else -1
        elif event.inaxes == self.ax_sagittal: self.current_point['x'] += 1 if event.button == 'up' else -1
        else: return
        self.current_point['z'] = np.clip(self.current_point['z'], 0, self.z_dim-1)
        self.current_point['y'] = np.clip(self.current_point['y'], 0, self.y_dim-1)
        self.current_point['x'] = np.clip(self.current_point['x'], 0, self.x_dim-1)
        self.update_plots()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ct', required=True)
    parser.add_argument('--mask', required=False)
    parser.add_argument('--json', required=False)
    args = parser.parse_args()
    Cardiac4PanelViewer(args.ct, args.mask, args.json)

if __name__ == '__main__':
    main()