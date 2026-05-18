import os
import argparse
import logging
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Cardiac4PanelViewer:
    def __init__(self, ct_path, mask_path):
        logger.info(f"Loading CT Volume: {ct_path}")
        self.ct_img = sitk.ReadImage(ct_path)

        logger.info(f"Loading Multi-Class Mask: {mask_path}")
        self.mask_img = sitk.ReadImage(mask_path)
        
        # Convert SimpleITK images to NumPy arrays (Z, Y, X)
        self.ct_arr = sitk.GetArrayFromImage(self.ct_img)
        self.mask_arr = sitk.GetArrayFromImage(self.mask_img)

        # Convert SimpleITK images to NumPy arrays (Z, Y, X)
        self.ct_arr = sitk.GetArrayFromImage(self.ct_img)
        self.mask_arr = sitk.GetArrayFromImage(self.mask_img)
        
        # FOR DEBUGGING:
        print(f"DEBUG: Mask Array Shape: {self.mask_arr.shape}")
        print(f"DEBUG: Mask Array Sum:   {np.sum(self.mask_arr)}")
        print(f"DEBUG: Max Value in Mask: {np.max(self.mask_arr)}")

        
        # Check mask properties
        unique_labels = np.unique(self.mask_arr)
        logger.info(f"Unique labels found in mask: {unique_labels}")
        
        # Clip CT intensity for high soft-tissue contrast windowing (-175 to 250 HU)
        self.ct_arr = np.clip(self.ct_arr, -175, 250)
        
        self.z_dim, self.y_dim, self.x_dim = self.ct_arr.shape
        
        # Start crosshairs perfectly dead-center in the 3D volume
        self.current_point = {
            'z': self.z_dim // 2,  # Axial index
            'y': self.y_dim // 2,  # Coronal index
            'x': self.x_dim // 2   # Sagittal index
        }
        
        # Define an explicit, vivid color map for the 4 heart chambers
        # 0: transparent, 1(LV): Red, 2(RV): Blue, 3(LA): Green, 4(RA): Yellow
        colors = [
            [0.0, 0.0, 0.0, 0.0],  # 0: Background transparent
            [1.0, 0.2, 0.2, 0.5],  # 1: Left Ventricle (Red, 50% opacity)
            [0.2, 0.4, 1.0, 0.5],  # 2: Right Ventricle (Blue, 50% opacity)
            [0.2, 0.9, 0.2, 0.5],  # 3: Left Atrium (Green, 50% opacity)
            [0.9, 0.9, 0.1, 0.5]   # 4: Right Atrium (Yellow, 50% opacity)
        ]
        self.cmap = ListedColormap(colors)
        
        # Create a 2x2 grid panel layout
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.canvas.manager.set_window_title(f"Synchronized Cardiac QA: {os.path.basename(ct_path)}")
        
        self.ax_axial = self.axes[0, 0]
        self.ax_coronal = self.axes[0, 1]
        self.ax_sagittal = self.axes[1, 0]
        self.ax_info = self.axes[1, 1]  # 4th panel used for stats/metadata
        
        self.update_plots()
        
        # Link mouse scroll wheel events across the canvas
        self.fig.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)
        
        print("\n" + "="*60)
        print("  SYNCHRONIZED SCROLL INTERFACE READY:")
        print("  - Hover mouse over ANY window and scroll.")
        print("  - Slices will shift uniformly across all orthogonal views.")
        print("="*60 + "\n")
        
        plt.tight_layout()
        plt.show()

    def update_plots(self):
        """Re-renders all views dynamically keeping coordinates locked in sync."""
        z, y, x = self.current_point['z'], self.current_point['y'], self.current_point['x']
        
        # Clear previous frame renders
        for ax in self.axes.ravel():
            ax.clear()
            
        # Create masked arrays so background (0) doesn't render or alter the colormap scaling
        mask_z = np.ma.masked_where(self.mask_arr[z, :, :] == 0, self.mask_arr[z, :, :])
        mask_y = np.ma.masked_where(self.mask_arr[:, y, :] == 0, self.mask_arr[:, y, :])
        mask_x = np.ma.masked_where(self.mask_arr[:, :, x] == 0, self.mask_arr[:, :, x])
        
        # --------------------------------------------------------------------
        # 1. AXIAL PANEL (Z-Slice)
        # --------------------------------------------------------------------
        # Removed origin='lower' to fix the upside-down inversion
        self.ax_axial.imshow(self.ct_arr[z, :, :], cmap='gray')
        self.ax_axial.imshow(mask_z, cmap=self.cmap, vmin=0, vmax=4)
        self.ax_axial.axhline(y, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_axial.axvline(x, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_axial.set_title(f"Axial View (Z: {z}/{self.z_dim-1})")
        self.ax_axial.axis('off')
        
        # --------------------------------------------------------------------
        # 2. CORONAL PANEL (Y-Slice)
        # --------------------------------------------------------------------
        self.ax_coronal.imshow(self.ct_arr[:, y, :], cmap='gray', origin='lower', aspect='auto')
        self.ax_coronal.imshow(mask_y, cmap=self.cmap, vmin=0, vmax=4, origin='lower', aspect='auto')
        self.ax_coronal.axhline(z, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_coronal.axvline(x, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_coronal.set_title(f"Coronal View (Y: {y}/{self.y_dim-1})")
        self.ax_coronal.axis('off')
        
        # --------------------------------------------------------------------
        # 3. SAGITTAL PANEL (X-Slice)
        # --------------------------------------------------------------------
        self.ax_sagittal.imshow(self.ct_arr[:, :, x], cmap='gray', origin='lower', aspect='auto')
        self.ax_sagittal.imshow(mask_x, cmap=self.cmap, vmin=0, vmax=4, origin='lower', aspect='auto')
        self.ax_sagittal.axhline(z, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_sagittal.axvline(y, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_sagittal.set_title(f"Sagittal View (X: {x}/{self.x_dim-1})")
        self.ax_sagittal.axis('off')
        
        # --------------------------------------------------------------------
        # 4. METADATA INFO PANEL (4th Window)
        # --------------------------------------------------------------------
        self.ax_info.axis('off')
        info_text = (
            f"CARDIAC QA SNAPSHOT\n\n"
            f"Current 3D Voxel Coord:\n"
            f"  X: {x} | Y: {y} | Z: {z}\n\n"
            f"Volume Array Resolution:\n"
            f"  {self.ct_arr.shape} (Z, Y, X)\n\n"
            f"Chamber Color Mapping:\n"
            f"  ■ Class 1: Left Ventricle (Red)\n"
            f"  ■ Class 2: Right Ventricle (Blue)\n"
            f"  ■ Class 3: Left Atrium (Green)\n"
            f"  ■ Class 4: Right Atrium (Yellow)\n\n"
            f"Voxel Counts:\n"
            f"  Total Mask Voxels: {np.count_nonzero(self.mask_arr)}"
        )
        self.ax_info.text(0.1, 0.9, info_text, fontsize=11, family='monospace',
                           verticalalignment='top', transform=self.ax_info.transAxes)
        self.ax_info.set_title("Dataset Summary")

        self.fig.canvas.draw_idle()

    def update_tttplots(self):
        """Re-renders all views dynamically keeping coordinates locked in sync."""
        z, y, x = self.current_point['z'], self.current_point['y'], self.current_point['x']
        
        # Clear previous frame renders
        for ax in self.axes.ravel():
            ax.clear()
            
        # 1. Extract raw 2D slices for this specific 3D coordinate point
        ct_z = self.ct_arr[z, :, :]
        lbl_z = self.mask_arr[z, :, :]
        
        ct_y = self.ct_arr[:, y, :]
        lbl_y = self.mask_arr[:, y, :]
        
        ct_x = self.ct_arr[:, :, x]
        lbl_x = self.mask_arr[:, :, x]
        
        # 2. Create explicit overlay masks where label > 0 (ignoring background)
        # This guarantees Matplotlib won't scale or crush the low integer values
        mask_z = np.ma.masked_where(lbl_z == 0, lbl_z)
        mask_y = np.ma.masked_where(lbl_y == 0, lbl_y)
        mask_x = np.ma.masked_where(lbl_x == 0, lbl_x)
        
        # --------------------------------------------------------------------
        # 1. AXIAL PANEL (Z-Slice)
        # --------------------------------------------------------------------
        self.ax_axial.imshow(ct_z, cmap='gray')
        if np.any(lbl_z > 0):  # Only render if target pixels exist on this slice layer
            self.ax_axial.imshow(mask_z, cmap=self.cmap, vmin=0.5, vmax=4.5, interpolation='nearest')
            
        self.ax_axial.axhline(y, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_axial.axvline(x, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_axial.set_title(f"Axial View (Z: {z}/{self.z_dim-1})")
        self.ax_axial.axis('off')
        
        # --------------------------------------------------------------------
        # 2. CORONAL PANEL (Y-Slice)
        # --------------------------------------------------------------------
        self.ax_coronal.imshow(ct_y, cmap='gray', origin='lower', aspect='auto')
        if np.any(lbl_y > 0):
            self.ax_coronal.imshow(mask_y, cmap=self.cmap, vmin=0.5, vmax=4.5, origin='lower', aspect='auto', interpolation='nearest')
            
        self.ax_coronal.axhline(z, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_coronal.axvline(x, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_coronal.set_title(f"Coronal View (Y: {y}/{self.y_dim-1})")
        self.ax_coronal.axis('off')
        
        # --------------------------------------------------------------------
        # 3. SAGITTAL PANEL (X-Slice)
        # --------------------------------------------------------------------
        self.ax_sagittal.imshow(ct_x, cmap='gray', origin='lower', aspect='auto')
        if np.any(lbl_x > 0):
            self.ax_sagittal.imshow(mask_x, cmap=self.cmap, vmin=0.5, vmax=4.5, origin='lower', aspect='auto', interpolation='nearest')
            
        self.ax_sagittal.axhline(z, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_sagittal.axvline(y, color='cyan', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_sagittal.set_title(f"Sagittal View (X: {x}/{self.x_dim-1})")
        self.ax_sagittal.axis('off')
        
        # --------------------------------------------------------------------
        # 4. METADATA INFO PANEL (4th Window)
        # --------------------------------------------------------------------
        self.ax_info.axis('off')
        info_text = (
            f"CARDIAC QA SNAPSHOT\n\n"
            f"Current 3D Voxel Coord:\n"
            f"  X: {x} | Y: {y} | Z: {z}\n\n"
            f"Volume Array Resolution:\n"
            f"  {self.ct_arr.shape} (Z, Y, X)\n\n"
            f"Chamber Color Mapping:\n"
            f"  ■ Class 1: Left Ventricle (Red)\n"
            f"  ■ Class 2: Right Ventricle (Blue)\n"
            f"  ■ Class 3: Left Atrium (Green)\n"
            f"  ■ Class 4: Right Atrium (Yellow)\n\n"
            f"Voxel Counts:\n"
            f"  Total Mask Voxels: {np.count_nonzero(self.mask_arr)}\n"
            f"  Active Slice Voxels: {np.count_nonzero(lbl_z)}"
        )
        self.ax_info.text(0.1, 0.9, info_text, fontsize=11, family='monospace',
                           verticalalignment='top', transform=self.ax_info.transAxes)
        self.ax_info.set_title("Dataset Summary")

        self.fig.canvas.draw_idle()

    def on_mouse_scroll(self, event):
        """Catches scroll events in ANY panel and moves the central point symmetrically."""
        if event.button is None:
            return
            
        direction = 1 if event.button == 'up' else -1
        
        # Change the specific coordinate component based on which window the mouse is hovering over
        if event.inaxes == self.ax_axial:
            self.current_point['z'] = np.clip(self.current_point['z'] + direction, 0, self.z_dim - 1)
        elif event.inaxes == self.ax_coronal:
            self.current_point['y'] = np.clip(self.current_point['y'] + direction, 0, self.y_dim - 1)
        elif event.inaxes == self.ax_sagittal:
            self.current_point['x'] = np.clip(self.current_point['x'] + direction, 0, self.x_dim - 1)
        else:
            return # Scrolled out of bounds
            
        self.update_plots()

def main():
    parser = argparse.ArgumentParser(description="Synchronized 4-Panel Cardiac Mask Viewer.")
    parser.add_argument('--ct', type=str, required=True, help="Path to NIfTI CT image file")
    parser.add_argument('--mask', type=str, required=True, help="Path to NIfTI mask file")
    args = parser.parse_args()
    
    Cardiac4PanelViewer(args.ct, args.mask)

if __name__ == '__main__':
    main()
