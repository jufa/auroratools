import os
import cv2
import numpy as np
import argparse

class ColorCorrector:
    """
    Adjust Red and Blue channels to match green channel average in sample black area of frame (top left).
    Allows positive (lift) and negative (cut) offsets at black, keeping maximum at 1.0.
    """

    def __init__(self, red_offset=0.0):
        """
        red_offset: linear y-intercept for black (-0.05 to 0.05 typical)
        """
        self.red_offset = red_offset
        if not (-0.5 <= red_offset <= 0.5):
            raise ValueError("red_offset should be in -0.5 to 0.5 range")

    @staticmethod
    def srgb_to_linear(img):
        return img
        return np.where(img <= 0.04045, img / 12.92, ((img + 0.055)/1.055)**2.4)

    @staticmethod
    def linear_to_srgb(img):
        return img
        return np.where(img <= 0.0031308, img*12.92, 1.055*(img**(1/2.4))-0.055)
    
    def sample_black_corner(self, img):
        # Sample a 16x16 corner patch (assumes top-left is in shadow)
        sample_size = 300
        corner = img[:sample_size, :sample_size]
        print(f"sampled corner mean (per channel): {corner.mean(axis=(0,1))}")
        return corner.mean(axis=(0,1))  # mean per channel, BGR order
    

    def apply(self, img_path, out_path=None):
        img = cv2.imread(img_path).astype(np.float32)/255.0
        # sample_black_corner = self.sample_black_corner(img)
        img_lin = self.srgb_to_linear(img)
        sample_black_corner = self.sample_black_corner(img_lin)
        b = 0
        g = 1
        r = 2
        correction_red =  -sample_black_corner[r]
        correction_blue = -sample_black_corner[b]
        correction_green = -sample_black_corner[g]
        # Apply linear curve: R_out = (1-b)*R_in + b
        slope = correction_red
        m = 1.0 - slope
        img_lin[:,:,r] = img_lin[:,:,r] * m + slope  # Red channel (OpenCV BGR)

        slope = correction_blue
        m = 1.0 - slope
        img_lin[:,:,b] = img_lin[:,:,b] * m + slope  # Red channel (OpenCV BGR)

        slope = correction_green
        m = 1.0 - slope
        img_lin[:,:,g] = img_lin[:,:,g] * m + slope  # Red channel (OpenCV BGR)

        # Clip to [0,1]
        img_lin = np.clip(img_lin, 0.0, 1.0)

        # Back to sRGB
        img_srgb = self.linear_to_srgb(img_lin)
        img_out = np.clip(img_srgb*255,0,255).astype(np.uint8)

        cv2.imwrite(out_path, img_out)

# ---------- CLI ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Adjust Red and Blue channel y-intercept for shadows in a Sony JPEG to remove colour cast"
    )
    parser.add_argument('--folder', required=True, help='Path to the JPEG images')
    args = parser.parse_args()

    shifter = ColorCorrector()
    output_folder = os.path.join(args.folder, "WBFixed")
    if not os.path.exists(output_folder):
      os.mkdir(output_folder)
    files = os.listdir(args.folder)
    print(f"Found {len(files)} files in {args.folder}, processing JPEGs...")
    
    jpgs = [f for f in files if f.lower().endswith('.jpg')]
    for jpg in jpgs:
      image_path = os.path.join(args.folder, jpg)
      out_path = os.path.join(output_folder, jpg)
      print(f"Processing {image_path}...")
      corrected_path = shifter.apply(image_path, out_path=out_path)
      print(f"Corrected image saved to: {output_folder}")
