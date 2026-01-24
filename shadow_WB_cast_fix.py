import os
import cv2
import numpy as np
import argparse

class RedChannelCurveShift:
    """
    Adjust Red channel linear curve (slope + y-intercept) in linear-light.
    Allows positive (lift) and negative (cut) offsets at black, keeping maximum at 1.0.

    Currently this is a non working experiment - the match looks very off. need more empirical tuning
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
        return np.where(img <= 0.04045, img / 12.92, ((img + 0.055)/1.055)**2.4)

    @staticmethod
    def linear_to_srgb(img):
        return np.where(img <= 0.0031308, img*12.92, 1.055*(img**(1/2.4))-0.055)

    def apply(self, img_path, out_path=None):
        img = cv2.imread(img_path).astype(np.float32)/255.0
        img_lin = self.srgb_to_linear(img)

        # Apply linear red curve: R_out = (1-b)*R_in + b
        b = self.red_offset
        m = 1.0 - b
        img_lin[:,:,2] = img_lin[:,:,2] * m + b  # Red channel (OpenCV BGR)

        # Clip to [0,1]
        img_lin = np.clip(img_lin, 0.0, 1.0)

        # Back to sRGB
        img_srgb = self.linear_to_srgb(img_lin)
        img_out = np.clip(img_srgb*255,0,255).astype(np.uint8)

        if out_path is None:
            base, ext = os.path.splitext(os.path.basename(img_path))
            out_path = os.path.join(os.path.dirname(img_path), f'{base}_corrected{ext}')

        if os.path.exists(out_path):
          os.remove(out_path)
        cv2.imwrite(out_path, img_out)
        return out_path

# ---------- CLI ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Adjust Red channel linear curve y-intercept for shadows in a Sony JPEG"
    )
    parser.add_argument('--image', required=True, help='Path to the JPEG image')
    parser.add_argument('--red-offset', type=float, default=0.0,
                        help='Red channel linear y-intercept (positive lift, negative cut, typical -0.005 to 0.005)')
    args = parser.parse_args()

    shifter = RedChannelCurveShift(red_offset=args.red_offset)
    corrected_path = shifter.apply(args.image)
    print(f"Corrected image saved to: {corrected_path}")
