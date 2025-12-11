#!/usr/bin/env python3
import cv2
import argparse
from pathlib import Path
import numpy as np

"""
usage:
python diffimages.py --path "/Volumes/T7 Drive/sequence/" --start 1 --end 20 --contrast 1.8

output:
/Volumes/T7 Drive/sequence/diff/00000002-00000001.png
/Volumes/T7 Drive/sequence/diff/00000003-00000002.png
"""

def parse_args():
    p = argparse.ArgumentParser(description="Compute sequential image differences.")
    p.add_argument("--path", required=True,
                  help="Folder containing sequential images like 00000001.jpg")
    p.add_argument("--start", type=int, required=True,
                  help="First frame number (e.g., 1)")
    p.add_argument("--end", type=int, required=True,
                  help="Last frame number (e.g., 2000)")
    p.add_argument("--contrast", type=float, default=1.0,
                  help="Contrast multiplier")
    return p.parse_args()

def load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def main():
    args = parse_args()
    folder = Path(args.path)

    # Create diff subfolder
    diff_dir = folder / "diff"
    diff_dir.mkdir(exist_ok=True)

    for i in range(args.start, args.end):
        img1_path = folder / f"DSC0{i:04d}.JPG"
        img2_path = folder / f"DSC0{i+1:04d}.JPG"

        img1 = load_img(img1_path)
        img2 = load_img(img2_path)

        # Convert to int16 to prevent wraparound
        img1_i = img1.astype(np.int16)
        img2_i = img2.astype(np.int16)

        diff = img2_i - img1_i
        diff = diff * args.contrast  # contrast boost
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        # Output filename goes into diff/ subfolder
        outname = f"{i+1:08d}-{i:08d}.png"
        outpath = diff_dir / outname

        cv2.imwrite(str(outpath), diff, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
