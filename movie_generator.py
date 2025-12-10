#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import cv2

def generate_mp4(path: Path, framerate_in: int, framerate_out: int, min_duration=60):
    composed_dir = path / "composed_frames"
    if not composed_dir.exists():
        raise FileNotFoundError(f"{composed_dir} does not exist")

    # Find first PNG to get resolution
    first_frame = composed_dir / "00000000.png"
    if not first_frame.exists():
        png_files = sorted(composed_dir.glob("*.png"))
        if not png_files:
            raise FileNotFoundError(f"No PNG files in {composed_dir}")
        first_frame = png_files[0]

    img = cv2.imread(str(first_frame))
    if img is None:
        raise RuntimeError(f"Failed to read {first_frame}")
    height, width = img.shape[:2]

    # Generate output filename
    output_file = f"{path.name}_{height}px_{framerate_in}infps_{framerate_out}outfps.mp4"
    output_path = path / output_file

    # Pad to 3840 width (centered) and convert to yuv420p
    pad_width = max(3840, width)
    pad_x = (pad_width - width) // 2
    pad_filter = f"pad={pad_width}:{height}:{pad_x}:0:color=black,format=yuv420p"

    # Frame blending: simple average to increase frame rate
    if framerate_out > framerate_in:
      print("using frame blending...")
      blend_filter = f"{pad_filter},tblend=all_mode=average"
    else:
      blend_filter = f"{pad_filter}"
    time_pad_filter = f"{blend_filter},tpad=stop_mode=clone:stop_duration={min_duration}"

    use_hw = True  # True = VideoToolbox, False = libx264
    if use_hw:
        encoder = "h264_videotoolbox"
        bitrate = "20M"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(framerate_in),  # input frame rate (e.g., 20)
            # "-pattern_type", "glob", # allow skipped numbered frames but still encode alphabetically
            # "-i", str(composed_dir / "*.png"), # ^ cont
            "-i", str(composed_dir / "%08d.png"),
            "-color_range", "pc",
            "-pix_fmt", "yuv420p",
            "-vf", time_pad_filter,
            "-r", str(framerate_out),      # output frame rate (e.g., 60)
            "-c:v", encoder,
            "-b:v", bitrate,
            str(output_path)
        ]

    print("Running ffmpeg command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved MP4 to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert PNG sequence to MP4")
    parser.add_argument("--path", type=str, required=True, help="Path to folder containing 'composed_frames'")
    parser.add_argument("--framerate_in", type=int, default=20, help="Output framerate (default: 20)")
    parser.add_argument("--framerate_out", type=int, default=60, help="Output framerate (default: 60)")
    args = parser.parse_args()

    generate_mp4(Path(args.path), args.framerate_in, args.framerate_out)

if __name__ == "__main__":
    main()
