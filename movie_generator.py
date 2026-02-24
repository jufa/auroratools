#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import cv2

def generate_mp4(path: Path, framerate_in: int, framerate_out: int, min_duration=65, padding_image: Path = None):
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
    else:
        png_files = sorted(composed_dir.glob("*.png"))

    img = cv2.imread(str(first_frame))
    if img is None:
        raise RuntimeError(f"Failed to read {first_frame}")
    height, width = img.shape[:2]

    # Calculate existing duration and padding needed
    existing_frames = len(png_files)
    existing_duration = existing_frames / framerate_out  # duration in seconds at output framerate
    padding_duration = max(2, min_duration - existing_duration) # seconds

    print(f"Padding duration: {padding_duration:.2f} seconds (existing: {existing_duration:.2f} seconds)")

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
    
    # Build padding filter
    if padding_image and padding_image.exists():
        # Calculate number of frames needed for padding
        pad_frames = int(padding_duration * framerate_out)
        # Use custom padding image - scale to fit within target dimensions while maintaining aspect ratio, then pad to exact size
        time_pad_filter = f"{blend_filter}[v];movie=filename='{padding_image}':loop={pad_frames},setpts=N/(TB*{framerate_out}),scale={pad_width}:{height}:force_original_aspect_ratio=decrease,pad={pad_width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p[pad];[v][pad]concat=n=2:v=1:a=0"
    else:
        # Fallback to cloning last frame
        time_pad_filter = f"{blend_filter},tpad=stop_mode=clone:stop_duration={padding_duration}"

    use_hw = True  # True = VideoToolbox, False = libx264
    if use_hw:
        encoder = "h264_videotoolbox"
        bitrate = "40M"
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
    parser.add_argument("--framerate_in", type=int, default=24, help="Output framerate (default: 20)")
    parser.add_argument("--framerate_out", type=int, default=24, help="Output framerate (default: 60)")
    parser.add_argument("--padding_image", type=str, help="Path to image to use for padding (optional)")
    args = parser.parse_args()

    padding_path = Path(args.padding_image) if args.padding_image else None
    generate_mp4(Path(args.path), args.framerate_in, args.framerate_out, padding_image=padding_path)

if __name__ == "__main__":
    main()
