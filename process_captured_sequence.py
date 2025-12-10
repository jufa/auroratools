#seq_2025-11-13T01-41-35
#!/usr/bin/env python3
import subprocess
import argparse
import shlex
import csv
from pathlib import Path

def get_frame_count(metadata_path):
    metadata_path = Path(metadata_path)
    with metadata_path.open(newline='') as f:
        reader = csv.reader(f)
        frame_count = sum(1 for _ in reader) - 1  # subtract 1 if there's a header row
    return frame_count

def main():
    parser = argparse.ArgumentParser(description="AurorEye processing pipeline wrapper")
    parser.add_argument("--path", required=True, help="Path to the sequence folder")
    parser.add_argument("--unit_text", required=True, help="Text string for unit annotation, can include newlines with \\n")
    parser.add_argument("--gamma", required=False, type=float, default=1.0, help="Gamma correction, defaults to 1 \\n")
    args = parser.parse_args()

    # Ensure path is absolute and normalized
    path = Path(args.path).resolve()
    unit_text = args.unit_text

    # Step 1: keogram.py
    cmd_keogram = [
        "python", "keogram.py",
        "--transform-metadata",
        "--metadata-ZVE10",
        "--keogram",
        "--contactsheet",
        "--path", str(path)
    ]

    print("Running keogram.py...")
    # subprocess.run(cmd_keogram, check=True)

    frame_count = get_frame_count(path / "metadata.csv")
    print(f"Frames: {frame_count}")

    # Step 2: polarwarp.py
    keogram_file = path / "keogram.png"
    cmd_polarwarp = [
        "python", "polarwarp.py",
        "--path", str(keogram_file),
        "--inner-diameter", "3200",
        "--outer-diameter", "4000",
        "--padding", "0",
        "--angle", "360"
    ]
    print("Running polarwarp.py...")
    # subprocess.run(cmd_polarwarp, check=True)

    # Step 3: batch_compositor.py
    cmd_compositor = [
        "python", "batch_compositor.py",
        "--path", str(path),
        "--output_size", "2160",
        "--gamma", str(args.gamma),
        "--frame_count", str(frame_count),
        "--skip_frames", "1",
        "--unit_text", unit_text
    ]
    print("Running batch_compositor.py...")
    subprocess.run(cmd_compositor, check=True)

    # Step 4: movie_generator.py
    cmd_movie = [
        "python", "movie_generator.py",
        "--path", str(path),
        "--framerate_in", "24",
        "--framerate_out", "24"
    ]
    print("Running movie_generator.py...")
    subprocess.run(cmd_movie, check=True)

    print("Pipeline complete!")

if __name__ == "__main__":
    main()
