#seq_2025-11-13T01-41-35
#!/usr/bin/env python3
import subprocess
import argparse
import shlex
import csv
from pathlib import Path
from youtubeuploader import get_youtube_client, upload_video

# pipeline steps:
steps = {
  "download": False,
  "keogram": False,
  "polarwarp": False,
  "composite": True,
  "mp4": False,
  "upload": False
}


def get_frame_count(metadata_path):
    metadata_path = Path(metadata_path)
    with metadata_path.open(newline='') as f:
        reader = csv.reader(f)
        frame_count = sum(1 for _ in reader) - 1  # subtract 1 if there's a header row
    return frame_count

def latest_mp4(path: Path) -> Path | None:
    mp4_files = list(path.glob("*.mp4"))
    if not mp4_files:
        return None
    return max(mp4_files, key=lambda f: f.stat().st_mtime)

def main():
    parser = argparse.ArgumentParser(description="AurorEye processing pipeline wrapper")
    parser.add_argument("--path", required=True, help="Path to the sequence folder")
    parser.add_argument("--unit_text", required=True, help="Text string for unit annotation, can include newlines with \\n")
    parser.add_argument("--gamma", required=False, type=float, default=1.0, help="Gamma correction, defaults to 1 \\n")
    args = parser.parse_args()

    # Ensure path is absolute and normalized
    path = Path(args.path)
    unit_text = args.unit_text
    seq_name = path.name #i.e. seq_2025....

    local_root = Path("/Volumes/T7 Shield/AurorEye/")
    local_path = str(local_root / path)
    google_bucket_name = "auroreye-storage-558"

    # Step 0: Download from google cloud:
    cmd_download = [
      "gsutil",
      "-m", 
      "cp",
      "-r",
      f"gs://{google_bucket_name}/{seq_name}",
      str(local_root)
    ]
    print(f"Downloading from cloud {seq_name}...")
    print(cmd_download)
    if steps["download"]:
      subprocess.run(cmd_download, check=True)

    # Step 1: keogram.py
    cmd_keogram = [
        "python", "keogram.py",
        "--transform-metadata",
        "--metadata-ZVE10",
        "--keogram",
        "--contactsheet",
        "--path", local_path
    ]

    print("Running keogram.py...")
    if steps["keogram"]:
      subprocess.run(cmd_keogram, check=True)

    frame_count = get_frame_count(str(Path(local_path) / "metadata.csv"))
    print(f"Frames: {frame_count}")

    # Step 2: polarwarp.py
    keogram_file = f"{local_path}/keogram.png"
    cmd_polarwarp = [
        "python", "polarwarp.py",
        "--path", str(keogram_file),
        "--inner-diameter", "3200",
        "--outer-diameter", "4000",
        "--padding", "0",
        "--angle", "360"
    ]
    print("Running polarwarp.py...")
    if steps["polarwarp"]:
       subprocess.run(cmd_polarwarp, check=True)

    # Step 3: batch_compositor.py
    cmd_compositor = [
        "python", "batch_compositor.py",
        "--path", local_path,
        "--output_size", "2160",
        "--gamma", str(args.gamma),
        "--frame_count", str(frame_count),
        "--skip_frames", "1",
        "--unit_text", unit_text
    ]
    print("Running batch_compositor.py...")
    if steps["composite"]:
      subprocess.run(cmd_compositor, check=True)

    # Step 4: movie_generator.py
    cmd_movie = [
        "python", "movie_generator.py",
        "--path", local_path,
        "--framerate_in", "24",
        "--framerate_out", "24"
    ]
    print("Running movie_generator.py...")
    if steps["mp4"]:
      subprocess.run(cmd_movie, check=True)

    movie_path = latest_mp4(Path(local_path))

    if steps["upload"]:
      youtube = get_youtube_client()
      title = f"AurorEye {unit_text.replace("\n"," ")} {seq_name.replace("seq_","")}"
      description = title
      playlist_id = "PLXVlyzeh2wiG8LTGsbsH7KegGRlX-qFuf"

      print(f"Uploading {str(movie_path)}to youtube as {title}")
    
      upload_video( youtube, file_path=str(movie_path), title=title, description=description, playlist_id=playlist_id)



    print("Pipeline complete!")

if __name__ == "__main__":
    main()
