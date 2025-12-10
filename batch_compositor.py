import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time
from compositor import Compositor


def main():
  parser = argparse.ArgumentParser(description="Parallel PNG compositor")
  parser.add_argument("--path", required=True, help="Path to working directory with source files")
  parser.add_argument("--asi_diameter", type=float, default=0.75)
  parser.add_argument("--keogram_diameter", type=float, default=0.98)
  parser.add_argument("--unit_text", required=True, help="Text to display on images")
  parser.add_argument("--gamma", type=float, default=1.0)
  parser.add_argument("--output_size", type=int, required=True)
  parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
  parser.add_argument("--frame_count", type=int, required=True, help="Total number of frames")
  parser.add_argument("--skip_frames", type=int, default=1, required=False, help="Total number of frames")
  args = parser.parse_args()

  base_params = {
      "root_path": args.path,
      "frame_number": None,
      "asi_diameter": args.asi_diameter,
      "keogram_diameter": args.keogram_diameter,
      "unit_text": args.unit_text,
      "output_size": args.output_size,
      "gamma": args.gamma,
  }

  # Split frames for workers
  frames = np.array_split(range(0, args.frame_count, args.skip_frames), args.workers)
  frames_args = [(batch, base_params) for batch in frames]

  start_time = time.perf_counter()
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
      pool.map(render_batch_with_params, frames_args)
  end_time = time.perf_counter()

  elapsed_time = end_time - start_time
  print(f"{elapsed_time:.4f} sec to composite {args.frame_count} frames")

def render_one_frame(frame_number, base_params):
  try:
    frame_number = int(frame_number)  # ensure Python int
    params = base_params | {"frame_number": frame_number}
    print(f"Processing frame {frame_number}")
    c = Compositor(**params)  # <-- ** is required
    c.compose_frame()
  except Exception as e:
    print(e)
        

def render_batch_with_params(args):
    batch, base_params = args
    for i in batch:
        render_one_frame(i, base_params)

if __name__ == '__main__':
  main()