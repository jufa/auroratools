#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import os
import glob
import subprocess
from pathlib import Path

"""
This is a failure so far in black aurora
python optical_flow.py --start 1 --end 100 --input_folder ./images [--step_thru] # - Shows debug windows
"""


def create_flow_visualization(flow, clahe_frame, filename="", step=18, draw_legend=True):
  """Create a vector field visualization of optical flow overlaid on CLAHE frame.
  
  Args:
    flow: Optical flow field from calcOpticalFlowFarneback
    clahe_frame: CLAHE-enhanced grayscale frame to overlay on
    filename: Input filename to display (optional)
    step: Grid spacing for flow vectors (default 23)
    draw_legend: Whether to draw the direction legend (default True)
  """
  h, w = flow.shape[:2]
  
  # Convert grayscale to BGR for color overlay
  vis = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR).astype(np.uint8)
  
  # Create a grid of points
  y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
  
  # Get flow vectors at grid points
  fx, fy = flow[y, x].T
  
  # Draw flow vectors as lines with circles
  for i in range(len(x)):
    magnitude = np.sqrt(fx[i]**2 + fy[i]**2)
    if magnitude > 0.0:
      scale = 2.0  # Scale factor for better visibility
      pt1 = (x[i], y[i])
      pt2 = (int(x[i] + fx[i] * scale), int(y[i] + fy[i] * scale))
      
      # Calculate angle for color based on cardinal directions
      angle = np.arctan2(fy[i], fx[i])
      # Normalize angle to 0-1 range
      normalized_angle = (angle + np.pi) / (2 * np.pi)
      
      # Define cardinal direction colors in BGR
      # East (0°): Green, North (90°): Cyan, West (180°): Magenta, South (270°): White
      base_val = 64
      cardinal_colors = [
        (base_val, 255, base_val),  # East: Green
        (base_val, 255, 255),  # North: Cyan
        (255, base_val, 255),  # West: Magenta
        (255, 255, 255),  # South: White
      ]
      
      # Interpolate between cardinal colors based on angle
      color_index = normalized_angle * 4
      color1_idx = int(color_index) % 4
      color2_idx = (int(color_index) + 1) % 4
      blend = color_index - int(color_index)
      
      c1 = np.array(cardinal_colors[color1_idx], dtype=float)
      c2 = np.array(cardinal_colors[color2_idx], dtype=float)
      bgr_color = tuple(int(c) for c in (c1 * (1 - blend) + c2 * blend))
      
      # Draw line with increased thickness for visibility
      cv2.line(vis, pt1, pt2, bgr_color, 1)
      
      # Draw small open circle at origin
      # cv2.circle(vis, pt1, 3, bgr_color, 1)
  
  # Add filename text if provided
  if filename:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0, 255, 255)  # Cyan
    text_pos = (10, h - 10)
    cv2.putText(vis, filename, text_pos, font, font_scale, text_color, font_thickness)
  
  # Add flow direction legend in top left
  if draw_legend:
    draw_flow_legend(vis, 0, 70)
  
  return vis

def create_sidebyside_visualization(flow, clahe_frame, filename, step=18, frame_opacity=0.6):
  """Create side-by-side visualization: full CLAHE on left, low opacity CLAHE + flow on right (2160x1080).
  
  Args:
    flow: Optical flow field from calcOpticalFlowFarneback
    clahe_frame: CLAHE-enhanced grayscale frame
    filename: Input filename to display
    step: Grid spacing for flow vectors
    frame_opacity: Opacity of frame background behind flow field (0.0-1.0, default 0.3)
  """
  # Target output size: 2160x1080 (left 1080x1080, right 1080x1080)
  output_width = 2160
  output_height = 1080
  half_width = 1080
  
  # Left side: full brightness CLAHE frame
  clahe_resized = cv2.resize(clahe_frame, (half_width, output_height))
  clahe_left = cv2.cvtColor(clahe_resized, cv2.COLOR_GRAY2BGR).astype(np.uint8)
  
  # Right side: flow visualization with opacity-reduced CLAHE background
  flow_resized = cv2.resize(flow, (half_width, output_height))
  
  # Create dimmed CLAHE frame for right side background
  clahe_right = cv2.cvtColor(clahe_resized, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
  clahe_right = (clahe_right * frame_opacity * 255).astype(np.uint8)
  clahe_right_gray = cv2.cvtColor(clahe_right, cv2.COLOR_BGR2GRAY)
  
  # Create flow visualization on top of dimmed CLAHE (no legend, we'll add it separately)
  flow_right = create_flow_visualization(flow_resized, clahe_right_gray, "", step=12, draw_legend=False)
  
  # Combine left and right
  combined = np.hstack([clahe_left, flow_right])
  
  # Add SINGLE legend to top left of right frame (within combined image)
  draw_flow_legend(combined, half_width, 70)
  
  # Add filename text spanning across bottom
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.8
  font_thickness = 2
  text_color = (0, 255, 255)  # Cyan
  text_pos = (20, output_height - 20)
  cv2.putText(combined, filename, text_pos, font, font_scale, text_color, font_thickness)
  
  return combined

def process_optical_flow(input_folder, start_frame, end_frame, output_folder, roi=None, step_thru=False, prescale=1.0, rotate=0, temporal_window=7, sidebyside=False, frame_opacity=0.3):
  """Process sequence of images to generate optical flow visualization."""

  print(f"sidebyside: {sidebyside}  ")
  
  # Create output folder
  Path(output_folder).mkdir(parents=True, exist_ok=True)
  
  # Create CLAHE object once with fixed parameters for all frames
  clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
  
  # Get list of input files
  files = []
  for i in range(start_frame, end_frame + 1):
    filename = f"{i:08d}.jpg"
    filepath = os.path.join(input_folder, filename)
    if os.path.exists(filepath):
      files.append(filepath)
  
  if len(files) < 2:
    print("Need at least 2 frames for optical flow")
    return
  
  prev_frame = None
  prev_clahe_frame = None
  flow_history = []
  
  for frame_num, filepath in zip(range(start_frame + 1, end_frame + 1), files[1:]):
    i = frame_num - start_frame
    print(f"Processing frame {i}/{len(files)-1}")
    
    # Read current frame
    curr_img = cv2.imread(filepath)
    if curr_img is None:
      continue
    
    # Apply rotation
    if rotate != 0:
      h, w = curr_img.shape[:2]
      center = (w // 2, h // 2)
      rotation_matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
      curr_img = cv2.warpAffine(curr_img, rotation_matrix, (w, h))
    
    # Mirror horizontally
    curr_img = cv2.flip(curr_img, 1)
    
    # Apply prescaling
    if prescale != 1.0:
      new_width = int(curr_img.shape[1] * prescale)
      new_height = int(curr_img.shape[0] * prescale)
      curr_img = cv2.resize(curr_img, (new_width, new_height))
    
    # Apply ROI if specified
    if roi:
      tl_x, tl_y, br_x, br_y = roi
      curr_img = curr_img[tl_y:br_y, tl_x:br_x]
    
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    # Denoise before CLAHE to reduce grain amplification
    curr_gray = cv2.bilateralFilter(curr_gray, 9, 75, 75)
    
    # Apply the same CLAHE object to all frames
    curr_clahe = clahe.apply(curr_gray)
    
    # Additional contrast stretching
    curr_clahe = cv2.normalize(curr_clahe, None, 0, 255, cv2.NORM_MINMAX)
    
    # Visualize CLAHE enhanced frame if step_thru enabled
    if step_thru:
      print(f"  Frame {i}: CLAHE applied. Press any key to continue...")
      cv2.imshow("CLAHE Enhanced Frame", curr_clahe)
      cv2.waitKey(0)
    
    if prev_frame is not None:
      # Check if output already exists before expensive calculations
      output_path = os.path.join(output_folder, f"flow_{frame_num:08d}.png")
      if os.path.exists(output_path):
        print(f"  Frame {frame_num:08d} already exists, skipping...")
        prev_frame = curr_clahe.copy()
        prev_clahe_frame = curr_clahe.copy()
        continue
      
      # Calculate dense optical flow with sensitive parameters
      flow = cv2.calcOpticalFlowFarneback(
        prev_frame, curr_clahe, None, 
        pyr_scale=0.2, levels=9, winsize=5,
        iterations=5, poly_n=5, poly_sigma=0.5, flags=0
      )
      
      # Add to history for temporal smoothing
      flow_history.append(flow)
      if len(flow_history) > temporal_window:
        flow_history.pop(0)
      
      # Average flows in history
      smoothed_flow = np.mean(flow_history, axis=0)
      
      # Create flow visualization with filename
      display_filename = os.path.basename(os.path.dirname(files[i-1])) + "/" + os.path.basename(files[i-1])
      
      if sidebyside:
        # Create side-by-side visualization (2160x1080)
        final_vis = create_sidebyside_visualization(smoothed_flow, prev_clahe_frame, display_filename, frame_opacity=frame_opacity)
      else:
        # Create regular flow visualization
        final_vis = create_flow_visualization(smoothed_flow, prev_clahe_frame, display_filename)
      
      # Visualize optical flow if step_thru enabled
      if step_thru:
        print(f"  Frame {i}: Optical flow calculated. Press any key to continue...")
        cv2.imshow("Optical Flow Visualization", final_vis)
        cv2.waitKey(0)
      
      # Save visualization
      cv2.imwrite(output_path, final_vis)
    
    prev_frame = curr_clahe.copy()
    prev_clahe_frame = curr_clahe.copy()
  
  # Close all windows when done
  cv2.destroyAllWindows()

def create_video(output_folder, output_video, fps=30):
  """Create video from PNG sequence using ffmpeg with ProRes codec for color fidelity."""
  # Determine output codec based on file extension
  if output_video.lower().endswith('.mov'):
    # ProRes for .mov files (best color preservation)
    cmd = [
      'ffmpeg', '-y',
      '-framerate', str(fps),
      '-pattern_type', 'glob',
      '-i', os.path.join(output_folder, 'flow_*.png'),
      '-c:v', 'prores_ks',
      '-pix_fmt', 'yuv422p10le',
      '-profile:v', '3',
      '-q:v', '6',
      output_video
    ]
  else:
    # Fall back to h.264 for MP4, with best possible settings
    cmd = [
      'ffmpeg', '-y',
      '-framerate', str(fps),
      '-pattern_type', 'glob',
      '-i', os.path.join(output_folder, 'flow_*.png'),
      '-c:v', 'libx264rgb',
      '-crf', '18',
      '-preset', 'slow',
      output_video
    ]
  
  try:
    subprocess.run(cmd, check=True)
    print(f"Video created: {output_video}")
  except subprocess.CalledProcessError as e:
    print(f"Error creating video: {e}")

def draw_flow_legend(image, x, y, size=60):
  """Draw a compass rose legend showing flow direction colors.
  
  Args:
    image: Image to draw legend on (modified in place)
    x, y: Top-left corner position
    size: Size of legend in pixels (60x60)
  """
  # Colors in BGR: East=Magenta, North=Cyan, West=Green, South=White
  colors = {
    'N': (128, 255, 255),  # Cyan
    'E': (255, 128, 255),  # Magenta
    'S': (255, 255, 255),  # White
    'W': (128, 255, 128),  # Green
  }
  
  center_x = x + size // 2
  center_y = y + size // 2
  
  # Draw semi-transparent background box (60x60)
  overlay = image.copy()
  cv2.rectangle(overlay, (x, y), (x + size, y + size), (0, 0, 0), -1)
  cv2.addWeighted(overlay, 0.9, image, 0.5, 0, image)
  
  # Arrow length
  arrow_len = size // 3
  
  # Draw compass rose arrows
  directions = [
    ('N', (0, -1)),   # Up - Cyan
    ('E', (1, 0)),    # Right - Magenta
    ('S', (0, 1)),    # Down - White
    ('W', (-1, 0)),   # Left - Green
  ]
  
  for label, (dx, dy) in directions:
    # Arrow endpoint
    end_x = int(center_x + dx * arrow_len)
    end_y = int(center_y + dy * arrow_len)
    
    # Draw arrow line
    cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), colors[label], 2, tipLength=0.4)
    
    # Draw label with increased padding
    label_padding = size // 2 + 8
    label_x = int(center_x + dx * label_padding)
    label_y = int(center_y + dy * label_padding)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    
    # Get text size for centering
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    label_x -= text_size[0] // 2
    label_y += text_size[1] // 2
    
    # Draw label in the same color as the arrow
    cv2.putText(image, label, (label_x, label_y), font, font_scale, colors[label], font_thickness)

def main():
  parser = argparse.ArgumentParser(description='Generate optical flow visualization from image sequence')
  parser.add_argument('--input_folder', help='Folder containing input JPG files')
  parser.add_argument('--start', type=int, required=True, help='Starting frame number')
  parser.add_argument('--end', type=int, required=True, help='Ending frame number')
  parser.add_argument('--output', default='flow_movie.mp4', help='Output video filename')
  parser.add_argument('--prescale', type=float, default=1.0, help='Prescale factor (e.g., 0.5 for 50% reduction)')
  parser.add_argument('--rotate', type=float, default=0, help='Rotate image N degrees clockwise before processing')
  parser.add_argument('--roi', nargs=4, type=int, metavar=('TL_X', 'TL_Y', 'BR_X', 'BR_Y'),
            help='Region of interest: top-left and bottom-right coordinates')
  parser.add_argument('--fps', type=int, default=30, help='Output video framerate')
  parser.add_argument('--step_thru', action='store_true', help='Enable interactive step-through with imshow debug windows')
  parser.add_argument('--sidebyside', action='store_true', help='Display CLAHE on left, flow field on right in 2160x1080 format')
  parser.add_argument('--frame_opacity', type=float, default=0.3, help='Opacity of CLAHE frame behind flow field (0.0-1.0)')
  
  args = parser.parse_args()
  
  # Derive output folder from output video path
  output_video_path = Path(args.output)
  output_folder = str(output_video_path.parent / 'flow_analysis')
  
  # Process optical flow
  process_optical_flow(
    input_folder=args.input_folder,
    start_frame=args.start,
    end_frame=args.end,
    output_folder=output_folder,
    roi=args.roi,
    step_thru=args.step_thru,
    prescale=args.prescale,
    rotate=args.rotate,
    temporal_window=7,
    sidebyside=args.sidebyside,
    frame_opacity=args.frame_opacity
  )
  
  # Create video
  create_video(output_folder, args.output, args.fps)

if __name__ == '__main__':
  main()