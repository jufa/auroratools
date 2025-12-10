import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import time
from functools import wraps
import cv2

"""
usage:
python compositor.py --path "/Volumes/T7 Shield/AurorEye/seq_2025-12-06T07-28-12/" --output-size 2160 --gamma 0.5 --frame-count 100 --unit-text "UNIT 12\nFAIRBANKS, AK"
"""
perf_mon=False


def timeit(func):
    """
    Decorator to measure the execution time of a function.
    """
    if not perf_mon:
        return func 
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{elapsed_time:.4f} seconds for FUNCTION {func.__name__!r}")
        return result
    return wrapper

class Compositor:
    """
    Handles the compositing of a single image frame based on metadata,
    transform configuration, and CLI-like parameters.
    """
    def __init__(self, root_path, output_size,
                 frame_number, asi_diameter, 
                 keogram_diameter, unit_text, gamma):

        self.root_path = Path(root_path)
        self.output_path = self.root_path / "composed_frames"
        self.output_path.mkdir(parents=False, exist_ok=True)
        self.metadata_csv_path = self.root_path / "metadata.csv"
        self.transform_json_path = self.root_path / "transform_metadata.json"
        self.annulus_png_path = self.root_path / "keogram_annular.png"
        self.frame_number = frame_number
        self.asi_diameter = asi_diameter
        self.keogram_diameter = keogram_diameter
        self.logo_file = Path("/Users/jeremy/projects/auroratools/auroreyelogo128.png")
        self.unit_text = unit_text
        self.output_size = output_size # Standard output frame size (e.g., 1000x1000 pixels)
        self.last_time_check = None
        self.mask_enabled = False
        self.gamma = gamma
        
        # Internal configuration
        self.margin = 30
        self.font_path = "HelveticaNeue.ttc"
        self.font_size_smallest = 22
        self.font_size_small = 28
        self.font_size_large = 36
        self.font_scale = 1500

        # Data placeholders
        self.metadata_df = None
        self.transform_data = None
        self.target_row = None
        self.total_rows = 0
        self.gamma_lut = self.build_gamma_lut(gamma=self.gamma)
        
        # Run setup to load data
        self._load_data()

    def build_gamma_lut(self, gamma):
      inv = 1.0 / gamma
      lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
      return lut

    def time_check(self, msg):
        if not perf_mon:
            return
        if (self.last_time_check):
          elapsed_time = time.perf_counter() - self.last_time_check
          print(f"{elapsed_time:.4f} seconds FOR {msg}")
        self.last_time_check = time.perf_counter()
    
    @timeit
    def composite_layers(self, background, foreground):
      """
      background, foreground: BGRA uint8 images, same size
      Returns composited BGRA image.
      """

      # Ensure float32 for blending
      bg = background.astype(np.float32)
      fg = foreground.astype(np.float32)

      # Normalize alpha channel of foreground to 0..1
      alpha = fg[:, :, 3:4] / 255.0  # shape (H, W, 1)

      # Blend BGR channels
      bg[:, :, :3] = alpha * fg[:, :, :3] + (1 - alpha) * bg[:, :, :3]

      # Optional: update alpha channel
      bg[:, :, 3] = np.clip(fg[:, :, 3] + bg[:, :, 3] * (1 - alpha[:, :, 0]), 0, 255)

      return bg.astype(np.uint8)


    @timeit
    def cv2_to_pil_rgba(self, cv2_array):
        """Converts an OpenCV BGR or BGRA array to a PIL Image (RGBA)."""
        if cv2_array.shape[2] == 3: # BGR (3 channels)
            # Convert BGR to RGB
            rgb_array = cv2.cvtColor(cv2_array, cv2.COLOR_BGR2RGB)
            # Use 'RGB' mode
            return Image.fromarray(rgb_array, 'RGB')
        elif cv2_array.shape[2] == 4: # BGRA (4 channels)
            # Convert BGRA to RGBA (required for PIL's alpha handling)
            rgb_array = cv2.cvtColor(cv2_array, cv2.COLOR_BGRA2RGBA)
            # FIX: Explicitly specify 'RGBA' mode to ensure alpha channel integrity
            return Image.fromarray(rgb_array, 'RGBA') 
        else:
            raise ValueError("Unsupported channel depth in OpenCV array.")

    @timeit
    def pil_to_cv2_bgra(self, pil_image):
        """Converts a PIL Image (must be RGBA) to an OpenCV BGRA array."""
        if pil_image.mode != 'RGBA':
            # This conversion itself might be slow if the image is large, 
            # but it is necessary if the logo/text drawing changed the mode.
            pil_image = pil_image.convert('RGBA')
            
        np_array = np.array(pil_image)
        # Convert RGBA (Pillow) to BGRA (OpenCV standard with alpha)
        return cv2.cvtColor(np_array, cv2.COLOR_RGBA2BGRA)
    
    @timeit
    def _load_data(self):
        """Loads CSV and JSON data and selects the target row."""
        try:
            # Load Metadata CSV
            self.metadata_df = pd.read_csv(self.metadata_csv_path)
            self.total_rows = len(self.metadata_df)
            
            # Row number is 1-based, index is 0-based
            if 0 < self.frame_number <= self.total_rows:
                self.target_row = self.metadata_df.iloc[self.frame_number]
            else:
                raise IndexError(f"Row number {self.frame_number} is out of bounds (1 to {self.total_rows}).")
            
            # Load Transform JSON
            with open(self.transform_json_path, 'r') as f:
                self.transform_data = json.load(f)
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required file not found: {e.filename}")
        except Exception as e:
            raise Exception(f"Error during data loading: {e}")
    
    @timeit
    def _load_and_process_asi_image(self, input_img_path, total_rotation_angle):
        """
        Loads the ASI image, crops it, and composites it onto an opaque black 
        background, masking the corners if apply_mask is True.
        Returns a BGRA NumPy array.
        """
        # 1. Load BGR Image (OpenCV default, fastest I/O)
        img_bgr = cv2.imread(str(input_img_path), cv2.IMREAD_COLOR)


        if img_bgr is None:
            raise FileNotFoundError(f"OpenCV could not load image at {input_img_path}")
            
        # rescale to output size (usually smaller so in anything improves perf)
        scale_px = int(self.output_size * self.asi_diameter)
        scale_px += scale_px % 2
        img_bgr_rescaled = self._rotate_and_scale(img_bgr, scale_px, total_rotation_angle) # fraction of full output image size
        img_bgr_rescaled = self._flip_horizontal(img_bgr_rescaled)

        border_size =  (self.output_size - scale_px) // 2 # Thickness of the border in pixels
        border_color = [0, 0, 0]  # BGR color for the border (e.g., [0, 0, 255] for red)

        image_with_border = cv2.copyMakeBorder(
            img_bgr_rescaled,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )

        # Convert the cropped BGR image to BGRA (adding an opaque alpha channel)
        asi_img_bgra = cv2.cvtColor(image_with_border, cv2.COLOR_BGR2BGRA)
    
        return asi_img_bgra
    
    @timeit
    def _flip_horizontal(self, cv2_array):
        return cv2.flip(cv2_array, 1)

    @timeit
    def _rotate_and_scale(self, cv2_array, target_diameter, angle):
        """
        Scales an OpenCV BGRA array and rotates it using cv2.warpAffine.
        Returns a BGRA NumPy array.
        """
        target_diameter = int(target_diameter)
        h, w = cv2_array.shape[:2]
        
        # 1. Scaling (Resize)
        if w != target_diameter:
            # Use INTER_AREA for optimized downscaling, which is common here.
            scaled_array = cv2.resize(cv2_array, (target_diameter, target_diameter), 
                                      interpolation=cv2.INTER_AREA)
        else:
            scaled_array = cv2_array
            
        h_s, w_s = scaled_array.shape[:2]
        center_s = (w_s // 2, h_s // 2)
        
        # 2. Rotation
        # Get the rotation matrix (Note: cv2 uses counter-clockwise, so we negate the angle)
        M = cv2.getRotationMatrix2D(center_s, -angle, 1.0) 

        # Perform the rotation using INTER_LINEAR, keeping the original size
        rotated_array = cv2.warpAffine(scaled_array, M, (w_s, h_s), 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=(0, 0, 0, 255)) # Transparent background
        
        return rotated_array # Returns BGRA NumPy array
    
    @timeit
    def color_dodge_roi(self, img, cx, cy, half_w, half_h, strength=1.0):
      """
      img: BGRA or BGR uint8 OpenCV image
      cx, cy: center of rectangle
      half_w, half_h: half-dimensions of rect
      strength: 1.0 = normal color dodge, <1 weaker, >1 stronger
      """

      h, w = img.shape[:2]

      # Rectangle bounds
      x1 = max(0, cx - half_w)
      x2 = min(w, cx + half_w)
      y1 = max(0, cy - half_h)
      y2 = min(h, cy + half_h)

      # Extract ROI view
      roi = img[y1:y2, x1:x2]

      # Work only on BGR channels
      base = roi[:, :, :3].astype(np.float32) / 255.0
      # Use a constant light layer (white) scaled by strength
      light = np.clip(strength, 0, 10.0)  
      light_layer = (np.ones_like(base) * light).clip(0, 1)

      # Color Dodge: base / (1 - light)
      # But with safety clamp to avoid blowouts or division by zero
      denom = np.clip(1.0 - light_layer, 1e-6, 1.0)
      dodged = base / denom
      dodged = np.clip(dodged, 0.0, 1.0)

      roi[:, :, :3] = (dodged * 255).astype(np.uint8)

      return img

    @timeit
    def gamma_roi(self, img, cx, cy, half_w, half_h, gamma):
      """
      img: BGRA or BGR uint8 OpenCV image
      cx, cy: center of rectangle
      half_w, half_h: rectangle extends symmetrically
      gamma: gamma correction factor
      """

      h, w = img.shape[:2]

      # Compute bounding box
      x1 = max(0, cx - half_w)
      x2 = min(w, cx + half_w)
      y1 = max(0, cy - half_h)
      y2 = min(h, cy + half_h)

      # Extract ROI
      roi = img[y1:y2, x1:x2, :3]  # BGR color only

      img[y1:y2, x1:x2, :3] = cv2.LUT(roi, self.gamma_lut)

      return img

    @timeit
    def _format_coordinates(self):
        """Formats lat/lon data into the required string format."""
        lat = self.target_row['latitude']
        lat_ref = self.target_row['latitude_ref']
        lon = self.target_row['longitude']
        lon_ref = self.target_row['longitude_ref']
        
        # Assume lat and lon are strings with decimal values
        lat_str = f"{float(lat):.1f}°{lat_ref}"
        lon_str = f"{float(lon):.1f}°{lon_ref}"
        
        return f"{lat_str}, {lon_str}"
    
    @timeit
    def compose_frame(self):
        """Performs the full compositing procedure using OpenCV for heavy lifting."""
        
        # 1. Initialize the Base Canvas (Always the final output size)
        # Must be BGRA for compositing layers, then converted for text
        base_frame_cv2 = np.full((self.output_size, self.output_size, 4), (0, 0, 0, 255), dtype=np.uint8)
        
        # --- LAYER 1: FISHEYE IMAGE (INPUT 1) ---
        
        root_path = Path(self.transform_data.get('root_path', ''))
        input1_path = root_path / self.target_row['filename']
        
        # a. Mask and Crop (Returns BGRA NumPy array)
        total_rotation_angle = self.transform_data.get('rotation', 0) + self.transform_data.get('declination', 0)
        masked_cropped_img_cv2 = self._load_and_process_asi_image(input1_path, total_rotation_angle)

        # --- LAYER 2: ANNULUS (INPUT 4) ---
        
        # Load Annulus as PIL, convert to BGRA NumPy array
        self.time_check("reset")
        annulus_cv2 = cv2.imread(self.annulus_png_path, cv2.IMREAD_UNCHANGED)
        self.time_check("keo file open ")
        # annulus_cv2 = self.pil_to_cv2_bgra(annulus_pil)
        
        # b. Annulus Rotation
        annulus_rotation_angle = -((self.frame_number - 1) / self.total_rows) * 360
        
        # a. Scale and Rotate (Returns BGRA NumPy array)
        scale_px = int(self.output_size * self.keogram_diameter)
        scale_px += scale_px % 2
        rotated_annulus_cv2 = self._rotate_and_scale(
            annulus_cv2, scale_px, annulus_rotation_angle)
        
        border_size =  (self.output_size - scale_px) // 2 # Thickness of the border in pixels
        border_color = [0, 0, 0]  # BGR color for the border (e.g., [0, 0, 255] for red)

        rotated_annulus_cv2_with_border = cv2.copyMakeBorder(
            rotated_annulus_cv2,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )
        
        base_frame_cv2 = self.composite_layers(masked_cropped_img_cv2, rotated_annulus_cv2_with_border)


        #adjust global gamma:
        base_frame_cv2 = self.gamma_roi(img=base_frame_cv2, cx=(self.output_size//2), cy=(self.output_size//2), half_w=(self.output_size//2), half_h=(self.output_size//2), gamma=self.gamma)

        # add "now" marker
        base_frame_cv2 = self.color_dodge_roi(img=base_frame_cv2, cx=(self.output_size//2), cy=(self.output_size//20), half_w=10, half_h=(self.output_size//18), strength=0.5)

        self.time_check("annulus alpha channel BS ")
        # --- LAYER 3 & 4: TEXT AND LOGO OVERLAYS (PIL) ---
        
        # Convert the final composite base frame from OpenCV BGRA to PIL RGBA
        base_frame_pil = self.cv2_to_pil_rgba(base_frame_cv2)
        draw = ImageDraw.Draw(base_frame_pil)
        self.time_check("drawing composite frame without text ")
        
        # ... (Load Fonts) ...
        # ... (Text data preparation) ...

        # --- Layer 4: Logo (Must be done via PIL after conversion) ---
        try:
            logo = Image.open(self.logo_file).convert("RGBA")
            # ... (Scaling logo) ...
            # ... (Calculate position and composite using base_frame_pil.paste) ...
        except FileNotFoundError:
            # ... (Handle missing logo) ...
            print(f"logo file not found: {self.logo_file}" )

        # --- Layer 3 & 4: Text Drawing (using draw object on base_frame_pil) ---
        try:
            font_size_smallest_scaled = int(self.font_size_smallest * self.output_size / self.font_scale)
            font_size_small_scaled = int(self.font_size_small * self.output_size / self.font_scale)
            font_size_large_scaled = int(self.font_size_large * self.output_size / self.font_scale)
            line_spacing = 1.2
            margin_scaled = int(self.margin * self.output_size / self.font_scale)

            # Compact Sans-Serif Fonts (Adjust path/name as needed)
            font_smallest = ImageFont.truetype(self.font_path, font_size_smallest_scaled, index=0)
            font_small = ImageFont.truetype(self.font_path, font_size_small_scaled, index=0)
            font_large = ImageFont.truetype(self.font_path, font_size_large_scaled, index=0)
            font_cardinal = ImageFont.truetype(self.font_path, font_size_small_scaled, index=1)

        except IOError:
            print(f"Warning: Could not load font {self.font_path}. Using default Pillow font.")
            font_small = ImageFont.load_default()
            font_large = ImageFont.load_default()

        self.time_check("loading logo and fonts ")

        text_color = (255, 255, 255, 255) # White
        outline_color = (64, 64, 64, 255)

        # --- LAYER 3: BOTTOM RIGHT (RIGHT JUSTIFIED) ---
        
        x_right_anchor = self.output_size - margin_scaled
        y_start = self.output_size - margin_scaled - (5 * font_size_small_scaled * line_spacing ) # Start above the bottom margin

        # Data preparation
        geo_text = self._format_coordinates()
        iso = int(self.target_row['photographic_sensitivity'])
        exp = round(float(self.target_row['exposure_time']), 1)
        exposure_text = f"ISO {iso} | EXP {exp:.1f}s"
        date_text = f"{self.target_row['date']}"
        time_text = f"{self.target_row['time']} UTC"
        filename_text = self.target_row['filename']
        
        # List of text strings to draw
        right_texts = [time_text, date_text, filename_text, exposure_text, geo_text]
        right_texts_colors = [
            (255,255,255,255),
            (255,255,255,255),
            (255,255,255,255),
            (0,255,255,255),
            (0,255,255,255),
        ]
        
        # Draw the right-justified text
        y_current = y_start
        for i, text in enumerate(right_texts):
            font = font_small
            font_size_scaled = font_size_small_scaled
            if i==0:
              font = font_large
              font_size_scaled = font_size_large_scaled
            text_width = draw.textlength(text, font=font)
            # Anchor text to the right edge
            x_pos = x_right_anchor - text_width
            draw.text((x_pos, y_current), text, font=font, fill=right_texts_colors[i])
            y_current += font_size_scaled * line_spacing # Move to the next row

        self.time_check("drawing bottom right text ")

        # --- LAYER 4: BOTTOM LEFT (LEFT JUSTIFIED) ---
      
        x_left_anchor = margin_scaled
        logo_height = font_size_large_scaled * 5
        y_start = self.output_size - margin_scaled - (3 * font_size_small_scaled * line_spacing) - logo_height # Align with row 2 of the right text
        y_start = int(y_start)
        # Row 1: Logo PNG
        try:
            logo = Image.open(self.logo_file).convert("RGBA")
            
            logo_width = int(logo.width * (logo_height / logo.height))
            logo = logo.resize((logo_width, logo_height))
            
            # Composite logo 
            base_frame_pil.paste(logo, (x_left_anchor, y_start), logo)
            y_current_left = int(y_start + logo_height * 0.8 + font_size_small_scaled * line_spacing ) # 0.8 ius a fudge number for the white space border around the logo image
            
        except FileNotFoundError:
            print(f"Warning: Logo file {self.logo_file} not found. Skipping logo.")
            y_current_left = y_start # Start at the same line if logo is skipped

        # Row 2 & 3: Text
        draw.text((x_left_anchor, y_current_left), "AUROREYE.CA", font=font_small, fill=(0,255,128,255))
        y_current_left += font_size_large_scaled * line_spacing
        
        draw.multiline_text((x_left_anchor, y_current_left), self.unit_text, font=font_large, fill=text_color)

        self.time_check("drawing bootom left text ")

        # cardinal points:
        cardinal_color=(0,255,255,255)
        pad = -0.01 * self.asi_diameter * self.output_size
        x = self.output_size * (1 - self.asi_diameter) // 2
        y = self.output_size // 2
        text="W"
        text_width = draw.textlength(text, font=font_cardinal)
        x_centered = x - text_width // 2 + pad
        y_centered = y - text_width // 2  # optional: to also center vertically
        draw.text((x_centered, y_centered), text, font=font_cardinal, fill=cardinal_color)

        x = self.output_size - x
        text="E"
        x_centered = x - text_width // 2 - pad
        draw.text((x_centered, y_centered), text, font=font_cardinal, fill=cardinal_color)
        
        x = self.output_size // 2
        y = self.output_size * (1 - self.asi_diameter) // 2
        text="N"
        text_width = draw.textlength(text, font=font_cardinal)
        x_centered = x - text_width // 2
        y_centered = y - text_width * 1.75 // 2 + pad  # optional: to also center vertically
        draw.text((x_centered, y_centered), text, font=font_cardinal, fill=cardinal_color)
        
        y = self.output_size - y
        text="S"
        x_centered = x - text_width // 2
        y_centered = y - text_width * 1.75 // 2 - pad # optional: to also center vertically
        draw.text((x_centered, y_centered), text, font=font_cardinal, fill=cardinal_color)

        
        # --- FINAL SAVE ---
        
        # Save the final PIL image
        #base_frame_pil.convert("RGB").save(output_path) # takes 1.4sec at 4000px res!
        
        draw = ImageDraw.Draw(base_frame_pil)
        final_save_cv2 = self.pil_to_cv2_bgra(base_frame_pil)
        final_save_cv2_uint8 = np.clip(final_save_cv2, 0, 255).astype(np.uint8)
        output_path_str = f"{str(self.output_path)}/{str(self.frame_number).zfill(8)}.png"
        print(output_path_str)
        # Remove alpha channel if present
        if final_save_cv2_uint8.shape[2] == 4:
          final_save_cv2_uint8 = final_save_cv2_uint8[:, :, :3]  # keep only RGB

        success = cv2.imwrite(output_path_str, final_save_cv2_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 4])

        # print(f"Successfully composited frame {self.frame_number} and saved to {output_path}")
        self.time_check("saving PIL final image ")

if __name__ == '__main__':
    print("use batch_compositor.py for multiple frame processing from the command line")