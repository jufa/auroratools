import rawpy
import cv2
import numpy as np
from pathlib import Path
import argparse
from scipy.ndimage import uniform_filter
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
import subprocess

class ARWProcessor:
  GAMMA_SRGB = (2.222, 4.5)

  # def __init__(self):
  #   print("Building Gamma LUT...", "")
  #   self.gamma_lut_16 = self.make_gamma_lut()
  #   print("done")

  # def make_gamma_lut(self, bits_in=16, gamma=2.2):
  #   max_in = (1 << bits_in) - 1
  #   x = np.arange(max_in + 1, dtype=np.float32) / max_in
  #   y = np.power(x, 1.0 / gamma)
  #   return np.clip(y * 255, 0, 255).astype(np.uint8)

  def extract_ccm_from_exiftool(self, arw_path, matrix_tag='ColorMatrix'):
    """
    Extract Sony Color Matrix (CCM) from an ARW file using exiftool.
    Returns a 3x3 numpy float matrix scaled to normal float values.

    Parameters:
    - arw_path: path to Sony ARW file
    - matrix_tag: 'ColorMatrix1' or 'ColorMatrix2'

    Returns:
    - ccm: 3x3 numpy float array, scaled by 1/1000
    """
    try:
      # Run exiftool and capture output
      result = subprocess.run(
        ['exiftool', '-b', f'-{matrix_tag}', arw_path],
        capture_output=True,
        text=True,
        check=True
      )
      raw_text = result.stdout.strip()
      if not raw_text:
        raise ValueError(f"{matrix_tag} not found in {arw_path}")

      # Split into integers
      numbers = [int(x) for x in raw_text.split()]
      if len(numbers) < 9:
        raise ValueError(f"{matrix_tag} has fewer than 9 values: {numbers}")

      # Take first 9 numbers, reshape into 3x3
      ccm = np.array(numbers[:9], dtype=float).reshape((3,3))
      print(ccm)

      return ccm

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Error running exiftool:", e)
        # fallback to identity
        return np.eye(3, dtype=float)
    except ValueError as ve:
        print("Warning:", ve)
        return np.eye(3, dtype=float)
    
  def prepare_ccm_for_pidng(self, ccm_float, scale=1000):
    """
    Convert a 3x3 float CCM into the 3x3x2 format expected by piDNG:
    [[value, scale], ...]
    
    Parameters:
    - ccm_float: 3x3 numpy float array
    - scale: integer scale factor (default 10000)
    
    Returns:
    - ccm_pidng: list of 9 [value, scale] pairs flattened for DNGTags
    """
    ccm_pidng = []
    for row in ccm_float:
        for val in row:
            # Convert to integer using the scale
            int_val = int(round(val * scale))
            ccm_pidng.append([int_val, scale])
    return ccm_pidng
  
  def float_to_uint16_bayer(self, bayer_float, target_bits=16):
    """
    Convert a float binned Bayer image to uint16 for piDNG.

    Parameters:
    - bayer_float: numpy array of shape (H,W), float type
    - target_bits: 8, 12, or 16. Specifies how many bits to keep.

    Returns:
    - bayer_uint16: numpy uint16 array ready for RAW2DNG.convert()
    """
    if not issubclass(bayer_float.dtype.type, np.floating):
      raise TypeError("Input must be a float numpy array.")

    if target_bits > 16 or target_bits < 8:
      raise ValueError("target_bits must be between 8 and 16.")

    # First, clip to full 16-bit range
    bayer_clipped = np.clip(bayer_float, 0, 2**16 - 1)

    # If reducing bits, shift right
    if target_bits < 16:
      shift = 16 - target_bits
      bayer_clipped = np.right_shift(bayer_clipped.astype(np.uint16), shift)
    else:
      bayer_clipped = bayer_clipped.astype(np.uint16)

    return bayer_clipped

  def bayer_aware_bin(self, raw_bayer):
    """
    Bayer-aware 2x2 binning for RGGB Bayer array.
    raw_bayer: 2D numpy array (16-bit)
    Returns: binned Bayer array
    """
    # raw_bayer shape: H x W
    H, W = raw_bayer.shape
    
    R = raw_bayer[0:H:2, 0:W:2]
    G1 = raw_bayer[0:H:2, 1:W:2]
    G2 = raw_bayer[1:H:2, 0:W:2]
    B = raw_bayer[1:H:2, 1:W:2]
    
    # R, G1, G2, B: each H/2 x W/2
    # Apply 2x2 mean filter
    R_bin = uniform_filter(R.astype(np.float32), size=2)[::2, ::2]
    G1_bin = uniform_filter(G1.astype(np.float32), size=2)[::2, ::2]
    G2_bin = uniform_filter(G2.astype(np.float32), size=2)[::2, ::2]
    B_bin = uniform_filter(B.astype(np.float32), size=2)[::2, ::2]
  
    H, W = R_bin.shape
    binned = np.zeros((H*2, W*2), dtype=np.float32)
    
    binned[0::2, 0::2] = R_bin
    binned[0::2, 1::2] = G1_bin
    binned[1::2, 0::2] = G2_bin
    binned[1::2, 1::2] = B_bin
    
    return binned

  def extract_sony_params(self, raw):

    metadata = {}
    
    # Black/White levels
    metadata["black"] = raw.black_level_per_channel.copy()
    metadata["white"] = raw.white_level
    
    # Camera WB multipliers (Kelvin mode)
    metadata["wb_multipliers"] = raw.camera_whitebalance.copy()  # [R, G, B]
    
    # Color matrices
    metadata["color_matrix"] = raw.color_matrix.copy()           # ColorMatrix1 / ColorMatrix2
    # metadata["calib_matrix"] = raw.calibration_matrix.copy()     # CameraCalibration1 / 2
    
    # CFA pattern
    metadata["cfa_pattern"] = raw.raw_pattern  # RGGB = [[0,1],[1,2]] or similar
    
    return metadata

  def processARW(self, path:Path):
    dir_path = path.parent
    file_name = path.name
    print(f"parsing image {path}")
    ccm = self.extract_ccm_from_exiftool(str(path))
    scale = 1000
    ccm_with_scale = self.prepare_ccm_for_pidng(ccm)

    with rawpy.imread(str(path)) as raw:
      raw = rawpy.imread(str(path)) 
      bayer = raw.raw_image
      
      binned_bayer = self.bayer_aware_bin(bayer)
      binned_bayer_uint16 = self.float_to_uint16_bayer(binned_bayer)
      metadata = self.extract_sony_params(raw)
      wbm = metadata["wb_multipliers"]
      print(metadata)
      print(metadata["color_matrix"])
      w = 3000
      h = 2000

      t = DNGTags()
      t.set(Tag.ImageWidth, w)
      t.set(Tag.ImageLength, h)
      t.set(Tag.TileWidth, w)
      t.set(Tag.TileLength, h)
      t.set(Tag.Orientation, Orientation.Horizontal)
      t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
      t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
      t.set(Tag.CFARepeatPatternDim, [2,2])
      t.set(Tag.CFAPattern, CFAPattern.RGGB)  # or whatever pattern
      t.set(Tag.BitsPerSample, 12)            # if packing 12-bit
      t.set(Tag.BlackLevel, metadata["black"])
      t.set(Tag.WhiteLevel, metadata["white"])
      t.set(Tag.ColorMatrix1, ccm_with_scale)           # from rawpy
      # t.set(Tag.CameraCalibration1, metadata["calib_matrix"])
      t.set(Tag.AsShotNeutral, [[1,int(wbm[0])],[1,int(wbm[2])],[1,int(wbm[3])]]) # 1/WB
      
      r = RAW2DNG()
      r.options(t, path="", compress=True)
      r.convert(binned_bayer_uint16, filename=Path(dir_path, f"{file_name}_proc.png"))
      # RAW2DNG(binned_bayer, Path(dir_path, f"{file_name}_proc.png"), metadata=dng_meta)



      # raw.raw_image[:] = binned_bayer

      # rgb = raw.postprocess(
      #   gamma=ARWProcessor.GAMMA_SRGB,
      #   no_auto_bright=True,
      #   bright=1.0,
      #   use_camera_wb=True,
      #   use_auto_wb=False,
      #   demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
      #   output_bps=8,
      #   output_color=rawpy.ColorSpace.sRGB
      # )
      # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
      # cv2.imwrite(Path(dir_path, f"{file_name}_proc.png"), bgr)

if __name__=="__main__":
  parser = argparse.ArgumentParser(
    description="Process Sony ARW files, specifically from ZV-E10 mk I"
  )
  parser.add_argument('--image', required=True, help='Path to the JPEG image')
  args = parser.parse_args()

  ap = ARWProcessor()
  ap.processARW(Path(args.image))




