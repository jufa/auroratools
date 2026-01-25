import cv2
import numpy as np
import tifffile as tiff
from pathlib import Path
import argparse
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from time import perf_counter


class CFAProcessor:

    # ===== Sony ZV-E10 constants =====

    bits_per_sample = 12
    black_level = 512
    white_level = (2 ** bits_per_sample) - 1  # 4095

    # WB RGGB Levels from EXIF (normalized to G=1.0)
    wb_rggb = np.array([
        2396 / 1024,  # R
        1.0,          # G
        1.0,          # G
        2188 / 1024   # B
    ], dtype=np.float32)

    # Sony color matrix / 1000
    color_matrix = np.array([
        [ 1.280, -0.211, -0.012],
        [ 0.064,  1.151, -0.159],
        [ 0.031, -0.259,  1.284]
    ], dtype=np.float32)

    gamma = 2.22
    bayer_pattern = "RGGB"

    # =================================

    def __init__(self, 
                 exposure=0.0,
                 gamma=2.22,
                 contrast=0.0,
                 saturation=1.0,
                 black_floor = 0.5, # percentile
                 rolloff_threshold=0.8,
                 rolloff_slope=0.05,
                 wb_rggb=None,
                 bpp=8,
                 denoise=False,
                 color_matrix=None):
        """
        gamma: gamma correction value
        saturation: linear-space saturation multiplier
        rolloff_threshold: linear value to start highlight rolloff
        rolloff_slope: controls how aggressive rolloff is
        wb_rggb: 4-element RGGB white balance array (normalized)
        color_matrix: 3x3 color correction matrix
        """
        # gamma & saturation
        self.exposure = exposure
        self.gamma = gamma
        self.contrast = contrast
        self.saturation = saturation
        self.black_floor = black_floor
        self.bpp = bpp
        self.denoise = denoise

        # highlight rolloff
        self.rolloff_threshold = rolloff_threshold
        self.rolloff_slope = rolloff_slope

        # build 1D LUT for 16-bit linear input (0..65535)
        x = np.linspace(0, 1, 65536, dtype=np.float32)
        mask = x > self.rolloff_threshold
        lut = np.copy(x)
        lut[mask] = self.rolloff_threshold + \
            (1 - np.exp(-(x[mask] - self.rolloff_threshold)/self.rolloff_slope)) * (1 - self.rolloff_threshold)
        self.rolloff_lut = lut

        # build contrast curve LUT:
        self.contrast_lut = self.generate_contrast_lut(self.contrast)

        # Exposure LUT:
        self.exposure_lut = self.generate_exposure_lut(self.exposure)

        # WB and color matrix
        if wb_rggb is None:
            self.wb_rggb = CFAProcessor.wb_rggb
        else:
            self.wb_rggb = wb_rggb.astype(np.float32)

        if color_matrix is None:
            self.color_matrix = CFAProcessor.color_matrix
        else:
            self.color_matrix = color_matrix.astype(np.float32)

        # black & white levels for linearization
        self.black_level = CFAProcessor.black_level
        self.white_level = CFAProcessor.white_level

        # Bayer pattern
        self.bayer_pattern = CFAProcessor.bayer_pattern

    def generate_exposure_lut(self, exposure_ev: float) -> np.ndarray:
      """
      Generate a 16-bit LUT for exposure adjustment.
      exposure_ev: exposure in stops (0.0 = no change)
      """
      gain = 2.0 ** exposure_ev

      x = np.linspace(0, 1, 65536, dtype=np.float32)
      x_new = x * gain
      x_new = np.clip(x_new, 0.0, 1.0)

      return x_new

    def generate_contrast_lut(self, contrast: float) -> np.ndarray:
      """
      Generate a 16-bit LUT for contrast adjustment.
      contrast: 0.0 = no change, positive = increase, negative = decrease
      """
      # linear 0..65535
      x = np.linspace(0, 1, 65536, dtype=np.float32)
      
      # S-curve style contrast
      x_new = 0.5 + (x - 0.5) * (1.0 + contrast)
      x_new = np.clip(x_new, 0.0, 1.0)
      
      return x_new

    def extract_tif_bayer_data(self, path: Path) -> np.ndarray:
      data = tiff.imread(path)
      if data.dtype != np.uint16:
        raise ValueError("Expected uint16 Bayer TIFF")
      return data

    def linearize(self, data: np.ndarray) -> np.ndarray:
      """
      Subtract black level and normalize to [0,1]
      """
      data = data.astype(np.float32)
      data -= self.black_level
      data = np.clip(data, 0, None)
      data /= (self.white_level - self.black_level)
      return data
    
    def apply_exposure(self, rgb: np.ndarray) -> np.ndarray:
      """
      Apply exposure LUT (linear space).
      """
      idx = np.clip((rgb * 65535).astype(np.int32), 0, 65535)
      return self.exposure_lut[idx]


    def apply_white_balance_cfa(self, bayer: np.ndarray) -> np.ndarray:
      """
      Apply WB directly on CFA (RGGB)
      """
      out = bayer.copy()

      out[0::2, 0::2] *= self.wb_rggb[0]  # R
      out[0::2, 1::2] *= self.wb_rggb[1]  # G
      out[1::2, 0::2] *= self.wb_rggb[2]  # G
      out[1::2, 1::2] *= self.wb_rggb[3]  # B

      return out

    def demosaic(self, bayer: np.ndarray) -> np.ndarray:
      rgb = demosaicing_CFA_Bayer_Menon2007(
        bayer,
        pattern=self.bayer_pattern
      )
      return rgb.astype(np.float32)

    def apply_color_matrix(self, rgb: np.ndarray) -> np.ndarray:
      h, w, _ = rgb.shape
      reshaped = rgb.reshape(-1, 3)
      corrected = reshaped @ self.color_matrix.T
      return corrected.reshape(h, w, 3)

    def apply_gamma(self, rgb: np.ndarray) -> np.ndarray:
      rgb = np.clip(rgb, 0.0, 1.0)
      return np.power(rgb, 1.0 / self.gamma)
    
    def apply_highlight_rolloff(self, rgb: np.ndarray, threshold=0.8, slope=0.1) -> np.ndarray:
      """
      Compress highlights smoothly.
      rgb : linear float32 array [0,1]
      threshold: linear value where rolloff starts
      slope: how aggressive the rolloff is
      """
      out = rgb.copy()
      mask = out > threshold
      # exponential rolloff
      out[mask] = threshold + (1 - np.exp(-(out[mask] - threshold)/slope)) * (1 - threshold)
      # ensure still in 0..1
      out = np.clip(out, 0.0, 1.0)
      return out
    
    def apply_contrast(self, rgb: np.ndarray) -> np.ndarray:
      """
      Apply LUT-based contrast enhancement
      """
      idx = np.clip((rgb * 65535).astype(np.int32), 0, 65535)
      return self.contrast_lut[idx]
    
    def subtract_black_floor(self, rgb: np.ndarray) -> np.ndarray:
      return np.clip(rgb - self.black_floor, 0.0, 1.0)
    
    def downsample_to_8bit(self, rgb: np.ndarray, dithering=True) -> np.ndarray:
      """
      Convert linear or gamma-corrected 16-bit float RGB [0,1] to 8-bit.
      rgb: float32 array [0,1]
      dithering: apply small random noise to reduce banding
      """
      rgb8 = rgb * 255.0
      if dithering:
          rgb8 += np.random.uniform(-0.5, 0.5, rgb.shape)
      return np.clip(rgb8, 0, 255).astype(np.uint8)

    
    def adjust_saturation_hue_preserving(self, rgb: np.ndarray) -> np.ndarray:
      """
      Hue-preserving saturation adjustment using YCbCr-like method.
      rgb: linear float32 array [0,1]
      alpha: saturation multiplier (1.0 = no change)
      """
      alpha = self.saturation

      R = rgb[..., 0]
      G = rgb[..., 1]
      B = rgb[..., 2]

      # BT.709 luma (linear RGB)
      Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

      # Chroma
      Cb = B - Y
      Cr = R - Y

      # Scale chroma
      Cb *= alpha
      Cr *= alpha

      # Reconstruct RGB
      Rn = Y + Cr
      Bn = Y + Cb
      Gn = (Y - 0.2126 * Rn - 0.0722 * Bn) / 0.7152

      out = np.stack((Rn, Gn, Bn), axis=-1)
      return np.clip(out, 0.0, 1.0)
    
    def apply_denoise(self, rgb: np.ndarray,
            h: float = 1,
            hColor: float = 2,
            templateWindowSize: int = 7,
            searchWindowSize: int = 21,
            bit_depth: int = 8) -> np.ndarray:
      """
      Apply OpenCV fastNlMeansDenoisingColored to linear RGB image.

      Args:
          rgb: float32 array [0,1], shape (H,W,3)
          h: filter strength for luminance
          hColor: filter strength for color
          templateWindowSize: patch size for comparison
          searchWindowSize: window size for search
          bit_depth: 8 or 16 (output type for OpenCV denoising)

      Returns:
          denoised float32 array [0,1], same shape as input
      """
      if rgb.shape[2] != 3:
          raise ValueError("Input RGB must have 3 channels (no alpha).")

      if bit_depth == 8:
          # Convert to CV_8UC3
          img = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
      elif bit_depth == 16:
          # Convert to CV_16UC3
          img = np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)
      else:
          raise ValueError("bit_depth must be 8 or 16.")

      # Apply OpenCV denoising
      denoised = cv2.fastNlMeansDenoisingColored(
          img,
          None,
          h=h,
          hColor=hColor,
          templateWindowSize=templateWindowSize,
          searchWindowSize=searchWindowSize
      )

      # Convert back to float32 [0,1]
      if bit_depth == 8:
          return denoised.astype(np.float32) / 255.0
      else:
          return denoised.astype(np.float32) / 65535.0



    def to_uint16(self, rgb: np.ndarray) -> np.ndarray:
      return np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)
    
    def perf_counter_pretty(self, start, msg):
      print(f"{((perf_counter()-start) * 1000):>8.0f} ms\t{msg}")

    def process_tif(self, src: Path, dst: Path):

      s = perf_counter()
      start = s
      bayer_u16 = self.extract_tif_bayer_data(src)
      self.perf_counter_pretty(s, "extract_tif_bayer_data")

      s = perf_counter()
      bayer = self.linearize(bayer_u16)
      self.perf_counter_pretty(s, "linearize")

      s = perf_counter()
      bayer = self.apply_white_balance_cfa(bayer)
      self.perf_counter_pretty(s, "apply_white_balance_cfa")
      
      s = perf_counter()
      rgb = self.demosaic(bayer)
      self.perf_counter_pretty(s, "demosaic")

      s = perf_counter()
      rgb = self.apply_color_matrix(rgb)
      self.perf_counter_pretty(s, "apply_color_matrix")

      s = perf_counter()
      rgb = self.apply_exposure(rgb)
      self.perf_counter_pretty(s, "apply_exposure")
      
      s = perf_counter()
      rgb = self.adjust_saturation_hue_preserving(rgb)
      self.perf_counter_pretty(s, "adjust_saturation")
      
      s = perf_counter()
      rgb = self.apply_highlight_rolloff(rgb, threshold=0.8, slope=0.05)
      self.perf_counter_pretty(s, "apply_highlight_rolloff")

      s = perf_counter()
      rgb = self.apply_contrast(rgb)
      self.perf_counter_pretty(s, "apply_contrast")

      s = perf_counter()
      rgb = self.subtract_black_floor(rgb) 
      self.perf_counter_pretty(s, "subtract_black_floor")
      
      if self.denoise:
        if self.bpp != 8:
           print(f"WARNING: could not apply denoise, it only works at 8bpp, not {self.bpp}bpp. Skipping.")
        s = perf_counter()
        rgb = self.apply_denoise(rgb)
        self.perf_counter_pretty(s, "denoise")
      
      s = perf_counter()
      rgb = self.apply_gamma(rgb)
      self.perf_counter_pretty(s, "apply_gamma")
      
      if self.bpp == 8:
        s = perf_counter()
        out = self.downsample_to_8bit(rgb)
        self.perf_counter_pretty(s, "downsample_to_8bit")
      else:
        s = perf_counter()
        out = self.to_uint16(rgb)
        self.perf_counter_pretty(s, "to_uint16")
      
      s = perf_counter()
      cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
      self.perf_counter_pretty(s, "imwrite")

      self.perf_counter_pretty(start, "TOTAL")


if __name__=="__main__":

  """
  example use:
  python cfa_processor.py --src ./sampleimages/00000007-20260125.tif --dst ./sampleimages/00000007-20260125.png

  CFAProcessor Pipeline Overview (Aurora / Timelapse Friendly)

  Bayer Raw Input (from 12-bit Sony ARW)
  [Linearization / EXIF Black Subtraction]  <-- subtract sensor black level
  [White Balance]  <-- per-channel WB from EXIF or fixed
  [Demosaic]  <-- color interpolation (Menon2007)
  [Color Matrix]  <-- camera-specific 3x3 transform
  [Exposure Adjustment]  <-- linear scale to brighten/darken
  [Hue-Preserving Saturation]  <-- boosts color without hue drift
  [Highlight Rolloff LUT]  <-- compress bright highlights smoothly
  [Contrast LUT]  <-- contrast enhancement (-/+ adjustable)
  [Black Floor Subtraction]  <-- fixed, frame-independent for timelapse
  [Noise Reduction]  <-- OpenCV fastNlMeansDenoisingColored, linear space
  [Gamma Correction]  <-- optional, display gamma (1.0=linear, 1.4-1.6=astro video)
  [Output]
    ├─ 16-bit PNG/TIFF for archival
    └─ 8-bit PNG/Video export (optional dithering)

    
    PERFORMANCE (Mac M1)
      35 ms     extract_tif_bayer_data
      20 ms     linearize
       8 ms     apply_white_balance_cfa
    2020 ms     demosaic
      97 ms     apply_color_matrix
      69 ms     apply_exposure
      84 ms     adjust_saturation
      32 ms     apply_highlight_rolloff
      42 ms     apply_contrast
      33 ms     subtract_black_floor
    1804 ms     denoise
     146 ms     apply_gamma
     309 ms     downsample_to_8bit
     109 ms     imwrite
    4809 ms     TOTAL
  """

  parser = argparse.ArgumentParser(
    description="Process CFA Bayer data from ZV-E10 mk I"
  )
  parser.add_argument('--src', required=True, help='Path and name of the source tif image')
  parser.add_argument('--dst', required=True, help='Path and name of the output png16 image')
  args = parser.parse_args()

  ap = CFAProcessor(bpp=8, denoise=True, saturation=2.8, contrast=0.0, gamma=1.6, exposure=0.5, black_floor=0.005)
  ap.process_tif(Path(args.src), Path(args.dst))




