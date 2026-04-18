import os
import rawpy
import numpy as np
import scipy.stats as ss
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # non-interactive, file output only — no Tk needed
import matplotlib.pyplot as plt
import imageio

"""
This demo takes in sony ARW files in a folder with all sky camera circular fisheye images in the center 4000x4000 px region
it uses either median or skewed gaussian curve fit to determin an estimate per channel blackpoint for each image
and corrects the blackpoint using these values, then saves a postprocessed JPG for visual comparison. The histograms of the sampled blackpoint areas are plotted with the estimated black levels marked.
"""

print(f"rawpy version: {rawpy.__version__}")

USE_GAUSSIAN_FIT = True
USE_NOISE_REDUCTION = False
EXPOSURE_SHIFT = 1.0  # 1.0=no change,
GAMMA = 1.6
GAMMA_SLOPE_CUTOFF = 0.3  # higher values for darker lowlights

def read_arw_and_extract_channels(file_path):
  # Read the raw file without debayering
  with rawpy.imread(file_path) as raw:
    # Get the raw Bayer pattern data
    raw_data = raw.raw_image.copy()
    # optical_black_data = raw_data[0:20, 0:20]  # Example: top-left 10x10 area for black level
    # print(f"Optical black data (10x10):\n{optical_black_data}")

    print(f"Raw data shape: {raw_data.shape}, dtype: {raw_data.dtype}")
    
    # Get the Bayer pattern info
    pattern = raw.color_desc.decode('utf-8')  # Usually 'RGGB'
    
    # Extract RGGB channels using array slicing (optimized)
    # Assuming RGGB pattern starting at (0,0)
    R = raw_data[0::2, 0::2].astype(np.float32)   # Red
    G1 = raw_data[0::2, 1::2].astype(np.float32)  # Green 1
    G2 = raw_data[1::2, 0::2].astype(np.float32)  # Green 2  
    B = raw_data[1::2, 1::2].astype(np.float32)   # Blue
    
    # border in channel-space: 6px full-res top/bottom = 3ch, 12px full-res left/right = 6ch
    BORDER_V = int((4024-4000)/2/2)   # vertical border in channel pixels (6048x4032), bayer halved per channel, halved again for left/right/top bottom
    BORDER_H = int((6048-5000)/2/2)   # horizontal border in channel pixels

    # channel dims are half of full res: ~1500x1000 for 3000x2000 sensor
    ch_h, ch_w = R.shape

    MARGIN = 20   # 20px full-res / 2
    SAMPLE = 400   #
    # bottom right region
    sample_mask = np.zeros((ch_h, ch_w), dtype=bool)
    # bottom right region
    sample_mask[ch_h - MARGIN - SAMPLE : ch_h - MARGIN,
                ch_w - MARGIN - SAMPLE : ch_w - MARGIN] = True
    # bottom left region
    sample_mask[MARGIN : MARGIN + SAMPLE,
                ch_w - MARGIN - SAMPLE : ch_w - MARGIN] = True
    # top left region
    sample_mask[MARGIN : MARGIN + SAMPLE,
                MARGIN : MARGIN + SAMPLE] = True
    # top right region
    sample_mask[ch_h - MARGIN - SAMPLE : ch_h - MARGIN,
                MARGIN : MARGIN + SAMPLE] = True
    
    sample_R  = R[sample_mask]
    sample_G1 = G1[sample_mask]
    sample_G2 = G2[sample_mask]
    sample_B  = B[sample_mask]
    
    # Calculate median values
    reference = 512
    median_R =  np.median(sample_R)  - reference
    median_G1 = np.median(sample_G1) - reference
    median_G2 = np.median(sample_G2) - reference
    median_B =  np.median(sample_B)  - reference
    median_G = (median_G1 + median_G2) / 2.0

    # calculate using parabolic fit:
    if USE_GAUSSIAN_FIT:
      estimate_black_R,  popt_R  = estimate_black(sample_R)
      estimate_black_G1, popt_G1 = estimate_black(sample_G1)
      estimate_black_G2, popt_G2 = estimate_black(sample_G2)
      estimate_black_B,  popt_B  = estimate_black(sample_B)
      estimate_black_R  -= reference
      estimate_black_G1 -= reference
      estimate_black_G2 -= reference
      estimate_black_B  -= reference
      estimate_black_G = (estimate_black_G1 + estimate_black_G2) / 2.0

  with rawpy.imread(file_path) as raw:
    raw_visible = raw.raw_image_visible.astype(np.float32)

    if USE_GAUSSIAN_FIT:
      raw_visible[0::2, 0::2] -= (estimate_black_R)
      raw_visible[0::2, 1::2] -= (estimate_black_G)
      raw_visible[1::2, 0::2] -= (estimate_black_G)
      raw_visible[1::2, 1::2] -= (estimate_black_B)
    else:
      raw_visible[0::2, 0::2] -= (median_R)
      raw_visible[0::2, 1::2] -= (median_G)
      raw_visible[1::2, 0::2] -= (median_G)
      raw_visible[1::2, 1::2] -= (median_B)

    # clip to 0 (handles negative values after subtraction)
    np.clip(raw_visible, 0, 16383, out=raw_visible)

    # write back as uint16
    raw.raw_image_visible[:] = raw_visible.astype(np.uint16)

    jpg_path = file_path.replace('.ARW', '_corrected_output.jpg') 
    rgb = raw.postprocess(
      exp_shift=EXPOSURE_SHIFT,  # 1.0=no change, 2.0=1 stop up, 4.0=2 stops up
      exp_preserve_highlights=1.0,  # 0.0-1.0, prevents highlight clippin
      gamma=(GAMMA, GAMMA_SLOPE_CUTOFF),
      user_black=reference,  # already subtracted
      use_camera_wb=True,
      no_auto_bright=True,
      output_bps=8,
      # fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Full, # Apply FBDD
      noise_thr=(500.0 if USE_NOISE_REDUCTION else 0), # Set wavelet threshold
      median_filter_passes=(1 if USE_NOISE_REDUCTION else 0) # Remove color artifacts
    )
    imageio.imwrite(jpg_path, rgb)

    
    return {
      'channels': (R, G1, G2, B),
      'channels_sample': (sample_R, sample_G1, sample_G2, sample_B),
      'medians': (median_R, median_G1, median_G2, median_G, median_B),
      'estimates': (estimate_black_R, estimate_black_G1, estimate_black_G2, estimate_black_G, estimate_black_B),
      'popts': (popt_R, popt_G1, popt_G2, popt_B),
      'pattern': pattern
    }

def estimate_black(channel_corner):
    """
    1. Bin sample into 4-DN histogram
    2. Fit skewed Gaussian to the binned counts
    3. Return (peak_x, popt) where popt is the fitted parameters (amp, loc, scale, alpha)
       or None if fitting failed.
    """
    flat = channel_corner.flatten().astype(np.float64)

    bins    = np.arange(flat.min(), flat.max() + 4, 4)
    counts, edges = np.histogram(flat, bins=bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2

    def skewed_gaussian(x, amp, loc, scale, alpha):
        return amp * ss.skewnorm.pdf(x, alpha, loc, scale)

    try:
        init_scale = max(np.std(flat), 4.0)  # never let scale start below 1 bin width
        p0 = [counts.max(), centres[np.argmax(counts)], init_scale, 1.0]
        bounds = (
            [0,       flat.min(), 1.0,  -20],   # lower: amp>0, loc in range, scale>=1, alpha free
            [np.inf,  flat.max(), np.inf, 20],   # upper
        )
        popt, _ = curve_fit(skewed_gaussian, centres, counts, p0=p0, bounds=bounds, maxfev=5000)
        amp, loc, scale, alpha = popt
        x   = np.linspace(flat.min(), flat.max(), 10_000)
        pdf = skewed_gaussian(x, amp, loc, scale, alpha)
        print(f"  fit: amp={amp:.4f}, loc={loc:.2f}, scale={scale:.2f}, alpha={alpha:.2f}")
        return float(x[np.argmax(pdf)]), popt
    except Exception as e:
        print(f"  curve_fit failed: {e} — falling back to peak bin")
        return float(centres[np.argmax(counts)]), None


def plot_histograms(results: list[tuple[str, dict]], reference: int = 512):
    """
    Plot overlaid histograms of R, G1, G2, B samples for a list of ARW results.
    results: list of (filename, result_dict) tuples
    """

    def skewed_gaussian(x, amp, loc, scale, alpha):
        return amp * ss.skewnorm.pdf(x, alpha, loc, scale)
    
    if USE_GAUSSIAN_FIT:
      fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)), squeeze=False)

    for idx, (filename, result) in enumerate(results):
        R, G1, G2, B = result['channels_sample']
        R_est, G1_est, G2_est, G_est, B_est = result['estimates']
        if USE_GAUSSIAN_FIT:
          R_popt, G1_popt, G2_popt, B_popt = result['popts']
        ax = axes[idx][0]

        bins = np.arange(-64, 256, 4)  # relative to black level, 1-DN bins

        ax.hist((R  - reference).flatten(), bins=bins, color='red',   alpha=1.0, label='R',  density=True, histtype='step', linewidth=1.2)
        ax.hist((G1 - reference).flatten(), bins=bins, color='green', alpha=1.0, label='G1', density=True, histtype='step', linewidth=1.2)
        ax.hist((G2 - reference).flatten(), bins=bins, color='lime',  alpha=1.0, label='G2', density=True, histtype='step', linewidth=1.2)
        ax.hist((B  - reference).flatten(), bins=bins, color='blue',  alpha=1.0, label='B',  density=True, histtype='step', linewidth=1.2)

        # Plot fitted skewed Gaussian curves
        if USE_GAUSSIAN_FIT:
          x_fit = np.linspace(-64, 255, 2000) + reference  # in raw DN space to match popt loc
          for popt, color in [(R_popt, 'red'), (G1_popt, 'green'), (G2_popt, 'lime'), (B_popt, 'blue')]:
              if popt is not None:
                  y_fit = skewed_gaussian(x_fit, *popt)
                  ax.plot(x_fit - reference, y_fit, color=color, linewidth=1.0, alpha=0.6, linestyle='-')

        # estimate peak black level from histogram and mark with vertical line:
        ax.axvline(R_est,  color='red',   linewidth=1.0, linestyle='-', alpha=0.7)
        ax.axvline(G1_est, color='green', linewidth=1.0, linestyle='--', alpha=0.7)
        ax.axvline(G2_est, color="lime",  linewidth=1.0, linestyle=':', alpha=0.7)
        ax.axvline(B_est,  color='blue',  linewidth=1.0, linestyle='-.', alpha=0.7)

        ax.set_title(f"{filename} {"Guassian fit" if USE_GAUSSIAN_FIT else "Median fit"}", color='white')
        ax.set_xlabel('DN (relative to black level 512)', color='white')
        ax.set_ylabel('Density', color='white')
        ax.legend(loc='upper right', facecolor='#222222', labelcolor='white')
        ax.set_xlim(-64, 128)
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_facecolor("#000000")
        ax.tick_params(colors='white')
        ax.spines[:].set_color('white')

        tick_positions = np.arange(-64, 129, 8)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(t) for t in tick_positions], rotation=45, ha='right', fontsize=7)

    fig.patch.set_facecolor('#111111')
    plt.tight_layout()
    plt.savefig('/Volumes/T7 Shield/AurorEye/Unit12 WB test/20260409 ARW+JPG PP OFF ISO6400 2s/test/histograms.png',
                dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    print("Histogram saved to histograms.png")

def summarize(path):
  def color(r,g,b):
    RED   = "\033[31m"
    GREEN = "\033[32m"
    BLUE  = "\033[34m"
    RESET = "\033[0m"

    r_str = f"{RED}R: {r:6.2f}{RESET}"
    g_str = f"{GREEN}G: {g:6.2f}{RESET}"
    b_str = f"{BLUE}B: {b:6.2f}{RESET}"
    return f"{r_str}, {g_str}, {b_str}"
  
  result = read_arw_and_extract_channels(path)
  R, G1, G2, B = result['channels']
  med_R, med_G1, med_G2, med_G, med_B = result['medians']
  est_R, est_G1, est_G2, est_G, est_B = result['estimates']

  print(f"{path.split('/')[-1]}",end="\n")
  # print(f"sample blackpoint area {x=}, {y=}, {size=} ")
  print(f"Median values   - {color(med_R, med_G, med_B)}")
  print(f"Estimate values - {color(est_R, est_G, est_B)}")

  return result['medians'], result

# Usage example:

files = os.listdir('/Volumes/T7 Shield/AurorEye/Unit12 WB test/20260409 ARW+JPG PP OFF ISO6400 2s/test/')
arw_files = [f for f in files if f.lower().endswith('.arw')]
print(f"Found ARW files: {arw_files}")  

BLACK_LEVEL = 512
histogram_data = []
for arw_file in arw_files:
  medians, result = summarize(f'/Volumes/T7 Shield/AurorEye/Unit12 WB test/20260409 ARW+JPG PP OFF ISO6400 2s/test/{arw_file}')
  histogram_data.append((arw_file, result))
plot_histograms(histogram_data)



"""
ARW image: /Volumes/T7 Shield/AurorEye/Unit12 WB test/20260409 ARW+JPG PP OFF ISO6400 2s/test/capt_DSC08424.ARW
sample blackpoint area x=300, y=300, size=300 
Median values - R: 516.00, G1: 516.00, G2: 516.00, Gavg: 516.00, B: 512.00
ARW image: /Volumes/T7 Shield/AurorEye/Unit12 WB test/20260409 ARW+JPG PP OFF ISO6400 2s/test/capt_DSC07400.ARW
sample blackpoint area x=300, y=300, size=300 
Median values - R: 512.00, G1: 520.00, G2: 516.00, Gavg: 518.00, B: 516.00
"""
"""
capt_DSC07400.ARW       Median values - R:      4, Gavg:     16, B:      4
capt_DSC03304.ARW       Median values - R:      8, Gavg:     18, B:      4
capt_DSC08424.ARW       Median values - R:     12, Gavg:     16, B:      4
capt_DSC02280.ARW       Median values - R:      8, Gavg:     14, B:      0
capt_DSC06376.ARW       Median values - R:      4, Gavg:     16, B:      4
capt_DSC09208.ARW       Median values - R:     92, Gavg:    166, B:     60
capt_DSC04328.ARW       Median values - R:      8, Gavg:     14, B:      4
capt_DSC05352.ARW       Median values - R:      8, Gavg:     14, B:      4
capt_DSC01256.ARW       Median values - R:      8, Gavg:     16, B:      4

Note on quantization values:
Hypothesis: we are seeing minimum step of 4 since we are getting 12 out of 14 possible bits in the ADC
for electronic shutter. loss of two bits means steps of 4.
"""