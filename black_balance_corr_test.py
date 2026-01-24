import cv2
import numpy as np


"""
This demo test is to adjust the JPG output of the ZVE-10
The issue seems to be that even with colour temp WB, there is still frame to frame variance in the black point

For the relatively flat picture profile 3, this can be mitigated with the algorithm below,
effectively adjusting the y-intercept of the curve adjustment, and keeping 255 pinned for all channels

The correction_factor scales the adjustment and is empirical. 
0 = no adjust, 1 = full adjust

The sample used is the top left 100x100px area of the images which is a 
pseudo dark frame outside the image circle (except for massive overexposure)

Designed to work on output JPGs 
"""
def adjust_to_match_blacklift(image_A, image_B, sample_size=100, correction_factor=1.0, set_level=[0,0,0]):
    hA, wA, _ = image_A.shape
    hB, wB, _ = image_B.shape

    sample_A = image_A[0:sample_size, wA - sample_size:wA]
    sample_B = image_B[0:sample_size, wB - sample_size:wB]

    mean_A = sample_A.mean(axis=(0, 1))  # BGR
    mean_B = sample_B.mean(axis=(0, 1))

    # Delta (A - B)
    if set_level:
        delta = mean_A - set_level 
    else:
        delta = mean_A = mean_B
    delta = delta * correction_factor

    # Black lift per channel
    black = -delta
    black = np.clip(black, -128, 128)

    # Prepare output
    adjusted_channels = []

    x = np.arange(256, dtype=np.float32)

    # Split channels
    channels = cv2.split(image_A)

    for c in range(3):
        b = black[c]
        slope = (255.0 - b) / 255.0

        y = slope * x + b
        lut = np.clip(y, 0, 255).astype(np.uint8)

        adjusted_c = cv2.LUT(channels[c], lut)
        adjusted_channels.append(adjusted_c)

    adjusted = cv2.merge(adjusted_channels)
    return adjusted

def show_half_overlay(adjusted, image_B):
    """
    Display a half-and-half overlay: top half image_B, bottom half adjusted.
    Both images must be the same shape.
    """
    assert adjusted.shape == image_B.shape, "Images must be same shape for overlay"

    h, w, c = adjusted.shape
    mid = h // 2

    # Create overlay
    overlay = np.zeros_like(adjusted)
    overlay[0:mid, :, :] = image_B[0:mid, :, :]      # top half B
    overlay[mid:h, :, :] = adjusted[mid:h, :, :]     # bottom half adjusted

    # Show
    cv2.imshow("Overlay: top=B, bottom=Adjusted", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return adjusted




image_A = cv2.imread("./sampleimages/00000687.jpg", cv2.IMREAD_COLOR)
image_B = cv2.imread("./sampleimages/00001728.jpg", cv2.IMREAD_COLOR)

if image_A is None or image_B is None:
    raise IOError("Failed to load one or both images")

# Adjust image_A to match image_B
adjusted_A = adjust_to_match_blacklift(image_A, image_B, set_level=[0,0,0])
show_half_overlay(adjusted_A, image_B)

# (Optional) save result
cv2.imwrite("./sampleimages/00000687_adjusted.jpg", adjusted_A)