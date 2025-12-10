import cv2
import numpy as np
import argparse
import os

PI = np.pi

def annulus_warp(src_path: str, angular_coverage: float, inner_diameter: int, outer_diameter: int, padding: int) -> None:
    """
    Performs a custom polar warp (rectangle to annulus) with explicit inner/outer diameters,
    using precise conditional mapping for seamless 360-degree closure.
    """
    # 1. Load the image and validate
    src = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if src is None:
        print(f"Error: Could not load image from {src_path}. Check if the file exists and is a valid image.")
        return

    H_in, W_in = src.shape[:2]
    
    # 2. Define Geometry and Canvas
    
    R_inner = inner_diameter // 2
    R_outer = outer_diameter // 2
    
    if R_outer <= R_inner:
        print("Error: Outer diameter must be larger than the inner diameter.")
        return
        
    # Canvas size: diameter of the outer ring plus padding
    final_size = outer_diameter + 2 * padding
    C = final_size // 2  # Center coordinate
    
    # Angular mapping (in radians)
    angular_radians = angular_coverage * (PI / 180.0)
    
    # Calculate the starting angle offset needed to center the arc at the top (PI/2).
    excluded_angle = (2 * PI) - angular_radians
    A_start = excluded_angle / 2.0
    
    # Check for the 360 case precisely
    is_full_circle = (angular_coverage >= 359.99)

    # 3. Create the Inverse Coordinate Map
    
    print(f"Generating precise map for {W_in}x{H_in} image to {angular_coverage}° annulus...")
    
    Y, X = np.mgrid[0:final_size, 0:final_size]

    R = np.sqrt((X - C)**2 + (Y - C)**2)
    Theta = np.arctan2(Y - C, X - C)
    Theta[Theta < 0] += 2 * PI
    
    # 3a. Source Y-coordinate (Radial Mapping) - UNCHANGED
    R_norm = (R - R_inner) / (R_outer - R_inner)
    Y_s = H_in * (1 - R_norm)
    
    # 3b. Source X-coordinate (Angular Mapping) - PRECISE CONDITIONAL LOGIC
    
    if is_full_circle:
        # For a full 360-degree circle, map the entire 2*PI range to [0, W_in].
        # This ignores A_start and ensures perfect wrap-around.
        X_s = (Theta / (2 * PI)) * W_in
        
        # Since we use Theta and not Theta_shifted, the angular mask needs to cover everything
        Theta_shifted = None # Placeholder, not used for mapping or masking below
        mask_angular = np.full_like(R, True, dtype=bool)
    else:
        # For a partial arc, use the shifted and scaled method.
        Theta_shifted = Theta - A_start
        Theta_shifted[Theta_shifted < 0] += 2 * PI 
        
        X_s = (Theta_shifted / angular_radians) * W_in
        mask_angular = (Theta_shifted >= 0) & (Theta_shifted <= angular_radians)
    
    # 4. Define Final Maps and Mask

    # Combine X and Y maps into the required CV_32FC2 format
    map_xy = np.stack((X_s, Y_s), axis=-1).astype(np.float32)

    # Define the final mask conditions
    mask_radial = (R >= R_inner) & (R <= R_outer)
    final_mask_bool = mask_radial & mask_angular
    
    # 5. Apply the Transformation (Remapping)
    
    print("Applying remapping and interpolation...")
    
    # The cv2.remap function performs the final high-performance pixel sampling.
    temp_circular_image = cv2.remap(
        src, 
        map_xy, 
        None,    
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0) # Black where outside the source coordinates
    )
    
    # 6. Create Final Transparent PNG
    
    # Create the Alpha channel based on the boolean mask
    alpha = np.zeros((final_size, final_size), dtype=np.uint8)
    alpha[final_mask_bool] = 255
    
    # Combine the color (BGR) and the alpha channels
    b, g, r = cv2.split(temp_circular_image)
    final_ring_bgr_a = cv2.merge([b, g, r, alpha])
    final_ring_bgr_a = cv2.rotate(final_ring_bgr_a, cv2.ROTATE_90_COUNTERCLOCKWISE)

    print("Warp and transparency complete.")

    # 7. Save the output
    dirname = os.path.dirname(src_path)
    basename, _ = os.path.splitext(os.path.basename(src_path))
    
    output_filename = f"ring_output_{basename}_a{int(angular_coverage)}_i{inner_diameter}_o{outer_diameter}_p{padding}.png"
    output_filename = "keogram_annular.png"
    output_path = os.path.join(dirname, output_filename)
    
    if cv2.imwrite(output_path, final_ring_bgr_a, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        print(f"Output saved successfully to: {output_path}")
    else:
        print(f"Error: Failed to save the output image to {output_path}")

# CLI Setup
def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform a custom polar warp (rectangle to sector ring) with explicit inner/outer diameters.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--path', required=True, help="Path to the input rectangular image file.")
    
    parser.add_argument(
        '--angle', type=float, default=360.0,
        help="The angle in degrees (e.g., 260) that the input image width maps to (default: 359.0). Use 360 for a seamless circle."
    )
    
    parser.add_argument(
        '--inner-diameter', type=int, default=800,
        help="The inner diameter in pixels of the ring (default: 800)."
    )
    
    parser.add_argument(
        '--outer-diameter', type=int, default=1000,
        help="The outer diameter in pixels of the ring."
    )
    
    parser.add_argument(
        '--padding', type=int, default=0,
        help="Number of transparent pixels to pad around the outside of the ring (default: 0)."
    )

    args = parser.parse_args()
    
    if args.inner_diameter <= 0 or args.outer_diameter <= 0 or args.padding < 0 or args.angle <= 0 or args.angle > 360:
        print("Error: Invalid diameter, padding, or angle values.")
        return
    
    if args.inner_diameter >= args.outer_diameter:
        print("Error: The outer diameter must be strictly greater than the inner diameter.")
        return

    annulus_warp(args.path, args.angle, args.inner_diameter, args.outer_diameter, args.padding)

if __name__ == "__main__":
    import argparse
    main()