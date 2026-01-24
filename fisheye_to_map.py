import numpy as np
import cv2
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use('Agg')  # headless backend, no Tkinter needed
from matplotlib import pyplot as plt



"""
mapper = FisheyeAuroraMapper(
    xc=2025, yc=2010, r_max=2000,
    alt_poly_coeffs=[-49.50085744, 62.16763995, -91.67967310, 94.83971196],
    observer_latlon=(65.0, -147.5),
    feature_altitude=110_000
)

output_img = mapper.project_image(fisheye_img,
                                  output_dims=(1024,1024),
                                  input_radius_cutoff=1950,
                                  include_map=True,
                                  include_lat_lon=10,
                                  interpolation='bilinear')
"""

class FisheyeAuroraMapper:
    
    COEFFS_MEIKE65_SONYZVE10 = [-49.50085744, 62.16763995, -91.67967310, 94.83971196]

    def __init__(self, xc, yc, r_max, alt_poly_coeffs, observer_latlon, feature_altitude):
        """
        xc, yc : optical center (pixels)
        r_max : maximum radius in pixels for normalization
        alt_poly_coeffs : 3rd-order polynomial coefficients (list/tuple)
        observer_latlon : (lat, lon) in fractional degrees
        feature_altitude : altitude of aurora features in meters
        """
        self.xc = xc
        self.yc = yc
        self.r_max = r_max
        self.alt_poly_coeffs = alt_poly_coeffs
        self.observer_latlon = observer_latlon
        self.feature_altitude = feature_altitude
        self.earth_radius = 6371000  # meters

    # -------------------------
    # Core mapping functions
    # -------------------------
    def pixel_to_alt(self, x, y, input_radius_cutoff=None):
        """Compute altitude (deg) from pixel position using normalized radius
           Optionally mask pixels outside input_radius_cutoff (pixels from center)"""
        r = np.sqrt((x - self.xc)**2 + (y - self.yc)**2)
        if input_radius_cutoff is not None and r > input_radius_cutoff:
            return 0.0  # treated as invalid, will appear black
        rho = r / self.r_max
        return np.polyval(self.alt_poly_coeffs, rho)

    def slant_distance_from_altitude_deg(self, alt_deg):
        alpha = np.deg2rad(alt_deg)
        R = self.earth_radius
        h = self.feature_altitude
        return -R*np.sin(alpha) + np.sqrt((R*np.sin(alpha))**2 + h*(2*R+h))

    def ground_range_from_altitude_deg(self, alt_deg):
        """Horizontal distance along Earth's surface from observer"""
        alpha = np.deg2rad(alt_deg)
        s = self.slant_distance_from_altitude_deg(alt_deg)
        R = self.earth_radius
        return R * np.arcsin((s * np.cos(alpha)) / (R + self.feature_altitude))

    def pixel_to_latlon(self, x, y, projection_center=None):
        """
        Convert pixel to lat/lon.
        projection_center: tuple (lat, lon). If None, observer's location is center.
        """
        alt_deg = self.pixel_to_alt(x, y)
        az_deg = np.rad2deg(np.arctan2(x - self.xc, -(y - self.yc))) % 360  # North = top

        s = self.slant_distance_from_altitude_deg(alt_deg)
        lat0, lon0 = self.observer_latlon if projection_center is None else projection_center
        R = self.earth_radius

        # Spherical Earth approximation
        lat = np.rad2deg(np.arcsin(
            np.sin(np.deg2rad(lat0)) * np.cos(s/R) +
            np.cos(np.deg2rad(lat0)) * np.sin(s/R) * np.cos(np.deg2rad(az_deg))
        ))

        lon = lon0 + np.rad2deg(
            np.arctan2(
                np.sin(np.deg2rad(az_deg)) * np.sin(s/R) * np.cos(np.deg2rad(lat0)),
                np.cos(s/R) - np.sin(np.deg2rad(lat0)) * np.sin(np.deg2rad(lat))
            )
        )
        return lat, lon
    

    def generate_map_overlay(self, output_dims, include_map=True, include_lat_lon=0,
                         projection_center=None):
      """
      Generate a transparent overlay with map outlines and optional lat/lon lines.
      
      Returns a numpy array of shape (H, W, 3), dtype uint8.
      """
      import matplotlib
      matplotlib.use('Agg')  # headless
      import matplotlib.pyplot as plt
      import cartopy.crs as ccrs
      import cartopy.feature as cfeature
      import numpy as np

      out_w, out_h = output_dims
      fig = plt.figure(figsize=(out_w / 100, out_h / 100), dpi=100)
      
      # North polar stereographic
      central_lon = projection_center[1] if projection_center else 0.0
      ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=central_lon))
      
      # Ensure axes fill the figure
      fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
      ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
      ax.axis('off')
      ax.patch.set_alpha(0)  # transparent background

      # Add features
      if include_map:
          ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='white', linewidth=1)
          ax.add_feature(cfeature.OCEAN, facecolor='none', edgecolor='white', linewidth=1)

      # Lat/lon grid
      if include_lat_lon > 0:
          for lat in range(-90, 91, include_lat_lon):
              ax.plot(np.linspace(-180, 180, 360), [lat]*360, 'w:', transform=ccrs.PlateCarree())
          for lon in range(-180, 181, include_lat_lon):
              ax.plot([lon]*181, np.linspace(-90, 90, 181), 'w:', transform=ccrs.PlateCarree())

      # Render figure to RGBA buffer
      fig.canvas.draw()
      # Get RGBA buffer as NumPy array
      map_img = np.array(fig.canvas.renderer.buffer_rgba())
      map_img = map_img[...,:3]  # keep only RGB

      plt.close(fig)

      # Resize to desired output
      map_img = cv2.resize(map_img, (out_w, out_h))
      
      return map_img


    # -------------------------
    # Visualization function
    # -------------------------
    def project_image(self, input_image, output_dims=(1024,1024),
                      include_map=True, include_lat_lon=0,
                      overlay_azimuth_zenith=False,
                      projection_center=None,
                      interpolation='nearest',
                      input_radius_cutoff=None):
        """
        Project fisheye image onto polar stereographic map with see-through overlays.

        Returns RGB output image with all requested overlays.
        """

        import matplotlib
        matplotlib.use('Agg')  # headless, no Tkinter
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        out_w, out_h = output_dims

        # ----------------------
        # Backproject fisheye image
        # ----------------------
        y_idx, x_idx = np.indices((out_h, out_w))
        x_c = out_w / 2
        y_c = out_h / 2
        r_max_out = min(out_w, out_h) / 2

        dx = x_idx - x_c
        dy = y_c - y_idx  # y-axis inverted
        rho = np.sqrt(dx**2 + dy**2) / r_max_out
        az = np.rad2deg(np.arctan2(dx, dy)) % 360
        alt = np.polyval(self.alt_poly_coeffs, rho)

        r_fish = rho * self.r_max
        x_fish = self.xc + r_fish * np.sin(np.deg2rad(az))
        y_fish = self.yc - r_fish * np.cos(np.deg2rad(az))

        # Apply input radius cutoff
        if input_radius_cutoff is not None:
            r_from_center = np.sqrt((x_fish - self.xc)**2 + (y_fish - self.yc)**2)
            mask = r_from_center > input_radius_cutoff
            x_fish[mask] = 0
            y_fish[mask] = 0

        interp_flag = cv2.INTER_NEAREST if interpolation=='nearest' else cv2.INTER_LINEAR
        map_x = x_fish.astype(np.float32)
        map_y = y_fish.astype(np.float32)

        output_image = cv2.remap(input_image, map_x, map_y,
                                interpolation=interp_flag,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)  # black outside fisheye

        # ----------------------
        # Prepare overlay layer
        # ----------------------
        overlay = np.zeros_like(output_image, dtype=np.uint8)

        # ---- Map outlines and lat/lon lines using Cartopy ----
        if include_map or include_lat_lon:
            map_img = self.generate_map_overlay((out_w, out_h),
                                          include_map=include_map,
                                          include_lat_lon=include_lat_lon,
                                          projection_center=projection_center)
            overlay = cv2.add(overlay, map_img)

            # Resize to output dimensions
            map_img = cv2.resize(map_img, (out_w, out_h))
            overlay = cv2.add(overlay, map_img)

        # ---- Azimuth/zenith grid ----
        if overlay_azimuth_zenith:
            center_x = out_w // 2
            center_y = out_h // 2
            max_radius = min(out_w, out_h) // 2

            # Concentric zenith circles every 10°
            for alt_deg in range(10, 91, 10):
                r_px = int(max_radius * (1 - alt_deg/90.0))
                cv2.circle(overlay, (center_x, center_y), r_px, (255,255,255), 1)

            # Radial azimuth lines every 30°
            for az_deg in range(0, 360, 30):
                angle_rad = np.deg2rad(az_deg)
                x_end = int(center_x + max_radius * np.sin(angle_rad))
                y_end = int(center_y - max_radius * np.cos(angle_rad))
                cv2.line(overlay, (center_x, center_y), (x_end, y_end), (255,255,255), 1)

        # ----------------------
        # Alpha-blend overlay over backprojected image
        # ----------------------
        alpha = 1.0
        mask = np.any(overlay != 0, axis=2)  # only blend non-black pixels
        output_image[mask] = cv2.addWeighted(output_image[mask], 1.0, overlay[mask], alpha, 0)

        return output_image



    

if __name__ == "__main__":
    
    """
    python fisheye_to_map.py \
    --input_image_path sampleimages/00001273_geographic_north_EW_flipped.png \
    --feature_altitude 110000 \
    --observer_lat 50.3 \
    --observer_lon -96.5 \
    --output_image_path sampleimages/00001273_mapped.png \
    --overlay_map \
    --overlay_lat_lon 10 \
    --overlay_azimuth_zenith \
    --radius_cutoff_px 1900 \
    --output_size 1024 \
    --interpolation bilinear
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Project fisheye aurora image to polar stereographic map")
    parser.add_argument('--input_image_path', required=True, help='Path to input fisheye image (RGB)')
    parser.add_argument('--feature_altitude', type=float, default=110000, help='Altitude of aurora features in meters')
    parser.add_argument('--observer_lat', type=float, required=True, help='Observer latitude (decimal degrees)')
    parser.add_argument('--observer_lon', type=float, required=True, help='Observer longitude (decimal degrees)')
    parser.add_argument('--output_image_path', required=True, help='Path to save output projected map')
    parser.add_argument('--overlay_map', action='store_true', help='Include map outlines')
    parser.add_argument('--overlay_lat_lon', type=int, default=0, help='Draw lat/lon lines every N degrees (0 = none)')
    parser.add_argument('--radius_cutoff_px', type=float, default=None, help='Radius cutoff in pixels from fisheye center')
    parser.add_argument('--output_size', type=int, default=1024, help='Width and height of output image (square)')
    parser.add_argument('--interpolation', choices=['bilinear','nearest'], default='nearest', help='Resampling method')
    parser.add_argument('--overlay_azimuth_zenith', action='store_true', help='Overlay azimuth and zenith reference circles')
    
    args = parser.parse_args()

    # Load input image
    if not os.path.exists(args.input_image_path):
        raise FileNotFoundError(f"Input image not found: {args.input_image_path}")
    fisheye_img = cv2.imread(args.input_image_path, cv2.IMREAD_COLOR)
    if fisheye_img is None:
        raise ValueError(f"Failed to load image: {args.input_image_path}")
    fisheye_img = cv2.cvtColor(fisheye_img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Assume center is middle of image, max radius = min(H,W)/2
    H, W, _ = fisheye_img.shape
    xc, yc = W/2, H/2
    r_max = min(H, W)/2

    # Example polynomial coefficients (replace with your calibration)
    alt_poly_coeffs = [-49.50085744, 62.16763995, -91.67967310, 94.83971196]

    mapper = FisheyeAuroraMapper(
        xc=xc,
        yc=yc,
        r_max=r_max,
        alt_poly_coeffs=alt_poly_coeffs,
        observer_latlon=(args.observer_lat, args.observer_lon),
        feature_altitude=args.feature_altitude
    )

    output_img = mapper.project_image(
        fisheye_img,
        output_dims=(args.output_size, args.output_size),
        include_map=args.overlay_map,
        include_lat_lon=args.overlay_lat_lon,
        overlay_azimuth_zenith=args.overlay_azimuth_zenith,
        projection_center=None,
        interpolation=args.interpolation,
        input_radius_cutoff=args.radius_cutoff_px
    )

    # Save output image
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_image_path, output_bgr)
    print(f"Projected map saved to: {args.output_image_path}")

