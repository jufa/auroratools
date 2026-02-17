import aacgmv2
from astropy.time import Time
import astropy.units as u
import numpy as np

"""
Approximation of MLM That this module is more accurately modelling:

Example:
Yellowknife on 2026-02-17
declination is 16 deg east (+16deg)
16/360 = 0.0444 of a day
0.0444 * 24 = 1.07 hours until MLM after local midnight (00:00)
so local midnight + 1.07 hours = 01:04  is the approximate local MLM time
now add UTC offset (7 hours East of 00:00UTC) to get UTC time of MLM: 01:04 + 7 hours = 08:04 UTC
Module says 08:37 UTC using a more accurate library

tromso on 2026-02-17
declination is 15 deg east (+15deg)
0.0444 * 24 = 1.07 hours until MLM at local geographic midnight
now add UTC offset (+1 hours, or 1 hour west of UTC so opposite sign from declination)
and they should approximately cancel out, giving an MLM time close to 00:00 UTC
MLM:            2026-02-17 22:35:00.000 UTC
"""

# sample sites
sites = {
    "plumas": (50.3, -99.1, 0.281),  # (lat, lon, alt_km)
    "fairbanks": (64.8, -147.7, 0.133),
    "calgary": (51.0, -114.1, 0.000),
    "tromso": (69.7, 18.9, 0.000),
    "sodankyla": (67.4, 26.6, 0.000),
    "kiruna": (67.8, 20.4, 0.000),
    "yellowknife": (62.5, -114.4, 0.000), # typ 08:37 UTC
    "reykjavik": (64.1, -21.9, 0.000),
    "longyearbyen": (78.2, 15.6, 0.000)
}

site = "plumas"
date_str = "2026-02-17"
geo_lat, geo_lon, alt_km = sites[site.lower()]

# Use a fine 1-minute time grid
t0 = Time(date_str + "T00:00:00", scale="utc")
times = t0 + np.arange(0, 1440) * (1/60) * u.hour  # 1-min steps

# First convert geographic to magnetic coordinates to get magnetic longitude
mag_coords = aacgmv2.get_aacgm_coord(geo_lat, geo_lon, alt_km, t0.to_datetime())
mag_lon = mag_coords[1]  # Magnetic longitude

# Compute magnetic local time directly using magnetic longitude
# convert_mlt takes (magnetic longitude, datetime) and returns MLT in hours
mlt_list = [aacgmv2.convert_mlt(mag_lon, Time(t).to_datetime()) for t in times.iso]

# Find time when MLT ≈ 0 → Magnetic Local Midnight
mlt_array = np.array(mlt_list)
idx = np.argmin(np.abs(mlt_array))
mlm_time = times[idx]

# print("Magnetic Local Midnight (UTC):", mlm_time.utc.iso)
idx = np.argmin(np.abs(mlt_array))
mlm_time = times[idx]

print(f"-------------------------------------------\nApproximate Magnetic Local Midnight\n-------------------------------------------\nLocation: \t{site}\nDate:\t\t{date_str}\nMLM:\t\t{mlm_time.utc.iso} UTC\n-------------------------------------------")
