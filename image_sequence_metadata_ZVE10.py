#!/bin/env python
import pathlib
from argparse import ArgumentParser
from exif import Image, Orientation
import csv
from pprint import pprint
import os
import re
import traceback
import sys

# Sony ZVE10 mk I data processing

# exif library users' page: https://exif.readthedocs.io/en/latest/usage.html
# constants and enums: https://gitlab.com/TNThieding/exif/-/blob/master/src/exif/_constants.py



remap_orientation = {
  Orientation.TOP_LEFT: 0, # normal
  Orientation.LEFT_BOTTOM: 90, # rotate 90 CW to match normal
  Orientation.RIGHT_TOP: 270, # rotate 270 CW to match normal
}


def get_exif_object(path):
  try:
    with open(path, 'rb') as img:
      exif_object = Image(img)
      return exif_object
  except:
    print(f"ERROR: could not access exif data for {path}")


def get_file_name(image):
  return pathlib.Path(image).name


def  build_metadata_object_for_file(exif_object, filepath):
    selected_tags = [
      "orientation_degrees",
      "make",
      "model",
      "orientation",
      "software",
      "exposure_time",
      "f_number",
      "photographic_sensitivity",
      "exif_version",
      "offset_time_original",
      "brightness_value",
      "focal_length",
      "pixel_x_dimension",
      "pixel_y_dimension",
      "digital_zoom_ratio",
      "scene_capture_type",
      "contrast",
      "saturation",
      "sharpness",
      "lens_model",
      "gps_altitude",
      "gps_altitude_ref"
    ]

    md = {}
    try:
      tags = exif_object.list_all()
      md["filename"] = get_file_name(filepath)
      md["filepath"] = filepath
      md["date"] = exif_object.get("datetime_original").split()[0].replace(":", "-")
      md["time"] = exif_object.get("datetime_original").split()[1]
      md["orientation_degrees"] = remap_orientation.get(exif_object.get("orientation"), 0)
      for tag in selected_tags:
        try:
          md[tag] = exif_object.get(tag)
        except:
          print(f"tag {tag} could not be retrieved")
      md['latitude'] = dms_to_decimal(exif_object.get("gps_latitude"))
      md['latitude_ref'] = exif_object.get("gps_latitude_ref")
      md['longitude'] = dms_to_decimal(exif_object.get("gps_longitude"))
      md['longitude_ref'] = exif_object.get("gps_longitude_ref")
      return md
    except Exception as e:
      print("ERROR building exif metadata dict\nDETAILS:\n")
      print(e)
      print(sys.exc_info()[0])
      print(sys.exc_info()[1])
      print(sys.exc_info()[2])
      print(traceback.format_exc())

"""
  GPS coordinates are stored in degrees, minutes, seconds format
  Convert to decimal format
  input : "(degrees, minutes, seconds)"
  output: degrees decimal, i.e. "64.45225"
"""
def dms_to_decimal(dms_str):
  # print(f"dms_to_decimal: {str(dms_str)}")
  dms_str = str(dms_str).strip()
  match = re.match(r"\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)", dms_str)
  if not match:
    raise ValueError("Input string does not match the expected format (d, m, s)")
  
  degrees, minutes, seconds = map(float, match.groups())
  decimal = degrees + minutes / 60 + seconds / 3600
  return round(decimal, 6)  # Rounded to 6 decimal places for precision


def iterate_through_image_directory(path, forcelatlon=False):
  image_files = []
  metadata = []
  file_name_pattern = r".*.JPG"
  folder = pathlib.Path(path)
  for item in folder.iterdir():
    if item.is_file() and re.search(file_name_pattern, item.name.upper()):
      image_files.append(str(item))
    else:
      print(f"file {item} not added to image list")
  image_files.sort()
  files_to_scan_count = len(image_files)
  print(f"files_to_scan: {files_to_scan_count} starting at {image_files[0]}, ending at {image_files[-1]}")
  for i, image in enumerate(image_files):
    try:
      print(f"extracting metadata for: [{i}/{files_to_scan_count}]\r", end="")
      metadata.append( build_metadata_object_for_file( get_exif_object(image), image ) )
      if forcelatlon:
        print("forcing lat lon:")
        print(f"old/new LAT: {metadata[i]['LAT']} -> {forcelatlon[0]}")
        print(f"old/new LON: {metadata[i]['LON']} -> {forcelatlon[1]}")
        if metadata[i].get("LAT", None):
          metadata[i]["LAT"] = forcelatlon[0]
        if metadata[i].get("LON", None):
          metadata[i]["LON"] = forcelatlon[1]
    except:
      print(f"{image} error in retrieving metadata...continuing")
      break
  save_metadata_as_csv(metadata, os.path.join(path, "metadata.csv") )
  return True


def save_metadata_as_csv(metadata, path):
  # remove trailing empty rows
  while metadata and not metadata[-1]:
    metadata.pop()
  
  keys = metadata[0].keys()

  with open(path, 'w', newline='') as output_file:
      # reference: https://docs.python.org/3/library/csv.html#dialects-and-formatting-par--forcelatlon ameters
      dict_writer = csv.DictWriter(output_file, keys)
      dict_writer.writeheader()
      dict_writer.writerows(metadata)

  print("==========\ncsv file written:")
  print(path)
  print(f"rows (images): {len(metadata)}")
  print(f"cols (tags):   {len(keys)}")
        


def process_cli():
  parser = ArgumentParser()
  parser.add_argument('--path', help='path to folder of source images')
  # 62.552213, -114.026510
  parser.add_argument('--forcelatlon', help='forced value of latitude and longitude in the format 123.346,-123.456')
  args = vars(parser.parse_args())
  path = args["path"]
  forcelatlon = args.get("forcelatlon", None)
  forcelatlon_parsed = None
  if forcelatlon:
    forcelatlon_parsed = [float(v) for v in args["forcelatlon"].split(",")]
  iterate_through_image_directory(path, forcelatlon_parsed)


if __name__ == '__main__':
  process_cli()


"""
['image_description',
'make',
'model',
'orientation',
'x_resolution',
'y_resolution',
'resolution_unit',
'datetime',
'y_and_c_positioning',
'copyright',
'_exif_ifd_pointer',
'_gps_ifd_pointer',
'artist',
'compression',
'jpeg_interchange_format',
'jpeg_interchange_format_length',
'exposure_time',
'f_number',
'exposure_program',
'photographic_sensitivity',
'sensitivity_type',
'recommended_exposure_index',
'exif_version',
'datetime_original',
'datetime_digitized',
'components_configuration',
'compressed_bits_per_pixel',
'shutter_speed_value',
'aperture_value',
'exposure_bias_value',
'metering_mode',
'flash',
'focal_length',
'maker_note',
'user_comment',
'subsec_time',
'subsec_time_original',
'subsec_time_digitized',
'flashpix_version',
'color_space',
'pixel_x_dimension',
'pixel_y_dimension',
'_interoperability_ifd_Pointer',
'focal_plane_x_resolution',
'focal_plane_y_resolution',
'focal_plane_resolution_unit',
'sensing_method',
'file_source',
'custom_rendered',
'exposure_mode',
'white_balance',
'digital_zoom_ratio',
'scene_capture_type',
'camera_owner_name',
'body_serial_number',
'lens_specification',
'lens_model',
'lens_serial_number',
'gps_version_id',
'gps_latitude',
'gps_latitude_ref',
'gps_longitude',
'gps_longitude_ref',
'gps_altitude',
'gps_altitude_ref']
"""
