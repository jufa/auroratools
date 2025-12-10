#!/bin/env python
import pathlib
from argparse import ArgumentParser
from exif import Image
import csv
from pprint import pprint
import os
import re

"""
path: path to a folder containing an image sequence
"""

# get list of files in path
# for each file: extract exif data as an array of dict
# save aas csv with header column being the keys of the dict


# reference

# to_csv = [
#     {'name': 'bob', 'age': 25, 'weight': 200},
#     {'name': 'jim', 'age': 31, 'weight': 180},
# ]

# keys = to_csv[0].keys()

# with open('people.csv', 'w', newline='') as output_file:
#     dict_writer = csv.DictWriter(output_file, keys)
#     dict_writer.writeheader()
#     dict_writer.writerows(to_csv)


def get_exif_object(path):
  try:
    with open(path, 'rb') as img:
      exif_object = Image(img)
      # print(exif_object.list_all())
      return exif_object
  except:
    print(f"ERROR: could not access exif data for {path}")


# def read_exif(self, exif_object=None, tag=None, data=None, pad=True):
   
#     try:
#       if pad and isinstance(data, str):
        
#         try:
#           length_target = len(exif_object.get(tag)) # owner_name can be written then read, but not read first, maybe this is an EXIF lib bug
#         except:
#           length_target = len(data)
#         length_data = len(data)
#         pad_length = length_target - length_data - 1
#         data_padded = data.ljust(pad_length, " ") # we pad because exif seems to have fixed length fields and once they are shortened, they can't be lengthened, at least by this library
#         exif_object.set(tag, data_padded)
#       else:
#         exif_object.set(tag, data)
#     except:
#       self.log(f"ERROR: Cannot set EXIF tag {tag} to {data} on exif object {exif_object}")
#       exc_type, exc_obj, exc_tb = sys.exc_info()
#       fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#       self.log(f'exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}')
#       self.log(f'{exc_type} {fname} {exc_tb.tb_lineno}')
#       return -1
#     return 1


def  build_metadata_object_for_file(exif_object):
    md = {}
    try:
      # md["datetime"] = exif_object.get("datetime")
      md["date"] = exif_object.get("datetime").split()[0].replace(":", "-")
      md["time"] = exif_object.get("datetime").split()[1]
      uc = exif_object.get("user_comment")
      ori = exif_object.get("orientation")
      print(f"\norientation: {ori}\n")
      uc_params = uc.split("|")
      for param in uc_params:
        key, val = param.split()
        md[key] = val
      # md['latitude'] = f' \
      #   { exif_object.get("gps_latitude") } \
      #   { exif_object.get("gps_latitude_ref").strip() }'

# 'gps_latitude_ref',
# 'gps_longitude',
# 'gps_longitude_ref',

      return md
    except:
      print("ERROR building exif metadata dict")


def iterate_through_image_directory(path, forcelatlon):
  image_files = []
  metadata = []
  file_name_pattern = r"\d{8}\.jpg"
  folder = pathlib.Path(path)
  for item in folder.iterdir():
    if item.is_file() and re.search(file_name_pattern, item.name):
      image_files.append(str(item))
    else:
      print(f"file {item} not added to image list")
  image_files.sort()
  files_to_scan_count = len(image_files)
  for i, image in enumerate(image_files):
    print(f"extracting metadata for: [{i}/{files_to_scan_count}]\r", end="")
    metadata.append( build_metadata_object_for_file( get_exif_object(image) ) )
    if forcelatlon:
      print("forcing lat lon:")
      print(f"old/new LAT: {metadata[i]['LAT']} -> {forcelatlon[0]}")
      print(f"old/new LON: {metadata[i]['LON']} -> {forcelatlon[1]}")
      if metadata[i].get("LAT", None):
        metadata[i]["LAT"] = forcelatlon[0]
      if metadata[i].get("LON", None):
        metadata[i]["LON"] = forcelatlon[1]
  print("\nSample metadata:")
  pprint(metadata[0])
  save_metadata_as_csv(metadata, os.path.join(path, "metadata.csv") )
  return True


def save_metadata_as_csv(metadata, path):
  keys = metadata[0].keys()

  with open(path, 'w', newline='') as output_file:
      dict_writer = csv.DictWriter(output_file, keys)
      dict_writer.writeheader()
      dict_writer.writerows(metadata)



    # stamp.append(f'VER 001')
    # if config:
    #   stamp.append(f'UNT {config["unit"]}')
    # else:
    #   stamp.append(f'UNT {UNKNOWN_STRING}')
    # stamp.append(f'DAT {gps_data["date"]}T{gps_data["time"]}Z')
    # stamp.append(f"EXP {self.status['exposure']}")
    # stamp.append(f"ISO {self.status['iso']}")

    # if gps_data and ( type(gps_data['lat']) is not str ):
    #   lat = gps_data['lat']
    #   lon = gps_data['lon']
    #   alt = gps_data['alt']
    #   stamp.append(f'LAT {lat:0.6f}')
    #   stamp.append(f'LON {lon:0.6f}')
    #   stamp.append(f'ALT {alt:0.1f}')
    # else:
    #   stamp.append(f'LAT {UNKNOWN_STRING}')
    #   stamp.append(f'LON {UNKNOWN_STRING}')
    #   stamp.append(f'ALT {UNKNOWN_STRING}')
    
    # if accel_data:
    #   degX = accel_data['degX']
    #   degY = accel_data['degY']
    #   stamp.append(f'DGX {degX:0.2f}')
    #   stamp.append(f'DGY {degY:0.2f}')
    # else:
    #   stamp.append(f'DGX {UNKNOWN_STRING}')
    #   stamp.append(f'DGY {UNKNOWN_STRING}')

    # if th_data:
    #   temperature = th_data['temperature']
    #   humidity = th_data['humidity']
    #   stamp.append(f'TMP {temperature:0.1f}')
    #   stamp.append(f'HUM {humidity:0.1f}')
    # else:
    #   stamp.append(f'TMP {UNKNOWN_STRING}')
    #   stamp.append(f'HUM {UNKNOWN_STRING}')

    # stamp.append(f'CAM {config["camera"].replace(" ", "").upper()}')
    # stamp.append(f'LNS {config["lens"].replace(" ", "").upper()}')

    # metadata_string = "|".join(stamp)
    # return metadata_string

def process_cli():
  parser = ArgumentParser()
  parser.add_argument('--path', help='path to folder to create keogram')
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
