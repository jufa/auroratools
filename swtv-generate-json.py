"""
Summary:
    This script extracts data from the metadata.csv files in the auroreye timelapse folders and generates a JSON file
    that can be used in the space weather timeline viewer at:

Input: 
    metatdata.csv files in the auroreye timelapse folders found directly under specified parent folder.
    1 image per row with the following schema (header  samplevalue)


    filename              00000000.jpg
    filepath             /Volumes/T7 Shield/AurorEye/seq_2026-04-18T07-14-35/00000000.jpg
    date                 2026-04-18
    time                 07:15:10
    orientation_degrees  
    make                 SONY
    model                ZV-E10
    orientation          1
    software             ZV-E10 v2.02
    exposure_time        8.0
    f_number             0.0
    photographic_sensitivity 3200
    exif_version         0232
    offset_time_original +00:00
    brightness_value     -14.18203125
    focal_length         0.0
    pixel_x_dimension    4000
    pixel_y_dimension    4000
    digital_zoom_ratio   1.0
    scene_capture_type   0
    contrast             0
    saturation           0
    sharpness            0
    lens_model           ----
    gps_altitude         213.0
    gps_altitude_ref     0
    latitude             44.973725
    latitude_ref         S
    longitude            169.241233
    longitude_ref        E

Transform:
  [
    {
      "id": "",                             # youtube video id, not present in metadata
      "source": "auroreye",                 # fixed
      "url": "https://auroreye.ca",         # fixed
      "location": "44.69, -63.13",          # latitude and longitude, with - for S and W
      "start_elapsed": 0,                   # constant
      "stop_elapsed": 63.0,                 # seconds time elapsed in youtube video playback head, calculated with 24fps
      "start_utc": "2023-04-24T03:48:40Z",  # from date and time data in metadata.csv, the first data row after the header
      "stop_utc": "2023-04-24T05:33:04Z"    # from date and time data in metadata.csv, the last data row 
    },
    ...
  ]

"""

import csv
import argparse
from pathlib import Path
from pprint import pprint
import json
import re
from googleapiclient.discovery import build

class MetadataProcessor:

  def __init__(self, folder:str=None, parent_folder:str=None) -> None:
    self.METADATA_FILENAME = "metadata.csv"
    self.parent_folder = parent_folder
    self.folder = folder
    self.output = []
    self.playlist_data = self.get_youtube_playlist_data()
    print(self.playlist_data)


  def get_youtube_playlist_data(self) -> dict:
    """
    [
      {
        'title': 'AurorEye UNIT 15 PLUMAS, MB 2026-01-21T02-29-22',
        'video_id': 'SMSlmCsCOnw'
      },
      ...
    ]
    """
    playlist_id = "PLXVlyzeh2wiG8LTGsbsH7KegGRlX-qFuf"
    api_key = ""

    with open(Path("secrets", "youtubeapiv3.txt")) as txt:
      api_key = txt.readline()
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    videos = []
    next_page_token = None

    print("retrieving video ids from youtube...", end="")
    while True:
      # Call the playlistItems.list method
      request = youtube.playlistItems().list(
        part='snippet,contentDetails',
        playlistId=playlist_id,
        maxResults=500,
        pageToken=next_page_token
      )
      response = request.execute()

      # Extract titles and video IDs
      for item in response['items']:
        title = item['snippet']['title']
        video_id = item['contentDetails']['videoId']
        videos.append({'title': title, 'video_id': video_id})

      # Check if there's a next page
      next_page_token = response.get('nextPageToken')
      if not next_page_token:
        break

      print("complete")

    return videos
  
  def find_video_id(self, pattern:str) -> str:
    """
    matching to string 2026-01-21T02-29-22
    """
    match = next((item for item in self.playlist_data if pattern in item.get('title')), None)
    return match["video_id"] if match else ""


  def process_folders(self, parent_folder:Path):
    success = 0
    total = 0
    for folder in parent_folder.iterdir():
      if folder.is_dir():
        pattern=r"seq_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}"
        re.match(pattern, folder.name)
        total += 1
        try:
          processed = self.parse_metadata_file_in_folder(folder)
          print (f"{folder} processed")
        except Exception as e:
          print(f"Could not process folder {folder}")
        if processed:
          success += 1
          self.output.append(processed)
    print(f"{success}/{total} folders successfully parsed")
    return self.output

  def parse_metadata_file_in_folder(self, folder:Path):
    csv_path = folder / self.METADATA_FILENAME
    folder_name = folder.name # seq_2029-01-31T13-13-59
    first:dict = None
    last:dict = None
    try:
      with open(str(csv_path), newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='"',)
        for i, row in enumerate(reader):
          data = row
          if i == 0:
            first = data
        last = data
        frames = i
        # print(f"frame count : {i}")
      
      start_datetime_utc = self.parse_datetime_utc(first)
      end_datetime_utc = self.parse_datetime_utc(last)
      location = self.parse_location(last)
      start_elapsed =  0.0
      stop_elapsed = round(frames / 24.0, 1)
      id = self.find_video_id(folder_name.replace("seq_",""))

      processed = {
        "id": id,
        "source": "auroreye",
        "url": "https://auroreye.ca",
        "start_datetime_utc": start_datetime_utc,
        "end_datetime_utc": end_datetime_utc,
        "location": location,
        "start_elapsed": start_elapsed,
        "stop_elapsed": stop_elapsed,
      }

      return processed
    
    except Exception as e:
      pass


  def parse_datetime_utc(self, row_data):
    # return ISO string, ie "2023-04-24T05:33:04Z"
    return (f"{row_data["date"]}T{row_data["time"]}Z")

  def parse_location(self, row_data):
    """
    builds lat and lon with negative for south and west, rounded to 2 decimals as float
    latitude             44.973725
    latitude_ref         S
    longitude            169.241233
    longitude_ref        E

    "44.69, -63.13"
    """

    lat = float(row_data["latitude"])
    lon = float(row_data["longitude"])
    lat = -lat if row_data["latitude_ref"].lower() == "S" else lat
    lon = -lon if row_data["longitude_ref"].lower() == "E" else lon

    return f"{lat:0.2f}, {lon:0.2f}"

  def process_metadata(self):
    return self.process_folders(Path(self.parent_folder))
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--parent_folder", help="parent folder of folders containing metadata.csv files")
  parser.add_argument("--folder", help="folder containing metadata.csv file")
  args = parser.parse_args()

  mp = MetadataProcessor(folder=args.folder, parent_folder=args.parent_folder)
  processed = mp.process_metadata()
  with open("SWTV.json", "w") as f:
    json.dump(processed, f, indent=4, sort_keys=False)
  


    


    






