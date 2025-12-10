# import os
# import sys
# import json
# from datetime import datetime
# from re import X
# import subprocess
# from argparse import ArgumentParser, ArgumentError
# import fnmatch
# from jpegtran import JPEGImage
# from sensor import Sensor
# from pathlib import Path
# from PIL import Image
# import numpy as np
# import cv2
# from utils import exception_details
# from google.cloud import storage


# OK so we need to loook throught the root folder and check for any folders that do not have a timelapse
# we basiaclly want a button called create and upload timelapses
# this means we need an ftp or http target to upload to wityh some kind of authentication
# which only accepts mp4, jpg, raw
# auroreye needs and option to crerate and upload the mjpeg or raw? yeah.
# also need to clear out the memory space
# also need software upgrade
# maybe some kind of ssh over internet?

RESOLUTION_2K = "2k"
RESOLUTION_4K = "4k"
RESIZE_DIMENSION = 1080
SCALED_FOLDER_NAME= "scaled2k"
ROOT_PATH = "/mnt/extstore/"
GCP_STORAGE_BUCKET = "auroreye-storage-558"
TIMELAPSE_FILE_NAME = "timelapse2k.mp4"
TIMELAPSE_4K_FILE_NAME = "timelapse4k.mkv"
ARCHIVE_FILE_NAME = "archive.tar"
SEQUENCE_LOG_FILE_NAME = "sequence_log.txt"
AUROREYE_LOG_FILE_NAME = "auroreye_log.txt"
SUCCESS = True
ERROR = False

class PostProcessor(Sensor):

  def __init__(self, logger=None):
    super().__init__(logger=logger, module_name='PostProc')

  # def log(self, msg):
  #   print(msg)


  def count_images(self, path):
    try:
      files = fnmatch.filter(os.listdir(path), '*.jpg')
      return len(files)
    except:
      return None
    # nl = "\n"
    # tab = "\t"
    # print(f"JPG images:{nl}{tab.join(files)}")
    # files.sort()
    # for f in files:
      # file_path = os.path.join(path, f)
      # modified_ts = os.path.getmtime(file_path)
      # print(f"{f} {datetime.fromtimestamp(modified_ts)}")
    # self.log('-' * 20)
    # self.log(f"jpg images:{len(files)}")
    # self.log('-' * 20)

  def make_keogram(self, path):
    input_path = path
    try:
      files = fnmatch.filter(os.listdir(input_path), '*.jpg')
    except:
      self.log(f"ERROR: could not open folder for resizing operation: {input_path}")
      self.log(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
      return
    # output_path = os.path.join(input_path,self.scaled_folder_name)

    files.sort()
    files_to_scan_count = len(files)

    on_first_image = True
    keogram = None
    curr_column = 0

    try:
      for f in files:
        file_path = os.path.join(input_path, f)
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        sample_width = 4
        height = img.shape[0]
        width = img.shape[1]
        keo_width = files_to_scan_count
        col_start = int(width / 2) - 1
        col_end = int(width / 2)
        slice = img[0:height, col_start:col_end] # r, c
        if on_first_image:
          on_first_image = False
          keogram = np.zeros((height, keo_width, 3), np.uint8)
        keogram[0:height, curr_column:curr_column + 1] = slice
        # keogram = np.append(keogram, slice, axis=1) # slow
        curr_column += 1
      
      output_path = os.path.join(path, "keogram.png")
      cv2.imwrite(output_path, keogram)

      self.log(f"keogram complete without error: dimensions:{keogram.shape}")
      with open(os.path.join(path, 'sequence_log.txt'), 'a') as f:
        f.write("# KEOGRAM GENERATED\n")
    except:
      self.log(f"ERROR: could not create keogram: {input_path}")
      self.log(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
      return


  def make_timelapse(self, path, resolution=RESOLUTION_2K):
    if resolution == RESOLUTION_2K:
      image_folder_path = os.path.join(path, SCALED_FOLDER_NAME)
    elif resolution == RESOLUTION_4K:
      image_folder_path = os.path.join(path, "")

    output_file_path = path
    if os.path.exists(image_folder_path):
      success = self.run_ffmpeg(image_folder_path, output_file_path, resolution)
      if success:
        with open(os.path.join(path, 'sequence_log.txt'), 'a') as f:
          if resolution == RESOLUTION_2K:
            f.write("# TIMELAPSE 2K GENERATED\n")
          if resolution == RESOLUTION_4K:
            f.write("# TIMELAPSE 4K GENERATED\n")
    else:
      self.log("ERROR: make_timelapse: {SCALED_FOLDER_NAME} folder not found at {path}")
      return False


  def make_archive(self, folder_name):
    self.log(f"make_archive: {folder_name} to {folder_name}/archive.tar")
    # command = f"tar -C {folder_name} -cvf archive.tar *.jpg"
    
    command = f"cd {folder_name};tar -cvf archive.tar *.jpg"

    so = open(f"{folder_name}/archive_output.txt", "w+")
    se = open(f"{folder_name}/archive_err.txt", "w+")
    if subprocess.run(command, shell=True, stdout=so, stderr=se).returncode == 0:
      so.close()
      se.close()
      self.log("Success: archive created")
      with open(os.path.join(folder_name, 'sequence_log.txt'), 'a') as f:
        f.write("# ARCHIVE GENERATED\n")
      return SUCCESS
    else:
      so.close()
      se.close()
      self.log("ERROR: archive creation")
      return False


  def upload_archive(self, folder_path):
    folder_name = str( Path(*folder_path.parts[3:]) ) # strip /mnt/exstore parent portions of path to get a useful path to use in the bucket
    files_to_upload = [
      {
        "src":os.path.join(folder_path, SEQUENCE_LOG_FILE_NAME),
        "dst":os.path.join(folder_name, SEQUENCE_LOG_FILE_NAME)
      },
      {
        "src":os.path.join(folder_path, "../", AUROREYE_LOG_FILE_NAME),
        "dst":os.path.join(folder_name, AUROREYE_LOG_FILE_NAME)
      },
      {
        "src":os.path.join(folder_path, ARCHIVE_FILE_NAME),
        "dst":os.path.join(folder_name, ARCHIVE_FILE_NAME)
      },
    ]
    
    for file_paths in files_to_upload:
      success_status = True
      src = file_paths["src"]
      dst = file_paths["dst"]
      self.log(f"attempting to upload file: {src} to {dst}")

      if Path(src).is_file():
        result = self.upload_gcp(source_path=src, dest_path=dst)
        if result is SUCCESS:
          self.log(f"file uploaded to remote at {dst}")
        else:
          self.log(f"ERROR: GCP Upload issue")
          success_status = False
      else:
        self.log(f"ERROR: file not found: {src}")
        success_status = False

    if success_status == True:
      self.log("SUCCESS: all archive files uploaded")
      with open(os.path.join(folder_path, 'sequence_log.txt'), 'a') as f:
        f.write("# ARCHIVE UPLOADED\n")
        return SUCCESS
    else:
      self.log("ERROR: some or all archive files not uploaded")
      return ERROR


  def run_ffmpeg(self, image_folder_path, output_path, resolution=RESOLUTION_2K):
    # ffmpeg = "/usr/bin/ffmpeg"
    self.log(f"run_ffmpeg: {image_folder_path} to {output_path} at resolution {resolution}")
    command = ""

    # optional 2 pass encoding with omx: http://raspberrypithoughts.blogspot.com/2019/03/raspberry-pi-3b-video-encoding-with.html
    # h264_v4l2m2m
    command_2k = f"ffmpeg -y -i {image_folder_path}/%08d.jpg -b:v 12M -crf 18 -codec:v h264_v4l2m2m  {output_path}/{TIMELAPSE_FILE_NAME}"
    command_4k = f"ffmpeg -y -i {image_folder_path}/%08d.jpg -codec:v copy {output_path}/{TIMELAPSE_4K_FILE_NAME}"

    if resolution == RESOLUTION_2K:
      command = command_2k
    elif resolution == RESOLUTION_4K:
      command = command_4k

    so = open(f"{output_path}/ffmpeg_output.txt", "w+")
    se = open(f"{output_path}/ffmpeg_err.txt", "w+")
    if subprocess.run(command, shell=True, stdout=so, stderr=se).returncode == 0:
      so.close()
      se.close()
      self.log("Success: timelapse created")
      return SUCCESS
    else:
      so.close()
      se.close()
      self.log("ERROR: timelapse creation")
      return False
    
  def upload_gcp(self, source_path=None, dest_path=None):
    try:
      # source_path = Path(source_path)
      dest_path = str(dest_path) # str( Path(*source_path.parts[3:]) ) # strip /mnt/exstore parent portions of path to get a useful path to use in the bucket
      source_path = str(source_path)
      # Setting credentials using the downloaded JSON file
      client = storage.Client.from_service_account_json(json_credentials_path="auroreye-dfa929a4d933.json")
      # Creating bucket object
      bucket = client.get_bucket(GCP_STORAGE_BUCKET)
      # Name of the object to be stored in the bucket
      object_name_in_gcs_bucket = bucket.blob( dest_path )
      # Name of the object in local file system
      result = object_name_in_gcs_bucket.upload_from_filename(source_path)
      self.log(f"Upload complete to {dest_path}. Result status: {result}")
      return SUCCESS if result is None else ERROR
    except:
      self.log(f"Upload GCP exception: {exception_details()}")
      return ERROR


  def upload_timelapse(self, path, resolution=RESOLUTION_2K):
    # check if timelpase exists at path, if so, upload gcp the expected file and path:
    if resolution == RESOLUTION_4K:
      timelapse_file_path = os.path.join(path, TIMELAPSE_4K_FILE_NAME)
    else:
      timelapse_file_path = os.path.join(path, TIMELAPSE_FILE_NAME)

    self.log(f"attempting to upload timelapse file: {timelapse_file_path}")

    if Path(timelapse_file_path).is_file():
      result = self.upload_gcp(timelapse_file_path)
      if result == SUCCESS:
        with open(os.path.join(path, 'sequence_log.txt'), 'a') as f:
          if resolution == RESOLUTION_4K:
            f.write("# TIMELAPSE 4K UPLOADED\n")
          else:
            f.write("# TIMELAPSE 2K UPLOADED\n")
        self.log(f"timelapse uploaded to remote at {timelapse_file_path}")
      else:
        self.log(f"ERROR: GCP Upload issue for: {timelapse_file_path}")
    else:
      self.log(f"ERROR: timelapse file not found: {timelapse_file_path}")


  def offload(self):
    """
    assesses audits folders and uploads completed timelapses and keograms
    """
    pass


  def process_cli(self):
    parser = ArgumentParser()
    parser.add_argument('--path', help='path to folder to create keogram')
    parser.add_argument('--scan-subfolders', action='store_true', help='scan all subfolders and report processing status')
    parser.add_argument('--process', action='store_true', help='process images specified by --path')
    parser.add_argument('--process-all', action='store_true', help='scan all subfolders under --path and process all images found')
    parser.add_argument('--resize', action='store_true', help='resize images in folder --path and store under /scaled subfolder')
    parser.add_argument('--make-timelapse', action='store_true', help='make timelapse folder --path, 2k (1080p) or 4k (2500p) timelapse resolution')
    parser.add_argument('--make-keogram', action='store_true', help='make keogram from images in folder --path/scaled')
    parser.add_argument('--upload', action='store_true', help='upload file in --path to cloud storage')
    parser.add_argument('--offload', action='store_true', help='fully audit, process and upload timelapses to cloud storage')
    parser.add_argument('--resolution', choices=[RESOLUTION_2K, RESOLUTION_4K], default=RESOLUTION_2K, help='2k (1080p) or 4k (2560p) timelapse resolution')
    self.args = vars(parser.parse_args())
    for arg, value in self.args.items():
      print(arg, value)
    path = self.args["path"]
    if self.args["scan_subfolders"]:
      self.audit_folders(path)
    if self.args["resize"]:
      self.resize_images(path)
    if self.args["make_timelapse"]:
      self.make_timelapse(path, self.args["resolution"])
    if self.args["process"]:
      self.process_folder(path, self.args["resolution"])
    if self.args["process_all"]:
      self.process_all(root_path=path)
    if self.args["make_keogram"]:
      self.make_keogram(path)
    if self.args["upload"]:
      self.upload_gcp(path)


  def get_image_size(self, image_path):
    dim = None
    try:
      im = Image.open(image_path)
      width, height = im.size
      dim = { "width": width, "height": height }
    except:
      pass
    return dim


  def resize_images(self, input_path):
    try:
      files = fnmatch.filter(os.listdir(input_path), '*.jpg')
    except:
      self.log(f"ERROR: could not open folder for resizing operation: {input_path}")
      self.log(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
      return
    output_path = os.path.join(input_path,SCALED_FOLDER_NAME)

    try:
      os.mkdir(output_path)
    except OSError as error:
      self.log(error)

    files.sort()
    files_to_scan_count = len(files)
    for f in files:
      file_path = os.path.join(input_path, f)
      im = JPEGImage(file_path)
      image_dims = self.get_image_size(file_path)
      operations = {"crop": False, "resize": False}

      if image_dims:
        if image_dims["height"] > RESIZE_DIMENSION:
          operations["resize"] = True
        if image_dims["height"] < image_dims["width"]:
          operations["crop"] = True
      else:
        # could not determine image dimesnsions: skip:
         self.log(f"ERROR: {file_path}: image size could not be determined")

      if operations["crop"]:
        x = image_dims["width"] - image_dims["height"]
        x = int(x * 0.5 / 16) * 16
        y = 0
        height = image_dims["height"]
        width = height
        im = im.crop(x, y, width, height)
      if operations["resize"]:
        im = im.downscale(RESIZE_DIMENSION, RESIZE_DIMENSION)
     
      self.log(f"{file_path}: {image_dims['width']}x{image_dims['height']}\tcrop: {operations['crop']}\tresize:{operations['resize']}")

      if any(value == True for value in operations.values()):
        try:
          save_path = os.path.join(output_path, f)
          im.save(save_path)
          files_to_scan_count -= 1
        except:
          self.log(f"ERROR: could not save {save_path}")
          self.log(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
      else:
        files_to_scan_count -= 1
    if files_to_scan_count == 0:
      self.log("resize complete without error")
      with open(os.path.join(input_path, 'sequence_log.txt'), 'a') as f:
        f.write("# RESIZE COMPLETE\n")


    else:
      self.log(f"ERROR: unresized image count: {files_to_scan_count}") 
    return files_to_scan_count == 0 # if 0 then successfully handled all files



  def parse_sequence_log(self, path):
    status = {
      "sequence_log_found": False,
      "timelapse_2k_generated": False,
      "timelapse_4k_generated": False,
      "sequence_ended": False,
      "resize_complete": False,
      "keogram_generated": False,
      "timelapse_2k_uploaded": False,
      "timelapse_4k_uploaded": False,
      "archive_generated": False,
      "archive_uploaded": False,
      "skip": False,
    }
    try:
      fp = open(os.path.join(path, 'sequence_log.txt'), 'r')
      status["sequence_log_found"] = True
      lines = fp.readlines()
      for line in lines:
        if line.find("# SKIP") != -1:
          status['skip'] = True
        if line.find("# TIMELAPSE 2K GENERATED") != -1:
          status["timelapse_2k_generated"] = True
        if line.find("# TIMELAPSE 4K GENERATED") != -1:
          status["timelapse_4k_generated"] = True
        if line.find("# SEQUENCE ENDED") != -1:
          status["sequence_ended"] = True
        if line.find("# RESIZE COMPLETE") != -1:
          status["resize_complete"] = True
        if line.find("# KEOGRAM GENERATED") != -1:
          status["keogram_generated"] = True
        if line.find("# TIMELAPSE 2K UPLOADED") != -1:
          status["timelapse_2k_uploaded"] = True
        if line.find("# TIMELAPSE 4K UPLOADED") != -1:
          status["timelapse_4k_uploaded"] = True
        if line.find("# ARCHIVE GENERATED") != -1:
          status["archive_generated"] = True
        if line.find("# ARCHIVE UPLOADED") != -1:
          status["archive_uploaded"] = True
    except:
      pass
    return status


  def folder_status(self, path):
    status = self.parse_sequence_log(path)
    status["file_count"] = self.count_images(path)

    return status


  def audit_folders(self, root_path):
    results = self.scan_folders(root_path)
    for folder, status in results.items():
      print(str(folder))
      print(json.dumps(status, sort_keys=True, indent=4))
      # for folder, status in results.items():
      #   print(str(folder))
      # print(r)


  def scan_folders(self, root_path):
    p = Path(root_path)
    subfolders = [x for x in p.iterdir() if (x.is_dir() and os.path.basename(x)[0:3] == "seq" ) ]
    folder_status_summary = {}
    for folder in subfolders:
      status = self.folder_status(folder)
      folder_status_summary[folder] = status
    return folder_status_summary


  def process_all(self, complete_handler=None, root_path=ROOT_PATH, resolution=None, archive=True, keogram=False):
    self.log("-" * 40)
    self.log("Processing and Uploading all new image sequences...")
    self.log("-" * 40)
    folder_status_summary = self.scan_folders(root_path)
    for folder, status in folder_status_summary.items():
      self.log(f"processing folder {folder}...{status}")
      self.process_folder(path=folder, resolution=resolution, archive=archive, keogram=keogram)
    self.log(f"processing complete for all folders")
    self.log(f"complete_handler: {complete_handler}")
    if complete_handler:
      try:
        complete_handler()
      except:
        self.log("process_all: complete_handler callback")
    return


  def scan_single_folder(path):
    pass


  def process_folder(self, path, resolution=None, archive=True, keogram=False):
    status = self.folder_status(path)
    if not status["sequence_log_found"]:
      self.log(f"SKIPPING: sequence log not found in {path}")
      return ERROR
    if status["skip"]:
      self.log(f"SKIPPING: found # SKIP directive at {path}")
      return ERROR
    if not status["sequence_ended"]:
      self.log(f"WARNING: sequence log does not indicate sequence was ended gracefully in {path}. Possible battery exhaustion during shooting. Continuing anyways")
    if not status["resize_complete"] and resolution == RESOLUTION_2K:
      self.log(f"Sequence log does not show resizing was completed. Resizing...")
      self.resize_images(path)
    if not status["keogram_generated"] and keogram == True:
      self.log(f"Sequence log does not show keogram was completed. Generating...")
      self.make_keogram(path)
    if not status["timelapse_2k_generated"] and resolution == RESOLUTION_2K:
      self.log(f"Sequence log does not show 2K timelapse was created. Generating...")
      self.make_timelapse(path, resolution)
    if not status["timelapse_4k_generated"] and resolution == RESOLUTION_4K:
      self.log(f"Sequence log does not show 4K timelapse was created. Generating...")
      self.make_timelapse(path, resolution)
    if not status["timelapse_2k_uploaded"] and resolution == RESOLUTION_2K:
      self.log(f"Sequence log does not show 2K timelapse was uploaded. Uploading...")
      self.upload_timelapse(path, resolution)
    if not status["timelapse_4k_uploaded"] and resolution == RESOLUTION_4K:
      self.log(f"Sequence log does not show 4K timelapse was uploaded. Uploading...")
      self.upload_timelapse(path, resolution)
    if not status["archive_generated"] and archive == True:
      self.log(f"Sequence log does not show archive was created. Generating...")
      self.make_archive(path)
    if not status["archive_uploaded"] and archive == True:
      self.log(f"Sequence log does not show archive was uploaded. Uploading...")
      self.upload_archive(path)
    self.log(f"processing complete for {path}")



if __name__ == '__main__':
  kg = PostProcessor()
  kg.process_cli()

# go to the folder
# count the images
# start ffmpeg at 00000000.jpg
# use the ffmpeg callback to generate a new background file for each ffmpeg as per http://underpop.online.fr/f/ffmpeg/help/drawtext.htm.gz

