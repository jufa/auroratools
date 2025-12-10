import os
import sys
from re import X
from argparse import ArgumentParser, ArgumentError
import fnmatch
from pathlib import Path
import numpy as np
import json
import cv2
from exif import Image, Orientation

remap_orientation = {
  Orientation.TOP_LEFT: 0, # normal
  Orientation.LEFT_BOTTOM: 90, # rotate 90 CW to match normal
  Orientation.RIGHT_TOP: 270, # rotate 270 CW to match normal
}

class SequenceGenerator():

  def __init__(self):
    self.image_file_list = []
    pass

  def get_exif_object(self, path):
    try:
      with open(path, 'rb') as img:
        exif_object = Image(img)
        return exif_object
    except:
      print(f"ERROR: could not access exif data for {path}")

  def get_image_file_list(self, path):
    if self.image_file_list:
      return self.image_file_list
    input_path = path
    try:
      files = fnmatch.filter(os.listdir(input_path), '*.jpg')
    except:
      print(f"ERROR: could not open folder for resizing operation: {input_path}")
      print(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
      return
    # output_path = os.path.join(input_path,self.scaled_folder_name)
    files.sort()
    return files

  def load_image(self):
    img = cv2.imread(os.path.join(self.path, self.image_file_list[self.index]), cv2.IMREAD_COLOR)
    return img

  def rotate_image_and_square_canvas(self, image, angle):
    height, width = image.shape[:2]
    max_dimension = max(height, width)
    centerX, centerY = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D((centerX, centerY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (max_dimension, max_dimension))
    return rotated
  

  def get_rotation_angle(self, path):
    files = self.get_image_file_list(path)
    self.index = 0
    file_count = len(self.image_file_list)
    windowName = 'main'
    self.window = windowName
    cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(self.window, 800, 800)
    cv2.createTrackbar('Image', self.window, 0, file_count, self.on_change_image)
    cv2.createTrackbar('Rotation', self.window, 0, 360, self.on_change_rotation)
    cv2.createTrackbar('Declination', self.window, 50, 100, self.on_change_declination)
    cv2.resizeWindow(self.window, 800, 800)
    self.draw()
    print("ready to align keogram. press any key when ready to make keogram")
    cv2.waitKey(0)
    if self.generate_transform_metadata_file_flag:
      print("generating transform metadata file...")
      self.generate_transform_metadata_file()
      print("generating image sequence metadata file...")
      if self.generate_metadata_file_flag_M100:
        import image_sequence_metadata_M100
        image_sequence_metadata_M100.iterate_through_image_directory(self.path)
      elif self.generate_metadata_file_flag_ZVE10:
        import image_sequence_metadata_ZVE10
        image_sequence_metadata_ZVE10.iterate_through_image_directory(self.path)
    if self.generate_contact_sheet_flag:
      print("generating contact_sheet...")
      self.generate_contact_sheet()
      print("contact_sheet generation complete. press any key")
    if self.generate_keogram_flag:
      print("generating keogram...")
      self.generate_keogram()
      print("keogram generation complete. press any key")
    print("complete!")
    cv2.destroyAllWindows()

  def generate_transform_metadata_file(self):
    metadata = {}
    metadata["keogram_height"] = int(self.prescale)
    metadata["root_path"] = self.path
    metadata["rotation"] = self.rotation
    metadata["declination"] = self.declination

    json_object = json.dumps(metadata, indent=2)

    with open(os.path.join(self.path, "transform_metadata.json"),"w") as f:
      f.write(json_object)


  def draw(self):
    img = self.load_image()
    img_rotated = self.rotate_image(img)
    cv2.line(img_rotated, (img_rotated.shape[1]//2,0), (img_rotated.shape[1]//2,img_rotated.shape[0]), (255, 0, 128), 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3
    fontColor = (255,255,255)
    thickness = 3
    lineType = 2

    cv2.putText(img_rotated, f"DECLINATION: {self.declination}", (10, 100), font, fontScale, fontColor, thickness, lineType)
    cv2.putText(img_rotated, f"IMG ROTATION: {self.rotation}", (10, 200), font, fontScale, fontColor, thickness, lineType)
    cv2.putText(img_rotated, f"IMG INDEX: {self.index}", (10, 300), font, fontScale, fontColor, thickness, lineType)

    cv2.imshow(self.window, img_rotated)

  def process_cli(self):
    parser = ArgumentParser()
    parser.add_argument('--srcfolder', required=True, help='path to folder with source images')
    parser.add_argument('--dstfolder', required=True, help='path to folder to deposit transformed images')
    parser.add_argument('--size', default="4000", required=False, help='image size in pixels to scale the image (image will be square dimensions 1:1)')

    self.args = vars(parser.parse_args())
    self.srcfolder = self.args["srcfolder"]
    self.dstfolder = self.args["dstfolder"]
    self.size = int(self.args["size"])
    self.image_file_list = self.get_image_file_list(self.srcfolder)
    print(f"image_file_list: {self.image_file_list}")

  def get_image_size(self, image_path):
    dim = None
    try:
      im = Image.open(image_path)
      width, height = im.size
      dim = { "width": width, "height": height }
    except:
      pass
    return dim


  # def resize_images(self, input_path):
  #   try:
  #     files = fnmatch.filter(os.listdir(input_path), '*.jpg')
  #   except:
  #     print(f"ERROR: could not open folder for resizing operation: {input_path}")
  #     print(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
  #     return
  #   output_path = os.path.join(input_path,SCALED_FOLDER_NAME)

  #   try:
  #     os.mkdir(output_path)
  #   except OSError as error:
  #     print(error)

  #   files.sort()
  #   files_to_scan_count = len(files)
  #   for f in files:
  #     file_path = os.path.join(input_path, f)
  #     im = JPEGImage(file_path)
  #     image_dims = self.get_image_size(file_path)
  #     operations = {"crop": False, "resize": False}

  #     if image_dims:
  #       if image_dims["height"] > RESIZE_DIMENSION:
  #         operations["resize"] = True
  #       if image_dims["height"] < image_dims["width"]:
  #         operations["crop"] = True
  #     else:
  #       # could not determine image dimesnsions: skip:
  #        print(f"ERROR: {file_path}: image size could not be determined")

  #     if operations["crop"]:
  #       x = image_dims["width"] - image_dims["height"]
  #       x = int(x * 0.5 / 16) * 16
  #       y = 0
  #       height = image_dims["height"]
  #       width = height
  #       im = im.crop(x, y, width, height)
  #     if operations["resize"]:
  #       im = im.downscale(RESIZE_DIMENSION, RESIZE_DIMENSION)
     
  #     print(f"{file_path}: {image_dims['width']}x{image_dims['height']}\tcrop: {operations['crop']}\tresize:{operations['resize']}")

  #     if any(value == True for value in operations.values()):
  #       try:
  #         save_path = os.path.join(output_path, f)
  #         im.save(save_path)
  #         files_to_scan_count -= 1
  #       except:
  #         print(f"ERROR: could not save {save_path}")
  #         print(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
  #     else:
  #       files_to_scan_count -= 1
  #   if files_to_scan_count == 0:
  #     print("resize complete without error")
  #     with open(os.path.join(input_path, 'sequence_log.txt'), 'a') as f:
  #       f.write("# RESIZE COMPLETE\n")


    # else:
    #   print(f"ERROR: unresized image count: {files_to_scan_count}") 
    # return files_to_scan_count == 0 # if 0 then successfully handled all files


  def folder_status(self, path):
    status = self.parse_sequence_log(path)
    status["file_count"] = self.count_images(path)
    return status
  
  def rotateAndResizeImage(self, image_path, size):
    self.get_exif_object(image_path)
    exif_object = self.getExifOrientation(image_path)
    orientation = exif_object.get("orientation")
    orientation_degrees = remap_orientation.get(orientation, 0)
    image = cv2.imread(image_path)
    image = self.rotate_image_and_square_canvas(image, orientation_degrees)
    image = cv2.resize(image, (size, size))
    return image

if __name__ == '__main__':
  sg = SequenceGenerator()
  sg.process_cli()

