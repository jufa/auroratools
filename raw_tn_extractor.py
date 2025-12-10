import os
import sys
from re import X
from argparse import ArgumentParser, ArgumentError
import fnmatch
from pathlib import Path
import numpy as np
import json
import cv2
import image_sequence_metadata


class KeogramGenerator():

  def __init__(self):
    self.image_file_list = None
    self.path = None
    self.window = None
    self.rotation = 0
    self.index = 0
    self.declination = 0
    self.prerotate = 0
    self.prescale = 400

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
    self.image_file_list = files
    return files

  def make_keogram(self, path, prescale, prerotate):
    files = self.get_image_file_list(path)
    files_to_scan_count = len(files)

    on_first_image = True
    keogram = None
    curr_column = 0

    cv2.namedWindow("keo", cv2.WINDOW_NORMAL)
    
    try:
      for i, f in enumerate(files):
        print(f"generating keogram: [{i}/{files_to_scan_count}]\r", end="")
        file_path = os.path.join(path, f)
        input_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        
        # prerotate
        height, width = input_img.shape[:2]
        centerX, centerY = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D((centerX, centerY), -prerotate, 1.0)
        rotated = cv2.warpAffine(input_img, M, (width, height))

        img = cv2.resize(rotated, (prescale, prescale))

        # if i % 10 == 0:
        #   cv2.imshow("prescaled and rotated", img)

        height = img.shape[0]
        width = img.shape[1]
        keo_width = files_to_scan_count
        col_start = int(width / 2) - 1
        col_end = int(width / 2)
        slice = img[0:height, col_start:col_end] # r, c
        if on_first_image:
          on_first_image = False
          keogram = np.zeros((height, keo_width, 3), np.uint8)
          cv2.resizeWindow("keo", keo_width, height)
        keogram[0:height, curr_column:curr_column + 1] = slice
        # keogram = np.append(keogram, slice, axis=1) # slow
        curr_column += 1
        cv2.imshow("keo", keogram)
        cv2.waitKey(1)

      
      output_path = os.path.join(path, "keogram.png")
      cv2.imwrite(output_path, keogram)

      print(f"keogram complete without error: dimensions:{keogram.shape}")
    except:
      print(f"ERROR: could not create keogram")
      print(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
      return
    

  def make_contact_sheet(self, path, prescale, prerotate, rows=1, columns=20, image_size=200):
    image_padding_pixels = 16
    contact_sheet_width = (image_size + image_padding_pixels) * columns
    contact_sheet_height = image_size * rows
    contact_sheet = np.zeros((contact_sheet_height, contact_sheet_width, 3), np.uint8)
    files = self.get_image_file_list(path)
    total_files = len(files)
    file_sample_count = rows * columns
    offset_start = 3 # make sure we don't get setup frames from start or end of sequence
    offset_end = 1
    total_files = total_files - offset_start - offset_end
    step = total_files // file_sample_count
    files_for_contact_sheet = files[offset_start: total_files - offset_end: step]
    files_for_contact_sheet = files_for_contact_sheet[0:-1] # drop last one since it is over what we want by one
    files_to_scan_count = len(files_for_contact_sheet)

    print(f"files_for_contact_sheet:{files_for_contact_sheet}")

    on_first_image = True
    # contact_sheet = None

    cv2.namedWindow("keo", cv2.WINDOW_NORMAL)
    
    try:
      for i, f in enumerate(files_for_contact_sheet):
        print(f"generating contact_sheet: [{i}/{files_to_scan_count}]\r", end="\n")
        file_path = os.path.join(path, f)
        input_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        
        # prerotate
        height, width = input_img.shape[:2]

        centerX, centerY = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D((centerX, centerY), -prerotate, 1.0)
        rotated = cv2.warpAffine(input_img, M, (width, height))

        img = cv2.resize(rotated, (image_size, image_size))

        # if i % 10 == 0:
        #   cv2.imshow("prescaled and rotated", img)
        

        if on_first_image:
          on_first_image = False
          # contact_sheet = np.zeros((height, keo_width, 3), np.uint8)
          cv2.resizeWindow("keo", contact_sheet_width, contact_sheet_height)
        col_start = i * (image_size + image_padding_pixels)
        col_end = col_start + image_size
        contact_sheet[0:image_size, col_start:col_end] = img
        
        center_y = image_size // 2
        center_x = col_start + image_size // 2
        thickness = 4
        color = (0,0,0)
        # cv2.circle(contact_sheet, (center_x, center_y), (image_size + thickness) // 2, color, thickness, lineType=cv2.LINE_AA)

        # contact_sheet = np.append(contact_sheet, slice, axis=1) # slow
        cv2.imshow("keo", contact_sheet)
        cv2.waitKey(1)

      
      output_path = os.path.join(path, "contact_sheet.png")
      cv2.imwrite(output_path, contact_sheet)

      print(f"contact_sheet complete without error: dimensions:{contact_sheet.shape}")
    except:
      print(f"ERROR: could not create contact_sheet")
      print(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
      return

  def load_image(self):
    img = cv2.imread(os.path.join(self.path, self.image_file_list[self.index]), cv2.IMREAD_COLOR)
    return img

  def rotate_image(self, image):
    height, width = image.shape[:2]
    centerX, centerY = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D((centerX, centerY), -self.rotation - self.declination, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
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
    print("generating transform metadata file...")
    self.generate_transform_metadata_file()
    print("generating image sequence metadata file...")
    image_sequence_metadata.iterate_through_image_directory(self.path, None)
    print("generating contact_sheet...")
    self.generate_contact_sheet()
    print("contact_sheet generation complete. press any key")
    cv2.waitKey(0)
    print("generating keogram...")
    # self.generate_keogram()
    print("keogram generation complete. press any key")
    cv2.waitKey(0)
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
    # cv2.resizeWindow(window, 400, 400)


  def on_change_image(self, index):
    self.index = int(index)
    self.draw()


  def on_change_rotation(self, angle):
    self.rotation = angle
    self.draw()


  def on_change_declination(self, value):
    self.declination = value - 50 # slider only goes from 0
    self.draw()

  def generate_keogram(self):
    self.make_keogram(self.path, int(self.prescale), float(self.rotation) + float(self.declination))

  def generate_contact_sheet(self):
    self.make_contact_sheet(self.path, int(self.prescale), float(self.rotation) + float(self.declination))

  def process_cli(self):
    parser = ArgumentParser()
    parser.add_argument('--path', required=True, help='path to folder to create keogram')
    parser.add_argument('--keogram', default=False, action='store_true', required=False, help='create keogram png file')
    parser.add_argument('--contactsheet', default=False, action='store_true', required=False, help='create contact sheet png file')
    parser.add_argument('--metadata', default=False, action='store_true', required=False, help='create metadata file')
    parser.add_argument('--prerotate', default="0", required=False, help='angle in degrees clockwise to rotate the image before sampling the keogram (assumes square image)')
    parser.add_argument('--prescale', default="400", required=False, help='image size in pixels to scale the image before sampling (assumes square image)')

    self.args = vars(parser.parse_args())
    for arg, value in self.args.items():
      print(arg, value)
    self.path = self.args["path"]
    self.prerotate = int(self.args["prerotate"])
    self.prescale = int(self.args["prescale"])
    self.get_rotation_angle(self.path)
    # self.make_keogram(path, int(prescale), float(prerotate))


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




if __name__ == '__main__':
  kg = KeogramGenerator()
  kg.process_cli()

