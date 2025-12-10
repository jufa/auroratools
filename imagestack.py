import os
import sys
from argparse import ArgumentParser, ArgumentError
import fnmatch
from pathlib import Path
import numpy as np
from PIL import ImageChops, Image


class ImageStack():

  def __init__(self):
    self.image_file_list = None
    self.path = None
    self.window = None
    self.index = 0

  def get_image_file_list(self, path):
      if self.image_file_list:
        return self.image_file_list
      input_path = path
      try:
        files = fnmatch.filter(os.listdir(input_path), '*.jpg')
      except Exception:
        print(f"ERROR: could not open folder: {input_path}")
        print(f"exception details: {sys.exc_info()[0]}: {sys.exc_info()[1]}")
        return
      # output_path = os.path.join(input_path,self.scaled_folder_name)
      files.sort()
      self.image_file_list = files
      return files
  
  def trim_file_range(self, file_list, first, last):
    def file_number(file_path):
      path = Path(file_path)
      try:
        file_number = int(path.stem)
        return file_number
      except Exception as e:
        print(f"ERROR: trim_file_range: {e}")

    def file_number_in_range(file_path, first, last):
      fn = file_number(file_path)
      return fn >= first and fn <= last
    
    trimmed_file_list = [item for item in file_list if file_number_in_range(item, first, last)]
    return trimmed_file_list
  
  def stack_image_sequence(self):
    files = self.get_image_file_list(self.path)
    trimmed_files = self.trim_file_range(files, self.first, self.last)
    # print(trimmed_files) # 0000004.xyz etc in order

    final_image = Image.open( Path(self.path, trimmed_files[0]) )
    file_count = len(trimmed_files)
    for i, file in enumerate(trimmed_files):
      current_image = Image.open( Path(self.path, file) )
      final_image = ImageChops.lighter(final_image, current_image)
      sys.stdout.write( f"\rprocessing file {i}/{file_count}")
    final_image.save("allblended.jpg","JPEG")



  def process_cli(self):
    parser = ArgumentParser()
    parser.add_argument('--path', required=True, help='path to folder to create keogram')
    parser.add_argument('--first', default="0", required=True, help='angle in degrees clockwise to rotate the image before sampling the keogram (assumes square image)')
    parser.add_argument('--last', default="0", required=False, help='image size in pixels to scale the image before sampling (assumes square image)')
    self.args = vars(parser.parse_args())
    for arg, value in self.args.items():
      print(arg, value)
    self.path = self.args["path"]
    self.first = int(self.args["first"])
    self.last = int(self.args["last"])



if __name__ == '__main__':
  app = ImageStack()
  app.process_cli()
  app.stack_image_sequence()






