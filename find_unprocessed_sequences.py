import os
import sys
import re
from argparse import ArgumentParser, ArgumentError
import fnmatch
from pathlib import Path
import numpy as np
import json
import cv2
import image_sequence_metadata
from pprint import pprint

"""
du -sh * | sort -h -r :
 79G    20220413 Yellowknife Auroreye
 74G    20220401 Yellowknife Auroreye
 41G    20220412 Yellowknife Auroreye
 18G    20220416 yellowknife auroreye
8.6G    seq_2023-04-03T05-49-53
8.6G    AUROREYE04-2023-04-03
7.8G    seq_2023-04-08T05-40-56
7.8G    AUROREYE04-2023-04-08T05-40-56
7.4G    20211101 Edmonton auroreye
7.3G    AUROREYE04-2023-04-15
6.7G    seq_2023-05-12T19-24-14
6.3G    seq_2023-03-24T03-22-19
4.3G    AUROREYE04-2023-04-16
3.5G    seq_2023-03-27T05-16-52_00000004
3.2G    seq_2023-03-24T05-18-22
2.7G    seq_2023-03-25T03-40-23
2.0G    seq_2023-03-26T04-49-37
1.9G    seq_2023-03-25T05-20-59
1.7G    seq_2023-04-24T03-48-25
1.7G    seq_2023-04-22T23-57-58
1.7G    seq_2023-03-23T04-33-09
1.7G    seq_2023-03-14T01-09-14
1.0G    seq_2023-04-24T05-33-34
677M    seq_2023-03-25T05-55-03
542M    seq_2023-03-23T07-46-51
465M    seq_2023-03-23T04-12-20
450M    seq_2023-04-24T02-06-53
450M    seq_2023-03-13T01-20-41
443M    seq_2023-04-24T00-30-38
353M    seq_2023-03-23T08-44-19
337M    seq_2023-03-25T05-39-05
305M    seq_2023-03-24T09-14-15 NA
258M    seq_2023-03-24T05-04-56
123M    seq_2023-03-23T23-47-07
117M    seq_2023-04-16T21-03-11
 57M    seq_2023-03-27T19-16-11
 38M    seq_2023-04-16T20-40-11
 35M    seq_2023-04-16T21-01-11
 35M    seq_2023-04-16T20-46-34
 35M    seq_2023-04-16T20-35-17
 32M    SD card not auto purged
 20M    seq_2023-04-16T06-26-55
 14M    seq_2023-03-27T05-16-52_00000003
 10M    seq_2023-04-14T09-23-49
8.1M    seq_2023-04-16T22-12-56
 80K    auroreyelogo1280.jpg
 16K    auroreyelogo128.png
 16K    Theauroreyelogo128.png
 12K    logo
4.0K    seq_2023-04-16T20-33-47
"""

class SequenceStatus():

  def __init__(self):
    self.sequence_folder_list = []
    self.path = None
    self.sequences = {}


  def analyze_directory(self, path):
    files = os.listdir( os.path.join(self.path, path) )
    r = re.compile("[0-9]{8}.jpg")
    jpgs = list(filter(r.match, files))
    print(f"{path}: {len(jpgs)} sequence jpgs found")


  def get_sequence_status(self, path):
    # find all folders in root path

    # for each folder find the count of sequence jpgs (or the max file number)
    # look for csv in the folder
    # look for json in the fodler
    # look mp4 in the folder

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for dir in dirs:
      self.analyze_directory(dir)


  def process_cli(self):
    parser = ArgumentParser()
    parser.add_argument('--path', required=False, default="/Volumes/X8 Kuzub/AurorEye/", help='path to root folder containing sequence folders')
    self.args = vars(parser.parse_args())
    for arg, value in self.args.items():
      print(arg, value)
    self.path = self.args["path"]
    self.get_sequence_status(self.path)


  def folder_status(self, path):
    status = self.parse_sequence_log(path)
    status["file_count"] = self.count_images(path)
    return status


if __name__ == '__main__':
  app = SequenceStatus()
  app.process_cli()

