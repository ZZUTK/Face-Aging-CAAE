import argparse
import sys
import os
import cv2

from os import getcwd
from os.path import join
from FaceCropAlign import align_and_crop_face

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dirs", required=True,
                help="directory path to input images")
ap.add_argument("-o", "--output_dirs", required=True,
                help="directory path to output images")
args = vars(ap.parse_args())

# constant
input_dirs = join(os.getcwd(), args["input_dirs"])
output_dirs = join(os.getcwd(), args["output_dirs"])

# Iterate all images in input_dirs
image_exts = ['jpg', 'png']
file_list = [fn for fn in os.listdir(input_dirs)
             if fn.split(".")[-1] in image_exts]
for file_name in file_list:
    # Read
    input_path = join(input_dirs, file_name)
    print("Read ", input_path, "...")
    image = cv2.imread(input_path)

    # Process
    outputs = align_and_crop_face(image)

    # Write
    for i, output in enumerate(outputs):
        splited_file_name = file_name.split(".")
        only_name, ext = splited_file_name[0], '.'.join(splited_file_name[1:])
        output_path = join(output_dirs, '%s_%d.%s' % (only_name, i, ext))
        print("Write ", output_path, "...")
        cv2.imwrite(output_path, output)
