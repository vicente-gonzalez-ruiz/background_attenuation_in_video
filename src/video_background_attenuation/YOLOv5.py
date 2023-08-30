'''Video background attenuation using YOLOv5 (https://github.com/vicente-gonzalez-ruiz/yolov5_no_bounding_boxes).
'''

# https://colab.research.google.com/drive/1XqAqfYMthcMk-tL9SBzfF-b9FPUc7G4J?usp=sharing

import numpy as np
import cv2
import os

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except:
    IN_COLAB = False
logger.info(f"Running in Google Colab: {IN_COLAB}")


def attenuate_background_seq(
        input_sequence_prefix="/tmp/input/",
        output_sequence_prefix="/tmp/output/",
        img_extension=".jpg",
        first_img_index=0,
        last_img_index=120):

    os.system("rm -rf yolov5/runs/predict-seg")
    os.system(f"python yolov5/segment/predict.py --save-crop --save-txt --weights yolov5x-seg.pt --source {input_sequence_prefix} --retina-masks --classes 0")
    images_folder = "yolov5/runs/predict-seg/exp/"
    os.system("ls yolov5/runs/predict-seg/exp/")
    list_of_imagenames = np.array([img for img in os.listdir(images_folder) if img_extension in img])
    counter = 0
    print(list_of_imagenames)
    for image_name in list_of_imagenames:
        orig_img = cv2.imread(os.path.join(images_folder, image_name))
        zeros_img = np.zeros_like(orig_img).astype(np.uint8)
        masked_img = cv2.imread(os.path.join(images_folder, image_name))
        print(counter, image_name)
        zeroed_img = np.where(masked_img[...,0] != masked_img[...,2], orig_img[...,0], 0)
        cv2.imwrite(os.path.join(output_sequence_prefix, image_name), zeroed_img)
        counter += 1

if __name__ == "__main__":

    def int_or_str(text):
        '''Helper function for argument parsing.'''
        try:
            return int(text)
        except ValueError:
            return text

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.description = __doc__

    parser.add_argument("-i", "--input", type=int_or_str,
                        help="Prefix of the input image sequence",
                        default="/tmp/input/")
    
    parser.add_argument("-o", "--output", type=int_or_str,
                        help="Prefix of the output image sequence",
                        default="/tmp/output/")

    parser.add_argument("-e", "--extension", type=int_or_str,
                        help="Image extension",
                        default=".jpg")

    parser.add_argument("-f", "--first", type=int_or_str,
                        help="Index of the first image",
                        default=0)

    parser.add_argument("-l", "--last", type=int_or_str,
                        help="Index of the last image",
                        default=120)

    args = parser.parse_args()

    os.system("rm -rf yolov5")
    os.system("git clone https://github.com/vicente-gonzalez-ruiz/yolov5_no_bounding_boxes.git")
    os.system("mv yolov5_no_bounding_boxes yolov5")
    os.system("pip install -r yolov5/requirements.txt")
    
    attenuate_background_seq(
        input_sequence_prefix=args.input,
        output_sequence_prefix=args.output,
        img_extension=args.extension,
        first_img_index=args.first,
        last_img_index=args.last)
