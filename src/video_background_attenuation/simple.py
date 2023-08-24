import numpy as np
import cv2

def attenuate_background(prev_img, next_img):
    background = prev_img
    difference = next_img.astype(np.int16) - background
    difference = np.clip(difference, 0, 255).astype(np.uint8)

    if __debug__:
        cv2_imshow(difference)

    return difference
