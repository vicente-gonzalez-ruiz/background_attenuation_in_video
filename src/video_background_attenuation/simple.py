import numpy as np
import cv2

def attenuate_background_img(
        prev_img,
        next_img,
        background_img,
        alpha=0.99,
        threshold=10):
    difference_img = next_img.astype(np.int16) - background_img
    background_pixels = np.where((difference_img < threshold), prev_img, 0)
    difference_img = np.clip(difference_img, 0, 255).astype(np.uint8)
    background_img = alpha*background_img + (1 - alpha)*background_pixels

    if __debug__:
        cv2_imshow(difference_img)

    return difference_img, background_img

def attenuate_background_seq(
        input_sequence_path='input',
        output_sequence_path='output',
        img_extension=".jpg",
        first_img_index=0,
        last_img_index=120,
        alpha=0.99):
    first_img_path = input_sequence_path + str(first_img_index) + img_extension
    prev_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
    background_img = prev_img # Ojo
    for i in range(first_img_index, last_img_index):
        next_img_path = output_sequence_path + str(i) + img_extension
        next_img = cv2.imread(next_img_path, cv2.IMREAD_UNCHANGED)
        difference_img, background_img = attenuate_background_img(
            prev_img,
            next_img,
            background_img)
        prev_img = next_img

    if __debug__:
        cv2_imshow(difference_img)

        difference_img_path = sequence_path + str(i) + "_BG_attenuated_" + img_extension
        cv2.imwrite(difference_img_path, difference_img)
