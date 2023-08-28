import numpy as np
import cv2

def attenuate_background_img(
        prev_img,
        next_img,
        background_img,
        alpha=0.99,
        threshold=10):

    if logger.getEffectiveLevel() <= logging.DEBUG:
        logging.debug(f"threshold={threshold}")
        print(f"prev ({prev_img.dtype})")
        cv2_imshow(prev_img)
        print(f"next ({next_img.dtype})")
        cv2_imshow(next_img)
        print(f"background ({background_img.dtype})")
        cv2_imshow(background_img)

    difference_img = next_img - background_img
    difference_img = np.clip(difference_img, 0.0, 255.0)
    background_pixels = np.where((difference_img > threshold), prev_img, 0.0)

    if logger.getEffectiveLevel() <= logging.DEBUG:
        print(f"background_pixels ({background_pixels.dtype})")
        cv2_imshow(background_pixels)

    background_img = alpha*background_img + (1 - alpha)*background_pixels
    #background_img = next_img

    if logger.getEffectiveLevel() < logging.WARNING:
        print(f"attenuated ({difference_img.dtype})")
        cv2_imshow(difference_img)

    return difference_img, background_img

def attenuate_background_seq(
        input_sequence_path='input/',
        output_sequence_path='output/',
        img_extension=".jpg",
        first_img_index=0,
        last_img_index=120,
        alpha=0.99,
        threshold=10):

    first_img_path = input_sequence_path + str(first_img_index) + img_extension
    prev_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    assert prev_img is not None, first_img_path
    background_img = np.zeros_like(prev_img)
    for i in range(first_img_index + 1, last_img_index):
        next_img_path = input_sequence_path + str(i) + img_extension
        next_img = cv2.imread(next_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        assert next_img is not None, next_img_path
        difference_img, background_img = attenuate_background_img(
            prev_img,
            next_img,
            background_img,
            alpha,
            threshold)
        prev_img = next_img

        difference_img_path = output_sequence_path + str(i) + img_extension
        cv2.imwrite(difference_img_path, difference_img.astype(np.uint8))

    if logger.getEffectiveLevel() < logging.WARNING:
        print(f"background ({background_img.dtype})")
        cv2_imshow(background_img)

if __main__:
    attenuate_background_seq(
        input_sequence_path="img_paper/Alicia/ImagesGRAYSCALE/",
        output_sequence_path="/tmp/",
        img_extension=".jpg",
        first_img_index=0,
        last_img_index=120,
        alpha=0.99,
        threshold=3)
