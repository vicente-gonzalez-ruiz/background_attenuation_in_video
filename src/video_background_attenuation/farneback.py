import numpy as np
import cv2
import motion_estimation

def attenuate_background_img(
        prev_img,
        next_img,
        background_img,
        alpha=0.99,
        initial_flow=None,
        levels=3,
        winsize=17,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2):
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_img.astype(np.float32),
        next=next_img.astype(np.float32),
        flow=initial_flow,
        pyr_scale=0.5,
        levels=levels,
        winsize=ME_winsize,
        iterations=ME_iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW) #cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    motion_magnitude, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    background_pixels = np.where((motion_magnitude < 1), prev_img, 0)
    difference_img = next_img.astype(np.int16) - background_img
    difference_img = np.clip(difference_img, 0, 255).astype(np.uint8)
    background_img = alpha*background_img + (1 - alpha)*background_pixels

    if __debug__:
        cv2_imshow(difference_img)

    return difference_img, background_img, flow

def attenuate_background_seq(
        input_sequence_path='input/',
        output_sequence_path='output/',
        img_extension=".jpg",
        first_img_index=0,
        last_img_index=120,
        alpha=0.99,
        levels=3,
        winsize=17,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2):
    first_img_path = input_sequence_path + str(first_img_index) + img_extension
    prev_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
    assert prev_img is not None, first_img_path
    background_img = np.zeros_like(prev_img) # Ojo
    initial_flow = np.zeros((background_img.shape[0], background_img.shape[1], 2), dtype=np.float32)
    for i in range(first_img_index, last_img_index):
        next_img_path = input_sequence_path + str(i) + img_extension
        next_img = cv2.imread(next_img_path, cv2.IMREAD_UNCHANGED)
        assert next_img is not None, next_img_path
        difference_img, background_img, flow = attenuate_background_img(
            prev_img,
            next_img,
            background_img,
            alpha,
            initial_flow,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma)
        prev_img = next_img
        initial_flow = flow

        if __debug__:
          print("background")
          cv2_imshow(background_img)
          print("prev")
          cv2_imshow(prev_img)
          print("difference")
          cv2_imshow(difference_img)

        difference_img_path = output_sequence_path + str(i) + img_extension
        cv2.imwrite(difference_img_path, difference_img)

