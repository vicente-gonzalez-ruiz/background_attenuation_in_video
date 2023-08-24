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
    no_motion_pixels = np.where((motion_magnitude < 1), prev_img, 0)
    difference = next_img.astype(np.int16) - background_img
    difference = np.clip(difference, 0, 255).astype(np.uint8)
    background = alpha*background_img + (1 - alpha)*no_motion_pixels

    if __debug__:
        cv2_imshow(difference)

    return difference, background

def attenuate_background(
        sequence_path='.',
        extension=".jpg",
        first_img_index=0,
        last_img_index=120,
        alpha=0.99):
    first_img_path = sequence_path + str(first_img_index) + extension
    prev_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
    background_img = np.zeros_like(prev_img)
    initial_flow = np.zeros((background.shape[0], background.shape[1], 2), dtype=np.float32)
    for i in range(first_img_index, last_img_index):
        next_img_path = sequence_path + str(i) + extension
        next_img = cv2.imread(next_img_path, cv2.IMREAD_UNCHANGED)
