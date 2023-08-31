'''Video background attenuation using motion estimation (optical flow
by Farneback).

'''

import numpy as np
import cv2
import motion_estimation # pip install "motion_estimation @ git+https://github.com/vicente-gonzalez-ruiz/motion_estimation"

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
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

def attenuate_background_img(
        prev_img,
        next_img,
        background_img,
        alpha=0.99,
        threshold=1,
        initial_flow=None,
        levels=3,
        winsize=17,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2):
  
    if IN_COLAB:
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logging.debug(f"threshold={threshold}")
            print(f"prev ({prev_img.dtype})")
            cv2_imshow(prev_img)
            print(f"next ({next_img.dtype})")
            cv2_imshow(next_img)
            print(f"background ({background_img.dtype})")
            cv2_imshow(background_img)

    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_img.astype(np.float32),
        next=next_img.astype(np.float32),
        flow=initial_flow,
        pyr_scale=0.5,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW) #cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    motion_magnitude, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    background_pixels = np.where((motion_magnitude < 1), prev_img, 0)

    if IN_COLAB:
        if logger.getEffectiveLevel() <= logging.DEBUG:
            print(f"background_pixels ({background_pixels.dtype})")
            cv2_imshow(background_pixels)

    difference_img = next_img.astype(np.int16) - background_img
    difference_img = np.clip(difference_img, 0, 255).astype(np.uint8)
    background_img = alpha*background_img + (1 - alpha)*background_pixels

    if IN_COLAB:
        if logger.getEffectiveLevel() < logging.WARNING:
            print(f"attenuated ({difference_img.dtype})")
            cv2_imshow(difference_img)
  
    return difference_img, background_img, flow

def attenuate_background_seq(
        input_sequence_prefix="/tmp/input/",
        output_sequence_prefix="/tmp/output/",
        img_extension=".jpg",
        first_img_index=0,
        last_img_index=120,
        alpha=0.99,
        threshold=1,
        levels=3,
        winsize=17,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2):

    assert alpha <=1.0
    first_img_prefix = input_sequence_prefix + str(first_img_index) + img_extension
    prev_img = cv2.imread(first_img_prefix, cv2.IMREAD_UNCHANGED).astype(np.float32)
    assert prev_img is not None, first_img_prefix
    background_img = np.zeros_like(prev_img) # Ojo
    initial_flow = np.zeros((background_img.shape[0], background_img.shape[1], 2), dtype=np.float32)
    for i in range(first_img_index + 1, last_img_index):
        next_img_prefix = input_sequence_prefix + str(i) + img_extension
        next_img = cv2.imread(next_img_prefix, cv2.IMREAD_UNCHANGED).astype(np.float32)
        assert next_img is not None, next_img_prefix
        difference_img, background_img, flow = attenuate_background_img(
            prev_img,
            next_img,
            background_img,
            alpha,
            threshold,
            initial_flow,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma)
        prev_img = next_img
        initial_flow = flow

        difference_img_prefix = output_sequence_prefix + str(i) + img_extension
        cv2.imwrite(difference_img_prefix, difference_img)

    if IN_COLAB:
        if logger.getEffectiveLevel() < logging.WARNING:
            print(f"background ({background_img.dtype})")
            cv2_imshow(background_img)

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
    
    parser.add_argument("-a", "--inertia", type=int_or_str,
                        help="Background inertia (difficulty to change)",
                        default=0.99)
    
    parser.add_argument("-t", "--threshold", type=int_or_str,
                        help="Below threshold no motion is detected",
                        default=1)

    parser.add_argument("-v", "--levels", type=int_or_str,
                        help="Number of levels in the Gaussian pyramid",
                        default=3)

    parser.add_argument("-w", "--winsize", type=int_or_str,
                        help="Side (in pixels) of the applicability window",
                        default=17)

    parser.add_argument("-r", "--iterations", type=int_or_str,
                        help="Number of iterations of the Farneback's motion estimator",
                        default=17)

    parser.add_argument("-n", "--poly_n", type=int_or_str,
                        help="Degree of the polynomial expansion used by the Farneback's motion estimator",
                        default=5)

    parser.add_argument("-s", "--poly_sigma", type=int_or_str,
                        help="Variance of the Gaussian window used by the Farneback's motion estimator",
                        default=1.2)

    args = parser.parse_args()

    attenuate_background_seq(
        input_sequence_prefix=args.input,
        output_sequence_prefix=args.output,
        img_extension=args.extension,
        first_img_index=args.first,
        last_img_index=args.last,
        alpha=args.inertia,
        threshold=args.threshold,
        levels=args.levels,
        winsize=args.winsize,
        iterations=args.iterations,
        poly_n=args.poly_n,
        poly_sigma=args.poly_sigma)
