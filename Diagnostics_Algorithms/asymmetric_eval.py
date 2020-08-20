import cv2
import utils
import numpy as np


def eval_asymmetric(aligned_mask):
    """
    :param mask: segmentation mask of mole.
    :return: ratio between uncommon pixels and common pixels
             of the 2 parts that divided from the center both horizontal and vertical.
    """
    bias = 0.0  # for nice appearance in the bar (too avoid small scores)
    width_center = aligned_mask.shape[1] // 2
    height_center = aligned_mask.shape[0] // 2
    if aligned_mask.shape[1] % 2 == 0:
        left_half = aligned_mask[:, 0: width_center]
    else:
        left_half = aligned_mask[:, 0: width_center + 1]
    right_half = aligned_mask[:, width_center: aligned_mask.shape[1]]
    if aligned_mask.shape[0] % 2 == 0:
        upper_half = aligned_mask[0: height_center, :]
    else:
        upper_half = aligned_mask[0: height_center + 1, :]
    bottom_half = aligned_mask[height_center: aligned_mask.shape[0], :]
    left_half = cv2.flip(left_half, 1)
    bottom_half = cv2.flip(bottom_half, 0)
    or_vertical = cv2.bitwise_or(left_half, right_half)
    and_vertical = cv2.bitwise_and(left_half, right_half)
    result_vertical = cv2.bitwise_xor(or_vertical, and_vertical)
    or_horizontal = cv2.bitwise_or(upper_half, bottom_half)
    and_horizontal = cv2.bitwise_and(upper_half, bottom_half)
    result_horizontal = cv2.bitwise_xor(or_horizontal, and_horizontal)
    horizontal_score = np.sum(result_horizontal) / np.sum(or_horizontal)
    vertical_score = np.sum(result_vertical) / np.sum(or_vertical)
    return min(1.0, (horizontal_score + vertical_score) / 2 + bias)


if __name__ == '__main__':
    seg_mask = cv2.imread('/home/haimzis/Downloads/masks(1)/masks/5.png',  -1)
    aligned_mask = utils.align(seg_mask)
    cv2.imwrite('/home/haimzis/Downloads/masks/masks/7.png', seg_mask[:, :, 0])
    print(eval_asymmetric(aligned_mask))