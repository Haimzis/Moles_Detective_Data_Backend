import cv2
import numpy as np
import utils

## TODO: need to improve this algorithm - take the distance parameter into account
def distance(pixel1, pixel2):
    return ((pixel1[1] - pixel2[1]) ** 2 + (pixel1[0] - pixel2[0]) ** 2) ** 0.5


def calculate_max_diameter(mask):
    aligned_mask = utils.align(mask)
    max_diameter = 0
    for i in range(0, aligned_mask.shape[0]):
        for j in range(0, aligned_mask.shape[0]):
            if aligned_mask[i, 0] != 0 and aligned_mask[j, aligned_mask.shape[1] - 1] != 0:
                diameter = distance([i, 0], [j, aligned_mask.shape[1] - 1])
                if max_diameter < diameter:
                    max_diameter = diameter
    return max_diameter


def size_eval(mask):
    # dpi = dots per inch
    # https://www.howtogeek.com/339665/how-to-find-your-android-devices-info-for-correct-apk-downloads/
    # that how to find the dpi.
    dpi = 96 # stam
    diameter_mm = (calculate_max_diameter(mask)*1.27) / dpi
    if diameter_mm > 5.0:
        return 1.0
    else:
        return ((diameter_mm / 0.5) + 1) / 10

if __name__ == '__main__':
    mask = cv2.imread('/home/haimzis/Downloads/masks(1)/masks/7.png' , -1)
    print(size_eval(mask))