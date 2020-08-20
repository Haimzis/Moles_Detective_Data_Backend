import cv2
import numpy as np
import math

# colors_dict = {'White': [[142, 18, 16], [255, 255, 255]],
#                'Black': [[0, 0, 0], [49, 49, 49]],
#                'Red': [[150, 50, 50], [255, 0, 0]],
#                'Light-brown': [[150, 50, 0], [200, 150, 100]],
#                'Dark-brown': [[50, 0, 0], [150, 100, 100]],
#                'Blue-gray': [[0, 100, 150], [150, 125, 150]]}
colors_dict = {#'White': [[142, 18, 16], [255, 255, 255]],
               #'Black': [[0, 0, 0], [49, 49, 49]],
               'Red': [[150, 60, 50], [255, 0, 0]],
               'Light-brown': [[92, 60, 39], [95, 66, 50]],
               'Dark-brown': [[50, 0, 0], [150, 100, 100]],
               'Blue-gray': [[0, 100, 150], [150, 125, 150]]}

# stuck with mean and covariance
def color_eval(segmented_mole_image):
    b, g, r = cv2.split(segmented_mole_image)

    # mean
    b_sum, g_sum, r_sum = np.sum(b), np.sum(g), np.sum(r)
    n = np.count_nonzero(segmented_mole_image) // 3
    channels_mean_tuple = (r_sum // n, g_sum // n, b_sum // n)
    # covariance
    # rr rg rb gr gg gb br bg bb
    covariance_channels_mul_sum = []
    channels_mean_mul_mat = []
    covariance_mat = []
    for i, channel_mean1 in enumerate(channels_mean_tuple):
        for j, channel_mean2 in enumerate(channels_mean_tuple):
            covariance_channels_mul_sum.append(segmented_mole_image[:, :, i].astype('float32') * segmented_mole_image[:, :, j].astype('float32'))
            channels_mean_mul_mat.append(channel_mean1*channel_mean2)
    for i in range(0, 9):
        covariance_mat.append(np.sum(covariance_channels_mul_sum[i]) // n - channels_mean_mul_mat[i])  # covariance formula

    return None


if __name__ == '__main__':
    img = cv2.imread(
        '/Output/objects_extraction/segmentation_purpose/transparent/ISIC_0000006.png', -1)
    color_eval(img)
