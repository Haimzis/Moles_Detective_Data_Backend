import cv2
import numpy as np
import params


def pyramid_blending(l_container_img, l_transplanted_img, mask=None, maxLevels=None, filterSizeIm=None,
                     filterSizeMask=None):
    c_container_img = l_container_img.copy()
    c_transplanted_img = l_transplanted_img.copy()
    gp_container_img = [c_container_img]
    gp_transplanted_img = [c_transplanted_img]
    for i in range(params.laplacian_pyramid_level):
        c_container_img = cv2.pyrDown(c_container_img)
        gp_container_img.append(c_container_img)
        c_transplanted_img = cv2.pyrDown(c_transplanted_img)
        gp_transplanted_img.append(c_transplanted_img)

    c_container_img = gp_container_img[params.laplacian_pyramid_level - 1]
    lp_container_img = [c_container_img]
    for i in range(params.laplacian_pyramid_level - 1, 0, -1):
        gaussian_expanded = cv2.pyrUp(gp_container_img[i])
        laplacian = cv2.subtract(gp_container_img[i - 1], gaussian_expanded)
        lp_container_img.append(laplacian)

    c_transplanted_img = gp_transplanted_img[params.laplacian_pyramid_level - 1]
    lp_transplanted_img = [c_transplanted_img]
    for i in range(params.laplacian_pyramid_level - 1, 0, -1):
        gaussian_expanded = cv2.pyrUp(gp_transplanted_img[i])
        laplacian = cv2.subtract(gp_transplanted_img[i - 1], gaussian_expanded)
        lp_transplanted_img.append(laplacian)

    blended_image_pyramid = []
    for transplanted_img_lap, container_img_lap in zip(lp_transplanted_img, lp_container_img):
        laplacian = np.add(transplanted_img_lap, container_img_lap)
        blended_image_pyramid.append(laplacian)

    blended_image_reconstruct = blended_image_pyramid[0]
    for i in range(1, params.laplacian_pyramid_level):
        blended_image_reconstruct = cv2.pyrUp(blended_image_reconstruct)
        blended_image_reconstruct = cv2.add(blended_image_pyramid[i], blended_image_reconstruct)

    return blended_image_reconstruct
