import cv2
import numpy as np
from numpy.core._multiarray_umath import ndarray
import params


############ Laplacian Pyramid Image Blending ############
def laplacian_pyramid_image_blending(transplanted_object_img, transplanted_object_mask, container_img,
                                     container_updated_mask, transplant_coords, filter_coords):
    """
    :param transplanted_object_img: object image
    :param transplanted_object_mask: object mask
    :param transplant_coords: where to transplant the object
    :param filter_coords: transplantation area
    """
    ymin, ymax, xmin, xmax = transplant_coords
    fymin, fymax, fxmin, fxmax = filter_coords
    try:
        container_updated_mask[ymin: ymax, xmin: xmax] = cv2.dilate(transplanted_object_mask,
                                                                    np.ones((3, 3), np.uint8), iterations=1)
        container_updated_mask = container_updated_mask.astype('uint8')
        transplanted_img = np.zeros(tuple(container_img.shape), dtype=np.uint8)
        transplanted_img[ymin: ymax, xmin: xmax, :] = transplanted_object_img
        temp_mask = np.zeros(tuple(container_updated_mask.shape), dtype=np.uint8)
        temp_mask[ymin: ymax, xmin: xmax] = transplanted_object_mask
        nbitw_full_container_mask = cv2.bitwise_not(temp_mask.astype('uint8'))
        container_img = cv2.bitwise_and(container_img, container_img, mask=nbitw_full_container_mask)
        l_transplanted_img = transplanted_img[fymin: fymax, fxmin: fxmax]
        l_container_img = container_img[fymin: fymax, fxmin: fxmax]
        container_img[fymin: fymax, fxmin: fxmax] = pyramid_blending(l_container_img, l_transplanted_img)
        pass
    ## TODO: prevent glur

    except cv2.error:
        print(transplant_coords)
        print('container_img ', container_img.dtype)
        print(cv2.error.msg)


def pyramid_blending(l_container_img, l_transplanted_img):
    """
    laplacian pyramid image blending algorithm
    :return: blended image
    """
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


############ Simple Image Blending ############
def my_image_blending(transplanted_object_img, transplanted_object_mask, container_img, container_mask, transplant_coords):
    """
    simple algorithm that I wrote for images blending
    :param transplanted_object_img: image to transplant in
    :param transplanted_object_mask: mask to transplant in
    :param container_img: the object image
    :param container_mask: the object mask
    :param transplant_coords: where to transplant
    :return: result image after blending
    """
    ymin, ymax, xmin, xmax = transplant_coords
    container_updated_mask = np.copy(container_mask)

    # 1) add the object mask to the updated container mask
    container_updated_mask[ymin: ymax, xmin: xmax] = transplanted_object_mask
    container_updated_mask = container_updated_mask.astype('uint8')

    # 2) set to 0 the transplantation area in the container image
    temp_mask = np.zeros(tuple(container_updated_mask.shape), dtype=np.uint8)
    temp_mask[ymin: ymax, xmin: xmax] = transplanted_object_mask
    nbitw_full_container_mask = cv2.bitwise_not(temp_mask.astype('uint8'))
    container_updated_img = cv2.bitwise_and(container_img, container_img, mask=nbitw_full_container_mask)

    # 3) put the object in the transplantation area in a container image size
    temp_img = np.zeros(tuple(container_updated_img.shape))
    temp_img[ymin: ymax, xmin: xmax] = transplanted_object_img
    temp_img: ndarray = temp_img.astype('uint8')

    cv2.imwrite('./Stam/6_mask.png', transplanted_object_mask)
    cv2.imwrite('./Stam/6_source.png', transplanted_object_img)
    cv2.imwrite('./Stam/6_target.png', container_img)


    # 4) add the two image, to complete container image
    container_updated_img = container_updated_img + temp_img

    return container_updated_img, container_updated_mask


