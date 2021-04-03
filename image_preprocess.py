import cv2
import numpy as np


def clear_noise(img):
    """
    clear noise func
    :param img: input
    :return: img after noise cleaning
    """
    return cv2.bilateralFilter(img, 9, 10, 75)


def improve_contrast(img):
    """
    :param img: input
    :return: histogram of contrast
    """
    if len(img.shape) > 2:
        raise NameError('expect for gray-scale input')
    return cv2.equalizeHist(img)


def change_color_space(img):
    """
    :param img: bgr / rgb input
    :return: gray-scale img
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def hair_removal(img):
    """
    hair-removal algorithm
    :param img: input
    :return: image after hair removal
    """
    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (5, 5))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 5, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)
    return dst


def gaussian_blur(img_crop):
    """
    :param img_crop: input
    :return: blurred image
    """
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(img_crop, -1, kernel)


def HZ_preprocess(img, hair=False):
    """
    my algorithm for image for training - based on many articles
    :param img: input
    :param hair: has hair or not
    """
    if hair:
        return clear_noise(improve_contrast(change_color_space(hair_removal(img))))
    else:
        return clear_noise(improve_contrast(change_color_space(img)))


if __name__ == '__main__':
    img_path = '/home/haimzis/PycharmProjects/DL_training_preprocessing/Data/img/28_02_2020_20_37/VID_20200228_202138/frame0.jpg'
    input = cv2.imread(img_path)
    #dst = HZ_preprocess(input, True)
    dst = hair_removal(input)
    cv2.imwrite(img_path + 'w.png', dst)



