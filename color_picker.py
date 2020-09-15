import cv2
import numpy as np
import glob

image_hsv = None
pixel = (20, 60, 80)
image_src = None


# mouse callback function
def pick_color(event, x, y, flags, param):
    """
    clicking event - prints the color range of the area that have been clicked
    :param event: click event on specific img
    :param x: coord
    :param y: coord
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y, x]

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 15])
        lower = np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 15])
        print(list([list(lower), list(upper)]))

        image_mask = cv2.inRange(image_hsv, lower, upper)
        RGB_image_mask = cv2.merge((image_mask, image_mask, image_mask))
        image_src_without_wanted_color = cv2.bitwise_and(image_src, image_src, mask=cv2.bitwise_not(image_mask))
        overlay = cv2.addWeighted(image_src_without_wanted_color, 1.0, RGB_image_mask, 1.0, 0)
        cv2.imshow("overlay", overlay)


def main():
    import sys
    global image_hsv, pixel, image_src  # so we can use it in mouse callback
    files = glob.glob('/media/haimzis/Extreme SSD/Moles_Detector_Dataset/Classification/ISIC_2019_Training_Input/*.jpg')
    for file in files:
        image_src = cv2.imread(file, -1)  # pick.py my.png
        if image_src is None:
            print("the image read is None............")
            return
        cv2.imshow('bgr', image_src)

        ## NEW ##
        cv2.setMouseCallback('bgr', pick_color)

        # now click into the hsv img , and look at values:
        image_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
