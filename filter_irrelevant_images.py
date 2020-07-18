import cv2
import numpy as np
import os
import ctypes
from params import color_elimination_const
import fake_data_creation


def filter_flow(data):
    for data_dict in data:
        img_path = data_dict['input']
        img = cv2.imread(img_path, -1)
        cv2.imshow('img', img)
        key = cv2.waitKey(0)
        if key == ord('d'):
            remove(data_dict)
            try:
                ctypes.windll.user32.MessageBoxW(0, "image and mask removed", "Action Succeed", 1)
            except:
                print("image and mask removed")
        elif key == ord('m'):
            try:
                ctypes.windll.user32.MessageBoxW(0, "moved on", "Action Succeed", 1)
            except:
                print("moved on")
        elif key == ord('a'):
            print("setting color threshold: starting from {0}".format(color_elimination_const))
            mask_path = data_dict['label']
            mask = cv2.imread(mask_path, -1)
            cv2.imshow('img', img)
            cv2.imshow('mask', mask)
            key = cv2.waitKey(0)
            color_elimination = color_elimination_const
            while key != ord('s') or key != ord('m'):
                if key == ord('u'):
                    color_elimination += 1
                elif key == ord('j'):
                    color_elimination -= 1
                f_img, f_mask = fake_data_creation.fix_object_crop(img, mask,color_elimination)
                img = cv2.imread(img_path, -1)
                mask = cv2.imread(mask_path, -1)
                cv2.imshow('img', f_img)
                cv2.imshow('mask', f_mask)
                key = cv2.waitKey(0)
            if key == ord('s'):
                # save changes
                pass
        else:
            try:
                ctypes.windll.user32.MessageBoxW(0, "wrong key", "Action Failed", 1)
            except:
                print("wrong key")


def remove(data_dict):
    img_path = data_dict['input']
    mask_path = data_dict['label']
    os.remove(img_path)
    os.remove(mask_path)