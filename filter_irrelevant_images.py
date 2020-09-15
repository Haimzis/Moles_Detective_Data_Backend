import sys
import cv2
import numpy as np
import os
import ctypes
import params
import utils
from params import color_elimination_const
import fake_data_creation
import tensorflow as tf
import pb_inference
import shutil


def filter_segmented_segmentation_data_examples(data):
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
                f_img, f_mask = fake_data_creation.fix_object_crop(img, mask, color_elimination)
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


def filter_segmented_classification_data_examples(classification_data_path):
    images = tf.gfile.Glob(classification_data_path + '/*.jpg')  # All images
    extracted_object_images = tf.gfile.Glob(params.object_img_dir + '/*')
    extracted_object_images = list(map((lambda x: x.split('/')[-1].split('.')[0]), extracted_object_images))  # Already approved
    pb_inference.init_inference()

    for image_path in images:
        img_name = image_path.split('/')[-1].split('.')[0]
        if img_name in extracted_object_images:
            continue
        img, mask, overlay = pb_inference.quick_inference(image_path)
        results_container = np.zeros((params.INPUT_SIZE, params.INPUT_SIZE * 3, 3), np.uint8)
        results_container[0: params.INPUT_SIZE, 0: params.INPUT_SIZE, :] = img
        results_container[0: params.INPUT_SIZE, params.INPUT_SIZE: params.INPUT_SIZE * 2, :] = overlay
        results_container[0: params.INPUT_SIZE,params.INPUT_SIZE * 2: params.INPUT_SIZE * 3, :] = cv2.merge((mask, mask, mask))
        if not mask.any():
            shutil.move(image_path, image_path.replace(img_name + '.jpg', 'rejected/' + img_name + '.jpg'))
            print(img_name + ' rejected! - mask is empty')
            continue
        cv2.imshow('segmentation result', results_container)
        key = None
        while key != ord('n') and key != ord('s'):
            key = cv2.waitKey(0)
            if key == ord('n'):
                shutil.move(image_path, image_path.replace(img_name + '.jpg', 'rejected/' + img_name + '.jpg'))
                print(img_name + ' rejected!')
                continue
            elif key == ord('s'):
                utils.extract_single_object_from_both_img_mask(img, mask, img_name)
                print(img_name + ' approved!')
            else:
                print('Yogev you can only choose S to save or N to reject')
            sys.stdout.flush()
        cv2.destroyAllWindows()


def remove(data_dict):
    img_path = data_dict['input']
    mask_path = data_dict['label']
    os.remove(img_path)
    os.remove(mask_path)
