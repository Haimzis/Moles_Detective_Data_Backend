import params
import numpy as np
import cv2
import tensorflow as tf
import produce_crops
import utils
import os


def create_artifact_name(artifacts_amount, transplanted_data_list):
    name = 'fk_{0}'.format(artifacts_amount)
    for data_name in transplanted_data_list:
        name += '_{0}'.format(data_name)
    name += '_'
    return name


def produce_fake_data(transplanted_data, containers_data, amount=params.fake_data_artifacts_amount):
    artifacts_amount = 0
    utils.generated_data_directories_init()
    while artifacts_amount < amount:
        for container in containers_data:
            if artifacts_amount >= amount:
                break
            container_mask = cv2.imread(container['label'], -1)
            if not container_mask.any():
                continue
            container_img = cv2.imread(container['input'], -1)
            full_container_img, full_container_mask, transplanted_data_list = fill_container(container_img, container_mask, transplanted_data)
            produce_crops.generate_crops_from_image(create_artifact_name(artifacts_amount, transplanted_data_list), full_container_img, full_container_mask)
            print('fk_{0} created'.format(artifacts_amount))
            artifacts_amount += 1


def is_area_valid_for_transplantation(transplanted_object_mask, container_mask, transplant_coords):# crop_coords = [ymin, ymax, xmin, xmax]
    temp_obj_mask = np.zeros(tuple(container_mask.shape))
    temp_obj_mask[transplant_coords[0]: transplant_coords[1], transplant_coords[2]: transplant_coords[3]] = transplanted_object_mask
    temp_container_mask = (container_mask // np.max(container_mask)).astype('uint8') * 255
    temp_container_mask = cv2.bitwise_not(temp_container_mask)
    temp_obj_mask = cv2.bitwise_and(temp_obj_mask, temp_obj_mask, mask=temp_container_mask)
    if not temp_obj_mask.any():
        return True
    return False


def fix_object_crop(transplanted_object_img, transplanted_object_mask):
    gray_scale_object_img = cv2.cvtColor(transplanted_object_img.astype('float32'), cv2.COLOR_RGB2GRAY).astype('uint8')
    color_high = np.array(np.max(gray_scale_object_img))
    color_low = np.array(np.subtract(np.max(gray_scale_object_img), params.color_elimination_const), dtype=np.uint8)
    skin_mask = cv2.inRange(gray_scale_object_img, color_low, color_high)
    com_skin_transplanted_object_mask = cv2.bitwise_not(skin_mask)
    transplanted_object_mask = cv2.bitwise_and(transplanted_object_mask, transplanted_object_mask,
                                               mask=com_skin_transplanted_object_mask)

    # fix single pixels
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(transplanted_object_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    transplanted_object_mask = cv2.bitwise_and(transplanted_object_mask, opening,
                                               mask=closing)

    transplanted_object_img = cv2.bitwise_and(transplanted_object_img, transplanted_object_img,
                                              mask=transplanted_object_mask)
    return transplanted_object_img, transplanted_object_mask


def transplant_object_by_filter(container_img, container_mask, transplanted_data, filter_coords, container_updated_mask):
    appearance_amount = [1]  # ready for addition
    rand_index = np.random.choice(len(appearance_amount), 1, p=[1.0])[0]#0.61, 0.23, 0.095, 0.05, 0.01, 0.005])[0]
    rand_indexes = np.random.choice(len(transplanted_data), appearance_amount[rand_index])
    chosen_objects_to_transplant = np.array(transplanted_data)[rand_indexes]
    transplanted_data_names_list = []
    for object in chosen_objects_to_transplant:
        transplanted_object_size = (np.random.randint(40, 70, 1))[0]
        xmin = np.random.randint(filter_coords[2], filter_coords[3] - transplanted_object_size)
        ymin = np.random.randint(filter_coords[0], filter_coords[1] - transplanted_object_size)
        xmax = xmin + transplanted_object_size
        ymax = ymin + transplanted_object_size
        transplant_coords = [ymin, ymax, xmin, xmax]
        transplanted_object_img_path = object['input']
        transplanted_object_mask_path = object['label']
        transplanted_data_names_list.append(transplanted_object_img_path.split('/')[-1].split('.')[0])
        transplanted_object_img = cv2.imread(transplanted_object_img_path, -1)
        transplanted_object_mask = cv2.imread(transplanted_object_mask_path, -1)
        transplanted_object_img = cv2.resize(transplanted_object_img, (xmax - xmin, ymax - ymin))
        transplanted_object_mask = cv2.resize(transplanted_object_mask, (xmax - xmin, ymax - ymin), cv2.INTER_NEAREST)
        transplanted_object_img, transplanted_object_mask = fix_object_crop(transplanted_object_img, transplanted_object_mask)
        if is_area_valid_for_transplantation(transplanted_object_mask, container_mask, transplant_coords):
            container_updated_mask[ymin: ymax, xmin: xmax] = cv2.dilate(transplanted_object_mask,
                                                                        np.ones((3, 3), np.uint8), iterations=1)
            container_updated_mask = container_updated_mask.astype('uint8')
            temp_mask = np.zeros(tuple(container_updated_mask.shape), dtype=np.uint8)
            temp_mask[ymin: ymax, xmin: xmax] = transplanted_object_mask
            temp_img = np.zeros(tuple(container_img.shape))
            temp_img[ymin: ymax, xmin: xmax] = transplanted_object_img
            temp_img = temp_img.astype('uint8')
            nbitw_full_container_mask = cv2.bitwise_not(temp_mask.astype('uint8'))
            container_img = cv2.bitwise_and(container_img, container_img, mask=nbitw_full_container_mask)
            try:
                container_img = cv2.addWeighted(container_img, 1.0, temp_img, 1.0, 0)
            except cv2.error:
                print(transplant_coords)
                print('container_img ', container_img.dtype)
                print('temp ', temp_img.dtype)
                print(cv2.error.msg)
    return container_img, container_updated_mask, transplanted_data_names_list


def fill_container(container_img, container_mask, transplanted_data, filter_size=params.filter_size, stride=params.filter_size):
    stride = min(stride, container_img.shape[0], container_img.shape[1])
    filter_coords = [0, min(filter_size, container_img.shape[0]), 0, min(filter_size, container_img.shape[1])]  # crop_coords = [ymin, ymax, xmin, xmax]
    container_updated_mask = np.zeros(tuple(container_mask.shape), dtype=np.uint8)
    transplanted_data_list = []
    while filter_coords[1] <= container_img.shape[0]:
        while filter_coords[3] <= container_img.shape[1]:
            container_img, container_updated_mask, transplanted_data_names_list = transplant_object_by_filter(container_img, container_mask, transplanted_data, filter_coords, container_updated_mask)
            transplanted_data_list += transplanted_data_names_list
            filter_coords[2] += stride
            filter_coords[3] += stride
        filter_coords[0] += stride
        filter_coords[1] += stride
        filter_coords[2] = 0
        filter_coords[3] = min(filter_size, container_img.shape[1])
    return container_img, container_updated_mask, transplanted_data_list
