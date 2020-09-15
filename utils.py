import os
import numpy as np
import tensorflow as tf
import cv2
import params


def find_object_coords(object_mask, coords=None):
    """
    find object coordination in a mask input
    :param object_mask: mask input
    :param coords: [ymin, ymax, xmin, xmax] - sub coords that contains the object
    :return: [ymin, ymax, xmin, xmax] - minimal size image coords that contains the object
    """
    min_mask_size = 5
    if object_mask.shape[0] < min_mask_size or object_mask.shape[1] < min_mask_size:
        raise Exception('mask size too small for analyze, must be atleast ({0}, {0})'.format(min_mask_size))

    if coords is None:
        min_y = 0
        max_y = object_mask.shape[0] - 1
        min_x = 0
        max_x = object_mask.shape[1] - 1
    else:
        min_y = coords[0]
        max_y = coords[1]
        min_x = coords[2]
        max_x = coords[3]

    # y coords
    while not object_mask[min_y, min_x:max_x].any():
        min_y += 1
    while not object_mask[max_y, min_x:max_x].any():
        max_y -= 1

    # x coords
    while not object_mask[min_y:max_y, min_x].any():
        min_x += 1
    while not object_mask[min_y:max_y, max_x].any():
        max_x -= 1

    return [min_y, max_y, min_x, max_x]


def extract_object_from_both_img_mask(data):
    """
    objects extraction func, by bitwise operations
    :param data:many data frames: each one is: img and mask of the object
    :return: writes img and label of the object only - every pixel that dont belongs to the object will be black
    """
    for data_dict in data:
        object_img_path = data_dict['input']
        object_mask_path = data_dict['label']
        object_img = cv2.imread(object_img_path, -1)
        object_mask = cv2.imread(object_mask_path, -1)
        if not object_mask.any():
            continue
        object_coords = find_object_coords(object_mask)
        object_img_cropped = object_img[object_coords[0]:object_coords[1], object_coords[2]:object_coords[3], :]
        object_mask_cropped = object_mask[object_coords[0]:object_coords[1], object_coords[2]:object_coords[3]]
        output_img_path = params.output_img_extraction + '/' + object_img_path.split('/')[-1].split('.')[0] + '.png'
        try:
            final_output_img_cropped = cv2.bitwise_and(object_img_cropped, object_img_cropped, mask=object_mask_cropped)
            cv2.imwrite(output_img_path, final_output_img_cropped)
            cv2.imwrite(params.output_mask_extraction + '/' + object_mask_path.split('/')[-1], object_mask_cropped)

        except cv2.error:
            print('failed!')
            print(object_img_path)
            print(object_coords)
            print(cv2.error.msg)
            break
        print(object_img_path, ' object extracted')


def extract_single_object_from_both_img_mask(img, mask, img_name):
    """
    specific extraction for classification data
    """
    if not mask.any():
        return
    object_coords = find_object_coords(mask)
    object_img_cropped = img[object_coords[0]:object_coords[1], object_coords[2]:object_coords[3], :]
    object_mask_cropped = mask[object_coords[0]:object_coords[1], object_coords[2]:object_coords[3]]
    output_img_path = params.output_img_extraction + '/' + img_name + '.png'
    try:
        final_output_img_cropped = cv2.bitwise_and(object_img_cropped, object_img_cropped, mask=object_mask_cropped)
        cv2.imwrite(output_img_path, final_output_img_cropped)
        cv2.imwrite(params.output_mask_extraction + '/' + img_name + '.png', object_mask_cropped)
    except cv2.error:
        print('failed!')
        print(img_name)
        print(object_coords)
        print(cv2.error.msg)
    #print(output_img_path, ' object extracted')


def read_data(images_dir=params.input_images_dir, masks_dir=params.input_masks_dir, labels_prefix=''):
    """
    read data from path
    :param images_dir: directory of images
    :param masks_dir: directory of masks
    :param labels_prefix: specific prefix of labels - optional.
    :return: returns data frames array
    """
    data = []
    img_names = []
    for depth in range(0, params.max_depth):
        for format in params.formats:
            img_names += tf.gfile.Glob(os.path.join(images_dir + '/*' * depth, '*.{0}'.format(format)))
    for img_name in img_names:
        if img_name[0] == '.':
            mask_name = '.' + img_name.replace(images_dir, masks_dir).split('.')[1] + labels_prefix + '.png'
        else:
            mask_name = img_name.replace(images_dir, masks_dir).split('.')[0] + labels_prefix + '.png'
        data.append({'input': img_name, 'label': mask_name})
    return data


def generated_data_directories_init():
    """
    create directories func
    """
    if not os.path.exists(params.output_dir):
        os.mkdir(params.output_dir)
        os.mkdir(params.output_img_data)
        os.mkdir(params.output_mask_data)
    else:
        if not os.path.exists(params.output_img_data):
            os.mkdir(params.output_img_data)
        if not os.path.exists(params.output_mask_data):
            os.mkdir(params.output_mask_data)


def cut_roi_from_mask(mask, coords):  # crop_coords = [ymin, ymax, xmin, xmax]
    """
    extraction of roi with coords, from mask
    :param mask: input mask
    :param coords: input coords
    :return: the desired roi of the mask
    """
    return mask[coords[0]: coords[1], coords[2]: coords[3]]


def cut_roi_from_tensor(tensor, coords):  # crop_coords = [ymin, ymax, xmin, xmax]
    """
    extraction of roi with coords, from img
    :param tensor: input img
    :param coords: input coords
    :return: the desired roi of the img
    """
    return tensor[coords[0]: coords[1], coords[2]: coords[3], :]


def rotate(mat, angle):
    """
    rotation by angle func
    :param mat: matrix or tensor
    :param angle: angle of rotation
    :return: rotated matrix without information loss
    """
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def align(mask):
    """
    :param mask: segmentation mask of mole.
    :return: aligned segmentation mask of the mole.
    """
    alignment_res = mask.copy()
    best_rotation_size = mask.shape[0]
    for angle in range(0, 180):
        res = rotate(mask, angle)
        res = cut_roi_from_mask(res, find_object_coords(res))
        if best_rotation_size < res.shape[0]:
            best_rotation_size = res.shape[0]
            alignment_res = res
    for i in range(0, alignment_res.shape[1]):
        for j in range(0, alignment_res.shape[0]):
            if alignment_res[j, i].any():
                alignment_res[j, i] = 100
    return alignment_res


