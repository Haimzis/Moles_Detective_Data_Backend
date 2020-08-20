import glob
import os
import cv2
import params
import numpy as np
import tensorflow as tf
import image_preprocess
import utils

labels_stats = {0: 0, 1: 0}


# TODO: 1. encapsulate the flip and the rotation
#      2. fix the save for small crops

def produce_crops(data):
    for data_dict in data:
        img_path = data_dict['input']
        mask_path = data_dict['label']
        img = cv2.imread(img_path, -1)
        mask = cv2.imread(mask_path, -1)
        name = img_path.split('/')[-1].split('.')[0]
        generate_crops_from_image(name, img, mask)
        print(img_path)
    print("crops preprocess done!")
    for label in labels_stats:
        print('label {0}: {1} samples.'.format(label, labels_stats[label]))


def save_crop(img_crop, mask_crop, name, crop_coords):
    """
    saves crop after different preprocesses activation.
    :param img_crop: input image.
    :param mask_crop: input mask.
    :param name: name of the output crop.
    :param crop_coords: coordinates of the crop.
    :return: None
    """
    crop_versions = [(img_crop, mask_crop, name)]

    # rotation
    while len(crop_versions) != 0:
        rotated_crops = []
        img, mask, name = crop_versions.pop()
        center = (mask.shape[1] / 2, mask.shape[0] / 2)
        for angle, rot_str in {0: '', 90: '_90d'}.items():  # , 180: '_180d', 270: '_270d'}.items():
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_result = cv2.warpAffine(img_crop, M, (mask_crop.shape[1], mask_crop.shape[0]))
            mask_result = cv2.warpAffine(mask_crop, M, (mask_crop.shape[1], mask_crop.shape[0]))
            new_name = name + '_' + rot_str
            rotated_crops.append((img_result, mask_result, new_name))
    crop_versions += rotated_crops

    # flip
    while len(crop_versions) != 0:
        flipped_crops = []
        img, mask, name = crop_versions.pop()
        for flip, f_name in {0: 'v', 2: '', 1: 'h', -1: 'vh'}.items():
            if flip <= 1:
                img = cv2.flip(img, flip)
                mask = cv2.flip(mask, flip)
                name = name + '_' + f_name
            flipped_crops.append((img, mask, name))
    crop_versions += flipped_crops

    # preprocess algorithms
    # if params.hair_removal and 'ha' in new_name:
    #     img_crop = image_preprocess.hair_removal(img_crop)
    # if params.HZ_preprocess:
    #     img_result = image_preprocess.HZ_preprocess(img_result)
    for img, mask, name in crop_versions:
        final_name = name + 'cor_' + str(crop_coords[0]) + '_' + str(crop_coords[1])
        cv2.imwrite(params.output_img_data + '/' + final_name + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(params.output_mask_data + '/' + final_name + '.png', mask)


def generate_crops_from_image(name, img, mask):  # crop_coords = [ymin, ymax, xmin, xmax]
    """
    :param name: name of the image
    :param img: input image
    :param mask: input mask
    :return: None
    """
    crop_size = np.array(params.crop_size)
    if img.shape[0] <= crop_size[0] * 2 or img.shape[1] <= crop_size[1] * 2:
        #  resize to the crop size and save
        img, mask = resize_to_desired_crop_size(img, mask)
        save_crop(img, mask, name + '_no_dup', np.array([0, params.crop_size[1] - 1, 0, params.crop_size[0] - 1]))

    else:
        # produce crops
        crops_per_image = get_reasonable_crops_amount_per_image(crop_size[0])
        crop_sizes_samples = generate_crop_sizes_samples(crop_size, crops_per_image)
        crop_coords = generate_random_crops_coordinates(img.shape, crop_sizes_samples)

        # ensure that the crops sizes are correct by coordinates calculation.
        for i, coords in enumerate(crop_coords):
            try:
                if coords[1] - coords[0] == crop_sizes_samples[i][1] and \
                        coords[3] - coords[2] == crop_sizes_samples[i][0]:
                    raise Exception('crop size is not right')

                if coords[0] > 0 and coords[2] > 0 and \
                        coords[1] < img.shape[0] and coords[3] < img.shape[1]:
                    raise Exception('there is a deviation')

            except Exception as e:
                print('there was an error: {error} . throwed by {img_name}, problematic coords: {coords},'
                      ' image shape: {img_shape} wanted crop size is : {crop_size} '
                      .format(error=str(e), img_name=name, coords=coords, img_shape=img.shape,
                              crop_size=crop_sizes_samples[i]))
                exit(1)

        # generate crops for each given coordinates.
        for i in range(0, crops_per_image):
            mask_crop = utils.cut_roi_from_mask(mask, crop_coords[i])

            if mask_crop.any() == 0:  # mask crop doesnt contain any label
                if np.random.random_sample() <= params.background_prob:
                    img_crop = utils.cut_roi_from_tensor(img, crop_coords[i])
                    img_crop, mask_crop = resize_to_desired_crop_size(img_crop, mask_crop)
                    save_crop(img_crop, mask_crop, name, crop_coords[i])
                    labels_stats[0] += 1

            else:  # mask crop contain some label
                valid_crop_coords = get_valid_coords_for_label_crop(mask, crop_coords[i])
                img_crop = utils.cut_roi_from_tensor(img, valid_crop_coords)
                mask_crop = utils.cut_roi_from_mask(mask, valid_crop_coords)
                img_crop, mask_crop = resize_to_desired_crop_size(img_crop, mask_crop)
                save_crop(img_crop, mask_crop, name, valid_crop_coords)
                labels_stats[1] += 1


def resize_to_desired_crop_size(img, mask):
    """
    :param img: input image.
    :param mask: input mask.
    :return: resized image and mask.
    """
    img = cv2.resize(img, (params.crop_size[0], params.crop_size[1]))
    mask = cv2.resize(mask, (params.crop_size[0], params.crop_size[1]), interpolation=cv2.INTER_NEAREST)
    return img, mask


def label_on_mask_border(mask):
    """
    :param mask: input mask
    :return: True, if there is a label on some border of the mask.
    """
    return mask[:, [0, params.crop_size[0] - 1]].any() or mask[[0, params.crop_size[1] - 1], :].any()


def generate_crop_sizes_samples(crop_size, how_many_crops):
    """
    :param crop_size: the desired crop size after resize.
    :param how_many_crops: how many crops sizes to generate.
    :return: np array of 'how_many_crops' - crops sizes (before the final resize)
    """
    xy = np.array([tuple(crop_size)])
    random_indices = np.random.choice(len(xy), how_many_crops, p=[1.0])
    random_crops_sizes_additions = np.random.randint(0, crop_size - 1, how_many_crops)
    return xy[random_indices] + np.dstack([random_crops_sizes_additions, random_crops_sizes_additions])[0]


def generate_random_crops_coordinates(img_shape, crop_sizes_samples):  # crop_coords = [ymin, ymax, xmin, xmax]
    """
    :param img_shape: the image shape.
    :param crop_sizes_samples: size of each desired crop.
    :return: np array of all crops coordinates.
    """
    how_many_crops = len(crop_sizes_samples)
    ymax = [0] * how_many_crops
    xmax = [0] * how_many_crops
    ymin = [0] * how_many_crops
    xmin = [0] * how_many_crops

    for i in range(0, how_many_crops):
        ymin[i] = np.random.randint(0, img_shape[0] - crop_sizes_samples[i][1], how_many_crops)
        xmin[i] = np.random.randint(0, img_shape[1] - crop_sizes_samples[i][0], how_many_crops)

        # if the random coordinates are over the image board
        # fix the coordinates.
        ymax[i] = ymin[i] + crop_sizes_samples[i][0]
        if ymax[i] >= img_shape[0]:
            ymin[i] -= ymax[i] - img_shape[0]
            ymax[i] -= ymax[i] - img_shape[0]
        xmax[i] = xmin[i] + crop_sizes_samples[i][1]
        if xmax[i] >= img_shape[1]:
            xmin[i] -= xmax[i] - img_shape[1]
            xmax[i] -= xmax[i] - img_shape[1]
    return np.dstack([ymin, ymax, xmin, xmax])[0]


def get_reasonable_crops_amount_per_image(img_size):
    """
    :param img_size: input image size
    :return: the amount of crops to take from this specific image.
    """
    if img_size >= params.crop_size[0] * 5:
        return params.crops_per_image

    elif img_size >= params.crop_size[0] * 4:
        return max(params.crops_per_image / 2, 1)

    elif img_size >= params.crop_size[0] * 3:
        return max(params.crops_per_image / 4, 1)

    else:
        return max(params.crops_per_image / 8, 1)


def get_valid_coords_for_label_crop(mask, crop_coords):  # crop_coords = [ymin, ymax, xmin, xmax]
    """
    :param mask: input mask that contains a label.
    :param crop_coords: the coordinates of the desired crop.
    :return: fixed coordinates without a sliced label.
    """
    new_coords = crop_coords.copy()
    mask_crop = utils.cut_roi_from_mask(mask, new_coords)
    mask_crop_resized = cv2.resize(mask_crop, (params.crop_size[0], params.crop_size[1]),
                                   interpolation=cv2.INTER_NEAREST)
    tries = 0
    while label_on_mask_border(mask_crop_resized):
        if mask_crop_resized[:, 0].any():
            new_coords[2] = max(new_coords[2] - np.random.randint(params.min_shift_for_fix,
                                                                  params.max_shift_for_fix), 1)
        if mask_crop_resized[:, params.crop_size[0] - 1].any():
            new_coords[3] = min(new_coords[3] + np.random.randint(params.min_shift_for_fix,
                                                                  params.max_shift_for_fix), mask.shape[1] - 1)
        if mask_crop_resized[0, :].any():
            new_coords[0] = max(new_coords[0] - np.random.randint(params.min_shift_for_fix,
                                                                  params.max_shift_for_fix), 1)
        if mask_crop_resized[params.crop_size[1] - 1, :].any():
            new_coords[1] = min(new_coords[1] + np.random.randint(params.min_shift_for_fix,
                                                                  params.max_shift_for_fix), mask.shape[0] - 1)

        mask_crop = utils.cut_roi_from_mask(mask, new_coords)
        mask_crop_resized = cv2.resize(mask_crop, (params.crop_size[0], params.crop_size[1]),
                                       interpolation=cv2.INTER_NEAREST)
        tries += 1
        if tries >= 20:
            new_coords = crop_coords.copy()

    return new_coords
