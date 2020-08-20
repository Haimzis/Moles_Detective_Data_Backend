import cv2
import params
import numpy as np
import image_preprocess
import utils

labels_stats = {0: 0, 1: 0}


# TODO: 1. encapsulate the flip and the rotation
#      2. fix the save for small crops

def produce_crops(data):
    for data_dict in data:
        img_path = data_dict['input']
        mask_path = data_dict['label']
        for flip, f_name in {0: 'v', 2: '', 1: 'h', -1: 'vh'}.items():
            img = cv2.imread(img_path, -1)
            mask = cv2.imread(mask_path, -1)
            if flip <= 1:
                img = cv2.flip(img, flip)
                mask = cv2.flip(mask, flip)
            name = img_path.split('/')[-1].split('.')[0] + '_' + f_name
            generate_crops_from_image(name, img, mask)
        print(img_path)
    print("crops preprocess done!")
    for label in labels_stats:
        print('label {0}: {1} samples.'.format(label, labels_stats[label]))


def save_crop(img_crop, mask_crop, name, crop_coords):
    center = (mask_crop.shape[1] / 2, mask_crop.shape[0] / 2)
    for angle, rot_str in {0: '', 90: '_90d'}.items():  # , 180: '_180d', 270: '_270d'}.items():
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_result = cv2.warpAffine(img_crop, M, (mask_crop.shape[1], mask_crop.shape[0]))
        mask_result = cv2.warpAffine(mask_crop, M, (mask_crop.shape[1], mask_crop.shape[0]))
        new_name = '/' + name + 'c_' + str(crop_coords[0]) + '_' + str(crop_coords[1]) + rot_str

        if params.hair_removal and 'ha' in new_name:
            img_crop = image_preprocess.hair_removal(img_crop)
        if params.HZ_preprocess:
            img_result = image_preprocess.HZ_preprocess(img_result)
        cv2.imwrite(params.output_img_data + new_name + '.jpg', img_result, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(params.output_mask_data + new_name + '.png', mask_result)


def generate_crops_from_image(name, img, mask):
    crop_size = np.array(params.crop_size)
    if img.shape[0] <= crop_size[0] * 4 or img.shape[1] <= crop_size[1] * 4:
        pass
        # img, mask = resize_to_crop_size(img, mask)
        # save_crop(img, mask, name + '_no_dup', np.array([0, params.crop_size[1] - 1, 0, params.crop_size[0] - 1]))
    else:
        crop_sizes_samples = generate_crop_sizes_samples(crop_size)
        crop_coords = generate_random_crops_coordinates(img.shape, crop_sizes_samples)

        for i, coords in enumerate(crop_coords):
            if coords[1] - coords[0] != crop_sizes_samples[i][1] or coords[3] - coords[2] != crop_sizes_samples[i][0]:
                print('crop size invalid, check code.')
                exit(1)
        for i in range(0, params.crops_per_image):
            mask_crop = utils.cut_roi_from_mask(mask, crop_coords[i])
            if mask_crop.any() == 0:  # Background
                if np.random.random_sample() <= params.background_prob:
                    img_crop = utils.cut_roi_from_tensor(img, crop_coords[i])
                    img_crop, mask_crop = resize_to_crop_size(img_crop, mask_crop)
                    save_crop(img_crop, mask_crop, name, crop_coords[i])
                    labels_stats[0] += 1
            else:  # Mole
                save_crop_with_object(img, mask, name, crop_coords[i])
                labels_stats[1] += 1


def resize_to_crop_size(img, mask):
    img = cv2.resize(img, (params.crop_size[0], params.crop_size[1]))
    mask = cv2.resize(mask, (params.crop_size[0], params.crop_size[1]), interpolation=cv2.INTER_NEAREST)
    return img, mask


def no_label_on_mask_border(mask):
    return not mask[:, [0, params.crop_size[0] - 1]].any() and not mask[[0, params.crop_size[1] - 1], :].any()


def generate_crop_sizes_samples(crop_size):
    xy = np.array([tuple(crop_size), tuple(crop_size * 2), tuple(crop_size * 3), tuple(crop_size * 4)])
    random_indices = np.random.choice(len(xy), params.crops_per_image, p=[0.5, 0.35, 0.1, 0.05])
    random_crops_sizes_additions = np.random.randint(0, min(crop_size) - 1, params.crops_per_image)
    return xy[random_indices] + np.dstack([random_crops_sizes_additions, random_crops_sizes_additions])[0]


def generate_random_crops_coordinates(img_shape, crop_sizes_samples):
    ymin = np.random.randint(0, img_shape[0] - params.crop_size[1], params.crops_per_image)
    xmin = np.random.randint(0, img_shape[1] - params.crop_size[0], params.crops_per_image)
    ymax = [0] * params.crops_per_image
    xmax = [0] * params.crops_per_image

    for i in range(0, params.crops_per_image):
        ymax[i] = ymin[i] + crop_sizes_samples[i][0]
        if ymax[i] >= img_shape[0]:
            ymin[i] -= ymax[i] - img_shape[0]
            ymax[i] -= ymax[i] - img_shape[0]
        xmax[i] = xmin[i] + crop_sizes_samples[i][1]
        if xmax[i] >= img_shape[1]:
            xmin[i] -= xmax[i] - img_shape[1]
            xmax[i] -= xmax[i] - img_shape[1]
    return np.dstack([ymin, ymax, xmin, xmax])[0]


def save_crop_with_object(img, mask, name, crop_coords):  # crop_coords = [ymin, ymax, xmin, xmax]
    mask_crop = utils.cut_roi_from_mask(mask, crop_coords)
    img_crop = utils.cut_roi_from_tensor(img, crop_coords)
    img_crop_resized, mask_crop_resized = resize_to_crop_size(img_crop, mask_crop)
    if no_label_on_mask_border(mask_crop_resized):
        save_crop(img_crop_resized, mask_crop_resized, name, crop_coords)
    else:
        shifted = False
        if mask_crop_resized[:, 0].any():
            if crop_coords[2] > 0:
                crop_coords[2] = max(crop_coords[2] - np.random.randint(params.min_shift_for_fix,
                                                                        params.max_shift_for_fix), 0)
                shifted = True
        if mask_crop_resized[:, params.crop_size[0] - 1].any():
            if crop_coords[3] < img.shape[1]:
                crop_coords[3] = min(crop_coords[3] + np.random.randint(params.min_shift_for_fix,
                                                                        params.max_shift_for_fix), img.shape[1])
                shifted = True
        if mask_crop_resized[0, :].any():
            if crop_coords[0] > 0:
                crop_coords[0] = max(crop_coords[0] - np.random.randint(params.min_shift_for_fix,
                                                                        params.max_shift_for_fix), 0)
                shifted = True
        if mask_crop_resized[params.crop_size[1] - 1, :].any():
            if crop_coords[1] < img.shape[0]:
                crop_coords[1] = min(crop_coords[1] + np.random.randint(params.min_shift_for_fix,
                                                                        params.max_shift_for_fix), img.shape[0])
                shifted = True
        if shifted:
            save_crop_with_object(img, mask, name, crop_coords)
