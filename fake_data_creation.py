import params
import numpy as np
import cv2
from produce_crops import generate_crops_from_image
import utils


def create_artifact_name(artifacts_amount, transplanted_data_list):
    """
    :param artifacts_amount: counter of created fake data until now.
    :param transplanted_data_list: transplanted data that used.
    :return: name of the generated data.
    """
    name = 'fk_{0}'.format(artifacts_amount)
    for data_name in transplanted_data_list:
        name += '_{0}'.format(data_name)
    name += '_'
    return name


def produce_fake_data(transplant_data, placeholders_data, amount=params.fake_data_artifacts_amount):
    """
    :param transplant_data: data for transplantation, segmented images with masks.
    :param placeholders_data: data to transplant into it.
    :param amount: desired amount of fake data images.
    :return: None
    """
    artifacts_amount = 0
    utils.generated_data_directories_init()
    while artifacts_amount < amount:
        for container in placeholders_data:
            if artifacts_amount >= amount:
                break
            container_mask = cv2.imread(container['label'], -1)
            if not container_mask.any():
                continue
            container_img = cv2.imread(container['input'], -1)
            filled_placeholder_img, filled_placeholder_mask, transplanted_data_list = fill_container(container_img,
                                                                                                     container_mask,
                                                                                                     transplant_data)
            if filled_placeholder_mask.any():
                generate_crops_from_image(create_artifact_name(artifacts_amount, transplanted_data_list),
                                          filled_placeholder_img, filled_placeholder_mask)

            print('fk_{0} created'.format(artifacts_amount))
            artifacts_amount += 1


def is_transplantation_coords_are_valid(transplanted_object_mask, container_mask,
                                        transplant_coords):  # crop_coords = [ymin, ymax, xmin, xmax]
    temp_obj_mask = np.zeros(tuple(container_mask.shape))
    temp_obj_mask[transplant_coords[0]: transplant_coords[1],
    transplant_coords[2]: transplant_coords[3]] = transplanted_object_mask
    temp_container_mask = (container_mask // np.max(container_mask)).astype('uint8') * 255
    temp_container_mask = cv2.bitwise_not(temp_container_mask)
    temp_obj_mask = cv2.bitwise_and(temp_obj_mask, temp_obj_mask, mask=temp_container_mask)
    if not temp_obj_mask.any():
        return True
    return False


def opening_holes_in_img(transplanted_object_mask, iter, init_kernel, operator='open'):
    if operator == 'open':
        operator = cv2.MORPH_OPEN
    elif operator == 'close':
        operator = cv2.MORPH_CLOSE
    else:
        return None
    op = transplanted_object_mask
    for _ in range(iter):
        kernel_size = init_kernel[0] + 1
        init_kernel = (kernel_size, kernel_size)
        kernel = np.ones(init_kernel, np.uint8)
        op = cv2.morphologyEx(op, operator, kernel)
    return op


def fix_object_crop(transplanted_object_img, transplanted_object_mask,
                    color_elimination=params.color_elimination_const):
    gray_scale_object_img = cv2.cvtColor(transplanted_object_img.astype('float32'), cv2.COLOR_RGB2GRAY).astype('uint8')
    color_high = np.array(np.max(gray_scale_object_img))
    color_low = np.array(np.subtract(np.max(gray_scale_object_img), color_elimination), dtype=np.uint8)
    skin_mask = cv2.inRange(gray_scale_object_img, color_low, color_high)
    com_skin_transplanted_object_mask = cv2.bitwise_not(skin_mask)
    transplanted_object_mask = cv2.bitwise_and(transplanted_object_mask, transplanted_object_mask,
                                               mask=com_skin_transplanted_object_mask)

    # fix single pixels
    transplanted_object_mask_opening = opening_holes_in_img(transplanted_object_mask, iter=15, init_kernel=(2, 2),
                                                            operator='open')
    transplanted_object_mask_closing = opening_holes_in_img(transplanted_object_mask, iter=7, init_kernel=(2, 2),
                                                            operator='close')
    transplanted_object_mask = cv2.bitwise_and(transplanted_object_mask, transplanted_object_mask_opening)
    transplanted_object_mask = cv2.bitwise_or(transplanted_object_mask, transplanted_object_mask_closing,
                                              mask=transplanted_object_mask_opening)
    transplanted_object_mask = opening_holes_in_img(transplanted_object_mask, iter=2, init_kernel=(2, 2),
                                                    operator='close')

    transplanted_object_img = cv2.bitwise_and(transplanted_object_img, transplanted_object_img,
                                              mask=transplanted_object_mask)
    return transplanted_object_img, transplanted_object_mask


def pixels_checksum(img, mask):
    return abs(
        cv2.countNonZero((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) + mask // 255) // 2) - cv2.countNonZero(mask)) < 10


def transplant_object_in_filter(container_img, container_mask, transplanted_data,
                                filter_coords, container_updated_mask):
    transplanted_data_names_list = []
    container_updated_img = np.copy(container_img)

    for transplant_object in get_objects_for_transplant(transplanted_data):
        transplant_coords = get_transplantation_coords(filter_coords)
        transplanted_object_img, transplanted_object_mask, objects_names_list = \
            load_transplant_object_datadict(transplant_coords, transplant_object)

        #  remove noises and irrelevant features from the object crop
        # TODO: improve the fix.
        # transplanted_object_img, transplanted_object_mask = fix_object_crop(transplanted_object_img,
        #                                                                     transplanted_object_mask)
        # transplanted_object_img, transplanted_object_mask = \
        #     fix_uncommon_pixels_between_mask_and_img(transplanted_object_img, transplanted_object_mask)
        # assert pixels_checksum(img=transplanted_object_img, mask=transplanted_object_mask)
        if is_transplantation_coords_are_valid(transplanted_object_mask, container_mask, transplant_coords):
            transplanted_data_names_list += objects_names_list
            container_updated_img, container_updated_mask = params.image_blending_func(transplanted_object_img,
                                                                                       transplanted_object_mask,
                                                                                       container_updated_img,
                                                                                       container_updated_mask,
                                                                                       transplant_coords)

    return container_updated_img, container_updated_mask, transplanted_data_names_list


def load_transplant_object_datadict(transplant_coords, transplant_object):
    transplanted_data_names_list = []
    #  get the datadict image paths
    transplanted_object_img_path = transplant_object['input']
    transplanted_object_mask_path = transplant_object['label']
    #  open object images
    transplanted_data_names_list.append(transplanted_object_img_path.split('/')[-1].split('.')[0])
    transplanted_object_img = cv2.imread(transplanted_object_img_path, -1)
    transplanted_object_mask = cv2.imread(transplanted_object_mask_path, -1)
    #  resize images of the object
    transplanted_object_img = cv2.resize(transplanted_object_img, get_image_size_by_coords(transplant_coords))
    transplanted_object_mask = cv2.resize(transplanted_object_mask, get_image_size_by_coords(transplant_coords),
                                          cv2.INTER_NEAREST)
    return transplanted_object_img, transplanted_object_mask, transplanted_data_names_list


def get_image_size_by_coords(coords):  # crop_coords = [ymin, ymax, xmin, xmax]
    return coords[3] - coords[2], coords[1] - coords[0]


def get_transplantation_coords(filter_coords):
    transplanted_object_size = (np.random.randint(int(params.crop_size[0] * 0.5), params.crop_size[0], 1))[0]
    xmin = np.random.randint(filter_coords[2], filter_coords[3] - transplanted_object_size)
    ymin = np.random.randint(filter_coords[0], filter_coords[1] - transplanted_object_size)
    xmax = xmin + transplanted_object_size
    ymax = ymin + transplanted_object_size
    transplant_coords = [ymin, ymax, xmin, xmax]
    return transplant_coords


def get_objects_for_transplant(transplanted_data):
    appearance_amount = [1]  # ready for addition
    rand_index = np.random.choice(len(appearance_amount), 1, p=[1.0])[0]  # 0.61, 0.23, 0.095, 0.05, 0.01, 0.005])[0]
    rand_indexes = np.random.choice(len(transplanted_data), appearance_amount[rand_index])
    chosen_objects_to_transplant = np.array(transplanted_data)[rand_indexes]
    return chosen_objects_to_transplant


def fill_container(placeholder_img, placeholder_mask,
                   transplant_data,
                   filter_size=params.filter_size,
                   stride=params.filter_size):
    """
    :param placeholder_img:
    :param placeholder_mask:
    :param transplant_data:
    :param filter_size:
    :param stride:
    :return:
    """
    container_updated_mask = np.zeros(tuple(placeholder_mask.shape), dtype=np.uint8)
    transplanted_data_list = []

    for filter_coords in next_filter_coords(placeholder_img, filter_size, stride):
        #  transplant object image into iterated filter coords. (in both container img & mask)
        placeholder_img, container_updated_mask, transplanted_data_names_list = \
            transplant_object_in_filter(placeholder_img, placeholder_mask, transplant_data,
                                        filter_coords, container_updated_mask)

        #  collect the transplanted objects images names.
        transplanted_data_list += transplanted_data_names_list

    return placeholder_img, container_updated_mask, transplanted_data_list


def next_filter_coords(placeholder_img, filter_size, stride):
    """
    :param placeholder_img: input image for creation of filters.
    :param filter_size: output size of filter.
    :param stride: pixels strides between filters.
    :return: filter coords for transplantation.
    """
    stride = min(stride, placeholder_img.shape[0], placeholder_img.shape[1])
    filter_coords = [0, min(filter_size, placeholder_img.shape[0]), 0, min(filter_size, placeholder_img.shape[1])]

    while filter_coords[1] <= placeholder_img.shape[0]:
        while filter_coords[3] <= placeholder_img.shape[1]:
            yield filter_coords
            filter_coords[2] += stride
            filter_coords[3] += stride
        filter_coords[0] += stride
        filter_coords[1] += stride
        filter_coords[2] = 0
        filter_coords[3] = min(filter_size, placeholder_img.shape[1])


if __name__ == '__main__':
    objects_data = utils.read_data(params.output_img_extraction, params.output_mask_extraction, '_segmentation')
    objects_placeholders_data = utils.read_data(params.object_placeholder_imgs_dir, params.object_placeholder_masks_dir)
    produce_fake_data(objects_data, objects_placeholders_data)
