import params
import fake_data_creation
import produce_crops
from utils import read_data


def data_crop_preprocess():
    data = read_data()
    produce_crops.produce_crops(data)


def fake_data_generate_preprocess():
    objects_data = read_data(params.output_img_extraction, params.output_mask_extraction, '_segmentation')
    objects_placeholders_data = read_data(params.object_placeholder_imgs_dir, params.object_placeholder_masks_dir)
    fake_data_creation.produce_fake_data(objects_data, objects_placeholders_data)


if __name__ == '__main__':
    data_crop_preprocess()
