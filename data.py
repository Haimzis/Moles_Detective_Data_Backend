import params
import fake_data_creation
import produce_crops
from filter_irrelevant_images import filter_flow
from utils import read_data
import _thread


def data_crop_preprocess():
    data = read_data()
    produce_crops.produce_crops(data)


def fake_data_generate_preprocess():
    objects_data = read_data(params.output_img_extraction, params.output_mask_extraction, '_segmentation')
    objects_placeholders_data = read_data(params.object_placeholder_imgs_dir, params.object_placeholder_masks_dir)
    fake_data_creation.produce_fake_data(objects_data, objects_placeholders_data)


def remove_irrelevant_images():
    data = read_data(params.output_img_extraction, params.output_mask_extraction, '_segmentation')
    filter_flow(data)


if __name__ == '__main__':
    remove_irrelevant_images()
    # _thread.start_new_thread(fake_data_generate_preprocess, ())
    # _thread.start_new_thread(fake_data_generate_preprocess, ())
    # _thread.start_new_thread(fake_data_generate_preprocess, ())
    # _thread.start_new_thread(data_crop_preprocess, ())
    # while True:
    #     pass


