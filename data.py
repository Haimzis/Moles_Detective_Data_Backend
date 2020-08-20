import params
import fake_data_creation
import produce_crops
from filter_irrelevant_images import filter_segmented_segmentation_data_examples, filter_segmented_classification_data_examples
from utils import read_data
import _thread


def data_crop_preprocess():
    data = read_data()
    produce_crops.produce_crops(data)


def remove_irrelevant_images():
    data = read_data(params.output_img_extraction, params.output_mask_extraction, '_segmentation')
    filter_segmented_segmentation_data_examples(data)


if __name__ == '__main__':
    filter_segmented_classification_data_examples('/media/haimzis/Extreme SSD/Moles_Detector_Dataset/Classification/ISIC_2019_Training_Input')
    #remove_irrelevant_images()
    # _thread.start_new_thread(fake_data_generate_preprocess, ())
    # _thread.start_new_thread(fake_data_generate_preprocess, ())
    # _thread.start_new_thread(fake_data_generate_preprocess, ())
    #_thread.start_new_thread(data_crop_preprocess, ())
    # while True:
    #     pass


