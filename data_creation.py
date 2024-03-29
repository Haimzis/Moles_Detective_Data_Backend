import tensorflow as tf
from fake_data_creation import produce_fake_data
import produce_crops
from utils import read_data, generated_data_directories_init

flags = tf.app.flags
FLAGS = flags.FLAGS

# DATASET SETTINGS
flags.DEFINE_enum('dataset_activity', 'fake_data', ['training_data', 'fake_data'],
                  'which dataset creation process to perform')

# Training Data  - relevant only when dataset_activity == 'training_data'
flags.DEFINE_string('dataset_images_dir', './Data/img',
                    'where is the dataset images is found.')

flags.DEFINE_string('dataset_masks_dir', './Data/final_masks',
                    'where is the dataset masks is found.')

# Fake Data - relevant only when dataset_activity == 'fake_data'
flags.DEFINE_string('extracted_images_dir', 'Data/Input/img',
                    'where is the extracted images for transplantation is found.')

flags.DEFINE_string('extracted_masks_dir', 'Data/Input/label',
                    'where is the extracted masks for transplantation is found.')

flags.DEFINE_string('placeholders_images_dir', 'Data/fake/img',
                    'where is the placeholders images for transplantation is found.')

flags.DEFINE_string('placeholders_masks_dir', 'Data/fake/label',
                    'where is the placeholders masks for transplantation is found.')



if __name__ == '__main__':
    if FLAGS.dataset_activity == 'fake_data':
        prefix = '_segmentation'
        # prefix for the ISIC2019 seg masks.
        # (needed only when the masks aren't in the same name as the images.)
        objects_data = read_data(FLAGS.extracted_images_dir, FLAGS.extracted_masks_dir, '_segmentation')
        objects_placeholders_data = read_data(FLAGS.placeholders_images_dir, FLAGS.placeholders_masks_dir)
        produce_fake_data(objects_data, objects_placeholders_data)

    elif FLAGS.dataset_activity == 'training_data':
        generated_data_directories_init()
        data = read_data(FLAGS.dataset_images_dir, FLAGS.dataset_masks_dir)
        produce_crops.produce_crops(data)

    else:
        raise Exception('wrong activity.')
