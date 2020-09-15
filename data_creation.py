import tensorflow as tf
from fake_data_creation import produce_fake_data
import produce_crops
from utils import read_data, generated_data_directories_init

flags = tf.app.flags
FLAGS = flags.FLAGS

# DATASET
flags.DEFINE_integer('image_size', 200, 'output data size (X,X)')

flags.DEFINE_enum('dataset_activity', 'training_data', ['training_data', 'fake_data'],
                  'which dataset creation process to perform')

# Training Data  - relevant only when dataset_activity == 'training_data'
flags.DEFINE_string('dataset_images_dir', './Data/img',
                    'where is the dataset images is found.')

flags.DEFINE_string('dataset_masks_dir', './Data/final_masks',
                    'where is the dataset masks is found.')

flags.DEFINE_string('output_dir', '/output/dir', 'where to export the generated data')

# Fake Data - relevant only when dataset_activity == 'fake_data'
flags.DEFINE_string('extracted_images_dir', 'Output/objects_extraction/segmentation_purpose/transparent',
                    'where is the extracted images for transplantation is found.')

flags.DEFINE_string('extracted_masks_dir', 'Output/objects_extraction/segmentation_purpose/annotations',
                    'where is the extracted masks for transplantation is found.')

flags.DEFINE_string('placeholders_images_dir', 'Data/fake_data/img',
                    'where is the placeholders images for transplantation is found.')

flags.DEFINE_string('placeholders_masks_dir', 'Data/fake_data/masks',
                    'where is the placeholders masks for transplantation is found.')

flags.DEFINE_integer('fake_artifacts_amount', 14000, 'how many fake data to generate')

# image preprocess
flags.DEFINE_enum('data_preprocess', None, ['hair_removal', 'HZ'],
                  'which preprocess to activate on generated data')

flags.DEFINE_enum('image_blending', 'regular', ['regular', 'laplacian_pyramid'],
                  'how to blend the images when create fake data')


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
