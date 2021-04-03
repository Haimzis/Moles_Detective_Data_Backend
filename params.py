crop_size = (200, 200)
input_images_dir = './Data/img'
input_masks_dir = './Data/final_masks'
max_depth = 4
crops_per_image = 15
max_shift_for_fix = 20
min_shift_for_fix = 3
formats = ['png', 'jpeg', 'jpg']
output_dir = './Output/generated_training_data'
output_img_data = './Output/generated_training_data/images'
output_mask_data = './Output/generated_training_data/annotations'
output_img_extraction = 'Output/objects_extraction/classification_purpose/transparent'
output_mask_extraction = 'Output/objects_extraction/classification_purpose/annotations'
fake_data_artifacts_amount = 1200
object_placeholder_imgs_dir = 'Data/fake_data/img'
object_placeholder_masks_dir = 'Data/fake_data/masks'
# object_img_dir = './Output/objects_extraction/images'
object_img_dir = 'Output/objects_extraction/classification_purpose/transparent'
object_mask_dir = 'Output/objects_extraction/classification_purpose/annotations'
filter_size = 300
background_prob = 0.00
color_elimination_const = 20
HZ_preprocess = False
hair_removal = False
laplacian_pyramid_image_blending = False
laplacian_pyramid_level = 2

### pb inference ###

### COMMON ###
INPUT_SIZE = 250
INPUT_CHANNELS = 3

### INFERENCE MODEL ###
HZ_preprocess_activate = False
image_preprocess_func = HZ_preprocess
SEC_BETWEEN_PREDICTION = 1.5

### EXPORT MODEL ###
EXPORT_MODEL_PATH = './deployed/MobileNet_V3_large_ISIC_ver1.pb'
CHECKPOINT_DIR = './train_log/30_07_2020_00_40_45'

### EXPORT TFLITE ###
graph_def_file = './deployed/MobileNet_V3_large_ver2.pb'
input_arrays = 'ImageTensor'
output_arrays = 'SemanticPredictions'
input_shape = '1,{0},{0},{1}'.format(INPUT_SIZE, INPUT_CHANNELS)
inference_input_type = 'QUANTIZED_UINT8'
inference_type = 'FLOAT'
mean_values = 128
std_dev_values = 127


from image_blending import laplacian_pyramid_image_blending, my_image_blending

image_blending_func = my_image_blending
