## Moles Detective: Data Augmentations & preprocessing
This is the Data Backend of MolesDetective Project,  
In this part of the project we created augmentation and image preprocess flow  
that will makes more learning samples, and will make our small datasets to be more balanced.  

###**Usage**:
1. Download the source code from github
2. In data_creation.py script, chose between the enums ['training_data', 'fake_data'],
   set input and output directories paths.
3. modify params.py as you desire (augmentation activation + paths setting).
4. execute data_creation.py.
5. all artifacts, masks and images after flipping/rotation and other augmentations
   will be generated in output_dir path.

**Augmentations & Preprocesses**:
- Flip
- Rotation
- Fake Image Creation
- Random Resize and Crop
- Crop
- Balanced labels artifacts augmentation
- colors threshold
- image blending(binary wise-operation + fixing boundaries, laplacian)
- hair removal

**Example**:  
Real Data Augmentation: (without image blending on synthetic data)

| Preprocess | `Image` |  `Mask` |
| :---: | :---: | :---: |
| Original | ![](Data/Input/go_63.jpg)| ![](Data/Input/go_63.png) |
| Random Crop & Flip |  ![](Output/RandomCropFlipped/go_63_no_dup__v_h_vhcor_0_199.jpg) | ![](Output/RandomCropFlipped/go_63_no_dup__v_h_vhcor_0_199.png) |
|  Equalize Histogram & Greyscale |  ![](Output/Guassian_grey/go_63_no_dup__v_hcor_0_199.jpg) | ![](Output/Guassian_grey/go_63_no_dup__v_hcor_0_199.png) |
|  Hair Removal | ![](Output/without_hair/go_63_no_dup__v_h_vhcor_0_199.jpg) | ![](Output/without_hair/go_63_no_dup__v_h_vhcor_0_199.png) |

Fake Data Augmentation:  
(image blending between Object image + mask into Template Object + Mask)

| Fake Augmentation | `Image` |  `Mask` |
| :---: | :---: | :---: |
| Object To Transplant | ![](Data/Input/go_63.jpg)| ![](Data/Input/go_63.png) |
| Placeholder Template | ![](Data/fake/fgo_num26.jpg)| ![](Data/fake/fgo_num26.png) |
| Augmented Result| ![](Output/fake/fk_1_go_63_go_63_go_63_go_63_go_63_go_63___vcor_282_631.jpg)| ![](Output/fake/fk_1_go_63_go_63_go_63_go_63_go_63_go_63___vcor_282_631.png) |

As you can see, the object transplanted with a chosen technique of  
image blending into a randomly chosen area in the template image.

