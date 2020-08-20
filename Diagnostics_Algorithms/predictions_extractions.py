import cv2
import numpy as np
import utils


def separate_objects_from_mask(mask):  # [ymin, ymax, xmin, xmax]
    coords = utils.find_object_coords(mask)
    separated_objects_coords_col = separate_object_anchor(mask, coords, separation_op='col')
    separated_objects_coords = []
    for separated_object_coords_col in separated_objects_coords_col:
        row_separation_res = separate_object_anchor(mask, separated_object_coords_col, separation_op='row')
        separated_objects_coords += row_separation_res
    separated_objects_masks = [utils.cut_roi_from_mask(mask, coords) for coords in separated_objects_coords]
    return separated_objects_masks


def separate_object_anchor(mask, coords, res=None, separation_op='col'):
    if res is None:
        res = []
    temp_coords = coords.copy()
    if separation_op == 'col':
        while mask[:, temp_coords[3]].any():
            if temp_coords[3] == temp_coords[2]:
                res.append(coords)
                return res
            temp_coords[3] -= 1
        anchor1 = [coords[0], coords[1], temp_coords[3], coords[3]]
        anchor2 = [coords[0], coords[1], coords[2], temp_coords[3]]
    else:
        while mask[temp_coords[1], :].any():
            if temp_coords[1] == temp_coords[0]:
                res.append(coords)
                return res
            temp_coords[1] -= 1
        anchor1 = [temp_coords[1], coords[1], coords[2], coords[3]]
        anchor2 = [coords[0], temp_coords[1], coords[2], coords[3]]

    res.append(utils.find_object_coords(mask, anchor1))
    anchor2 = utils.find_object_coords(mask, anchor2)
    separate_object_anchor(mask, anchor2, res)
    return res


if __name__ == '__main__':
    example = cv2.imread('/home/haimzis/Downloads/masks(1)/masks/5.png', -1)
    separated_masks = separate_objects_from_mask(example)
    for index, separated_mask in enumerate(separated_masks):
        cv2.imwrite('/home/haimzis/Downloads/separated_mask' + str(index) + '.png', separated_mask)
    print('Done!')
