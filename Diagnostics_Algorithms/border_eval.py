import cv2
import numpy as np
import utils


def eval_border_irregularities(border_irregularities_number):
    return min((border_irregularities_number / 100)**1.5, 1.0)


def border_eval(aligned_mask):
    # find 4 edge of the lesion from the center
    y1, x1 = find_quarter_coords(aligned_mask, 1, -1)
    y2, x2 = find_quarter_coords(aligned_mask, -1, -1)
    y3, x3 = find_quarter_coords(aligned_mask, -1, 1)
    y4, x4 = find_quarter_coords(aligned_mask, 1, 1)

    # partition to 4 graphs images
    graph1_img = aligned_mask[y1:y4, min(x1, x4): aligned_mask.shape[1] - 1]  # right
    graph2_img = aligned_mask[0: max(y1, y2), x2: x1]  # up
    graph3_img = aligned_mask[y2:y3, 0: min(x2, x3)]  # left
    graph4_img = aligned_mask[min(y3, y4): aligned_mask.shape[0] - 1, x3: x4]  # bottom

    # rotation img to look as graph
    graph1_img = utils.rotate(graph1_img, 90)
    graph3_img = utils.rotate(graph3_img, 270)
    graph4_img = utils.rotate(graph4_img, 180)

    # smoothing
    filter_size = (3, 3)
    graph1_img = cv2.GaussianBlur(graph1_img, filter_size, 0)
    graph2_img = cv2.GaussianBlur(graph2_img, filter_size, 0)
    graph3_img = cv2.GaussianBlur(graph3_img, filter_size, 0)
    graph4_img = cv2.GaussianBlur(graph4_img, filter_size, 0)

    full_graph = []
    border_irregularities_number = 0
    for graph in [graph1_img, graph2_img, graph3_img, graph4_img]:
        for i in range(0, graph.shape[1]):
            value = np.count_nonzero(graph[:, i])
            full_graph.append(value)
        full_graph.append(-1)

    for i in range(1, len(full_graph) - 1):
        if full_graph[i] == -1 \
                or full_graph[i - 1] == -1 \
                or full_graph[i + 1] == -1:
            continue
        else:
            if full_graph[i-1] < full_graph[i] < full_graph[i+1]:
                border_irregularities_number += 1
    if border_irregularities_number == 0:
        print('something wrong')
        return None
    return eval_border_irregularities(border_irregularities_number)


def find_quarter_coords(aligned_mask, x_dir, y_dir):
    x_dir_steps = x_dir
    y_dir_steps = y_dir
    width_center = aligned_mask.shape[1] // 2
    height_center = aligned_mask.shape[0] // 2
    if not aligned_mask[height_center, width_center].any():
        print('something went wrong')
        return None
    if not aligned_mask.any():
        print('mask empty')
        return None
    while aligned_mask[height_center + y_dir_steps, width_center + x_dir_steps].any():
        x_dir_steps += x_dir
        y_dir_steps += y_dir
    return [height_center + y_dir_steps, width_center + x_dir_steps]


if __name__ == '__main__':
    mask = cv2.imread('/home/haimzis/Downloads/masks(1)/masks/7.png', -1)
    aligned_mask = utils.align(mask)
    print(border_eval(aligned_mask))
