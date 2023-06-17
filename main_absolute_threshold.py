# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from collections import defaultdict
from functools import reduce
from itertools import product
from queue import PriorityQueue
from statistics import mean

import numpy as np
from disjoint_set import DisjointSet

import cv2
from scipy.spatial import KDTree
from PIL import Image

THRESHOLD = 190
CONTOUR_SIZE_RATIO = 0.001
FIXED_CONSIDERED_DISTANCE = 12
CONNECTIVITY_THRESHOLD = 0.1


def get_considerable_distance(contour_size, largest_contour_size):
    return FIXED_CONSIDERED_DISTANCE * (2 - contour_size / largest_contour_size)


def combine_contours(contours):
    contour_queue = PriorityQueue()
    ds = DisjointSet()
    contour_point_mapping = {}
    for index, contour in enumerate(contours):
        contour_point_mapping.update({(point[0][0], point[0][1]): index for point in contour})
        contour_queue.put((-cv2.contourArea(contour), index))
    largest_size = -contour_queue.queue[0][0]
    points = list(contour_point_mapping.keys())
    point_tree = KDTree(points)
    i = 1
    while not contour_queue.empty():
        area, index = contour_queue.get()
        area = -area
        # print(i)
        i += 1
        if area < largest_size * CONTOUR_SIZE_RATIO:
            break
        connectivity = defaultdict(lambda: 0)
        contour_points = contours[index]
        for point in contour_points:
            considerable_distance = get_considerable_distance(area, largest_size)
            neighboring_points = point_tree.query_ball_point(point, considerable_distance)[0]
            for neighbor_point_index in neighboring_points:
                contour_index = contour_point_mapping[points[neighbor_point_index]]
                connectivity[contour_index] += contour_index != index
        for neighboring_contour, neighboring_points_count in connectivity.items():
            if neighboring_points_count / len(contours[neighboring_contour]) > CONNECTIVITY_THRESHOLD:
                ds.union(index, neighboring_contour)
    return list(ds.itersets())


def calc_variance_from_lighter_neighbors(p, r, image):
    min_x, max_x = max(0, p[0] - r), min(image.shape[0] - 1, p[0] + r)
    min_y, max_y = max(0, p[1] - r), min(image.shape[1] - 1, p[1] + r)
    total, count = 0, 0
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if image[x][y] > image[p[0]][p[1]]:
                total += (image[x][y] - image[p]) ** 2
                count += 1
    return count and total / count


def calc_contour_variance_neighbors_mean(contour, r, image):
    res = [calc_variance_from_lighter_neighbors(p[0], r, image) for p in contour]
    return sum(res) / len(res)


def rate_cluster(cluster, image, contours):
    # variances = [calc_contour_variance_neighbors_mean(contours[contour_index], 5, image) for contour_index in cluster]
    # average_variance = sum(variances) / len(variances)
    return (255 - get_cluster_mean(cluster, contours, image)) ** 3 * sum(
        len(contours[contour_index]) for contour_index in cluster)  # * average_variance ** 2



def get_cluster_mean(cluster, contours, image):
    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.drawContours(mask, tuple(contours[contour_index] for contour_index in cluster), -1, 255, -1)
    return cv2.mean(image, mask=mask)[0]


def get_cluster_peek(cluster, func, index, contours):
    return func(func(point[0][index] for point in contours[contour_index]) for contour_index in cluster)


def get_cluster_bounding_rectangle(cluster, contours):
    # min_x, max_x, min_y, max_y = (get_cluster_peek(cluster, func, index, contours) for func, index in
    # product([min, max], [0, 1]))
    min_x = min(min(p[0][0] for p in contours[contour_index]) for contour_index in cluster)
    max_x = max(max(p[0][0] for p in contours[contour_index]) for contour_index in cluster)
    min_y = min(min(p[0][1] for p in contours[contour_index]) for contour_index in cluster)
    max_y = max(max(p[0][1] for p in contours[contour_index]) for contour_index in cluster)
    n = 20
    return (min_x - n, min_y - n), (max_x + n, max_y + n)


def detect(img):
    img = np.asarray(img)
    image = cv2.bitwise_not(img)
    imageb  = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 50)
    # print(np.max(image))
    # ret, image = cv2.threshold(image, THRESHOLD, 255, cv2.THRESH_TRUNC)
    # ret, imageb = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)
    #imgain = image  # cv2.imread(img_path)
    # mask = cv2.inRange(image, 140, 141)
    # image[mask > 0] = 255
    contours, hierarchy = cv2.findContours(imageb, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # imgain = cv2.drawContours(imgain, contours, -1, (0, 255, 0), 3)
    combinos = combine_contours(contours)
    cluster = max(combinos, key=lambda clus: rate_cluster(clus, image, contours))
    top_left_corner, bottom_right_corner = get_cluster_bounding_rectangle(cluster, contours)
    imgain = cv2.rectangle(img, top_left_corner, bottom_right_corner, (0, 255, 0), 2)
    imdraw = imgain
    #imdraw = cv2.resize(imdraw, (1200, 700))
    #imdraw = cv2.bitwise_not(imdraw)

    #cv2.imshow('uh oh', imdraw)
    #Image.fromarray(imdraw).save(rf"uh_oh.jpg")
    cv2.waitKey()
    color_converted = cv2.cvtColor(imdraw, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    return pil_image
    # pil_image.show()
    # pil_image.save("un_proccessed_detected.jpg")
    # print(len(contours))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # detect(r'results/egoz_461/res0.jpg')
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
