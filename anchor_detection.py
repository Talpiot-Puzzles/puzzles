import cv2
import heapq


def _get_anchors_in_all_images(images, sift=None):
    """
    Apply sift anchor detector for all image in array images
    """
    if sift is None:
        sift = cv2.SIFT_create()

    vals = [sift.detectAndCompute(image, None) for image in images]
    return [[couple[i] for couple in vals] for i in range(2)]


def _get_all_matches(descs, flann=None):
    """
    Find all matches required between images.
    Current implementation: Find matches between 2 adjacent images (i and i + 1).
    """
    if flann is None:
        index_params = dict(algorithm=2, trees=10)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

    return [flann.knnMatch(descs[i], descs[i + 1], k=2) for i in range(len(descs) - 1)]


def _get_good_points(all_matches, ratio=0.8):
    """
    Filter only images which uphold Lowe's ratio test.
    :return: An array of all relevant matches alongside Lowe's ratio (array of couples).
    """
    return [[(m, m.distance / n.distance) for m, n in m_array if m.distance < ratio * n.distance] for m_array in
            all_matches]


def _filter_by_max_amount(arr, k=50):
    """
    Get top 10 matches for every 2 images.
    """
    return [[el[0] for el in heapq.nsmallest(k, m_array, key=lambda el: el[1])] for m_array in arr]


def _prepare_images_to_draw(kps, chosen):
    """
    Get the pixels for every match according to all the images in the match.
    """
    points_to_draw = [[] for _ in range(2 * (len(kps) - 1))]

    for i in range(0, 2 * (len(kps) - 1), 2):
        points_to_draw[i] = [kps[i // 2][el.queryIdx].pt for el in chosen[i // 2]]
        points_to_draw[i + 1] = [kps[i // 2 + 1][el.trainIdx].pt for el in chosen[i // 2]]

    return points_to_draw


def detect_anchors(images):
    kps, descs = _get_anchors_in_all_images(images)
    print(1)
    matches = _get_all_matches(descs)
    print(2)
    good_points = _get_good_points(matches)
    print(3)
    good_points = _filter_by_max_amount(good_points)
    print(4)
    pixels = _prepare_images_to_draw(kps, good_points)
    print(5)

    return pixels


if __name__ == '__main__':
    paths = ['imgs/1.jpg', 'imgs/2.jpg', 'imgs/3.jpg']
    images = [cv2.imread(path)[..., ::-1] for path in paths]
    res = detect_anchors(images)