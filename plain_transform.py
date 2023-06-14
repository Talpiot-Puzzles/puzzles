import cv2
import numpy as np
import matplotlib.pyplot as plt

DEGREES_IN_CIRCLE = 360

def calculate_descriptors(img):
    """
    Calculate descriptors for the given image
    :param img: the image
    :return: the descriptors
    """
    algorithm = cv2.KAZE_create()
    kps, dsc = algorithm.detectAndCompute(img, None)

    return kps, dsc


def match_extraction(img1, img2):
    kps1, dsc1 = calculate_descriptors(img1)
    kps2, dsc2 = calculate_descriptors(img2)

    # convert the descriptors to uint8
    dsc1 *= 255
    dsc2 *= 255
    dsc1 = dsc1.astype("uint8")
    dsc2 = dsc2.astype("uint8")

    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = brute_force.match(dsc1, dsc2)

    return kps1, kps2, dsc1, dsc2, matches


def find_homography(src_pts, dst_pts):
    homograph_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homograph_matrix

def extract_movement_from_homography(homography):
    # print(homography)
    # Extract the rotation matrix and translation vector from the homography matrix
    R = homography[:, :2]
    t = homography[:2, 2]
    # Compute the rotation angle and translation distance
    theta = -np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

    return (t[0], t[1], theta)

def project_image(src, dst, homograph_matrix):
    height = src.shape[1] + dst.shape[1]
    width = src.shape[0] + dst.shape[0]

    plane = np.zeros((height, width))
    plane[0: dst.shape[0], 0: dst.shape[1]] = dst

    wrap_src = cv2.warpPerspective(src, homograph_matrix, (width, height))
    plt.imshow(wrap_src, cmap="gray")
    plt.show()
    plane = np.where(wrap_src > 0, wrap_src, plane)

    plt.imshow(plane, cmap="gray")
    plt.show()
    return plane


def load_images(path):
    # load the images using cv2
    img1 = cv2.imread(path + "1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path + "2.jpg", cv2.IMREAD_GRAYSCALE)

    return img1, img2


def load_and_show_images(path):
    # load the images using cv2
    img1 = cv2.imread(path + "1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path + "2.jpg", cv2.IMREAD_GRAYSCALE)

    plt.imshow(img1, cmap="gray")
    plt.show()
    plt.imshow(img2, cmap="gray")
    plt.show()

    return img1, img2


def get_match_list(kps1, kps2, matches):
    img1_pts = np.array([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    img2_pts = np.array([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return img1_pts, img2_pts


def find_adjacent_image_movement(matches):
    """
    calculate distance and rotation between every two adjacent images
    :param matches: list of list of tuple with matches between every two adjacent:
                    [[[(1,1), (2,2)], [(1,1), (2,2)]], [[(3,3), (4,4)], [(3,3), (4,4)]]]
    :return: list of distance and rotation between every two images, the format is:
                [(x_d_i, y_d_i, rotation_i]
    """
    movement = []
    for matches_pair in matches:
        # calculate the homographs using ransac algorithm
        img1_pts = np.array([[pt[0], pt[1]] for pt in matches_pair[0]]).reshape(-1, 1, 2)
        img2_pts = np.array([[pt[0], pt[1]] for pt in matches_pair[1]]).reshape(-1, 1, 2)

        homography_matrix = find_homography(img2_pts, img1_pts)
        movement.append(extract_movement_from_homography(homography_matrix))

    return movement


def find_relative_movement(adjacent_movement):
    """
    :param adjacent_movement: list of distance and rotation between every two images, the format is:
                            [(x_d_i, y_d_i, rotation_i]
    :return: list of distance and rotation between every image to the base image, the format is:
                            [(x_d_i, y_d_i, rotation_i]
    """
    adjacent_movement = np.array(adjacent_movement)
    relative_movement = np.cumsum(adjacent_movement, axis=0)
    # relative_movement[2, :] = relative_movement[2, :] % DEGREES_IN_CIRCLE
    relative_movement[:, 2] = relative_movement[:, 2] % DEGREES_IN_CIRCLE

    return relative_movement

def plain_transform(mathces):
    """
    calculate distance and rotation between every two adjacent images
    :param matches: list of list of tuple with matches between every two adjacent:
                    [[[(1,1), (2,2)], [(1,1), (2,2)]], [[(3,3), (4,4)], [(3,3), (4,4)]]]
    :return: list of distance and rotation between every image to the base image, the format is:
                            [(x_d_i, y_d_i, rotation_i]
    """
    return find_relative_movement(find_adjacent_image_movement(mathces))


def main():
    # load the images using cv2
    img1 = cv2.imread("database/halva/pic1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("database/halva/pic2.jpg", cv2.IMREAD_GRAYSCALE)

    kps1, kps2, dsc1, dsc2, matches = match_extraction(img1, img2)
    # calculate the homographs using ransac algorithm
    img1_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2).astype(int)

    img2_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2).astype(int)
    homograph_matrix, mask = cv2.findHomography(img2_pts, img1_pts, cv2.RANSAC, 5.0)

    height = img1.shape[1] + img2.shape[1]
    width = img1.shape[0] + img2.shape[0]

    plane = np.zeros((height, width))
    plane[0 : img1.shape[0], 0 : img1.shape[1]] = img1

    plt.figure()
    plt.imshow(img1, cmap="gray")
    plt.show()
    plt.imshow(img2, cmap="gray")
    plt.show()


    result = cv2.warpPerspective(img2, homograph_matrix, (width, height))
    plt.imshow(result, cmap="gray")
    plt.show()
    plane = np.where(result > 0, result, plane)

    plt.imshow(plane, cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
