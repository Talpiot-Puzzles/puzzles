from typing import List, Tuple, Dict

import cv2
import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from tqdm import tqdm


def unite_sigmoid(x, upper_width=1, lower_width=30, meanval=127.5):
    # condition1 = x >= meanval
    # condition2 = x < meanval
    # result1 = 255 / (1 + np.exp((meanval - x) / upper_width))
    # result2 = 255 / (1 + np.exp((meanval - x) / lower_width))
    # result = np.where(condition2, result2, 0) + np.where(condition1, result1, 0)
    return x


def highlights_black_areas(image: np.ndarray, thresh_value: int) -> np.ndarray:
    """
    This function highlights the hot areas in the input image by performing a
    bitwise 'AND' operation between the thresholded image and the original image.

    :param image: A PIL Image object representing the input image
    :param thresh_value: An integer threshold value to use for binarization
    :return: A PIL Image object representing the image with hot areas highlighted
    """
    # Convert the input image to a NumPy array
    image_array = np.array(image)

    # Perform thresholding on the image to create a binary inverted threshold image
    _, thresh_img = cv2.threshold(image_array, thresh_value, 255, cv2.THRESH_BINARY_INV)
    inv_thresh_img = cv2.bitwise_and(thresh_img, image_array)

    # Convert the inverted thresholded image back to a PIL Image object
    highlighted_image = Image.fromarray(inv_thresh_img)
    return highlighted_image


def highlights_hot_areas(image: Image.Image, thresh_value: int = 150) -> Image.Image:
    """
    Highlights the white areas in the input image.

    Args:
        image (PIL.Image.Image): A PIL Image object representing the input image.
        thresh_value (int, optional): An integer threshold value to use for binarization.
                                      Pixels below this value are set to 255 (white) and pixels
                                      above are set to 0 (black). Default is 150.

    Returns:
        PIL.Image.Image: A PIL Image object representing the image with white areas highlighted.

    Raises:
        None.
    """
    print("### Highlights hot areas")
    # Convert the input image to a NumPy array
    image_array = np.array(image)

    # Perform thresholding on the image to create a binary inverted threshold image
    _, thresh_img = cv2.threshold(image_array, thresh_value, 255, cv2.THRESH_BINARY_INV)
    inv_thresh_img = cv2.bitwise_not(thresh_img)

    # Convert the inverted thresholded image back to a PIL Image object
    highlighted_image = Image.fromarray(inv_thresh_img)
    return highlighted_image


def apply_mean_filter(image: np.ndarray, overlap_size: int) -> np.ndarray:
    """
    Apply a 3x3 mean filter to the overlapping region of an image.
    :param image: Input image as a numpy array
    :param overlap_size: Size of the overlapping region
    :return: Filtered image as a numpy array
    """
    kernel = np.ones((3, 3)) / 9.0
    filtered_image = image.copy()
    filtered_image[overlap_size:-overlap_size, overlap_size:-overlap_size] = convolve2d(
        image[overlap_size:-overlap_size, overlap_size:-overlap_size], kernel, mode='same', boundary='symm'
    )
    return filtered_image.astype(np.uint8)


def load_image(path):
    # load the images using cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img


def load_images(shifted_images: List[Tuple[str, Tuple[int, int, int]]]):
    """
    Read a list of image paths and return a list of corresponding images.

    :param paths: A list of image paths
    :return: A list of images
    """
    print("### Load the image to the shifted_images ...")
    return [(load_image(path), *rest) for path, *rest in shifted_images]


def white_is_most_important(overlap_img: List[List[List[int]]]):
    # If this is the best algo could to optimize run time and cancel the overlap array
    result = np.zeros((len(overlap_img), len(overlap_img[0])), dtype=np.int32)
    for i in range(len(overlap_img)):
        for j in range(len(overlap_img[i])):
            if len(overlap_img[i][j]) == 0:
                # Not should to be when fix width calc problem
                result[i][j] = 0
            else:
                result[i][j] = max(overlap_img[i][j])
        # result.append(row_result)
    combine_img = Image.fromarray(result)
    combine_img = combine_img.convert('RGB')
    return combine_img


def get_white(pixel: list):
    thresh_factor = 150
    # Comment out for get also black pixel
    filtered = list(filter(lambda a: a > thresh_factor, pixel))
    if len(filtered) > 0:  # len(pixel):
        return list_mean(filtered)
    return list_mean(pixel)


def kernel_mean(overlap_img: List[List[List[int]]], kernel_size: int = 3):
    result = np.zeros((len(overlap_img), len(overlap_img[0])), dtype=np.int32)
    for i in tqdm(range(len(overlap_img))):
        for j in range(len(overlap_img[i])):
            kernel_sum = []
            for k in range(0, kernel_size):
                for x in range(max(0, i - k), min(len(overlap_img) - 1, i + k)):
                    for y in range(max(0, j - k), min(len(overlap_img[x]) - 1, j + k)):
                        kernel_sum.extend(overlap_img[x][y])
            result[i][j] = get_white(kernel_sum)  # list_mean(kernel_sum)
    combine_img = Image.fromarray(result)
    combine_img = combine_img.convert('RGB')
    return combine_img

def list_mean(lst: List[int]):
    if len(lst) == 0:
        return 0
    return int(np.ceil((sum(lst) / len(lst))))


def simple_mean(overlap_img):
    # overlap_img = np.array(overlap_img)
    counter = (overlap_img == 0).sum(axis=-1)
    summed = overlap_img.sum(axis=-1)
    res = summed / (overlap_img.shape[-1] - counter)

    res = np.where(res == np.inf, 0, res)

    return Image.fromarray(res).convert('RGB')


def milo_simple_mean(overlap_img):
    # alpha = 1.1
    # alpha = 1.2
    alpha = 1.15
    counter = (overlap_img == 0).sum(axis=-1)
    to_mean = np.power(overlap_img, 1 / alpha)
    summed = to_mean.sum(axis=-1)
    summed = np.power(summed, alpha)
    res = summed / (overlap_img.shape[-1] - counter)

    res = np.where(res == np.inf, 0, res)

    return Image.fromarray(res).convert('RGB')

# def milomilo_simple_mean(overlap_img):
#     alpha = 1.15
#     padded = np.pad(overlap_img, ((0, 0), (1, 1), (1, 1)))
#     overlap_img -= np.apply_along_axis(close_mean)
#     counter = (overlap_img == 0).sum(axis=-1)
#     to_mean = np.power(overlap_img, 1 / alpha)
#     summed = to_mean.sum(axis=-1)
#     summed = np.power(summed, alpha)
#     res = summed / (overlap_img.shape[-1] - counter)
#
#     res = np.where(res == np.inf, 0, res)
#
#     return Image.fromarray(res).convert('RGB')


def simple_mean2(overlap_img: List[List[List[int]]]) -> Image.Image:
    combine_mean = []

    for inner_list in tqdm(overlap_img):
        sublist_mean = []
        for values in inner_list:
            values = list(filter(lambda a: a != 0, values))
            if len(values) > 0:
                sublist_mean.append(np.ceil(list_mean(values)))
            else:
                sublist_mean.append(0)  # Handle empty sublists by adding 255
        combine_mean.append(sublist_mean)
    combine_mean = np.array(combine_mean)
    # Calculate the mean along axis 0
    # combine_mean = np.mean(combine_mean, axis=0).astype(int)
    # Convert the NumPy array to PIL Image
    combine_img = Image.fromarray(combine_mean)
    combine_img = combine_img.convert('RGB')
    return combine_img


# Preprocess functions:
def split_pixels(img: np.ndarray, split_factor: int) -> np.ndarray:
    """
    Splits each pixel in an image into smaller pixels to increase its resolution.

    Args:
        image (np.ndarray): The original image as a numpy array.
        split_factor (int): The factor by which to split each pixel.

    Returns:
        np.ndarray: The new image as a numpy array.

    Raises:
        ValueError: If the split factor is less than or equal to 0.
    """
    if split_factor <= 0:
        raise ValueError("Split factor must be greater than 0.")

    # Get the height and width of the image
    height, width = img.shape

    # Create a new array with the specified split factor
    new_height = height * split_factor
    new_width = width * split_factor
    new_array = np.zeros((new_height, new_width), dtype=np.uint8)

    # Copy the pixels from the original image to the new image, splitting each pixel into smaller pixels
    for i in range(height):
        for j in range(width):
            new_array[i * split_factor:(i + 1) * split_factor, j * split_factor:(j + 1) * split_factor] = img[i, j]
    return new_array


def rotate_image(image: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Rotates an image by the specified angle in degree.

    Args:
        image (np.ndarray): The input image as a numpy array.
        angle_deg (float): The rotation angle in degree.

    Returns:
        tuple[np.ndarray, tuple[int, int]]: A tuple containing the rotated image as a numpy array
            and the position of the top-left vertex after rotation.

    Raises:
        None.
    """

    # Convert the angle to degrees
    # angle_deg = math.degrees(angle_rad)

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Calculate the center of the image
    center = (width // 2, height // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Calculate the new dimensions of the rotated image
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Adjust the rotation matrix to include translation
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Rotate the image
    # rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderMode=1, borderValue=-1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    # Calculate the position of the top-left vertex after rotation
    rotated_position = (
        int(rotation_matrix[0, 2]),
        int(rotation_matrix[1, 2])
    )
    # im_array = Image.fromarray(rotated_position).Convert('RBG')

    # saving the final output
    # as a PNG file
    # im_array.save('gfg_dummy_pic.png')
    return rotated_image, rotated_position


def split_and_update_shift(shifted_image: Tuple[np.ndarray, Tuple[float, float, float]], split_factor: int) -> Tuple[
    np.ndarray, Tuple[int, int, float]]:
    """
    Splits the image into smaller pixels and updates the shift accordingly.

    Args:
        shifted_image (tuple[np.ndarray, tuple[float, float, float]]): A tuple containing the shifted image
            as a numpy array and the shift values (x, y, angle).
        split_factor (int): The split factor to determine the number of smaller pixels per original pixel.

    Returns:
        tuple[np.ndarray, tuple[int, int, float]]: A tuple containing the updated image with smaller pixels
            as a numpy array and the updated shift values (x, y, angle).

    Raises:
        None.

    """

    image, shift = shifted_image
    updated_shift = (round(shift[0] * split_factor), round(shift[1] * split_factor), shift[2])
    # Split the image into smaller pixels
    image = split_pixels(image, split_factor)

    return image, updated_shift


def calculate_vertices(upper_left, height, width):
    # upper right vertex
    upper_right = (upper_left[0] + width, upper_left[1])
    # lower right vertex
    lower_right = (upper_left[0] + width, upper_left[1] + height)
    # lower left vertex
    lower_left = (upper_left[0], upper_left[1] + height)
    return [upper_left, upper_right, lower_right,lower_left]

def find_min_max_coordinates(corners):
    x_values = [corner[0] for corner in corners]
    y_values = [corner[1] for corner in corners]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    return min_x, max_x, min_y, max_y


def calculate_combined_size(combine_min, combine_max, shift, image):
    image_height, image_width = image.shape[:2]
    # TODO: do more efficiently calculate of min_x, max_x, min_y, max_y
    corners = calculate_vertices(shift, image_height, image_width)
    min_x, max_x, min_y, max_y = find_min_max_coordinates(corners)
    combine_min['x'] = min(combine_min['x'], min_x)
    combine_max['x'] = max(combine_max['x'], max_x)
    combine_min['y'] = min(combine_min['y'], min_y)
    combine_max['y'] = max(combine_max['y'], max_y)


def preprocess_combine(shifted_images: List[Tuple[np.ndarray, Tuple[float, float, float]]]) -> Tuple[
    Dict[str, int], Tuple[int, int], List[Tuple[np.ndarray, Tuple[int, int]]]]:
    """
    Preprocesses the shifted images for combining by performing the necessary steps.

    Args:
        shifted_images (list[tuple[np.ndarray, tuple[float, float, float]]]): A list of tuples containing the shifted
            images as numpy arrays and their corresponding shift values (x, y, angle).

    Returns:
        tuple[dict[str, int], tuple[int, int], list[tuple[np.ndarray, tuple[int, int]]]]: A tuple containing
            the updated mother image position (x, y) as a dictionary, the combined image size as a tuple (height, width),
            and the list of updated shifted images as tuples containing the images as numpy arrays and their updated shift
            values (x, y, angle).

    Raises:
        None.

    """

    update_shifted_images = []
    combine_min = {'x': 0, 'y': 0}
    combine_max = {'x': 0, 'y': 0}
    m_image_position = {'x': 0, 'y': 0}

    print("### Start preprocesses combine ...")

    for shifted_image in tqdm(shifted_images):
        split_factor = 3

        # Step 1 - split pixels:
        update_shifted_image = split_and_update_shift(shifted_image, split_factor)

        # Step 2 - rotate image:
        rotated_image, rotated_position = rotate_image(update_shifted_image[0], update_shifted_image[1][2])

        # Step 3 - update shift value:
        update_shifted_image = (rotated_image, (
            update_shifted_image[1][0] - rotated_position[0],
            update_shifted_image[1][1] - rotated_position[1]
        ))

        # Step 4 - calculate the combined image size:
        calculate_combined_size(combine_min, combine_max, update_shifted_image[1], rotated_image)

        # Step 5 - update the new shift list:
        update_shifted_images.append(update_shifted_image)
    print("### End preprocesses combine.")
    # Step 6 - calculate combined image size:
    combine_size = (combine_max['y'] - combine_min['y'], combine_max['x'] - combine_min['x'])

    # Update mother image position (x, y)
    m_image_position['x'] = abs(combine_min['x'])
    m_image_position['y'] = abs(combine_min['y'])

    return m_image_position, combine_size, update_shifted_images


def append_to_combine_img(x: int, y: int, combined_overlap: np.ndarray, image: np.ndarray,
                          shape: Tuple[int, int], index) -> None:
    """
    Appends the pixel values of an image to the corresponding location in the combined overlap image.

    Args:
        x (int): The x-coordinate of the top-left corner of the image within the combined overlap image.
        y (int): The y-coordinate of the top-left corner of the image within the combined overlap image.
        combined_overlap (list[list[list[int]]]): The combined overlap image, represented as a 3D list of pixel values.
        image (np.ndarray): The image to be appended.
        shape (tuple[int, int]): The shape of the combined overlap image (height, width).

    Returns:
        None

    Raises:
        None
    """

    combine_img_h, combine_img_w = shape
    image_h, image_w = image.shape[0], image.shape[1]
    image = unite_sigmoid(image)

    # for i in [*range(-y, 0), *range(image_h, combine_img_h - y)]:
    #     for j in range(combine_img_w):
    #         combined_overlap[y + i][j].append(-1)
    #
    #
    # for j in [*range(-x, 0), *range(image_w, combine_img_w - x)]:
    #     for i in range(image_h):
    #         combined_overlap[y + i][x + j].append(-1)

    combined_overlap[y:y+image_h, x:x+image_w, index] = image


    # for i in range(image_h):
    #     for j in range(image_w):
    #         if 0 <= y + i < combine_img_h and 0 <= x + j < combine_img_w:
    #             # TODO: Add sigmoid function here on image[i, j]
    #             combined_overlap[y + i][x + j].append(image[i, j])
    #         else:
    #             # combined_overlap[y + i][x + j].append(-1)
    #             print("WARNING: Should solve wrong calculation issue")
    #             break


def calculate_position_in_combine_image(shift: Tuple[int, int], m_image_position: Dict[str, int]) -> Tuple[int, int]:
    """
    Calculates the position of an image in the combined image based on its shift values and the mother image position.

    Args:
        shift (tuple[int, int]): The shift values of the image (x, y).
        m_image_position (dict[str, int]): The position of the mother image in the combined image (x, y).

    Returns:
        tuple[int, int]: The updated position of the image in the combined image (x, y).

    Raises:
        None.
    """

    x, y = shift[0], shift[1]
    x += m_image_position['x']
    y += m_image_position['y']
    return x, y


def combine(m_image_position: Dict[str, int], combine_size: Tuple[int, int],
            update_shifted_images: List[Tuple[np.ndarray, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Combines the shifted images into a single combined image.

    Args:
        m_image_position (dict[str, int]): The position of the mother image in the combined image (x, y).
        combine_size (tuple[int, int]): The size of the combined image (height, width).
        update_shifted_images (list[tuple[np.ndarray, tuple[int, int]]]): The list of shifted images with their shifts.

    Returns:
        np.ndarray: The combined image.

    Raises:
        None
    """
    print("### Start combine ...")
    combined_height, combined_width = combine_size

    print("### Create overlap array ... ")
    # Create an empty array to hold the combined image
    # combined_overlap = [[[] for _ in range(combined_width)] for _ in range(combined_height)]
    # combined_overlap = -1 * np.ones(shape=(combined_height, combined_width, len(update_shifted_images)))
    # combined_overlap = np.zeros(shape=(combined_height, combined_width, len(update_shifted_images)))
    combined_overlap = np.memmap("temp.dat", dtype=np.float32, mode='w+', shape=(combined_height, combined_width, len(update_shifted_images)))
    # Combine the images by pasting them into the empty array

    # x0s, y0s = zip(*[calculate_position_in_combine_image(shift, m_image_position) for image, shift in tqdm(update_shifted_images)])
    # images, shift = zip(*update_shifted_images)
    # x1s = [el + image.shape[0] for el, image in zip(x0s, images)]
    # y1s = [el + image.shape[1] for el, image in zip(y0s, images)]
    #
    # indices = np.array([np.stack(np.meshgrid(np.linspace(x0, x1 - 1, (x1 - x0)).astype(np.int32), np.linspace(y0, y1 - 1, (y1 - y0)).astype(np.int32))) for x0, x1, y0, y1 in zip(x0s, x1s, y0s, y1s)])
    # np.put_along_axis(combined_overlap, indices, images, axis=-1)
    for i, (image, shift) in tqdm(enumerate(update_shifted_images)):
        x, y = calculate_position_in_combine_image(shift, m_image_position)
        # combined_overlap[y:y + image.shape[0], x:x + image.shape[1], i] = unite_sigmoid(image)
        combined_overlap[y:y + image.shape[0], x:x + image.shape[1], i] = image
        # append_to_combine_img(x, y, combined_overlap, image, (combined_height, combined_width), i)

    print("### Combine overlap array ... ")
    # TODO: Implement the more method for combining the overlapping pixels
    combined_image = milo_simple_mean(combined_overlap)
    # combined_image = simple_mean(combined_overlap)
    # combined_image = kernel_mean(combined_overlap, kernel_size=2)
    # combined_image = white_is_most_important(combined_overlap)
    print("### End combine ...")
    return combined_image


def smart_combine_images(shifted_images: List[Tuple[Image.Image, Tuple[float, float, float]]]) -> Image.Image:
    """
    Combines a list of shifted images into a single combined image.

    Args:
        shifted_images (list[tuple[Image.Image, tuple[float, float, float]]]): A list of shifted images, where each image is represented as a tuple containing the image object and its shift values (x, y, angle).

    Returns:
        Image.Image: The combined image.

    Raises:
        None.

    Examples:
        shifted_images = [(image1, (0, 0, 0)), (image2, (15.0, -25.0, 2))]
        combined_image = smart_combine_images(shifted_images)
    """
    # TODO: Remove in the pipeline
    # shifted_images = load_images(shifted_images)
    m_image_position, combine_size, update_shifted_images = preprocess_combine(shifted_images)
    combined_image = combine(m_image_position, combine_size, update_shifted_images)
    # Make transform to get highlights on the hot areas
    # combined_image = highlights_hot_areas(combined_image, 50)
    # combined_image = highlights_black_areas(combined_image, 100)
    return combined_image


# Todo - list:
# 1. Add more function
# 2. Fix wight problem
# 3. Optimize the run time

# Suggest:
# In the real time suggest to remove the output to improve run time
def main():
    shifted_images = []
    # img1 = cv2.imread("./Test/21.jpg", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread("./Test/22.jpg", cv2.IMREAD_GRAYSCALE)
    # img3 = cv2.imread("./Test/22.jpg", cv2.IMREAD_GRAYSCALE)
    imt1 = ("Test/21.jpg", (0, 0, 0))
    # imt2 = ("Test/22.jpg", (232, 0, 0))#232, 0
    imt2 = ("Test/22.jpg", (232, 1, -1))  # 232, 0
    imt3 = ("Test/22.jpg", (232.65, -10.3, 0))  # 232, 0
    # imt2 = ("Test/22.jpg", (232, 0, 0))
    shifted_images.append(imt1)
    shifted_images.append(imt2)
    shifted_images.append(imt3)
    merged = smart_combine_images(shifted_images)
    # cv2.imshow('Transofrm', merged)
    # merged = Image.fromarray(merged)
    merged.save("merged.jpg")


if __name__ == "__main__":
    main()
