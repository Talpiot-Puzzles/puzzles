from typing import List, Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Preprocess functions:
def split_pixels(img: np.ndarray, split_factor: int) -> np.ndarray:
    """
    Splits each pixel in an image into smaller pixels to increase its resolution.

    Args:
        img (np.ndarray): The original image as a numpy array.
        split_factor (int): The factor by which to split each pixel.

    Returns:
        np.ndarray: The new image with increased resolution as a numpy array.

    Raises:
        ValueError: If the split factor is less than or equal to 0.
    """
    if split_factor <= 0:
        raise ValueError("Split factor must be greater than 0.")

    # Splitting pixels by repeating along both axes
    split_img = np.repeat(np.repeat(img, split_factor, axis=0), split_factor, axis=1)

    return split_img

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
    image = split_pixels(image, split_factor)

    return image, updated_shift

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
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    # Calculate the position of the top-left vertex after rotation
    rotated_position = (
        int(rotation_matrix[0, 2]),
        int(rotation_matrix[1, 2])
    )

    return rotated_image, rotated_position

def calculate_vertices(upper_left: Tuple[int, int], height: int, width: int) -> List[Tuple[int, int]]:
    """
    Calculate the vertices of an image given the upper-left corner, height, and width.

    Args:
        upper_left (Tuple[int, int]): The coordinates of the upper-left corner of the rectangle.
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        List[Tuple[int, int]]: A list of tuples representing the vertices of the rectangle in the order
                               [upper_left, upper_right, lower_right, lower_left].

    """
    upper_right = (upper_left[0] + width, upper_left[1])
    lower_right = (upper_left[0] + width, upper_left[1] + height)
    lower_left = (upper_left[0], upper_left[1] + height)

    return [upper_left, upper_right, lower_right, lower_left]

def find_min_max_coordinates(corners: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """
    Find the minimum and maximum x and y coordinates from a list of corners.

    Args:
        corners (List[Tuple[int, int]]): A list of corner coordinates.

    Returns:
        Tuple[int, int, int, int]: A tuple containing the minimum and maximum x and y coordinates in the order
                                   (min_x, max_x, min_y, max_y).

    """
    x_values = [corner[0] for corner in corners]
    y_values = [corner[1] for corner in corners]

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    return min_x, max_x, min_y, max_y

def calculate_combined_size(combine_min: Dict[str, int], combine_max: Dict[str, int], shift: Tuple[int, int], image: np.ndarray) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Calculate the combined size of an image and update the minimum and maximum coordinates accordingly.

    Args:
        combine_min (Dict[str, int]): The dictionary storing the minimum x and y coordinates.
        combine_max (Dict[str, int]): The dictionary storing the maximum x and y coordinates.
        shift (Tuple[int, int]): The shift coordinates.
        image (np.ndarray): The image.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: A tuple containing the updated combine_min and combine_max dictionaries.

    """
    image_height, image_width = image.shape[:2]

    # Get the corners of checked image
    corners = calculate_vertices(shift, image_height, image_width)
    min_x, max_x, min_y, max_y = find_min_max_coordinates(corners)

    # Update combine min and max values
    combine_min['x'] = min(combine_min['x'], min_x)
    combine_max['x'] = max(combine_max['x'], max_x)
    combine_min['y'] = min(combine_min['y'], min_y)
    combine_max['y'] = max(combine_max['y'], max_y)

    return combine_min, combine_max

def preprocess_combine(shifted_images: List[Tuple[np.ndarray, Tuple[float, float, float]]], filters) -> Tuple[
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
        split_factor = filters['split_factor']

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

# Combine process functions:

def milo_simple_mean(overlap_img: np.ndarray, ws: np.ndarray) -> Image:
    """
    Applies the Milo Simple Mean fusion algorithm to an overlap image.

    Args:
        overlap_img (np.ndarray): The overlap image as a numpy array.
        ws (np.ndarray): The weight matrix for fusion.

    Returns:
        Image: The fused image as a PIL Image object.

    """
    alpha = 1
    # alpha = 1.1
    # alpha = 1.15
    to_mean = np.power(overlap_img, 1 / alpha)
    summed = to_mean.sum(axis=-1)
    summed = np.power(summed, alpha)
    res = np.divide(summed, ws, where=ws != 0)
    res = np.nan_to_num(res, nan=0)

    return Image.fromarray(res.astype(np.uint8)).convert('RGB')

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

    shift_x, shift_y = shift[0], shift[1]
    shift_x += m_image_position['x']
    shift_y += m_image_position['y']
    return shift_x, shift_y

def stretch_histogram(img: np.ndarray, max_value: float, min_value: float) -> np.ndarray:
    """
    Stretch the histogram of an image to utilize the full dynamic range.

    Args:
        img (np.ndarray): Input image array.
        max_value (float): Maximum value for histogram stretching.
        min_value (float): Minimum value for histogram stretching.

    Returns:
        np.ndarray: Image array after histogram stretching.

    """
    # Calculate the stretching factor
    stretching_factor = 255.0 / (max_value - min_value)

    # Perform histogram stretching
    stretched_img = ((stretching_factor * (img - min_value))).astype(np.uint8)

    return stretched_img

def combine_process(m_image_position: Dict[str, int], combine_size: Tuple[int, int],
            update_shifted_images: List[Tuple[np.ndarray, Tuple[int, int, int]]], filters) -> np.ndarray:
    """
    Combines the shifted images into a single combined image.

    Args:
        m_image_position (dict[str, int]): The position of the mother image in the combined image (x, y).
        combine_size (tuple[int, int]): The size of the combined image (height, width).
        update_shifted_images (list[tuple[np.ndarray, tuple[int, int]]]): The list of shifted images with their shifts.
        TODO: filters ():
    Returns:
        np.ndarray: The combined image.

    Raises:
        None
    """
    print("### Start combine ...")
    combined_height, combined_width = combine_size

    print("### Create overlap array ... ")
    # Create an empty array to hold the combined image
    combined_overlap = np.zeros(shape=(combined_height, combined_width, len(update_shifted_images)))
    ws = np.zeros(shape=(combined_height, combined_width))
    max_val = filters['max_val']
    min_val = filters['min_val']
    contrast_factor = filters['contrast_factor']
    for i, (image, shift) in tqdm(enumerate(update_shifted_images)):
        shift_x, shift_y = calculate_position_in_combine_image(shift, m_image_position)

        if max_val != 255 or min_val != 0:
            image = stretch_histogram(image, max_val, min_val)
        w = np.power(image / 255, contrast_factor)
        height, width = image.shape[0], image.shape[1]

        ws[shift_y: shift_y + height, shift_x: shift_x + width] += w
        combined_overlap[shift_y: shift_y + height, shift_x: shift_x + width, i] = image * w

    print("### End to create overlap array ... ")
    print("### Start do mean on the overlap array ...")
    combined_image = milo_simple_mean(combined_overlap, ws)
    print("### End do mean on the overlap array ...")
    print("### End combine process ...")
    return combined_image


def smart_combine_images(shifted_images: List[Tuple[Image.Image, Tuple[float, float, float]]], filters) -> Image.Image:
    """
    Combines a list of shifted images into a single combined image.

    Args:
        shifted_images (list[tuple[Image.Image, tuple[float, float, float]]]): A list of shifted images, where each image is represented as a tuple containing the image object and its shift values (x, y, angle).
        TODO: filters ():
    Returns:
        Image.Image: The combined image.

    Raises:
        None.

    Examples:
        shifted_images = [(image1, (0, 0, 0)), (image2, (15.0, -25.0, 2))]
        combined_image = smart_combine_images(shifted_images)
    """
    # Do preprocess steps
    m_image_position, combine_size, update_shifted_images = preprocess_combine(shifted_images, filters)
    # Do combine process to make combined image
    combined_image = combine_process(m_image_position, combine_size, update_shifted_images, filters)
    return combined_image

