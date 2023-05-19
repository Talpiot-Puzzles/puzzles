import cv2
import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def highlights_hot_areas(image: np.ndarray, thresh_value: int) -> np.ndarray:
    """
    This function highlights the hot areas in the input image by performing a
    bitwise 'AND' operation between the thresholded image and the original image.

    :param image: A PIL Image object representing the input image
    :param thresh_value: An integer threshold value to use for binarization
    :return: A PIL Image object representing the image with hot areas highlighted
    """
    _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(thresh_img, image)
    return result

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

def combine_v1(shifted_images):
    """
    Combine multiple images into a single, large image by aligning them using
    their corresponding shift vectors.

    :param shifted_images: List of tuples, where each tuple contains an image path
                           and its corresponding shift vector (as a tuple of integers)
                           (x, y, angle)
    :return: Combined image as a numpy array
    """
    # TODO: get specfific value for the overlap_size=(x_s, x_e, y_s, y_e)
    overlap_size = 300
    # Determine the size of the combined image
    combined_width = max([image.shape[1] + shift[0] for image, shift in shifted_images])
    combined_height = max([image.shape[0] + shift[1] for image, shift in shifted_images])

    # Create an empty array to hold the combined image
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Combine the images by pasting them into the empty array
    for image, shift in shifted_images:
        # If the image is grayscale, convert it to RGB
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        # image = highlights_hot_areas(image, 100)
        # Compute the upper-left corner of the current image in the combined image
        x, y = shift[0], shift[1]

        # Paste the current image into the combined image
        combined_image[y:y+image.shape[0], x:x+image.shape[1], :] = image

        # Apply a 3x3 mean filter to the overlapping region of the current image
        if x > overlap_size and y > overlap_size:
            overlap_region = combined_image[y-overlap_size:y+image.shape[0], x-overlap_size:x+image.shape[1]]
            filtered_overlap_region = apply_mean_filter(overlap_region, overlap_size)
            combined_image[y:y+image.shape[0], x:x+image.shape[1], :] = filtered_overlap_region[overlap_size:-overlap_size, overlap_size:-overlap_size]

    return combined_image

def smart_combine_images(shifted_images: list[tuple[str, tuple[int, int, int]]]):
    """
    Combine multiple images into a single, large image by aligning them using
    their corresponding shift vectors and .

    :param shifted_images: List of tuples, where each tuple contains an image path
                           and its corresponding shift vector (as a tuple of integers)
                           (x, y, angle)
    :return: combine image - single image
    """
    shifted_images = load_images(shifted_images)
    #
    combined_image = combine(shifted_images)
    combined_image = combined_image.convert('RGB')
    # Make transform for get highlights on the hot areas
    # combined_image = highlights_hot_areas(combined_image, 100)
    return combined_image

def load_images(shifted_images: list[tuple[str, tuple[int, int, int]]]):
    """
    Read a list of image paths and return a list of corresponding images.

    :param paths: A list of image paths
    :return: A list of images
    """
    return [(load_image(path), *rest) for path, *rest in shifted_images]

def calculate_vertices(upper_left, angle_rad, length, width):
    # Calculate the cosine and sine of the angle
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    # upper right vertex
    upper_right = (upper_left[0] + length * cos_angle, upper_left[1] + length * sin_angle)
    # lower right vertex
    lower_right = (upper_right[0] - width * sin_angle, upper_right[1] + width * cos_angle)
    # lower left vertex
    lower_left = (upper_left[0] - width * sin_angle, upper_left[1] + width * cos_angle)
    return [upper_left, upper_right, lower_right, lower_left]


def find_min_max_coordinates(corners):
    x_values = [corner[0] for corner in corners]
    y_values = [corner[1] for corner in corners]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    return min_x, max_x, min_y, max_y

def calculate_combined_size(shifted_images):
    combine_min_x = 0
    combine_max_x = 0
    combine_min_y = 0
    combine_max_y = 0
    for image, (x, y, degree) in shifted_images:
        image_height, image_width = image.shape[:2]
        corners = calculate_vertices((x, y), degree, image_height, image_width)
        min_x, max_x, min_y, max_y = find_min_max_coordinates(corners)
        combine_min_x = min(combine_min_x, min_x)
        combine_max_x = max(combine_max_x, max_x)
        combine_min_y = min(combine_min_y, min_y)
        combine_max_y = max(combine_max_y, max_y)
    m_img_pos = (int(abs(combine_min_x)), int(abs(combine_min_y)))
    combine_size = (combine_max_x - combine_min_x, combine_max_y - combine_min_y)
    return m_img_pos, combine_size

def rotate_image(image, angle_rad):
    angle_deg = math.degrees(angle_rad)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

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

def append_to_combine_img(x: int, y: int, combined_overlap, image: np.ndarray, shape):
    combine_img_h, combine_img_w = shape
    image_h, image_w = image.shape[0], image.shape[1]

    for i in range(image_h):
        for j in range(image_w):
            if 0 <= y + i < combine_img_h and 0 <= x + j < combine_img_w:
                combined_overlap[y + i][x + j].append(image[i, j])
            # else:
            #     print("WARNING: Should to solve worng calculate issue")

def list_mean(lst):
    return int(np.ceil((sum(lst) / len(lst))))
def simple_mean(overlap_img):
    combine_mean = []

    for inner_list in overlap_img:
        sublist_mean = []
        for values in inner_list:
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
    return combine_img

def combine(shifted_images):
    # Determine the size of the combined image
    m_img_pos, combine_size = calculate_combined_size(shifted_images)
    combined_height = int(combine_size[0])
    combined_width = int(combine_size[1])
    # Create an empty array to hold the combined image
    combined_overlap = [[[] for _ in range(combined_width)] for _ in range(combined_height)]
    combined_image = np.zeros((combined_height, combined_width, 2), dtype=np.uint8)
    combined_image = np.array(combined_image)
    # Combine the images by pasting them into the empty array

    for image, shift in shifted_images:
        # image = highlights_hot_areas(image, 100)
        x, y = shift[0], shift[1]
        x += m_img_pos[0]
        y += m_img_pos[1]
        if shift[2] != 0:
            image , pos = rotate_image(image, shift[2])
            # TODO: check if this necessary x, y += pos[0], pos[1]
        # Compute the upper-left corner of the current image in the combined image
        append_to_combine_img(x, y, combined_overlap, image, (combined_height, combined_width))
    combined_image = simple_mean(combined_overlap)
    return combined_image

# Todo - list:
# 1. find the over lap area
# 2. Add image rotation to the shift
# 3. Connect to pipline and suit the param
# 4. Ask for test it on more images
def main():
    shifted_images = []
    img1 = cv2.imread("./Test/21.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./Test/22.jpg", cv2.IMREAD_GRAYSCALE)
    imt1 = ("Test/21.jpg", (0, 0, 0))
    imt2 = ("Test/22.jpg", (-10, 1, 0.5))#232, 0
    shifted_images.append(imt1)
    shifted_images.append(imt2)
    merged = smart_combine_images(shifted_images)
    # cv2.imshow('Transofrm', merged)
    # merged = Image.fromarray(merged)
    merged.save("merged.jpg")

if __name__ == "__main__":
    main()