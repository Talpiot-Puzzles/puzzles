import cv2
import math
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import time


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

def load_images(shifted_images: list[tuple[str, tuple[int, int, int]]]):
    """
    Read a list of image paths and return a list of corresponding images.

    :param paths: A list of image paths
    :return: A list of images
    """
    return [(load_image(path), *rest) for path, *rest in shifted_images]



def list_mean(lst):
    return int(np.ceil((sum(lst) / len(lst))))

def simple_mean(overlap_img):
    combine_mean = []

    for inner_list in overlap_img:
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
def split_pixels(img, split_factor):
    # Convert the image to a numpy array
    # img_array = np.array(img)

    # Get the height and width of the image
    channels = 2
    height, width = img.shape

    # Create a new array with the specified split factor
    new_height = height * split_factor
    new_width = width * split_factor
    new_array = np.zeros((new_height, new_width), dtype=np.uint8)

    # Copy the pixels from the original image to the new image, splitting each pixel into smaller pixels
    for i in range(height):
        for j in range(width):
            new_array[i*split_factor:(i+1)*split_factor, j*split_factor:(j+1)*split_factor] = img[i, j]

    # Convert the image to RGB mode
    # new_img = Image.fromarray(new_array)
        # .convert('RGB')

    # Save the new image
    # new_img.save("Split.jpg")
    return new_array


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
    # Convert the image to a numpy array
    # image_array = np.array(image)
    # Rotate the image
    # TODO: Check if possible to warp with -1 instead 0
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    # Calculate the position of the top-left vertex after rotation
    rotated_position = (
        int(rotation_matrix[0, 2]),
        int(rotation_matrix[1, 2])
    )
    # new_img = Image.fromarray(rotated_image)
        # .convert('RGB')
    # TODO: Check why return 0 array
    # new_img = np.array(rotated_image)
    # Save the new image
    # path = "rotated_image.jpg"
    # new_img.save(path)
    # rotated_image = load_image(path)
    return rotated_image, rotated_position

def split_and_update_shift(shifted_image, split_factor):
    image , shift = shifted_image
    shift = (round(shift[0] * split_factor), round(shift[1] * split_factor), shift[2])
    image = split_pixels(image, split_factor)
    return (image, shift)

def calculate_vertices(upper_left, height, width):
    # upper right vertex
    upper_right = (upper_left[0], upper_left[1] + width)
    # lower right vertex
    lower_right = (upper_right[0] + height, upper_right[1] + width)
    # lower left vertex
    lower_left = (upper_left[0] + height, upper_left[1])
    return [upper_left, upper_right, lower_right, lower_left]


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



def preprocess_combine(shifted_images):
    update_shifted_images = []
    combine_min = {'x': 0, 'y': 0}
    combine_max = {'x': 0, 'y': 0}
    m_image_position = {'x': 0, 'y': 0}
    for shifted_image in shifted_images:
        split_factor = 3
        # Step 1 - spilt pixels:
        update_shifted_image = split_and_update_shift(shifted_image, split_factor)
        # Step 2 - rotate_image:
        rotated_image, rotated_position = rotate_image(update_shifted_image[0], update_shifted_image[1][2])
        # Step 3 - update shift value:
        update_shifted_image = (rotated_image, (update_shifted_image[1][0] - rotated_position[0], update_shifted_image[1][1] - rotated_position[1]))
        # Step 4 - calculate the combine image size:
        calculate_combined_size(combine_min, combine_max, update_shifted_image[1], rotated_image)
        # Step 5 - update the new shift list:
        update_shifted_images.append(update_shifted_image)
    # Step 6 - calculate combine image size:
    combine_size = (combine_max['y'] - combine_min['y'], combine_max['x'] - combine_min['x'])
    # Update mother image position (x, y)
    m_image_position['x'] = abs(combine_min['x'])
    m_image_position['y'] = abs(combine_min['y'])
    return m_image_position, combine_size, update_shifted_images

def append_to_combine_img(x: int, y: int, combined_overlap, image: np.ndarray, shape):
    combine_img_h, combine_img_w = shape
    image_h, image_w = image.shape[0], image.shape[1]

    for i in range(image_h):
        for j in range(image_w):
            if 0 <= y + i < combine_img_h and 0 <= x + j < combine_img_w:
                combined_overlap[y + i][x + j].append(image[i, j])
            else:
                print("WARNING: Should to solve wrong calculate issue")
                break

def calculate_position_in_combine_image(shift, m_image_position):
    x, y = shift[0], shift[1]
    x += m_image_position['x']
    y += m_image_position['y']
    return x, y


def combine(m_image_position, combine_size, update_shifted_images):
    combined_height, combined_width = combine_size
    # Create an empty array to hold the combined image
    combined_overlap = [[[] for _ in range(combined_width)] for _ in range(combined_height)]

    # Combine the images by pasting them into the empty array
    for image, shift in update_shifted_images:
        # image = highlights_hot_areas(image, 100)
        x, y = calculate_position_in_combine_image(shift, m_image_position)
        append_to_combine_img(x, y, combined_overlap, image, (combined_height, combined_width))
    # TODO: return the list array and build new layer or get function as param like simple_mean
    combined_image = simple_mean(combined_overlap)
    return combined_image

def smart_combine_images(shifted_images: list[tuple[Image.Image, tuple[float, float, float]]]):
    """
    Combine multiple images into a single, large image by aligning them using
    their corresponding shift vectors and .

    :param shifted_images: List of tuples, where each tuple contains an image
                           and its corresponding shift vector (as a tuple of float)
                           (x, y, angle (in radian))
    :return: combine image - single image
    """
    # TODO: Remove in the pipeline
    shifted_images = load_images(shifted_images)
    m_image_position, combine_size, update_shifted_images = preprocess_combine(shifted_images)
    start_time = time.time()

    combined_image = combine(m_image_position, combine_size, update_shifted_images)

    end_time = time.time()
    execution_time = end_time - start_time

    print("Execution time:", execution_time, "seconds")

    # Make transform for get highlights on the hot areas
    # combined_image = highlights_hot_areas(combined_image, 100)
    return combined_image


# Todo - list:
# 3. Connect to pipline and suit the param
# 4. Ask for test it on more images
def main():
    shifted_images = []
    img1 = cv2.imread("./Test/21.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./Test/22.jpg", cv2.IMREAD_GRAYSCALE)
    imt1 = ("Test/21.jpg", (0, 0, 0))
    # imt2 = ("Test/22.jpg", (-10, 1, 0.5))#232, 0
    imt2 = ("Test/22.jpg", (232, 1, -1))  # 232, 0
    shifted_images.append(imt1)
    shifted_images.append(imt2)
    merged = smart_combine_images(shifted_images)
    # cv2.imshow('Transofrm', merged)
    # merged = Image.fromarray(merged)
    merged.save("merged.jpg")

if __name__ == "__main__":
    main()