import cv2
import numpy as np
from PIL import Image
import math
from scipy.signal import convolve2d


def highlights_hot_areas(image: Image.Image, thresh_value: int) -> np.ndarray:
    """
    This function highlights the hot areas in the input image by performing a
    bitwise 'AND' operation between the thresholded image and the original image.

    :param image: A PIL Image object representing the input image
    :param thresh_value: An integer threshold value to use for binarization
    :return: A PIL Image object representing the image with hot areas highlighted
    """
    _, thresh_img = cv2.threshold(image, thresh_value, 0, cv2.THRESH_BINARY)
    result = cv2.bitwise_or(thresh_img, image)
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



def combine(shifted_images):
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
    combined_width = math.ceil(max([image.shape[1] + math.ceil(abs(shift[0])) for image, shift in shifted_images]))
    combined_height = math.ceil(max([image.shape[0] + math.ceil(abs(shift[1])) for image, shift in shifted_images]))

    # Create an empty array to hold the combined image
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Combine the images by pasting them into the empty array
    for image, shift in shifted_images:
        # If the image is grayscale, convert it to RGB
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        # image = highlights_hot_areas(image, 2)
        # Compute the upper-left corner of the current image in the combined image
        x, y = math.ceil(shift[0]), math.ceil(shift[1])

        # remove if negative - fix
        # my intuition: move the combined image to the other size if shift is negative
        if x < 0:
            image = image[:, abs(x):, :]
            x = 0
        if y < 0:
            image = image[abs(y):, :, :]
            y = 0

        # Paste the current image into the combined image
        combined_image[y:y+image.shape[0], x:x+image.shape[1], :] = image
        # image_height, image_width, _ = image.shape
        # x_end = min(x + image_width, combined_width)
        # y_end = min(y + image_height, combined_height)
        # combined_image[y:y_end, x:x_end, :] = image[:y_end - y, :x_end - x, :]

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
    # shifted_images = load_images(shifted_images)
    combined_image = combine(shifted_images)
    # Make transform for get highlights on the hot areas
    # combined_image = highlights_hot_areas(combined_image, 100)
    return combined_image

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
    return [(img, *rest) for img, *rest in shifted_images]

# Todo - list:
# 1. find the over lap area
# 2. Add image rotation to the shift
# 3. Connect to pipline and suit the param
# 4. Ask for test it on more images
def main():
    shifted_images = []
    # imt0 = ("Test/21.jpg", (0, 0, 0))
    imt1 = ("Test/21.jpg", (0, 0, 0))
    imt2 = ("Test/22.jpg", (230, 0, 0))#232, 0
    # shifted_images.append(imt0)
    shifted_images.append(imt1)
    shifted_images.append(imt2)
    merged = smart_combine_images(shifted_images)
    # cv2.imshow('Transofrm', merged)
    merged = Image.fromarray(merged)
    merged.save("Withshift.jpg")

if __name__ == "__main__":
    main()