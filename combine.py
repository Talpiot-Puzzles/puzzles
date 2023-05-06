import cv2

def highlightsHotAreas(image, thresh_value):
    """
    This function highlights the hot areas in the image with bitwise and
    :param image: ...
    :param thresh_value: ...
    :return: ...
    """
    # img = cv2.imread(path, 0)
    _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(thresh_img, image)
    return result


def smart_combine_images(shifted_images):
    """
    ...
    :param shifted_images: ...
    :return: ...
    """
    combined_image = None
    return combined_image