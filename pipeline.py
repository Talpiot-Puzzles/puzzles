import os

import cv2
import numpy as np
from PIL import Image

import anchor_detection
import combine
import plain_movement
import plain_transform


class Pipeline:
    def __init__(self, steps, pipeline_input=None):
        self.accessible_data = {}
        self.pipeline_input = pipeline_input
        self.steps = steps

    def run(self):
        # Run the pipeline steps in sequence
        data = self.pipeline_input
        for step_name, step_func in self.steps:
            data = step_func(data, self.accessible_data)
            print(f"successfully run '{step_name}'")
        return data


# Step 1: Load images from directory
def load_images(input_data, pipeline_data):
    if not input_data["is_video"]:
        image_dir = input_data["path"]
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                       f.endswith(".png") or f.endswith('.jpg')]
        # image_paths = [image_paths[i] for i in range(83, 85)]
        image_paths = [image_paths[i] for i in range(230, 250)]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        for i, img in enumerate(images):
            Image.fromarray(img).save(f".\imgs\img{i}.jpg")
    else:
        vidcap = cv2.VideoCapture(input_data['path'])
        images = [cv2.cvtColor(vidcap.read()[1], cv2.COLOR_BGR2GRAY) for _ in range(80)]

    pipeline_data['images'] = images
    return images


# Step 2: Preprocess images
def preprocess_images(images, pipeline_data):
    height, width = images[0].shape
    barrel_coef = -5.15e-5
    distCoeff = np.array([[barrel_coef], [0], [0], [0]], dtype=np.float64)
    cam = np.eye(3, dtype=np.float32)
    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = 10.0  # define focal length x
    cam[1, 1] = 10.0  # define focal length y
    # Compute the coordinate transform only once
    map1, map2 = cv2.initUndistortRectifyMap(cam, distCoeff, None, None, (width, height), cv2.CV_32FC1)

    def preprocess_image(img):
        # Apply the coordinate transform using cv2.remap
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    preprocessed_images = [preprocess_image(image) for image in images]
    return preprocessed_images
        # image = images[i]
        # # Get the dimensions of the image
        # height, width = image.shape
        #
        # # Calculate the boundaries for the middle third
        # start_row = height // 4
        # end_row = (height * 3) // 4
        # start_col = width // 4
        # end_col = (width * 3) // 4
        #
        # # Extract the middle third of the image
        # middle_third = image[start_row:end_row, start_col:end_col]
        # images[i] = middle_third
        # images[i] = np.where(images[i] == 0, 1, images[i])


# Step 3: Anchor detection
def detect_anchors(preprocessed_images, pipeline_data):
    return anchor_detection.detect_anchors(preprocessed_images)


# Step 4: Connect images
def connect_images(anchors, pipeline_data):
    to_use = [[anchors[i], anchors[i + 1]] for i in range(0, len(anchors) - 1, 2)]
    return plain_transform.plain_transform(to_use)


# Step 5: Shift images
def shift_images(shifts, pipeline_data):
    h_enkor = 40
    h_ground = 40
    dh = h_ground - h_enkor
    refocused_shifts = []
    for tup in shifts:
        # print("tup", tup)
        (dx, dy, tet) = tup
        refocse_x = -dh * dx / h_enkor
        refocse_y = -dh * dy / h_enkor
        refocused_shifts.append((dx + refocse_x, dy + refocse_y, tet))
    return refocused_shifts


# Step 6: Combine images
def combine_images(shifts, pipeline_data):
    images = pipeline_data['images']
    shifts = np.insert(shifts, 0, [0, 0, 0], axis=0)
    # print(shifts)
    shifted_images = [(image, (*shifted, 0)) for image, shifted in zip(images, shifts)]
    combined_image = combine.smart_combine_images(shifted_images)
    combined_image.save(r".\res\combined_image.jpg")
    return combined_image


# Step 7: Object detection
def detect_objects(combined_image):
    # labeled_image = detect_objects_in_image(combined_image, images)
    # return labeled_image
    return combined_image


def make_pipeline(start_step=None, end_step=None, pipeline_input=None):
    # Define the full pipeline
    full_pipeline = [
        ('load_images', load_images),
        ('preprocess_images', preprocess_images),
        ('detect_anchors', detect_anchors),
        ('connect_images', connect_images),
        ('shift_images', shift_images),
        ('combine_images', combine_images),
        ('detect_objects', detect_objects)]

    # Get the start and end indices of the pipeline
    if start_step is not None:
        try:
            start_index = [i for i, (name, _) in enumerate(full_pipeline) if name == start_step][0]
        except IndexError:
            raise ValueError(f"Invalid start step: {start_step}")
    else:
        start_index = 0

    if end_step is not None:
        try:
            end_index = [i for i, (name, _) in enumerate(full_pipeline) if name == end_step][0] + 1
        except IndexError:
            raise ValueError(f"Invalid end step: {end_step}")
    else:
        end_index = len(full_pipeline)

    steps = full_pipeline[start_index:end_index]
    return Pipeline(steps, pipeline_input)


if __name__ == '__main__':
    # Example usage
    # input_data = r'C:\Users\t9146472\Documents\name'
    # input_data = {"path": r'C:\Users\t9146472\Documents\DJI_04_310_320', "is_video": False}
    input_data = {'path': 'DJI_0603_T.MP4', "is_video": True}
    p = make_pipeline(start_step='load_images', end_step='combine_images', pipeline_input=input_data)
    output_data = p.run()
    print(output_data)

    # p = make_pipeline(start_step='load_images', end_step='detect_anchors', pipeline_input=input_data)
    # images = p.accessible_data['images']
    # res = [images[0], *(2 * [images[i] for i in range(1, len(images) - 1)]), images[-1]]
    # res = [np.array(el) for el in res]
    #
    # for i, img in enumerate(res):
    #     for point in output_data[i]:
    #         res[i] = cv2.circle(img, [int(el) for el in point], 100, (255, 0, 0))
    #
    # fig, ax = plt.subplots(1, len(res))
    # for i in range(len(res)):
    #     ax[i].imshow(res[i])
    #
    # plt.show()

