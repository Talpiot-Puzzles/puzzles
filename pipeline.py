import os
import traceback

import cv2
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt


import anchor_detection
import combine
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
            try:
                data = step_func(data, self.accessible_data)
            except Exception as e:
                traceback.format_exc()
                raise ValueError(f"Error running step '{step_name}': {str(e)}")
            print(f"successfully run '{step_name}'")
        return data


# Step 1: Load images from directory
def load_images(image_dir, pipeline_data):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
    image_paths = [image_paths[i] for i in [220, 180]]
    images = [cv2.imread(path)[..., ::-1] for path in image_paths]
    # print(len(images))
    # for img in images:
    #     plt.imshow(img)
    #     plt.show()
    for i, img in enumerate(images):
        Image.fromarray(img).save(f"img{i}.jpg")
    pipeline_data['images'] = images
    return images


# Step 2: Preprocess images
def preprocess_images(images, pipeline_data):
    fig, ax = plt.subplots(1, len(images))
    for i in range(len(images)):
        ax[i].imshow(images[i])
    return images
    # preprocessed_images = [preprocess_image(image) for image in images]
    # return preprocessed_images


# Step 3: Anchor detection
def detect_anchors(preprocessed_images, pipeline_data):
    return anchor_detection.detect_anchors(preprocessed_images)


# Step 4: Connect images
def connect_images(anchors, pipeline_data):
    to_use = [[anchors[i], anchors[i + 1]] for i in range(0, len(anchors) - 1, 2)]
    return plain_transform.plain_transform(to_use)
    # connected_images = connect_images_from_anchors(anchors)
    # return connected_images


# Step 5: Shift images
def shift_images(shifts, pipeline_data):
    shifted_images = shifts
    # # Todo rotate images in advance
    # shifts_only = [el[:2] for el in shifts]
    # # TODO iterate over h2 options
    # return plain_movement.refocuse_multipule_images(shifts_only, plain_movement.H1 + plain_movement.H2, plain_movement.H2)
    # shifted_images = focus_by_lower_plane(shifts)
    return shifted_images


# Step 6: Combine images
def combine_images(shifted_images, pipeline_data):
    print(shifted_images)
    images = pipeline_data['images']
    shifted_images = [(images[0], (0, 0, 0)), (images[1], (shifted_images[0][0], shifted_images[0][1], 0))]
    combined_image = combine.smart_combine_images(shifted_images)

    merged = Image.fromarray(combined_image)
    merged.save("combined.jpg")
    return combined_image


# Step 7: Object detection
def detect_objects(combined_image):
    labeled_image = detect_objects_in_image(combined_image, images)
    return labeled_image


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
    input_data = r'./imgs'
    p = make_pipeline(start_step='load_images', end_step='combine_images', pipeline_input=input_data)
    # p = make_pipeline(start_step='load_images', end_step='detect_anchors', pipeline_input=input_data)
    output_data = p.run()
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
    print(output_data)
