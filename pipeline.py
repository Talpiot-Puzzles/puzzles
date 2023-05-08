import os
import traceback

import cv2
from matplotlib import pyplot as plt

import anchor_detection
import plain_transform


class Pipeline:
    def __init__(self, steps, pipeline_input=None):
        self.accessible_data = None
        self.pipeline_input = pipeline_input
        self.steps = steps

    def run(self):
        # Run the pipeline steps in sequence
        data = self.pipeline_input
        for step_name, step_func in self.steps:
            try:
                data = step_func(data)
            except Exception as e:
                traceback.format_exc()
                raise ValueError(f"Error running step '{step_name}': {str(e)}")
        return data


# Step 1: Load images from database
def load_images(image_dir):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
    images = [cv2.imread(path)[..., ::-1] for path in image_paths]
    return images


# Step 2: Preprocess images
def preprocess_images(images):
    return images
    preprocessed_images = [preprocess_image(image) for image in images]
    return preprocessed_images


# Step 3: Anchor detection
def detect_anchors(preprocessed_images):
    return anchor_detection.detect_anchors(preprocessed_images)


# Step 4: Connect images
def connect_images(anchors):
    to_use = [[anchors[i], anchors[i+1]] for i in range(0, len(anchors) - 1, 2)]
    return plain_transform.plain_transform(to_use)
    connected_images = connect_images_from_anchors(anchors)
    return connected_images


# Step 5: Shift images
def shift_images(connected_images):
    shifted_images = focus_by_lower_plane(connected_images)
    return shifted_images


# Step 6: Combine images
def combine_images(shifted_images):
    combined_image = smart_combine_images(shifted_images)
    return combined_image


# Step 7: Object detection
def detect_objects(combined_image):
    labeled_image = detect_objects_in_image(combined_image)
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
    input_data = 'images'
    # p = make_pipeline(start_step='load_images', end_step='combine_images', pipeline_input=input_data)
    p = make_pipeline(start_step='load_images', end_step='connect_images', pipeline_input=input_data)
    output_data = p.run()
    print(output_data)
