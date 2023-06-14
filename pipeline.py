import os

import cv2
import numpy as np
from PIL import Image

import anchor_detection
import combine
import plain_transform
from utils import make_dir_handle_duplicate_name, timeit, capture_frames, pretty_print_pipeline_data, compute_maps, \
    undistort, crop
from plain_movement import calculate_shifts_for_layers


class Pipeline:
    def __init__(self, steps, pipeline_input=None, accessible_data=None):
        self.accessible_data = accessible_data
        self.pipeline_input = pipeline_input
        self.steps = steps

    @timeit
    def run(self):
        # Run the pipeline steps in sequence
        data = self.pipeline_input
        for step_name, step_func in self.steps:
            data = step_func(data, self.accessible_data)
            print(f"successfully run '{step_name}'")
        return data


# Step 1: Load images from directory
@timeit
def load_images(input_data, pipeline_data):
    result = pipeline_data['result']
    images_path = result['img_path']
    if result['save_images']:
        images_path = os.path.join(images_path, result['name'])
        if not os.path.exists(images_path):
            os.makedirs(images_path)

    if not input_data["is_video"]:
        image_dir = input_data["path"]
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                       f.endswith(".png") or f.endswith('.jpg')]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        for i, img in enumerate(images):
            Image.fromarray(img).save(f"{images_path}\img{i}.jpg")
    else:
        path, start_time, duration = input_data['path'], input_data["start_time"], input_data["duration"]
        images = capture_frames(path, start_time, duration)

    for i, img in enumerate(images):
        Image.fromarray(img).save(rf"{images_path}\img{i}.jpg")

    pipeline_data['images'] = images
    return images


# Step 2: Preprocess images
@timeit
def preprocess_images(images, pipeline_data):
    height, width = images[0].shape
    filters = pipeline_data["filters"]
    crop_from_frame = filters['crop_from_frame']
    barrel_coef = filters["dist_coef"]

    map1, map2 = compute_maps(width, height, barrel_coef)

    preprocessed_images = images
    if filters["distortion"]:
        preprocessed_images = [undistort(image, map1, map2) for image in preprocessed_images]
    if filters["crop"]:
        preprocessed_images = [crop(image, crop_from_frame) for image in preprocessed_images]
    if filters["stretch"]:
        min_value = np.min(images)
        max_value = np.max(images)
        print(f"min: {min_value}, max: {max_value}")
        filters['min_val'] = min_value
        filters['max_val'] = max_value

    return preprocessed_images


# Step 3: Anchor detection
@timeit
def detect_anchors(preprocessed_images, pipeline_data):
    anchors = anchor_detection.detect_anchors(preprocessed_images)
    return anchors


# Step 4: Connect images
@timeit
def connect_images(anchors, pipeline_data):
    to_use = [[anchors[i], anchors[i + 1]] for i in range(0, len(anchors) - 1, 2)]
    return plain_transform.plain_transform(to_use)


# Step 5: Shift images
@timeit
def shift_images(shifts, pipeline_data):
    h_anchor, h_ground = pipeline_data["heights"]["anchor"], pipeline_data["heights"]["ground"]
    layers_num = pipeline_data["heights"]["layers_around"]
    thickness = pipeline_data["heights"]["layer_thickness"]
    layers_below = layers_num // 2

    lowest_h_anchor = h_anchor - (layers_below * thickness)
    refocused_shifts = calculate_shifts_for_layers(shifts, lowest_h_anchor, h_ground, layers_num, thickness)
    return refocused_shifts


# Step 6: Combine images
@timeit
def combine_images(shifts, pipeline_data):
    filters = pipeline_data["filters"]
    path, name = pipeline_data['result']['path'], pipeline_data['result']['name']
    res_path = make_dir_handle_duplicate_name(path, name)

    # create a file that documents the configurations of the pipeline without the images
    with open(os.path.join(res_path, "config.txt"), "w") as f:
        f.write(pretty_print_pipeline_data(pipeline_data))

    combined_images = []
    for i, shifts_group in enumerate(shifts):
        print(f"layer number {i}")
        images = pipeline_data['images']
        # insert (0, 0, 0) so that the first image will not be shifted
        shifts_group = np.insert(shifts_group, 0, [0, 0, 0], axis=0)
        shifted_images = [(image, (*shifted, 0)) for image, shifted in zip(images, shifts_group)]
        combined_image = combine.smart_combine_images(shifted_images, filters)

        combined_image.save(rf"{res_path}\res{i}.jpg")
        combined_images.append(combined_image)
    return combined_images


# Step 7: Object detection
@timeit
def detect_objects(combined_image):
    # labeled_image = detect_objects_in_image(combined_image, images)
    # return labeled_image
    return combined_image


def make_pipeline(start_step=None, end_step=None, pipeline_input=None, accessible_data=None):
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
    return Pipeline(steps, pipeline_input, accessible_data)


# TODO: make the averaging weighted in favour of white pixles
if __name__ == '__main__':
    dist_coef = -5.15e-5

    input_path = r"C:\Users\t9146472\Documents\61.MP4"
    is_video = True
    start_time = (0 * 60 + 14)
    duration = 1

    distort_filter = True
    crop_filter = True
    stretch_histogram = True
    crop_from_frame = 13
    contrast_factor = 10
    split_factor = 1

    anchor_height = 50.5
    ground_height = 50
    num_of_layers = 3
    layer_thickness = 0.5

    name = "new"
    res_path = r".\results"
    img_path = r".\data"
    save_images = True

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    input_data = {'path': input_path, "is_video": is_video, "start_time": start_time, "duration": duration}
    accessible_data = {'filters': {'crop': crop_filter, 'distortion': distort_filter, 'dist_coef': dist_coef,
                                   'crop_from_frame': crop_from_frame, 'stretch': stretch_histogram,
                                   'contrast_factor': contrast_factor, 'min_val': 0, 'max_val': 255,
                                   'split_factor': split_factor},
                       'result': {'path': res_path, 'img_path': img_path, 'name': name, 'save_images': save_images},
                       'heights': {'anchor': anchor_height, 'ground': ground_height, 'layers_around': num_of_layers,
                                   'layer_thickness': layer_thickness},
                       'input_data': input_data}
    p = make_pipeline(start_step='load_images', end_step='combine_images', pipeline_input=input_data,
                      accessible_data=accessible_data)
    output_data = p.run()
    print(output_data)
