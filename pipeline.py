import math
import os
import time

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import anchor_detection
import combine
import plain_transform


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


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


@timeit
def capture_frames(path, start_time, duration):
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Get the frames per second (FPS) of the video
    start_frame = int(start_time * fps)  # Calculate the start frame based on the start time
    end_frame = int(
        math.ceil((start_time + duration) * fps))  # Calculate the end frame based on the start time and duration

    num_frames = end_frame - start_frame
    start_seconds, start_ms = divmod(start_time, 1)
    start_minutes, start_seconds = divmod(int(start_seconds), 60)
    end_seconds, end_ms = divmod(start_time + duration, 1)
    end_minutes, end_seconds = divmod(int(end_seconds), 60)

    start_time_str = f"{start_minutes:02d}:{start_seconds:02d}.{int(start_ms * 1000):03d}"
    end_time_str = f"{end_minutes:02d}:{end_seconds:02d}.{int(end_ms * 1000):03d}"

    print(f"{num_frames} frames - {start_time_str} to {end_time_str}")
    # Set the initial frame to the start frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    images = []
    for _ in range(start_frame, end_frame):
        ret, frame = vidcap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            images.append(gray_frame)
        else:
            break

    vidcap.release()
    return images


def show_first_and_last_image_of_scan(images):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(images[0])
    ax[1].imshow(images[-1])
    plt.show()


# Step 1: Load images from directory
@timeit
def load_images(input_data, pipeline_data):
    result = pipeline_data['result']
    img_path = result['img_path']
    if result['save_images']:
        img_path = os.path.join(img_path, result['name'])
        if not os.path.exists(img_path):
            os.makedirs(img_path)

    if not input_data["is_video"]:
        image_dir = input_data["path"]
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                       f.endswith(".png") or f.endswith('.jpg')]
        # image_paths = [image_paths[i] for i in range(83, 85)]
        image_paths = [image_paths[i] for i in range(230, 250)]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        for i, img in enumerate(images):
            Image.fromarray(img).save(f"{img_path}\img{i}.jpg")
    else:
        path, start_time, duration = input_data['path'], input_data["start_time"], input_data["duration"]
        images = capture_frames(path, start_time, duration)

    for i, img in enumerate(images):
        Image.fromarray(img).save(rf"{img_path}\img{i}.jpg")

    pipeline_data['images'] = images
    # show_first_and_last_image_of_scan(images)
    return images


# Step 2: Preprocess images
@timeit
def preprocess_images(images, pipeline_data):
    height, width = images[0].shape
    filters = pipeline_data["filters"]
    crop_from_frame = filters['crop_from_frame']
    barrel_coef = filters["dist_coef"]
    distCoeff = np.array([[barrel_coef], [0], [0], [0]], dtype=np.float64)
    cam = np.eye(3, dtype=np.float32)
    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = 10.0  # define focal length x
    cam[1, 1] = 10.0  # define focal length y
    # Compute the coordinate transform only once
    map1, map2 = cv2.initUndistortRectifyMap(cam, distCoeff, None, None, (width, height), cv2.CV_32FC1)

    def undistort(img):
        # Apply the coordinate transform using cv2.remap
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    def crop(img):
        # Get the dimensions of the image
        height, width = img.shape

        # Calculate the boundaries for the middle third
        start_row = height // crop_from_frame
        end_row = (height * (crop_from_frame - 1)) // crop_from_frame
        start_col = width // crop_from_frame
        end_col = (width * (crop_from_frame - 1)) // crop_from_frame

        # Extract the middle third of the image
        middle_third = img[start_row:end_row, start_col:end_col]
        img = middle_third
        return np.where(img == 0, 1, img)

    preprocessed_images = images
    if filters["distortion"]:
        preprocessed_images = [undistort(image) for image in preprocessed_images]
    if filters["crop"]:
        preprocessed_images = [crop(image) for image in preprocessed_images]
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
    # images = pipeline_data['images']
    # res = [images[0], *(2 * [images[i] for i in range(1, len(images) - 1)]), images[-1]]
    # res = [np.array(el) for el in res]
    #
    # for i, img in enumerate(res):
    #     for point in anchors[i]:
    #         res[i] = cv2.circle(img, [int(el) for el in point], 100, (255, 0, 0))
    #
    # fig, ax = plt.subplots(1, len(res))
    # for i in range(len(res)):
    #     ax[i].imshow(res[i])
    #
    # plt.show()
    # return anchors


# Step 4: Connect images
@timeit
def connect_images(anchors, pipeline_data):
    to_use = [[anchors[i], anchors[i + 1]] for i in range(0, len(anchors) - 1, 2)]
    return plain_transform.plain_transform(to_use)


# Step 5: Shift images
@timeit
def shift_images(shifts, pipeline_data):
    new_shifts = []
    anchor, h_ground = pipeline_data["heights"]["anchor"], pipeline_data["heights"]["ground"]
    layers_num = pipeline_data["heights"]["layers_around"]
    thickness = pipeline_data["heights"]["layer_thickness"]
    layers_below = layers_num // 2
    layers_below *= thickness

    h_anchor = anchor - layers_below
    # for h_anchor in range(37, 43, 1):
    for _ in range(layers_num):
        # h_ground = 40
        dh = h_ground - h_anchor
        refocused_shifts = []
        for tup in shifts:
            # print("tup", tup)
            (dx, dy, tet) = tup
            refocuse_x = -dh * dx / h_anchor
            refocuse_y = -dh * dy / h_anchor
            refocused_shifts.append((dx + refocuse_x, dy + refocuse_y, tet))

        new_shifts.append(refocused_shifts)
        h_anchor += thickness
    return new_shifts

# pretty print of pipeline data to configuration file
def pretty_print_pipeline_data(pipeline_data):
    res = ""
    # remove images from the configuration file
    pipeline_data = pipeline_data.copy()
    pipeline_data['images'] = f"{len(pipeline_data['images'])} images"
    for key, value in pipeline_data.items():
        # handle subdictionaries
        if isinstance(value, dict):
            res += f"{key}:\n"
            for subkey, subvalue in value.items():
                res += f"\t{subkey}: {subvalue}\n"
        else:
            res += f"{key}: {value}\n"
        res += "\n"
    return res



# Step 6: Combine images
@timeit
def combine_images(shifts, pipeline_data):
    filters = pipeline_data["filters"]
    path, name = pipeline_data['result']['path'], pipeline_data['result']['name']
    res_path = os.path.join(path, name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    else:
        res_path = os.path.join(path, name + "+")
        os.makedirs(res_path)

    # create a file that documents the configurations of the pipeline without the images
    with open(os.path.join(res_path, "config.txt"), "w") as f:
        f.write(pretty_print_pipeline_data(pipeline_data))

    combined_images = []
    for i, shifts_group in enumerate(shifts):
        print(f"layer number {i}")
        images = pipeline_data['images']
        shifts_group = np.insert(shifts_group, 0, [0, 0, 0], axis=0)
        # print(shifts_group)
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
    # start_time = datetime.datetime.now()

    # Example usage
    dist_coef = -5.15e-5
    anchor_height = 50.5
    ground_height = 50

    # input_path = r'C:\Users\t9146472\Documents\third_run.MP4'
    input_path = r"C:\Users\t9146472\Documents\61.MP4"
    is_video = True
    crop_from_frame = 13
    start_time = (0 * 60 + 10)
    duration = 1
    distort_filter = True
    crop_filter = True
    stretch_histogram = True
    split_factor = 1
    contrast_factor = 10
    name = "test2"
    save_images = True
    num_of_layers = 3
    layer_thickness = 0.5

    res_path = r".\results"
    img_path = r".\data"
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

    # end_time = datetime.datetime.now()
    # elapsed_time = end_time - start_time
    # minutes = elapsed_time.seconds // 60
    # seconds = elapsed_time.seconds % 60
    # print(f"runtime: {int(minutes)} minutes {int(seconds)} seconds")

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
