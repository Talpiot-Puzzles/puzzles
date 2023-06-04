import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import anchor_detection
import combine
import plain_transform


class Pipeline:
    def __init__(self, steps, pipeline_input=None, accessible_data=None):
        self.accessible_data = accessible_data
        self.pipeline_input = pipeline_input
        self.steps = steps

    def run(self):
        # Run the pipeline steps in sequence
        data = self.pipeline_input
        for step_name, step_func in self.steps:
            data = step_func(data, self.accessible_data)
            print(f"successfully run '{step_name}'")
        return data


def capture_frames(path, start_time, duration):
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Get the frames per second (FPS) of the video
    start_frame = int(start_time * fps)  # Calculate the start frame based on the start time
    end_frame = int((start_time + duration) * fps)  # Calculate the end frame based on the start time and duration

    print("frames", end_frame - start_frame)

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
        path, start_time, duration = input_data['path'], input_data["start_time"], input_data["duration"]
        images = capture_frames(path, start_time, duration)

    pipeline_data['images'] = images
    return images


# Step 2: Preprocess images
def preprocess_images(images, pipeline_data):
    height, width = images[0].shape
    filters = pipeline_data["filters"]
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
        start_row = height // 13
        end_row = (height * 12) // 13
        start_col = width // 13
        end_col = (width * 12) // 13

        # Extract the middle third of the image
        middle_third = img[start_row:end_row, start_col:end_col]
        img = middle_third
        return np.where(img == 0, 1, img)

    preprocessed_images = images
    if filters["distortion"]:
        preprocessed_images = [undistort(image) for image in preprocessed_images]
    if filters["crop"]:
        preprocessed_images = [crop(image) for image in preprocessed_images]

    return preprocessed_images


# Step 3: Anchor detection
def detect_anchors(preprocessed_images, pipeline_data):
    anchors = anchor_detection.detect_anchors(preprocessed_images)
    return anchors
    images = pipeline_data['images']
    res = [images[0], *(2 * [images[i] for i in range(1, len(images) - 1)]), images[-1]]
    res = [np.array(el) for el in res]

    for i, img in enumerate(res):
        for point in anchors[i]:
            res[i] = cv2.circle(img, [int(el) for el in point], 100, (255, 0, 0))

    fig, ax = plt.subplots(1, len(res))
    for i in range(len(res)):
        ax[i].imshow(res[i])

    plt.show()
    return anchors



# Step 4: Connect images
def connect_images(anchors, pipeline_data):
    to_use = [[anchors[i], anchors[i + 1]] for i in range(0, len(anchors) - 1, 2)]
    return plain_transform.plain_transform(to_use)


# Step 5: Shift images
def shift_images(shifts, pipeline_data):
    new_shifts = []
    # h_anchor, h_ground = pipeline_data["heights"]["anchor"], pipeline_data["heights"]["ground"]
    # for h_anchor in range(37, 43, 1):
    for h_anchor in range(30, 43, 2):
        h_ground = 40
        dh = h_ground - h_anchor
        refocused_shifts = []
        for tup in shifts:
            # print("tup", tup)
            (dx, dy, tet) = tup
            refocuse_x = -dh * dx / h_anchor
            refocuse_y = -dh * dy / h_anchor
            refocused_shifts.append((dx + refocuse_x, dy + refocuse_y, tet))

        new_shifts.append(refocused_shifts)
    return new_shifts


# Step 6: Combine images
def combine_images(shifts, pipeline_data):
    combined_images = []
    for i, shifts_group in enumerate(shifts):
        images = pipeline_data['images']
        shifts_group = np.insert(shifts_group, 0, [0, 0, 0], axis=0)
        # print(shifts_group)
        shifted_images = [(image, (*shifted, 0)) for image, shifted in zip(images, shifts_group)]
        combined_image = combine.smart_combine_images(shifted_images)
        path, name = pipeline_data['result']['path'], pipeline_data['result']['name']
        combined_image.save(rf"{path}\{name}{i}.jpg")
        combined_images.append(combined_image)
    return combined_images


# Step 7: Object detection
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


if __name__ == '__main__':
    # Example usage
    dist_coef = -5.15e-5
    anchor_height = ground_height = 40

    input_path = r'C:\Users\t9146472\Documents\DJI_0004_T.MP4'
    is_video = True
    start_time = (3 * 60 + 3)
    start_time = 54
    duration = 3
    distort_filter = True
    crop_filter = True
    name = "combination"

    res_path = r".\res"
    input_data = {'path': input_path, "is_video": is_video, "start_time": start_time, "duration": duration}
    accessible_data = {'filters': {'crop': crop_filter, 'distortion': distort_filter, 'dist_coef': dist_coef},
                       'result': {'path': res_path, 'name': name},
                       'heights': {'anchor': anchor_height, 'ground': ground_height}}
    p = make_pipeline(start_step='load_images', end_step='combine_images', pipeline_input=input_data,
                      accessible_data=accessible_data)
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
