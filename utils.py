import math
import os
import time
import cv2

import numpy as np


def make_dir_handle_duplicate_name(path, name):
    index = 0
    while True:
        try:
            if index == 0:
                res_path = os.path.join(path, name)
            else:
                # if the path exists, try another one with an incremental index
                res_path = os.path.join(path, f"{name}_{index}")
            os.makedirs(res_path)
            # if the directory was successfully created, break the loop
            break
        except FileExistsError:
            # if the directory already exists, increment the index and try again
            index += 1
    return res_path

def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print(f"Processing time of {func.__qualname__}: {round(time.time() - start_time,2)} seconds.")
        return result

    return measure_time


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

# for preprocessing: undistort and crop

def compute_maps(width, height, barrel_coef):
    distCoeff = np.array([[barrel_coef], [0], [0], [0]], dtype=np.float64)
    cam = np.eye(3, dtype=np.float32)
    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = 10.0  # define focal length x
    cam[1, 1] = 10.0  # define focal length y
    return cv2.initUndistortRectifyMap(cam, distCoeff, None, None, (width, height), cv2.CV_32FC1)

def undistort(img, map1, map2):
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

def crop(img, crop_from_frame):
    height, width = img.shape
    start_row = height // crop_from_frame
    end_row = (height * (crop_from_frame - 1)) // crop_from_frame
    start_col = width // crop_from_frame
    end_col = (width * (crop_from_frame - 1)) // crop_from_frame
    middle_third = img[start_row:end_row, start_col:end_col]
    return np.where(middle_third == 0, 1, middle_third)

