# Puzzles Project

The Puzzles project is a program dedicated to identifying objects that are obscured beneath undergrowth in thermal images. By using multiple images captured from different angles, Puzzles can decipher hidden objects despite only a few visible pixels in the undergrowth.
## Getting Started

### Installation
1. Clone the repository: `git clone https://github.com/example-user/puzzles.git`
2. Navigate to the cloned repository: `cd puzzles`
3. Install the required dependencies: `pip install -r requirements.txt`

### Execution
1. Run the program: `python puzzles.py --config-path config.json`

## Overview of the Code

The codebase primarily comprises of the following key methods:
1. `load_images`: Reads thermal images from a specified directory.
2. `preprocess_images`: Conducts initial image processing tasks such as distortion correction, cropping, and histogram stretching.
3. `detect_anchors`: Identifies anchor points in the image to aid subsequent image alignment.
4. `align_images`: Leverages the identified anchor points to align the set of images.
5. `shift_focus_plain`: Refocuses the images on designated layer(s).
6. `combine_images`: Merges all the refocused images into a single comprehensive image.
7. `detect_objects`: Executes object detection on the combined image.
8. `save_output`: Stores the final outcome and labeled images to a pre-defined directory.

Each step of the pipeline prints a success message upon successful execution. The sequence of these steps can be customized using the `make_pipeline` function.


## Configuration

The program requires a configuration file (`config.json`) to specify the input data, filters, heights, and result parameters. Here is an example configuration:

```json

{
  "input_data": {
    "input_path": "C:/path/to/images",
    "is_video": true,
    "start_time": 37,
    "duration": 2,
    "max_images": 4
  },
  "filters": {
    "crop_filter": 0.1,
    "distort_filter": true,
    "stretch_histogram": true,
    "contrast_factor": 1,
    "split_factor": 3
  },
  "heights": {
    "anchor_height": 48.5,
    "ground_height": 50,
    "num_of_layers": 1,
    "layer_thickness": 0.5
  },
  "result": {
    "name": "for_split",
    "res_path": "./results",
    "img_path": "./data",
    "save_images": false
  }
}
```


### Input Data 
#### Defines the path to the input images, if the input is a video, the start time, duration, and the maximum number of images to be processed. 

- `input_path` (string): Path to the input images or video. 
- `is_video` (boolean): Set to `true` if the input is a video, `false` otherwise. 
- `start_time` (float): Start time (in seconds) for video input. 
- `duration` (float): Duration (in seconds) for video input. 
- `max_images` (int): Maximum number of images to use.
### Filters 
#### Defines various filter parameters for preprocessing the images such as cropping factor, distortion, histogram stretching, contrast factor, and split factor. 

- `crop_filter` (float): Crop filter factor for image preprocessing. 
- `distort_filter` (boolean): Enable distortion correction filter. 
- `stretch_histogram` (boolean): Enable histogram stretching filter. 
- `contrast_factor` (float): Contrast factor for histogram stretching. 
- `split_factor` (int): Split factor for splitting pixels.
### Heights 
#### Specifies the heights and layer details to be considered during the image shifting step. 
- `anchor_height` (float): Height of the anchor. 
- `ground_height` (float): Height of the ground. 
- `num_of_layers` (int): Number of layers. 
- `layer_thickness` (float): Thickness of each layer.
### Result 
#### Specifies the output details like the output directory path, whether to save images, etc.
- `name` (string): Name of the result. 
- `res_path` (string): Path to save the result. 
- `img_path` (string): Path to save the intermediate images. 
- `save_images` (boolean): Set to `true` to save intermediate images.



## Future Improvements
- Implement weighted averaging in favor of white pixels for more accurate object detection.
- Add support for different object detection algorithms.
- Improve user interface for easier configuration and visualization.
- Optimize performance and memory usage.
- Add unit tests for better code quality.
- Add support for different image formats. 
- Add support for different input types (e.g. video, live feed, etc.).
`

Once you have your configuration file set up, run the script from the command line with the configuration file as an argument:

```arduino

python puzzles.py --config-path /path/to/your/config.json
```


## Future Improvements

This is an active project and we plan to continually refine and expand the capabilities of Puzzles. Some of the areas we're currently focusing on include: 
- Improving the accuracy of object detection by fine-tuning the algorithm. 
- Extending the preprocessing steps to enhance the quality of images for better object detection. 
- Optimizing the performance of the pipeline to process images faster.
- Adding support for different object detection algorithms.
## Contributing

We welcome contributions to the Puzzles project. If you have an idea for improvement, please open an issue describing your idea before making changes. If you'd like to contribute code, please open a pull request.
