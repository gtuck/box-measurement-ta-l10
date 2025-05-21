# Box Measurement TA-L10

A Python application to calibrate your Tau LiDAR Camera TA-L10 and measure shipping-box dimensions with millimeter accuracy. This version includes improved command-line handling and saving/loading of calibration data.

## Features

* One-time intrinsic & depth-scale calibration with data persistence.

* Load existing calibration data for quick measurements.

* Real-time 3D point-cloud segmentation.

* Oriented bounding-box measurement.

* 3D visualization of captured data, segmented plane, and fitted box.

* Configurable parameters for calibration and measurement via command-line arguments.

## Prerequisites

* **Python 3.8+**

* **OS**: Windows 10/11, macOS, or Linux

* **USB-C port** (adapter if needed) for the Tau LiDAR Camera

* **Checkerboard** pattern printed (default: 7×6 inner corners, 25 mm squares)

## Installation

1. **Clone the repository** (if you haven't already)


git clone https://github.com/gtuck/box-measurement-ta-l10.git Or your fork/local copy
cd box-measurement-ta-l10


2. **Create & activate a virtual environment** (recommended)


python3 -m venv venv

macOS/Linux
source venv/bin/activate

Windows
venv\Scripts\activate


3. **Install dependencies**


pip install --upgrade pip
pip install -r requirements.txt


The `requirements.txt` should include:


tau-lidar-camera
open3d
numpy
opencv-python


## Usage

The main script for this project is `improved_box_measurement.py`. If you have renamed this file, please use your custom name in the example commands below. The script utilizes command-line arguments to specify different actions such as calibration or measurement.

**General Help:**


python improved_box_measurement.py -h


### 1. Intrinsic Calibration

Capture camera intrinsics using a checkerboard. This saves the calibration data to a file (default: `camera_intrinsics.npz`).


python improved_box_measurement.py calibrate_intrinsics --output camera_intrinsics.npz --dims 7,6 --size 0.025 --frames 20


* `--output`: File to save intrinsic parameters (default: `camera_intrinsics.npz`).

* `--dims`: Inner corners of the checkerboard (cols,rows, default: "7,6").

* `--size`: Size of a checkerboard square in meters (default: 0.025).

* `--frames`: Number of frames to capture (default: 20).

* Follow on-screen instructions: displays a live grayscale feed. Press 'c' to capture when the checkerboard is detected. Collect frames by moving the board.

### 2. Depth-Scale Calibration

Compute a scale factor for accurate depth readings. This saves the scale factor to a file (default: `depth_scale.json`).


python improved_box_measurement.py calibrate_depth --output depth_scale.json --distance 1.0 --samples 50


* `--output`: File to save the depth scale factor (default: `depth_scale.json`).

* `--distance`: Known distance in meters from the camera to a flat surface (default: 1.0).

* `--samples`: Number of depth samples to take (default: 50).

* Follow on-screen instructions: point the camera at a flat surface exactly at the specified `--distance`. Press 's' to start sampling.

### 3. Measuring Boxes

Run the measurement routine using previously saved calibration data.


python improved_box_measurement.py measure --intrinsics camera_intrinsics.npz --depth_scale depth_scale.json


* `--intrinsics`: Path to the saved intrinsic calibration file (default: `camera_intrinsics.npz`).

* `--depth_scale`: Path to the saved depth scale file (default: `depth_scale.json`).

* You can also adjust point cloud processing parameters:

  * `--plane_thresh`: Distance threshold for plane segmentation (default: 0.01m).

  * `--cluster_eps`: DBSCAN epsilon for clustering (default: 0.02m).

  * `--cluster_min_points`: DBSCAN min_points for clustering (default: 100).

* Prints width, height, and depth of the largest detected box in millimeters.

* Opens a 3D window showing the point cloud, segmented plane, and the fitted bounding box.

### 4. Full Run (Calibrate All then Measure)

Perform both calibration steps and then immediately measure a box.


python improved_box_measurement.py full_run --intrinsics_output camera_intrinsics.npz --depth_scale_output depth_scale.json


* This command accepts all parameters from `calibrate_intrinsics`, `calibrate_depth`, and `measure` (use `-h` for details). For example:


python improved_box_measurement.py full_run --dims 7,6 --size 0.025 --frames 15 --known_distance 0.75 --samples 30 --plane_thresh 0.015


## Troubleshooting & Tips

* **Camera Connection**: Ensure the Tau LiDAR camera is properly connected before running the script. If you see "Error initializing Tau Camera", check the USB connection and drivers.

* **Lighting**: Use even, diffuse light for intrinsic calibration to improve checkerboard corner detection.

* **Stability**: Keep the camera and targets (checkerboard, flat surface) steady during captures.

* **Clutter**: A clean, uncluttered tabletop or background yields better segmentation for box measurement.

* **Calibration Accuracy**:

* For intrinsic calibration, a low reprojection error (printed at the end) indicates good calibration. If the error is high (>1.0 pixels), re-calibrate with more varied checkerboard poses.

* For depth scale calibration, a low standard deviation of scale samples (printed at the end) is desirable. If high, ensure the target surface is flat, static, and at the correct distance.

* **Measurement Parameters**: If box measurement is not working well (e.g., not finding the box, or including parts of the table), try adjusting the following parameters for the `measure` command (or the equivalent parameters if using `full_run`):

* `--plane_thresh`: If the table is not being removed properly, try slightly increasing or decreasing this value.

* `--cluster_eps`: If the box points are not being grouped together or too much noise is included, adjust this. Smaller values make clusters tighter.

* `--cluster_min_points`: Minimum points to form a cluster. If your box is small or far away, you might need to decrease this.

## Contributing

Pull Requests (PRs) are welcome for:

* GUI front-ends (e.g., using PySimpleGUI, Qt, or a web interface).

* Batch-mode processing or automation scripts for measuring multiple boxes.

* Enhanced filtering, segmentation, or box-fitting algorithms.

* More robust error handling and user feedback.

* Support for different camera models or calibration patterns.
