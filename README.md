# Box Measurement TA-L10

A Python application to calibrate your Tau LiDAR Camera TA-L10 and measure shipping-box dimensions with millimeter accuracy.

## Features

* One-time intrinsic & depth-scale calibration
* Real-time 3D point-cloud segmentation
* Oriented bounding-box measurement
* 3D visualization of captured data

## Prerequisites

* **Python 3.8+**
* **OS**: Windows 10/11, macOS, or Linux
* **USB-C port** (adapter if needed)
* **Checkerboard pattern** printed (7×6 inner corners, 25 mm squares)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/gtuck/box-measurement-ta-l10.git
   cd box-measurement-ta-l10
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Calibration

### 1. Intrinsic Calibration

Capture camera intrinsics using a checkerboard:

```bash
python box_measurement_with_calibration.py --calibrate-intrinsics
```

* Displays a live grayscale feed.
* When the checkerboard is detected, captures a frame.
* Collects 20 frames by moving the board.
* Prints & saves camera matrix and distortion coefficients.

### 2. Depth-Scale Calibration

Compute a scale factor for accurate depth readings:

```bash
python box_measurement_with_calibration.py --calibrate-depth-scale --distance 1.0
```

* Point the camera at a flat surface exactly 1 m away.
* Samples the central region 50 times (default).
* Prints & saves your depth-scale factor.

## Measuring Boxes

Run the measurement routine:

```bash
python box_measurement_with_calibration.py --measure-box --scale-factor <YOUR_SCALE>
```

* Prints width, height, and depth in millimeters.
* Opens a 3D window showing the fitted bounding box.

## Troubleshooting & Tips

* **Lighting**: Use even, diffuse light to improve corner detection.
* **Stability**: Keep the camera and targets steady during captures.
* **Clutter**: A clean tabletop yields better segmentation.
* **Advanced Tuning**: Adjust RANSAC (`--plane-thresh`) or clustering (`--eps`, `--min-points`) via flags.

## Contributing

Welcome PRs for:

* GUI front-ends (PySimpleGUI, Qt)
* Batch-mode or automation scripts
* Enhanced filtering or segmentation algorithms
