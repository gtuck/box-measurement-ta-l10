# Box Measurement TA-L10

A Python application to calibrate your Tau LiDAR Camera TA-L10 and measure shipping-box dimensions with millimeter accuracy. This version features a Graphical User Interface (GUI), enhanced command-line handling, and improved saving and loading of calibration data.

## Features

* **Graphical User Interface (GUI)** for intuitive operation.
* One-time intrinsic & depth-scale calibration with data persistence.
* Load existing calibration data for quick measurements.
* Real-time 3D point-cloud segmentation (visualization primarily in CLI mode, GUI shows 2D feeds).
* Oriented bounding-box measurement.
* 3D visualization of captured data, segmented plane, and fitted box (primarily when using CLI, GUI provides logs and final dimensions).
* Configurable parameters for calibration and measurement via GUI or command-line arguments.

## Prerequisites

* **Python 3.8+**
* **OS**: Windows 10/11, macOS, or Linux
* **USB-C port** (adapter if needed) for the Tau LiDAR Camera
* **Checkerboard** pattern printed (default: 7×6 inner corners, 25 mm squares)

## Installation

1.  **Clone the repository** (if you haven't already)

    ```bash
    git clone https://github.com/gtuck/box-measurement-ta-l10.git # Or your fork/local copy
    cd box-measurement-ta-l10
    ```

2.  **Create & activate a virtual environment** (recommended)

    ```bash
    python3 -m venv venv
    ```
    *macOS/Linux*
    ```bash
    source venv/bin/activate
    ```
    *Windows*
    ```bash
    venv\Scripts\activate
    ```

3.  **Install dependencies**

    First, ensure `pip` is up-to-date:
    ```bash
    python -m pip install --upgrade pip
    ```
    Then, install the project requirements.
    ```bash
    pip install -r requirements.txt
    ```
    Your `requirements.txt` file should include the following packages:
    ```
    numpy
    opencv-python
    TauLidarCamera
    open3d
    PySimpleGUI
    ```
    If your `requirements.txt` has a different name for the Tau camera package (e.g., `tau-lidar-camera`), please update it to `TauLidarCamera` or ensure it matches the correct installable package name.

## Running the Application with GUI

The application now features a graphical user interface for easier operation.

**To launch the GUI:**
```bash
python gui.py
```
or, if `python` defaults to Python 2 on your system:
```bash
python3 gui.py
```

**GUI Overview:**

*   **Camera Connection**: Connect to and disconnect from your Tau LiDAR Camera. You can specify a port or let the application scan for available cameras.
*   **Parameter Input**: All parameters for calibration and measurement are organized into tabs ("Intrinsics Calib.", "Depth Scale Calib.", "Measurement Params") and can be set directly in the GUI.
*   **Operations**: Buttons are available to trigger:
    *   **Calibrate Intrinsics**: Calibrates the camera's internal parameters. The GUI will show a live feed, and the process will auto-capture frames when a checkerboard is detected.
    *   **Calibrate Depth Scale**: Calibrates the camera's depth measurement. The GUI shows a live depth feed. Press "Start Sampling" when you have aimed the camera at a flat surface at the known distance.
    *   **Measure Box**: Measures a box using previously saved (or just run) calibration data.
    *   **Full Run**: Sequentially performs intrinsic calibration, depth scale calibration, and then a box measurement.
*   **Live Feed**: An image area displays the live camera feed (intensity or depth) during calibration processes.
*   **Log Output**: A detailed log area shows progress, messages from the camera, errors, and results.

Follow the on-screen instructions and messages in the log area for guidance during operations.

## Command-Line Interface (CLI) Usage

For most users, the GUI (run via `python gui.py`) is recommended. The command-line interface is available for advanced users or scripting.

The main script for CLI operations is `measureBox.py`. (Note: the original README might have referred to `improved_box_measurement.py`; ensure you use the correct current script name if it differs).

**General Help:**
```bash
python measureBox.py -h
```

### 1. Intrinsic Calibration (CLI)

Capture camera intrinsics using a checkerboard. This saves the calibration data to a file (default: `camera_intrinsics.npz`).
```bash
python measureBox.py calibrate_intrinsics --output camera_intrinsics.npz --dims 7,6 --size 0.025 --frames 20
```
*   `--output`: File to save intrinsic parameters (default: `camera_intrinsics.npz`).
*   `--dims`: Inner corners of the checkerboard (cols, rows, default: "7,6").
*   `--size`: Size of a checkerboard square in meters (default: 0.025).
*   `--frames`: Number of frames to capture (default: 20).
*   `--port`: (Optional) Specify the serial port.
*   The CLI version for intrinsic calibration will auto-capture frames when a checkerboard is detected in the center of the view.

### 2. Depth-Scale Calibration (CLI)

Compute a scale factor for accurate depth readings. Saves to file (default: `depth_scale.json`).
```bash
python measureBox.py calibrate_depth --output depth_scale.json --distance 1.0 --samples 50
```
*   `--output`: File to save the depth scale factor (default: `depth_scale.json`).
*   `--distance`: Known distance in meters to a flat surface (default: 1.0).
*   `--samples`: Number of depth samples (default: 50).
*   `--port`: (Optional) Specify the serial port.
*   The CLI version for depth scale calibration will start sampling automatically after displaying the depth feed for a moment.

### 3. Measuring Boxes (CLI)

Run measurement using saved calibration data.
```bash
python measureBox.py measure --intrinsics camera_intrinsics.npz --depth_scale depth_scale.json
```
*   `--intrinsics`: Path to intrinsics file (default: `camera_intrinsics.npz`).
*   `--depth_scale`: Path to depth scale file (default: `depth_scale.json`).
*   `--port`: (Optional) Specify the serial port.
*   Adjust point cloud parameters: `--plane_thresh`, `--cluster_eps`, `--cluster_min_points`.
*   Prints dimensions and opens an Open3D window for visualization.

### 4. Full Run (CLI)

Perform all steps: intrinsic calibration, depth scale calibration, then measurement.
```bash
python measureBox.py full_run --intrinsics_output camera_intrinsics.npz --depth_scale_output depth_scale.json
```
*   Accepts all parameters from the individual steps (use `-h` for details). Example:
    ```bash
    python measureBox.py full_run --dims 7,6 --size 0.025 --frames 15 --known_distance 0.75 --samples 30 --plane_thresh 0.015 --port /dev/ttyUSB0
    ```

## Troubleshooting & Tips

*   **Package Installation**: If `pip install -r requirements.txt` fails for `TauLidarCamera` or `PySimpleGUI`, ensure your Python version is compatible and consider installing problematic packages individually. Check their respective documentation for specific OS or Python version needs. `TauLidarCamera` often depends on `TauLidarCommon`.
*   **Camera Connection**: If "No Tau Camera devices found" or similar errors occur:
    *   Check physical USB connection.
    *   Ensure drivers (if any) are installed.
    *   In the GUI, try "scan" or specify the correct port (e.g., COM3 on Windows, /dev/ttyUSB0 or /dev/ttyACM0 on Linux).
    *   In CLI, use the `--port` argument.
*   **Lighting**: Good, even lighting is crucial for intrinsic calibration (checkerboard detection). Avoid glare and shadows.
*   **Stability**: Keep the camera and calibration targets (checkerboard, flat surface for depth) very still during captures/sampling.
*   **Clutter**: For box measurement, a clear background helps segment the box from other objects.
*   **Calibration Accuracy**:
    *   *Intrinsics*: A low reprojection error (logged at the end, ideally < 1.0) indicates good calibration. If high, use more varied checkerboard poses and ensure good focus/lighting.
    *   *Depth Scale*: A low standard deviation of scale samples is good. Ensure the target surface is flat, perpendicular to the camera, and at the accurately measured known distance.
*   **Measurement Issues**: If box dimensions are incorrect or the box isn't found:
    *   Verify calibration files are correct and loaded.
    *   Adjust GUI/CLI parameters: `--plane_thresh`, `--cluster_eps`, `--cluster_min_points`.
*   **Open3D Visualization**: The 3D visualizations are primarily launched by the CLI mode (`measureBox.py`). The GUI provides logs and numerical results; its image element shows 2D feeds during calibration.

## Contributing

Pull Requests (PRs) are welcome for:
*   Further GUI enhancements.
*   Batch processing for multiple boxes.
*   Advanced filtering or segmentation algorithms.
*   Support for other camera models or calibration patterns.
