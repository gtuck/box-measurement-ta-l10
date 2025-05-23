import numpy as np
import cv2
from TauLidarCamera.camera import Camera 
from TauLidarCommon.frame import FrameType 
import open3d as o3d
import argparse
import os
import json
import logging

# ----------------------------------
# Calibration File Constants
# ----------------------------------
DEFAULT_INTRINSICS_FILE = 'camera_intrinsics.npz'
DEFAULT_DEPTH_SCALE_FILE = 'depth_scale.json'

# ----------------------------------
# Camera Setup Helper
# ----------------------------------
def setup_camera(port_arg=None):
    """
    Scans for available Tau Camera devices, connects to one, and configures it.
    - port_arg: Specific port to connect to. If None, scans all.
    Returns: Tuple (Configured Camera object or None, list of messages).
    """
    camera = None
    selected_port = port_arg
    messages = []

    if selected_port is None:
        try:
            ports = Camera.scan()
            msg = f'Found {len(ports)} possible device(s)'
            messages.append(msg)
            logging.info(msg)
            if not ports:
                msg = "No Tau Camera devices found. Please check connection."
                messages.append(msg)
                logging.warning(msg)
                return None, messages
            selected_port = ports[0]
            msg = f"Attempting to connect to the first available device on port '{selected_port}'"
            messages.append(msg)
            logging.info(msg)
        except Exception as e:
            msg = f"Error scanning for cameras: {e}"
            messages.append(msg)
            logging.error(msg)
            return None, messages
    else:
        msg = f"Attempting to connect to device on specified port '{selected_port}'"
        messages.append(msg)
        logging.info(msg)

    try:
        camera = Camera.open(selected_port)
        if camera:
            # Default configurations, adjust if needed
            camera.setModulationChannel(0)
            camera.setIntegrationTime3d(0, 1000)
            camera.setMinimalAmplitude(0, 10)

            camera_info = camera.info()
            msg = "ToF camera opened successfully:"
            messages.append(msg)
            logging.info(msg)
            for key_val in [
                ("model", camera_info.model),
                ("firmware", camera_info.firmware),
                ("uid", camera_info.uid),
                ("resolution", camera_info.resolution),
                ("port", camera_info.port)
            ]:
                msg = f"    {key_val[0]}:      {key_val[1]}"
                messages.append(msg)
                logging.info(msg)
            return camera, messages
        else:
            msg = f"Failed to open camera on port '{selected_port}'."
            messages.append(msg)
            logging.warning(msg)
            return None, messages
    except Exception as e:
        msg = f"Error during camera setup on port '{selected_port}': {e}"
        messages.append(msg)
        logging.error(msg)
        return None, messages

def get_intensity_image(camera_obj):
    """
    Reads an intensity (grayscale) image from the camera.
    Uses FrameType.DISTANCE_GRAYSCALE and processes frame.data_grayscale.
    Returns: Tuple (image_data_numpy_array or None, error_message or None)
    """
    if not camera_obj:
        logging.warning("get_intensity_image: camera_obj is None.")
        return None, "Camera object is None."
    try:
        logging.debug("Attempting to read FrameType.DISTANCE_GRAYSCALE for intensity data...")
        frame = camera_obj.readFrame(FrameType.DISTANCE_GRAYSCALE)
        if frame and hasattr(frame, 'data_grayscale'):
            res_str = camera_obj.info().resolution
            frame_width, frame_height = map(int, res_str.split('x'))

            mat_grayscale_uint16 = np.frombuffer(frame.data_grayscale, dtype=np.uint16, count=-1, offset=0).reshape(frame_height, frame_width)

            if np.max(mat_grayscale_uint16) > 0:
                 mat_grayscale_normalized = (mat_grayscale_uint16 / np.max(mat_grayscale_uint16) * 255)
            else:
                 mat_grayscale_normalized = mat_grayscale_uint16

            mat_grayscale_uint8 = mat_grayscale_normalized.astype(np.uint8)
            logging.debug("Successfully read and processed intensity frame.")
            return mat_grayscale_uint8, None
        else:
            err_msg = ""
            if frame is None:
                err_msg = "Failed to read DISTANCE_GRAYSCALE frame (frame object is None)."
                logging.warning(err_msg)
            else: # frame exists but no data_grayscale
                err_msg = "DISTANCE_GRAYSCALE frame read, but 'data_grayscale' attribute is missing."
                logging.warning(err_msg)
            return None, err_msg
    except AttributeError as ae: # This can happen if FrameType is not supported by firmware etc.
        err_msg = f"AttributeError: FrameType 'DISTANCE_GRAYSCALE' may not be valid or an attribute is missing. Details: {ae}"
        logging.error(err_msg)
        return None, err_msg
    except Exception as e:
        err_msg = f"General error reading or processing intensity frame: {e}"
        logging.error(err_msg)
        return None, err_msg

def get_raw_depth_image(camera_obj):
    """
    Reads a raw depth image from the camera.
    Uses FrameType.DISTANCE and processes frame.data.
    Returns: Tuple (image_data_numpy_array or None, error_message or None)
    """
    if not camera_obj:
        logging.warning("get_raw_depth_image: camera_obj is None.")
        return None, "Camera object is None."
    try:
        logging.debug("Attempting to read FrameType.DISTANCE for raw depth data...")
        frame = camera_obj.readFrame(FrameType.DISTANCE)
        if frame and hasattr(frame, 'data'):
            res_str = camera_obj.info().resolution
            frame_width, frame_height = map(int, res_str.split('x'))

            raw_depth_uint16 = np.frombuffer(frame.data, dtype=np.uint16, count=-1, offset=0).reshape(frame_height, frame_width)
            logging.debug("Successfully read raw depth frame.")
            return raw_depth_uint16, None
        else:
            err_msg = ""
            if frame is None:
                err_msg = "Failed to read DISTANCE frame (frame object is None)."
                logging.warning(err_msg)
            else: # frame exists but no data
                err_msg = "DISTANCE frame read, but 'data' attribute is missing or frame is invalid."
                logging.warning(err_msg)
            return None, err_msg
    except AttributeError as ae: # This can happen if FrameType is not supported by firmware etc.
        err_msg = f"AttributeError: FrameType 'DISTANCE' may not be valid or an attribute is missing. Details: {ae}"
        logging.error(err_msg)
        return None, err_msg
    except Exception as e:
        err_msg = f"General error reading or processing raw depth frame: {e}"
        logging.error(err_msg)
        return None, err_msg

# ----------------------------------
# Calibration Functions
# ----------------------------------
def calibrate_intrinsics(camera_obj, output_file, checkerboard_dims=(7, 6), square_size=0.025, num_frames=20, gui_window=None, image_key=None, capture_event_trigger=None, log_callback=None):
    messages = []
    def _log(message):
        if log_callback: log_callback(message)
        else: logging.info(message) # Default to logging if no callback
        messages.append(message)

    if not camera_obj:
        _log("Camera not initialized. Cannot perform intrinsic calibration.")
        return None, None, messages
        
    _log(f"Starting intrinsic calibration. Looking for {checkerboard_dims} inner corners. Need {num_frames} frames.")

    objp = np.zeros((checkerboard_dims[1] * checkerboard_dims[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []
    collected_count = 0
    last_gray_shape = None
    
    # This function will run a loop trying to collect num_frames.
    # In a GUI context, this function would ideally be called repeatedly for each frame,
    # or run in a thread. For now, it's a blocking call that auto-captures.
    # The `capture_event_trigger` is a placeholder for a more complex GUI interaction.
    # For this refactoring, if `gui_window` is present, we assume auto-capture if checkerboard is found.

    for frame_attempt in range(num_frames * 5): # Try more times than num_frames
        if collected_count >= num_frames:
            break

        gray, err_msg = get_intensity_image(camera_obj)
        
        if err_msg or gray is None:
            _log(f"Frame attempt {frame_attempt+1}: Failed to get intensity image. {err_msg if err_msg else 'No image data.'}")
            if gui_window and image_key: # Show a black screen or an error image
                # Use last known shape or a default
                h_shape = last_gray_shape[0] if last_gray_shape else 480
                w_shape = last_gray_shape[1] if last_gray_shape else 640
                black_img = np.zeros((h_shape, w_shape), dtype=np.uint8) 
                imgbytes = cv2.imencode('.png', black_img)[1].tobytes()
                try: gui_window[image_key].update(data=imgbytes)
                except Exception as e_gui: _log(f"GUI update error (black screen): {e_gui}")
            # In a real GUI, we wouldn't just continue rapidly. This needs a delay or external trigger.
            # For this blocking version, a small pause if in GUI mode to allow some visual feedback.
            if gui_window: cv2.waitKey(1) # Minimal wait to allow GUI to refresh if it can
            continue
        
        last_gray_shape = gray.shape
        display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        found, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)
        
        capture_this_frame = False
        if found:
            cv2.drawChessboardCorners(display_img, checkerboard_dims, corners, found)
            text = f"Found! ({collected_count}/{num_frames}). "
            if gui_window : text += "Auto-capturing." # Simplified: auto-capture if GUI
            else: text += "Press 'c' (CLI mode - not really, just for info)"
            cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if gui_window: # Auto-capture for GUI mode for this simplified refactor
                capture_this_frame = True
            # elif capture_event_trigger and capture_event_trigger(): # Ideal GUI check
            #    capture_this_frame = True
            # elif not gui_window: # CLI mode might have a manual capture trigger (not implemented here)
            #    pass


        else:
            cv2.putText(display_img, f"Aim at checkerboard... ({collected_count}/{num_frames})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if gui_window and image_key:
            try:
                imgbytes = cv2.imencode('.png', display_img)[1].tobytes()
                gui_window[image_key].update(data=imgbytes)
                cv2.waitKey(1) # Allow GUI to process update
            except Exception as e_gui:
                _log(f"Error updating GUI image: {e_gui}")
        
        if found and capture_this_frame: # Or based on a GUI flag
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
            collected_count += 1
            _log(f"Captured calibration frame {collected_count}/{num_frames}")
            # Brief "captured" message on image for GUI
            if gui_window and image_key:
                cv2.putText(display_img, f"FRAME {collected_count} CAPTURED!", 
                            (display_img.shape[1]//2 - 150, display_img.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
                try:
                    imgbytes = cv2.imencode('.png', display_img)[1].tobytes()
                    gui_window[image_key].update(data=imgbytes)
                    cv2.waitKey(1) # Show "CAPTURED" message briefly
                except Exception as e_gui: _log(f"GUI update error (captured msg): {e_gui}")
        
        if collected_count >= num_frames:
            _log(f"Collected {num_frames} frames. Proceeding to calibration.")
            break
        # If not gui_window, this loop runs very fast. Add a small delay for CLI if needed.
        # if not gui_window: time.sleep(0.1) # Requires import time

    if collected_count == 0:
        _log("No frames collected. Intrinsic calibration cannot proceed.")
        return None, None, messages
    if collected_count < num_frames:
         _log(f"Warning: Only {collected_count} frames collected, expected {num_frames}. Calibration might be suboptimal.")
    
    if not objpoints or not imgpoints or last_gray_shape is None:
        _log("Not enough points or valid image shape for calibration.")
        return None, None, messages
        
    _log("Calibrating camera...")
    mtx, dist = None, None
    try:
        # Ensure last_gray_shape is in (height, width) order for calibrateCamera
        h, w = last_gray_shape[:2]
        ret, mtx_calc, dist_calc, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w,h), None, None # OpenCV expects (width, height)
        )
        if not ret:
            _log("Intrinsic calibration failed (calibrateCamera returned False). Check your setup and images.")
            return None, None, messages
        mtx, dist = mtx_calc, dist_calc
    except cv2.error as e:
        _log(f"OpenCV Error during calibration: {e}")
        return None, None, messages
    except Exception as e_gen:
        _log(f"Unexpected error during cv2.calibrateCamera: {e_gen}")
        return None, None, messages

    _log("Intrinsic calibration successful.")
    _log(f"Camera matrix (mtx):\n{mtx}")
    _log(f"Distortion coefficients (dist):\n{dist.ravel()}")

    try:
        np.savez(output_file, mtx=mtx, dist=dist)
        _log(f"Intrinsic calibration data saved to {output_file}")
    except Exception e:
        _log(f"Error saving intrinsic calibration data: {e}")

    mean_error = 0
    if objpoints and rvecs and tvecs and mtx is not None and dist is not None:
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        total_reprojection_error = mean_error / len(objpoints)
        _log(f"Total reprojection error: {total_reprojection_error}")
        if total_reprojection_error > 1.0: # Threshold for good calibration
            _log("Warning: Reprojection error is high. Calibration might be inaccurate.")
    else:
        _log("Could not calculate reprojection error due to missing calibration products (rvecs, tvecs).")


    return mtx, dist, messages

def calibrate_depth_scale(camera_obj, output_file, known_distance=1.0, num_samples=50, gui_window=None, image_key=None, start_sampling_event_key=None, log_callback=None):
    messages = []
    def _log(message):
        if log_callback: log_callback(message)
        else: logging.info(message)
        messages.append(message)

    if not camera_obj:
        _log("Camera not initialized. Cannot perform depth scale calibration.")
        return None, messages

    _log(f"Starting depth scale calibration. Known distance: {known_distance}m. Samples to collect: {num_samples}")
        
    scales = []
    # Simplified: if gui_window is present, auto-starts sampling. 
    # `start_sampling_event_key` is a placeholder for future GUI interaction.
    sampling_active = bool(gui_window) or not bool(gui_window) # Auto-start for this version
    
    last_depth_shape = None

    for sample_attempt in range(num_samples * 3): # Try more times
        if len(scales) >= num_samples:
            break

        if not sampling_active and gui_window: # If GUI controlled start needed in future
            # Update GUI with a message "Waiting to start..."
            # For now, this path is not taken due to auto-start.
            if gui_window and image_key:
                # Placeholder: show message on image
                h_shape = last_depth_shape[0] if last_depth_shape else 480
                w_shape = last_depth_shape[1] if last_depth_shape else 640
                wait_img = np.zeros((h_shape, w_shape, 3), dtype=np.uint8)
                cv2.putText(wait_img, "Waiting to start sampling...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                imgbytes = cv2.imencode('.png', wait_img)[1].tobytes()
                try: gui_window[image_key].update(data=imgbytes); cv2.waitKey(1)
                except Exception as e_gui: _log(f"GUI error (wait_img): {e_gui}")
            continue


        raw_depth, err_msg = get_raw_depth_image(camera_obj) 
        
        if err_msg or raw_depth is None:
            _log(f"Sample attempt {sample_attempt+1}: Failed to read depth frame. {err_msg if err_msg else 'No depth data.'}")
            if gui_window and image_key:
                h_shape = last_depth_shape[0] if last_depth_shape else 480
                w_shape = last_depth_shape[1] if last_depth_shape else 640
                black_img = np.zeros((h_shape, w_shape, 3), dtype=np.uint8) # Color for depth
                imgbytes = cv2.imencode('.png', black_img)[1].tobytes()
                try: gui_window[image_key].update(data=imgbytes); cv2.waitKey(1)
                except Exception as e_gui: _log(f"GUI error (black_img depth): {e_gui}")
            continue
        
        last_depth_shape = raw_depth.shape
        depth_display_normalized = cv2.normalize(raw_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display_color = cv2.applyColorMap(depth_display_normalized, cv2.COLORMAP_JET)
        
        h, w = raw_depth.shape
        roi_size = 20 # pixels for ROI
        cv2.rectangle(depth_display_color, (w//2-roi_size//2, h//2-roi_size//2), 
                      (w//2+roi_size//2, h//2+roi_size//2), (0,255,0), 2) # Green ROI box

        if sampling_active:
            cv2.putText(depth_display_color, f"Sampling... {len(scales)}/{num_samples}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Process sample
            # ROI from center of raw_depth (not display image)
            depth_roi_raw = raw_depth[h//2-roi_size//2:h//2+roi_size//2, w//2-roi_size//2:w//2+roi_size//2]
            depth_m_at_center = depth_roi_raw.astype(np.float32) * 0.001 # Assuming depth is in mm
            
            valid_depths = depth_m_at_center[depth_m_at_center > 0.1] # Filter out very close/invalid
            
            if valid_depths.size > 0:
                measured_distance = np.median(valid_depths)
                if measured_distance > 0.01: # Ensure it's a somewhat sensible distance
                    current_scale = known_distance / measured_distance
                    scales.append(current_scale)
                    _log(f"Sample {len(scales)}/{num_samples}: Measured Median Distance (ROI) = {measured_distance:.3f}m, Scale = {current_scale:.4f}")
                else:
                    _log(f"Sample attempt {len(scales)+1}/{num_samples}: Warning - Median measured distance in ROI is zero or too small ({measured_distance:.3f}m).")
            else:
                _log(f"Sample attempt {len(scales)+1}/{num_samples}: Warning - No valid depth data in ROI.")
        else: # Should not happen with current auto-start logic
             cv2.putText(depth_display_color, f"Aim at target {known_distance:.2f}m away. Ready to start.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        if gui_window and image_key:
            try:
                imgbytes = cv2.imencode('.png', depth_display_color)[1].tobytes()
                gui_window[image_key].update(data=imgbytes)
                cv2.waitKey(1) # Allow GUI to process update
            except Exception as e_gui:
                _log(f"Error updating GUI image for depth scale: {e_gui}")
        
        if len(scales) >= num_samples:
            _log("Required depth samples collected.")
            break
        # if not gui_window: time.sleep(0.05) # Small delay for CLI

    if not scales:
        _log("No depth samples collected. Cannot calculate scale factor.")
        if gui_window and image_key: # Update GUI with error message
            h_shape = last_depth_shape[0] if last_depth_shape else 480
            w_shape = last_depth_shape[1] if last_depth_shape else 640
            err_img = np.zeros((h_shape, w_shape, 3), dtype=np.uint8)
            cv2.putText(err_img, "No samples collected!", (20,h_shape//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            imgbytes = cv2.imencode('.png', err_img)[1].tobytes()
            try: gui_window[image_key].update(data=imgbytes); cv2.waitKey(1)
            except Exception as e_gui: _log(f"GUI error (no samples img): {e_gui}")
        return None, messages

    scale_factor = float(np.mean(scales))
    std_dev_scale = float(np.std(scales))
    _log(f"Depth scale factor calculated: {scale_factor:.5f}")
    _log(f"Standard deviation of scale samples: {std_dev_scale:.5f}")
    if std_dev_scale > 0.05 * abs(scale_factor): # abs in case scale factor is weirdly negative
        _log("Warning: High standard deviation in depth scale samples. Measurement might be inconsistent.")
        _log("Ensure the target surface is flat, perpendicular to the camera, and at the correct distance.")

    try:
        with open(output_file, 'w') as f:
            json.dump({'depth_scale_factor': scale_factor, 
                       'known_distance_meters': known_distance, 
                       'num_samples': len(scales), 
                       'std_dev_scale': std_dev_scale}, f, indent=4)
        _log(f"Depth scale factor saved to {output_file}")
    except Exception as e:
        _log(f"Error saving depth scale factor: {e}")
        # scale_factor is still valid, so don't return None here unless saving is critical
        
    return scale_factor, messages

# ----------------------------------
# Point-Cloud Processing & Measurement
# ----------------------------------
def depth_to_pointcloud(depth_m, camera_matrix):
    # No changes to print, should be logging.warning as per instruction
    h, w = depth_m.shape
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)
    valid = depth_m > 0.1 # Filter out points too close, likely invalid

    z = depth_m[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    pts = np.stack((x, y, z), axis=-1)

    if pts.size == 0:
        logging.warning("Warning: No valid points found to create point cloud.")
        return o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def segment_plane_and_objects(pcd, dist_thresh=0.01, ransac_n=3, num_iter=1000):
    if not pcd.has_points():
        logging.warning("Warning: Empty point cloud passed to segment_plane_and_objects.")
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=ransac_n, num_iterations=num_iter)
    except Exception as e:
        logging.error(f"Error during plane segmentation: {e}")
        return o3d.geometry.PointCloud(), pcd # Return original as objects if error
    if not inliers: # Check if inliers list is empty
        logging.warning("Warning: No plane found (inliers list is empty). Returning original point cloud as objects.")
        return o3d.geometry.PointCloud(), pcd
    plane_pcd = pcd.select_by_index(inliers)
    objects_pcd = pcd.select_by_index(inliers, invert=True)
    plane_pcd.paint_uniform_color([0.7, 0.7, 0.7]) # Gray for plane
    logging.info(f"Plane segmented with {len(inliers)} inliers. Objects have {len(objects_pcd.points)} points.")
    return plane_pcd, objects_pcd

def cluster_and_fit_box(objects_pcd, eps=0.02, min_points=100):
    if not objects_pcd.has_points():
        logging.info("No object points to cluster.")
        return None
    try:
        # Set print_progress to False as it prints to stdout
        labels = np.array(objects_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except Exception as e:
        logging.error(f"Error during DBSCAN clustering: {e}")
        return None

    if labels.size == 0 or np.all(labels < 0) : # All points are noise or no points
        logging.warning("No clusters found or all points are noise.")
        if objects_pcd.has_points(): # Check if there were points to begin with
            logging.info("Attempting to fit OBB to all object points as a fallback since no clusters were found.")
            try:
                obb = objects_pcd.get_oriented_bounding_box()
                obb.color = (0, 1, 0) # Green for fallback OBB
                return obb
            except Exception as e_obb:
                logging.error(f"Error fitting OBB to all object points: {e_obb}")
                return None
        return None # No points, no OBB

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True) # Consider only non-noise labels
    if unique_labels.size == 0:
        logging.warning("No valid clusters found (all points considered noise after filtering).")
        return None

    largest_cluster_label = unique_labels[counts.argmax()]
    logging.info(f"Largest cluster label: {largest_cluster_label} with {counts.max()} points.")
    cluster_indices = np.where(labels == largest_cluster_label)[0]
    largest_cluster_pcd = objects_pcd.select_by_index(cluster_indices)

    if not largest_cluster_pcd.has_points():
        logging.warning("Largest cluster is empty after selection (should not happen if logic is correct).")
        return None
    try:
        obb = largest_cluster_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  # Red for largest cluster's OBB
    except Exception as e:
        logging.error(f"Error getting OBB for the largest cluster: {e}")
        return None
    logging.info(f"OBB fitted to largest cluster. Extent: {obb.extent}, Center: {obb.center}")
    return obb

def capture_frames_for_measurement(camera_obj, log_callback=None):
    """ 
    Captures intensity and raw depth frame for measurement.
    Returns: (intensity_image, raw_depth_image, messages_list)
    """
    _messages = []
    def _log_local(message):
        if log_callback: log_callback(message)
        else: logging.info(message) # Fallback to standard logging
        _messages.append(message)

    if not camera_obj:
        _log_local("capture_frames_for_measurement: Camera object is None.")
        return None, None, _messages
    
    intensity_image, intensity_err = get_intensity_image(camera_obj)
    if intensity_err:
        _log_local(f"Intensity capture error: {intensity_err}")
    
    raw_depth_image, depth_err = get_raw_depth_image(camera_obj)
    if depth_err:
        _log_local(f"Depth capture error: {depth_err}")
        
    return intensity_image, raw_depth_image, _messages

def measure_box_dimensions(camera_obj, intrinsics_file_path, depth_scale_file_path, 
                           plane_dist_thresh=0.01, cluster_eps=0.02, cluster_min_points=100, 
                           log_callback=None, gui_window=None, image_key=None): # Added gui_window and image_key
    messages = []
    def _log(message):
        if log_callback: log_callback(message)
        else: logging.info(message)
        messages.append(message)

    mtx, dist, depth_scale_factor = None, None, None

    # Load Intrinsics
    if not os.path.exists(intrinsics_file_path):
        _log(f"Error: Intrinsics file not found: {intrinsics_file_path}")
        return None, messages
    try:
        data = np.load(intrinsics_file_path)
        mtx = data['mtx']
        dist = data['dist']
        _log(f"Loaded intrinsics from {intrinsics_file_path}")
    except Exception as e:
        _log(f"Error loading intrinsics from {intrinsics_file_path}: {e}")
        return None, messages

    # Load Depth Scale
    if not os.path.exists(depth_scale_file_path):
        _log(f"Error: Depth scale file not found: {depth_scale_file_path}")
        return None, messages
    try:
        with open(depth_scale_file_path, 'r') as f:
            scale_data = json.load(f)
            depth_scale_factor = float(scale_data['depth_scale_factor'])
        _log(f"Loaded depth scale factor ({depth_scale_factor:.5f}) from {depth_scale_file_path}")
    except Exception as e:
        _log(f"Error loading depth scale from {depth_scale_file_path}: {e}")
        return None, messages

    if not camera_obj:
        _log("Camera not initialized for measurement.")
        return None, messages
    if mtx is None or dist is None or depth_scale_factor is None:
        _log("Error: Critical calibration data (mtx, dist, or depth_scale_factor) failed to load.")
        return None, messages
    
    _log("Attempting to capture frame for measurement...")
    # Update GUI with live feed if available
    if gui_window and image_key:
        try:
            temp_intensity, _ = get_intensity_image(camera_obj) # Get a frame for display
            if temp_intensity is not None:
                imgbytes = cv2.imencode('.png', cv2.cvtColor(temp_intensity, cv2.COLOR_GRAY2BGR))[1].tobytes()
                gui_window[image_key].update(data=imgbytes)
                gui_window.refresh() # Force update before potentially blocking operations
        except Exception as e_img:
            _log(f"Could not update live image for measurement: {e_img}")

    intensity_img, raw_depth, capture_messages = capture_frames_for_measurement(camera_obj, log_callback)
    messages.extend(capture_messages)

    if raw_depth is None: 
        _log("Failed to capture critical depth frame for measurement. Check camera.")
        return None, messages
    if intensity_img is None: 
        _log("Warning: Failed to capture intensity frame. Proceeding with depth data only for point cloud.")

    # Ensure depth_scale_factor is float
    try:
        current_depth_scale_factor = float(depth_scale_factor)
    except ValueError:
        _log(f"Error: depth_scale_factor '{depth_scale_factor}' is not a valid float.")
        return None, messages

    depth_m = raw_depth.astype(np.float32) * 0.001 * current_depth_scale_factor # raw depth is usually in mm
    
    _log("Generating point cloud...")
    pcd_full = depth_to_pointcloud(depth_m, mtx) # logging.warning is inside this function
    if not pcd_full.has_points():
        _log("Point cloud generation failed or resulted in an empty cloud.")
        return None, messages
    
    # Downsample before segmentation for performance
    pcd_downsampled = pcd_full.voxel_down_sample(voxel_size=0.005) 
    _log(f"Point cloud downsampled from {len(pcd_full.points)} to {len(pcd_downsampled.points)} points.")

    _log(f"Segmenting plane (e.g., table) with distance threshold: {plane_dist_thresh}...")
    # logging inside segment_plane_and_objects
    plane_pcd, objects_pcd = segment_plane_and_objects(pcd_downsampled, dist_thresh=plane_dist_thresh) 
    
    geometries_to_visualize = []
    if pcd_downsampled.has_points(): geometries_to_visualize.append(pcd_downsampled)
    if plane_pcd.has_points(): geometries_to_visualize.append(plane_pcd) # plane_pcd is already colored gray

    if not objects_pcd.has_points():
        _log("No object points found after plane segmentation.")
        if geometries_to_visualize: # Show what we have (e.g. downsampled cloud and plane)
            try: 
                _log("Visualizing: Point cloud and segmented plane (no objects).")
                o3d.visualization.draw_geometries(geometries_to_visualize, window_name="Measurement - No Objects Found", width=800, height=600)
            except Exception as e_vis: _log(f"Open3D visualization error (no objects): {e_vis}")
        return None, messages

    _log(f"Clustering objects (eps={cluster_eps}, min_pts={cluster_min_points}) and fitting bounding box...")
    # logging inside cluster_and_fit_box
    obb = cluster_and_fit_box(objects_pcd, eps=cluster_eps, min_points=cluster_min_points)

    dimensions_dict = None
    if obb is None:
        _log("Could not fit a bounding box to any detected objects.")
        if objects_pcd.has_points(): 
            objects_pcd.paint_uniform_color([0,0,1]) # Blue for unclustered objects
            geometries_to_visualize.append(objects_pcd) # Add remaining object points
        if geometries_to_visualize: # Show what we have
            try: 
                _log("Visualizing: Point cloud, plane, and unclustered objects (no box fitted).")
                o3d.visualization.draw_geometries(geometries_to_visualize, window_name="Measurement - No Box Fitted", width=800, height=600)
            except Exception as e_vis: _log(f"Open3D visualization error (no box): {e_vis}")
        return None, messages

    # obb is an OrientedBoundingBox object
    # extent gives dimensions along its local axes (x,y,z may not align with world L,W,H)
    # For consistency, sort them: Length (largest), Width (middle), Height (smallest)
    dims_m_sorted = np.sort(obb.extent)[::-1] # Sorts in descending order
    
    dimensions_dict = {
        "length_mm": dims_m_sorted[0] * 1000.0,
        "width_mm": dims_m_sorted[1] * 1000.0,
        "height_mm": dims_m_sorted[2] * 1000.0,
        "center_x_m": obb.center[0],
        "center_y_m": obb.center[1],
        "center_z_m": obb.center[2],
    }
    
    _log("\n--- Measured Box Dimensions (Sorted L, W, H based on OBB extent) ---")
    _log(f"  Length (mm): {dimensions_dict['length_mm']:.1f} (Largest extent of OBB)")
    _log(f"  Width (mm): {dimensions_dict['width_mm']:.1f} (Middle extent of OBB)")
    _log(f"  Height (mm): {dimensions_dict['height_mm']:.1f} (Smallest extent of OBB)")
    _log(f"  OBB Center (m): {obb.center[0]:.3f}, {obb.center[1]:.3f}, {obb.center[2]:.3f}")
    _log(f"  OBB Extent (m - unsorted): {obb.extent[0]:.3f}, {obb.extent[1]:.3f}, {obb.extent[2]:.3f}")
    _log("---------------------------------------------------------------------\n")
    
    _log("Displaying 3D visualization. Close the Open3D window to continue.")
    # Add the OBB itself to the visualization list. It's already colored.
    geometries_to_visualize.append(obb) 
    
    # Add a coordinate frame at the OBB center for better orientation understanding
    obb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=obb.center)
    geometries_to_visualize.append(obb_frame)
    
    # Add a world coordinate frame at origin for global reference
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    geometries_to_visualize.append(world_frame)

    try:
        o3d.visualization.draw_geometries(geometries_to_visualize, 
                                          window_name="Box Measurement Visualization", 
                                          width=1024, height=768)
    except Exception as e_vis: 
        _log(f"Open3D visualization error (final): {e_vis}")
        # If visualization fails, still return dimensions if calculated
        
    return dimensions_dict, messages

# ----------------------------------
# Main Execution
# ----------------------------------
def main():
    # Basic logger setup for CLI mode
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    def cli_log_callback(message):
        """Passes messages to print for CLI, could also use basic logging."""
        print(message)

    parser = argparse.ArgumentParser(description="Calibrate Tau LiDAR Camera and measure box dimensions.")
    parser.add_argument('--port', default=None, help='Specify serial port of the Tau LiDAR Camera. If None, scans for cameras.')
    
    # Open3D specific verbosity - maps to Open3D's own levels
    parser.add_argument('--o3d_log_level', default='Warning', 
                        choices=['Debug', 'Info', 'Warning', 'Error', 'Critical'], # Critical for o3d usually means no output
                        help='Set the logging level for Open3D library.')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands.', required=True)

    # Intrinsics Calibration
    parser_intrinsics = subparsers.add_parser('calibrate_intrinsics', help='Perform intrinsic camera calibration.')
    parser_intrinsics.add_argument('--output', default=DEFAULT_INTRINSICS_FILE, help=f'Output file for intrinsics. Default: {DEFAULT_INTRINSICS_FILE}')
    parser_intrinsics.add_argument('--dims', type=str, default="7,6", help='Checkerboard inner corners (cols,rows). Default: "7,6"')
    parser_intrinsics.add_argument('--size', type=float, default=0.025, help='Checkerboard square size in meters. Default: 0.025')
    parser_intrinsics.add_argument('--frames', type=int, default=20, help='Number of calibration frames. Default: 20')

    # Depth Scale Calibration
    parser_depth = subparsers.add_parser('calibrate_depth', help='Perform depth scale calibration.')
    parser_depth.add_argument('--output', default=DEFAULT_DEPTH_SCALE_FILE, help=f'Output file for depth scale. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    parser_depth.add_argument('--distance', type=float, default=1.0, help='Known distance to target (m). Default: 1.0')
    parser_depth.add_argument('--samples', type=int, default=50, help='Number of depth samples. Default: 50')

    # Measurement
    parser_measure = subparsers.add_parser('measure', help='Measure box dimensions using existing calibration.')
    parser_measure.add_argument('--intrinsics', default=DEFAULT_INTRINSICS_FILE, help=f'Input intrinsics file. Default: {DEFAULT_INTRINSICS_FILE}')
    parser_measure.add_argument('--depth_scale', default=DEFAULT_DEPTH_SCALE_FILE, help=f'Input depth scale file. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    parser_measure.add_argument('--plane_thresh', type=float, default=0.01, help='Plane segmentation threshold (m). Default: 0.01')
    parser_measure.add_argument('--cluster_eps', type=float, default=0.02, help='Clustering epsilon (m). Default: 0.02')
    parser_measure.add_argument('--cluster_min_points', type=int, default=100, help='Clustering min_points. Default: 100')

    # Full Run (All Steps)
    parser_full = subparsers.add_parser('full_run', help='Calibrate (intrinsics & depth) then measure.')
    # Intrinsics args for full_run
    parser_full.add_argument('--intrinsics_output', default=DEFAULT_INTRINSICS_FILE, help=f'Output intrinsics file. Default: {DEFAULT_INTRINSICS_FILE}')
    parser_full.add_argument('--dims', type=str, default="7,6", help='Checkerboard dims for intrinsics. Default: "7,6"')
    parser_full.add_argument('--size', type=float, default=0.025, help='Checkerboard square size (m) for intrinsics. Default: 0.025')
    parser_full.add_argument('--frames', type=int, default=20, help='Number of frames for intrinsic calibration. Default: 20')
    # Depth_scale args for full_run
    parser_full.add_argument('--depth_scale_output', default=DEFAULT_DEPTH_SCALE_FILE, help=f'Output depth scale file. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    parser_full.add_argument('--known_distance', type=float, default=1.0, help='Known distance to target for depth cal (m). Default: 1.0')
    parser_full.add_argument('--samples', type=int, default=50, help='Number of samples for depth calibration. Default: 50')
    # Measurement args for full_run
    parser_full.add_argument('--plane_thresh_measure', type=float, default=0.01, help='Plane segmentation threshold (m) for measurement. Default: 0.01')
    parser_full.add_argument('--cluster_eps_measure', type=float, default=0.02, help='Clustering epsilon (m) for measurement. Default: 0.02')
    parser_full.add_argument('--cluster_min_points_measure', type=int, default=100, help='Clustering min_points for measurement. Default: 100')
    
    args = parser.parse_args()

    # Set Open3D verbosity level
    o3d_log_levels_map = {
        'Debug': o3d.utility.VerbosityLevel.Debug,
        'Info': o3d.utility.VerbosityLevel.Info,
        'Warning': o3d.utility.VerbosityLevel.Warning,
        'Error': o3d.utility.VerbosityLevel.Error,
        'Critical': o3d.utility.VerbosityLevel.Critical # Maps to Off in some O3D versions
    }
    # Ensure the chosen level is valid for o3d.utility.VerbosityLevel
    o3d_verbosity = o3d_log_levels_map.get(args.o3d_log_level, o3d.utility.VerbosityLevel.Warning)
    o3d.utility.set_verbosity_level(o3d_verbosity)
    cli_log_callback(f"Open3D verbosity set to: {args.o3d_log_level}")


    camera = None
    camera_messages = []

    # Setup camera if any command requires it
    if args.command in ['calibrate_intrinsics', 'calibrate_depth', 'measure', 'full_run']:
        cli_log_callback("Attempting to set up camera...")
        camera_obj, cam_msgs = setup_camera(args.port) 
        camera = camera_obj 
        camera_messages.extend(cam_msgs)
        for msg in cam_msgs: cli_log_callback(msg) 

        if not camera:
            cli_log_callback("Failed to initialize camera. Exiting.")
            return

    try:
        if args.command == 'calibrate_intrinsics':
            cli_log_callback("--- Starting Intrinsic Calibration (CLI Mode) ---")
            try:
                dims = tuple(map(int, args.dims.split(',')))
                if len(dims) != 2: raise ValueError("Checkerboard dims must be two comma-separated integers (cols,rows)")
            except ValueError as e:
                cli_log_callback(f"Error parsing checkerboard dimensions: {e}"); return
            
            mtx, dist, cal_messages = calibrate_intrinsics(
                camera, args.output, 
                checkerboard_dims=dims, square_size=args.size, num_frames=args.frames,
                log_callback=cli_log_callback # Pass CLI logger
            )
            # Messages are already printed by log_callback within the function
            if mtx is not None:
                cli_log_callback("Intrinsic calibration completed successfully (CLI).")
            else:
                cli_log_callback("Intrinsic calibration failed (CLI).")
        
        elif args.command == 'calibrate_depth':
            cli_log_callback("--- Starting Depth Scale Calibration (CLI Mode) ---")
            scale_factor, cal_messages = calibrate_depth_scale(
                camera, args.output, 
                known_distance=args.distance, num_samples=args.samples,
                log_callback=cli_log_callback # Pass CLI logger
            )
            if scale_factor is not None:
                cli_log_callback(f"Depth scale calibration completed. Scale: {scale_factor:.5f} (CLI).")
            else:
                cli_log_callback("Depth scale calibration failed (CLI).")

        elif args.command == 'measure':
            cli_log_callback("--- Starting Measurement (CLI Mode) ---")
            mtx, dist, scale_factor = None, None, None # Initialize
            
            if not os.path.exists(args.intrinsics):
                cli_log_callback(f"Error: Intrinsics file not found: {args.intrinsics}"); return
            try:
                data = np.load(args.intrinsics); mtx = data['mtx']; dist = data['dist']
                cli_log_callback(f"Loaded intrinsics from {args.intrinsics}")
            except Exception as e: cli_log_callback(f"Error loading intrinsics: {e}"); return

            if not os.path.exists(args.depth_scale):
                cli_log_callback(f"Error: Depth scale file not found: {args.depth_scale}"); return
            try:
                with open(args.depth_scale, 'r') as f: 
                    scale_data = json.load(f)
                    scale_factor = float(scale_data['depth_scale_factor']) # Ensure it's float
                cli_log_callback(f"Loaded depth scale factor ({scale_factor:.5f}) from {args.depth_scale}")
            except Exception as e: cli_log_callback(f"Error loading depth scale: {e}"); return
            
            if mtx is not None and scale_factor is not None: # dist is also loaded
                dimensions, measure_messages = measure_box_dimensions(
                    camera, mtx, dist, scale_factor, 
                    plane_dist_thresh=args.plane_thresh, 
                    cluster_eps=args.cluster_eps, 
                    cluster_min_points=args.cluster_min_points,
                    log_callback=cli_log_callback # Pass CLI logger
                )
                if dimensions:
                    cli_log_callback(f"Measurement completed. Dimensions: {dimensions} (CLI).")
                else:
                    cli_log_callback("Measurement failed or no box found (CLI).")
            else: 
                cli_log_callback("Measurement cannot proceed: missing critical calibration data.")

        elif args.command == 'full_run':
            cli_log_callback("--- Starting Full Run (CLI Mode) ---")
            # 1. Intrinsics Calibration
            cli_log_callback("\n[Full Run Step 1/3: Intrinsic Calibration]")
            try:
                dims_intr = tuple(map(int, args.dims.split(',')))
                if len(dims_intr) != 2: raise ValueError("Checkerboard dims for intrinsics: cols,rows")
            except ValueError as e: cli_log_callback(f"Error parsing checkerboard dimensions for intrinsics: {e}"); return
                
            mtx_full, dist_full, intr_messages = calibrate_intrinsics(
                camera, args.intrinsics_output, 
                checkerboard_dims=dims_intr, square_size=args.size, num_frames=args.frames,
                log_callback=cli_log_callback
            )
            if mtx_full is None: cli_log_callback("Intrinsic calibration failed during full run. Aborting."); return
            cli_log_callback("Intrinsic calibration successful for full run.")

            # 2. Depth Scale Calibration
            cli_log_callback("\n[Full Run Step 2/3: Depth Scale Calibration]")
            scale_factor_full, depth_messages = calibrate_depth_scale(
                camera, args.depth_scale_output, 
                known_distance=args.known_distance, num_samples=args.samples,
                log_callback=cli_log_callback
            )
            if scale_factor_full is None: cli_log_callback("Depth scale calibration failed during full run. Aborting."); return
            cli_log_callback(f"Depth scale calibration successful for full run. Scale: {scale_factor_full:.5f}")

            # 3. Measurement
            cli_log_callback("\n[Full Run Step 3/3: Measuring Box]")
            dimensions_full, measure_messages_full = measure_box_dimensions(
                camera, mtx_full, dist_full, scale_factor_full,
                plane_dist_thresh=args.plane_thresh_measure, 
                cluster_eps=args.cluster_eps_measure, 
                cluster_min_points=args.cluster_min_points_measure,
                log_callback=cli_log_callback
            )
            if dimensions_full:
                cli_log_callback(f"Measurement completed in full run. Dimensions: {dimensions_full} (CLI).")
            else:
                cli_log_callback("Measurement failed or no box found during full run (CLI).")
            cli_log_callback("\n--- Full Run Completed ---")

    finally:
        if camera:
            cli_log_callback("Closing camera...")
            camera.close()
            cli_log_callback("Camera closed.")

if __name__ == "__main__":
    main()