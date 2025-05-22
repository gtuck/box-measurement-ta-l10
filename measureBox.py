import numpy as np
import cv2
from TauLidarCamera.camera import Camera 
from TauLidarCommon.frame import FrameType 
import open3d as o3d
import argparse
import os
import json

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
    Returns: Configured Camera object or None if connection failed.
    """
    camera = None
    selected_port = port_arg

    if selected_port is None:
        ports = Camera.scan()
        print(f'\nFound {len(ports)} possible device(s)')
        if not ports:
            print("No Tau Camera devices found. Please check connection.")
            return None
        selected_port = ports[0] 
        print(f"Attempting to connect to the first available device on port '{selected_port}'")
    else:
        print(f"Attempting to connect to device on specified port '{selected_port}'")

    try:
        camera = Camera.open(selected_port)
        if camera:
            # Default configurations, adjust if needed
            camera.setModulationChannel(0)      
            camera.setIntegrationTime3d(0, 1000) 
            camera.setMinimalAmplitude(0, 10)   

            camera_info = camera.info()
            print("\nToF camera opened successfully:")
            print(f"    model:      {camera_info.model}")
            print(f"    firmware:   {camera_info.firmware}")
            print(f"    uid:        {camera_info.uid}")
            print(f"    resolution: {camera_info.resolution}") 
            print(f"    port:       {camera_info.port}")
            return camera
        else:
            print(f"Failed to open camera on port '{selected_port}'.")
            return None
    except Exception as e:
        print(f"Error during camera setup on port '{selected_port}': {e}")
        return None

def get_intensity_image(camera_obj):
    """
    Reads an intensity (grayscale) image from the camera.
    Uses FrameType.DISTANCE_GRAYSCALE and processes frame.data_grayscale.
    """
    if not camera_obj: return None
    try:
        print("Attempting to read FrameType.DISTANCE_GRAYSCALE for intensity data...")
        frame = camera_obj.readFrame(FrameType.DISTANCE_GRAYSCALE)
        if frame and hasattr(frame, 'data_grayscale'):
            res_str = camera_obj.info().resolution 
            frame_width, frame_height = map(int, res_str.split('x'))
            
            # data_grayscale is uint16, needs conversion to uint8 for cv2.findChessboardCorners
            mat_grayscale_uint16 = np.frombuffer(frame.data_grayscale, dtype=np.uint16, count=-1, offset=0).reshape(frame_height, frame_width)
            
            # Normalize and convert to uint8
            if np.max(mat_grayscale_uint16) > 0: # Avoid division by zero if image is all black
                 mat_grayscale_normalized = (mat_grayscale_uint16 / np.max(mat_grayscale_uint16) * 255)
            else:
                 mat_grayscale_normalized = mat_grayscale_uint16 # Already zeros
            
            mat_grayscale_uint8 = mat_grayscale_normalized.astype(np.uint8)

            return mat_grayscale_uint8
        else:
            if frame is None:
                print("Failed to read DISTANCE_GRAYSCALE frame (frame object is None).")
            else:
                print("DISTANCE_GRAYSCALE frame read, but 'data_grayscale' attribute is missing.")
            return None
    except AttributeError as ae:
        print(f"Error: FrameType 'DISTANCE_GRAYSCALE' may not be valid or an attribute is missing. {ae}")
        return None
    except Exception as e:
        print(f"Error reading or processing intensity frame: {e}")
        return None

def get_raw_depth_image(camera_obj):
    """
    Reads a raw depth image from the camera.
    Uses FrameType.DISTANCE and processes frame.data.
    """
    if not camera_obj: return None
    try:
        print("Attempting to read FrameType.DISTANCE for raw depth data...")
        frame = camera_obj.readFrame(FrameType.DISTANCE)
        if frame and hasattr(frame, 'data'):
            res_str = camera_obj.info().resolution 
            frame_width, frame_height = map(int, res_str.split('x'))
            
            # Assuming frame.data for FrameType.DISTANCE is raw 16-bit depth
            raw_depth_uint16 = np.frombuffer(frame.data, dtype=np.uint16, count=-1, offset=0).reshape(frame_height, frame_width)
            return raw_depth_uint16
        else:
            if frame is None:
                print("Failed to read DISTANCE frame (frame object is None).")
            else:
                print("DISTANCE frame read, but 'data' attribute is missing or frame is invalid.")
            return None
    except AttributeError as ae:
        print(f"Error: FrameType 'DISTANCE' may not be valid or an attribute is missing. {ae}")
        return None
    except Exception as e:
        print(f"Error reading or processing raw depth frame: {e}")
        return None

# ----------------------------------
# Calibration Functions
# ----------------------------------
def calibrate_intrinsics(camera_obj, output_file, checkerboard_dims=(7, 6), square_size=0.025, num_frames=20):
    if not camera_obj:
        print("Camera not initialized. Cannot perform intrinsic calibration.")
        return None, None
        
    print(f"Starting intrinsic calibration. Looking for {checkerboard_dims} inner corners.")

    objp = np.zeros((checkerboard_dims[1] * checkerboard_dims[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []
    collected = 0
    print("Press 'c' to capture a frame when checkerboard is detected, 'q' to quit.")
    
    cv2.namedWindow('Intrinsics Calibration - Grayscale Feed', cv2.WINDOW_NORMAL)
    last_gray_shape = None

    while collected < num_frames:
        gray = get_intensity_image(camera_obj) 

        if gray is None:
            print("Failed to get intensity image for checkerboard detection.")
            key = cv2.waitKey(100) & 0xFF 
            if key == ord('q'):
                print("Quitting calibration due to frame reading issues.")
                break
            continue 
        
        last_gray_shape = gray.shape 

        display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        found, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)
        
        if found:
            cv2.drawChessboardCorners(display_img, checkerboard_dims, corners, found)
            cv2.putText(display_img, f"Checkerboard detected! Press 'c' to capture ({collected}/{num_frames})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, "Aim at checkerboard...", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Intrinsics Calibration - Grayscale Feed', display_img)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            print("Intrinsic calibration quit by user.")
            break
        elif key == ord('c') and found:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
            collected += 1
            print(f"Captured calibration frame {collected}/{num_frames}")
            cv2.putText(display_img, f"FRAME {collected} CAPTURED!", 
                        (display_img.shape[1]//2 - 100, display_img.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.imshow('Intrinsics Calibration - Grayscale Feed', display_img)
            cv2.waitKey(500) 

    cv2.destroyAllWindows()

    if collected == 0:
        print("No frames collected. Intrinsic calibration cannot proceed.")
        return None, None
    if collected < num_frames:
         print(f"Warning: Only {collected} frames collected, expected {num_frames}. Calibration might be suboptimal.")
    
    if not objpoints or not imgpoints or last_gray_shape is None:
        print("Not enough points or valid image shape for calibration.")
        return None, None
        
    print("Calibrating camera...")
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, last_gray_shape[::-1], None, None 
        )
    except cv2.error as e:
        print(f"OpenCV Error during calibration: {e}")
        return None, None

    if not ret:
        print("Intrinsic calibration failed. Check your setup and images.")
        return None, None
        
    print("Intrinsic calibration successful.")
    print("Camera matrix (mtx):\n", mtx)
    print("Distortion coefficients (dist):\n", dist.ravel())

    try:
        np.savez(output_file, mtx=mtx, dist=dist)
        print(f"Intrinsic calibration data saved to {output_file}")
    except Exception as e:
        print(f"Error saving intrinsic calibration data: {e}")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "Total reprojection error: {}".format(mean_error/len(objpoints)) )
    if (mean_error/len(objpoints)) > 1.0:
        print("Warning: Reprojection error is high. Calibration might be inaccurate.")

    return mtx, dist

def calibrate_depth_scale(camera_obj, output_file, known_distance=1.0, num_samples=50):
    if not camera_obj:
        print("Camera not initialized. Cannot perform depth scale calibration.")
        return None

    print(f"Starting depth scale calibration.")
        
    scales = []
    cv2.namedWindow('Depth Scale Calibration - Depth Feed', cv2.WINDOW_NORMAL)
    sampling_started = False

    while True:
        raw_depth = get_raw_depth_image(camera_obj) 
        if raw_depth is None:
            print("Failed to read depth frame from camera for depth calibration.")
            key = cv2.waitKey(100) & 0xFF 
            if key == ord('q'):
                print("Quitting depth calibration due to frame reading issues.")
                break
            continue

        depth_display_normalized = cv2.normalize(raw_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display_color = cv2.applyColorMap(depth_display_normalized, cv2.COLORMAP_JET)
        
        h, w = raw_depth.shape
        roi_size = 20 
        cv2.rectangle(depth_display_color, (w//2-roi_size//2, h//2-roi_size//2), 
                      (w//2+roi_size//2, h//2+roi_size//2), (0,255,0), 2)

        if not sampling_started:
            cv2.putText(depth_display_color, f"Aim at target {known_distance:.2f}m away. Press 's' to start.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(depth_display_color, f"Sampling... {len(scales)}/{num_samples}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(scales) < num_samples:
                depth_m_at_center = raw_depth[h//2-roi_size//2:h//2+roi_size//2, w//2-roi_size//2:w//2+roi_size//2].astype(np.float32) * 0.001
                valid_depths = depth_m_at_center[depth_m_at_center > 0.1] 
                if valid_depths.size > 0:
                    measured_distance = np.median(valid_depths)
                    if measured_distance > 0: 
                        current_scale = known_distance / measured_distance
                        scales.append(current_scale)
                        print(f"Sample {len(scales)}/{num_samples}: Measured Median Distance = {measured_distance:.3f}m, Scale = {current_scale:.4f}")
                    else:
                        print(f"Sample {len(scales)}/{num_samples}: Warning - Median measured distance is zero or invalid in ROI.")
                else:
                    print(f"Sample {len(scales)}/{num_samples}: Warning - No valid depth data in ROI.")
                cv2.waitKey(50) 
            else:
                print("Sampling complete.")
                break
        
        cv2.imshow('Depth Scale Calibration - Depth Feed', depth_display_color)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            print("Depth scale calibration quit by user.")
            cv2.destroyAllWindows() 
            return None
        elif key == ord('s') and not sampling_started:
            print("Starting sampling...")
            sampling_started = True
            scales = [] 

    cv2.destroyAllWindows()

    if not scales:
        print("No depth samples collected. Cannot calculate scale factor.")
        return None

    scale_factor = float(np.mean(scales))
    std_dev_scale = float(np.std(scales))
    print(f"Depth scale factor calculated: {scale_factor:.5f}")
    print(f"Standard deviation of scale samples: {std_dev_scale:.5f}")
    if std_dev_scale > 0.05 * scale_factor : 
        print("Warning: High standard deviation in depth scale samples. Measurement might be inconsistent.")
        print("Ensure the target surface is flat, perpendicular to the camera, and at the correct distance.")

    try:
        with open(output_file, 'w') as f:
            json.dump({'depth_scale_factor': scale_factor, 'known_distance_meters': known_distance, 'num_samples': len(scales), 'std_dev_scale': std_dev_scale}, f, indent=4)
        print(f"Depth scale factor saved to {output_file}")
    except Exception as e:
        print(f"Error saving depth scale factor: {e}")
        
    return scale_factor

# ----------------------------------
# Point-Cloud Processing & Measurement
# ----------------------------------
def depth_to_pointcloud(depth_m, camera_matrix):
    h, w = depth_m.shape
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)
    valid = depth_m > 0.1  
    
    z = depth_m[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    pts = np.stack((x, y, z), axis=-1)
    
    if pts.size == 0:
        print("Warning: No valid points found to create point cloud.")
        return o3d.geometry.PointCloud() 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def segment_plane_and_objects(pcd, dist_thresh=0.01, ransac_n=3, num_iter=1000):
    if not pcd.has_points():
        print("Warning: Empty point cloud passed to segment_plane_and_objects.")
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=ransac_n, num_iterations=num_iter)
    except Exception as e:
        print(f"Error during plane segmentation: {e}")
        return o3d.geometry.PointCloud(), pcd 
    if not inliers:
        print("Warning: No plane found. Returning original point cloud as objects.")
        return o3d.geometry.PointCloud(), pcd
    plane_pcd = pcd.select_by_index(inliers)
    objects_pcd = pcd.select_by_index(inliers, invert=True)
    plane_pcd.paint_uniform_color([0.7, 0.7, 0.7]) 
    return plane_pcd, objects_pcd

def cluster_and_fit_box(objects_pcd, eps=0.02, min_points=100):
    if not objects_pcd.has_points():
        print("No object points to cluster.")
        return None
    try:
        labels = np.array(objects_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except Exception as e:
        print(f"Error during DBSCAN clustering: {e}")
        return None
    if labels.size == 0 or np.all(labels < 0) : 
        print("No clusters found or all points are noise.")
        if objects_pcd.has_points():
            print("Attempting to fit OBB to all object points as a fallback.")
            try:
                obb = objects_pcd.get_oriented_bounding_box()
                obb.color = (0, 1, 0) 
                return obb
            except Exception as e_obb:
                print(f"Error fitting OBB to all object points: {e_obb}")
                return None
        return None
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if unique_labels.size == 0:
        print("No valid clusters found (all points considered noise).")
        return None
    largest_cluster_label = unique_labels[counts.argmax()]
    cluster_indices = np.where(labels == largest_cluster_label)[0]
    largest_cluster_pcd = objects_pcd.select_by_index(cluster_indices)
    if not largest_cluster_pcd.has_points():
        print("Largest cluster is empty.")
        return None
    try:
        obb = largest_cluster_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  
    except Exception as e:
        print(f"Error getting OBB for the cluster: {e}")
        return None
    return obb

def capture_frames_for_measurement(camera_obj):
    """ Captures intensity and raw depth frame for measurement. """
    if not camera_obj: return None, None
    
    intensity_image = get_intensity_image(camera_obj)
    raw_depth_image = get_raw_depth_image(camera_obj)
    
    return intensity_image, raw_depth_image

def measure_box_dimensions(camera_obj, mtx, dist, depth_scale_factor, 
                           plane_dist_thresh=0.01, cluster_eps=0.02, cluster_min_points=100):
    if not camera_obj:
        print("Camera not initialized for measurement.")
        return
    if mtx is None or depth_scale_factor is None:
        print("Error: Calibration data (mtx or depth_scale_factor) is missing.")
        return

    print("Attempting to capture frame for measurement...")
    intensity_img, raw_depth = capture_frames_for_measurement(camera_obj) 

    if raw_depth is None: 
        print("Failed to capture critical depth frame for measurement. Check camera.")
        return
    if intensity_img is None: 
        print("Warning: Failed to capture intensity frame. Proceeding with depth data only for point cloud.")

    depth_m = raw_depth.astype(np.float32) * 0.001 * depth_scale_factor
    
    print("Generating point cloud...")
    pcd_full = depth_to_pointcloud(depth_m, mtx)
    if not pcd_full.has_points():
        print("Point cloud generation failed or resulted in an empty cloud.")
        return
    
    pcd_downsampled = pcd_full.voxel_down_sample(voxel_size=0.005) 
    print(f"Point cloud downsampled from {len(pcd_full.points)} to {len(pcd_downsampled.points)} points.")

    print("Segmenting plane (e.g., table)...")
    plane_pcd, objects_pcd = segment_plane_and_objects(pcd_downsampled, dist_thresh=plane_dist_thresh)
    
    if not objects_pcd.has_points():
        print("No object points found after plane segmentation.")
        geometries_to_draw = [pcd_downsampled]
        if plane_pcd.has_points(): geometries_to_draw.append(plane_pcd)
        o3d.visualization.draw_geometries(geometries_to_draw, window_name="Measurement - No Objects Found")
        return

    print("Clustering objects and fitting bounding box...")
    obb = cluster_and_fit_box(objects_pcd, eps=cluster_eps, min_points=cluster_min_points)

    if obb is None:
        print("Could not fit a bounding box to any detected objects.")
        geometries_to_draw = [pcd_downsampled]
        if plane_pcd.has_points(): geometries_to_draw.append(plane_pcd)
        if objects_pcd.has_points(): 
            objects_pcd.paint_uniform_color([0,0,1]) 
            geometries_to_draw.append(objects_pcd)
        o3d.visualization.draw_geometries(geometries_to_draw, window_name="Measurement - No Box Fitted")
        return

    dims_meters = obb.extent
    dims_mm = dims_meters * 1000  
    print("\n--- Measured Box Dimensions ---")
    print(f"  Extent X (mm): {dims_mm[0]:.1f}")
    print(f"  Extent Y (mm): {dims_mm[1]:.1f}")
    print(f"  Extent Z (mm): {dims_mm[2]:.1f}")
    print(f"  Center (m): {obb.center[0]:.3f}, {obb.center[1]:.3f}, {obb.center[2]:.3f}")
    print("-------------------------------\n")
    print("Displaying 3D visualization. Close window to exit.")
    geometries_to_draw = [pcd_downsampled] 
    if plane_pcd.has_points(): geometries_to_draw.append(plane_pcd) 
    geometries_to_draw.append(obb)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    geometries_to_draw.append(coordinate_frame)
    o3d.visualization.draw_geometries(geometries_to_draw, window_name="Box Measurement Visualization", width=800, height=600)

# ----------------------------------
# Main Execution
# ----------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calibrate Tau LiDAR Camera and measure box dimensions.")
    parser.add_argument('--port', default=None, help='Specify serial port of the Tau LiDAR Camera. If None, scans for cameras.')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set the logging level for Open3D.')
    subparsers = parser.add_subparsers(dest='command', help='Available commands.', required=True)

    parser_intrinsics = subparsers.add_parser('calibrate_intrinsics', help='Perform intrinsic camera calibration.')
    parser_intrinsics.add_argument('--output', default=DEFAULT_INTRINSICS_FILE, help=f'Output file for intrinsics. Default: {DEFAULT_INTRINSICS_FILE}')
    parser_intrinsics.add_argument('--dims', type=str, default="7,6", help='Checkerboard inner corners (cols,rows). Default: "7,6"')
    parser_intrinsics.add_argument('--size', type=float, default=0.025, help='Checkerboard square size in meters. Default: 0.025')
    parser_intrinsics.add_argument('--frames', type=int, default=20, help='Number of calibration frames. Default: 20')

    parser_depth = subparsers.add_parser('calibrate_depth', help='Perform depth scale calibration.')
    parser_depth.add_argument('--output', default=DEFAULT_DEPTH_SCALE_FILE, help=f'Output file for depth scale. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    parser_depth.add_argument('--distance', type=float, default=1.0, help='Known distance to target (m). Default: 1.0')
    parser_depth.add_argument('--samples', type=int, default=50, help='Number of depth samples. Default: 50')

    parser_measure = subparsers.add_parser('measure', help='Measure box dimensions using existing calibration.')
    parser_measure.add_argument('--intrinsics', default=DEFAULT_INTRINSICS_FILE, help=f'Input intrinsics file. Default: {DEFAULT_INTRINSICS_FILE}')
    parser_measure.add_argument('--depth_scale', default=DEFAULT_DEPTH_SCALE_FILE, help=f'Input depth scale file. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    parser_measure.add_argument('--plane_thresh', type=float, default=0.01, help='Plane segmentation threshold (m). Default: 0.01')
    parser_measure.add_argument('--cluster_eps', type=float, default=0.02, help='Clustering epsilon (m). Default: 0.02')
    parser_measure.add_argument('--cluster_min_points', type=int, default=100, help='Clustering min_points. Default: 100')

    parser_full = subparsers.add_parser('full_run', help='Calibrate (intrinsics & depth) then measure.')
    parser_full.add_argument('--intrinsics_output', default=DEFAULT_INTRINSICS_FILE, help=f'Output intrinsics file. Default: {DEFAULT_INTRINSICS_FILE}')
    parser_full.add_argument('--depth_scale_output', default=DEFAULT_DEPTH_SCALE_FILE, help=f'Output depth scale file. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    parser_full.add_argument('--dims', type=str, default="7,6", help='Checkerboard dims. Default: "7,6"')
    parser_full.add_argument('--size', type=float, default=0.025, help='Checkerboard square size (m). Default: 0.025')
    parser_full.add_argument('--frames', type=int, default=20, help='Intrinsic frames. Default: 20')
    parser_full.add_argument('--known_distance', type=float, default=1.0, help='Known distance for depth cal (m). Default: 1.0')
    parser_full.add_argument('--samples', type=int, default=50, help='Depth samples. Default: 50')
    parser_full.add_argument('--plane_thresh', type=float, default=0.01, help='Plane segmentation threshold (m). Default: 0.01')
    parser_full.add_argument('--cluster_eps', type=float, default=0.02, help='Clustering epsilon (m). Default: 0.02')
    parser_full.add_argument('--cluster_min_points', type=int, default=100, help='Clustering min_points. Default: 100')

    args = parser.parse_args()

    log_levels_map = {
        'DEBUG': o3d.utility.VerbosityLevel.Debug,
        'INFO': o3d.utility.VerbosityLevel.Info,
        'WARNING': o3d.utility.VerbosityLevel.Warning,
        'ERROR': o3d.utility.VerbosityLevel.Error
    }
    o3d.utility.set_verbosity_level(log_levels_map.get(args.log_level, o3d.utility.VerbosityLevel.Info))

    camera = None 

    if args.command in ['calibrate_intrinsics', 'calibrate_depth', 'measure', 'full_run']:
        camera = setup_camera(args.port)
        if not camera:
            print("Failed to initialize camera. Exiting.")
            return 

    try:
        if args.command == 'calibrate_intrinsics':
            try:
                dims = tuple(map(int, args.dims.split(',')))
                if len(dims) != 2: raise ValueError("Checkerboard dims: cols,rows")
            except ValueError as e:
                print(f"Error parsing checkerboard dimensions: {e}"); return
            calibrate_intrinsics(camera, args.output, checkerboard_dims=dims, square_size=args.size, num_frames=args.frames)
        
        elif args.command == 'calibrate_depth':
            calibrate_depth_scale(camera, args.output, known_distance=args.distance, num_samples=args.samples)

        elif args.command == 'measure':
            mtx, dist, scale_factor = None, None, None
            if not os.path.exists(args.intrinsics):
                print(f"Error: Intrinsics file not found: {args.intrinsics}"); return
            try:
                data = np.load(args.intrinsics); mtx = data['mtx']; dist = data['dist']
                print(f"Loaded intrinsics from {args.intrinsics}")
            except Exception as e: print(f"Error loading intrinsics: {e}"); return

            if not os.path.exists(args.depth_scale):
                print(f"Error: Depth scale file not found: {args.depth_scale}"); return
            try:
                with open(args.depth_scale, 'r') as f: scale_data = json.load(f); scale_factor = scale_data['depth_scale_factor']
                print(f"Loaded depth scale factor ({scale_factor:.5f}) from {args.depth_scale}")
            except Exception as e: print(f"Error loading depth scale: {e}"); return
            
            if mtx is not None and scale_factor is not None:
                measure_box_dimensions(camera, mtx, dist, scale_factor, 
                                       plane_dist_thresh=args.plane_thresh, 
                                       cluster_eps=args.cluster_eps, 
                                       cluster_min_points=args.cluster_min_points)
            else: print("Measurement cannot proceed: missing calibration data.")

        elif args.command == 'full_run':
            print("--- Starting Full Run: Intrinsic Calibration ---")
            try:
                dims = tuple(map(int, args.dims.split(',')))
                if len(dims) != 2: raise ValueError("Checkerboard dims: cols,rows")
            except ValueError as e: print(f"Error parsing checkerboard dimensions: {e}"); return
                
            mtx, dist = calibrate_intrinsics(camera, args.intrinsics_output, 
                                             checkerboard_dims=dims, square_size=args.size, num_frames=args.frames)
            if mtx is None: print("Intrinsic calibration failed. Aborting."); return
            
            print("\n--- Full Run: Depth Scale Calibration ---")
            scale_factor = calibrate_depth_scale(camera, args.depth_scale_output, 
                                                 known_distance=args.known_distance, num_samples=args.samples)
            if scale_factor is None: print("Depth scale calibration failed. Aborting."); return

            print("\n--- Full Run: Measuring Box ---")
            measure_box_dimensions(camera, mtx, dist, scale_factor,
                                   plane_dist_thresh=args.plane_thresh, 
                                   cluster_eps=args.cluster_eps, 
                                   cluster_min_points=args.cluster_min_points)
    finally:
        if camera:
            print("Closing camera...")
            camera.close()

if __name__ == "__main__":
    main()