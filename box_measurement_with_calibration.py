import numpy as np
import cv2
import tau_lidar_camera as tau # type: ignore
import open3d as o3d
import argparse # Added for command-line arguments
import os       # Added for path operations
import json     # Added for saving/loading depth scale

# ----------------------------------
# Calibration File Constants
# ----------------------------------
DEFAULT_INTRINSICS_FILE = 'camera_intrinsics.npz'
DEFAULT_DEPTH_SCALE_FILE = 'depth_scale.json'

# ----------------------------------
# Calibration Functions
# ----------------------------------

def calibrate_intrinsics(output_file, checkerboard_dims=(7, 6), square_size=0.025, num_frames=20):
    """
    Perform intrinsic calibration using a checkerboard pattern.
    - output_file: Path to save the camera matrix and distortion coefficients.
    - checkerboard_dims: (cols, rows) of inner corners
    - square_size: size of each square in meters
    - num_frames: number of successful detections to collect
    Returns: camera matrix (mtx), distortion coefficients (dist)
    """
    print(f"Starting intrinsic calibration. Looking for {checkerboard_dims} inner corners.")
    print(f"Ensure your checkerboard squares are {square_size*1000} mm.")
    print(f"Move the checkerboard to {num_frames} different positions and orientations.")

    objp = np.zeros((checkerboard_dims[1] * checkerboard_dims[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []
    
    # Attempt to initialize the Tau Camera
    try:
        cam = tau.TauCamera()
    except Exception as e:
        print(f"Error initializing Tau Camera: {e}")
        print("Please ensure the Tau LiDAR Camera is connected and drivers are installed.")
        print("You can find more info at: https://www.onsemiconductor.com/products/sensors/lidar-sensors/SECO-RANGEFINDER-GEVK")
        return None, None

    cam.start()
    collected = 0
    print("Press 'c' to capture a frame when checkerboard is detected, 'q' to quit.")
    
    # Create a window that can be resized
    cv2.namedWindow('Intrinsics Calibration - Grayscale Feed', cv2.WINDOW_NORMAL)

    while collected < num_frames:
        gray, depth = cam.read()
        if gray is None:
            print("Failed to read frame from camera. Check connection.")
            cv2.waitKey(100) # Wait a bit before retrying or quitting
            continue

        # Make a copy for drawing
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
            # Briefly show it's captured
            cv2.putText(display_img, f"FRAME {collected} CAPTURED!", 
                        (display_img.shape[1]//2 - 100, display_img.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.imshow('Intrinsics Calibration - Grayscale Feed', display_img)
            cv2.waitKey(500) # Show message for 0.5s


    cam.stop()
    cv2.destroyAllWindows()

    if collected < num_frames and collected > 0 : # Check if at least some frames were collected
         print(f"Warning: Only {collected} frames collected, expected {num_frames}. Calibration might be suboptimal.")
    elif collected == 0:
        print("No frames collected. Intrinsic calibration cannot proceed.")
        return None, None


    if not objpoints or not imgpoints:
        print("Not enough points for calibration.")
        return None, None
        
    print("Calibrating camera...")
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
    except cv2.error as e:
        print(f"OpenCV Error during calibration: {e}")
        print("This can happen if the checkerboard points are collinear or not diverse enough.")
        return None, None


    if not ret:
        print("Intrinsic calibration failed. Check your setup and images.")
        return None, None
        
    print("Intrinsic calibration successful.")
    print("Camera matrix (mtx):\n", mtx)
    print("Distortion coefficients (dist):\n", dist.ravel())

    # Save calibration data
    try:
        np.savez(output_file, mtx=mtx, dist=dist)
        print(f"Intrinsic calibration data saved to {output_file}")
    except Exception as e:
        print(f"Error saving intrinsic calibration data: {e}")

    # Calculate and display reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "Total reprojection error: {}".format(mean_error/len(objpoints)) )
    if (mean_error/len(objpoints)) > 1.0:
        print("Warning: Reprojection error is high. Calibration might be inaccurate.")


    return mtx, dist

def calibrate_depth_scale(output_file, known_distance=1.0, num_samples=50):
    """
    Compute scale factor to convert raw depth units to accurate meters.
    - output_file: Path to save the depth scale factor.
    - known_distance: distance from camera to a flat surface in meters
    - num_samples: number of depth readings to average
    Returns: depth scale factor
    """
    print(f"Starting depth scale calibration.")
    print(f"Point the camera at a flat, static surface EXACTLY {known_distance:.2f} meters away.")
    print(f"The central region of the camera's view will be sampled {num_samples} times.")
    print("Press 's' to start sampling, or 'q' to quit.")

    try:
        cam = tau.TauCamera()
    except Exception as e:
        print(f"Error initializing Tau Camera: {e}")
        return None
        
    cam.start()
    scales = []
    
    cv2.namedWindow('Depth Scale Calibration - Depth Feed', cv2.WINDOW_NORMAL)
    sampling_started = False

    while True:
        gray, raw_depth = cam.read()
        if raw_depth is None:
            print("Failed to read depth frame from camera.")
            cv2.waitKey(100)
            continue

        # Normalize depth for display (0-255)
        depth_display = cv2.normalize(raw_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        
        h, w = raw_depth.shape
        roi_size = 20 # Region of interest size (pixels) for sampling
        cv2.rectangle(depth_display_color, (w//2-roi_size//2, h//2-roi_size//2), 
                      (w//2+roi_size//2, h//2+roi_size//2), (0,255,0), 2)

        if not sampling_started:
            cv2.putText(depth_display_color, f"Aim at target {known_distance:.2f}m away. Press 's' to start.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(depth_display_color, f"Sampling... {len(scales)}/{num_samples}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(scales) < num_samples:
                # Assuming raw_depth units are millimeters, convert to meters
                depth_m_at_center = raw_depth[h//2-roi_size//2:h//2+roi_size//2, w//2-roi_size//2:w//2+roi_size//2].astype(np.float32) * 0.001
                
                # Filter out zero or invalid depth values
                valid_depths = depth_m_at_center[depth_m_at_center > 0.1] # Ignore depths less than 10cm as likely invalid
                if valid_depths.size > 0:
                    measured_distance = np.median(valid_depths)
                    if measured_distance > 0: # Avoid division by zero
                        current_scale = known_distance / measured_distance
                        scales.append(current_scale)
                        print(f"Sample {len(scales)}/{num_samples}: Measured Median Distance = {measured_distance:.3f}m, Scale = {current_scale:.4f}")
                    else:
                        print(f"Sample {len(scales)}/{num_samples}: Warning - Median measured distance is zero or invalid in ROI.")
                else:
                    print(f"Sample {len(scales)}/{num_samples}: Warning - No valid depth data in ROI.")
                cv2.waitKey(50) # Short delay between samples
            else:
                print("Sampling complete.")
                break
        
        cv2.imshow('Depth Scale Calibration - Depth Feed', depth_display_color)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q'):
            print("Depth scale calibration quit by user.")
            cam.stop()
            cv2.destroyAllWindows()
            return None
        elif key == ord('s') and not sampling_started:
            print("Starting sampling...")
            sampling_started = True
            scales = [] # Reset scales if 's' is pressed again before finishing

    cam.stop()
    cv2.destroyAllWindows()

    if not scales:
        print("No depth samples collected. Cannot calculate scale factor.")
        return None

    scale_factor = float(np.mean(scales))
    std_dev_scale = float(np.std(scales))
    print(f"Depth scale factor calculated: {scale_factor:.5f}")
    print(f"Standard deviation of scale samples: {std_dev_scale:.5f}")
    if std_dev_scale > 0.05 * scale_factor : # If std dev is more than 5% of the mean
        print("Warning: High standard deviation in depth scale samples. Measurement might be inconsistent.")
        print("Ensure the target surface is flat, perpendicular to the camera, and at the correct distance.")


    # Save depth scale factor
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
    """
    Converts a depth image to a 3D point cloud.
    - depth_m: Depth image in meters.
    - camera_matrix: Calibrated camera intrinsic matrix (mtx).
    Returns: o3d.geometry.PointCloud
    """
    h, w = depth_m.shape
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    # Create a mask for valid depth values (e.g., > 0 and < some reasonable max if needed)
    valid = depth_m > 0.1  # Example: ignore points closer than 10cm or very far
    
    z = depth_m[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy

    pts = np.stack((x, y, z), axis=-1)
    
    if pts.size == 0:
        print("Warning: No valid points found to create point cloud.")
        return o3d.geometry.PointCloud() # Return empty point cloud

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def segment_plane_and_objects(pcd, dist_thresh=0.01, ransac_n=3, num_iter=1000):
    """
    Segments the dominant plane (e.g., table) from the point cloud.
    Returns: plane_pcd, objects_pcd (point cloud of objects above the plane)
    """
    if not pcd.has_points():
        print("Warning: Empty point cloud passed to segment_plane_and_objects.")
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=ransac_n,
            num_iterations=num_iter
        )
    except Exception as e:
        print(f"Error during plane segmentation: {e}")
        return o3d.geometry.PointCloud(), pcd # Return original pcd as objects if segmentation fails

    if not inliers:
        print("Warning: No plane found during segmentation. Returning original point cloud as objects.")
        return o3d.geometry.PointCloud(), pcd

    plane_pcd = pcd.select_by_index(inliers)
    objects_pcd = pcd.select_by_index(inliers, invert=True)
    
    # Optional: Color the plane for visualization
    plane_pcd.paint_uniform_color([0.7, 0.7, 0.7]) # Gray color for the plane
    
    return plane_pcd, objects_pcd

def cluster_and_fit_box(objects_pcd, eps=0.02, min_points=100):
    """
    Clusters the objects point cloud and fits an oriented bounding box (OBB)
    to the largest cluster.
    Returns: o3d.geometry.OrientedBoundingBox or None if no suitable cluster found.
    """
    if not objects_pcd.has_points():
        print("No object points to cluster.")
        return None

    try:
        # DBSCAN clustering
        # Note: o3d.geometry.PointCloud.cluster_dbscan returns a list of integers (labels)
        labels = np.array(objects_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except Exception as e:
        print(f"Error during DBSCAN clustering: {e}")
        return None

    if labels.size == 0 or np.all(labels < 0) : # Check if any valid clusters were found (labels >= 0)
        print("No clusters found or all points are noise.")
        # Try to fit OBB to the whole objects_pcd if no clusters
        if objects_pcd.has_points():
            print("Attempting to fit OBB to all object points as a fallback.")
            try:
                obb = objects_pcd.get_oriented_bounding_box()
                obb.color = (0, 1, 0) # Green for fallback OBB
                return obb
            except Exception as e_obb:
                print(f"Error fitting OBB to all object points: {e_obb}")
                return None
        return None

    # Find the largest cluster (ignoring noise points labeled -1)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if unique_labels.size == 0:
        print("No valid clusters found (all points considered noise).")
        return None
        
    largest_cluster_label = unique_labels[counts.argmax()]
    
    # Select points belonging to the largest cluster
    cluster_indices = np.where(labels == largest_cluster_label)[0]
    largest_cluster_pcd = objects_pcd.select_by_index(cluster_indices)

    if not largest_cluster_pcd.has_points():
        print("Largest cluster is empty (this shouldn't happen if labels were found).")
        return None

    # Get the oriented bounding box for the largest cluster
    try:
        obb = largest_cluster_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  # Red color for the main box
    except Exception as e:
        print(f"Error getting oriented bounding box for the cluster: {e}")
        return None
        
    return obb

def capture_raw_frame():
    """
    Captures a grayscale and raw depth frame from the Tau Camera.
    Returns: gray_image, raw_depth_image
    """
    try:
        cam = tau.TauCamera()
    except Exception as e:
        print(f"Error initializing Tau Camera: {e}")
        return None, None
        
    cam.start()
    gray, raw_depth = cam.read()
    cam.stop()
    return gray, raw_depth

def measure_box_dimensions(mtx, dist, depth_scale_factor, 
                           plane_dist_thresh=0.01, cluster_eps=0.02, cluster_min_points=100):
    """
    Captures a frame, processes point cloud, and measures the largest box.
    - mtx: Calibrated camera intrinsic matrix.
    - dist: Calibrated distortion coefficients.
    - depth_scale_factor: Factor to convert raw depth to meters.
    """
    if mtx is None or depth_scale_factor is None:
        print("Error: Calibration data (mtx or depth_scale_factor) is missing.")
        return

    print("Attempting to capture frame for measurement...")
    gray, raw_depth = capture_raw_frame()

    if raw_depth is None:
        print("Failed to capture frame for measurement. Check camera.")
        return

    # Convert raw depth (assumed mm) to meters using the scale factor
    depth_m = raw_depth.astype(np.float32) * 0.001 * depth_scale_factor
    
    # (Optional) Undistort grayscale image if needed for display or other processing
    # gray_undistorted = cv2.undistort(gray, mtx, dist, None, mtx)
    # cv2.imshow("Undistorted Grayscale", gray_undistorted)
    # cv2.waitKey(1)


    print("Generating point cloud...")
    pcd_full = depth_to_pointcloud(depth_m, mtx)
    if not pcd_full.has_points():
        print("Point cloud generation failed or resulted in an empty cloud.")
        return
    
    # Downsample for faster processing (optional, but can help)
    pcd_downsampled = pcd_full.voxel_down_sample(voxel_size=0.005) # 5mm voxel size
    print(f"Point cloud downsampled from {len(pcd_full.points)} to {len(pcd_downsampled.points)} points.")


    print("Segmenting plane (e.g., table)...")
    plane_pcd, objects_pcd = segment_plane_and_objects(pcd_downsampled, dist_thresh=plane_dist_thresh)
    
    if not objects_pcd.has_points():
        print("No object points found after plane segmentation.")
        geometries_to_draw = [pcd_downsampled]
        if plane_pcd.has_points():
            geometries_to_draw.append(plane_pcd)
        o3d.visualization.draw_geometries(geometries_to_draw, window_name="Measurement - No Objects Found")
        return

    print("Clustering objects and fitting bounding box...")
    obb = cluster_and_fit_box(objects_pcd, eps=cluster_eps, min_points=cluster_min_points)

    if obb is None:
        print("Could not fit a bounding box to any detected objects.")
        geometries_to_draw = [pcd_downsampled]
        if plane_pcd.has_points():
            geometries_to_draw.append(plane_pcd)
        if objects_pcd.has_points(): # Show remaining object points if OBB failed
            objects_pcd.paint_uniform_color([0,0,1]) # Blue for unboxed objects
            geometries_to_draw.append(objects_pcd)
        o3d.visualization.draw_geometries(geometries_to_draw, window_name="Measurement - No Box Fitted")
        return

    dims_meters = obb.extent
    dims_mm = dims_meters * 1000  # Convert meters to millimeters

    # Sort dimensions for consistency (optional, but often Length > Width > Height)
    # sorted_dims_mm = np.sort(dims_mm)[::-1] 

    print("\n--- Measured Box Dimensions ---")
    # The OBB extent does not directly map to Length, Width, Height in a fixed order.
    # It depends on the box's orientation. For simplicity, printing as X, Y, Z extents.
    print(f"  Extent X (mm): {dims_mm[0]:.1f}")
    print(f"  Extent Y (mm): {dims_mm[1]:.1f}")
    print(f"  Extent Z (mm): {dims_mm[2]:.1f}")
    print(f"  Center (m): {obb.center[0]:.3f}, {obb.center[1]:.3f}, {obb.center[2]:.3f}")
    print("-------------------------------\n")

    print("Displaying 3D visualization. Close window to exit.")
    # Visualize: original (downsampled) cloud, segmented plane, segmented objects, and OBB
    # Give objects_pcd a different color to distinguish from the fitted box cluster
    # (cluster_and_fit_box already colors the OBB red)
    # If objects_pcd contains more than just the fitted box, color it differently.
    
    geometries_to_draw = [pcd_downsampled] # Start with the downsampled full cloud
    if plane_pcd.has_points():
        geometries_to_draw.append(plane_pcd) # Add the gray plane
    
    # For better visualization, show the objects_pcd colored differently if it's not empty
    # and then the OBB on top.
    # If you want to show only the clustered box points with the OBB:
    # largest_cluster_pcd = objects_pcd.select_by_index(np.where(labels == largest_cluster_label)[0])
    # largest_cluster_pcd.paint_uniform_color([0,1,0]) # Green for the points inside OBB
    # geometries_to_draw.append(largest_cluster_pcd)
    
    geometries_to_draw.append(obb)
    
    # Add a coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    geometries_to_draw.append(coordinate_frame)


    o3d.visualization.draw_geometries(geometries_to_draw,
                                      window_name="Box Measurement Visualization",
                                      width=800, height=600)

# ----------------------------------
# Main Execution
# ----------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calibrate Tau LiDAR Camera and measure box dimensions.")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set the logging level for Open3D.')

    subparsers = parser.add_subparsers(dest='command', help='Available commands. Use a command followed by -h for more help.', required=True)

    # --- Calibrate Intrinsics Sub-parser ---
    parser_intrinsics = subparsers.add_parser('calibrate_intrinsics', help='Perform intrinsic camera calibration.')
    parser_intrinsics.add_argument('--output', default=DEFAULT_INTRINSICS_FILE, 
                                   help=f'Output file for intrinsics (mtx, dist). Default: {DEFAULT_INTRINSICS_FILE}')
    parser_intrinsics.add_argument('--dims', type=str, default="7,6", 
                                   help='Checkerboard inner corner dimensions (cols,rows). Default: "7,6"')
    parser_intrinsics.add_argument('--size', type=float, default=0.025, 
                                   help='Checkerboard square size in meters. Default: 0.025')
    parser_intrinsics.add_argument('--frames', type=int, default=20, 
                                   help='Number of calibration frames to capture. Default: 20')

    # --- Calibrate Depth Sub-parser ---
    parser_depth = subparsers.add_parser('calibrate_depth', help='Perform depth scale calibration.')
    parser_depth.add_argument('--output', default=DEFAULT_DEPTH_SCALE_FILE, 
                              help=f'Output file for depth scale factor. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    parser_depth.add_argument('--distance', type=float, default=1.0, 
                              help='Known distance to target surface in meters. Default: 1.0')
    parser_depth.add_argument('--samples', type=int, default=50, 
                              help='Number of depth samples to average. Default: 50')

    # --- Measure Box Sub-parser ---
    parser_measure = subparsers.add_parser('measure', help='Measure box dimensions using existing calibration data.')
    parser_measure.add_argument('--intrinsics', default=DEFAULT_INTRINSICS_FILE, 
                                help=f'Input file for intrinsics (mtx, dist). Default: {DEFAULT_INTRINSICS_FILE}')
    parser_measure.add_argument('--depth_scale', default=DEFAULT_DEPTH_SCALE_FILE, 
                                help=f'Input file for depth scale factor. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    # Measurement parameters
    parser_measure.add_argument('--plane_thresh', type=float, default=0.01, help='Distance threshold for plane segmentation. Default: 0.01m')
    parser_measure.add_argument('--cluster_eps', type=float, default=0.02, help='DBSCAN epsilon for clustering. Default: 0.02m')
    parser_measure.add_argument('--cluster_min_points', type=int, default=100, help='DBSCAN min_points for clustering. Default: 100')


    # --- Full Run (Calibrate All then Measure) Sub-parser ---
    parser_full = subparsers.add_parser('full_run', help='Perform full calibration (intrinsics and depth) and then measure a box.')
    parser_full.add_argument('--intrinsics_output', default=DEFAULT_INTRINSICS_FILE, 
                             help=f'Output file for intrinsics. Default: {DEFAULT_INTRINSICS_FILE}')
    parser_full.add_argument('--depth_scale_output', default=DEFAULT_DEPTH_SCALE_FILE, 
                             help=f'Output file for depth scale. Default: {DEFAULT_DEPTH_SCALE_FILE}')
    # Intrinsic params for full_run
    parser_full.add_argument('--dims', type=str, default="7,6", help='Checkerboard dims (cols,rows). Default: "7,6"')
    parser_full.add_argument('--size', type=float, default=0.025, help='Checkerboard square size (m). Default: 0.025')
    parser_full.add_argument('--frames', type=int, default=20, help='Number of intrinsic frames. Default: 20')
    # Depth params for full_run
    parser_full.add_argument('--known_distance', type=float, default=1.0, help='Known distance for depth cal (m). Default: 1.0')
    parser_full.add_argument('--samples', type=int, default=50, help='Number of depth samples. Default: 50')
    # Measurement params for full_run
    parser_full.add_argument('--plane_thresh', type=float, default=0.01, help='Plane segmentation threshold (m). Default: 0.01')
    parser_full.add_argument('--cluster_eps', type=float, default=0.02, help='Clustering epsilon (m). Default: 0.02')
    parser_full.add_argument('--cluster_min_points', type=int, default=100, help='Clustering min_points. Default: 100')


    args = parser.parse_args()

    # Set Open3D log level
    if args.log_level == 'DEBUG':
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    elif args.log_level == 'INFO':
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    elif args.log_level == 'WARNING':
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    elif args.log_level == 'ERROR':
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


    # --- Command Dispatch ---
    if args.command == 'calibrate_intrinsics':
        try:
            dims = tuple(map(int, args.dims.split(',')))
            if len(dims) != 2:
                raise ValueError("Checkerboard dimensions must be two comma-separated integers (cols,rows).")
        except ValueError as e:
            print(f"Error parsing checkerboard dimensions: {e}")
            return
        calibrate_intrinsics(args.output, checkerboard_dims=dims, square_size=args.size, num_frames=args.frames)
    
    elif args.command == 'calibrate_depth':
        calibrate_depth_scale(args.output, known_distance=args.distance, num_samples=args.samples)

    elif args.command == 'measure':
        mtx, dist, scale_factor = None, None, None
        # Load intrinsics
        if not os.path.exists(args.intrinsics):
            print(f"Error: Intrinsics file not found at {args.intrinsics}")
            print(f"Please run 'calibrate_intrinsics' first or provide a valid file path.")
            return
        try:
            data = np.load(args.intrinsics)
            mtx = data['mtx']
            dist = data['dist']
            print(f"Loaded intrinsics from {args.intrinsics}")
        except Exception as e:
            print(f"Error loading intrinsics from {args.intrinsics}: {e}")
            return

        # Load depth scale
        if not os.path.exists(args.depth_scale):
            print(f"Error: Depth scale file not found at {args.depth_scale}")
            print(f"Please run 'calibrate_depth' first or provide a valid file path.")
            return
        try:
            with open(args.depth_scale, 'r') as f:
                scale_data = json.load(f)
                scale_factor = scale_data['depth_scale_factor']
            print(f"Loaded depth scale factor ({scale_factor:.5f}) from {args.depth_scale}")
        except Exception as e:
            print(f"Error loading depth scale from {args.depth_scale}: {e}")
            return
        
        if mtx is not None and scale_factor is not None:
            measure_box_dimensions(mtx, dist, scale_factor, 
                                   plane_dist_thresh=args.plane_thresh, 
                                   cluster_eps=args.cluster_eps, 
                                   cluster_min_points=args.cluster_min_points)
        else:
            print("Measurement cannot proceed due to missing calibration data.")

    elif args.command == 'full_run':
        print("--- Starting Full Run: Intrinsic Calibration ---")
        try:
            dims = tuple(map(int, args.dims.split(',')))
            if len(dims) != 2:
                raise ValueError("Checkerboard dimensions must be two comma-separated integers (cols,rows).")
        except ValueError as e:
            print(f"Error parsing checkerboard dimensions for full_run: {e}")
            return
            
        mtx, dist = calibrate_intrinsics(args.intrinsics_output, 
                                         checkerboard_dims=dims, 
                                         square_size=args.size, 
                                         num_frames=args.frames)
        if mtx is None:
            print("Intrinsic calibration failed. Aborting full run.")
            return
        
        print("\n--- Full Run: Depth Scale Calibration ---")
        scale_factor = calibrate_depth_scale(args.depth_scale_output, 
                                             known_distance=args.known_distance, 
                                             num_samples=args.samples)
        if scale_factor is None:
            print("Depth scale calibration failed. Aborting full run.")
            return

        print("\n--- Full Run: Measuring Box ---")
        measure_box_dimensions(mtx, dist, scale_factor,
                               plane_dist_thresh=args.plane_thresh, 
                               cluster_eps=args.cluster_eps, 
                               cluster_min_points=args.cluster_min_points)

if __name__ == "__main__":
    main()
