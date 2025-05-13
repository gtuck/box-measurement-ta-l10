import numpy as np
import cv2
import tau_lidar_camera as tau
import open3d as o3d

# ----------------------------------
# Calibration Functions
# ----------------------------------

def calibrate_intrinsics(checkerboard_dims=(7, 6), square_size=0.025, num_frames=20):
    """
    Perform intrinsic calibration using a checkerboard pattern.
    - checkerboard_dims: (cols, rows) of inner corners
    - square_size: size of each square in meters
    - num_frames: number of successful detections to collect
    Returns: camera matrix (mtx), distortion coefficients (dist)
    """
    objp = np.zeros((checkerboard_dims[1] * checkerboard_dims[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []
    cam = tau.TauCamera()
    cam.start()
    collected = 0
    print("Starting intrinsic calibration...")
    while collected < num_frames:
        gray, depth = cam.read()
        found, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)
        if found:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
            collected += 1
            print(f"Captured calibration frame {collected}/{num_frames}")
        cv2.imshow('Intrinsics Cal', gray)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    if not ret:
        raise RuntimeError("Intrinsic calibration failed")
    print("Calibration successful.")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist.ravel())
    return mtx, dist

def calibrate_depth_scale(known_distance=1.0, num_samples=50):
    """
    Compute scale factor to convert raw depth units to accurate meters.
    - known_distance: distance from camera to a flat surface in meters
    """
    cam = tau.TauCamera()
    cam.start()
    scales = []
    print("Measuring depth scale...")
    for i in range(num_samples):
        gray, raw_depth = cam.read()
        depth_m = raw_depth.astype(np.float32) * 0.001  # raw → meters
        h, w = depth_m.shape
        region = depth_m[h//2-5:h//2+5, w//2-5:w//2+5]
        measured = np.median(region)
        scales.append(known_distance / measured)
    cam.stop()

    scale_factor = float(np.mean(scales))
    print(f"Depth scale factor: {scale_factor:.5f}")
    return scale_factor

# ----------------------------------
# Point-Cloud Processing & Measurement
# ----------------------------------

def depth_to_pointcloud(depth, intrinsics):
    h, w = depth.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    valid = depth > 0
    z = depth[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy

    pts = np.stack((x, y, z), axis=-1)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

def segment_plane(pcd, dist_thresh=0.005, ransac_n=3, num_iter=1000):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=ransac_n,
        num_iterations=num_iter
    )
    return pcd.select_by_index(inliers), pcd.select_by_index(inliers, invert=True)

def cluster_and_fit(pcd, eps=0.02, min_points=100):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.size == 0:
        raise RuntimeError("No clusters found")
    largest = np.bincount(labels[labels >= 0]).argmax()
    idx = np.where(labels == largest)[0]
    cluster = pcd.select_by_index(idx)
    obb = cluster.get_oriented_bounding_box()
    obb.color = (1, 0, 0)
    return obb

def capture_frame(depth_scale=1.0):
    cam = tau.TauCamera()
    cam.start()
    gray, raw_depth = cam.read()
    cam.stop()
    depth_m = raw_depth.astype(np.float32) * 0.001 * depth_scale
    return gray, depth_m, cam.intrinsics

def measure_box(depth_scale=1.0):
    gray, depth, intrinsics = capture_frame(depth_scale)
    pcd = depth_to_pointcloud(depth, intrinsics)
    _, rest = segment_plane(pcd)
    obb = cluster_and_fit(rest)

    dims = obb.extent * 1000  # meters → mm
    print("Measured box dimensions (mm):")
    print(f"  Width : {dims[0]:.1f}")
    print(f"  Height: {dims[1]:.1f}")
    print(f"  Depth : {dims[2]:.1f}")

    o3d.visualization.draw_geometries([obb, pcd])

if __name__ == "__main__":
    # 1. Calibrate intrinsics
    mtx, dist = calibrate_intrinsics()
    # 2. Calibrate depth scale at 1.0 m
    scale = calibrate_depth_scale(known_distance=1.0)
    # 3. Measure your box
    measure_box(depth_scale=scale)
