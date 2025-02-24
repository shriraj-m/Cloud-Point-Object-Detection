from ultralytics import YOLO
import torch
import cv2
import numpy as np
import open3d as o3d

def detect_2d_bounding_boxes(image_path):
    model = YOLO('runs/detect/train2/weights/best.pt').to('cuda')  # Load model

    results = model.predict(image_path, conf=0.4, imgsz=640)
    result = results[0]

    boxes = result.boxes.xywh
    print(boxes)
    return boxes

def calculate_corners(box):
    x, y, w, h = box.tolist()
    u1 = x - w / 2
    v1 = y - h / 2
    u2 = x + w / 2
    v2 = y + h / 2
    return u1, v1, u2, v2

def get_calibration(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    data['Tr_velo_to_cam'] = data['Tr_velo_to_cam'].reshape(3, 4)
    data['R0_rect'] = data['R0_rect'].reshape(3, 3)
    data['P2'] = data['P2'].reshape(3, 4)
    return data

def calibrate_point_cloud(point_cloud, calib_data):
    points = point_cloud[:, :3]
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = np.dot(calib_data['Tr_velo_to_cam'], points.T).T
    return points

def get_point_cloud_2d(point_cloud, calib_data):
    points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    points = np.dot(calib_data['P2'], points.T).T
    points = points[:, :2] / points[:, 2:3]
    return points


def extract_roi_from_point_cloud(lidar_2d, lidar_3d, boxes):
    rois = []
    rois_2d = []
    for box in boxes:
        u1, v1, u2, v2 = calculate_corners(box)
        in_box = (lidar_2d[:, 0] >= u1) & (lidar_2d[:, 0] < u2) & (lidar_2d[:, 1] >= v1) & (lidar_2d[:, 1] < v2)
        roi = lidar_3d[in_box]
        rois.append(roi)
        rois_2d.append(lidar_2d[in_box])
    return rois, rois_2d

def show_rois(image_path, rois_2d, rois):
    features = []
    features_3d = []
    for roi in rois_2d:
        features.extend(roi)
    features = np.array(features)
    for roi in rois:
        features_3d.extend(roi)
    features_3d = np.array(features_3d)
    
    
    image = cv2.imread(image_path)
    colormap = cv2.COLORMAP_HOT

    depth = features_3d[:, 2]
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    normalized_depth = 1 - (depth - depth_min) / (depth_max - depth_min)

    depth_colored = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), colormap)

    for idx, point in enumerate(features):
        x, y = point[:2]
        color = depth_colored[idx, 0, :].tolist()
        cv2.circle(image, (int(x), int(y)), 1, tuple(color), -1)
    
    cv2.imshow("Lidar Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_bounding_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    for box in boxes:
        u1, v1, u2, v2 = calculate_corners(box)
        u1, v1, u2, v2 = int(u1), int(v1), int(u2), int(v2)
        cv2.rectangle(image, (u1, v1), (u2, v2), (0, 255, 0), 2)
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def project_lidar_to_image(lidar_2d, image):
    for point in lidar_2d:
        x, y = point
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.imshow("Lidar Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

def main():
    image_path = "Data/training/image_2/000035.png"
    boxes = detect_2d_bounding_boxes(image_path)
    display_bounding_boxes(image_path, boxes)
    point_cloud = np.fromfile('Data/training/lidar/dataset/training/000035.bin', dtype=np.float32).reshape(-1, 4)
    calib_data = get_calibration('Data/training/calib/000035.txt')
    point_cloud = calibrate_point_cloud(point_cloud, calib_data)
    point_2d = get_point_cloud_2d(point_cloud, calib_data)
    rois, rois_2d = extract_roi_from_point_cloud(point_2d, point_cloud, boxes)
    project_lidar_to_image(point_2d, cv2.imread(image_path))
    show_rois(image_path, rois_2d, rois)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    o3d.visualization.draw_geometries([pcd])



main()
