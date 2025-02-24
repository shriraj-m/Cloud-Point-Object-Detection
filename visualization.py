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
    return boxes, result

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    matched_pairs = []  # Stores (pred_idx, gt_idx, IoU) for matched boxes
    unmatched_preds = []  # Stores unmatched predictions (false positives)
    unmatched_gts = set(range(len(gt_boxes)))  # Track unmatched ground truths

    for pred_idx, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred, gt)  # Compute IoU
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx in unmatched_gts:
            matched_pairs.append((pred_idx, best_gt_idx, best_iou))
            unmatched_gts.remove(best_gt_idx)  # Mark ground truth as matched
        else:
            unmatched_preds.append(pred_idx)  # No good match, it's a false positive

    return matched_pairs, unmatched_preds, list(unmatched_gts)

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
    colormap = cv2.COLORMAP_JET

    depth = features_3d[:, 2]
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)

    depth_colored = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), colormap)

    for idx, point in enumerate(features):
        x, y = point[:2]
        color = depth_colored[idx, 0, :].tolist()
        cv2.circle(image, (int(x), int(y)), 1, tuple(color), -1)
    
    cv2.imshow("Lidar Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_bounding_boxes(image_path, boxes, result):
    image = cv2.imread(image_path)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, box in enumerate(boxes):
        x, y, w, h = box.tolist()
        u1, v1, u2, v2 = calculate_corners(box)
        u1, v1, u2, v2 = int(u1), int(v1), int(u2), int(v2)
        conf = result.boxes.conf[i].item()
        label = result.names[result.boxes.cls[i].item()]
        cv2.rectangle(img_bgr, (u1, v1), (u2, v2), (0, 255, 0), 2)
        label_text = f"{label} {conf:.2f}"  # Format label and confidence score
        cv2.putText(img_bgr, label_text, (int(x - w / 2), int(y - h / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green text
    cv2.imshow("Detected Objects", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def project_lidar_to_image(lidar_2d, depth_map, image):

    depth = depth_map
    print(np.min(depth), np.max(depth))
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    
    depth_colored = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)

    for idx, point in enumerate(lidar_2d):
        x, y = point
        color = depth_colored[idx, 0, :].tolist()
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (int(x), int(y)), 1, tuple(color), -1)
    cv2.imshow("Lidar Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rois_3d(rois):
    feature = []
    for roi in rois:
        feature.extend(roi)
    feature = np.array(feature)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(feature)
    o3d.visualization.draw_geometries([pcd])
    

    

def main():
    x = 400
    x = str(x).zfill(6)
    image_path = f"Data/training/image_2/{x}.png"
    boxes, result = detect_2d_bounding_boxes(image_path)
    display_bounding_boxes(image_path, boxes, result)
    point_cloud = np.fromfile(f'Data/training/lidar/dataset/training/{x}.bin', dtype=np.float32).reshape(-1, 4)
    calib_data = get_calibration(f'Data/training/calib/{x}.txt')
    point_cloud = calibrate_point_cloud(point_cloud, calib_data)
    point_cloud = point_cloud[point_cloud[:, 2] > 0]
    point_2d = get_point_cloud_2d(point_cloud, calib_data)
    rois, rois_2d = extract_roi_from_point_cloud(point_2d, point_cloud, boxes)
    depth_map = point_cloud[:, 2]
    project_lidar_to_image(point_2d, depth_map, cv2.imread(image_path))
    show_rois(image_path, rois_2d, rois)

    rois_3d(rois)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    o3d.visualization.draw_geometries([pcd])

    for i in range(500, 700):
        img_testing_path = f"Data/training/images/val/{str(i).zfill(6)}.png"
        label_testing_path = f"Data/training/labels/val/{str(i).zfill(6)}.txt"
        boxes, results = detect_2d_bounding_boxes(img_testing_path)
        real_boxes = []
        matched_pairs = []
        unmatched_preds = []
        unmatched_gts = []
        image = cv2.imread(img_testing_path)
        with open(label_testing_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                x_center, y_center, width, height = map(float, parts)
                x_min = int((x_center - width / 2) * image.shape[1])
                y_min = int((y_center - height / 2) * image.shape[0])
                x_max = int((x_center + width / 2) * image.shape[1])
                y_max = int((y_center + height / 2) * image.shape[0])
                real_boxes.append([x_min, y_min, x_max, y_max])
        tmp_matched_pairs, tmp_unmatched_preds, tmp_unmatched_gts = match_boxes(boxes, real_boxes)

        matched_pairs.extend(tmp_matched_pairs)
        unmatched_preds.extend(tmp_unmatched_preds)
        unmatched_gts.extend(tmp_unmatched_gts)

    precision = len(matched_pairs) / (len(matched_pairs) + len(unmatched_preds))
    recall = len(matched_pairs) / (len(matched_pairs) + len(unmatched_gts))
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")


main()
