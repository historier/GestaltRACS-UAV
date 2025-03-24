# 该文件为调整标注，整合bbox，使标注更加符合格式塔规则，具体为：基于面积阈值过滤标注文件+角度简化处理+使用DBSCAN聚类合并边界框

import os
import numpy as np
from sklearn.cluster import DBSCAN
import cv2

def filter_annotations_by_area(annotation_dir, output_dir, area_threshold=30):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 基于面积阈值过滤标注文件
    for txt_file in os.listdir(annotation_dir):
        if txt_file.endswith('.txt'):
            file_path = os.path.join(annotation_dir, txt_file)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            filtered_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6:  # 检查是否包含角度θ
                    continue
                class_name, xc, yc, w, h, theta = parts[0], *map(float, parts[1:6])
                area = w * h
                if area >= area_threshold:
                    filtered_lines.append(f"{class_name} {xc} {yc} {w} {h} {theta}\n")

            output_file_path = os.path.join(output_dir, txt_file)
            with open(output_file_path, 'w') as f:
                f.writelines(filtered_lines)

            if filtered_lines != lines:
                print(f"Filtered {txt_file}: {len(lines) - len(filtered_lines)} boxes removed")
    
def filter_annotations_by_angle(annotation_dir, output_dir, angle_threshold=45):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    for txt_file in os.listdir(annotation_dir):
        if txt_file.endswith('.txt'):
            file_path = os.path.join(annotation_dir, txt_file)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            filtered_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                class_name, xc, yc, w, h, theta = parts[0], *map(float, parts[1:6])
                # 角度绝对值小于阈值（仅保留接近水平/垂直的框）
                if abs(theta) <= angle_threshold or abs(theta - 90) <= angle_threshold:
                    filtered_lines.append(line)

            output_file_path = os.path.join(output_dir, txt_file)
            with open(output_file_path, 'w') as f:
                f.writelines(filtered_lines)

            if filtered_lines != lines:
                print(f"Filtered {txt_file}: {len(lines) - len(filtered_lines)} boxes removed")
    


def merge_obbs_with_dbscan(annotation_dir, output_dir, eps=20, min_samples=1):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    for txt_file in os.listdir(annotation_dir):
        if txt_file.endswith('.txt'):
            file_path = os.path.join(annotation_dir, txt_file)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                continue

            # 提取OBB参数（中心点 + 宽高 + 角度）
            features = []
            obbs = []
            classes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                class_name, xc, yc, w, h, theta = parts[0], *map(float, parts[1:6])
                features.append([xc, yc, w, h, theta])  # 多特征输入
                obbs.append([xc, yc, w, h, theta])
                classes.append(class_name)

            if not features:
                continue

            # 标准化特征（可选，避免量纲影响）
            features = np.array(features)
            features[:, :2] /= 640  # 中心点坐标归一化到图像宽度
            features[:, 2:4] /= 640  # 宽高归一化到图像宽度
            features[:, 4] /= 180  # 角度归一化到[-1, 1]

            # 聚类
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
            labels = db.labels_

            # 合并同一聚类的OBB
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                if label not in clusters:
                    clusters[label] = {'obbs': [], 'class': classes[i]}
                clusters[label]['obbs'].append(obbs[i])

            # 生成合并后的OBB（使用最小外接矩形）
            merged_obbs = []
            for label in clusters:
                cluster = clusters[label]
                obbs_np = np.array(cluster['obbs'])

                # 转换为图像坐标并计算最小外接矩形
                rects = []
                for obb in obbs_np:
                    xc, yc, w, h, theta = obb
                    rect = ((xc, yc), (w, h), theta)
                    rects.append(rect)

                # 合并多个旋转矩形（需要特殊算法，此处简化）
                # 计算所有点的凸包，再求最小外接矩形
                points = []
                for rect in rects:
                    box = cv2.boxPoints(rect)
                    points.extend(box)
                points = np.array(points)
                if len(points) == 0:
                    continue

                merged_rect = cv2.minAreaRect(points)
                (xc, yc), (w, h), theta = merged_rect

                merged_obbs.append({
                    'class': cluster['class'],
                    'xc': xc,
                    'yc': yc,
                    'w': w,
                    'h': h,
                    'theta': theta
                })

            # 写入文件
            new_lines = []
            for obb in merged_obbs:
                new_lines.append(f"{obb['class']} {obb['xc']:.2f} {obb['yc']:.2f} {obb['w']:.2f} {obb['h']:.2f} {obb['theta']:.2f}\n")

            output_file_path = os.path.join(output_dir, txt_file)
            with open(output_file_path, 'w') as f:
                f.writelines(new_lines)

            if len(new_lines) != len(lines):
                print(f"Merged {txt_file}: {len(lines) - len(new_lines)} boxes removed")

def visualize_obbs(image_path, label_path, image_width=640, image_height=512):
    #可视化结果方便调整
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        
        class_name = parts[0]
        xc, yc, w, h, theta = map(float, parts[1:6])
        
        # 转换为 OpenCV 所需的矩形参数
        rect = ((xc, yc), (w, h), theta)
        box = cv2.boxPoints(rect)  # 获得矩形的四个顶点坐标
        box = np.int0(box)  # 转换为整数坐标
        
        # 确保所有点在图像范围内
        box[:, 0] = np.clip(box[:, 0], 0, image_width)
        box[:, 1] = np.clip(box[:, 1], 0, image_height)
        
        # 绘制矩形和标签
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        cv2.putText(img, class_name, (int(xc), int(yc)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Oriented BBoxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #面积
    filter_annotations_by_area(annotation_dir='HIT-UAV-Processed-Dataset/labels', output_dir='HIT-UAV-Processed-Dataset/area_labels',area_threshold=30)
    #角度
    filter_annotations_by_angle(annotation_dir='HIT-UAV-Processed-Dataset/area_labels',output_dir='HIT-UAV-Processed-Dataset/angle_labels', angle_threshold=45)
    #合并边界框
    merge_obbs_with_dbscan(annotation_dir='HIT-UAV-Processed-Dataset/angle_labels',output_dir='HIT-UAV-Processed-Dataset/final_labels', eps=20)
    #可视化
    visualize_obbs(image_path='HIT-UAV-Processed-Dataset/normalized_images/0_60_30_0_01609.jpg', label_path='HIT-UAV-Processed-Dataset/labels/0_60_30_0_01609.txt')
    visualize_obbs(image_path='HIT-UAV-Processed-Dataset/normalized_images/0_60_30_0_01609.jpg', label_path='HIT-UAV-Processed-Dataset/final_labels/0_60_30_0_01609.txt')