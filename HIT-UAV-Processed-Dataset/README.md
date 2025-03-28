# HIT-UAV Processed Dataset

本数据集基于 **HIT-UAV 数据集**（哈尔滨工业大学无人机红外数据集）进行二次处理，旨在优化目标检测任务的标注冗余问题，并适配格式塔感知驱动的分区域压缩任务。

---

## 数据集目录结构

```plaintext
your_dataset_root/
├── images/                     # 原始图片（来自 HIT-UAV 数据集）
│   ├── 0_60_30_0_01609.jpg
│   └── ...
├── labels/                     # 原始标注文件（.txt 格式，定向边界框）
│   ├── 0_60_30_0_01609.txt
│   └── ...
├── normalized_images/          # 归一化处理后的图片（Min-Max 归一化到 [0, 255]）
│   ├── 0_60_30_0_01609.jpg
│   └── ...
├── area_labels/                # 基于面积过滤后的标签（阈值：30 像素）
│   ├── 0_60_30_0_01609.txt
│   └── ...
├── angle_labels/               # 基于角度简化的标签（阈值：45°）
│   ├── 0_60_30_0_01609.txt
│   └── ...
├── final_labels/               # 合并后的最终标签（DBSCAN聚类，eps=20）
│   ├── 0_60_30_0_01609.txt
│   └── ...
└── final_dataset/              # 最终划分数据集（按 7:1.5:1.5 划分）
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

---

## 数据处理流程

### 1. 归一化处理（normalized_images）
```plaintext
方法：对 images/ 中的图片进行 Min-Max 归一化，将像素值线性映射到 [0, 255] 范围。
代码示例：
```
```python
normalized_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
```

### 2. 标注过滤（area_labels 和 angle_labels）
```plaintext
面积过滤：过滤面积小于 30 像素的噪声标注（w*h < 30）
角度简化：仅保留角度接近水平（θ ≤ 45°）或垂直（|θ-90°| ≤ 45°）的标注
代码示例：
```
```python
# 面积过滤
if w * h >= 30:
    keep_annotation()
# 角度过滤
if abs(theta) <= 45 or abs(theta - 90) <= 45:
    keep_annotation()
```

### 3. 标注合并（final_labels）
```plaintext
方法：使用 DBSCAN 聚类合并邻近的定向边界框（OBB）
参数：
- eps=20：邻域半径为 20 像素
- min_samples=1：允许单框独立存在
合并策略：计算聚类内所有 OBB 的最小外接矩形（cv2.minAreaRect）
```

---

## 数据集使用示例

### 加载最终数据集
```python
import os
import cv2
import numpy as np

def load_dataset(subset='train'):
    root = 'final_dataset'
    image_dir = os.path.join(root, subset, 'images')
    label_dir = os.path.join(root, subset, 'labels')
    
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # 加载图片
        image = cv2.imread(img_path)
        
        # 加载标注
        with open(label_path, 'r') as f:
            annotations = [line.strip().split() for line in f.readlines()]
        
        yield image, annotations
```

### 可视化标注
```python
import matplotlib.pyplot as plt

def visualize_obb(image_path, label_path):
    img = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        for line in f:
            class_name, xc, yc, w, h, theta = line.strip().split()
            rect = ((float(xc), float(yc)), (float(w), float(h)), float(theta))
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(img, [box], 0, (0,255,0), 2)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
```

---

## 注意事项
```plaintext
1. 标注格式：每个 .txt 文件中的每行表示一个定向边界框，格式为：
   <class_name> <xc> <yc> <w> <h> <theta>
   - (xc, yc)：边界框中心点坐标（绝对像素值）
   - (w, h)：边界框宽度和高度（绝对像素值）
   - theta：旋转角度（度数，从水平方向逆时针旋转）

2. 兼容性：最终数据集（final_dataset）可直接用于 YOLO、Faster R-CNN 等目标检测框架。

3. 引用声明：若使用本数据集，请同时引用原始 HIT-UAV 数据集：
```
```plaintext
@article{hit-uav,
  title={HIT-UAV: A High-altitude Infrared Thermal Dataset for Unmanned Aerial Vehicles},
  author={Zhang, Y., et al.},
  year={2022},
  journal={IEEE Transactions on Geoscience and Remote Sensing}
}
```