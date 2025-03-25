# 该文件作用为对处理后的数据集进行一些可视化

import os
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns

def visualize_random_samples(dataset_dir, num_samples=5):
    # 随机可视化数据集中的样本（原始 vs 处理后）
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        image_dir = os.path.join(dataset_dir, subset, 'images')
        label_dir = os.path.join(dataset_dir, subset, 'labels')
        
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        random.shuffle(image_files)
        
        for i in range(num_samples):
            img_file = image_files[i]
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
            
            # 加载原始图片和标注
            img = cv2.imread(img_path)
            original_height, original_width = img.shape[:2]
            
            # 加载处理后的标注
            with open(label_path, 'r') as f:
                annotations = [line.strip().split() for line in f.readlines()]
            
            # 绘制处理后的边界框
            for ann in annotations:
                class_name, xc, yc, w, h, theta = ann
                xc, yc, w, h, theta = map(float, [xc, yc, w, h, theta])
                
                # 转换为整数坐标
                rect = ((xc, yc), (w, h), theta)
                box = cv2.boxPoints(rect).astype(int)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            
            # 显示结果
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"{subset} - {img_file}")
            plt.axis('off')
            plt.show()

def collect_dataset_stats(dataset_dir):
    # 收集数据集参数统计信息
    stats = []
    subsets = ['train', 'val', 'test']
    
    for subset in subsets:
        label_dir = os.path.join(dataset_dir, subset, 'labels')
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        
        for label_file in label_files:
            with open(os.path.join(label_dir, label_file), 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                
                class_name, xc, yc, w, h, theta = parts[0], *map(float, parts[1:6])
                area = w * h
                stats.append({
                    'subset': subset,
                    'class': class_name,
                    'xc': xc,
                    'yc': yc,
                    'w': w,
                    'h': h,
                    'theta': theta,
                    'area': area
                })
    
    return pd.DataFrame(stats)

def visualize_dataset_stats(df):
    # 分别绘制参数统计图表
    # 角度分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['theta'], bins=30, kde=True, color='teal')
    plt.title('Angle Distribution (Degrees)')
    plt.xlabel('θ')
    plt.ylabel('Count')
    plt.show()

    # 面积分布（对数坐标）
    plt.figure(figsize=(10, 6))
    sns.histplot(df['area'], bins=30, kde=True, log_scale=True, color='coral')
    plt.title('Area Distribution (Log Scale)')
    plt.xlabel('Area (Pixels²)')
    plt.ylabel('Count')
    plt.show()

    # 中心点位置
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='xc', y='yc', data=df, alpha=0.5, color='darkblue')
    plt.title('Center Point Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # 宽高比例
    plt.figure(figsize=(10, 6))
    df['aspect_ratio'] = df['w'] / df['h']
    sns.histplot(df['aspect_ratio'], bins=30, kde=True, color='forestgreen')
    plt.title('Aspect Ratio (Width/Height)')
    plt.xlabel('Ratio')
    plt.ylabel('Count')
    plt.show()


if __name__ == "__main__":
    # 数据集处理前后对比
    visualize_random_samples('HIT-UAV-Processed-Dataset/final_dataset', num_samples=3)
    # 统计图与参数
    df = collect_dataset_stats('HIT-UAV-Processed-Dataset/final_dataset')
    visualize_dataset_stats(df)