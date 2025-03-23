# 该文件作用为完成所有图片的归一化

import os
import cv2
import numpy as np


def normalize_images(images_dir, labels_dir, output_images_dir):
    # 创建输出图像文件夹
    os.makedirs(output_images_dir, exist_ok=True)

    # 获取 images 文件夹内的所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # 构建图片文件的完整路径
        image_path = os.path.join(images_dir, image_file)
        # 读取图片
        image = cv2.imread(image_path)

        # 进行归一化处理
        normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 构建输出图片的完整路径
        output_image_path = os.path.join(output_images_dir, image_file)
        # 保存归一化后的图片
        cv2.imwrite(output_image_path, normalized_image)

        # 构建对应的标注文件路径
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        output_label_path = os.path.join(os.path.dirname(output_images_dir), 'labels', label_file)

        # 确保输出 labels 文件夹存在
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

        # 复制标注文件到新的位置
        if os.path.exists(label_path):
            with open(label_path, 'r') as src_file, open(output_label_path, 'w') as dst_file:
                dst_file.write(src_file.read())


if __name__ == "__main__":
    # 数据集文件夹路径
    dataset_dir = 'dataset'
    # images 文件夹路径
    images_dir = os.path.join(dataset_dir, 'images')
    # labels 文件夹路径
    labels_dir = os.path.join(dataset_dir, 'labels')
    # 输出归一化图片的文件夹路径
    output_images_dir = os.path.join(dataset_dir, 'normalized_images')

    normalize_images(images_dir, labels_dir, output_images_dir)
    