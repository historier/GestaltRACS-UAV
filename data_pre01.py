# 该文件作用为检查数据集是否归一化和整理图片与标注

import os
import cv2
import numpy as np
import pandas as pd
import shutil
import xml.etree.ElementTree as ET


# 检查原数据集是否完成归一化
def batch_check_normalization(data_dir):
    image_dir = os.path.join(data_dir, 'train')
    image_files = os.listdir(image_dir)

    stats = []
    for file in image_files[:10]:  # 随机选取10张检查
        path = os.path.join(image_dir, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        stats.append({
            'file': file,
            'dtype': img.dtype,
            'min': np.min(img),
            'max': np.max(img),
            'mean': np.mean(img),
            'std': np.std(img)
        })

    return pd.DataFrame(stats)


def check_all_normalized(data_dir):
    df = batch_check_normalization(data_dir)
    # 检查是否所有图像都满足归一化标准
    all_normalized = all(
        (row['dtype'] == np.float32) and (0 <= row['min'] <= 1) and (0 <= row['max'] <= 1)
        for _, row in df.iterrows()
    )
    if all_normalized:
        print("已全部归一化")
    else:
        print("未归一化")



def convert_xml_to_txt(annotations_dir, images_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # 遍历所有 XML 标注文件
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            try:
                # 以二进制模式读取文件
                with open(xml_path, 'rb') as file:
                    content = file.read()
                    # 去除 BOM
                    if content.startswith(b'\xef\xbb\xbf'):
                        content = content[3:]
                    try:
                        xml_content = content.decode('utf-8').lstrip()
                    except UnicodeDecodeError:
                        xml_content = content.decode('gbk').lstrip()
                if not xml_content.startswith('<'):
                    print(f"{xml_file} 可能不是有效的 XML 文件，跳过处理。")
                    continue
                root = ET.fromstring(xml_content)
            except Exception as e:
                print(f"解析 {xml_file} 时出错: {e}")
                continue
            # 解析 XML 文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图片文件名
            filename = root.find('filename').text
            # 构建图片文件的完整路径
            image_ext = '.jpg'
            image_path = os.path.join(images_dir, filename + image_ext)
            # 复制图片到输出目录
            output_image_path = os.path.join(output_images_dir, filename + image_ext)
            shutil.copyfile(image_path, output_image_path)

            # 构建对应的标注文本文件路径
            label_filename = filename + '.txt'
            output_label_path = os.path.join(output_labels_dir, label_filename)

            with open(output_label_path, 'w') as label_file:
                # 遍历每个目标对象
                for obj in root.findall('object'):
                    # 获取目标类别
                    class_name = obj.find('name').text
                    # 获取旋转边界框信息
                    robndbox = obj.find('robndbox')
                    cx = float(robndbox.find('cx').text)
                    cy = float(robndbox.find('cy').text)
                    w = float(robndbox.find('w').text)
                    h = float(robndbox.find('h').text)
                    angle = float(robndbox.find('angle').text)

                    # 将类别和边界框信息写入文本文件
                    label_file.write(f"{class_name} {cx} {cy} {w} {h} {angle}\n")
    


if __name__ == "__main__":
    # 检查图片是否归一化
    data_dir = 'HIT-UAV-Infrared-Thermal-Dataset-main/rotate_json'
    check_all_normalized(data_dir)
  
    annotations_dir = 'HIT-UAV-Infrared-Thermal-Dataset-main/rotate_xml/Annotations'
    images_dir = 'HIT-UAV-Infrared-Thermal-Dataset-main/rotate_xml/JPEGImages'
    output_dir = 'dataset'

    convert_xml_to_txt(annotations_dir, images_dir, output_dir)

    
   