# 检查数据集是否有不配对的文件并处理

import os

def check_and_clean_dataset(dataset_path):
    sub_folders = ['train', 'val', 'test']
    for sub_folder in sub_folders:
        image_folder = os.path.join(dataset_path, sub_folder, 'images')
        label_folder = os.path.join(dataset_path, sub_folder, 'labels')

        if not os.path.exists(image_folder) or not os.path.exists(label_folder):
            print(f"文件夹 {image_folder} 或 {label_folder} 不存在，跳过此子文件夹。")
            continue

        image_files = set([os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.endswith('.jpg')])
        label_files = set([os.path.splitext(file)[0] for file in os.listdir(label_folder) if file.endswith('.txt')])

        # 找出没有对应标注文件的图片
        images_without_labels = image_files - label_files
        for image_name in images_without_labels:
            image_path = os.path.join(image_folder, f"{image_name}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"删除了无对应标注的图片: {image_path}")

        # 找出没有对应图片的标注文件
        labels_without_images = label_files - image_files
        for label_name in labels_without_images:
            label_path = os.path.join(label_folder, f"{label_name}.txt")
            if os.path.exists(label_path):
                os.remove(label_path)
                print(f"删除了无对应图片的标注文件: {label_path}")

if __name__ == "__main__":
    check_and_clean_dataset(dataset_path='HIT-UAV-Processed-Dataset/final_dataset')