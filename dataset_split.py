import os
import random
import shutil

# 数据集根目录
dataset_dir = 'HIT-UAV-Processed-Dataset'
# 图像文件夹路径
images_dir = os.path.join(dataset_dir, 'normalized_images')
# 标签文件夹路径
labels_dir = os.path.join(dataset_dir, 'final_labels')

# 划分比例
train_ratio = 0.7
val_ratio = 0.10
test_ratio = 0.20

# 确保比例之和为 1
assert train_ratio + val_ratio + test_ratio == 1

# 获取所有图像文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
# 打乱图像文件列表
random.shuffle(image_files)

# 过滤掉没有对应标签文件的图片
paired_image_files = []
for image_file in image_files:
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file)
    if os.path.exists(label_path):
        paired_image_files.append(image_file)

# 计算划分的索引
num_images = len(paired_image_files)
train_index = int(num_images * train_ratio)
val_index = train_index + int(num_images * val_ratio)

# 划分数据集
train_files = paired_image_files[:train_index]
val_files = paired_image_files[train_index:val_index]
test_files = paired_image_files[val_index:]

# 创建输出文件夹
output_dir = 'HIT-UAV-Processed-Dataset/final_dataset'
os.makedirs(output_dir, exist_ok=True)
train_images_dir = os.path.join(output_dir, 'train', 'images')
train_labels_dir = os.path.join(output_dir, 'train', 'labels')
val_images_dir = os.path.join(output_dir, 'val', 'images')
val_labels_dir = os.path.join(output_dir, 'val', 'labels')
test_images_dir = os.path.join(output_dir, 'test', 'images')
test_labels_dir = os.path.join(output_dir, 'test', 'labels')

# 创建各个子集的文件夹
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# 复制训练集文件
for file in train_files:
    image_path = os.path.join(images_dir, file)
    label_file = os.path.splitext(file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file)
    shutil.copy(image_path, train_images_dir)
    shutil.copy(label_path, train_labels_dir)

# 复制验证集文件
for file in val_files:
    image_path = os.path.join(images_dir, file)
    label_file = os.path.splitext(file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file)
    shutil.copy(image_path, val_images_dir)
    shutil.copy(label_path, val_labels_dir)

# 复制测试集文件
for file in test_files:
    image_path = os.path.join(images_dir, file)
    label_file = os.path.splitext(file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file)
    shutil.copy(image_path, test_images_dir)
    shutil.copy(label_path, test_labels_dir)

print("数据集划分完成！")