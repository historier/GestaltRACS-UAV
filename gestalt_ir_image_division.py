import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import time
import numpy as np
from sklearn.metrics import jaccard_score

# 1. 增强型格式塔约束损失函数
# 用处：该损失函数用于训练格式塔模型，通过结合几何约束（均方误差损失）、闭合性约束和对称性约束，使模型预测的边界框更符合格式塔原理，从而更准确地划分图像的重要区域。
# 架构：继承自 nn.Module，包含均方误差损失和一个卷积层与 Sigmoid 激活函数组成的闭合性先验模块。在 forward 方法中计算三种约束的损失并加权求和。
class EnhancedGestaltConstraintLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.closure_prior = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, pred_boxes, true_boxes, images):
        # 几何约束
        true_boxes = true_boxes.squeeze(1) if true_boxes.dim() == 3 else true_boxes
        mse_loss = self.mse_loss(pred_boxes, true_boxes)

        # 闭合性约束
        closure_mask = self.closure_prior(images)
        closure_mask = F.adaptive_avg_pool2d(closure_mask, (1, 1)).squeeze()
        closure_loss = torch.mean(torch.abs(pred_boxes[:, 4] - closure_mask))

        # 对称性约束
        area = pred_boxes[:, 2] * pred_boxes[:, 3]
        area = torch.clamp(area, min=0)
        circle_ratio = 2 * torch.sqrt(area / torch.pi)
        symmetry_loss = torch.mean(torch.abs((pred_boxes[:, 2] + pred_boxes[:, 3])/2 - circle_ratio))

        return mse_loss + 0.1*closure_loss + 0.05*symmetry_loss

# 2. 多模态特征融合网络
# 用处：用于融合多模态特征，通过空间注意力和通道注意力机制，增强特征图中重要区域的特征表示，提高模型对图像重要区域的感知能力。
# 架构：包含空间注意力模块和通道注意力模块。空间注意力模块通过卷积层和 Sigmoid 函数生成空间掩码，通道注意力模块通过自适应平均池化、卷积层和 Sigmoid 函数生成通道掩码，最后将输入特征图与两个掩码相乘得到融合后的特征图。
class MultiModalFeatureFuser(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(base_channels*4, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels*4, base_channels*4//16, 1),
            nn.ReLU(),
            nn.Conv2d(base_channels*4//16, base_channels*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_mask = self.spatial_attn(x)
        channel_mask = self.channel_attn(x)
        return x * spatial_mask * channel_mask

# 3. 单通道图像数据集类
# 用处：用于加载和预处理单通道红外图像数据集，包括图像的读取、转换、变换和标签的读取与归一化。
# 架构：继承自 torch.utils.data.Dataset，实现了 __len__ 和 __getitem__ 方法。__len__ 方法返回数据集的长度，__getitem__ 方法根据索引返回图像和对应的边界框标签。
class SingleChannelIRDataset(Dataset):
    def __init__(self, data_dir, transform=None, normalize=True):
        super().__init__()
        self.data_dir = data_dir
        self.image_files = sorted(os.listdir(os.path.join(data_dir, 'images')))
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'images', self.image_files[idx])
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label_path = os.path.join(self.data_dir, 'labels', self.image_files[idx].replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            parts = line.split()
            coordinates = [float(part) for part in parts[1:6]]
            boxes = torch.tensor(coordinates, dtype=torch.float32).unsqueeze(0)

        if self.normalize:
            boxes = self.normalize_boxes(boxes)

        return image, boxes

    def normalize_boxes(self, boxes):
        img_width, img_height = 640, 512
        boxes = boxes.clone()
        boxes[:, 0] /= img_width
        boxes[:, 1] /= img_height
        boxes[:, 2] /= img_width
        boxes[:, 3] /= img_height
        return boxes

# 4. 增量式训练策略
# 用处：用于增量式训练格式塔模型，通过多个 epoch 迭代训练模型，同时在验证集上评估模型性能，并根据验证损失调整学习率。
# 架构：定义了优化器、损失函数和学习率调度器，使用混合精度训练加速训练过程。在每个 epoch 中，模型在训练集上进行训练，计算损失并更新参数；在验证集上进行评估，计算验证损失和平均交并比（IoU），并根据验证损失调整学习率。
def train_incremental(model, train_loader, val_loader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = EnhancedGestaltConstraintLoss().to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    start_time = time.time()
    print("模型开始训练...")

    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels, inputs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            elapsed_time = batch_end_time - start_time
            print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Batch Loss: {loss.item():.4f}, Batch Time: {batch_time:.4f}s, Elapsed Time: {elapsed_time:.4f}s')

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Epoch Time: {epoch_time:.4f}s')

        if val_loader:
            model.eval()
            val_loss = 0.0
            iou_scores = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels, inputs)
                    val_loss += loss.item()

                    pred_boxes = outputs.cpu().numpy()
                    true_boxes = labels.cpu().numpy()
                    pred_mask = (pred_boxes > 0.5).astype(int)
                    true_mask = (true_boxes > 0.5).astype(int)
                    iou = jaccard_score(true_mask.flatten(), pred_mask.flatten())
                    iou_scores.append(iou)
            mean_val_loss = val_loss / len(val_loader)
            mean_iou = np.mean(iou_scores)
            print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {mean_val_loss:.4f}, Validation Mean IoU: {mean_iou:.4f}')
            scheduler.step(mean_val_loss)
            # 打印当前学习率
            print(f'Epoch {epoch + 1}/{epochs}, Current Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]:.8f}')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"模型训练结束，总用时: {total_time:.4f}s")

# 5. 完整模型架构
# 用处：这是基于格式塔原理的红外图像区域划分模型，用于预测图像中重要区域的边界框。
# 架构：包含特征提取器、多模态特征融合网络和边界框预测器。特征提取器通过多个卷积层和批量归一化层提取图像特征；多模态特征融合网络对提取的特征进行融合；边界框预测器通过全连接层预测边界框的坐标和其他属性。
class OptimizedGestaltIRModel(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.correlation_miner = MultiModalFeatureFuser(base_channels*4)
        self.box_predictor = nn.Sequential(
            nn.Linear(base_channels*4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        correlated = self.correlation_miner(features)
        pooled = F.adaptive_avg_pool2d(correlated, (1, 1)).squeeze()
        # 调整 pooled 的形状
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
        return self.box_predictor(pooled)

# 6. 主程序入口
if __name__ == "__main__":
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 30

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = SingleChannelIRDataset('HIT-UAV-Processed-Dataset/final_dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = SingleChannelIRDataset('HIT-UAV-Processed-Dataset/final_dataset/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = OptimizedGestaltIRModel()
    train_incremental(model, train_loader, val_loader, epochs=EPOCHS)

    # 保存模型
    torch.save(model.state_dict(), 'Models/optimized_gestalt_model_3.0.pth')
    