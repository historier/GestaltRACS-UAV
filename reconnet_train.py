import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import time
import psutil
import os
import numpy as np
from gestalt_ir_image_division import (
    SingleChannelIRDataset,
    OptimizedGestaltIRModel,
    EnhancedGestaltConstraintLoss
)

# 测量矩阵生成模块
class ChaoticToeplitzSampler(nn.Module):
    def __init__(self, block_size=16, measurement_ratio=0.5):
        super().__init__()
        self.block_size = block_size
        self.measurement_dim = int(block_size**2 * measurement_ratio)
        # 生成测量矩阵并注册为缓冲区，确保随模型移动到相同设备
        self.register_buffer('phi', self.generate_01_chaotic_toeplitz())
        
    def generate_logistic_chaotic_sequence(self, length=256, r=3.99):
        x = np.random.rand()
        sequence = []
        for _ in range(length):
            x = r * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)
    
    def generate_chaotic_toeplitz_matrix(self, sequence):
        n = self.block_size**2
        m = self.measurement_dim
        c = sequence[:n]
        r = sequence[1:m+1]
        
        # 创建一个m行n列的零矩阵
        toeplitz_matrix = np.zeros((m, n))
        
        # 填充Toeplitz矩阵
        for i in range(m):
            for j in range(n):
                k = i - j
                if k >= 0:
                    # 使用列向量c，但确保索引不越界
                    idx = k % len(c)  # 循环使用序列，防止索引越界
                    toeplitz_matrix[i, j] = c[idx]
                else:
                    # 使用行向量r，但确保索引不越界
                    idx = (-k) % len(r)  # 循环使用序列，防止索引越界
                    toeplitz_matrix[i, j] = r[idx]
        
        return toeplitz_matrix
    
    def binarize_matrix(self, matrix):
        threshold = np.mean(matrix)
        return (matrix >= threshold).astype(np.float32)
    
    def generate_01_chaotic_toeplitz(self):
        seq = self.generate_logistic_chaotic_sequence()
        toeplitz = self.generate_chaotic_toeplitz_matrix(seq)
        return torch.from_numpy(self.binarize_matrix(toeplitz)).float()
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # (B, 256) for 16x16 block
        # 由于self.phi已注册为缓冲区，会自动与输入x在同一设备上
        y = torch.matmul(x_flat, self.phi.t())  # (B, m)
        return y

class GaussianSampler(nn.Module):
    def __init__(self, block_size=8, measurement_ratio=0.25):
        super().__init__()
        self.block_size = block_size
        self.measurement_dim = int(block_size**2 * measurement_ratio)
        self.phi = nn.Parameter(torch.randn(self.measurement_dim, block_size**2) * 1/np.sqrt(block_size**2), requires_grad=False)
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # (B, 64) for 8x8 block
        y = torch.matmul(x_flat, self.phi.t())  # (B, m)
        return y

# 边界融合模块 - 修改卷积层输入通道数为128
class BoundaryFusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改输入通道数为128，匹配imp_features的通道数
        self.conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# 完整的压缩感知ReconNet模型
class CSReconNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 测量层
        self.important_sampler = ChaoticToeplitzSampler(block_size=16, measurement_ratio=0.4)
        self.non_important_sampler = GaussianSampler(block_size=8, measurement_ratio=0.15)
        
        # 重要区域特征提取与重建
        self.imp_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.imp_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        )
        
        # 非重要区域特征提取与重建
        self.non_imp_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.non_imp_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        )
        
        # 边界融合模块
        self.boundary_fusion = BoundaryFusionModule()

    def forward(self, important_blocks, non_important_blocks, return_measurements=False):
        # 存储分块信息（用于重建拼接）
        self.imp_block_info = {
            'shape': important_blocks.shape,
            'n_blocks': important_blocks.shape[0] if important_blocks.numel() > 0 else 0
        }
        self.non_imp_block_info = {
            'shape': non_important_blocks.shape,
            'n_blocks': non_important_blocks.shape[0] if non_important_blocks.numel() > 0 else 0
        }

        # 测量过程
        y_imp = self.important_sampler(important_blocks) if important_blocks.numel() > 0 else torch.empty(0, self.important_sampler.measurement_dim)
        y_non = self.non_important_sampler(non_important_blocks) if non_important_blocks.numel() > 0 else torch.empty(0, self.non_important_sampler.measurement_dim)

        if return_measurements:
            return y_imp, y_non

        # 特征提取
        imp_features = self.imp_extractor(important_blocks) if important_blocks.numel() > 0 else torch.empty(0, 128, 16, 16)
        non_imp_features = self.non_imp_extractor(non_important_blocks) if non_important_blocks.numel() > 0 else torch.empty(0, 32, 8, 8)

        # 重建过程
        x_imp_recon = self.imp_decoder(imp_features) if imp_features.numel() > 0 else important_blocks
        x_non_recon = self.non_imp_decoder(non_imp_features) if non_imp_features.numel() > 0 else non_important_blocks

        # 边界融合处理
        if x_imp_recon.numel() > 0:
            boundary_info = self.boundary_fusion(imp_features)
            x_imp_recon = x_imp_recon + boundary_info

        return x_imp_recon, x_non_recon

def load_gestalt_model(ckpt_path='Models/optimized_gestalt_model_3.0.pth'):
    """加载预训练的格式塔区域划分模型"""
    model = OptimizedGestaltIRModel()
    # 修复FutureWarning提示，明确指定weights_only参数
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()  # 设置为评估模式
    return model

def get_rotated_bbox_coords(xc, yc, w, h, theta, img_h, img_w):
    """
    根据中心点坐标、宽度、高度和旋转角度计算旋转后的边界框坐标
    """
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    half_w = w / 2
    half_h = h / 2

    # 计算四个角点的坐标
    corners = [
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ]

    rotated_corners = []
    for corner in corners:
        x = corner[0] * cos_theta - corner[1] * sin_theta + xc
        y = corner[0] * sin_theta + corner[1] * cos_theta + yc
        rotated_corners.append([x, y])

    # 计算边界框的最小和最大坐标
    min_x = max(0, int(np.min([corner[0] for corner in rotated_corners])))
    max_x = min(img_w, int(np.max([corner[0] for corner in rotated_corners])))
    min_y = max(0, int(np.min([corner[1] for corner in rotated_corners])))
    max_y = min(img_h, int(np.max([corner[1] for corner in rotated_corners])))

    return min_x, min_y, max_x - min_x, max_y - min_y

def image_to_blocks_with_info(image, bbox, important_block_size=16, non_important_block_size=8):
    """将图像按边界框划分为重要区域和非重要区域块，并返回块的坐标信息"""
    # 获取原始图像尺寸（假设为640x512）
    original_width, original_height = 640, 512
    
    # 获取当前处理的图像张量尺寸（可能是经过resize后的尺寸）
    current_h, current_w = image.shape[-2], image.shape[-1]
    
    # 从归一化坐标转换回原始图像尺寸（640x512）
    bbox = bbox.squeeze().cpu().numpy()
    xc_normalized, yc_normalized, w_box_normalized, h_box_normalized, theta = bbox
    
    # 反归一化到原始图像尺寸
    xc_original = xc_normalized * original_width
    yc_original = yc_normalized * original_height
    w_box_original = w_box_normalized * original_width
    h_box_original = h_box_normalized * original_height
    
    # 计算在当前处理图像尺寸中的坐标
    xc = xc_original * current_w / original_width
    yc = yc_original * current_h / original_width
    w_box = w_box_original * current_w / original_width
    h_box = h_box_original * current_h / original_height

    # 计算旋转后的边界框坐标
    x1, y1, w_rot, h_rot = get_rotated_bbox_coords(xc, yc, w_box, h_box, theta, current_h, current_w)

    # 处理边界情况
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x1 + w_rot, current_w)
    y2 = min(y1 + h_rot, current_h)

    # 提取重要区域并分块（16x16）
    important_region = image[..., y1:y2, x1:x2]
    imp_h, imp_w = important_region.shape[-2], important_region.shape[-1]
    n_h_imp = imp_h // important_block_size
    n_w_imp = imp_w // important_block_size

    # 处理无法均匀分块的情况
    if imp_h % important_block_size != 0:
        important_region = important_region[..., :n_h_imp * important_block_size, :]
    if imp_w % important_block_size != 0:
        important_region = important_region[..., :, :n_w_imp * important_block_size]

    if n_h_imp == 0 or n_w_imp == 0:
        important_blocks = torch.empty(0, 1, important_block_size, important_block_size)
    else:
        important_blocks = important_region.reshape(
            -1, 1, important_block_size, important_block_size, n_h_imp, n_w_imp
        ).permute(0, 4, 5, 1, 2, 3).contiguous().reshape(-1, 1, important_block_size, important_block_size)

    # 提取非重要区域并分块（8x8）
    non_important_mask = torch.ones_like(image, dtype=bool)
    non_important_mask[..., y1:y2, x1:x2] = False
    non_important_region = image * non_important_mask.float()

    non_imp_h, non_imp_w = non_important_region.shape[-2], non_important_region.shape[-1]
    n_h_non = non_imp_h // non_important_block_size
    n_w_non = non_imp_w // non_important_block_size

    # 处理无法均匀分块的情况
    if non_imp_h % non_important_block_size != 0:
        non_important_region = non_important_region[..., :n_h_non * non_important_block_size, :]
    if non_imp_w % non_important_block_size != 0:
        non_important_region = non_important_region[..., :, :n_w_non * non_important_block_size]

    if n_h_non == 0 or n_w_non == 0:
        non_important_blocks = torch.empty(0, 1, non_important_block_size, non_important_block_size)
    else:
        non_important_blocks = non_important_region.reshape(
            -1, 1, non_important_block_size, non_important_block_size, n_h_non, n_w_non
        ).permute(0, 4, 5, 1, 2, 3).contiguous().reshape(-1, 1, non_important_block_size, non_important_block_size)

    # 记录块的坐标信息（用于重建时拼接）
    imp_coords = []
    for i in range(n_h_imp):
        for j in range(n_w_imp):
            y_start = y1 + i * important_block_size
            x_start = x1 + j * important_block_size
            imp_coords.append((y_start, x_start))
    
    non_imp_coords = []
    for i in range(n_h_non):
        for j in range(n_w_non):
            y_start = i * non_important_block_size
            x_start = j * non_important_block_size
            # 确保非重要区域坐标不与重要区域重叠
            if not (y_start >= y1 and y_start < y2 and x_start >= x1 and x_start < x2):
                non_imp_coords.append((y_start, x_start))
    
    return (important_blocks, imp_coords), (non_important_blocks, non_imp_coords)

def reconstruct_from_blocks(x_imp_recon, x_non_recon, imp_coords, non_imp_coords, original_size=(224, 224), important_block_size=16, non_important_block_size=8):
    """从分块重建完整图像，并处理块之间的边缘"""
    # 初始化重建图像
    reconstructed_image = torch.zeros((1, 1, *original_size), device=x_imp_recon.device)
    
    # 拼接重要区域
    if x_imp_recon.numel() > 0:
        for i, (y_start, x_start) in enumerate(imp_coords):
            block = x_imp_recon[i:i+1]
            reconstructed_image[..., y_start:y_start+important_block_size, x_start:x_start+important_block_size] = block
    
    # 拼接非重要区域
    if x_non_recon.numel() > 0:
        for i, (y_start, x_start) in enumerate(non_imp_coords):
            block = x_non_recon[i:i+1]
            reconstructed_image[..., y_start:y_start+non_important_block_size, x_start:x_start+non_important_block_size] = block
    
    # 边缘处理：使用高斯模糊平滑块边界
    kernel_size = 3
    sigma = 0.5
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    reconstructed_image = gaussian_blur(reconstructed_image)
    
    return reconstructed_image

def train_cs_reconnet(model, gestalt_model, train_loader, val_loader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    gestalt_model = gestalt_model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion_imp = nn.MSELoss()  # 重要区域损失
    criterion_non = nn.MSELoss()  # 非重要区域损失
    measurement_criterion = nn.MSELoss()  # 测量一致性损失

    print("CSReconNet 开始训练...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, (images, _) in enumerate(train_loader):  # 格式塔模型不需要标签
            batch_start_time = time.time()
            images = images.to(device)
            optimizer.zero_grad()

            # 1. 使用格式塔模型获取边界框
            with torch.no_grad():
                bboxes = gestalt_model(images)

            # 2. 图像分块处理并获取坐标信息
            batch_imp_blocks, batch_non_imp_blocks = [], []
            batch_imp_coords, batch_non_imp_coords = [], []
            
            for img, bbox in zip(images, bboxes):
                (imp_blocks, imp_coords), (non_imp_blocks, non_imp_coords) = image_to_blocks_with_info(img.unsqueeze(0), bbox.unsqueeze(0))
                batch_imp_blocks.append(imp_blocks)
                batch_non_imp_blocks.append(non_imp_blocks)
                batch_imp_coords.append(imp_coords)
                batch_non_imp_coords.append(non_imp_coords)

            # 3. 压缩感知测量和重建
            total_loss = 0
            for i in range(len(batch_imp_blocks)):
                imp_blocks = batch_imp_blocks[i].to(device)
                non_imp_blocks = batch_non_imp_blocks[i].to(device)
                imp_coords = batch_imp_coords[i]
                non_imp_coords = batch_non_imp_coords[i]

                # 测量过程
                y_imp, y_non = model(imp_blocks, non_imp_blocks, return_measurements=True)
                
                # 重建过程
                imp_recon, non_imp_recon = model(imp_blocks, non_imp_blocks)
                
                # 计算损失
                if imp_blocks.numel() > 0:
                    # 重建损失
                    loss_imp = criterion_imp(imp_recon, imp_blocks)
                    # 测量一致性损失（验证重建结果是否能产生相似的测量值）
                    y_imp_recon = model.important_sampler(imp_recon)
                    loss_measurement_imp = measurement_criterion(y_imp_recon, y_imp)
                    total_loss += loss_imp + 0.1 * loss_measurement_imp

                if non_imp_blocks.numel() > 0:
                    loss_non = criterion_non(non_imp_recon, non_imp_blocks)
                    y_non_recon = model.non_important_sampler(non_imp_recon)
                    loss_measurement_non = measurement_criterion(y_non_recon, y_non)
                    total_loss += 0.5 * (loss_non + 0.1 * loss_measurement_non)  # 非重要区域权重降低

            if total_loss > 0:
                # 反向传播
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

            # 打印训练日志
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            elapsed_time = batch_end_time - start_time
            print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss:.4f} | "
                  f"Batch Time: {batch_time:.4f}s | "
                  f"Elapsed Time: {elapsed_time:.4f}s")

        #  epoch结束统计
        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} 结束 |  epoch Loss: {epoch_loss / len(train_loader):.4f} | "
              f"Epoch Time: {epoch_end_time - epoch_start_time:.4f}s")

    print("CSReconNet 训练结束！")
    return model

def evaluate_cs_reconnet(model, gestalt_model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    gestalt_model = gestalt_model.to(device)
    model.eval()

    total_compression_ratio = 0
    total_processing_time = 0
    total_imp_loss = 0
    total_non_imp_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_images = 0

    criterion_imp = nn.MSELoss()
    criterion_non = nn.MSELoss()

    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        has_metrics = True
    except ImportError:
        has_metrics = False
        print("警告: 缺少skimage库，无法计算PSNR和SSIM指标")

    for images, _ in val_loader:
        images = images.to(device)
        start_time = time.time()

        # 1. 使用格式塔模型获取边界框
        with torch.no_grad():
            bboxes = gestalt_model(images)

        # 2. 图像分块处理并获取坐标信息
        for img, bbox in zip(images, bboxes):
            (imp_blocks, imp_coords), (non_imp_blocks, non_imp_coords) = image_to_blocks_with_info(img.unsqueeze(0), bbox.unsqueeze(0))
            imp_blocks = imp_blocks.to(device)
            non_imp_blocks = non_imp_blocks.to(device)

            # 计算原始图像的总元素数
            original_num_elements = img.numel()

            # 计算压缩后测量值的总元素数
            compressed_num_elements = 0
            if imp_blocks.numel() > 0:
                y_imp = model.important_sampler(imp_blocks)
                compressed_num_elements += y_imp.numel()
            if non_imp_blocks.numel() > 0:
                y_non = model.non_important_sampler(non_imp_blocks)
                compressed_num_elements += y_non.numel()

            # 计算压缩率
            compression_ratio = compressed_num_elements / original_num_elements

            # 3. 重建图像
            imp_recon, non_imp_recon = model(imp_blocks, non_imp_blocks)
            
            # 4. 从分块重建完整图像
            reconstructed_image = reconstruct_from_blocks(
                imp_recon, non_imp_recon, imp_coords, non_imp_coords, 
                original_size=img.shape[-2:]
            )

            # 计算损失
            if imp_blocks.numel() > 0:
                loss_imp = criterion_imp(imp_recon, imp_blocks)
                total_imp_loss += loss_imp.item()
            else:
                loss_imp = 0

            if non_imp_blocks.numel() > 0:
                loss_non = criterion_non(non_imp_recon, non_imp_blocks)
                total_non_imp_loss += loss_non.item()
            else:
                loss_non = 0

            # 计算PSNR和SSIM
            if has_metrics:
                # 将图像张量转换为numpy数组
                original_np = img.cpu().detach().squeeze().numpy()
                reconstructed_np = reconstructed_image.cpu().detach().squeeze().numpy()
                
                # 计算PSNR
                psnr_val = psnr(original_np, reconstructed_np, data_range=1.0)
                total_psnr += psnr_val
                
                # 计算SSIM
                ssim_val = ssim(original_np, reconstructed_np, data_range=1.0)
                total_ssim += ssim_val

            total_compression_ratio += compression_ratio
            num_images += 1

        end_time = time.time()
        total_processing_time += end_time - start_time

    # 计算平均值
    if num_images > 0:
        average_compression_ratio = total_compression_ratio / num_images
        average_processing_time = total_processing_time / num_images
        average_imp_loss = total_imp_loss / num_images
        average_non_imp_loss = total_non_imp_loss / num_images
        
        print(f"平均压缩率: {average_compression_ratio:.4f}")
        print(f"平均处理时间: {average_processing_time:.4f} 秒")
        print(f"重要区域平均损失: {average_imp_loss:.4f}")
        print(f"非重要区域平均损失: {average_non_imp_loss:.4f}")
        
        if has_metrics:
            average_psnr = total_psnr / num_images
            average_ssim = total_ssim / num_images
            print(f"平均PSNR: {average_psnr:.4f} dB")
            print(f"平均SSIM: {average_ssim:.4f}")

    return average_compression_ratio, average_processing_time, average_imp_loss, average_non_imp_loss

# 主程序入口
if __name__ == "__main__":
    # 1. 配置参数
    ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 640, 512
    IMAGE_SIZE = (224, 224)  # 模型输入尺寸
    BATCH_SIZE = 4
    EPOCHS = 20

    # 2. 加载格式塔模型
    gestalt_model = load_gestalt_model()

    # 3. 构建CSReconNet模型
    model = CSReconNet()

    # 4. 准备数据集
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = SingleChannelIRDataset(
        'HIT-UAV-Processed-Dataset/final_dataset/train',
        transform=transform,
        normalize=True  # 格式塔模型需要归一化坐标
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = SingleChannelIRDataset(
        'HIT-UAV-Processed-Dataset/final_dataset/val',
        transform=transform,
        normalize=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 开始训练
    model = train_cs_reconnet(
        model,
        gestalt_model,
        train_loader,
        val_loader,
        epochs=EPOCHS
    )

    # 6. 保存CSReconNet模型
    torch.save(model.state_dict(), 'Models/cs_reconnet_model.pth')

    # 7. 验证模块
    evaluate_cs_reconnet(model, gestalt_model, val_loader)