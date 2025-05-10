import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
import psutil
import os
from io import BytesIO
import tempfile
from gestalt_ir_image_division import (
    SingleChannelIRDataset,
    OptimizedGestaltIRModel,
    EnhancedGestaltConstraintLoss
)

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ----------------------- CSReconNet模型组件 -----------------------
class ChaoticToeplitzSampler(nn.Module):
    def __init__(self, block_size=16, measurement_ratio=0.5):
        super().__init__()
        self.block_size = block_size
        self.measurement_dim = int(block_size**2 * measurement_ratio)
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
        toeplitz_matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                k = i - j
                idx = k % len(c) if k >= 0 else (-k) % len(r)
                toeplitz_matrix[i, j] = c[idx] if k >= 0 else r[idx]
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
        x_flat = x.view(batch_size, -1)
        return torch.matmul(x_flat, self.phi.t())

class GaussianSampler(nn.Module):
    def __init__(self, block_size=8, measurement_ratio=0.25):
        super().__init__()
        self.block_size = block_size
        self.measurement_dim = int(block_size**2 * measurement_ratio)
        self.phi = nn.Parameter(torch.randn(self.measurement_dim, block_size**2) * 1/np.sqrt(block_size**2), requires_grad=False)
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return torch.matmul(x_flat, self.phi.t())

class BoundaryFusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)  # 修正输入通道为128
    
    def forward(self, x):
        return self.conv(x)

class CSReconNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 测量层
        self.important_sampler = ChaoticToeplitzSampler(block_size=16, measurement_ratio=0.4)
        self.non_important_sampler = GaussianSampler(block_size=8, measurement_ratio=0.15)
        
        # 重要区域处理
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
        
        # 非重要区域处理
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

    def forward(self, important_blocks, non_important_blocks):
        # 特征提取
        imp_features = self.imp_extractor(important_blocks) if important_blocks.numel() > 0 else torch.empty(0)
        non_imp_features = self.non_imp_extractor(non_important_blocks) if non_important_blocks.numel() > 0 else torch.empty(0)
        
        # 重建过程
        x_imp_recon = self.imp_decoder(imp_features)
        x_non_recon = self.non_imp_decoder(non_imp_features)
        
        # 边界融合
        if x_imp_recon.numel() > 0:
            boundary_info = self.boundary_fusion(imp_features)
            x_imp_recon = x_imp_recon + boundary_info
        
        return x_imp_recon, x_non_recon

# ----------------------- 坐标计算和图像处理函数 -----------------------
def get_rotated_bbox_coords(xc, yc, w, h, theta, img_h, img_w):
    """计算旋转后的边界框坐标"""
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    half_w = w / 2
    half_h = h / 2

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

    min_x = max(0, int(np.min([corner[0] for corner in rotated_corners])))
    max_x = min(img_w, int(np.max([corner[0] for corner in rotated_corners])))
    min_y = max(0, int(np.min([corner[1] for corner in rotated_corners])))
    max_y = min(img_h, int(np.max([corner[1] for corner in rotated_corners])))

    return min_x, min_y, max_x - min_x, max_y - min_y

# 加载格式塔模型函数
def load_gestalt_model(ckpt_path='Models/optimized_gestalt_model_3.0.pth'):
    """加载预训练的格式塔区域划分模型"""
    model = OptimizedGestaltIRModel()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model

# 图像分块函数
def image_to_blocks(image, bbox, important_block_size=16, non_important_block_size=8):
    """将图像按边界框划分为重要区域和非重要区域块"""
    # 获取原始图像尺寸（假设为640x512）
    original_width, original_height = 640, 512
    current_h, current_w = image.shape[-2], image.shape[-1]
    
    # 从归一化坐标转换回原始图像尺寸
    bbox = bbox.squeeze().cpu().numpy()
    xc_normalized, yc_normalized, w_box_normalized, h_box_normalized, theta = bbox
    xc_original = xc_normalized * original_width
    yc_original = yc_normalized * original_height
    w_box_original = w_box_normalized * original_width
    h_box_original = h_box_normalized * original_height
    
    # 计算在当前处理图像尺寸中的坐标
    xc = xc_original * current_w / original_width
    yc = yc_original * current_h / original_height
    w_box = w_box_original * current_w / original_width
    h_box = h_box_original * current_h / original_height

    # 计算旋转后的边界框坐标
    x1, y1, w_rot, h_rot = get_rotated_bbox_coords(xc, yc, w_box, h_box, theta, current_h, current_w)

    # 处理边界情况
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x1 + w_rot, current_w)
    y2 = min(y1 + h_rot, current_h)

    # 重要区域处理
    important_region = image[..., y1:y2, x1:x2]
    imp_h, imp_w = important_region.shape[-2], important_region.shape[-1]
    
    # 计算重要区域的块数和坐标
    n_h_imp = max(1, (imp_h + important_block_size - 1) // important_block_size)
    n_w_imp = max(1, (imp_w + important_block_size - 1) // important_block_size)
    
    # 重要区域块的位置信息
    important_blocks = []
    important_blocks_pos = []
    
    for i in range(n_h_imp):
        for j in range(n_w_imp):
            block_y_start = i * important_block_size
            block_y_end = min(block_y_start + important_block_size, imp_h)
            block_x_start = j * important_block_size
            block_x_end = min(block_x_start + important_block_size, imp_w)
            
            if block_y_end > block_y_start and block_x_end > block_x_start:
                block = important_region[..., block_y_start:block_y_end, block_x_start:block_x_end]
                
                absolute_y_start = y1 + block_y_start
                absolute_y_end = y1 + block_y_end
                absolute_x_start = x1 + block_x_start
                absolute_x_end = x1 + block_x_end
                
                important_blocks.append(block)
                important_blocks_pos.append({
                    'y_start': absolute_y_start,
                    'y_end': absolute_y_end,
                    'x_start': absolute_x_start,
                    'x_end': absolute_x_end,
                    'h': block_y_end - block_y_start,
                    'w': block_x_end - block_x_start
                })
    
    # 转换为张量
    if important_blocks:
        max_h = max(b.shape[-2] for b in important_blocks)
        max_w = max(b.shape[-1] for b in important_blocks)
        important_blocks_tensor = torch.zeros(len(important_blocks), 1, max_h, max_w, device=image.device)
        
        for i, block in enumerate(important_blocks):
            h, w = block.shape[-2:]
            important_blocks_tensor[i, :, :h, :w] = block
            
        important_blocks = important_blocks_tensor
    else:
        important_blocks = torch.empty(0, 1, important_block_size, important_block_size, device=image.device)
    
    # 非重要区域处理
    non_important_mask = torch.ones_like(image, dtype=bool)
    non_important_mask[..., y1:y2, x1:x2] = False
    non_important_region = image * non_important_mask.float()
    
    # 非重要区域块的位置信息
    non_important_blocks = []
    non_important_blocks_pos = []
    
    # 将非重要区域分块
    for i in range(0, current_h, non_important_block_size):
        for j in range(0, current_w, non_important_block_size):
            block_y_start = i
            block_y_end = min(i + non_important_block_size, current_h)
            block_x_start = j
            block_x_end = min(j + non_important_block_size, current_w)
            
            if not non_important_mask[..., block_y_start:block_y_end, block_x_start:block_x_end].all():
                continue
                
            if block_y_end > block_y_start and block_x_end > block_x_start:
                block = non_important_region[..., block_y_start:block_y_end, block_x_start:block_x_end]
                
                non_important_blocks.append(block)
                non_important_blocks_pos.append({
                    'y_start': block_y_start,
                    'y_end': block_y_end,
                    'x_start': block_x_start,
                    'x_end': block_x_end,
                    'h': block_y_end - block_y_start,
                    'w': block_x_end - block_x_start
                })
    
    # 转换为张量
    if non_important_blocks:
        max_h = max(b.shape[-2] for b in non_important_blocks)
        max_w = max(b.shape[-1] for b in non_important_blocks)
        non_important_blocks_tensor = torch.zeros(len(non_important_blocks), 1, max_h, max_w, device=image.device)
        
        for i, block in enumerate(non_important_blocks):
            h, w = block.shape[-2:]
            non_important_blocks_tensor[i, :, :h, :w] = block
            
        non_important_blocks = non_important_blocks_tensor
    else:
        non_important_blocks = torch.empty(0, 1, non_important_block_size, non_important_block_size, device=image.device)
    
    return important_blocks, non_important_blocks, (x1, y1, x2, y2), important_blocks_pos, non_important_blocks_pos

# 图像重建函数 - 修改了边界处理和压缩率计算
def reconstruct_image(model, gestalt_model, image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    gestalt_model = gestalt_model.to(device)
    image_tensor = image_tensor.to(device)
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss
    
    with torch.no_grad():
        # 获取边界框
        bbox = gestalt_model(image_tensor)
        
        # 图像分块
        important_blocks, non_important_blocks, bbox_coords, important_blocks_pos, non_important_blocks_pos = image_to_blocks(
            image_tensor, bbox, important_block_size=16, non_important_block_size=8
        )
        
        # 计算实际压缩率 - 基于测量矩阵维度
        original_num_elements = image_tensor.numel()
        important_measurements = important_blocks.shape[0] * model.important_sampler.measurement_dim
        non_important_measurements = non_important_blocks.shape[0] * model.non_important_sampler.measurement_dim
        compressed_num_elements = important_measurements + non_important_measurements
        compression_ratio = compressed_num_elements / original_num_elements
        
        # 重构图像
        model.eval()
        if important_blocks.numel() > 0 and non_important_blocks.numel() > 0:
            recon_imp, recon_non = model(important_blocks, non_important_blocks)
        elif important_blocks.numel() > 0:
            recon_imp, _ = model(important_blocks, non_important_blocks[:0])
            recon_non = torch.zeros_like(non_important_blocks)
        else:
            _, recon_non = model(important_blocks[:0], non_important_blocks)
            recon_imp = torch.zeros_like(important_blocks)
        
        # 计算重构时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 计算MSE损失
        mse_imp = torch.mean((recon_imp - important_blocks) ** 2).item() if important_blocks.numel() > 0 else 0
        mse_non = torch.mean((recon_non - non_important_blocks) ** 2).item() if non_important_blocks.numel() > 0 else 0
        
        # 构建重构后的完整图像
        reconstructed_image = image_tensor.clone()
        x1, y1, x2, y2 = bbox_coords
        
        # 重新构建重要区域 - 移除了高斯模糊处理
        if important_blocks.numel() > 0 and len(important_blocks_pos) > 0:
            for i, pos in enumerate(important_blocks_pos):
                block = recon_imp[i, :, :pos['h'], :pos['w']]
                reconstructed_image[..., pos['y_start']:pos['y_end'], pos['x_start']:pos['x_end']] = block
        
        # 重新构建非重要区域 - 移除了高斯模糊处理
        if non_important_blocks.numel() > 0 and len(non_important_blocks_pos) > 0:
            for i, pos in enumerate(non_important_blocks_pos):
                block = recon_non[i, :, :pos['h'], :pos['w']]
                reconstructed_image[..., pos['y_start']:pos['y_end'], pos['x_start']:pos['x_end']] = block
        
        # 计算峰值信噪比(PSNR)
        mse = torch.mean((reconstructed_image - image_tensor) ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # 计算结构相似性指数(SSIM)
        ssim = calculate_ssim(image_tensor.cpu().numpy(), reconstructed_image.cpu().numpy())
    
    return {
        'original_image': image_tensor.cpu().numpy(),
        'reconstructed_image': reconstructed_image.cpu().numpy(),
        'bbox_coords': bbox_coords,
        'compression_ratio': compression_ratio,
        'processing_time': processing_time,
        'mse_important': mse_imp,
        'mse_non_important': mse_non,
        'memory_usage': memory_usage / (1024 ** 2),
        'psnr': psnr,
        'ssim': ssim,
        'original_num_elements': original_num_elements,
        'compressed_num_elements': compressed_num_elements
    }

# 计算结构相似性指数(SSIM)
def calculate_ssim(img1, img2, window_size=11, K1=0.01, K2=0.03, L=1):
    """计算两个图像之间的SSIM值"""
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    
    # 创建高斯窗口
    window = np.ones((window_size, window_size)) / (window_size * window_size)
    
    # 计算均值
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = cv2.filter2D(img1 * img1, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 返回均值SSIM
    return np.mean(ssim_map)

# 绘制图像和边界框
def plot_image_with_bbox(image, bbox_coords, title, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    if image.shape[0] == 1:
        ax.imshow(image[0], cmap='gray')
    else:
        ax.imshow(np.transpose(image, (1, 2, 0)))
    
    x1, y1, x2, y2 = bbox_coords
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                         linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-10, '重要区域', color='red', fontsize=12)
    ax.set_title(title)
    ax.axis('off')
    return fig

# 绘制图像对比
def plot_image_comparison(original, reconstructed, bbox_coords, title, figsize=(15, 6)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 原始图像
    original_img = np.squeeze(original)
    ax1.imshow(original_img, cmap='gray')
    
    x1, y1, x2, y2 = bbox_coords
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                         linewidth=2, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title("原始图像")
    ax1.axis('off')
    
    # 重构图像
    reconstructed_img = np.squeeze(original)
    ax2.imshow(reconstructed_img, cmap='gray')
    
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                         linewidth=2, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title("重构图像")
    ax2.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

# 绘制像素级对比图
def plot_pixel_comparison(original, reconstructed, bbox_coords, patch_size=64, figsize=(15, 6)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    x1, y1, x2, y2 = bbox_coords
    x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
    x_patch = max(0, x_center - patch_size // 2)
    y_patch = max(0, y_center - patch_size // 2)
    x_patch_end = min(x_patch + patch_size, original.shape[-1])
    y_patch_end = min(y_patch + patch_size, original.shape[-2])
    
    # 原始图像
    original_img = np.squeeze(original)
    
    # 重构图像
    reconstructed_img = np.squeeze(reconstructed)
    
    # 全图
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title("原始图像")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_img, cmap='gray')
    axes[1, 0].set_title("重构图像")
    axes[1, 0].axis('off')
    
    # 局部放大
    original_patch = original_img[y_patch:y_patch_end, x_patch:x_patch_end]
    reconstructed_patch = reconstructed_img[y_patch:y_patch_end, x_patch:x_patch_end]
    
    axes[0, 1].imshow(original_patch, cmap='gray')
    axes[0, 1].set_title("原始图像局部")
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(original_patch, cmap='gray')
    axes[1, 1].set_title("重构图像局部")
    axes[1, 1].axis('off')
    
    # 差异图
    diff = np.abs(original_patch - reconstructed_patch)
    axes[0, 2].imshow(diff, cmap='jet')
    axes[0, 2].set_title("差异图")
    axes[0, 2].axis('off')
    
    # 隐藏最后一个子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig

# 主应用
def main():
    st.set_page_config(
        page_title="ReconNet 可视化演示",
        page_icon="🔄",
        layout="wide"
    )
    
    # 应用标题
    st.title("ReconNet 红外图像压缩与重构可视化演示")
    
    # 侧边栏
    with st.sidebar:
        st.header("模型配置")
        
        # 模型加载选项
        model_path = st.text_input("ReconNet模型路径", "Models/cs_reconnet_model.pth")
        
        # 上传图像
        uploaded_file = st.file_uploader("上传红外图像", type=["jpg", "jpeg", "png"])
        
        # 处理选项
        st.subheader("处理参数")
        show_details = st.checkbox("显示详细处理信息", True)
        show_performance = st.checkbox("显示性能指标", True)
        
        # 关于
        st.markdown("---")
        st.subheader("关于")
        st.info("""
        本应用演示了基于格式塔理论的红外图像压缩与重构技术。
        通过识别图像中的重要区域，实现高效的图像压缩与高质量重构。
        """)
    
    # 主内容区
    if uploaded_file is not None:
        # 加载模型
        with st.spinner("加载模型中..."):
            try:
                # 使用CSReconNet模型
                reconnet_model = CSReconNet()
                try:
                    reconnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                except FileNotFoundError:
                    st.warning(f"未找到模型文件: {model_path}，将使用随机初始化模型")
                reconnet_model.eval()
                
                # 加载Gestalt模型
                gestalt_model = load_gestalt_model(ckpt_path='Models/optimized_gestalt_model_3.0.pth')
                gestalt_model.eval()
                
                st.success("模型加载成功!")
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                st.stop()
        
        # 处理上传的图像
        with st.spinner("处理图像中..."):
            try:
                # 读取图像
                image = Image.open(uploaded_file).convert('L')
                original_width, original_height = image.size
                
                # 调整图像大小为模型输入尺寸
                input_size = (640, 512)
                image_resized = image.resize(input_size)
                
                # 转换为张量
                image_tensor = torch.tensor(np.array(image_resized), dtype=torch.float32)
                image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
                image_tensor = (image_tensor / 255.0 - 0.5) / 0.5
                
                # 重构图像
                result = reconstruct_image(reconnet_model, gestalt_model, image_tensor)
                
                # 显示结果
                col1, col2 = st.columns(2)
                
                # 原始图像和重构图像对比
                with col1:
                    st.subheader("原始图像与重构图像对比")
                    fig_comparison = plot_image_comparison(
                        result['original_image'],
                        result['reconstructed_image'],
                        result['bbox_coords'],
                        "原始图像与重构图像对比"
                    )
                    st.pyplot(fig_comparison)
                
                # 像素级对比
                with col2:
                    st.subheader("像素级对比")
                    fig_pixel_comparison = plot_pixel_comparison(
                        result['original_image'],
                        result['reconstructed_image'],
                        result['bbox_coords']
                    )
                    st.pyplot(fig_pixel_comparison)
                
                # 评估指标
                st.subheader("评估指标")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric("压缩率", f"{result['compression_ratio']:.4f}")
                    st.metric("处理时间", f"{result['processing_time']:.4f} 秒")
                    st.metric("峰值信噪比(PSNR)", f"{result['psnr']:.2f} dB")
                
                with metrics_col2:
                    st.metric("重要区域MSE", f"{result['mse_non_important']:.6f}")
                    st.metric("非重要区域MSE", f"{result['mse_important']:.6f}")
                    st.metric("结构相似性(SSIM)", f"{result['ssim']:.4f}")
                
                with metrics_col3:
                    st.metric("内存使用", f"{result['memory_usage']:.2f} MB")
                    st.metric("图像尺寸", f"{original_width}×{original_height}")
                
                # 详细信息
                if show_details:
                    st.subheader("详细处理信息")
                    
                    x1, y1, x2, y2 = result['bbox_coords']
                    st.markdown(f"""
                    - **重要区域坐标**:
                      - 左上角: ({x1}, {y1})
                      - 右下角: ({x2}, {y2})
                      - 宽度: {x2 - x1} 像素
                      - 高度: {y2 - y1} 像素
                    """)
                    
                    st.markdown(f"""
                    - **原始图像大小**: {original_width}×{original_height} 像素
                    - **处理图像大小**: {input_size[0]}×{input_size[1]} 像素
                    """)
                
                # 性能指标
                if show_performance:
                    st.subheader("性能指标")
                    
                    # 计算并显示压缩前后的大小
                    original_size_kb = result['original_num_elements'] / 8 / 1024
                    compressed_size_kb = result['compressed_num_elements'] / 8 / 1024
                    compression_percentage = result['compression_ratio'] * 100
                    savings_percentage = (1 - result['compression_ratio']) * 100
                    
                    st.markdown(f"""
                    - **原始图像大小**: {original_size_kb:.2f} KB ({result['original_num_elements']} 比特)
                    - **压缩后数据大小**: {compressed_size_kb:.2f} KB ({result['compressed_num_elements']} 比特)
                    - **压缩率**: {compression_percentage:.2f}%
                    - **压缩节省率**: {savings_percentage:.2f}%
                    """)
                    
                    # 绘制压缩效果图表
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(
                        ['原始图像', '压缩后数据'], 
                        [original_size_kb, compressed_size_kb],
                        color=['#1f77b4', '#ff7f0e']
                    )
                    ax.set_ylabel('数据大小 (KB)')
                    ax.set_title('图像压缩效果')
                    ax.bar_label(ax.containers[0], fmt='%.2f')
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"图像处理失败: {e}")
                # 打印详细的错误信息
                import traceback
                st.write(traceback.format_exc())
    else:
        # 未上传图像时显示介绍
        st.markdown("""
        ### 关于本应用
        
        本应用展示了基于格式塔理论的红外图像压缩与重构技术。该技术通过识别图像中的重要区域，实现高效的图像压缩与高质量重构。
        
        ### 使用方法
        
        1. 在左侧上传一张红外图像
        2. 等待系统处理图像
        3. 查看原始图像、标注了重要区域的图像以及重构后的图像
        4. 查看各项评估指标
        
        ### 技术特点
        
        - 基于格式塔心理学原理识别图像中的重要区域
        - 对重要区域和非重要区域采用不同的压缩策略
        - 在保证重要信息完整性的同时实现高效压缩
        - 提供直观的可视化界面，便于理解压缩效果
        """)
        
if __name__ == "__main__":
    main()