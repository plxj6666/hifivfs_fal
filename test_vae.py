import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoencoderKL

# 路径设置
output_dir = Path("/root/HiFiVFS/debug_vae")
output_dir.mkdir(exist_ok=True)

# 加载VAE模型
print("加载VAE模型...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")
vae_scale_factor = 0.18215

# 载入测试图像
test_image_path = "/root/HiFiVFS/data/curated/00110.mp4"
print(f"加载测试视频第一帧: {test_image_path}")
cap = cv2.VideoCapture(test_image_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("无法读取视频帧")
    exit()

# 调整图像大小并转换为PyTorch张量
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (640, 640))  # 调整为640x640
img_tensor = torch.from_numpy(img_resized).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to("cuda")

# 保存原始图像
plt.figure(figsize=(10, 10))
plt.imshow(img_resized)
plt.title("原始图像 (640x640)")
plt.savefig(str(output_dir / "original.png"))

# 编码和解码
print("进行VAE编码和解码...")
with torch.no_grad():
    # 编码到潜在空间
    latents = vae.encode(img_tensor).latent_dist.sample() * vae_scale_factor
    
    # 记录潜在空间统计信息
    print(f"潜在表示形状: {latents.shape}")  # 应该是 [1, 4, 80, 80]
    print(f"潜在表示范围: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
    print(f"潜在表示均值: {latents.mean().item():.4f}")
    print(f"潜在表示标准差: {latents.std().item():.4f}")
    
    # 解码回像素空间
    decoded = vae.decode(latents / vae_scale_factor).sample
    
    # 将解码结果保存为图像
    decoded_img = decoded[0].permute(1, 2, 0).cpu().numpy()
    # 裁剪到0-1范围
    decoded_img = np.clip(decoded_img, 0, 1)

# 显示并保存解码后的图像
plt.figure(figsize=(10, 10))
plt.imshow(decoded_img)
plt.title("VAE解码后的图像")
plt.savefig(str(output_dir / "decoded.png"))

# 保存增强后的图像版本
print("保存增强版本...")
# 对比度增强
enhanced = cv2.convertScaleAbs(
    (decoded_img * 255).astype(np.uint8), 
    alpha=1.2, beta=10
)
enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(enhanced_rgb)
plt.title("增强后的图像")
plt.savefig(str(output_dir / "enhanced.png"))

print(f"图像已保存到 {output_dir}")