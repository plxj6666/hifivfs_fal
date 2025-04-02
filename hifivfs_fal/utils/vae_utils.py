import torch
import numpy as np
import cv2
from diffusers import AutoencoderKL
from torch import nn
from .detect_align_face import detect_align_face

@torch.no_grad()
def encode_with_vae(vae, pixel_values, scale_factor):
    """使用VAE编码像素值到潜在空间"""
    # 将输入移到与VAE相同的设备上
    pixel_values = pixel_values.to(vae.device)
    
    # 编码为潜在表示
    latent_dist = vae.encode(pixel_values).latent_dist
    latent = latent_dist.sample() * scale_factor
    latent = latent.cpu()
    return latent

@torch.no_grad()
def decode_with_vae(vae: AutoencoderKL, latents: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Decodes VAE latents back to pixel space.
    
    Args:
        vae (AutoencoderKL): VAE模型实例
        latents (torch.Tensor): 潜在空间表示
        scale_factor (float): VAE缩放因子
    
    Returns:
        torch.Tensor: 解码后的像素值，范围[-1, 1]
    """
    latents = latents.float()
    latents_unscaled = latents / scale_factor
    image = vae.decode(latents_unscaled).sample
    # 图像范围通常是[-1, 1]
    return image

def convert_tensor_to_cv2_images(pixel_images):
    """转换张量图像为OpenCV格式，增加对比度和锐化"""
    images = []
    for img in pixel_images:
        # 转换到numpy并调整为0-255范围
        np_img = img.permute(1, 2, 0).cpu().numpy()
        np_img = np.clip(np_img, 0, 1) * 255
        np_img = np_img.astype(np.uint8)
        
        # 增强处理以提高人脸可见性
        # 1. 增加对比度
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 2. 适度锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_img, -1, kernel)
        
        images.append(sharpened)
    
    return images

def extract_gid_from_latent(vae, latents, vae_scale_factor, face_recognizer):
    """从VAE潜在表示中提取全局身份特征"""
    # 解码为像素空间
    pixel_images = decode_with_vae(vae, latents, vae_scale_factor)
    # 转换为OpenCV格式
    cv2_images = convert_tensor_to_cv2_images(pixel_images)
    
    # 提取每个图像的特征
    features = []
    for img in cv2_images:
        # 直接使用extract_identity而不是自己进行对齐和特征提取
        feature = face_recognizer.extract_identity(img)
        if feature is not None:
            features.append(feature)
        else:
            # 如果提取失败，使用零向量（或考虑其他回退策略）
            features.append(np.zeros(face_recognizer.embedding_size, dtype=np.float32))
    
    return torch.from_numpy(np.stack(features)).float()

class FridProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=1280, seq_len_kv=1): # seq_len_kv=1 是最简单的
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.seq_len_kv = seq_len_kv

    def forward(self, frid_merged):
        # frid_merged shape: (BatchSize, input_dim) = (B*N, 512)
        projected_frid = self.proj(frid_merged) # (B*N, 1280)
        # Reshape and repeat to match CrossAttention context format
        # Option A: seq_len_kv = 1
        frid_context = projected_frid.unsqueeze(1) # (B*N, 1, 1280)
        # Option B: seq_len_kv = H_latent * W_latent (如果希望每个像素都看到独立的投影) - 更复杂
        # Option C: seq_len_kv = 某个固定值 (e.g., 49 like CLIP) - 需要reshape?

        # 使用 Option A (seq_len_kv=1) 作为起点
        return frid_context # Shape (B*N, 1, 1280)