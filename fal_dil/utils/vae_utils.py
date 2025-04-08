# vae_utils.py

import torch
import numpy as np
import cv2
from diffusers import AutoencoderKL
from torch import nn
from pathlib import Path # 导入 Path
import logging # 导入 logging
from typing import List
# 内部依赖
try:
    from .detect_align_face import detect_align_face
except ImportError:
     # 如果在 utils 目录下运行，可能需要相对导入
     try:
          from detect_align_face import detect_align_face
     except ImportError:
          raise ImportError("无法导入 detect_align_face")

logger = logging.getLogger(__name__) # 使用 logger

# **** 定义失败图像保存目录 ****
# 注意：这个路径是硬编码的，更好的做法可能是通过参数传递进来
FAILED_IMAGE_DIR = Path("/root/HiFiVFS/samples/vox2_640/failed") 
FAILED_IMAGE_DIR.mkdir(parents=True, exist_ok=True) # 确保目录存在

@torch.no_grad()
def encode_with_vae(vae, pixel_values, scale_factor):
    """使用VAE编码像素值到潜在空间"""
    pixel_values = pixel_values.to(vae.device)
    latent_dist = vae.encode(pixel_values).latent_dist
    latent = latent_dist.sample() * scale_factor
    latent = latent.cpu() # 返回 CPU Tensor
    return latent

@torch.no_grad()
def decode_with_vae(vae: AutoencoderKL, latents: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """解码VAE潜变量到像素空间"""
    # 确保 latent 在 VAE 所在的设备上
    latents = latents.to(dtype=vae.dtype, device=vae.device) 
    latents_unscaled = latents / scale_factor
    
    # 调用解码并处理不同的返回类型
    decoded = vae.decode(latents_unscaled)
    
    # 处理可能的返回类型
    if hasattr(decoded, 'sample'):
        # 如果是分布对象 (如 DiagonalGaussianDistribution)
        image = decoded.sample
    else:
        # 如果直接返回张量
        image = decoded 
    # 返回的 image 在 VAE 设备上，范围 [-1, 1]
    return image

# **** 使用简化的、不带增强的转换函数 ****
def convert_tensor_to_cv2_images(pixel_images_rgb_neg1_1: torch.Tensor) -> List[np.ndarray]:
    """将范围在[-1, 1]的RGB Tensor转换为BGR uint8 NumPy列表"""
    images = []
    # 确保输入在 CPU 上
    pixel_images_cpu = pixel_images_rgb_neg1_1.detach().float().cpu() 
    for img_tensor in pixel_images_cpu:
        # Permute CHW -> HWC
        np_img_hwc = img_tensor.permute(1, 2, 0).numpy()
        # Denormalize [-1, 1] -> [0, 255]
        np_img_0_255 = ((np_img_hwc + 1.0) / 2.0 * 255.0)
        # Clip and convert to uint8
        np_img_uint8 = np.clip(np_img_0_255, 0, 255).astype(np.uint8)
        # Convert RGB -> BGR for OpenCV functions downstream
        np_img_bgr = cv2.cvtColor(np_img_uint8, cv2.COLOR_RGB2BGR)
        images.append(np_img_bgr)
    return images

# **** 修改 extract_gid_from_latent ****
def extract_gid_from_latent(vae, latents, vae_scale_factor, face_recognizer, global_step=None): # 添加 global_step 参数
    """从VAE潜在表示中提取全局身份特征，并在失败时保存图像"""
    features = []
    all_success = True # 标记是否所有图像都提取成功

    # 1. 解码为像素空间 (在 VAE 设备上)
    try:
        # 使用与样本保存相同的逻辑，解码到 CPU
        with torch.no_grad():
             pixel_images = decode_with_vae(vae.to('cpu'), latents.cpu(), vae_scale_factor)
             vae.to(latents.device) # 移回原设备
    except Exception as e_decode:
         logger.error(f"VAE解码失败 (步骤 {global_step}): {e_decode}", exc_info=True)
         # 如果解码失败，为批次中所有图像返回 None 或零向量
         for _ in range(latents.shape[0]):
              features.append(np.zeros(face_recognizer.embedding_size, dtype=np.float32))
         return torch.from_numpy(np.stack(features)).float() # 返回 CPU Tensor

    # 2. 转换为OpenCV格式 (BGR Uint8 CPU List)
    try:
        # 使用不带增强的版本
        cv2_images = convert_tensor_to_cv2_images(pixel_images) 
    except Exception as e_convert:
        logger.error(f"Tensor 到 CV2 图像转换失败 (步骤 {global_step}): {e_convert}", exc_info=True)
        for _ in range(latents.shape[0]):
              features.append(np.zeros(face_recognizer.embedding_size, dtype=np.float32))
        return torch.from_numpy(np.stack(features)).float()

    # 3. 提取每个图像的特征
    for idx, img_bgr in enumerate(cv2_images):
        feature = None
        try:
            # 使用 face_recognizer.extract_identity 进行对齐和提取
            feature = face_recognizer.extract_identity(img_bgr) 
        except Exception as e_extract:
             logger.warning(f"图像 {idx} 的特征提取过程中发生异常 (步骤 {global_step}): {e_extract}", exc_info=True)
             feature = None # 确保 feature 为 None

        if feature is not None:
            features.append(feature)
        else:
            # **** 提取失败，保存图像 ****
            all_success = False
            logger.warning(f"图像 {idx} 的身份特征提取失败 (步骤 {global_step})。")
            try:
                # 使用 global_step 和批次内索引命名文件
                filename = f"failed_gid_extraction_step{global_step}_idx{idx}.png"
                save_path = FAILED_IMAGE_DIR / filename
                cv2.imwrite(str(save_path), img_bgr)
                logger.info(f"已将失败的图像保存到: {save_path}")
            except Exception as e_save:
                logger.error(f"保存失败图像时出错: {e_save}", exc_info=True)
                
            # 添加零向量作为占位符
            features.append(np.zeros(face_recognizer.embedding_size, dtype=np.float32))

    # 4. 转换为 Tensor (CPU Tensor)
    # 如果所有图像都提取失败，也返回一个正确的形状
    if not features: # 如果列表为空（虽然不太可能发生）
         logger.error(f"特征列表为空 (步骤 {global_step})")
         # 返回一个形状正确的零张量
         return torch.zeros((len(cv2_images), face_recognizer.embedding_size), dtype=torch.float32)
         
    try:
        stacked_features = np.stack(features)
        return torch.from_numpy(stacked_features).float()
    except Exception as e_stack:
         logger.error(f"堆叠特征时出错 (步骤 {global_step}): {e_stack}", exc_info=True)
         # 尝试返回一个形状正确的零张量
         return torch.zeros((len(cv2_images), face_recognizer.embedding_size), dtype=torch.float32)


class FridProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=1280, seq_len_kv=1): 
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.seq_len_kv = seq_len_kv
    def forward(self, frid_merged):
        projected_frid = self.proj(frid_merged) 
        frid_context = projected_frid.unsqueeze(1) 
        return frid_context 