import torch
import numpy as np
import cv2
from diffusers import AutoencoderKL
from torch import nn
from .face_recognition import FaceRecognizer
from .detect_align_face import detect_align_face

@torch.no_grad()
def encode_with_vae(vae: AutoencoderKL, pixel_values: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Encodes pixel values to VAE latent space.
    
    Args:
        vae (AutoencoderKL): VAE模型实例
        pixel_values (torch.Tensor): 像素值，形状为(B, 3, H_vae, W_vae)，范围[-1, 1]
        scale_factor (float): VAE缩放因子
    
    Returns:
        torch.Tensor: 潜在空间表示
    """
    latent_dist = vae.encode(pixel_values).latent_dist
    latents = latent_dist.sample()  # 或使用 .mode()
    latents = latents * scale_factor
    return latents

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
    latents_unscaled = latents / scale_factor
    image = vae.decode(latents_unscaled).sample
    # 图像范围通常是[-1, 1]
    return image

def convert_tensor_to_cv2_images(tensor: torch.Tensor) -> list:
    """Converts a batch of tensors in [-1, 1] range to list of BGR uint8 numpy images.
    
    Args:
        tensor (torch.Tensor): 形状为(B, 3, H, W)，范围[-1, 1]的张量
    
    Returns:
        list: OpenCV BGR格式的图像列表
    """
    images = []
    tensor = tensor.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()  # B, H, W, 3
    tensor = ((tensor + 1.0) / 2.0 * 255.0).astype(np.uint8)
    for img_rgb in tensor:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        images.append(img_bgr)
    return images

def extract_gid_from_latent(vae: AutoencoderKL,
                            latents: torch.Tensor,
                            scale_factor: float,
                            face_recognizer: FaceRecognizer) -> torch.Tensor:
    """Decodes latents, detects/aligns faces, and extracts ID embeddings.
    
    Args:
        vae (AutoencoderKL): VAE模型实例
        latents (torch.Tensor): 潜在空间表示
        scale_factor (float): VAE缩放因子
        face_recognizer (FaceRecognizer): 人脸识别器实例
    
    Returns:
        torch.Tensor: 提取的身份嵌入
    """
    # 解码到像素空间
    pixel_images_tensor = decode_with_vae(vae, latents, scale_factor)
    # 转换为OpenCV图像列表
    cv2_images = convert_tensor_to_cv2_images(pixel_images_tensor)
    
    embeddings = []
    device = latents.device
    
    for img_bgr in cv2_images:
        # 使用实际的人脸检测和对齐
        aligned_face = detect_align_face(img_bgr, target_size=(112, 112), mode='arcface')
        
        if aligned_face is not None:
            # 获取人脸嵌入
            embedding_np = face_recognizer.get_embedding(aligned_face)
            if embedding_np is not None:
                embeddings.append(embedding_np)
            else:
                embeddings.append(np.zeros(512, dtype=np.float32))
        else:
            embeddings.append(np.zeros(512, dtype=np.float32))
    
    # 堆叠结果并移动到正确的设备
    f_gid_prime_batch = torch.from_numpy(np.stack(embeddings)).float().to(device)
    return f_gid_prime_batch

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