import torch
import numpy as np
import logging
from svd.sgm.models.diffusion import DiffusionEngine
from fal_dil.utils.face_parsing import FaceParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_mask")

def test_vae_encode_behavior(model):
    """测试VAE编码器的返回行为"""
    # 创建测试输入
    test_tensor = torch.randn(1, 3, 64, 64).to(model.device)
    
    # 测试VAE编码器行为
    with torch.no_grad():
        encoder_output = model.first_stage_model.encode(test_tensor)
    
    logger.info(f"VAE encode返回类型: {type(encoder_output)}")
    logger.info(f"是否有sample方法: {hasattr(encoder_output, 'sample')}")
    
    return encoder_output

def test_face_mask_generation(model, x):
    """逐步测试面部遮罩生成过程"""
    # 1. 确保有FaceParser
    if not hasattr(model, 'face_parser'):
        model.face_parser = FaceParser(device=model.device)
    
    logger.info(f"输入形状: {x.shape}")
    
    # 2. 解码到像素空间
    with torch.no_grad():
        try:
            pixel_x = model.decode_first_stage(x)
            logger.info(f"解码后形状: {pixel_x.shape}")
        except Exception as e:
            logger.error(f"解码失败: {e}")
            return None
    
    # 3. 处理一个样本
    try:
        frame = pixel_x[0] if len(pixel_x.shape) == 4 else pixel_x[0, 0]
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        frame_np = ((frame_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
        logger.info(f"帧形状: {frame_np.shape}, 类型: {frame_np.dtype}")
    except Exception as e:
        logger.error(f"帧处理失败: {e}")
        return None
    
    # 4. 执行面部解析
    try:
        mask = model.face_parser.parse(frame_np)
        logger.info(f"遮罩形状: {mask.shape}, 值范围: [{mask.min()}, {mask.max()}]")
    except Exception as e:
        logger.error(f"面部解析失败: {e}")
        return None
    
    # 5. 准备编码
    try:
        mask_tensor = torch.from_numpy(mask).float().permute(2, 0, 1).to(model.device)
        mask_tensor = mask_tensor.repeat(3, 1, 1)  # 3通道化
        mask_normalized = mask_tensor * 2.0 - 1.0
        logger.info(f"归一化后遮罩形状: {mask_normalized.shape}")
    except Exception as e:
        logger.error(f"遮罩处理失败: {e}")
        return None
    
    # 6. 尝试编码
    try:
        # 直接使用first_stage_model.encode
        encoder_output = model.first_stage_model.encode(mask_normalized)
        logger.info(f"编码器输出类型: {type(encoder_output)}, 是否有sample方法: {hasattr(encoder_output, 'sample')}")
        
        if hasattr(encoder_output, 'sample'):
            mask_latent = encoder_output.sample() * model.scale_factor
        else:
            mask_latent = encoder_output * model.scale_factor
        logger.info(f"最终潜变量形状: {mask_latent.shape}")
    except Exception as e:
        logger.error(f"编码失败: {e}")
        return None
        
    return mask_latent

# 使用方法(示例):
# model = ...  # 获取已初始化的DiffusionEngine实例
# x = ...      # 获取输入张量
# test_vae_encode_behavior(model)
# result = test_face_mask_generation(model, x)