import torch
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_mini_training")

project_root = Path("/root/HiFiVFS")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from svd.sgm.models.diffusion import DiffusionEngine
from svd.sgm.modules.diffusionmodules.wrappers import OpenAIWrapper
from svd.sgm.modules.diffusionmodules.openaimodel import UNetModel
from svd.sgm.modules.diffusionmodules.denoiser import DiscreteDenoiser

def create_mini_engine():
    """创建一个小型DiffusionEngine用于测试"""
    # 简化配置
    unet = UNetModel(
        in_channels=4, model_channels=32, out_channels=4,
        num_res_blocks=1, attention_resolutions=[1],
        channel_mult=[1, 2], dims=2, num_heads=1,
        context_dim=64, transformer_depth=1,
        spatial_transformer_attn_type="softmax"
    )
    
    wrapped_unet = OpenAIWrapper(unet)
    
    # 模拟一个mini版本的shared_step
    batch = {
        "image": torch.randn(2, 3, 64, 64),  # 原始图像
    }
    
    # 提取f_attr_low (模拟)
    f_attr_low = torch.randn(2, 32, 8, 8)  # 假设的特征
    
    # 模拟VAE编码
    x = torch.randn(2, 4, 8, 8)  # 假设的VAE潜变量
    
    # 添加噪声
    noise = torch.randn_like(x)
    t = torch.ones(2, dtype=torch.long) * 500  # 中间时间步
    sigmas = torch.ones(2) * 0.5
    noised_input = x + noise * sigmas.view(-1, 1, 1, 1)
    
    # 条件
    c = {"crossattn": torch.randn(2, 5, 64)}
    
    logger.info("测试UNet前向传播...")
    try:
        # 调用wrapped_unet，传递f_attr_low
        with torch.no_grad():
            output = wrapped_unet(
                noised_input, t, c,
                f_attr_low=f_attr_low
            )
        logger.info(f"✅ UNet前向传播成功! 输出形状: {output.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ UNet前向传播失败: {e}")
        return False

if __name__ == "__main__":
    create_mini_engine()