import torch
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_param_chain")

project_root = Path("/root/HiFiVFS")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from svd.sgm.modules.diffusionmodules.wrappers import OpenAIWrapper
from svd.sgm.modules.diffusionmodules.openaimodel import UNetModel
from svd.sgm.modules.diffusionmodules.denoiser import Denoiser, DiscreteDenoiser

def test_parameter_chain():
    """测试参数从Denoiser到UNet的传递链"""
    
    # 创建UNetModel
    unet = UNetModel(
        in_channels=4, model_channels=32, out_channels=4,
        num_res_blocks=1, attention_resolutions=[1],
        channel_mult=[1, 2], dims=2, num_heads=1,
        context_dim=64, transformer_depth=1,
        spatial_transformer_attn_type="softmax"
    )
    
    # 创建OpenAIWrapper
    wrapped_unet = OpenAIWrapper(unet)
    
    # 创建Denoiser (简化配置)
    denoiser = Denoiser({"target": "svd.sgm.modules.diffusionmodules.denoiser_scaling.VScalingWithEDMcNoise"})
    
    # 测试数据
    x = torch.randn(2, 4, 8, 8)  # 输入
    sigma = torch.ones(2)        # 噪声水平
    c = {"crossattn": torch.randn(2, 5, 64)}  # 条件
    f_attr_low = torch.randn(2, 32, 8, 8)  # 低层次属性特征
    
    # 测试传参
    logger.info("测试参数传递...")
    try:
        with torch.no_grad():
            # 调用denoiser，传递f_attr_low
            output = denoiser(
                wrapped_unet, x, sigma, c, 
                f_attr_low=f_attr_low
            )
        logger.info("✅ 参数传递测试成功!")
        return True
    except Exception as e:
        logger.error(f"❌ 参数传递测试失败: {e}")
        return False

if __name__ == "__main__":
    test_parameter_chain()