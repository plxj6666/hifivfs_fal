import torch
import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_f_attr_injection")

# 添加项目根目录到路径
project_root = Path("/root/HiFiVFS")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
logger.info(f"Project root: {project_root}")

# 导入UNetModel
from svd.sgm.modules.diffusionmodules.openaimodel import UNetModel

def test_f_attr_injection():
    """测试f_attr_low特征在UNet中的注入"""
    
    # 创建UNetModel实例 - 使用简化参数
    model = UNetModel(
        in_channels=4,           # 只有VAE潜变量 (4通道)
        model_channels=320,      # 基础通道数
        out_channels=4,          # 输出通道数
        num_res_blocks=2,        # 每个分辨率的残差块数
        attention_resolutions=[4, 2, 1],
        dropout=0.0,
        channel_mult=[1, 2, 4, 4],
        dims=2,
        use_checkpoint=False,
        num_heads=8,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        transformer_depth=1,
        context_dim=768,
        spatial_transformer_attn_type="softmax"
    )
    
    # 创建测试输入
    batch_size = 2
    height = 16
    width = 16
    
    # VAE潜变量 (4通道)
    x = torch.randn(batch_size, 4, height, width)
    
    # 时间步
    timesteps = torch.ones(batch_size) * 500
    
    # 上下文向量 (使用张量而非字典，测试兼容性)
    context = torch.randn(batch_size, 77, 768)
    
    # f_attr_low (320通道，与input_block_0输出匹配)
    f_attr_low = torch.randn(batch_size, 320, height, width)
    
    # 监控钩子
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    # 为input_block_0注册钩子
    input_block0_hook = model.input_blocks[0].register_forward_hook(hook_fn("input_block_0"))
    
    # 前向传播测试
    with torch.no_grad():
        logger.info("执行前向传播...")
        output = model(x, timesteps, context, f_attr_low=f_attr_low)
        logger.info(f"前向传播成功! 输出形状: {output.shape}")
    
    # 检查input_block_0输出形状
    if "input_block_0" in activations:
        ib0_shape = activations["input_block_0"].shape
        logger.info(f"input_block_0输出形状: {ib0_shape}")
        logger.info(f"f_attr_low形状: {f_attr_low.shape}")
        
        # 确认形状匹配
        if ib0_shape[1:] == f_attr_low.shape[1:]:
            logger.info("✅ 形状匹配，f_attr_low可以成功注入")
        else:
            logger.error(f"❌ 形状不匹配，注入可能失败")
    
    # 移除钩子
    input_block0_hook.remove()
    
    return "测试完成"

if __name__ == "__main__":
    test_f_attr_injection()