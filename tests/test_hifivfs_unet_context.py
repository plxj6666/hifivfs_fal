# tests/test_hifivfs_unet_context.py

import torch
import unittest
import sys
from pathlib import Path
import logging

# --- 临时添加项目根目录到 sys.path ---
# 这使得我们可以导入 sgm 和 hifivfs_fal
# 注意：在实际测试框架中可能有更好的方式处理路径
# current_dir = Path(__file__).resolve().parent.parent
# sys.path.append(str(current_dir))
# print(f"临时添加到 sys.path: {current_dir}")
# --- 路径添加结束 ---

# --- 配置日志，方便查看调试信息 ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 导入被测试的模块 ---
try:
    # 假设修改后的类在 video_attention.py 中
    from svd.sgm.modules.video_attention import SpatialVideoTransformer
    # 可能还需要导入我们创建的 BasicTransformerBlockWithHifiVFSContext
    # (如果它在不同的文件里)
    # from svd.sgm.modules.attention import BasicTransformerBlockWithHifiVFSContext
except ImportError as e:
    logger.error(f"无法导入测试所需的模块: {e}", exc_info=True)
    # 如果导入失败，测试将无法运行
    raise

class TestHiFiVFSUNetContext(unittest.TestCase):

    def test_spatial_video_transformer_with_dict_context(self):
        """测试 SpatialVideoTransformer 是否能处理字典 context"""

        # --- 定义测试参数 ---
        batch_size = 1 # 使用较小的 batch size 进行测试
        num_frames = 2 # 视频帧数
        in_channels = 320 # U-Net 中间层的通道数 (示例)
        height, width = 32, 32 # 特征图尺寸 (示例)
        n_heads = 8
        d_head = 64 # inner_dim = n_heads * d_head = 512
        depth = 1 # Transformer 块深度
        context_dim_dil = 768 # DIL context 维度
        context_dim_fal = 768 # FAL context 维度 (假设已投影)
        seq_len_dil = 9       # DIL token 数量
        seq_len_fal = 100     # FAL token 数量 (示例，例如 10x10 feature map)

        # --- 实例化修改后的 SpatialVideoTransformer ---
        try:
            transformer = SpatialVideoTransformer(
                in_channels=in_channels,
                n_heads=n_heads,
                d_head=d_head,
                depth=depth,
                context_dim=context_dim_dil, # DIL context dim
                f_attr_context_dim=context_dim_fal, # <<<--- 传入 FAL context dim
                attn_mode="softmax", # 使用内置 SDP backend
                use_checkpoint=False, # 测试时不使用 checkpoint 简化调试
                timesteps=num_frames # 传入帧数
            )
            transformer.eval() # 设置为评估模式
            logger.info("SpatialVideoTransformer 实例化成功。")
        except Exception as e:
            logger.error(f"实例化 SpatialVideoTransformer 失败: {e}", exc_info=True)
            self.fail("SpatialVideoTransformer 实例化失败")
            return

        # --- 创建假的输入数据 ---
        # x shape: (B*T, C, H, W)
        x = torch.randn(batch_size * num_frames, in_channels, height, width)
        logger.info(f"输入 x shape: {x.shape}")

        # --- 创建假的 context 字典 ---
        # context['crossattn'] shape: (B*T, N_dil, C_dil)
        # 注意：通常 context 的 Batch 维应该匹配 x，也是 B*T
        dil_tokens = torch.randn(batch_size * num_frames, seq_len_dil, context_dim_dil)
        # context['f_attr_tokens'] shape: (B*T, N_fal, C_fal)
        fal_tokens = torch.randn(batch_size * num_frames, seq_len_fal, context_dim_fal)

        context_dict = {
            "crossattn": dil_tokens,
            "f_attr_tokens": fal_tokens
        }
        logger.info(f"Context dict keys: {list(context_dict.keys())}")
        logger.info(f"Context['crossattn'] shape: {context_dict['crossattn'].shape}")
        logger.info(f"Context['f_attr_tokens'] shape: {context_dict['f_attr_tokens'].shape}")

        # --- 调用 forward 方法 ---
        output = None
        try:
            # 传递 context 字典和 timesteps
            output = transformer.forward(x, context=context_dict, timesteps=num_frames)
            logger.info("transformer.forward 调用成功。")
            logger.info(f"输出 output shape: {output.shape}")
        except Exception as e:
            logger.error(f"调用 transformer.forward 时出错: {e}", exc_info=True)
            self.fail("transformer.forward 调用失败")

        # --- 检查输出形状 ---
        self.assertIsNotNone(output, "输出不应为 None")
        self.assertEqual(output.shape, x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配")

        logger.info("测试通过！")

# --- 运行测试 ---
if __name__ == '__main__':
    unittest.main()