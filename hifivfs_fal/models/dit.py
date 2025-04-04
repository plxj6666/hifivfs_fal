# hifivfs_fal/models/dit.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class DetailedIdentityTokenizer(nn.Module):
    """
    详细身份标记器 (DIT)
    将从人脸识别模型中间层提取的 fdid 特征图转换为 UNet Cross-Attention 可用的 token 序列。
    """
    def __init__(self, 
                 input_channels: int,      # fdid 特征图的通道数 (例如 1792)
                 output_embedding_dim: int, # UNet Cross-Attention 期望的嵌入维度 (例如 768)
                 feature_map_size: tuple = (3, 3)): # fdid 特征图的空间尺寸 (H, W)
        """
        初始化 DIT。

        Args:
            input_channels (int): fdid 特征图的输入通道数。
            output_embedding_dim (int): 输出 token 的嵌入维度，应与 UNet 的 cross_attention_dim 匹配。
            feature_map_size (tuple): fdid 特征图的高度和宽度 (H, W)。用于计算和验证 token 数量。
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_embedding_dim
        self.feature_h, self.feature_w = feature_map_size
        self.num_tokens = self.feature_h * self.feature_w # 计算预期的 token 数量

        # 使用 1x1 卷积来调整通道数，同时保持空间维度
        # 输入: (B, input_channels, H, W)
        # 输出: (B, output_embedding_dim, H, W)
        self.channel_align_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_embedding_dim, 
            kernel_size=1
        )
        
        logger.info(f"DIT 初始化: 输入通道={input_channels}, 输出维度={output_embedding_dim}, 特征图尺寸={feature_map_size}, Token数量={self.num_tokens}")

    def forward(self, fdid_map: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            fdid_map (torch.Tensor): 输入的详细特征图，形状应为 (B, C_in, H, W)。
                                     例如 (BatchSize, 1792, 3, 3)。

        Returns:
            torch.Tensor: 输出的 token 序列，形状 (B, N_tokens, C_out)。
                          例如 (BatchSize, 9, 768)。
        """
        # 检查输入形状
        if fdid_map.ndim != 4 or fdid_map.shape[1] != self.input_channels \
           or fdid_map.shape[2] != self.feature_h or fdid_map.shape[3] != self.feature_w:
            logger.error(f"DIT 输入形状错误! 期望: (B, {self.input_channels}, {self.feature_h}, {self.feature_w}), 收到: {fdid_map.shape}")
            # 可以选择抛出错误或返回 None/空 Tensor
            raise ValueError("DIT 输入形状错误")
            # return torch.empty(fdid_map.shape[0], self.num_tokens, self.output_dim, device=fdid_map.device) # 返回空 Tensor

        # 1. 调整通道数
        # (B, C_in, H, W) -> (B, C_out, H, W)
        aligned_features = self.channel_align_conv(fdid_map)
        
        # 2. 展平空间维度 H 和 W
        # (B, C_out, H, W) -> (B, C_out, H*W)
        batch_size, channels, height, width = aligned_features.shape
        flattened_features = aligned_features.view(batch_size, channels, height * width) 
        
        # 3. 调整维度顺序以匹配 (Batch, SequenceLength, EmbeddingDim)
        # (B, C_out, H*W) -> (B, H*W, C_out)
        tdid_tokens = flattened_features.permute(0, 2, 1).contiguous() # contiguous() 确保内存连续

        # 验证 token 数量
        if tdid_tokens.shape[1] != self.num_tokens:
             # 这个情况理论上不应该发生，除非 view/permute 出错
             logger.warning(f"DIT 生成的 token 数量 {tdid_tokens.shape[1]} 与预期的 {self.num_tokens} 不符！")

        logger.debug(f"DIT 输出形状: {tdid_tokens.shape}") # 预期 (B, 9, output_dim)
        return tdid_tokens

# --- 可以在文件末尾添加一个简单的测试 ---
if __name__ == '__main__':
     logging.basicConfig(level=logging.DEBUG)
     
     # 假设参数
     batch_size = 4
     in_channels = 1792 # fdid 通道数
     out_dim = 768      # UNet cross attention 维度
     h, w = 3, 3        # fdid 特征图尺寸
     num_tok = h * w

     # 创建 DIT 实例
     dit = DetailedIdentityTokenizer(
         input_channels=in_channels,
         output_embedding_dim=out_dim,
         feature_map_size=(h, w)
     )
     print(f"DIT 模型:\n{dit}")

     # 创建假的 fdid 输入
     dummy_fdid = torch.randn(batch_size, in_channels, h, w)
     print(f"\n输入 fdid 形状: {dummy_fdid.shape}")

     # 通过 DIT
     try:
          output_tokens = dit(dummy_fdid)
          print(f"输出 tdid 形状: {output_tokens.shape}")
          # 验证形状
          assert output_tokens.shape == (batch_size, num_tok, out_dim)
          print("形状验证通过!")
     except Exception as e:
          print(f"测试失败: {e}")