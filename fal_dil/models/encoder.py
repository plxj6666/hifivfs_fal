# fal_dil/models/encoder.py

import torch
import torch.nn as nn
try:
    from .blocks import ResBlock, SelfAttentionBlock
except ImportError:
    from blocks import ResBlock, SelfAttentionBlock
import logging
logger = logging.getLogger(__name__)
import torch.nn.functional as F

class AttributeEncoder(nn.Module):
    # --- 修改 __init__ 添加 use_checkpoint ---
    def __init__(self, use_checkpoint: bool = False): # <<<--- 添加参数
        super().__init__()
        self.use_checkpoint = use_checkpoint # <<<--- 保存参数
        dtype = torch.float32
        self.initial_conv = nn.Conv2d(4, 320, kernel_size=1, stride=1, padding=0).to(dtype)

        # --- Stage 1 ---
        # --- 将 use_checkpoint 传递给子模块 ---
        self.stage1_res1 = ResBlock(320, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage1_sa1 = SelfAttentionBlock(320, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage1_res2 = ResBlock(320, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage1_sa2 = SelfAttentionBlock(320, use_checkpoint=self.use_checkpoint).to(dtype)
        # ---

        # --- Downsample 1 ---
        self.downsample1 = nn.Conv2d(320, 640, kernel_size=3, stride=2, padding=1).to(dtype)

        # --- Stage 2 ---
        # --- 将 use_checkpoint 传递给子模块 ---
        self.stage2_res1 = ResBlock(640, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage2_sa1 = SelfAttentionBlock(640, num_heads=8, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage2_res2 = ResBlock(640, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage2_sa2 = SelfAttentionBlock(640, num_heads=8, use_checkpoint=self.use_checkpoint).to(dtype)
        # ---

        # --- Downsample 2 ---
        self.downsample2 = nn.Conv2d(640, 1280, kernel_size=3, stride=2, padding=1).to(dtype)

        # --- Stage 3 ---
        # --- 将 use_checkpoint 传递给子模块 ---
        self.stage3_res1 = ResBlock(1280, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage3_sa1 = SelfAttentionBlock(1280, num_heads=16, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage3_res2 = ResBlock(1280, use_checkpoint=self.use_checkpoint).to(dtype)
        self.stage3_sa2 = SelfAttentionBlock(1280, num_heads=16, use_checkpoint=self.use_checkpoint).to(dtype)
        # ---

    def forward(self, vt):
        # forward 方法本身不需要修改，checkpoint 在子模块内部处理
        x = self.initial_conv(vt)
        x = self.stage1_res1(x)
        x = self.stage1_sa1(x)
        x = self.stage1_res2(x)
        f_low = self.stage1_sa2(x)
        x = self.downsample1(f_low)
        x = self.stage2_res1(x)
        x = self.stage2_sa1(x)
        x = self.stage2_res2(x)
        x = self.stage2_sa2(x)
        x = self.downsample2(x)
        x = self.stage3_res1(x)
        x = self.stage3_sa1(x)
        x = self.stage3_res2(x)
        f_attr = self.stage3_sa2(x)
        return f_attr, f_low
    
    def extract_low_level_features(self, x):
        """提取低层次特征 (f_low)，优化维度处理"""
        # 保存输入维度信息
        orig_shape = x.shape
        input_dtype = x.dtype
        
        # 确保是浮点型
        if input_dtype != torch.float32:
            x = x.float()
        
        # 检查维度
        if len(orig_shape) != 4:  # 不是标准的[B,C,H,W]
            logger.warning(f"输入维度不是标准的4D: {orig_shape}，尝试调整")
            if len(orig_shape) == 5:  # [B,T,C,H,W]
                B, T, C, H, W = orig_shape
                x = x.reshape(B*T, C, H, W)
            else:
                raise ValueError(f"无法处理的输入维度: {orig_shape}")
        
        # 空间尺寸检查和调整
        h, w = x.shape[2], x.shape[3]
        expected_latent_size = 40
        if h != expected_latent_size or w != expected_latent_size:
            logger.warning(f"输入特征尺寸 [{h}x{w}] 不匹配，调整到 [{expected_latent_size}x{expected_latent_size}]")
            x = F.interpolate(x, size=(expected_latent_size, expected_latent_size), mode='bilinear', align_corners=False)
        
        # 通道数检查和调整
        if x.shape[1] == 3:  # RGB -> RGBA
            alpha_channel = torch.ones_like(x[:, :1, :, :])
            x = torch.cat([x, alpha_channel], dim=1)
        
        # 提取特征
        _, f_low = self.forward(x)
        
        # 恢复原始精度
        if input_dtype != torch.float32:
            f_low = f_low.to(input_dtype)
        
        return f_low

# --- 测试代码 ---
if __name__ == "__main__":
    print("--- Testing AttributeEncoder Shape (640x640 VAE -> 80x80 Latent) ---")
    # 使用80x80输入测试
    dummy_vt = torch.randn(2, 4, 80, 80)
    encoder = AttributeEncoder()
    f_attr, f_low = encoder(dummy_vt)
    print(f"Input Vt shape: {dummy_vt.shape}")
    print(f"Output f_attr shape: {f_attr.shape}") # 应该为 (B, 1280, 20, 20)
    print(f"Output f_low shape: {f_low.shape}")   # 应该为 (B, 320, 80, 80)
    assert f_attr.shape == (2, 1280, 20, 20), "f_attr shape mismatch!"
    assert f_low.shape == (2, 320, 80, 80), "f_low shape mismatch!"