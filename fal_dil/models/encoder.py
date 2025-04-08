# fal_dil/models/encoder.py

import torch
import torch.nn as nn
try:
    from .blocks import ResBlock, SelfAttentionBlock
except ImportError:
    from blocks import ResBlock, SelfAttentionBlock


class AttributeEncoder(nn.Module):
    # --- 修改 __init__ 添加 use_checkpoint ---
    def __init__(self, use_checkpoint: bool = False): # <<<--- 添加参数
        super().__init__()
        self.use_checkpoint = use_checkpoint # <<<--- 保存参数

        self.initial_conv = nn.Conv2d(4, 320, kernel_size=1, stride=1, padding=0)

        # --- Stage 1 ---
        # --- 将 use_checkpoint 传递给子模块 ---
        self.stage1_res1 = ResBlock(320, use_checkpoint=self.use_checkpoint)
        self.stage1_sa1 = SelfAttentionBlock(320, use_checkpoint=self.use_checkpoint)
        self.stage1_res2 = ResBlock(320, use_checkpoint=self.use_checkpoint)
        self.stage1_sa2 = SelfAttentionBlock(320, use_checkpoint=self.use_checkpoint)
        # ---

        # --- Downsample 1 ---
        self.downsample1 = nn.Conv2d(320, 640, kernel_size=3, stride=2, padding=1)

        # --- Stage 2 ---
        # --- 将 use_checkpoint 传递给子模块 ---
        self.stage2_res1 = ResBlock(640, use_checkpoint=self.use_checkpoint)
        self.stage2_sa1 = SelfAttentionBlock(640, num_heads=8, use_checkpoint=self.use_checkpoint)
        self.stage2_res2 = ResBlock(640, use_checkpoint=self.use_checkpoint)
        self.stage2_sa2 = SelfAttentionBlock(640, num_heads=8, use_checkpoint=self.use_checkpoint)
        # ---

        # --- Downsample 2 ---
        self.downsample2 = nn.Conv2d(640, 1280, kernel_size=3, stride=2, padding=1)

        # --- Stage 3 ---
        # --- 将 use_checkpoint 传递给子模块 ---
        self.stage3_res1 = ResBlock(1280, use_checkpoint=self.use_checkpoint)
        self.stage3_sa1 = SelfAttentionBlock(1280, num_heads=16, use_checkpoint=self.use_checkpoint)
        self.stage3_res2 = ResBlock(1280, use_checkpoint=self.use_checkpoint)
        self.stage3_sa2 = SelfAttentionBlock(1280, num_heads=16, use_checkpoint=self.use_checkpoint)
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