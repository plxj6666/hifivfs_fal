import torch
import torch.nn as nn
try:
    from .blocks import ResBlock, SelfAttentionBlock
except ImportError:
    from blocks import ResBlock, SelfAttentionBlock


class AttributeEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 初始卷积层：4 -> 320 通道
        self.initial_conv = nn.Conv2d(4, 320, kernel_size=1, stride=1, padding=0)

        # --- Stage 1 (Input: 4x80x80, Output: 320x80x80) ---
        self.stage1_res1 = ResBlock(320)
        self.stage1_sa1 = SelfAttentionBlock(320)
        self.stage1_res2 = ResBlock(320)
        self.stage1_sa2 = SelfAttentionBlock(320)
        # f_low 从这里输出 (80x80)

        # --- Downsample 1 (80x80 -> 40x40) ---
        self.downsample1 = nn.Conv2d(320, 640, kernel_size=3, stride=2, padding=1)

        # --- Stage 2 (Output: 640x40x40) ---
        self.stage2_res1 = ResBlock(640)
        self.stage2_sa1 = SelfAttentionBlock(640, num_heads=8)
        self.stage2_res2 = ResBlock(640)
        self.stage2_sa2 = SelfAttentionBlock(640, num_heads=8)

        # --- Downsample 2 (40x40 -> 20x20) ---
        self.downsample2 = nn.Conv2d(640, 1280, kernel_size=3, stride=2, padding=1)

        # --- Stage 3 (Output: 1280x20x20) ---
        self.stage3_res1 = ResBlock(1280)
        self.stage3_sa1 = SelfAttentionBlock(1280, num_heads=16)
        self.stage3_res2 = ResBlock(1280)
        self.stage3_sa2 = SelfAttentionBlock(1280, num_heads=16)
        # f_attr 从这里输出 (20x20)

    def forward(self, vt):
        # vt shape: (B, 4, 80, 80) - 适应640x640输入的VAE潜在表示
        x = self.initial_conv(vt) # (B, 320, 80, 80)

        # Stage 1
        x = self.stage1_res1(x)
        x = self.stage1_sa1(x)
        x = self.stage1_res2(x)
        f_low = self.stage1_sa2(x) # (B, 320, 80, 80) - f_low output

        # Downsample 1
        x = self.downsample1(f_low) # (B, 640, 40, 40)

        # Stage 2
        x = self.stage2_res1(x)
        x = self.stage2_sa1(x)
        x = self.stage2_res2(x)
        x = self.stage2_sa2(x) # (B, 640, 40, 40)

        # Downsample 2
        x = self.downsample2(x) # (B, 1280, 20, 20)

        # Stage 3
        x = self.stage3_res1(x)
        x = self.stage3_sa1(x)
        x = self.stage3_res2(x)
        f_attr = self.stage3_sa2(x) # (B, 1280, 20, 20) - f_attr output

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