import torch
import torch.nn as nn
try:
    from .blocks import ResBlock, CrossAttentionBlock
except ImportError:
    from blocks import ResBlock, CrossAttentionBlock


class Decoder(nn.Module):
    def __init__(self, frid_channels=1280, allow_frid_projection=True):
        super().__init__()
        self.frid_channels = frid_channels
        self.allow_frid_projection = allow_frid_projection

        # 身份特征投影层
        self.frid_projection = nn.Sequential(
            nn.Linear(512, frid_channels),
            nn.LayerNorm(frid_channels),
            nn.SiLU()
        )

        # --- Stage 1 (Input: f_attr 1280x20x20, f_rid) ---
        ca1_heads = 16
        self.stage1_res1 = ResBlock(1280)
        self.stage1_ca1 = CrossAttentionBlock(query_channels=1280, kv_channels=self.frid_channels, num_heads=ca1_heads)
        self.stage1_res2 = ResBlock(1280)
        self.stage1_ca2 = CrossAttentionBlock(query_channels=1280, kv_channels=self.frid_channels, num_heads=ca1_heads)
        # Output: 1280x20x20

        # --- Upsample 1 (20x20 -> 40x40) ---
        self.upsample1 = nn.ConvTranspose2d(1280, 640, kernel_size=4, stride=2, padding=1)

        # --- Stage 2 (Input: 640x40x40, f_rid) ---
        ca2_heads = 8
        self.stage2_res1 = ResBlock(640)
        self.stage2_ca1 = CrossAttentionBlock(query_channels=640, kv_channels=self.frid_channels, num_heads=ca2_heads)
        self.stage2_res2 = ResBlock(640)
        self.stage2_ca2 = CrossAttentionBlock(query_channels=640, kv_channels=self.frid_channels, num_heads=ca2_heads)
        # Output: 640x40x40

        # --- Upsample 2 (40x40 -> 80x80) ---
        self.upsample2 = nn.ConvTranspose2d(640, 320, kernel_size=4, stride=2, padding=1)

        # --- Stage 3 (Input: 320x80x80) ---
        self.stage3_res1 = ResBlock(320)
        self.stage3_res2 = ResBlock(320)
        # Output: 320x80x80

        # --- Final Convolution ---
        self.final_conv = nn.Conv2d(320, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, f_attr, f_rid):
        # f_attr shape: (B, 1280, 20, 20)
        # f_rid shape: (B, C) 或 (B, SeqLen_kv, C)

        # 检查并投影身份特征
        if self.allow_frid_projection and hasattr(self, 'frid_projection'):
            if len(f_rid.shape) == 2:
                if f_rid.shape[1] != self.frid_channels:
                    f_rid = self.frid_projection(f_rid)
                f_rid = f_rid.unsqueeze(1) # (B, 1, C) for CrossAttn
            elif len(f_rid.shape) == 3:
                if f_rid.shape[2] != self.frid_channels:
                    batch_size, seq_len, channels = f_rid.shape
                    f_rid = f_rid.reshape(-1, channels)
                    f_rid = self.frid_projection(f_rid)
                    f_rid = f_rid.reshape(batch_size, seq_len, self.frid_channels)

        x = f_attr # (B, 1280, 20, 20)

        # --- Stage 1 ---
        x = self.stage1_res1(x)
        x = self.stage1_ca1(x, f_rid)
        x = self.stage1_res2(x)
        x = self.stage1_ca2(x, f_rid) # (B, 1280, 20, 20)

        # Upsample 1
        x = self.upsample1(x) # (B, 640, 40, 40)

        # --- Stage 2 ---
        x = self.stage2_res1(x)
        x = self.stage2_ca1(x, f_rid)
        x = self.stage2_res2(x)
        x = self.stage2_ca2(x, f_rid) # (B, 640, 40, 40)

        # Upsample 2
        x = self.upsample2(x) # (B, 320, 80, 80)

        # Stage 3
        x = self.stage3_res1(x)
        x = self.stage3_res2(x) # (B, 320, 80, 80)

        # Final Convolution
        vt_prime = self.final_conv(x) # (B, 4, 80, 80) - 适应640x640图像的VAE潜在表示

        return vt_prime

# --- 测试代码 ---
if __name__ == "__main__":
    print("--- Testing Decoder Shape (640x640 VAE -> 80x80 Latent) ---")
    # 输入现在是 20x20
    dummy_f_attr = torch.randn(2, 1280, 20, 20)
    
    # f_rid 形状保持不变
    batch_size = dummy_f_attr.shape[0]
    seq_len = 1
    frid_channels = 1280
    dummy_f_rid_dec = torch.randn(batch_size, seq_len, frid_channels)
    
    decoder = Decoder(frid_channels=frid_channels)
    vt_prime = decoder(dummy_f_attr, dummy_f_rid_dec)
    
    print(f"Input f_attr shape: {dummy_f_attr.shape}")
    print(f"Input f_rid shape: {dummy_f_rid_dec.shape}")
    print(f"Output V't shape: {vt_prime.shape}") # 应该为 (B, 4, 80, 80)
    assert vt_prime.shape == (batch_size, 4, 80, 80), "V't shape mismatch!"