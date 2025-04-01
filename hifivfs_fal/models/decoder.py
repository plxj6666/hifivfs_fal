import torch
import torch.nn as nn
# 假设 blocks.py 在同一目录下或已正确安装
try:
    from .blocks import ResBlock, CrossAttentionBlock
except ImportError: # 如果直接运行此文件，则尝试绝对导入
    from blocks import ResBlock, CrossAttentionBlock


class Decoder(nn.Module):
    def __init__(self, frid_channels=1280): # 假设 f_rid 的特征维度是 1280
        super().__init__()
        self.frid_channels = frid_channels

        # --- Stage 1 (Input: f_attr 1280x20x20, f_rid) ---
        # 顺序调整：ResBlock -> CrossAttn -> ResBlock -> CrossAttn
        ca1_heads = 16 # 1280 / 16 = 80
        self.stage1_res1 = ResBlock(1280)
        self.stage1_ca1 = CrossAttentionBlock(query_channels=1280, kv_channels=self.frid_channels, num_heads=ca1_heads)
        self.stage1_res2 = ResBlock(1280)
        self.stage1_ca2 = CrossAttentionBlock(query_channels=1280, kv_channels=self.frid_channels, num_heads=ca1_heads)


        # --- Upsample 1 (20x20 -> 40x40) ---
        self.upsample1 = nn.ConvTranspose2d(1280, 640, kernel_size=4, stride=2, padding=1)

        # --- Stage 2 (Input: 640x40x40, f_rid) ---
        # 顺序调整：ResBlock -> CrossAttn -> ResBlock -> CrossAttn
        ca2_heads = 8 # 640 / 8 = 80
        self.stage2_res1 = ResBlock(640)
        self.stage2_ca1 = CrossAttentionBlock(query_channels=640, kv_channels=self.frid_channels, num_heads=ca2_heads)
        self.stage2_res2 = ResBlock(640)
        self.stage2_ca2 = CrossAttentionBlock(query_channels=640, kv_channels=self.frid_channels, num_heads=ca2_heads)


        # --- Upsample 2 (40x40 -> 80x80) ---
        self.upsample2 = nn.ConvTranspose2d(640, 320, kernel_size=4, stride=2, padding=1)

        # --- Stage 3 (Input: 320x80x80) ---
        # 只有 ResBlock
        self.stage3_res1 = ResBlock(320)
        self.stage3_res2 = ResBlock(320)

        # --- Final Convolution ---
        # 使用 1x1 卷积
        self.final_conv = nn.Conv2d(320, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, f_attr, f_rid):
        # f_attr shape: (B, 1280, 20, 20)
        # f_rid shape: (B, SeqLen_kv, Ckv) - 假设 Ckv = self.frid_channels
        x = f_attr

        # --- Stage 1 (顺序修正) ---
        x = self.stage1_res1(x)      # ResBlock First
        x = self.stage1_ca1(x, f_rid) # Then CrossAttn
        x = self.stage1_res2(x)      # ResBlock
        x = self.stage1_ca2(x, f_rid) # Then CrossAttn -> 输出 (B, 1280, 20, 20)

        # Upsample 1
        x = self.upsample1(x) # (B, 640, 40, 40)

        # --- Stage 2 (顺序修正) ---
        x = self.stage2_res1(x)      # ResBlock First
        x = self.stage2_ca1(x, f_rid) # Then CrossAttn
        x = self.stage2_res2(x)      # ResBlock
        x = self.stage2_ca2(x, f_rid) # Then CrossAttn -> 输出 (B, 640, 40, 40)

        # Upsample 2
        x = self.upsample2(x) # (B, 320, 80, 80)

        # Stage 3
        x = self.stage3_res1(x)
        x = self.stage3_res2(x) # (B, 320, 80, 80)

        # Final Convolution
        vt_prime = self.final_conv(x) # (B, 4, 80, 80)

        return vt_prime

# --- 仅当直接运行此文件时执行测试 ---
# (测试代码与之前相同，可以保留)
if __name__ == "__main__":
    print("--- Testing Decoder Shape (Corrected Order) ---")
    # 假设 Encoder 存在且可以运行，或者直接创建 dummy f_attr
    # from encoder import AttributeEncoder # 假设 encoder.py 在同一目录
    # enc = AttributeEncoder()
    # dummy_vt_for_enc = torch.randn(2, 4, 80, 80)
    # dummy_f_attr_from_enc, _ = enc(dummy_vt_for_enc)

    # 或者直接创建 dummy tensor
    dummy_f_attr = torch.randn(2, 1280, 20, 20) # (B, C, H, W)

    # 假设 f_rid 是 (B, SeqLen, Ckv)，例如 (B, 49, 1280)
    batch_size = dummy_f_attr.shape[0]
    seq_len = 49
    frid_channels = 1280
    dummy_f_rid_dec = torch.randn(batch_size, seq_len, frid_channels) # (B, SeqLen_kv, Ckv)

    decoder = Decoder(frid_channels=frid_channels)
    # print(decoder) # 打印模型结构
    vt_prime = decoder(dummy_f_attr, dummy_f_rid_dec)

    print(f"Input f_attr shape: {dummy_f_attr.shape}")
    print(f"Input f_rid shape: {dummy_f_rid_dec.shape}")
    print(f"Output V't shape: {vt_prime.shape}") # 应该为 (B, 4, 80, 80)
    assert vt_prime.shape == (batch_size, 4, 80, 80), "V't shape mismatch!"
    print("-" * 20)

    # --- 梯度检查 ---
    print("\n--- Gradient Check (Corrected Order) ---")
    # 创建需要梯度的输入
    dummy_f_attr_grad = torch.randn(1, 1280, 20, 20, requires_grad=True)
    dummy_f_rid_dec_grad = torch.randn(1, seq_len, frid_channels, requires_grad=True)

    decoder_grad = Decoder(frid_channels=frid_channels)
    vt_prime_grad = decoder_grad(dummy_f_attr_grad, dummy_f_rid_dec_grad)

    # 定义虚拟损失
    dummy_loss = vt_prime_grad.mean()
    print(f"Dummy Loss: {dummy_loss.item()}")

    # 反向传播
    dummy_loss.backward()

    # 检查参数梯度
    all_grads_exist = True
    no_grads_params = []
    # 检查 Decoder 的参数
    for name, param in decoder_grad.named_parameters():
        if param.requires_grad and param.grad is None:
            all_grads_exist = False
            no_grads_params.append(f"Decoder: {name}")

    # 检查输入的梯度（可选）
    if dummy_f_attr_grad.grad is None:
        print("Warning: Gradient for input f_attr is None")
    if dummy_f_rid_dec_grad.grad is None:
        print("Warning: Gradient for input f_rid is None")


    if all_grads_exist:
        print("Gradient check passed: All Decoder parameters have gradients.")
    else:
        print(f"Gradient check failed: Decoder parameters without gradients: {no_grads_params}")
    print("-" * 20)