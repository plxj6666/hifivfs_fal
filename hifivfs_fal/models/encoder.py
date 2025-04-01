import torch
import torch.nn as nn
# 假设 blocks.py 在同一目录下或已正确安装
try:
    from .blocks import ResBlock, SelfAttentionBlock
except ImportError: # 如果直接运行此文件，则尝试绝对导入
    from blocks import ResBlock, SelfAttentionBlock


class AttributeEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 初始卷积层：4 -> 320 通道 (改为 1x1)
        self.initial_conv = nn.Conv2d(4, 320, kernel_size=1, stride=1, padding=0) # 使用 1x1 卷积

        # --- Stage 1 (Output: 320x80x80) ---
        self.stage1_res1 = ResBlock(320)
        self.stage1_sa1 = SelfAttentionBlock(320)
        self.stage1_res2 = ResBlock(320)
        self.stage1_sa2 = SelfAttentionBlock(320)
        # f_low 从这里输出

        # --- Downsample 1 (80x80 -> 40x40) ---
        # 使用 stride=2 的卷积进行下采样，同时增加通道数 320 -> 640
        self.downsample1 = nn.Conv2d(320, 640, kernel_size=3, stride=2, padding=1)

        # --- Stage 2 (Output: 640x40x40) ---
        self.stage2_res1 = ResBlock(640)
        self.stage2_sa1 = SelfAttentionBlock(640, num_heads=8) # 明确指定 num_heads
        self.stage2_res2 = ResBlock(640)
        self.stage2_sa2 = SelfAttentionBlock(640, num_heads=8) # 明确指定 num_heads

        # --- Downsample 2 (40x40 -> 20x20) ---
        # 下采样，增加通道数 640 -> 1280
        self.downsample2 = nn.Conv2d(640, 1280, kernel_size=3, stride=2, padding=1)

        # --- Stage 3 (Output: 1280x20x20) ---
        self.stage3_res1 = ResBlock(1280)
        # 注意：1280 可能不容易被常见的 num_heads (如 8) 整除，需要选择合适的头数
        # 常见的选择可能是 16? (1280 / 16 = 80) 或者调整通道数？
        # 这里我们先用 16 试试，或者如果报错，你需要调整 num_heads 或通道数 1280
        possible_num_heads = 16 # 1280 / 16 = 80
        self.stage3_sa1 = SelfAttentionBlock(1280, num_heads=possible_num_heads)
        self.stage3_res2 = ResBlock(1280)
        self.stage3_sa2 = SelfAttentionBlock(1280, num_heads=possible_num_heads)
        # f_attr 从这里输出

    def forward(self, vt):
        # vt shape: (B, 4, 80, 80)
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

# --- 仅当直接运行此文件时执行测试 ---
if __name__ == "__main__":
    print("--- Testing AttributeEncoder Shape---")
    dummy_vt = torch.randn(2, 4, 80, 80) # 使用 batch_size=2 测试
    encoder = AttributeEncoder()
    # print(encoder) # 打印模型结构
    f_attr, f_low = encoder(dummy_vt)
    print(f"Input Vt shape: {dummy_vt.shape}")
    print(f"Output f_attr shape: {f_attr.shape}") # 应该为 (B, 1280, 20, 20)
    print(f"Output f_low shape: {f_low.shape}")   # 应该为 (B, 320, 80, 80)
    assert f_attr.shape == (2, 1280, 20, 20), "f_attr shape mismatch!"
    assert f_low.shape == (2, 320, 80, 80), "f_low shape mismatch!"
    print("-" * 20)

    # --- 梯度检查 ---
    print("\n--- Gradient Check ---")
    # 创建需要梯度的输入
    dummy_vt_grad = torch.randn(1, 4, 80, 80, requires_grad=True)
    encoder_grad = AttributeEncoder()
    f_attr_grad, f_low_grad = encoder_grad(dummy_vt_grad)

    # 定义虚拟损失
    dummy_loss = (f_attr_grad.mean() + f_low_grad.mean()) * 0.5
    print(f"Dummy Loss: {dummy_loss.item()}")

    # 反向传播
    dummy_loss.backward()

    # 检查参数梯度
    all_grads_exist = True
    no_grads_params = []
    for name, param in encoder_grad.named_parameters():
        if param.requires_grad and param.grad is None: # 只检查需要梯度的参数
            all_grads_exist = False
            no_grads_params.append(name)
            # print(f"Warning: Gradient is None for parameter: {name}")

    if all_grads_exist:
        print("Gradient check passed: All parameters have gradients.")
    else:
        print(f"Gradient check failed: Parameters without gradients: {no_grads_params}")
    print("-" * 20)