import torch
import torch.nn as nn
try:
    from .blocks import ResBlock, SelfAttentionBlock
except ImportError:
    from blocks import ResBlock, SelfAttentionBlock

class Discriminator(nn.Module):
    """
    PatchGAN-style discriminator for 80x80 latent space.
    """
    def __init__(self, input_channels=4, base_channels=64, num_layers=3):
        """
        Args:
            input_channels (int): 4 for VAE latent.
            base_channels (int): Base channel count.
            num_layers (int): Number of downsampling conv blocks.
        """
        super().__init__()

        layers = []
        # --- Initial Layer (Input: 80x80)---
        layers.append(nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # Output: 40x40

        # --- Intermediate Layers ---
        channels_multiplier = 1
        channels_multiplier_prev = 1
        for n in range(1, num_layers): # n=1, n=2
            channels_multiplier_prev = channels_multiplier
            channels_multiplier = min(2**n, 8) # n=1 -> mult=2; n=2 -> mult=4
            layers.append(
                nn.Conv2d(base_channels * channels_multiplier_prev,
                          base_channels * channels_multiplier,
                          kernel_size=4, stride=2, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(base_channels * channels_multiplier))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        # n=1: input 40x40, output 20x20, channels base*2
        # n=2: input 20x20, output 10x10, channels base*4

        # --- Final Layer ---
        # 增加一层额外下采样以处理较大的输入
        channels_multiplier_prev = channels_multiplier  # base*4
        channels_multiplier = min(2**num_layers, 8)     # base*8
        layers.append(
            nn.Conv2d(base_channels * channels_multiplier_prev,
                      base_channels * channels_multiplier,
                      kernel_size=4, stride=2, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(base_channels * channels_multiplier))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # 额外层: input 10x10, output 5x5, channels base*8
        
        # 最终输出层
        layers.append(
            nn.Conv2d(base_channels * channels_multiplier, 1,
                      kernel_size=4, stride=1, padding=1)
        )
        # 最终输出: 4x4 patch scores

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input latent tensor (B, 4, 80, 80).
        Returns:
            torch.Tensor: Output patch scores.
        """
        return self.model(x)

# --- 测试代码 ---
if __name__ == "__main__":
    print("--- Testing Discriminator Shape (640x640 VAE -> 80x80 Latent) ---")
    # 使用80x80输入测试
    batch_size = 4
    input_channels = 4
    H, W = 80, 80
    dummy_input = torch.randn(batch_size, input_channels, H, W)

    discriminator = Discriminator(input_channels=input_channels, base_channels=64, num_layers=3)
    output = discriminator(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape[0] == batch_size and output.shape[1] == 1, "Output channel shape mismatch!"
    print(f"Output spatial dimensions: {output.shape[2]}x{output.shape[3]}")
