import torch
import torch.nn as nn
# 假设 blocks.py 在同一目录下或已正确安装
try:
    from .blocks import ResBlock, SelfAttentionBlock
except ImportError: # 如果直接运行此文件，则尝试绝对导入
    from blocks import ResBlock, SelfAttentionBlock

class Discriminator(nn.Module):
    """
    A PatchGAN-style discriminator, with a structure inspired by the AttributeEncoder.
    It outputs a feature map where each element represents the real/fake prediction
    for a corresponding patch in the input image/feature map.
    """
    def __init__(self, input_channels=4, base_channels=64, num_layers=3):
        """
        Initializes the Discriminator.

        Args:
            input_channels (int): Number of channels in the input tensor (e.g., 4 for VAE latent).
            base_channels (int): Number of channels in the first convolutional layer.
                                 Subsequent layers will multiply this number.
            num_layers (int): Number of downsampling layers (convolutional blocks).
                              The output feature map size depends on this.
        """
        super().__init__()

        # Use a sequence of convolutional blocks for downsampling
        layers = []

        # --- Initial Layer ---
        # No normalization in the first layer is common practice for PatchGAN
        layers.append(
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # --- Intermediate Downsampling Layers ---
        channels_multiplier = 1
        channels_multiplier_prev = 1
        for n in range(1, num_layers):
            channels_multiplier_prev = channels_multiplier
            channels_multiplier = min(2**n, 8) # Cap multiplier at 8 (e.g., 512 channels if base=64)
            layers.append(
                nn.Conv2d(base_channels * channels_multiplier_prev,
                          base_channels * channels_multiplier,
                          kernel_size=4, stride=2, padding=1, bias=False) # Use bias=False if using BatchNorm
            )
            layers.append(nn.BatchNorm2d(base_channels * channels_multiplier))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # Optionally add ResBlock or SelfAttention here if mimicking Eattr more closely
            # Example: layers.append(ResBlock(base_channels * channels_multiplier))

        # --- Final Layer ---
        # Output layer: map to a single channel output feature map (logits)
        channels_multiplier_prev = channels_multiplier
        layers.append(
            nn.Conv2d(base_channels * channels_multiplier_prev, 1,
                      kernel_size=4, stride=1, padding=1) # Stride 1 in the last conv layer
        )
        # No Sigmoid here if using BCEWithLogitsLoss or Hinge Loss

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, input_channels, H, W),
                              e.g., (B, 4, 80, 80).
        Returns:
            torch.Tensor: Output feature map of shape (B, 1, H', W'), representing patch scores.
                          H' and W' depend on num_layers and input H, W.
        """
        return self.model(x)

# --- 仅当直接运行此文件时执行测试 ---
if __name__ == "__main__":
    print("--- Testing Discriminator Shape ---")
    # Example input similar to Vt or V't (assuming VAE latent space)
    batch_size = 4
    input_channels = 4
    H, W = 80, 80
    dummy_input = torch.randn(batch_size, input_channels, H, W)

    # Initialize discriminator with default settings (3 downsampling layers)
    # Input: 80x80
    # Layer 1 (stride 2): 40x40
    # Layer 2 (stride 2): 20x20
    # Layer 3 (stride 2): 10x10
    # Final Conv (stride 1): 10x10
    # Expected output shape: (B, 1, 10, 10)
    discriminator = Discriminator(input_channels=input_channels, base_channels=64, num_layers=3)
    # print(discriminator) # Print model structure

    output = discriminator(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Basic assertion for output shape based on num_layers=3
    expected_H_out = H // (2**3) # 80 / 8 = 10
    expected_W_out = W // (2**3) # 80 / 8 = 10
    # The final conv layer's padding might affect the exact size slightly depending on kernel size
    # Let's check if it's close or calculate more precisely based on the Conv layers used
    # Conv(k=4, s=2, p=1) -> H_out = floor((H_in + 2*p - k)/s + 1)
    # 80 -> floor((80 + 2*1 - 4)/2 + 1) = floor(78/2 + 1) = floor(39+1) = 40
    # 40 -> floor((40 + 2*1 - 4)/2 + 1) = floor(38/2 + 1) = floor(19+1) = 20
    # 20 -> floor((20 + 2*1 - 4)/2 + 1) = floor(18/2 + 1) = floor(9+1) = 10
    # Final Conv(k=4, s=1, p=1) -> floor((10 + 2*1 - 4)/1 + 1) = floor(8 + 1) = 9
    # So the expected output size with kernel=4, padding=1 is actually 9x9 if num_layers=3
    expected_H_out = H // (2**3) -1 # More precise calculation for k=4,p=1
    expected_W_out = W // (2**3) -1 # More precise calculation for k=4,p=1
    print(f"Expected output spatial size: ({expected_H_out}, {expected_W_out})")


    # Let's recalculate based on code layers:
    h_out, w_out = H, W
    # Layer 0: k=4, s=2, p=1 => floor((h_in+2-4)/2+1) = floor(h_in/2)
    h_out, w_out = h_out // 2, w_out // 2 # 40x40
    # Layer 1: k=4, s=2, p=1 => floor(h_in/2)
    h_out, w_out = h_out // 2, w_out // 2 # 20x20
    # Layer 2: k=4, s=2, p=1 => floor(h_in/2)
    h_out, w_out = h_out // 2, w_out // 2 # 10x10
    # Final Layer: k=4, s=1, p=1 => floor((h_in+2-4)/1+1) = floor(h_in-2+1) = h_in-1
    h_out, w_out = h_out - 1, w_out - 1 # 9x9

    print(f"Recalculated expected output spatial size: ({h_out}, {w_out})")
    assert output.shape == (batch_size, 1, h_out, w_out), f"Output shape mismatch! Expected {(batch_size, 1, h_out, w_out)}"
    print("-" * 20)

    # --- 梯度检查 ---
    print("\n--- Gradient Check ---")
    dummy_input_grad = torch.randn(1, input_channels, H, W, requires_grad=True)
    discriminator_grad = Discriminator(input_channels=input_channels, base_channels=64, num_layers=3)
    output_grad = discriminator_grad(dummy_input_grad)

    dummy_loss = output_grad.mean()
    print(f"Dummy Loss: {dummy_loss.item()}")

    dummy_loss.backward()

    all_grads_exist = True
    no_grads_params = []
    for name, param in discriminator_grad.named_parameters():
        if param.requires_grad and param.grad is None:
            all_grads_exist = False
            no_grads_params.append(f"Discriminator: {name}")

    if all_grads_exist:
        print("Gradient check passed: All Discriminator parameters have gradients.")
    else:
        print(f"Gradient check failed: Discriminator parameters without gradients: {no_grads_params}")
    print("-" * 20)
