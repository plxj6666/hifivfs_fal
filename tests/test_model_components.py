# 创建 test_model_components.py
import torch
# 假设你的 hifivfs_fal 包在 Python 路径中，或者调整导入
from hifivfs_fal.models.encoder import AttributeEncoder
from hifivfs_fal.models.decoder import Decoder
from hifivfs_fal.models.discriminator import Discriminator
import torch.nn as nn # 需要导入 nn 来使用 Linear

# 定义批次大小和帧数
B, N = 2, 8  # 2个批次，每个批次8帧
batch_size_total = B * N

# --- 创建模拟输入 (修正尺寸为 64x64) ---
print("--- Creating dummy inputs ---")
# 使用标准SD VAE输出的隐空间尺寸64x64
H, W = 64, 64
vae_latent = torch.randn(batch_size_total, 4, H, W)  # [B*N, 4, 64, 64]
print(f"Input vae_latent shape: {vae_latent.shape}")

# 假设身份特征是 512 维
input_frid_dim = 512
frid = torch.randn(batch_size_total, input_frid_dim)
print(f"Input frid shape: {frid.shape}")
print("-" * 20)


# --- 测试 Encoder ---
print("--- Testing Encoder ---")
encoder = AttributeEncoder()
f_attr, f_low = encoder(vae_latent)
print(f"Encoder input shape: {vae_latent.shape}")
# 预期输出: f_attr=(B*N, 1280, 16, 16), f_low=(B*N, 320, 64, 64)
print(f"Encoder output shape: f_attr={f_attr.shape}, f_low={f_low.shape}")
assert f_attr.shape == (batch_size_total, 1280, 16, 16), "Encoder f_attr shape mismatch!"
assert f_low.shape == (batch_size_total, 320, 64, 64), "Encoder f_low shape mismatch!"
print("-" * 20)


# --- 测试 Decoder ---
print("--- Testing Decoder ---")
# 获取 Decoder 期望的 frid 内部通道数 (kv_channels)
decoder_frid_channels = 1280 # 与 Decoder.__init__ 中的默认值匹配
decoder = Decoder(frid_channels=decoder_frid_channels, allow_frid_projection=True) 

# --- 手动模拟 frid 投影和准备 ---
frid_projector_test = nn.Linear(input_frid_dim, decoder_frid_channels) 
projected_frid = frid_projector_test(frid) # (B*N, 1280)
context_frid = projected_frid.unsqueeze(1) # (B*N, 1, 1280)
print(f"frid shape after projection and unsqueeze: {context_frid.shape}")
# --- 结束模拟准备 ---

# Decoder 输入 f_attr 和准备好的 context_frid
# 直接传入原始的 frid，Decoder的forward会处理投影
generated_latent = decoder(f_attr, frid) 

print(f"Decoder input shapes: f_attr={f_attr.shape}, frid={frid.shape}")
# 预期输出: (B*N, 4, 64, 64)
print(f"Decoder output shape: {generated_latent.shape}")
assert generated_latent.shape == (batch_size_total, 4, 64, 64), "Decoder output shape mismatch!"
print("-" * 20)


# --- 测试 Discriminator ---
print("--- Testing Discriminator ---")
# 判别器输入 64x64
discriminator = Discriminator(input_channels=4, num_layers=3) 
real_score_map = discriminator(vae_latent)
fake_score_map = discriminator(generated_latent.detach()) 

print(f"Discriminator input shape: {vae_latent.shape}")
# 预期输出: (B*N, 1, 7, 7) - 基于64x64输入的计算
expected_H_dis = 7  # 64/8-1=7
expected_W_dis = 7  # 64/8-1=7
print(f"Discriminator output shape (score map): {real_score_map.shape}")
assert real_score_map.shape == (batch_size_total, 1, expected_H_dis, expected_W_dis), "Discriminator output shape mismatch!"

# 计算平均得分用于显示
real_score_mean = real_score_map.mean().item()
fake_score_mean = fake_score_map.mean().item()
print(f"Discriminator average score for real samples: {real_score_mean:.4f}")
print(f"Discriminator average score for fake samples: {fake_score_mean:.4f}")
print("-" * 20)

print("All component shape tests passed!")