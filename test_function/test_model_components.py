# 创建 test_model_components.py
import torch
from hifivfs_fal.models.encoder import AttributeEncoder
from hifivfs_fal.models.decoder import Decoder
from hifivfs_fal.models.discriminator import Discriminator

# 定义批次大小和帧数
B, N = 2, 8  # 2个批次，每个批次8帧

# 创建模拟的VAE编码输入 (假设VAE输出的潜在表示尺寸为64x64)
vae_latent = torch.randn(B*N, 4, 64, 64)  # [B*N, 4, 64, 64]

# 创建属性编码器并测试
encoder = AttributeEncoder()
attr_features = encoder(vae_latent)
print(f"Encoder输出形状: {attr_features.shape}")  # 根据您的设计，检查输出尺寸

# 创建解码器并测试
# 创建身份信息
frid = torch.randn(B*N, 512)  # 假设身份特征是512维
decoder = Decoder()
generated_latent = decoder(attr_features, frid)
print(f"Decoder输出形状: {generated_latent.shape}")  # 应该与VAE潜在尺寸匹配

# 测试判别器
discriminator = Discriminator()
real_score = discriminator(vae_latent)
fake_score = discriminator(generated_latent.detach())
print(f"判别器对真实样本的评分: {real_score.mean().item()}")
print(f"判别器对生成样本的评分: {fake_score.mean().item()}")