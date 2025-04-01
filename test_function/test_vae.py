# 创建 test_vae.py
import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL
from hifivfs_fal.utils.detect_align_face import detect_align_face
from hifivfs_fal.utils.vae_utils import encode_with_vae, decode_with_vae

# 加载测试图像
image_path = "/root/HiFiVFS/data/test_image.jpg"
img = cv2.imread(image_path)

# 人脸对齐
aligned_face = detect_align_face(img, target_size=(112, 112))
if aligned_face is None:
    print("人脸对齐失败")
    exit(1)

print(f"对齐后的人脸尺寸: {aligned_face.shape}")

# 调整大小为VAE期望的尺寸
vae_input_size = (512, 512)
vae_input = cv2.resize(aligned_face, vae_input_size)
print(f"调整为VAE输入尺寸: {vae_input.shape}")

# 预处理为VAE输入
vae_input_rgb = cv2.cvtColor(vae_input, cv2.COLOR_BGR2RGB)
vae_input_norm = (vae_input_rgb.astype(np.float32) / 127.5) - 1.0
vae_input_tensor = torch.from_numpy(vae_input_norm).permute(2, 0, 1).unsqueeze(0)

# 加载VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
vae.eval()

# 编码测试
latent = encode_with_vae(vae, vae_input_tensor, vae.config.scaling_factor)
print(f"潜在表示形状: {latent.shape}")  # 应该是 [1, 4, 64, 64]

# 解码测试
decoded = decode_with_vae(vae, latent, vae.config.scaling_factor)
print(f"解码后的形状: {decoded.shape}")  # 应该是 [1, 3, 512, 512]

# 将解码后的结果转回图像并保存
decoded_img = (decoded[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
decoded_img = decoded_img.astype(np.uint8)
decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("decoded_face.jpg", decoded_img)
print("已保存解码后的面部图像")