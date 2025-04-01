# train_fal.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
import logging
import os
from tqdm import tqdm # 用于显示进度条

# 导入你自己的模块
from hifivfs_fal.models.encoder import AttributeEncoder
from hifivfs_fal.models.decoder import Decoder
from hifivfs_fal.models.discriminator import Discriminator
from hifivfs_fal.utils.face_recognition import FaceRecognizer
from hifivfs_fal.dataset import FALDataset # 假设你的 Dataset 类叫这个名字
from hifivfs_fal.utils.vae_utils import encode_with_vae, decode_with_vae, convert_tensor_to_cv2_images, extract_gid_from_latent, FridProjector
from hifivfs_fal import losses # 导入整个 losses 模块

# --- 配置 Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- 加载 VAE 模型 ---
vae_model_name = "stabilityai/sd-vae-ft-mse" # 常用 VAE
try:
    logging.info(f"Loading VAE model: {vae_model_name}")
    # 加载时使用 float32 精度进行训练
    vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float32).to(device)
    vae.eval() # 设置为评估模式，不训练 VAE
    # 获取缩放因子
    vae_scale_factor = vae.config.scaling_factor
    logging.info(f"VAE loaded successfully. Scale factor: {vae_scale_factor}")
except Exception as e:
    logging.error(f"Failed to load VAE model: {e}")
    vae = None
    vae_scale_factor = 1.0 # Default value if VAE fails to load

# --- 后续的训练设置将会放在这里 ---
# e.g., 加载其他模型, 定义优化器, 创建 Dataset/DataLoader, 训练循环...

# --- 示例: 定义一个主函数来组织训练 ---
def main():
    if vae is None:
        logging.error("VAE failed to load. Exiting.")
        return

    # --- 超参数设置 (可以从配置文件加载) ---
    data_root = "/path/to/your/voxceleb2_or_similar_dataset" # !!! CHANGE THIS !!!
    num_frames = 8 # Number of frames per clip for Vt
    image_size_for_vae = (512, 512) # VAE 通常期望 512x512 输入
    latent_size = (image_size_for_vae[0] // 8, image_size_for_vae[1] // 8) # e.g., (64, 64) - 需要确认！SD VAE 通常是 H/8, W/8

    batch_size = 1 # 每个 GPU 处理一个视频片段序列 (N帧)
    learning_rate_g = 1e-5
    learning_rate_d = 4e-5 # Discriminator 通常学习率稍高
    epochs = 100
    lambda_adv, lambda_attr, lambda_rec, lambda_tid = 1.0, 10.0, 10.0, 1.0
    margin_tid = 0.5
    gan_loss_type = 'bce' # 'bce' or 'hinge'

    # --- 初始化模型 ---
    logging.info("Initializing models...")
    face_recognizer = FaceRecognizer() # 使用默认设置
    encoder = AttributeEncoder().to(device)
    # frid_channels 通常是 face_recognizer 输出的维度 (512)
    # 但 CrossAttention 的 kv_channels 需要匹配 f_rid 投影后的维度
    # 假设我们将 512 维投影到 1280 维 (这需要一个投影层, 暂时省略)
    # 或者修改 CrossAttention 以接受 512 维？
    # 我们先假设 Decoder 的 frid_channels 仍是 1280, 但需注意 f_rid 需要处理
    decoder = Decoder(frid_channels=1280).to(device) # 注意 frid_channels 的匹配
    discriminator = Discriminator(input_channels=4).to(device) # VAE latent channels = 4

    # --- 定义优化器 ---
    optimizer_g = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate_g, betas=(0.9, 0.999))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=learning_rate_d, betas=(0.9, 0.999))

    # --- 创建数据集和 DataLoader ---
    # VAE 和 FaceRecognizer 实例需要传递给 Dataset 用于预处理
    logging.info("Creating Dataset and DataLoader...")
    # 注意：将 vae 和 device 传递给 Dataset
    # 注意：将 image_size_for_vae 传递给 Dataset
    # 注意：Dataset 内部需要处理 VAE 编码
    dataset = FALDataset(data_root=data_root,
                         face_recognizer=face_recognizer,
                         num_frames=num_frames,
                         use_vae_latent=True, # 启用 VAE
                         vae_encoder_fn=lambda x: encode_with_vae(vae, x, vae_scale_factor), # 传递编码函数
                         image_size_for_vae=image_size_for_vae)

    # 注意：collate_fn 可能需要自定义来处理 batching 序列数据
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # --- 训练循环 ---
    logging.info("Starting training loop...")
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()
        # vae.eval() # VAE 始终是评估模式
        # face_recognizer.app # 内部模型通常也是评估模式

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            # --- 数据准备 ---
            # Dataset 返回的是处理好的数据
            # 注意：DataLoader 会在最前面加一个 Batch 维度 B (B=batch_size)
            # vt_latent: (B, N, 4, H_latent, W_latent)
            # fgid: (B, N, 512)
            # frid: (B, N, 512)
            # is_same_identity: (B, N, 1)

            # 将数据移到设备
            vt_latent = batch['vt'].to(device)
            fgid = batch['fgid'].to(device)
            frid = batch['frid'].to(device)
            is_same_identity = batch['is_same_identity'].to(device) # (B, N, 1) boolean

            # --- 处理维度 ---
            # 模型通常期望 (B*N, C, H, W) 或 (B, C, N, H, W)
            # 简单起见，我们将 B 和 N 合并
            B, N, C_lat, H_lat, W_lat = vt_latent.shape
            vt_latent_merged = vt_latent.view(B * N, C_lat, H_lat, W_lat)

            _, _, C_id = fgid.shape
            fgid_merged = fgid.view(B * N, C_id)
            frid_merged = frid.view(B * N, C_id)
            is_same_identity_merged = is_same_identity.view(B * N, 1) # (B*N, 1) boolean

            # --- !!! f_rid 投影/处理 (Placeholder) !!! ---
            # Decoder 的 CrossAttention 需要 kv_channels=1280 的 context
            # 而 frid_merged 是 (B*N, 512)
            # 需要一个处理步骤将 frid_merged 变为 (B*N, SeqLen, 1280)
            # 简单方法: 加一个线性层 + reshape + repeat
            # frid_processed = project_and_prepare_frid(frid_merged) # 你需要实现这个函数或模块
            # 暂时用假的代替，这部分必须实现
            dummy_seq_len = 1 # Or maybe H_lat*W_lat for full context? Let's use 1.
            frid_processed = FridProjector(input_dim=512, output_dim=1280).to(device) # 使用类版本


            # --- 训练 Discriminator ---
            optimizer_d.zero_grad()

            # 生成 V't
            with torch.no_grad(): # 生成时不需要计算 G 的梯度
                f_attr_merged, _ = encoder(vt_latent_merged)
                vt_prime_latent_merged = decoder(f_attr_merged, frid_processed)

            # 计算 D 的分数
            real_scores = discriminator(vt_latent_merged)
            fake_scores = discriminator(vt_prime_latent_merged.detach()) # detach G 的输出

            # 计算 D 损失
            d_loss = losses.compute_D_adv_loss(real_scores, fake_scores, loss_type=gan_loss_type)

            # 更新 D
            d_loss.backward()
            optimizer_d.step()

            # --- 训练 Generator (Encoder + Decoder) ---
            optimizer_g.zero_grad()

            # 1. 生成 V't (这次需要计算 G 的梯度)
            f_attr_merged_g, f_low_merged_g = encoder(vt_latent_merged)
            vt_prime_latent_merged_g = decoder(f_attr_merged_g, frid_processed) # 使用处理过的 frid

            # 2. 计算 G 的对抗损失
            fake_scores_g = discriminator(vt_prime_latent_merged_g) # 不需要 detach
            g_adv_loss = losses.compute_G_adv_loss(fake_scores_g, loss_type=gan_loss_type)

            # 3. 计算 Attribute Loss (Lattr)
            with torch.no_grad(): # 不需要计算 Eattr 内部梯度
                 f_attr_prime_merged_g = encoder(vt_prime_latent_merged_g)[0] # 只取 f_attr
            attr_loss = losses.compute_attribute_loss(f_attr_merged_g, f_attr_prime_merged_g)

            # 4. 计算 Reconstruction Loss (Lrec) - 在隐空间计算
            rec_loss = losses.compute_reconstruction_loss(vt_latent_merged,
                                                          vt_prime_latent_merged_g,
                                                          is_same_identity_merged,
                                                          loss_type='l1') # 或 'l2'

            # 5. 计算 Triplet Identity Loss (Ltid)
            #    需要将 vt_prime_latent_merged_g 解码回像素空间，提取 f'gid
            f_gid_prime_merged = extract_gid_from_latent(vae,
                                                         vt_prime_latent_merged_g,
                                                         vae_scale_factor,
                                                         face_recognizer) # 需要实现这个辅助函数
            tid_loss = losses.compute_triplet_identity_loss(fgid_merged,
                                                            f_gid_prime_merged,
                                                            frid_merged,
                                                            is_same_identity_merged,
                                                            margin=margin_tid)

            # 6. 计算总 G 损失
            g_total_loss = losses.compute_G_total_loss(g_adv_loss, attr_loss, rec_loss, tid_loss,
                                                      lambda_adv, lambda_attr, lambda_rec, lambda_tid)

            # 更新 G
            g_total_loss.backward()
            optimizer_g.step()

            # --- 记录和打印日志 ---
            if i % 50 == 0: # 每 50 个 batch 打印一次
                pbar.set_postfix({
                    'D Loss': f"{d_loss.item():.4f}",
                    'G Total': f"{g_total_loss.item():.4f}",
                    'G Adv': f"{g_adv_loss.item():.4f}",
                    'Lattr': f"{attr_loss.item():.4f}",
                    'Lrec': f"{rec_loss.item():.4f}",
                    'Ltid': f"{tid_loss.item():.4f}"
                })

        # --- Epoch 结束 ---
        logging.info(f"Epoch {epoch+1} finished. D Loss: {d_loss.item():.4f}, G Loss: {g_total_loss.item():.4f}")
        # --- Optional: 保存模型 checkpoint ---
        # save_checkpoint(epoch, encoder, decoder, discriminator, optimizer_g, optimizer_d)

# --- Entry Point ---
if __name__ == "__main__":
    # Make sure to change the data_root in main() before running
    # It's better to run this via command line: python train_fal.py
    main()