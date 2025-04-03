# train_fal.py - 加入 TensorBoard 支持

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# **** 导入 TensorBoard ****
from torch.utils.tensorboard import SummaryWriter 
from diffusers import AutoencoderKL
import logging
import os
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import time
import numpy as np
import traceback # 导入 traceback
import torchvision
# 导入模块
from hifivfs_fal.models.encoder import AttributeEncoder
from hifivfs_fal.models.decoder import Decoder
from hifivfs_fal.models.discriminator import Discriminator
from hifivfs_fal.utils.face_recognition import DeepFaceRecognizer
from hifivfs_fal.dataset import FALDataset
from hifivfs_fal.utils.vae_utils import encode_with_vae, decode_with_vae, convert_tensor_to_cv2_images, extract_gid_from_latent, FridProjector
from hifivfs_fal import losses
from torch.cuda.amp import GradScaler, autocast

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser(description='HiFiVFS FAL模型训练')
parser.add_argument('--config', type=str, default='./config/fal_config.yaml', 
                    help='配置文件路径')
parser.add_argument('--debug', action='store_true', help='启用调试模式')
# **** 添加 TensorBoard 日志目录参数 ****
parser.add_argument('--logdir', type=str, default='./logs/fal_runs', 
                    help='TensorBoard 日志保存目录')
args = parser.parse_args()

# --- 配置日志 ---
log_filename = f"train_fal_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename), # 保存到文件
        logging.StreamHandler() # 输出到控制台
    ]
)
logger = logging.getLogger('train_fal')
logger.info(f"日志将保存在: {log_filename}")
# 在文件顶部 import 之后
print("--- Imports completed ---")
logger.debug("--- Imports completed (debug) ---") 

# --- 加载配置 ---
def load_config(config_path):
    logger.info(f"加载配置文件: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

config = load_config(args.config)

# --- 确保输出目录存在 ---
checkpoint_dir = Path(config['training']['checkpoint_dir'])
sample_dir = Path(config['training']['sample_dir'])
# **** TensorBoard 日志目录 ****
tensorboard_log_dir = Path(args.logdir) / f"fal_train_{time.strftime('%Y%m%d_%H%M%S')}"

checkpoint_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)
tensorboard_log_dir.mkdir(parents=True, exist_ok=True) # 创建 TensorBoard 日志目录

logger.info(f"检查点目录: {checkpoint_dir}")
logger.info(f"样本保存目录: {sample_dir}")
logger.info(f"TensorBoard 日志目录: {tensorboard_log_dir}")

# --- 配置设备 ---
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    logger.info(f"检测到 {n_gpus} 个GPU")
    
    # 暂时禁用 DataParallel 以简化调试
    # if n_gpus > 1:
    #     logger.info(f"将使用所有 {n_gpus} 个GPU进行训练")
    #     device = torch.device("cuda:0")  
    #     use_data_parallel = True
    # else:
    device = torch.device("cuda")
    use_data_parallel = False
else:
    device = torch.device("cpu")
    use_data_parallel = False

logger.info(f"使用设备: {device}")

# --- 保存测试样本 ---
# save_sample_images 函数保持不变
def save_sample_images(vt_original, vt_prime, epoch, global_step, writer=None, prefix="sample"):
    """保存原始图像和生成图像的对比, 并可选地记录到 TensorBoard"""
    try:
        vt_original_np = vt_original.detach().cpu() # (N, C, H, W)
        vt_prime_np = vt_prime.detach().cpu()     # (N, C, H, W)
        
        # Denormalize [-1, 1] -> [0, 1] for visualization
        vt_original_vis = (vt_original_np + 1.0) / 2.0
        vt_prime_vis = (vt_prime_np + 1.0) / 2.0
        
        # --- 保存到文件 ---
        num_frames_to_show = min(4, vt_original_vis.shape[0])
        fig, axes = plt.subplots(2, num_frames_to_show, figsize=(4 * num_frames_to_show, 8))
        # 如果只有一个样本，axes不是二维数组
        if num_frames_to_show == 1:
             axes = axes.reshape(2, 1)
             
        for i in range(num_frames_to_show):
            img_orig = vt_original_vis[i].permute(1, 2, 0).numpy() # CHW -> HWC
            img_gen = vt_prime_vis[i].permute(1, 2, 0).numpy() # CHW -> HWC
            
            axes[0, i].imshow(np.clip(img_orig, 0, 1)) # Clip以防数值误差
            axes[0, i].set_title(f"原始 {i}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(np.clip(img_gen, 0, 1)) # Clip以防数值误差
            axes[1, i].set_title(f"生成 {i}")
            axes[1, i].axis('off')

        save_path = sample_dir / f"{prefix}_epoch{epoch}_step{global_step}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"保存样本图像到: {save_path}")

        # --- 记录到 TensorBoard ---
        if writer is not None:
             try:
                  # 只记录第一帧对比
                  img_grid_orig = torchvision.utils.make_grid(vt_original_vis[0:1]) # 取第一个样本的第一帧
                  img_grid_gen = torchvision.utils.make_grid(vt_prime_vis[0:1]) # 取第一个样本的第一帧
                  writer.add_image(f'{prefix}/original_frame_0', img_grid_orig, global_step)
                  writer.add_image(f'{prefix}/generated_frame_0', img_grid_gen, global_step)
                  
                  # 或者记录整个批次的前几帧网格 (如果显存允许)
                  # grid_orig = torchvision.utils.make_grid(vt_original_vis[:num_frames_to_show], nrow=num_frames_to_show)
                  # grid_gen = torchvision.utils.make_grid(vt_prime_vis[:num_frames_to_show], nrow=num_frames_to_show)
                  # writer.add_image(f'{prefix}/original_grid', grid_orig, global_step)
                  # writer.add_image(f'{prefix}/generated_grid', grid_gen, global_step)
                  
             except ImportError:
                  logger.warning("未安装 torchvision，无法将图像记录到 TensorBoard。请运行 'pip install torchvision'")
             except Exception as e_tb:
                  logger.error(f"记录图像到 TensorBoard 失败: {e_tb}")

    except Exception as e:
        logger.error(f"保存样本图像失败: {e}", exc_info=True)


# --- 训练函数 ---
def train_fal():
    print("--- Entering train_fal() function ---")
    logger.info("--- Entering train_fal() function ---")
    # **** 初始化 TensorBoard Writer ****
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard writer 初始化完成，日志目录: {tensorboard_log_dir}")

    # --- 加载VAE模型 ---
    # ... (保持不变) ...
    vae_model_name = "stabilityai/sd-vae-ft-mse"
    try:
        logger.info(f"加载VAE模型: {vae_model_name}")
        vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float32).to(device)
        vae.eval()
        vae_scale_factor = vae.config.scaling_factor
        logger.info(f"VAE加载成功. 缩放因子: {vae_scale_factor}")
    except Exception as e:
        logger.error(f"加载VAE模型失败: {e}")
        writer.close() # 关闭 writer
        return

    # --- 超参数 ---
    # ... (保持不变) ...
    train_list = config['data']['train_list']
    num_frames = config['data']['num_frames']
    image_size_for_vae = tuple(config['data']['image_size_for_vae'])
    latent_size = (image_size_for_vae[0] // 8, image_size_for_vae[1] // 8)
    
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    
    lambda_rec = config['training']['loss_weights']['reconstruction']
    lambda_attr = config['training']['loss_weights']['attribute'] 
    lambda_tid = config['training']['loss_weights']['identity']
    lambda_adv = config['training']['loss_weights']['adversarial']
    
    save_interval = config['training']['save_interval']
    log_interval = config['training']['log_interval'] # TensorBoard 记录间隔
    sample_save_interval = config['training'].get('sample_save_interval', 500) # 新增：保存图片样本的间隔步数

    # --- 模型初始化 ---
    # ... (保持不变) ...
    logger.info("初始化模型...")
    face_recognizer = DeepFaceRecognizer(model_name="Facenet512", detector_backend="retinaface")
    if not face_recognizer.initialized:
        logger.error("人脸识别器初始化失败，无法继续训练")
        writer.close()
        return
    logger.info(f"人脸识别器初始化成功，使用模型: {face_recognizer.model_name}")

    encoder = AttributeEncoder().to(device)
    decoder = Decoder(frid_channels=1280, allow_frid_projection=True).to(device)
    discriminator = Discriminator(input_channels=4).to(device)
    frid_projector = FridProjector(input_dim=512, output_dim=1280).to(device)

    # --- 优化器 ---
    # ... (保持不变) ...
    optimizer_g = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(frid_projector.parameters()),
        lr=learning_rate,
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    optimizer_d = optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    use_amp = True
    scaler = GradScaler(enabled=use_amp)

    # --- 创建数据集 ---
    # ... (保持不变) ...
    logger.info(f"创建数据集，使用训练列表: {train_list}")
    try:
        dataset = FALDataset(
            video_list_file=train_list,
            face_recognizer=face_recognizer,
            num_frames=num_frames,
            use_vae_latent=config['data']['vae_latent'],
            vae_encoder_fn=lambda x: encode_with_vae(vae, x, vae_scale_factor),
            image_size_for_vae=image_size_for_vae,
            target_img_size=tuple(config['data']['target_img_size'])
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, # 建议 num_workers > 0 以加速
            pin_memory=False, # 设为 False 可能更稳定
            drop_last=True
        )
        logger.info(f"数据集创建成功，共{len(dataset)}个视频片段")
    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        logger.error(traceback.format_exc())
        writer.close()
        return
    
    # --- 训练循环 ---
    logger.info("===== 开始训练 =====")
    logger.info(f"配置: 批大小={batch_size}, 学习率={learning_rate}, 轮数={num_epochs}")
    
    global_step = 0 # 初始化全局步数计数器

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        discriminator.train()
        
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_adv_losses = []
        epoch_attr_losses = []
        epoch_rec_losses = []
        epoch_tid_losses = []
        
        pbar = tqdm(dataloader, desc=f"轮次 {epoch+1}/{num_epochs}")
        for i, batch in enumerate(pbar):
            # **** 计算当前全局步数 ****
            current_step = epoch * len(dataloader) + i 
            
            try:
                # --- 数据准备 ---
                # ... (保持不变) ...
                vt_latent = batch['vt'].to(device)
                fgid = batch['fgid'].to(device)
                frid = batch['frid'].to(device)
                is_same_identity = batch['is_same_identity'].to(device)
                v_prime_latent = batch.get('v_prime_latent', None)
                
                if v_prime_latent is None:
                    v_prime_latent = vt_latent[:, 0:1].clone()
                v_prime_latent = v_prime_latent.to(device)
                
                # --- 维度处理 ---
                # ... (保持不变) ...
                B, N, C_lat, H_lat, W_lat = vt_latent.shape
                vt_latent_merged = vt_latent.view(B * N, C_lat, H_lat, W_lat)
                _, _, C_id = fgid.shape
                fgid_merged = fgid.view(B * N, C_id)
                frid_merged = frid.view(B * N, C_id)
                is_same_identity_merged = is_same_identity.view(B * N, 1)
                if len(v_prime_latent.shape) == 5: v_prime_latent_merged = v_prime_latent.view(B, C_lat, H_lat, W_lat)
                else: v_prime_latent_merged = v_prime_latent
                
                # --- 处理身份特征 ---
                frid_processed = frid_projector(frid_merged)
                
                # --- 判别器训练 ---
                # ... (保持不变, 但记录损失值) ...
                optimizer_d.zero_grad()
                with autocast(enabled=use_amp):
                    with torch.no_grad():
                        f_attr_merged, _ = encoder(vt_latent_merged)
                        vt_prime_latent_merged = decoder(f_attr_merged, frid_processed)
                    real_scores = discriminator(vt_latent_merged)
                    fake_scores = discriminator(vt_prime_latent_merged.detach())
                    d_loss = losses.compute_D_adv_loss(real_scores, fake_scores, loss_type='bce')
                scaler.scale(d_loss).backward()
                # **** 梯度裁剪 (可选但推荐) ****
                # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0) 
                scaler.step(optimizer_d)
                scaler.update()
                    
                # --- 生成器训练 ---
                # ... (保持不变, 但记录损失值) ...
                optimizer_g.zero_grad()
                with autocast(enabled=use_amp):
                    f_attr_merged_g, f_low_merged_g = encoder(vt_latent_merged)
                    vt_prime_latent_merged_g = decoder(f_attr_merged_g, frid_processed)
                    fake_scores_g = discriminator(vt_prime_latent_merged_g)
                    g_adv_loss = losses.compute_G_adv_loss(fake_scores_g, loss_type='bce')
                    with torch.no_grad(): f_attr_prime_real, _ = encoder(v_prime_latent_merged)
                    attr_loss = losses.compute_attribute_loss(f_attr_merged_g, f_attr_prime_real)
                    rec_loss = losses.compute_reconstruction_loss(
                        vt_latent_merged, vt_prime_latent_merged_g, is_same_identity_merged, loss_type='l1'
                    )
                    try:
                        f_gid_prime_merged_cpu = extract_gid_from_latent(
                            vae, vt_prime_latent_merged_g, vae_scale_factor, face_recognizer
                        )
                        # **** 检查 extract_gid_from_latent 是否返回 None ****
                        if f_gid_prime_merged_cpu is None:
                             logger.warning(f"步骤 {current_step}: extract_gid_from_latent 返回 None，跳过 Ltid 计算。")
                             tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                        else:
                             f_gid_prime_merged = f_gid_prime_merged_cpu.to(device)
                             tid_loss = losses.compute_triplet_identity_loss(
                                 fgid_merged, f_gid_prime_merged, frid_merged,
                                 is_same_identity_merged.to(device), margin=0.5
                             )
                    except Exception as e:
                        logger.warning(f"步骤 {current_step}: 计算身份损失失败: {e}", exc_info=True)
                        tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                        
                    g_total_loss = losses.compute_G_total_loss(
                        g_adv_loss, attr_loss, rec_loss, tid_loss,
                        lambda_adv, lambda_attr, lambda_rec, lambda_tid
                    )
                scaler.scale(g_total_loss).backward()
                # **** 梯度裁剪 (可选但推荐) ****
                # scaler.unscale_(optimizer_g) # 需要先 unscale
                # torch.nn.utils.clip_grad_norm_(
                #     list(encoder.parameters()) + list(decoder.parameters()) + list(frid_projector.parameters()), 
                #     max_norm=1.0
                # )
                scaler.step(optimizer_g)
                scaler.update()
                
                # --- 记录损失到列表 ---
                epoch_g_losses.append(g_total_loss.item())
                epoch_d_losses.append(d_loss.item())
                epoch_adv_losses.append(g_adv_loss.item())
                epoch_attr_losses.append(attr_loss.item())
                epoch_rec_losses.append(rec_loss.item())
                # **** 记录有效的 Ltid (非零时) ****
                valid_tid_loss = tid_loss.item() if tid_loss.item() != 0.0 else np.nan # 使用 NaN 标记失败
                epoch_tid_losses.append(valid_tid_loss)
                
                # --- 更新进度条 ---
                pbar.set_postfix({
                    'D_loss': f"{d_loss.item():.4f}",
                    'G_loss': f"{g_total_loss.item():.4f}",
                    'G_adv': f"{g_adv_loss.item():.4f}",
                    'Lattr': f"{attr_loss.item():.4f}",
                    'Lrec': f"{rec_loss.item():.4f}",
                    'Ltid': f"{tid_loss.item():.4f}" # tqdm 显示瞬时值
                })
                
                # **** 记录损失到 TensorBoard (按 log_interval) ****
                if current_step % log_interval == 0:
                    writer.add_scalar('Loss/Discriminator', d_loss.item(), current_step)
                    writer.add_scalar('Loss/Generator_Total', g_total_loss.item(), current_step)
                    writer.add_scalar('Loss/Generator_Adversarial', g_adv_loss.item(), current_step)
                    writer.add_scalar('Loss/Attribute', attr_loss.item(), current_step)
                    writer.add_scalar('Loss/Reconstruction', rec_loss.item(), current_step)
                    writer.add_scalar('Loss/Identity_Triplet', tid_loss.item(), current_step) # 记录瞬时 Ltid
                    # 可以额外记录学习率
                    writer.add_scalar('LearningRate/Generator', optimizer_g.param_groups[0]['lr'], current_step)
                    writer.add_scalar('LearningRate/Discriminator', optimizer_d.param_groups[0]['lr'], current_step)
                
                # --- 保存样本图像 (按 sample_save_interval) ---
                if current_step % sample_save_interval == 0:
                    try:
                        sample_idx = 0 
                        # 使用 detach() 避免梯度追踪
                        vt_latent_sample = vt_latent[sample_idx].detach()
                        vt_prime_latent_sample = vt_prime_latent_merged_g.view(B*N, C_lat, H_lat, W_lat)[sample_idx*N:(sample_idx+1)*N].detach()
                        
                        # 解码前确保在 CPU 上
                        vt_decoded = decode_with_vae(
                            vae, vt_latent_sample.to(vae.device), vae_scale_factor # 确保在 VAE 设备上
                        ).cpu() # 解码后移回 CPU
                        vt_prime_decoded = decode_with_vae(
                            vae, vt_prime_latent_sample.to(vae.device), vae_scale_factor # 确保在 VAE 设备上
                        ).cpu() # 解码后移回 CPU
                        
                        save_sample_images(
                            vt_decoded, vt_prime_decoded, 
                            epoch + 1, current_step, 
                            writer=writer # 传递 writer 以记录图像
                        )
                    except Exception as e:
                        logger.error(f"步骤 {current_step}: 保存样本失败: {e}", exc_info=True)
                
                # 清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"处理批次 {i} (全局步骤 {current_step}) 时出错: {e}")
                logger.error(traceback.format_exc())
                # 考虑是否需要跳过此批次或停止训练
                continue
        
        # --- 轮次结束 ---
        # 计算轮次平均损失 (忽略 NaN 的 Ltid)
        avg_g_loss = np.nanmean(epoch_g_losses) if epoch_g_losses else 0
        avg_d_loss = np.nanmean(epoch_d_losses) if epoch_d_losses else 0
        avg_adv_loss = np.nanmean(epoch_adv_losses) if epoch_adv_losses else 0
        avg_attr_loss = np.nanmean(epoch_attr_losses) if epoch_attr_losses else 0
        avg_rec_loss = np.nanmean(epoch_rec_losses) if epoch_rec_losses else 0
        avg_tid_loss = np.nanmean(epoch_tid_losses) if epoch_tid_losses else 0 # nanmean 会忽略 NaN
        
        logger.info(f"轮次 {epoch+1} 完成. 平均 D损失: {avg_d_loss:.4f}, 平均 G损失: {avg_g_loss:.4f}")
        logger.info(f"  平均 G对抗: {avg_adv_loss:.4f}, Lattr: {avg_attr_loss:.4f}, Lrec: {avg_rec_loss:.4f}, Ltid (有效均值): {avg_tid_loss:.4f}")

        # **** 记录轮次平均损失到 TensorBoard ****
        writer.add_scalar('Loss_Epoch/Discriminator', avg_d_loss, epoch + 1)
        writer.add_scalar('Loss_Epoch/Generator_Total', avg_g_loss, epoch + 1)
        writer.add_scalar('Loss_Epoch/Generator_Adversarial', avg_adv_loss, epoch + 1)
        writer.add_scalar('Loss_Epoch/Attribute', avg_attr_loss, epoch + 1)
        writer.add_scalar('Loss_Epoch/Reconstruction', avg_rec_loss, epoch + 1)
        writer.add_scalar('Loss_Epoch/Identity_Triplet', avg_tid_loss, epoch + 1) # 记录有效均值

        # --- 保存检查点 ---
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
            # **** 保存时注意 DataParallel ****
            # 如果使用了 DataParallel, 需要保存 .module.state_dict()
            encoder_sd = encoder.module.state_dict() if use_data_parallel else encoder.state_dict()
            decoder_sd = decoder.module.state_dict() if use_data_parallel else decoder.state_dict()
            discriminator_sd = discriminator.module.state_dict() if use_data_parallel else discriminator.state_dict()
            frid_projector_sd = frid_projector.module.state_dict() if use_data_parallel else frid_projector.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'global_step': current_step, # 保存当前的全局步数
                'encoder_state_dict': encoder_sd,
                'decoder_state_dict': decoder_sd,
                'discriminator_state_dict': discriminator_sd,
                'frid_projector_state_dict': frid_projector_sd,
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'scaler_state_dict': scaler.state_dict(), # 保存 GradScaler 的状态
            }, checkpoint_path)
            logger.info(f"保存检查点到: {checkpoint_path}")

    # --- 训练结束 ---
    logger.info("训练完成。")
    writer.close() # 关闭 TensorBoard writer

# ... (train_fal.py 文件中所有之前的代码，包括 train_fal 函数的定义) ...

# --- 主执行入口 ---
if __name__ == "__main__":
    # **** 添加 torchvision 导入以记录图像 ****
    try:
        # 尝试导入 torchvision，如果失败则发出警告
        import torchvision
        logger.debug("torchvision 导入成功，可以记录图像到 TensorBoard。")
    except ImportError:
        logger.warning("未找到 torchvision，将无法在 TensorBoard 中记录图像样本。请运行 'pip install torchvision'")
        # 将 torchvision 设为 None 或其他标记，以便 save_sample_images 知道不要尝试记录图像
        torchvision = None # 或者在 save_sample_images 中再次检查

    # **** 调用训练函数 ****
    try:
        # 开始训练过程
        train_fal() 
    except Exception as e:
        # 捕获训练过程中未被内部处理的严重错误
        logger.error(f"训练主流程发生严重错误: {e}")
        logger.error(traceback.format_exc())
        # 尝试关闭 writer (如果它在 train_fal 内部被初始化但未关闭)
        # 注意：这里 'writer' 可能未定义，更好的做法是在 train_fal 内部使用 finally 关闭
        # if 'writer' in locals() and writer is not None:
        #     writer.close()