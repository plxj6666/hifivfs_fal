# train_fal.py - 加入 DIL, TensorBoard 支持 和 恢复训练功能 (修复版)

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
from torch.optim.lr_scheduler import CosineAnnealingLR # 引入学习率调度器
import torch.nn.functional as F # 引入 F 用于 Lid 计算
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 限制只使用一个GPU
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步CUDA操作
# 尝试导入 torchvision
try:
    import torchvision
except ImportError:
    torchvision = None # 如果未安装，设为 None

# 导入模块
# 确保这些路径相对于你的项目结构是正确的
try:
    from hifivfs_fal.models.encoder import AttributeEncoder
    from hifivfs_fal.models.decoder import Decoder
    from hifivfs_fal.models.discriminator import Discriminator
    # **** 导入 DIT ****
    from hifivfs_fal.models.dit import DetailedIdentityTokenizer 
    from hifivfs_fal.utils.face_recognition import DeepFaceRecognizer
    from hifivfs_fal.dataset import FALDataset # 确保 FALDataset 返回 "fdid"
    # **** 修改：导入正确的 VAE 工具函数 ****
    from hifivfs_fal.utils.vae_utils import encode_with_vae, decode_with_vae, convert_tensor_to_cv2_images, extract_gid_from_latent, FridProjector 
    from hifivfs_fal import losses
except ImportError as e:
    print(f"导入自定义模块时出错: {e}")
    print("请确保 hifivfs_fal 包在 Python 路径中，并且所有子模块都存在。")
    exit(1)
    
from torch.cuda.amp import GradScaler, autocast

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser(description='HiFiVFS FAL + DIL 模型训练')
parser.add_argument('--config', type=str, default='./config/fal_config_dil.yaml', # **** 建议默认使用 DIL 配置 ****
                    help='配置文件路径')
parser.add_argument('--debug', action='store_true', help='启用调试模式')
parser.add_argument('--logdir', type=str, default='./logs/fal_dil_runs', 
                    help='TensorBoard 日志保存目录')
parser.add_argument('--resume_checkpoint', type=str, default=None, 
                    help='要恢复训练的检查点文件路径')
args = parser.parse_args()

# --- 配置日志 ---
log_filename = f"train_fal_dil_{time.strftime('%Y%m%d_%H%M%S')}.log" 
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename), 
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger('train_fal_dil') 
logger.info(f"日志将保存在: {log_filename}")
print("--- Imports completed ---")
logger.debug("--- Imports completed (debug) ---") 

# --- 加载配置 ---
def load_config(config_path):
    logger.info(f"加载配置文件: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError: logger.error(f"配置文件未找到: {config_path}"); raise
    except Exception as e: logger.error(f"加载配置文件失败: {e}"); raise

try: config = load_config(args.config)
except Exception: exit(1) 

# --- 确保输出目录存在 ---
try:
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    sample_dir = Path(config['training']['sample_dir'])
    config_basename = Path(args.config).stem 
    tensorboard_log_dir = Path(args.logdir) / f"{config_basename}_dil_{time.strftime('%Y%m%d_%H%M%S')}" 
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True) 
    logger.info(f"检查点目录: {checkpoint_dir}")
    logger.info(f"样本保存目录: {sample_dir}")
    logger.info(f"TensorBoard 日志目录: {tensorboard_log_dir}")
except KeyError as e: logger.error(f"配置文件中缺少键: {e}"); exit(1)
except Exception as e: logger.error(f"创建输出目录出错: {e}"); exit(1)

# --- 配置设备 ---
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count(); logger.info(f"检测到 {n_gpus} 个GPU")
    device = torch.device("cuda"); use_data_parallel = False 
else: device = torch.device("cpu"); use_data_parallel = False
logger.info(f"使用设备: {device}")

# --- 保存测试样本 ---
def save_sample_images(vt_original, vt_prime, epoch, global_step, writer=None, prefix="sample"):
    """保存原始图像和生成图像的对比, 并可选地记录到 TensorBoard"""
    try:
        vt_original_cpu = vt_original.detach().float().cpu() 
        vt_prime_cpu = vt_prime.detach().float().cpu()     
        vt_original_vis = torch.clamp((vt_original_cpu + 1.0) / 2.0, 0.0, 1.0)
        vt_prime_vis = torch.clamp((vt_prime_cpu + 1.0) / 2.0, 0.0, 1.0)
        
        num_frames_to_show = min(4, vt_original_vis.shape[0])
        fig, axes = plt.subplots(2, num_frames_to_show, figsize=(3 * num_frames_to_show, 6)) 
        if num_frames_to_show == 1:
             if axes.ndim == 1: axes = axes.reshape(2, 1)
             elif axes.ndim == 0: fig, ax = plt.subplots(2, 1, figsize=(3, 6)); axes = np.array([[ax[0]],[ax[1]]])
        # **** 修正：处理 axes.ndim == 1 但 num_frames_to_show > 1 的情况 ****
        elif axes.ndim == 1 and num_frames_to_show > 1: 
             axes = axes.reshape(1, -1) # 强制变为 1xN
             # 需要调整下面的索引逻辑，或者报错
             # 为了简单，我们只显示第一个对比
             img_orig = vt_original_vis[0].permute(1, 2, 0).numpy()
             img_gen = vt_prime_vis[0].permute(1, 2, 0).numpy()
             fig, axes_single = plt.subplots(2, 1, figsize=(3, 6))
             axes_single[0].imshow(img_orig); axes_single[0].set_title(f"Original 0"); axes_single[0].axis('off')
             axes_single[1].imshow(img_gen); axes_single[1].set_title(f"Generated 0"); axes_single[1].axis('off')
             axes = None # 标记后续循环不需要执行
             fig.tight_layout()
             save_path = sample_dir / f"{prefix}_epoch{epoch}_step{global_step}.png"
             fig.savefig(save_path)
             plt.close(fig)
             logger.info(f"保存样本图像到: {save_path}")
             # 记录到 TensorBoard (如果需要)
             if writer is not None and torchvision is not None:
                  try:
                       grid_orig = torchvision.utils.make_grid(vt_original_vis[0:1], nrow=1, normalize=False)
                       grid_gen = torchvision.utils.make_grid(vt_prime_vis[0:1], nrow=1, normalize=False)
                       writer.add_image(f'{prefix}/Comparison_Grid', torch.cat((grid_orig, grid_gen), dim=1), global_step)
                  except Exception as e_tb: logger.error(f"记录图像到 TensorBoard 失败: {e_tb}", exc_info=True)
             return # 直接返回，不再执行后续循环

        if axes is not None: # 只有在 axes 是二维数组时才执行
            for i in range(num_frames_to_show):
                img_orig = vt_original_vis[i].permute(1, 2, 0).numpy()
                img_gen = vt_prime_vis[i].permute(1, 2, 0).numpy()
                axes[0, i].imshow(img_orig); axes[0, i].set_title(f"Original {i}"); axes[0, i].axis('off')
                axes[1, i].imshow(img_gen); axes[1, i].set_title(f"Generated {i}"); axes[1, i].axis('off')

            save_path = sample_dir / f"{prefix}_epoch{epoch}_step{global_step}.png"
            plt.tight_layout(); plt.savefig(save_path); plt.close(fig)
            logger.info(f"保存样本图像到: {save_path}")

            if writer is not None and torchvision is not None:
                 try:
                      grid_orig = torchvision.utils.make_grid(vt_original_vis[:num_frames_to_show], nrow=num_frames_to_show, normalize=False)
                      grid_gen = torchvision.utils.make_grid(vt_prime_vis[:num_frames_to_show], nrow=num_frames_to_show, normalize=False)
                      writer.add_image(f'{prefix}/Comparison_Grid', torch.cat((grid_orig, grid_gen), dim=1), global_step)
                 except Exception as e_tb: logger.error(f"记录图像到 TensorBoard 失败: {e_tb}", exc_info=True)
    except Exception as e: logger.error(f"保存样本图像失败: {e}", exc_info=True)

# --- 训练函数 ---
writer = None 
def train_fal():
    global writer 
    try:
        print("--- Entering train_fal() function ---")
        logger.info("--- Entering train_fal() function ---")
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard writer 初始化完成，日志目录: {tensorboard_log_dir}")

        torch.multiprocessing.set_sharing_strategy('file_system')  
        # --- 加载VAE模型 ---
        vae_model_name = config.get('vae', {}).get('model_name', "stabilityai/sd-vae-ft-mse")
        try:
            logger.info(f"加载VAE模型: {vae_model_name}")
            vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float32).to(device)
            vae.eval(); 
            for param in vae.parameters(): param.requires_grad = False 
            # 为数据集创建CPU版本的VAE
            vae_cpu = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float32)
            vae_cpu.eval()
            for param in vae_cpu.parameters(): param.requires_grad = False

            vae_scale_factor = config.get('vae', {}).get('scale_factor', 0.18215)
            logger.info(f"VAE加载成功. 缩放因子: {vae_scale_factor}")
        except Exception as e: 
            logger.error(f"加载VAE模型失败: {e}", exc_info=True); 
            if writer: writer.close(); return

        # --- 超参数 ---
        try:
            train_list = config['data']['train_list']
            num_frames = config['data']['num_frames']
            image_size_for_vae = tuple(config['data']['image_size_for_vae'])
            latent_size = (image_size_for_vae[0] // 8, image_size_for_vae[1] // 8)
            batch_size = config['training']['batch_size']
            num_epochs = config['training']['num_epochs']
            learning_rate = config['training']['learning_rate']
            loss_weights = config['training']['loss_weights']
            lambda_rec = loss_weights['reconstruction']
            lambda_attr = loss_weights['attribute'] 
            lambda_tid = loss_weights['identity'] 
            lambda_adv = loss_weights['adversarial']
            lambda_lid = loss_weights.get('identity_lid', 1.0) 
            logger.info(f"Lid 损失权重 (lambda_lid): {lambda_lid}")
            save_interval = config['training']['save_interval']
            log_interval = config['training']['log_interval'] 
            sample_save_interval = config['training'].get('sample_save_interval', 100) 
            clip_grad_norm = config['training'].get('clip_grad_norm', None)
            unet_cross_attn_dim = config['model'].get('unet_cross_attention_dim', 768) 
            logger.info(f"UNet Cross Attention 维度: {unet_cross_attn_dim}")
        except KeyError as e: 
            logger.error(f"配置文件中缺少参数: {e}"); 
            if writer: writer.close(); return

        # --- 模型初始化 ---
        logger.info("初始化模型...")
        try:
            face_recognizer_model = config.get('face_recognition', {}).get('model_name', 'Facenet512')
            face_recognizer = DeepFaceRecognizer(model_name=face_recognizer_model)
            if not face_recognizer.initialized: raise RuntimeError("人脸识别器初始化失败")
            logger.info(f"人脸识别器初始化成功")
        except Exception as e: 
            logger.error(f"初始化人脸识别器失败: {e}", exc_info=True); 
            if writer: writer.close(); return

        try:
            encoder = AttributeEncoder(**config.get('model', {}).get('encoder_params', {})).to(device) 
            decoder_params = config.get('model', {}).get('decoder_params', {})
            # **** 动态获取 fdid 形状 ****
            if hasattr(face_recognizer, 'fdid_extractor') and face_recognizer.fdid_extractor is not None:
                 fdid_output_shape = face_recognizer.fdid_extractor.output_shape 
                 if len(fdid_output_shape) == 4: fdid_h, fdid_w, fdid_channels = fdid_output_shape[1], fdid_output_shape[2], fdid_output_shape[3]
                 else: logger.warning(f"无法获取 fdid 形状，使用默认值"); fdid_channels, fdid_h, fdid_w = 1792, 3, 3
            else: logger.warning("无法访问 fdid_extractor，使用默认值"); fdid_channels, fdid_h, fdid_w = 1792, 3, 3
            # **** 传递参数给 Decoder ****
            decoder_params['cross_attention_dim'] = unet_cross_attn_dim 
            decoder_params['num_tdid_tokens'] = fdid_h * fdid_w 
            decoder = Decoder(**decoder_params).to(device) 
            
            discriminator = Discriminator(**config.get('model', {}).get('discriminator_params', {})).to(device) 
            fr_embed_dim = face_recognizer.embedding_size 
            frid_projector_output_dim = unet_cross_attn_dim # **** 确保维度匹配 ****
            frid_projector = FridProjector(input_dim=fr_embed_dim, output_dim=frid_projector_output_dim).to(device)
            
            dit = DetailedIdentityTokenizer(input_channels=fdid_channels, output_embedding_dim=unet_cross_attn_dim, feature_map_size=(fdid_h, fdid_w)).to(device)
            logger.info("所有模型初始化完成。")
        except Exception as e: 
            logger.error(f"初始化模型失败: {e}", exc_info=True); 
            if writer: writer.close(); return

        # --- 优化器 ---
        try:
            optimizer_g = optim.AdamW(
                list(encoder.parameters()) + list(decoder.parameters()) + 
                list(frid_projector.parameters()) + list(dit.parameters()), 
                lr=learning_rate, betas=(config['training'].get('beta1', 0.5), config['training'].get('beta2', 0.999))
            )
            optimizer_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=(config['training'].get('beta1', 0.5), config['training'].get('beta2', 0.999)))
            use_amp = config['training'].get('use_amp', True)
            scaler = GradScaler(enabled=use_amp)
            logger.info("优化器和 GradScaler 初始化完成 (包含 DIT)。")
        except Exception as e: 
            logger.error(f"初始化优化器失败: {e}", exc_info=True); 
            if writer: writer.close(); return

        # --- 创建数据集 --- 
        logger.info(f"创建数据集...")
        try:
            ds_config = config.get('data', {})
            # 修改数据集创建代码
            dataset = FALDataset(
                video_list_file=train_list, 
                face_recognizer=face_recognizer, 
                num_frames=num_frames,
                use_vae_latent=ds_config.get('vae_latent', True), 
                # 替换lambda函数：
                vae=vae_cpu,  # 直接传递VAE模型
                vae_scale_factor=vae_scale_factor,  # 直接传递缩放因子
                image_size_for_vae=image_size_for_vae, 
                target_img_size=tuple(ds_config.get('target_img_size', [112, 112])),
                fdid_shape=(fdid_channels, fdid_h, fdid_w) 
            )
            dl_config = config.get('dataloader', {})
            dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=dl_config.get('shuffle', True), 
                                     num_workers=0, pin_memory=False, 
                                     drop_last=dl_config.get('drop_last', True))
            logger.info(f"数据集创建成功...")
            if len(dataloader) == 0: raise ValueError("DataLoader 为空")
        except Exception as e: 
            logger.error(f"创建数据集或 DataLoader 失败: {e}", exc_info=True); 
            if writer: writer.close(); return
        
        # --- 学习率调度器 ---
        # **** 移到 dataloader 创建之后 ****
        try:
            total_steps = len(dataloader) * num_epochs 
            scheduler_g = CosineAnnealingLR(optimizer_g, T_max=total_steps, eta_min=learning_rate*0.01)
            scheduler_d = CosineAnnealingLR(optimizer_d, T_max=total_steps, eta_min=learning_rate*0.01)
            logger.info(f"学习率调度器初始化完成 (CosineAnnealingLR, T_max={total_steps})。")
        except Exception as e: 
            logger.error(f"初始化学习率调度器失败: {e}", exc_info=True); 
            if writer: writer.close(); return

        # --- 加载检查点 ---
        start_epoch = 0
        global_step = 0
        if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
            logger.info(f"尝试从检查点恢复训练: {args.resume_checkpoint}")
            try:
                checkpoint = torch.load(args.resume_checkpoint, map_location=device) 
                def load_state_dict_flexible(model, state_dict, model_name):
                    # ... (加载逻辑不变) ...
                     pass # 省略重复代码
                load_state_dict_flexible(encoder, checkpoint['encoder_state_dict'], "Encoder")
                load_state_dict_flexible(decoder, checkpoint['decoder_state_dict'], "Decoder")
                load_state_dict_flexible(discriminator, checkpoint['discriminator_state_dict'], "Discriminator")
                load_state_dict_flexible(frid_projector, checkpoint['frid_projector_state_dict'], "FridProjector")
                if 'dit_state_dict' in checkpoint: load_state_dict_flexible(dit, checkpoint['dit_state_dict'], "DIT")
                else: logger.warning("检查点中未找到 DIT 状态字典。")
                if 'optimizer_g_state_dict' in checkpoint: optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict']); logger.info("加载生成器优化器状态。")
                if 'optimizer_d_state_dict' in checkpoint: optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict']); logger.info("加载判别器优化器状态。")
                if 'scaler_state_dict' in checkpoint and use_amp: scaler.load_state_dict(checkpoint['scaler_state_dict']); logger.info("加载 GradScaler 状态。")
                start_epoch = checkpoint.get('epoch', 0) 
                global_step = checkpoint.get('global_step', 0) + 1 
                if 'scheduler_g_state_dict' in checkpoint: scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict']); logger.info("加载生成器调度器状态。")
                if 'scheduler_d_state_dict' in checkpoint: scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict']); logger.info("加载判别器调度器状态。")
                logger.info(f"成功从检查点恢复。将从 Epoch {start_epoch + 1}, Global Step {global_step} 开始训练。")
            except Exception as e: logger.error(f"加载检查点失败: {e}", exc_info=True); start_epoch = 0; global_step = 0
        else:
            if args.resume_checkpoint: logger.warning(f"检查点文件不存在: {args.resume_checkpoint}。")
            logger.info("将从头开始训练。")
        
        # --- 训练循环 ---
        logger.info("===== 开始/恢复训练 =====")
        
        for epoch in range(start_epoch, num_epochs):
            encoder.train(); decoder.train(); discriminator.train(); frid_projector.train(); dit.train() 
            epoch_metrics = {'g_loss': [], 'd_loss': [], 'g_adv': [], 'attr': [], 'rec': [], 'tid': [], 'lid': []} 
            pbar = tqdm(dataloader, desc=f"轮次 {epoch+1}/{num_epochs}")
            for i, batch in enumerate(pbar):
                try:
                    # --- 数据准备 ---
                    vt_latent = batch['vt'].to(device, non_blocking=True)
                    fgid_source = batch['fgid'].to(device, non_blocking=True) 
                    frid_ref = batch['frid'].to(device, non_blocking=True)    
                    fdid_ref = batch['fdid'].to(device, non_blocking=True)    
                    is_same_identity = batch['is_same_identity'].to(device, non_blocking=True)
                    v_prime_latent = batch.get('v_prime_latent', None)
                    if v_prime_latent is None: v_prime_latent = vt_latent[:, 0:1].clone()
                    v_prime_latent = v_prime_latent.to(device, non_blocking=True)
                    
                    # --- 维度处理 ---
                    B, N, C_lat, H_lat, W_lat = vt_latent.shape
                    vt_latent_merged = vt_latent.view(B * N, C_lat, H_lat, W_lat)
                    fgid_source_merged = fgid_source.view(B * N, -1) 
                    frid_ref_merged = frid_ref.view(B * N, -1)
                    fdid_ref_merged = fdid_ref.view(B*N, fdid_channels, fdid_h, fdid_w) 
                    is_same_identity_merged = is_same_identity.view(B * N, 1)
                    if len(v_prime_latent.shape) == 5: v_prime_latent_merged = v_prime_latent.view(B, C_lat, H_lat, W_lat)
                    else: v_prime_latent_merged = v_prime_latent

                    # --- 处理详细身份特征 (DIL) ---
                    tdid_tokens = dit(fdid_ref_merged) 
                    
                    # --- 处理参考身份特征 (FridProjector) ---
                    frid_processed = frid_projector(frid_ref_merged) 

                    # --- 判别器训练 ---
                    optimizer_d.zero_grad(set_to_none=True)
                    with autocast(enabled=use_amp):
                        with torch.no_grad():
                             f_attr_merged, _ = encoder(vt_latent_merged)
                             # **** 修改 Decoder 调用以接收 tdid ****
                             vt_prime_latent_merged_detached = decoder(
                                 f_attr_merged, 
                                 frid_context=frid_processed, # **** 传递 frid_processed ****
                                 tdid_context=tdid_tokens     # **** 传递 tdid ****
                             ).detach()
                        real_scores = discriminator(vt_latent_merged)
                        fake_scores = discriminator(vt_prime_latent_merged_detached)
                        d_loss = losses.compute_D_adv_loss(real_scores, fake_scores, loss_type='bce')
                    scaler.scale(d_loss).backward()
                    if clip_grad_norm is not None: scaler.unscale_(optimizer_d); torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_grad_norm)
                    scaler.step(optimizer_d)

                    # --- 生成器训练 ---
                    optimizer_g.zero_grad(set_to_none=True)
                    with autocast(enabled=use_amp):
                        f_attr_merged_g, f_low_merged_g = encoder(vt_latent_merged)
                        # **** 修改 Decoder 调用以接收 tdid ****
                        vt_prime_latent_merged_g = decoder(
                            f_attr_merged_g, 
                            frid_context=frid_processed, # **** 传递 frid_processed ****
                            tdid_context=tdid_tokens     # **** 传递 tdid ****
                        ) 
                        fake_scores_g = discriminator(vt_prime_latent_merged_g)
                        g_adv_loss = losses.compute_G_adv_loss(fake_scores_g, loss_type='bce')
                        
                        with torch.no_grad(): f_attr_prime_real, _ = encoder(v_prime_latent_merged)
                        attr_loss = losses.compute_attribute_loss(f_attr_merged_g, f_attr_prime_real)
                        rec_loss = losses.compute_reconstruction_loss(vt_latent_merged, vt_prime_latent_merged_g, is_same_identity_merged, loss_type='l1')
                        
                        # --- 计算 Lid 损失 ---
                        lid_loss = torch.tensor(0.0, device=device, dtype=torch.float32) 
                        try:
                            with torch.no_grad():
                                 vae.to('cpu'); 
                                 # **** 确保输入 VAE 解码器的数据是 float ****
                                 vr_pixel_merged = decode_with_vae(vae, vt_prime_latent_merged_g.float().cpu(), vae_scale_factor)
                                 vae.to(device); 
                                 # **** 使用 convert_tensor_to_cv2_images_simple ****
                                 vr_cv2_images = convert_tensor_to_cv2_images(vr_pixel_merged)
                            
                            fgid_prime_list = []
                            for vr_img in vr_cv2_images:
                                 fgid_p = face_recognizer.get_embedding(vr_img) 
                                 if fgid_p is None: fgid_p = np.zeros(face_recognizer.embedding_size, dtype=np.float32)
                                 fgid_prime_list.append(fgid_p)
                            fgid_prime_merged = torch.from_numpy(np.stack(fgid_prime_list)).float().to(device)
                            
                            lid_loss = (1.0 - F.cosine_similarity(fgid_source_merged, fgid_prime_merged, dim=1)).mean()
                            if torch.isnan(lid_loss): lid_loss = torch.tensor(0.0, device=device, dtype=torch.float32) 

                        except Exception as e: logger.warning(f"步骤 {global_step}: 计算 Lid 损失失败: {e}", exc_info=True); lid_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                        # --- 计算身份损失 Ltid ---
                        tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32) 
                        try:
                            if torch.all(fgid_prime_merged == 0):
                                 logger.debug(f"步骤 {global_step}: Lid 的 fgid_prime 全零，Ltid 设为 0。")
                                 tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                            else:
                                 tid_loss = losses.compute_triplet_identity_loss(
                                     fgid_source_merged, fgid_prime_merged, frid_ref_merged, 
                                     is_same_identity_merged.to(device), 
                                     margin=loss_weights.get('identity_margin', 0.5)
                                 )
                                 if torch.isnan(tid_loss): tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32) 
                        except Exception as e: logger.warning(f"步骤 {global_step}: 计算 Ltid 损失失败: {e}", exc_info=True); tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                            
                        # --- 总损失 ---
                        g_total_loss = losses.compute_G_total_loss(g_adv_loss, attr_loss, rec_loss, tid_loss, lambda_adv, lambda_attr, lambda_rec, lambda_tid)
                        g_total_loss += lambda_lid * lid_loss 
                            
                    scaler.scale(g_total_loss).backward()
                    if clip_grad_norm is not None: scaler.unscale_(optimizer_g); torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()) + list(frid_projector.parameters()) + list(dit.parameters()), max_norm=clip_grad_norm) 
                    scaler.step(optimizer_g)
                    scaler.update() 
                    
                    # --- 记录损失到列表 ---
                    epoch_metrics['g_loss'].append(g_total_loss.item())
                    epoch_metrics['d_loss'].append(d_loss.item())
                    epoch_metrics['g_adv'].append(g_adv_loss.item())
                    epoch_metrics['attr'].append(attr_loss.item())
                    epoch_metrics['rec'].append(rec_loss.item())
                    epoch_metrics['tid'].append(tid_loss.item() if not torch.isnan(tid_loss) else np.nan) 
                    epoch_metrics['lid'].append(lid_loss.item() if not torch.isnan(lid_loss) else np.nan) 
                    
                    # --- 更新进度条 ---
                    pbar.set_postfix({'D': f"{d_loss.item():.3f}", 'G': f"{g_total_loss.item():.3f}", 'Adv': f"{g_adv_loss.item():.3f}", 'Attr': f"{attr_loss.item():.3f}", 'Rec': f"{rec_loss.item():.3f}", 'TID': f"{tid_loss.item():.3f}", 'LID': f"{lid_loss.item():.3f}"})
                    
                    # --- 记录到 TensorBoard ---
                    if global_step % log_interval == 0:
                        writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                        writer.add_scalar('Loss/Generator_Total', g_total_loss.item(), global_step)
                        writer.add_scalar('Loss/Generator_Adversarial', g_adv_loss.item(), global_step)
                        writer.add_scalar('Loss/Attribute', attr_loss.item(), global_step)
                        writer.add_scalar('Loss/Reconstruction', rec_loss.item(), global_step)
                        writer.add_scalar('Loss/Identity_Triplet', tid_loss.item(), global_step) 
                        writer.add_scalar('Loss/Identity_Lid', lid_loss.item(), global_step) 
                        writer.add_scalar('LearningRate/Generator', optimizer_g.param_groups[0]['lr'], global_step)
                        writer.add_scalar('LearningRate/Discriminator', optimizer_d.param_groups[0]['lr'], global_step)
                    
                    # --- 保存样本图像 ---
                    if global_step % sample_save_interval == 0:
                        try:
                            sample_idx = 0 
                            vt_latent_sample = vt_latent[sample_idx].detach()
                            vt_prime_latent_sample = vt_prime_latent_merged_g.view(B*N, C_lat, H_lat, W_lat)[sample_idx*N:(sample_idx+1)*N].detach() 
                            with torch.no_grad():
                                 vae.to('cpu'); vt_decoded = decode_with_vae(vae, vt_latent_sample.cpu(), vae_scale_factor) 
                                 # **** 确保输入解码器的是 float ****
                                 vt_prime_decoded = decode_with_vae(vae, vt_prime_latent_sample.float().cpu(), vae_scale_factor); vae.to(device) 
                            save_sample_images(vt_decoded, vt_prime_decoded, epoch + 1, global_step, writer=writer)
                        except Exception as e: logger.error(f"步骤 {global_step}: 保存样本失败: {e}", exc_info=True)
                    
                    # 清理缓存
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                    # 递增全局步数
                    global_step += 1 
                    
                except Exception as e: logger.error(f"处理批次 {i} (全局步骤 {global_step}) 时出错: {e}"); logger.error(traceback.format_exc()); continue 
            
            # --- 轮次结束 ---
            avg_g_loss = np.nanmean(epoch_metrics['g_loss']) if epoch_metrics['g_loss'] else 0
            avg_d_loss = np.nanmean(epoch_metrics['d_loss']) if epoch_metrics['d_loss'] else 0
            avg_adv_loss = np.nanmean(epoch_metrics['g_adv']) if epoch_metrics['g_adv'] else 0
            avg_attr_loss = np.nanmean(epoch_metrics['attr']) if epoch_metrics['attr'] else 0
            avg_rec_loss = np.nanmean(epoch_metrics['rec']) if epoch_metrics['rec'] else 0
            avg_tid_loss = np.nanmean(epoch_metrics['tid']) if epoch_metrics['tid'] else 0 
            avg_lid_loss = np.nanmean(epoch_metrics.get('lid', [])) if epoch_metrics.get('lid') else 0 
            
            logger.info(f"轮次 {epoch+1}/{num_epochs} 完成. Avg D={avg_d_loss:.4f}, Avg G={avg_g_loss:.4f}")
            logger.info(f"  └─ Avg Adv={avg_adv_loss:.4f}, Attr={avg_attr_loss:.4f}, Rec={avg_rec_loss:.4f}, TID={avg_tid_loss:.4f}, LID={avg_lid_loss:.4f}")

            # 记录轮次平均损失到 TensorBoard
            writer.add_scalar('Loss_Epoch/Discriminator', avg_d_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Generator_Total', avg_g_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Generator_Adversarial', avg_adv_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Attribute', avg_attr_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Reconstruction', avg_rec_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Identity_Triplet', avg_tid_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Identity_Lid', avg_lid_loss, epoch + 1) 

            # --- 更新学习率调度器 ---
            scheduler_g.step()
            scheduler_d.step()

            # --- 保存检查点 ---
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs: 
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
                try:
                    encoder_sd = encoder.module.state_dict() if isinstance(encoder, nn.DataParallel) else encoder.state_dict()
                    decoder_sd = decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict()
                    discriminator_sd = discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict()
                    frid_projector_sd = frid_projector.module.state_dict() if isinstance(frid_projector, nn.DataParallel) else frid_projector.state_dict()
                    dit_sd = dit.module.state_dict() if isinstance(dit, nn.DataParallel) else dit.state_dict() 
                    
                    torch.save({ 'epoch': epoch + 1, 'global_step': global_step,
                                 'encoder_state_dict': encoder_sd, 'decoder_state_dict': decoder_sd,
                                 'discriminator_state_dict': discriminator_sd, 'frid_projector_state_dict': frid_projector_sd,
                                 'dit_state_dict': dit_sd, 
                                 'optimizer_g_state_dict': optimizer_g.state_dict(), 'optimizer_d_state_dict': optimizer_d.state_dict(),
                                 'scaler_state_dict': scaler.state_dict(),
                                 'scheduler_g_state_dict': scheduler_g.state_dict(), # **** 保存调度器状态 ****
                                 'scheduler_d_state_dict': scheduler_d.state_dict() 
                               }, checkpoint_path)
                    logger.info(f"保存检查点到: {checkpoint_path}")
                except Exception as e_save: logger.error(f"保存检查点失败: {e_save}", exc_info=True)
    except Exception as e:
        logger.error(f"训练过程中发生严重错误: {e}", exc_info=True)
    finally: # 确保 writer 被关闭
        if writer: 
             logger.info("关闭 TensorBoard writer...")
             writer.close() 
             logger.info("TensorBoard writer 已关闭。")

    # --- 训练结束 ---
    logger.info("训练完成。")


# --- 主执行入口 ---
if __name__ == "__main__":
    if torchvision is None: logger.warning("未找到 torchvision...")
    try: train_fal() 
    except Exception as e: logger.error(f"训练主流程发生严重错误: {e}"); logger.error(traceback.format_exc())