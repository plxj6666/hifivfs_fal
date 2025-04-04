# train_fal.py - 加入 TensorBoard 支持 和 恢复训练功能

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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

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
    from hifivfs_fal.utils.face_recognition import DeepFaceRecognizer
    from hifivfs_fal.dataset import FALDataset
    from hifivfs_fal.utils.vae_utils import encode_with_vae, decode_with_vae, convert_tensor_to_cv2_images, extract_gid_from_latent, FridProjector
    from hifivfs_fal import losses
except ImportError as e:
    print(f"导入自定义模块时出错: {e}")
    print("请确保 hifivfs_fal 包在 Python 路径中，并且所有子模块都存在。")
    exit(1)
    
from torch.cuda.amp import GradScaler, autocast

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser(description='HiFiVFS FAL模型训练')
parser.add_argument('--config', type=str, default='./config/fal_config.yaml', 
                    help='配置文件路径')
parser.add_argument('--debug', action='store_true', help='启用调试模式')
parser.add_argument('--logdir', type=str, default='./logs/fal_runs', 
                    help='TensorBoard 日志保存目录')
# **** 新增：恢复训练检查点路径参数 ****
parser.add_argument('--resume_checkpoint', type=str, default=None, 
                    help='要恢复训练的检查点文件路径 (例如: checkpoints/checkpoint_epoch100.pth)')
args = parser.parse_args()

# --- 配置日志 ---
log_filename = f"train_fal_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO, # 根据 debug 参数设置级别
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename), # 保存到文件
        logging.StreamHandler() # 输出到控制台
    ]
)
logger = logging.getLogger('train_fal')
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
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

try:
    config = load_config(args.config)
except Exception:
    exit(1) # 加载配置失败则退出

# --- 确保输出目录存在 ---
try:
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    sample_dir = Path(config['training']['sample_dir'])
    # **** TensorBoard 日志目录 ****
    # 使用更具体的名称，包含配置文件的基本名称，以便区分不同的训练运行
    config_basename = Path(args.config).stem 
    tensorboard_log_dir = Path(args.logdir) / f"{config_basename}_{time.strftime('%Y%m%d_%H%M%S')}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True) # 创建 TensorBoard 日志目录

    logger.info(f"检查点目录: {checkpoint_dir}")
    logger.info(f"样本保存目录: {sample_dir}")
    logger.info(f"TensorBoard 日志目录: {tensorboard_log_dir}")
except KeyError as e:
     logger.error(f"配置文件中缺少必要的键: {e}")
     exit(1)
except Exception as e:
     logger.error(f"创建输出目录时出错: {e}")
     exit(1)

# --- 配置设备 ---
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    logger.info(f"检测到 {n_gpus} 个GPU")
    device = torch.device("cuda")
    use_data_parallel = False # 保持禁用以简化
    # 可以在这里添加显卡选择逻辑，例如使用 CUDA_VISIBLE_DEVICES 环境变量
else:
    device = torch.device("cpu")
    use_data_parallel = False
logger.info(f"使用设备: {device}")

# --- 保存测试样本 ---
def save_sample_images(vt_original, vt_prime, epoch, global_step, writer=None, prefix="sample"):
    """保存原始图像和生成图像的对比, 并可选地记录到 TensorBoard"""
    try:
        # 确保张量在 CPU 上并且是 float 类型用于计算
        vt_original_cpu = vt_original.detach().float().cpu() 
        vt_prime_cpu = vt_prime.detach().float().cpu()     
        
        # Denormalize [-1, 1] -> [0, 1] for visualization
        vt_original_vis = torch.clamp((vt_original_cpu + 1.0) / 2.0, 0.0, 1.0)
        vt_prime_vis = torch.clamp((vt_prime_cpu + 1.0) / 2.0, 0.0, 1.0)
        
        # --- 保存到文件 ---
        num_frames_to_show = min(4, vt_original_vis.shape[0])
        # 调整 figsize 使其更紧凑
        fig, axes = plt.subplots(2, num_frames_to_show, figsize=(3 * num_frames_to_show, 6)) 
        # 处理 axes 不是二维数组的情况
        if num_frames_to_show == 1:
             if axes.ndim == 1: # 如果只有一行或一列
                  axes = axes.reshape(2, 1) # 强制变为 2x1
             elif axes.ndim == 0: # 如果只有一个图
                  fig, ax = plt.subplots(2, 1, figsize=(3, 6))
                  axes = np.array([[ax[0]],[ax[1]]]) # 模拟二维结构
        elif axes.ndim == 1: # 如果只有一行（例如 batch_size > 1 但 < 4）
             axes = axes.reshape(1, -1) # 强制变为 1xN
             # 需要调整下面的索引
             raise NotImplementedError("Handling axes with ndim=1 and multiple columns needs adjustment")

        for i in range(num_frames_to_show):
            img_orig = vt_original_vis[i].permute(1, 2, 0).numpy() # CHW -> HWC
            img_gen = vt_prime_vis[i].permute(1, 2, 0).numpy() # CHW -> HWC
            
            axes[0, i].imshow(img_orig) # 不需要 clip，因为上面已经 clamp
            axes[0, i].set_title(f"Original {i}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(img_gen) # 不需要 clip
            axes[1, i].set_title(f"Generated {i}")
            axes[1, i].axis('off')

        save_path = sample_dir / f"{prefix}_epoch{epoch}_step{global_step}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"保存样本图像到: {save_path}")

        # --- 记录到 TensorBoard ---
        if writer is not None and torchvision is not None:
             try:
                  # 使用 make_grid 创建网格图像
                  # nrow 控制每行显示的图片数量
                  grid_orig = torchvision.utils.make_grid(vt_original_vis[:num_frames_to_show], nrow=num_frames_to_show, normalize=False) # 已经在 [0,1]
                  grid_gen = torchvision.utils.make_grid(vt_prime_vis[:num_frames_to_show], nrow=num_frames_to_show, normalize=False) # 已经在 [0,1]
                  
                  writer.add_image(f'{prefix}/Comparison_Grid', torch.cat((grid_orig, grid_gen), dim=1), global_step) # 上下拼接

             except Exception as e_tb:
                  logger.error(f"记录图像到 TensorBoard 失败: {e_tb}", exc_info=True)

    except Exception as e:
        logger.error(f"保存样本图像失败: {e}", exc_info=True)

# --- 训练函数 ---
writer = None 
def train_fal():
    # 在函数开始时初始化 writer 为 None
    try:
        print("--- Entering train_fal() function ---")
        logger.info("--- Entering train_fal() function ---")
        # **** 初始化 TensorBoard Writer ****
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard writer 初始化完成，日志目录: {tensorboard_log_dir}")

        # --- 加载VAE模型 ---
        vae_model_name = config.get('vae', {}).get('model_name', "stabilityai/sd-vae-ft-mse") # 从配置读取或用默认值
        try:
            logger.info(f"加载VAE模型: {vae_model_name}")
            vae = AutoencoderKL.from_pretrained(vae_model_name, torch_dtype=torch.float32).to(device)
            vae.eval() # 设置为评估模式
            # VAE梯度不计算
            for param in vae.parameters():
                 param.requires_grad = False 
            vae_scale_factor = config.get('vae', {}).get('scale_factor', 0.18215) # 从配置读取或用默认值
            logger.info(f"VAE加载成功. 缩放因子: {vae_scale_factor}")
        except Exception as e:
            logger.error(f"加载VAE模型失败: {e}", exc_info=True)
            if writer: writer.close(); 
            return

        # --- 超参数 ---
        try:
            train_list = config['data']['train_list']
            num_frames = config['data']['num_frames']
            image_size_for_vae = tuple(config['data']['image_size_for_vae'])
            latent_size = (image_size_for_vae[0] // 8, image_size_for_vae[1] // 8) # VAE通常下采样8倍
            
            batch_size = config['training']['batch_size']
            num_epochs = config['training']['num_epochs']
            learning_rate = config['training']['learning_rate']
            
            # 损失权重
            loss_weights = config['training']['loss_weights']
            lambda_rec = loss_weights['reconstruction']
            lambda_attr = loss_weights['attribute'] 
            lambda_tid = loss_weights['identity']
            lambda_adv = loss_weights['adversarial']
            
            save_interval = config['training']['save_interval']
            log_interval = config['training']['log_interval'] # TensorBoard 记录间隔
            sample_save_interval = config['training'].get('sample_save_interval', 100) 
            clip_grad_norm = config['training'].get('clip_grad_norm', None) # 可选的梯度裁剪范数
        except KeyError as e:
             logger.error(f"配置文件中缺少必要的训练参数: {e}")
             if writer: writer.close();
             return

        # --- 模型初始化 ---
        logger.info("初始化模型...")
        try:
            face_recognizer_model = config.get('face_recognition', {}).get('model_name', 'Facenet512')
            face_recognizer = DeepFaceRecognizer(model_name=face_recognizer_model)
            if not face_recognizer.initialized: raise RuntimeError("人脸识别器初始化失败")
            logger.info(f"人脸识别器初始化成功，使用模型: {face_recognizer.model_name}")
        except Exception as e:
            logger.error(f"初始化人脸识别器失败: {e}", exc_info=True)
            if writer: writer.close(); return

        # 初始化 FAL 相关模型
        try:
            encoder = AttributeEncoder(**config.get('model', {}).get('encoder_params', {})).to(device) # 允许传入参数
            decoder = Decoder(**config.get('model', {}).get('decoder_params', {})).to(device) # 允许传入参数
            discriminator = Discriminator(**config.get('model', {}).get('discriminator_params', {})).to(device) # 允许传入参数
            # 获取人脸识别 embedding 大小
            fr_embed_dim = face_recognizer.embedding_size 
            # 获取 decoder 输出通道数，通常等于 encoder 输入通道数
            decoder_frid_channels = config.get('model', {}).get('decoder_params', {}).get('frid_channels', 1280) 
            frid_projector = FridProjector(input_dim=fr_embed_dim, output_dim=decoder_frid_channels).to(device)
            logger.info("FAL 模型初始化完成。")
        except Exception as e:
            logger.error(f"初始化 FAL 模型失败: {e}", exc_info=True)
            if writer: writer.close(); return

        # --- 优化器 ---
        try:
            optimizer_g = optim.AdamW(
                list(encoder.parameters()) + list(decoder.parameters()) + list(frid_projector.parameters()),
                lr=learning_rate,
                betas=(config['training'].get('beta1', 0.5), config['training'].get('beta2', 0.999))
            )
            optimizer_d = optim.AdamW(
                discriminator.parameters(),
                lr=learning_rate,
                betas=(config['training'].get('beta1', 0.5), config['training'].get('beta2', 0.999))
            )
            use_amp = config['training'].get('use_amp', True) # 从配置读取是否使用 AMP
            scaler = GradScaler(enabled=use_amp)
            logger.info("优化器和 GradScaler 初始化完成。")
        except Exception as e:
             logger.error(f"初始化优化器失败: {e}", exc_info=True)
             if writer: writer.close(); return

        # --- 学习率调度器 ---
        try:
            scheduler_g = CosineAnnealingLR(optimizer_g, T_max=num_epochs, eta_min=learning_rate*0.01)
            scheduler_d = CosineAnnealingLR(optimizer_d, T_max=num_epochs, eta_min=learning_rate*0.01)
            logger.info("学习率调度器初始化完成。")
        except Exception as e:
            logger.error(f"初始化学习率调度器失败: {e}", exc_info=True)
            if writer: writer.close(); return

        # --- 加载检查点 ---
        start_epoch = 0
        global_step = 0
        if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
            logger.info(f"尝试从检查点恢复训练: {args.resume_checkpoint}")
            try:
                checkpoint = torch.load(args.resume_checkpoint, map_location=device) 

                # 辅助函数加载状态字典
                def load_state_dict_flexible(model, state_dict, model_name):
                    try:
                        # 检查是否需要移除 'module.' 前缀
                        is_parallel_saved = list(state_dict.keys())[0].startswith('module.')
                        is_current_parallel = isinstance(model, nn.DataParallel)

                        if is_parallel_saved and not is_current_parallel:
                            from collections import OrderedDict
                            new_state_dict = OrderedDict()
                            for k, v in state_dict.items():
                                name = k[7:] # 移除 'module.'
                                new_state_dict[name] = v
                            model.load_state_dict(new_state_dict)
                        elif not is_parallel_saved and is_current_parallel:
                            model.module.load_state_dict(state_dict)
                        else: # 状态匹配
                            model.load_state_dict(state_dict)
                        logger.info(f"成功加载 {model_name} 的状态字典。")
                    except Exception as e_load:
                         logger.error(f"加载 {model_name} 状态字典失败: {e_load}", exc_info=True)
                         raise # 重新抛出异常，让外部捕获

                load_state_dict_flexible(encoder, checkpoint['encoder_state_dict'], "Encoder")
                load_state_dict_flexible(decoder, checkpoint['decoder_state_dict'], "Decoder")
                load_state_dict_flexible(discriminator, checkpoint['discriminator_state_dict'], "Discriminator")
                load_state_dict_flexible(frid_projector, checkpoint['frid_projector_state_dict'], "FridProjector")

                if 'optimizer_g_state_dict' in checkpoint:
                    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                    logger.info("成功加载生成器优化器状态。")
                else: logger.warning("检查点中未找到生成器优化器状态。")
                
                if 'optimizer_d_state_dict' in checkpoint:
                    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
                    logger.info("成功加载判别器优化器状态。")
                else: logger.warning("检查点中未找到判别器优化器状态。")
                    
                if 'scaler_state_dict' in checkpoint and use_amp: # 只有启用 AMP 时才加载
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logger.info("成功加载 GradScaler 状态。")
                elif use_amp: logger.warning("检查点中未找到 GradScaler 状态。")

                start_epoch = checkpoint.get('epoch', 0) 
                global_step = checkpoint.get('global_step', 0) + 1 

                logger.info(f"成功从检查点恢复。将从 Epoch {start_epoch + 1}, Global Step {global_step} 开始训练。")

            except Exception as e:
                logger.error(f"加载检查点失败: {e}", exc_info=True)
                logger.warning("将从头开始训练。")
                start_epoch = 0
                global_step = 0
        else:
            if args.resume_checkpoint: logger.warning(f"指定的检查点文件不存在: {args.resume_checkpoint}。")
            logger.info("将从头开始训练。")

        # --- 创建数据集 ---
        logger.info(f"创建数据集，使用训练列表: {train_list}")
        try:
            # 从配置读取数据集相关参数
            ds_config = config.get('data', {})
            dataset = FALDataset(
                video_list_file=train_list,
                face_recognizer=face_recognizer,
                num_frames=num_frames,
                use_vae_latent=ds_config.get('vae_latent', True),
                vae_encoder_fn=lambda x: encode_with_vae(vae, x, vae_scale_factor),
                image_size_for_vae=image_size_for_vae,
                target_img_size=tuple(ds_config.get('target_img_size', [112, 112])) # 对齐尺寸
            )
            # 从配置读取 DataLoader 参数
            dl_config = config.get('dataloader', {})
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=dl_config.get('shuffle', True), 
                num_workers=dl_config.get('num_workers', 0), 
                pin_memory=dl_config.get('pin_memory', False), 
                drop_last=dl_config.get('drop_last', True)
            )
            logger.info(f"数据集创建成功，共{len(dataset)}个视频片段 (如果可用)。")
            if len(dataloader) == 0:
                 logger.error("DataLoader 为空，请检查数据集路径和内容。")
                 if writer: writer.close();
                 return
        except Exception as e:
            logger.error(f"创建数据集或 DataLoader 失败: {e}", exc_info=True)
            if writer: writer.close();
            return
        
        # --- 训练循环 ---
        logger.info("===== 开始/恢复训练 =====")
        logger.info(f"配置: Batch={batch_size}, LR={learning_rate}, Epochs=[{start_epoch+1}..{num_epochs}], AMP={use_amp}")
        
        for epoch in range(start_epoch, num_epochs):
            encoder.train(); decoder.train(); discriminator.train(); frid_projector.train()
            
            epoch_metrics = {'g_loss': [], 'd_loss': [], 'g_adv': [], 'attr': [], 'rec': [], 'tid': []}
            
            pbar = tqdm(dataloader, desc=f"轮次 {epoch+1}/{num_epochs}")
            for i, batch in enumerate(pbar):
                try:
                    # --- 数据准备 ---
                    vt_latent = batch['vt'].to(device, non_blocking=True)
                    fgid = batch['fgid'].to(device, non_blocking=True)
                    frid = batch['frid'].to(device, non_blocking=True)
                    is_same_identity = batch['is_same_identity'].to(device, non_blocking=True)
                    v_prime_latent = batch.get('v_prime_latent', None)
                    if v_prime_latent is None: v_prime_latent = vt_latent[:, 0:1].clone()
                    v_prime_latent = v_prime_latent.to(device, non_blocking=True)
                    
                    # --- 维度处理 ---
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
                    optimizer_d.zero_grad(set_to_none=True) # 优化
                    with autocast(enabled=use_amp):
                        with torch.no_grad():
                            f_attr_merged, _ = encoder(vt_latent_merged)
                            vt_prime_latent_merged_detached = decoder(f_attr_merged, frid_processed).detach() # 生成并 detach
                        real_scores = discriminator(vt_latent_merged)
                        fake_scores = discriminator(vt_prime_latent_merged_detached)
                        d_loss = losses.compute_D_adv_loss(real_scores, fake_scores, loss_type='bce')
                    scaler.scale(d_loss).backward()
                    if clip_grad_norm is not None: # 梯度裁剪 D
                         scaler.unscale_(optimizer_d)
                         torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_grad_norm)
                    scaler.step(optimizer_d)
                    # scaler.update() # D 和 G 共用一个 scaler 时，update 在 G 之后做

                    # --- 生成器训练 ---
                    optimizer_g.zero_grad(set_to_none=True) # 优化
                    with autocast(enabled=use_amp):
                        f_attr_merged_g, f_low_merged_g = encoder(vt_latent_merged)
                        vt_prime_latent_merged_g = decoder(f_attr_merged_g, frid_processed) # 重建
                        fake_scores_g = discriminator(vt_prime_latent_merged_g) # D 对 G 输出评分
                        g_adv_loss = losses.compute_G_adv_loss(fake_scores_g, loss_type='bce')
                        
                        # 属性损失
                        with torch.no_grad(): f_attr_prime_real, _ = encoder(v_prime_latent_merged)
                        attr_loss = losses.compute_attribute_loss(f_attr_merged_g, f_attr_prime_real)
                        
                        # 重建损失
                        rec_loss = losses.compute_reconstruction_loss(vt_latent_merged, vt_prime_latent_merged_g, is_same_identity_merged, loss_type='l1')
                        
                        # **** 修改：调用 extract_gid_from_latent 时传递 global_step ****
                        try:
                            f_gid_prime_merged_cpu = extract_gid_from_latent(
                                vae, vt_prime_latent_merged_g, vae_scale_factor, 
                                face_recognizer, 
                                global_step=global_step # 传递 global_step
                            )
                            if f_gid_prime_merged_cpu is None:
                                # 这个分支理论上不会被触发了，因为 extract_gid_from_latent 现在总是返回 Tensor
                                logger.error(f"步骤 {global_step}: extract_gid_from_latent 返回 None，这是异常情况！")
                                tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                            else:
                                # 检查返回的 Tensor 是否包含 NaN 或 Inf (虽然不太可能)
                                if not torch.all(torch.isfinite(f_gid_prime_merged_cpu)):
                                    logger.warning(f"步骤 {global_step}: extract_gid_from_latent 返回的 Tensor 包含 NaN 或 Inf！跳过 Ltid 计算。")
                                    tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                                else:
                                    f_gid_prime_merged = f_gid_prime_merged_cpu.to(device)
                                    # 检查是否全零（表示提取失败）
                                    if torch.all(f_gid_prime_merged == 0):
                                        logger.debug(f"步骤 {global_step}: 提取的 f_gid_prime 全零，Ltid 将为 0 或基于零向量计算。")
                                    tid_loss = losses.compute_triplet_identity_loss(
                                        fgid_merged, f_gid_prime_merged, frid_merged,
                                        is_same_identity_merged.to(device), 
                                        margin=loss_weights.get('identity_margin', 0.5)
                                    )
                        except Exception as e:
                            logger.warning(f"步骤 {global_step}: 计算身份损失失败: {e}", exc_info=True)
                            tid_loss = torch.tensor(0.0, device=device, dtype=torch.float32) 
                            # 总损失
                        g_total_loss = losses.compute_G_total_loss(g_adv_loss, attr_loss, rec_loss, tid_loss, lambda_adv, lambda_attr, lambda_rec, lambda_tid)
                        
                    scaler.scale(g_total_loss).backward()
                    if clip_grad_norm is not None: # 梯度裁剪 G
                         scaler.unscale_(optimizer_g)
                         torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()) + list(frid_projector.parameters()), max_norm=clip_grad_norm)
                    scaler.step(optimizer_g)
                    scaler.update() # 更新 scaler
                    
                    # --- 记录损失到列表 ---
                    epoch_metrics['g_loss'].append(g_total_loss.item())
                    epoch_metrics['d_loss'].append(d_loss.item())
                    epoch_metrics['g_adv'].append(g_adv_loss.item())
                    epoch_metrics['attr'].append(attr_loss.item())
                    epoch_metrics['rec'].append(rec_loss.item())
                    epoch_metrics['tid'].append(tid_loss.item() if tid_loss.item() != 0.0 else np.nan)
                    
                    # --- 更新进度条 ---
                    pbar.set_postfix({
                        'D': f"{d_loss.item():.3f}", 'G': f"{g_total_loss.item():.3f}",
                        'Adv': f"{g_adv_loss.item():.3f}", 'Attr': f"{attr_loss.item():.3f}",
                        'Rec': f"{rec_loss.item():.3f}", 'TID': f"{tid_loss.item():.3f}"
                    })
                    
                    # --- 记录到 TensorBoard ---
                    if global_step % log_interval == 0:
                        writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                        writer.add_scalar('Loss/Generator_Total', g_total_loss.item(), global_step)
                        writer.add_scalar('Loss/Generator_Adversarial', g_adv_loss.item(), global_step)
                        writer.add_scalar('Loss/Attribute', attr_loss.item(), global_step)
                        writer.add_scalar('Loss/Reconstruction', rec_loss.item(), global_step)
                        writer.add_scalar('Loss/Identity_Triplet', tid_loss.item(), global_step) 
                        writer.add_scalar('LearningRate/Generator', optimizer_g.param_groups[0]['lr'], global_step)
                        writer.add_scalar('LearningRate/Discriminator', optimizer_d.param_groups[0]['lr'], global_step)
                    
                    # --- 保存样本图像 ---
                    if global_step % sample_save_interval == 0:
                        try:
                            sample_idx = 0 
                            vt_latent_sample = vt_latent[sample_idx].detach()
                            # 使用 vt_prime_latent_merged_detached (来自判别器训练部分) 可能更稳定？或者用 g 的输出
                            vt_prime_latent_sample = vt_prime_latent_merged_g.view(B*N, C_lat, H_lat, W_lat)[sample_idx*N:(sample_idx+1)*N].detach() 
                            
                            # 解码在 CPU 进行以防干扰训练
                            with torch.no_grad():
                                 vt_decoded = decode_with_vae(vae.to('cpu'), vt_latent_sample.cpu(), vae_scale_factor) 
                                 vt_prime_decoded = decode_with_vae(vae.to('cpu'), vt_prime_latent_sample.cpu(), vae_scale_factor)
                                 vae.to(device) # 确保 VAE 回到 GPU

                            save_sample_images(vt_decoded, vt_prime_decoded, epoch + 1, global_step, writer=writer)
                        except Exception as e:
                            logger.error(f"步骤 {global_step}: 保存样本失败: {e}", exc_info=True)
                    
                    # 清理缓存
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                    # 递增全局步数
                    global_step += 1 
                    
                except Exception as e:
                    logger.error(f"处理批次 {i} (全局步骤 {global_step}) 时出错: {e}")
                    logger.error(traceback.format_exc())
                    continue # 跳过这个批次
            
            # --- 轮次结束 ---
            avg_g_loss = np.nanmean(epoch_metrics['g_loss']) if epoch_metrics['g_loss'] else 0
            avg_d_loss = np.nanmean(epoch_metrics['d_loss']) if epoch_metrics['d_loss'] else 0
            avg_adv_loss = np.nanmean(epoch_metrics['g_adv']) if epoch_metrics['g_adv'] else 0
            avg_attr_loss = np.nanmean(epoch_metrics['attr']) if epoch_metrics['attr'] else 0
            avg_rec_loss = np.nanmean(epoch_metrics['rec']) if epoch_metrics['rec'] else 0
            avg_tid_loss = np.nanmean(epoch_metrics['tid']) if epoch_metrics['tid'] else 0 
            
            logger.info(f"轮次 {epoch+1}/{num_epochs} 完成. 平均 D损失: {avg_d_loss:.4f}, 平均 G损失: {avg_g_loss:.4f}")
            logger.info(f"  └─ 平均 G对抗: {avg_adv_loss:.4f}, Lattr: {avg_attr_loss:.4f}, Lrec: {avg_rec_loss:.4f}, Ltid: {avg_tid_loss:.4f}")

            # 记录轮次平均损失到 TensorBoard
            writer.add_scalar('Loss_Epoch/Discriminator', avg_d_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Generator_Total', avg_g_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Generator_Adversarial', avg_adv_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Attribute', avg_attr_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Reconstruction', avg_rec_loss, epoch + 1)
            writer.add_scalar('Loss_Epoch/Identity_Triplet', avg_tid_loss, epoch + 1)

            # --- 更新学习率调度器 ---
            scheduler_g.step()
            scheduler_d.step()

            # --- 保存检查点 ---
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs: # 每隔 interval 或最后一轮保存
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch+1}.pth"
                try:
                    encoder_sd = encoder.module.state_dict() if isinstance(encoder, nn.DataParallel) else encoder.state_dict()
                    decoder_sd = decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict()
                    discriminator_sd = discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict()
                    frid_projector_sd = frid_projector.module.state_dict() if isinstance(frid_projector, nn.DataParallel) else frid_projector.state_dict()
                    
                    torch.save({
                        'epoch': epoch + 1, 'global_step': global_step,
                        'encoder_state_dict': encoder_sd, 'decoder_state_dict': decoder_sd,
                        'discriminator_state_dict': discriminator_sd, 'frid_projector_state_dict': frid_projector_sd,
                        'optimizer_g_state_dict': optimizer_g.state_dict(), 'optimizer_d_state_dict': optimizer_d.state_dict(),
                        'scaler_state_dict': scaler.state_dict()
                    }, checkpoint_path)
                    logger.info(f"保存检查点到: {checkpoint_path}")
                except Exception as e_save:
                     logger.error(f"保存检查点失败: {e_save}", exc_info=True)
    except Exception as e:
        logger.error(f"训练过程中发生严重错误: {e}", exc_info=True)
        if writer: writer.close();
        return
    # --- 训练结束 ---
    logger.info("训练完成。")
    if writer: writer.close() # 确保关闭 writer

# --- 主执行入口 ---
if __name__ == "__main__":
    # 检查 torchvision 是否可用
    if torchvision is None:
        logger.warning("未找到 torchvision，将无法在 TensorBoard 中记录图像样本。请运行 'pip install torchvision'")
        
    try:
        train_fal() 
    except Exception as e:
        logger.error(f"训练主流程发生严重错误: {e}")
        logger.error(traceback.format_exc())
        # 尝试关闭 writer (如果 train_fal 内部异常退出前未关闭)
        if 'writer' in locals() and writer is not None and hasattr(writer, 'close'):
             try:
                  writer.close()
             except Exception as e_close:
                  logger.error(f"关闭 TensorBoard writer 时出错: {e_close}")