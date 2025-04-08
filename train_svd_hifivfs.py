# train_svd_hifivfs.py

import argparse
import os
import sys
from pathlib import Path
import logging
import yaml
from datetime import timedelta
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf
# CUDA 初始化设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" # 设置 PyTorch 分布式调试级别
# --- 确保项目路径在 sys.path 中 ---
# (如果你的 sgm 和 hifivfs_fal 不是可安装包)
# current_dir = Path(__file__).resolve().parent
# project_root = current_dir.parent # 假设脚本在项目根目录或 scripts/ 下
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))
#     print(f"--- INFO: Prepended project root to sys.path: {project_root} ---")
# --- 路径添加结束 ---

# --- 导入必要的模块 ---
try:
    from svd.sgm.models.diffusion import DiffusionEngine # 导入修改后的 DiffusionEngine
    from svd.sgm.util import instantiate_from_config # SGM 的实例化工具
    # 导入 HiFiVFS 组件 (确保路径正确)
    from fal_dil.dataset import FALDataset
    from fal_dil.models.encoder import AttributeEncoder
    from fal_dil.utils.face_recognition import DeepFaceRecognizer
    from fal_dil.models.dit import DetailedIdentityTokenizer
    # 导入可能需要的 VAE
    from diffusers import AutoencoderKL # 或者 sgm.models.autoencoder.AutoencoderKL
except ImportError as e:
    print(f"导入模块失败: {e}. 请确保所有依赖项已安装且路径正确。")
    sys.exit(1)

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('train_svd_hifivfs')

# --- Pytorch Lightning DataModule ---
class HiFiVFSDataModule(pl.LightningDataModule):
    def __init__(self, data_config: OmegaConf, dataloader_config: OmegaConf,
                 face_recognizer_instance: DeepFaceRecognizer,
                 vae_instance: AutoencoderKL, vae_scale_factor: float):
        super().__init__()
        self.data_config = data_config # data_config 包含了 target 和 params
        self.dataloader_config = dataloader_config
        self.face_recognizer = face_recognizer_instance
        self.vae = vae_instance
        self.vae_scale_factor = vae_scale_factor

    def setup(self, stage: str = None):
        logger.info("Setting up dataset...")
        try:
            # 动态获取 fdid_shape
            fdid_shape = (1792, 3, 3) # 默认值
            # ... (获取 fdid_shape 的逻辑不变) ...
            if hasattr(self.face_recognizer, 'fdid_extractor') and self.face_recognizer.fdid_extractor is not None:
                output_shape = self.face_recognizer.fdid_extractor.output_shape
                if len(output_shape) == 4: fdid_shape = (output_shape[3], output_shape[1], output_shape[2])
                else: logger.warning(f"Face recognizer fdid_extractor output shape {output_shape} 不是 4D，使用默认 fdid_shape {fdid_shape}")
            else: logger.warning("无法从 face recognizer 获取 fdid_extractor，使用默认 fdid_shape {fdid_shape}")

            # --- 修改实例化方式 ---
            # 1. 将整个 data_config 转换为 Python 字典
            data_config_dict = OmegaConf.to_container(self.data_config, resolve=True)

            # 2. 使用转换后的字典和额外的 kwargs 调用 instantiate_from_config
            self.train_dataset = instantiate_from_config(
                data_config_dict, # <<<--- 传递转换后的字典
                # 将实例作为额外的 kwargs 传递
                face_recognizer=self.face_recognizer,
                vae=self.vae,
                vae_scale_factor=self.vae_scale_factor,
                fdid_shape=fdid_shape,
            )
            # --- 修改结束 ---

            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            if len(self.train_dataset) == 0:
                 raise ValueError("训练数据集为空！")
        except Exception as e:
            logger.error(f"创建训练数据集失败: {e}", exc_info=True)
            raise

    def train_dataloader(self):
        logger.info(f"Creating train dataloader with batch size {self.dataloader_config.batch_size}...")
        # DataLoader 参数需要从 dataloader_config 获取
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.dataloader_config.batch_size,
            shuffle=self.dataloader_config.get('shuffle', True),
            num_workers=self.dataloader_config.get('num_workers', 0),
            pin_memory=self.dataloader_config.get('pin_memory', False),
            drop_last=self.dataloader_config.get('drop_last', True),
            persistent_workers=self.dataloader_config.get('num_workers', 0) > 0 # 如果 num_workers > 0，建议启用
        )

    # (可选) 实现 val_dataloader()
    # def val_dataloader(self):
    #     logger.info(f"Creating val dataloader with batch size {self.dataloader_config.batch_size}...")
    #     return torch.utils.data.DataLoader(...)
# --- 主训练函数 ---
# train_svd_hifivfs.py

# ... (导入和日志配置不变) ...

def main(cfg: OmegaConf, args):
    logger.info("===== 开始 SVD + HiFiVFS 训练 =====")
    pl.seed_everything(cfg.get("seed", 42))

    # --- 1. 实例化核心组件 ---
    logger.info("实例化核心组件 (VAE, FaceRecognizer, DIT, AttributeEncoder)...")
    try:
        # VAE (CPU version for DataModule)
        # --- 使用 OmegaConf.to_container 转换配置 ---
        vae_cfg_dict = OmegaConf.to_container(cfg.model.params.first_stage_config, resolve=True)
        vae_model_name = vae_cfg_dict.get('params', {}).get('vae_model_name', "stabilityai/sd-vae-ft-mse")
        vae_scale_factor = cfg.model.params.scale_factor # scale_factor 通常是基本类型，直接获取
        vae_cpu = AutoencoderKL.from_pretrained(vae_model_name).eval()
        for param in vae_cpu.parameters(): param.requires_grad = False
        logger.info("VAE (CPU) 实例化完成。")

        # Face Recognizer
        # --- 使用 OmegaConf.to_container 转换配置 ---
        fr_cfg_dict = OmegaConf.to_container(cfg.model.params.face_recognizer_config, resolve=True)
        face_recognizer_instance = instantiate_from_config(fr_cfg_dict)
        if not face_recognizer_instance.initialized: raise RuntimeError("Face Recognizer 初始化失败")
        logger.info("Face Recognizer 实例化完成。")

        # DIT
        fdid_channels, fdid_h, fdid_w = 1792, 3, 3 # 默认值
        try: # 尝试获取形状
             if hasattr(face_recognizer_instance, 'fdid_extractor') and face_recognizer_instance.fdid_extractor is not None:
                  output_shape = face_recognizer_instance.fdid_extractor.output_shape
                  if len(output_shape) == 4:
                       fdid_channels, fdid_h, fdid_w = output_shape[3], output_shape[1], output_shape[2]
        except Exception: pass # 忽略错误，使用默认值
        unet_context_dim = cfg.model.params.network_config.params.context_dim

        # --- 修改 DIT 配置处理 ---
        # 1. 尝试获取 dit_config OmegaConf 对象
        dit_config_omegaconf = cfg.model.params.get('dit_config') # 使用 get 获取，不存在则为 None

        # 2. 如果存在，则转换为字典；否则使用空字典
        dit_params_dict = {}
        if dit_config_omegaconf is not None:
            try:
                # 只转换 params 部分，因为 target 是用来实例化的
                dit_params_dict = OmegaConf.to_container(dit_config_omegaconf.get('params', {}), resolve=True)
            except Exception as e_conv:
                logger.warning(f"转换 dit_config.params 为字典失败: {e_conv}. 将使用空参数字典。")
        # --- 修改结束 ---

        try:
             # 直接使用动态获取和从 U-Net 配置获取的参数实例化
             dit_instance = DetailedIdentityTokenizer(
                 input_channels=fdid_channels,
                 output_embedding_dim=unet_context_dim,
                 feature_map_size=(fdid_h, fdid_w)
                 # 如果 DIT 还有其他参数，可以从 dit_params_dict 合并
                 # **dit_params_dict
             )
             logger.info(f"DIT 实例化完成。")
        except Exception as e_dit:
             logger.error(f"实例化 DetailedIdentityTokenizer 失败: {e_dit}", exc_info=True)
             return

        # Attribute Encoder
        # --- 使用 OmegaConf.to_container 转换配置 ---
        ae_cfg_dict = OmegaConf.to_container(cfg.model.params.attribute_encoder_config, resolve=True)
        attr_encoder_instance = instantiate_from_config(ae_cfg_dict)
        logger.info("Attribute Encoder 实例化完成。")

    except Exception as e:
        logger.error(f"实例化核心组件失败: {e}", exc_info=True)
        return

    # --- 2. 实例化 DataModule (保持之前的修改) ---
    logger.info("实例化 DataModule...")
    try:
        # 注意：传递给 DataModule 的 data_config 仍然是 OmegaConf 对象
        datamodule = HiFiVFSDataModule(
            data_config=cfg.data, # 传递 OmegaConf 对象
            dataloader_config=cfg.dataloader, # 传递 OmegaConf 对象
            face_recognizer_instance=face_recognizer_instance, # 传递实例
            vae_instance=vae_cpu,
            vae_scale_factor=vae_scale_factor
        )
        datamodule.setup()
    except Exception as e:
        logger.error(f"实例化 DataModule 或调用 setup 失败: {e}", exc_info=True)
        return

    # --- 3. 实例化 DiffusionEngine 模型 ---
    logger.info("实例化 DiffusionEngine 模型...")
    try:
        # 将整个 cfg.model 转为字典，因为它包含了所有子配置
        model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
        # 直接将实例作为 kwargs 传递给 instantiate_from_config
        model = instantiate_from_config(
            model_cfg_dict, # 传递转换后的字典
            attribute_encoder_instance=attr_encoder_instance,
            face_recognizer_instance=face_recognizer_instance,
            dit_instance=dit_instance,
        )
    except Exception as e:
        logger.error(f"实例化 DiffusionEngine 失败: {e}", exc_info=True)
        return

    # --- 4. 配置 Logger, Callbacks, Trainer ---
    logger.info("配置 Logger, Callbacks, Trainer...")
    # ... (Trainer 配置不变，确保 trainer_params 和 checkpoint_params 是字典) ...
    tensorboard_logger = TensorBoardLogger( save_dir=args.logdir, name=Path(args.config).stem )
    # 确保 checkpoint 配置是字典
    ckpt_params_dict = OmegaConf.to_container(cfg.lightning.modelcheckpoint.params, resolve=True)
    checkpoint_callback = ModelCheckpoint( **ckpt_params_dict )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]
    # 确保 trainer 配置是字典
    trainer_params_dict = OmegaConf.to_container(cfg.lightning.trainer, resolve=True)
    trainer_params_dict["logger"] = tensorboard_logger
    trainer_params_dict["callbacks"] = callbacks
    # ... (恢复检查点逻辑不变) ...
    resume_ckpt_path = args.resume_checkpoint
    last_ckpt_path = Path(ckpt_params_dict['dirpath']) / "last.ckpt"
    if resume_ckpt_path is None and last_ckpt_path.is_file(): resume_ckpt_path = str(last_ckpt_path)
    elif resume_ckpt_path and not os.path.isfile(resume_ckpt_path): resume_ckpt_path = None

    try:
        trainer = pl.Trainer(**trainer_params_dict)
    except Exception as e:
        logger.error(f"配置 Trainer 失败: {e}", exc_info=True)
        return

    # --- 5. 启动训练 ---
    # ... (启动训练不变) ...
    logger.info("开始训练...")
    
    # 强制设置resume_ckpt_path为None，跳过检查点加载
    resume_ckpt_path = None
    logger.info("已禁用检查点恢复，将从头开始训练")
    
    # 修改 trainer 配置
    trainer_params_dict["precision"] = "16-mixed"
    trainer_params_dict["strategy"] = "ddp"  # 使用原生 DDP

    # 添加内存优化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try: trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_path)
    except Exception as e: logger.error(f"训练过程中发生错误: {e}", exc_info=True); # ...
    finally: logger.info("===== 训练结束 =====")


if __name__ == "__main__":
    # ... (命令行解析和主配置加载不变) ...
    parser = argparse.ArgumentParser(description='SVD + HiFiVFS Training')
    parser.add_argument('--config', type=str, default='config/svd_hifivfs_train.yaml', help='Path to the configuration file.')
    parser.add_argument('--logdir', type=str, default='./logs/svd_hifivfs_runs', help='Directory for TensorBoard logs.')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to the checkpoint file to resume training from.')
    args = parser.parse_args()
    try: logger.info(f"Loading configuration from: {args.config}"); cfg = OmegaConf.load(args.config)
    except Exception as e: logger.error(f"Failed to load configuration: {e}", exc_info=True); sys.exit(1)
    # 确保目录创建使用普通字符串路径
    log_parent_dir = Path(args.logdir)
    ckpt_parent_dir = Path(cfg.lightning.modelcheckpoint.params.dirpath) # OmegaConf 支持路径属性访问
    try: log_parent_dir.mkdir(parents=True, exist_ok=True); ckpt_parent_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e: logger.warning(f"创建日志/检查点目录时出错: {e}")
    main(cfg, args)