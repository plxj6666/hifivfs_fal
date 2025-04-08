# hifivfs_fal/models/embedders.py

import torch
import torch.nn as nn
from einops import rearrange
import logging
import cv2
import numpy as np

# 导入必要的父类和模块 (请确保路径正确)
try:
    from svd.sgm.modules.encoders.modules import AbstractEmbModel
except ImportError:
    # Placeholder - 实际项目中需要确保能正确导入
    class AbstractEmbModel(nn.Module):
         def __init__(self): super().__init__(); self._is_trainable = False; self._ucg_rate = 0.0; self._input_key = "dummy"
         print("警告：使用了临时的 AbstractEmbModel Placeholder！")

# 导入你自己的模块 (请确保路径正确)
try:
    from .dit import DetailedIdentityTokenizer
    from .encoder import AttributeEncoder
    from ..utils.face_recognition import DeepFaceRecognizer # 假设在 utils 下
except ImportError:
     try:
          from dit import DetailedIdentityTokenizer
          from encoder import AttributeEncoder
          from utils.face_recognition import DeepFaceRecognizer
     except ImportError as e:
          print(f"导入 DIT/AttributeEncoder/DeepFaceRecognizer 时出错: {e}")
          raise

logger = logging.getLogger(__name__)

class DILEmbedder(AbstractEmbModel):
    """
    将源图像编码为 DIL token 序列 (tdid)，用于 U-Net Cross-Attention。
    输出键: 'crossattn' (默认)
    """
    def __init__(self,
                 face_recognizer: DeepFaceRecognizer,
                 dit: DetailedIdentityTokenizer,
                 ucg_rate: float = 0.0,
                 input_key: str = "source_image",
                 output_key: str = "crossattn"): # 输出键
        super().__init__()
        if face_recognizer is None or dit is None:
            raise ValueError("必须提供 face_recognizer 和 dit 实例。")

        self.face_recognizer = face_recognizer
        self.dit = dit
        self._ucg_rate = ucg_rate
        self._input_key = input_key
        self.output_key = output_key # 存储输出键
        self._is_trainable = False # DIL 不训练
        self.legacy_ucg_val = None # 兼容旧版 UCG

        # 设置为 eval 模式并禁用梯度
        try: # 假设 face_recognizer 有 model 属性
             self.face_recognizer.model.eval()
             for param in self.face_recognizer.model.parameters(): param.requires_grad = False
        except AttributeError: logger.warning("Face Recognizer 可能没有 'model' 属性或无法设置 eval/requires_grad。")

        self.dit.eval()
        for param in self.dit.parameters(): param.requires_grad = False

        logger.info(f"DILEmbedder 初始化完成 (Input: '{input_key}', Output: '{output_key}'). Face Recognizer 和 DIT 设置为非训练模式。")

    def forward(self, source_image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args: source_image_tensor (torch.Tensor): (B, C_img, H_img, W_img), 值 [-1, 1]。
        Returns: torch.Tensor: DIL token 序列, (B, N_tokens, C_out=dit.output_dim)。
        """
        if source_image_tensor.ndim != 4:
            logger.error(f"DILEmbedder 输入形状错误! 期望 (B, C, H, W), 收到: {source_image_tensor.shape}")
            b = source_image_tensor.shape[0] if source_image_tensor.ndim > 0 else 1
            return torch.zeros((b, self.dit.num_tokens, self.dit.output_dim), device=source_image_tensor.device, dtype=source_image_tensor.dtype)

        device = source_image_tensor.device
        dtype = source_image_tensor.dtype
        batch_size = source_image_tensor.shape[0]
        all_tdid_tokens = []

        # 逐个处理 batch 中的图像 (如果 face_recognizer 不支持批处理)
        for i in range(batch_size):
            single_image_tensor = source_image_tensor[i]

            # 尝试直接使用 Tensor 调用 extract_identity，如果失败则回退到 NumPy
            fdid_np = None
            try:
                # 假设 extract_identity_tensor 接受 (C, H, W), [-1, 1] 的 Tensor
                # _, fdid_map_tensor = self.face_recognizer.extract_identity_tensor(single_image_tensor) # 如果有这个方法
                # if fdid_map_tensor is None: raise TypeError("extract_identity_tensor 返回 None")
                # fdid_map = fdid_map_tensor.unsqueeze(0).to(device) # (1, C_fdid, H_fdid, W_fdid)

                # --- 如果没有 extract_identity_tensor, 使用 NumPy 回退 ---
                img_np_permuted = single_image_tensor.cpu().numpy()
                img_np_rgb = ((img_np_permuted.transpose(1, 2, 0) + 1.0) / 2.0 * 255.0)
                img_np_bgr_uint8 = cv2.cvtColor(img_np_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                # 假设 extract_identity 接受 BGR uint8 numpy
                _, fdid_np = self.face_recognizer.extract_identity(img_np_bgr_uint8)
                # --- NumPy 回退结束 ---

            except Exception as e:
                logger.warning(f"DILEmbedder: 第 {i} 个样本提取 fdid 时出错 (将使用零向量): {e}")
                fdid_np = None # 确保出错时 fdid_np 为 None

            if fdid_np is None:
                # 创建零的 fdid map
                C_fdid, H_fdid, W_fdid = self.dit.input_channels, self.dit.feature_h, self.dit.feature_w
                fdid_map = torch.zeros((1, C_fdid, H_fdid, W_fdid), device=device, dtype=dtype)
            else:
                 # 将 fdid NumPy 转换为 Tensor (C_fdid, H_fdid, W_fdid) 并添加 Batch 维
                 # 假设 fdid_np shape: (H_fdid, W_fdid, C_fdid)
                 try:
                     fdid_tensor = torch.from_numpy(fdid_np).permute(2, 0, 1).float().to(device) # (C, H, W)
                     fdid_map = fdid_tensor.unsqueeze(0) # (1, C, H, W)
                 except Exception as e_conv:
                     logger.error(f"DILEmbedder: 第 {i} 个样本转换 fdid NumPy 到 Tensor 失败 (形状: {fdid_np.shape if fdid_np is not None else 'None'}): {e_conv}")
                     C_fdid, H_fdid, W_fdid = self.dit.input_channels, self.dit.feature_h, self.dit.feature_w
                     fdid_map = torch.zeros((1, C_fdid, H_fdid, W_fdid), device=device, dtype=dtype)


            # 通过 DIT 获取 tokens
            tdid_tokens_single = self.dit(fdid_map) # (1, N_tokens, C_out)
            all_tdid_tokens.append(tdid_tokens_single)

        # 合并 batch 结果
        tdid_tokens_batch = torch.cat(all_tdid_tokens, dim=0) # (B, N_tokens, C_out)

        # 应用 UCG (可选，通常为 False)
        if self.ucg_rate > 0.0:
             mask = torch.bernoulli(
                 (1.0 - self.ucg_rate) * torch.ones(tdid_tokens_batch.shape[0], 1, 1, device=device)
             )
             tdid_tokens_batch = mask * tdid_tokens_batch

        logger.debug(f"DILEmbedder 输出形状: {tdid_tokens_batch.shape}")
        return tdid_tokens_batch


class AttributeEmbedder(AbstractEmbModel):
    """
    将目标视频 VAE latent (vt) 编码为属性特征 token 序列 (f_attr_tokens)。
    f_low 的处理在外部进行。
    输出键: 'f_attr_tokens' (可自定义)
    """
    def __init__(self,
                 attribute_encoder: AttributeEncoder,
                 unet_cross_attn_dim: int, # 目标 Cross-Attention 维度 (必须提供)
                 ucg_rate: float = 0.0,
                 input_key: str = "vt",
                 output_key: str = "f_attr_tokens"): # 自定义输出键
        super().__init__()
        if attribute_encoder is None:
            raise ValueError("必须提供 attribute_encoder 实例。")

        self.attribute_encoder = attribute_encoder
        self._ucg_rate = ucg_rate
        self._input_key = input_key
        self.output_key = output_key # 存储输出键
        self._is_trainable = True # FAL Encoder 通常是需要训练的
        self.legacy_ucg_val = None
        self.f_attr_channels = 1280 # 从 AttributeEncoder 定义可知
        self.target_dim = unet_cross_attn_dim

        # 总是创建投影层，即使维度相同，保持接口一致性
        self.projection = nn.Linear(self.f_attr_channels, self.target_dim)
        logger.info(f"AttributeEmbedder 初始化完成 (Input: '{input_key}', Output: '{output_key}'). "
                    f"f_attr 将从 {self.f_attr_channels} 维投影到 {self.target_dim} 维。")

        # 如果需要训练，确保梯度开启
        if self._is_trainable:
             for param in self.attribute_encoder.parameters(): param.requires_grad = True
             for param in self.projection.parameters(): param.requires_grad = True

    def forward(self, vt_latent: torch.Tensor) -> torch.Tensor:
        """
        Args: vt_latent (torch.Tensor): (B*T, C_lat, H_lat, W_lat)。
        Returns: torch.Tensor: f_attr token 序列, (B*T, N_tokens=400, C_out=unet_cross_attn_dim)。
        """
        if vt_latent.ndim != 4:
            logger.error(f"AttributeEmbedder 输入形状错误! 期望 (B*T, C, H, W), 收到: {vt_latent.shape}")
            bt = vt_latent.shape[0] if vt_latent.ndim > 0 else 1
            return torch.zeros((bt, 20*20, self.target_dim), device=vt_latent.device, dtype=vt_latent.dtype)

        # 1. 获取 f_attr
        # 使用 torch.no_grad() 包裹 f_low 的计算，如果我们只关心 f_attr 的梯度
        # 但如果 AttributeEncoder 的浅层也需要训练，则不应使用 no_grad
        # 假设 AttributeEncoder 整体都需要训练
        f_attr, _ = self.attribute_encoder(vt_latent) # (B*T, 1280, 20, 20)

        # 2. Reshape 为 token 序列
        f_attr_tokens = rearrange(f_attr, 'bt c h w -> bt (h w) c') # (B*T, 400, 1280)

        # 3. 应用投影层
        f_attr_tokens_projected = self.projection(f_attr_tokens) # (B*T, 400, target_dim)

        # 4. 应用 UCG
        if self.ucg_rate > 0.0:
            batch_size = f_attr_tokens_projected.shape[0] # B*T
            mask = torch.bernoulli(
                (1.0 - self.ucg_rate) * torch.ones(batch_size, 1, 1, device=f_attr_tokens_projected.device)
            )
            f_attr_tokens_projected = mask * f_attr_tokens_projected

        logger.debug(f"AttributeEmbedder 输出形状: {f_attr_tokens_projected.shape}")
        return f_attr_tokens_projected