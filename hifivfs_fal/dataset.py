# hifivfs_fal/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import random
from pathlib import Path
import logging
from torch import nn
from diffusers import AutoencoderKL
from typing import Tuple, Optional
# 内部依赖 - 确保路径正确
try:
    from .utils.detect_align_face import detect_align_face 
    from .utils.face_recognition import DeepFaceRecognizer
    from .utils.vae_utils import encode_with_vae # 假设 encode_with_vae 在这里
except ImportError:
     # 尝试相对导入
     try:
          from utils.detect_align_face import detect_align_face
          from utils.face_recognition import DeepFaceRecognizer
          from utils.vae_utils import encode_with_vae
     except ImportError as e:
          print(f"导入 Dataset 依赖时出错: {e}")
          raise

logger = logging.getLogger(__name__)

class FALDataset(Dataset):
# 修改 FALDataset 类的 __init__ 方法

    def __init__(self, video_list_file=None, data_root=None, 
                face_recognizer: DeepFaceRecognizer = None,
                num_frames=16, 
                target_img_size=(112, 112), 
                use_vae_latent=True,
                vae=None,  # 新增：直接接收VAE模型
                vae_scale_factor=None,  # 新增：直接接收缩放因子
                vae_encoder_fn=None,  # 保留向后兼容
                image_size_for_vae=(640, 640),
                fdid_shape=(1792, 3, 3)
                ):
        super().__init__()
        
        if video_list_file is None and data_root is None: raise ValueError("...")
        if face_recognizer is None: raise ValueError("...")
        if use_vae_latent and vae is None and vae_encoder_fn is None: 
            raise ValueError("当use_vae_latent=True时，必须提供vae或vae_encoder_fn")
            
        self.face_recognizer = face_recognizer
        self.num_frames = num_frames
        self.target_img_size = target_img_size
        self.use_vae_latent = use_vae_latent
        self.vae = vae  # 存储VAE模型
        self.vae_scale_factor = vae_scale_factor  # 存储缩放因子
        self.vae_encoder_fn = vae_encoder_fn  # 向后兼容
        self.image_size_for_vae = image_size_for_vae
        self.fdid_shape = fdid_shape

        # --- 初始化视频文件列表 ---
        # ... (加载视频列表逻辑保持不变) ...
        self.video_files = []
        if video_list_file:
            logger.info(f"从文件加载视频列表: {video_list_file}")
            try:
                with open(video_list_file, 'r') as f:
                    for line in f:
                        video_path = line.strip()
                        if os.path.exists(video_path): self.video_files.append(Path(video_path))
                        else: logger.warning(f"视频文件不存在: {video_path}")
            except Exception as e: raise RuntimeError(f"无法加载视频列表文件: {e}")
        elif data_root:
            # ... (扫描目录逻辑) ...
            pass # 省略重复代码
        logger.info(f"找到{len(self.video_files)}个视频文件。")
        if not self.video_files: raise FileNotFoundError("未找到视频文件。")


    def __len__(self):
        return len(self.video_files)

    # _preprocess_face_for_recognition 保持不变
    def _preprocess_face_for_recognition(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        aligned_face_bgr_uint8 = self.face_recognizer.detect_and_align(frame_bgr) 
        if aligned_face_bgr_uint8 is None:
            # logger.warning("_preprocess_face_for_recognition 对齐失败。") # 日志可能过于频繁
            return None
        if aligned_face_bgr_uint8.dtype != np.uint8:
            # ... (类型转换) ...
             if np.issubdtype(aligned_face_bgr_uint8.dtype, np.floating):
                  if aligned_face_bgr_uint8.min() < 0: aligned_face_bgr_uint8 = ((aligned_face_bgr_uint8 + 1.0) / 2.0 * 255.0).astype(np.uint8)
                  else: aligned_face_bgr_uint8 = (aligned_face_bgr_uint8 * 255.0).astype(np.uint8)
             else: logger.error("Aligner returned unexpected type."); return None 
        return aligned_face_bgr_uint8 

    # **** 修改：加载随机人脸时同时获取 fgid 和 fdid ****
    def _load_random_face_features_for_frid(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从随机视频加载 fgid (frid) 和 fdid (fdid_random)"""
        max_retries = 10 # 防止无限循环
        for _ in range(max_retries):
            random_video_path = random.choice(self.video_files)
            cap = cv2.VideoCapture(str(random_video_path))
            if not cap.isOpened(): continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1: cap.release(); continue
            
            # 尝试从视频中随机找一帧有效的
            frame_indices = list(range(total_frames))
            random.shuffle(frame_indices)
            found_features = False
            frid_embedding, fdid_embedding = None, None
            
            for frame_idx in frame_indices[:min(5, total_frames)]: # 最多尝试 5 帧
                 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                 ret, frame = cap.read()
                 if ret:
                      # 调用修改后的 extract_identity 获取两者
                      frid_embedding, fdid_embedding = self.face_recognizer.extract_identity(frame) 
                      if frid_embedding is not None and fdid_embedding is not None:
                           found_features = True
                           break # 找到有效的就跳出内层循环
                 
            cap.release()
            if found_features:
                 logger.debug(f"成功加载随机 frid 和 fdid from {random_video_path}")
                 return frid_embedding, fdid_embedding # 返回两者
                 
        logger.warning(f"尝试 {max_retries} 次后未能加载有效的随机人脸特征。")
        return None, None # 都返回 None

    def __getitem__(self, index):
        # 重试逻辑，防止因单个视频问题导致训练卡住
        max_item_retries = 5
        for retry_count in range(max_item_retries):
            video_path = self.video_files[index] if retry_count == 0 else self.video_files[random.randint(0, len(self)-1)]
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"尝试 {retry_count+1}/{max_item_retries}: 打开视频失败 {video_path}")
                if retry_count == max_item_retries - 1: raise IOError(f"多次尝试后仍无法打开视频: {video_path}")
                continue # 尝试下一个随机视频

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.num_frames:
                cap.release()
                logger.warning(f"尝试 {retry_count+1}/{max_item_retries}: 视频 {video_path} 帧数不足 ({total_frames} < {self.num_frames})")
                if retry_count == max_item_retries - 1: raise ValueError(f"多次尝试后仍未找到足够帧数的视频")
                continue # 尝试下一个随机视频

            # --- 加载 Vt 视频帧 ---
            try:
                start_frame = random.randint(0, total_frames - self.num_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                vt_frames_bgr = []
                for _ in range(self.num_frames):
                    ret, frame = cap.read()
                    if not ret: raise IOError("读取 Vt 帧失败")
                    vt_frames_bgr.append(frame)
            except Exception as e:
                 cap.release()
                 logger.warning(f"尝试 {retry_count+1}/{max_item_retries}: 加载 Vt 帧失败 from {video_path}: {e}")
                 if retry_count == max_item_retries - 1: raise e
                 continue # 尝试下一个随机视频

            # --- 加载 V' 参考帧 ---
            try:
                reference_frame_idx = random.randint(0, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_idx)
                ret, v_prime_frame = cap.read()
                if not ret: raise IOError("读取 V' 帧失败")
            except Exception as e:
                 cap.release()
                 logger.warning(f"尝试 {retry_count+1}/{max_item_retries}: 加载 V' 帧失败 from {video_path}: {e}")
                 if retry_count == max_item_retries - 1: raise e
                 continue # 尝试下一个随机视频
            finally:
                 cap.release() # 确保 VideoCapture 被释放

            # --- 处理 V' (逻辑不变) ---
# 修改V'处理部分

# --- 处理 V' ---
            try:
                if self.use_vae_latent:
                    if self.vae is not None:
                        # 直接使用VAE模型
                        v_prime_resized = cv2.resize(v_prime_frame, (self.image_size_for_vae[1], self.image_size_for_vae[0]))
                        v_prime_rgb = cv2.cvtColor(v_prime_resized, cv2.COLOR_BGR2RGB)
                        v_prime_normalized = (v_prime_rgb.astype(np.float32) / 127.5) - 1.0
                        v_prime_chw = np.transpose(v_prime_normalized, (2, 0, 1))
                        v_prime_tensor = torch.from_numpy(v_prime_chw).unsqueeze(0).cpu()
                        with torch.no_grad():
                            v_prime_latent = encode_with_vae(self.vae.cpu(), v_prime_tensor, self.vae_scale_factor)
                    elif self.vae_encoder_fn:
                        # 原有逻辑作为备选
                        v_prime_resized = cv2.resize(v_prime_frame, (self.image_size_for_vae[1], self.image_size_for_vae[0]))
                        v_prime_rgb = cv2.cvtColor(v_prime_resized, cv2.COLOR_BGR2RGB)
                        v_prime_normalized = (v_prime_rgb.astype(np.float32) / 127.5) - 1.0
                        v_prime_chw = np.transpose(v_prime_normalized, (2, 0, 1))
                        v_prime_tensor = torch.from_numpy(v_prime_chw).unsqueeze(0) 
                        v_prime_latent = self.vae_encoder_fn(v_prime_tensor)
                    else: raise RuntimeError("vae和vae_encoder_fn都为None")
                else:
                    v_prime_resized = cv2.resize(v_prime_frame, (self.target_img_size[1], self.target_img_size[0]))
                    v_prime_normalized = (v_prime_resized.astype(np.float32) / 127.5) - 1.0
                    v_prime_latent = torch.from_numpy(v_prime_normalized).permute(2, 0, 1) # 返回 (C, H, W)
            except Exception as e:
                 logger.warning(f"尝试 {retry_count+1}/{max_item_retries}: 处理 V' 失败 from {video_path}: {e}")
                 if retry_count == max_item_retries - 1: raise e
                 continue # 尝试下一个随机视频

            # --- 处理 Vt (逻辑不变) ---
            # 修改Vt处理部分

# --- 处理 Vt ---
            try:
                if self.use_vae_latent:
                    if self.vae is not None:
                        # 直接使用VAE模型
                        vae_ready_frames = []
                        for frame in vt_frames_bgr:
                            frame_resized = cv2.resize(frame, (self.image_size_for_vae[1], self.image_size_for_vae[0]))
                            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                            frame_normalized = (frame_rgb.astype(np.float32) / 127.5) - 1.0
                            frame_chw = np.transpose(frame_normalized, (2, 0, 1))
                            vae_ready_frames.append(frame_chw)
                        frames_tensor = torch.from_numpy(np.stack(vae_ready_frames)).cpu()
                        with torch.no_grad():
                            vt = encode_with_vae(self.vae.cpu(), frames_tensor, self.vae_scale_factor)
                    elif self.vae_encoder_fn:
                        vae_ready_frames = []
                        for frame in vt_frames_bgr:
                            frame_resized = cv2.resize(frame, (self.image_size_for_vae[1], self.image_size_for_vae[0]))
                            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                            frame_normalized = (frame_rgb.astype(np.float32) / 127.5) - 1.0
                            frame_chw = np.transpose(frame_normalized, (2, 0, 1))
                            vae_ready_frames.append(frame_chw)
                        frames_tensor = torch.from_numpy(np.stack(vae_ready_frames))
                        vt = self.vae_encoder_fn(frames_tensor) # (N, C_lat, H_lat, W_lat)
                    else: raise RuntimeError("vae_encoder_fn is None")
                else:
                    processed_frames = []
                    for frame in vt_frames_bgr:
                        frame_resized = cv2.resize(frame, (self.target_img_size[1], self.target_img_size[0]))
                        frame_normalized = (frame_resized.astype(np.float32) / 127.5) - 1.0
                        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
                        processed_frames.append(frame_tensor)
                    vt = torch.stack(processed_frames) # (N, 3, H, W)
            except Exception as e:
                 logger.warning(f"尝试 {retry_count+1}/{max_item_retries}: 处理 Vt 失败 from {video_path}: {e}")
                 if retry_count == max_item_retries - 1: raise e
                 continue # 尝试下一个随机视频
                 
            # --- **** 修改：获取 fgid 和 fdid **** ---
            # 从 vt_frames_bgr 中随机选一帧用于提取源身份
            random_frame_for_source_id = random.choice(vt_frames_bgr)
            
            # 调用修改后的 extract_identity 获取 fgid 和 fdid
            fgid_np, fdid_np = self.face_recognizer.extract_identity(random_frame_for_source_id)

            if fgid_np is None or fdid_np is None:
                logger.warning(f"尝试 {retry_count+1}/{max_item_retries}: 未能从 {video_path} 提取源身份特征 (fgid 或 fdid 为 None)。")
                if retry_count == max_item_retries - 1: raise ValueError("多次尝试后仍无法提取源身份特征")
                continue # 尝试下一个随机视频
                
            # 将 NumPy 转换为 Tensor
            fgid_source = torch.from_numpy(fgid_np).float() # (embed_dim,)
            # fdid 是 (H', W', C')，需要转为 (C', H', W')
            fdid_source = torch.from_numpy(fdid_np).permute(2, 0, 1).float() # (C_fdid, H_fdid, W_fdid)

            # --- **** 修改：选择 frid 和对应的 fdid_ref **** ---
            if random.random() < 0.5:
                # 使用相同身份
                frid = fgid_source.clone()
                fdid_ref = fdid_source.clone() # **** 使用相同的 fdid ****
                is_same_identity = torch.tensor(True)
            else:
                # 使用不同身份 (加载随机人脸的 fgid 和 fdid)
                frid_numpy, fdid_random_numpy = self._load_random_face_features_for_frid()
                
                if frid_numpy is None or fdid_random_numpy is None:
                     logger.warning(f"尝试 {retry_count+1}/{max_item_retries}: 未能加载随机参考身份特征。")
                     if retry_count == max_item_retries - 1: raise ValueError("多次尝试后仍无法加载随机参考身份特征")
                     continue # 尝试下一个随机视频
                     
                frid = torch.from_numpy(frid_numpy).float()
                # **** 使用随机加载的 fdid ****
                fdid_ref = torch.from_numpy(fdid_random_numpy).permute(2, 0, 1).float() 
                is_same_identity = torch.tensor(False)

            # --- **** 修改：调整返回字典和形状 **** ---
            # fgid 和 frid 需要扩展到 (N, embed_dim)
            # fdid_ref 需要扩展到 (N, C_fdid, H_fdid, W_fdid)
            # is_same_identity 需要扩展到 (N, 1)
            
            # unsqueeze(0) 在第0维增加一个维度, repeat 在第0维重复 num_frames 次
            fgid_source_repeated = fgid_source.unsqueeze(0).repeat(self.num_frames, 1) 
            frid_repeated = frid.unsqueeze(0).repeat(self.num_frames, 1)
            fdid_ref_repeated = fdid_ref.unsqueeze(0).repeat(self.num_frames, 1, 1, 1) # 在第0维增加并重复
            is_same_identity_repeated = is_same_identity.unsqueeze(0).repeat(self.num_frames, 1)

            # 如果所有步骤都成功，跳出重试循环
            break 
            
        # --- 返回结果 ---
        return {
            "vt": vt,                         # (N, C_lat, H_lat, W_lat) 或 (N, 3, H, W)
            "fgid": fgid_source_repeated,     # **** 源 fgid (N, 512) ****
            "frid": frid_repeated,            # **** 参考 frid (N, 512) ****
            "fdid": fdid_ref_repeated,        # **** 参考 fdid (N, C_fdid, H_fdid, W_fdid) ****
            "is_same_identity": is_same_identity_repeated, # (N, 1)
            "v_prime_latent": v_prime_latent  # (1, C_lat, H_lat, W_lat) 或 (C, H, W)
        }
    
# test_dataset.py
if __name__ == '__main__':
     logging.basicConfig(level=logging.DEBUG) # 启用 DEBUG 日志看详细信息
     
     # 初始化 face_recognizer (假设代码在 utils 下)
     from hifivfs_fal.utils.face_recognition import DeepFaceRecognizer
     recognizer = DeepFaceRecognizer()
     
     # 初始化 dataset (需要 VAE 和 encoder fn，或者设置 use_vae_latent=False)
     # 这里用一个假的 encoder_fn 示例
     dummy_encoder = lambda x: torch.randn(x.shape[0], 4, x.shape[2]//8, x.shape[3]//8) 
     
     dataset = FALDataset(
          video_list_file='/root/HiFiVFS/data/vox2_curated_train.txt', # 使用你的调试列表
          face_recognizer=recognizer,
          num_frames=2, # 测试时用少量帧
          use_vae_latent=True,
          vae_encoder_fn=dummy_encoder, 
          image_size_for_vae=(640, 640) # 保持与训练一致
     )
     
     if len(dataset) > 0:
          print(f"数据集大小: {len(dataset)}")
          # 获取第一个样本
          try:
               sample = dataset[0]
               print("\n成功获取样本:")
               for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                         print(f"- {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    else:
                         print(f"- {key}: {value}")
                         
               # 检查关键形状
               assert sample['vt'].shape[0] == 2 # num_frames
               assert sample['fgid'].shape == (2, recognizer.embedding_size)
               assert sample['frid'].shape == (2, recognizer.embedding_size)
               # 检查 fdid 形状 (N, C, H, W)
               C_fdid, H_fdid, W_fdid = dataset.fdid_shape
               assert sample['fdid'].shape == (2, C_fdid, H_fdid, W_fdid) 
               assert sample['is_same_identity'].shape == (2, 1)
               print("\n形状检查通过!")
               
          except Exception as e:
               print(f"\n获取或检查样本时出错: {e}")
               import traceback
               traceback.print_exc()
     else:
          print("数据集为空！")