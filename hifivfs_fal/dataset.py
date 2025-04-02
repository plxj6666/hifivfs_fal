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
from .utils.detect_align_face import detect_align_face # 一个人脸检测和对齐的工具类
# 假设 FaceRecognizer 在这里可以导入
from .utils.face_recognition import DeepFaceRecognizer
from hifivfs_fal.utils.vae_utils import extract_gid_from_latent, decode_with_vae, convert_tensor_to_cv2_images, FridProjector

class FALDataset(Dataset):
    def __init__(self, video_list_file=None, data_root=None, face_recognizer=None, 
                 num_frames=16, target_img_size=(80, 80), 
                 use_vae_latent=False, vae_encoder_fn=None,
                 image_size_for_vae=(640, 640)):
        """
        Args:
            video_list_file (str, optional): 包含视频文件路径列表的文本文件
            data_root (str, optional): 数据集根目录路径（如果不使用video_list_file）
            face_recognizer (FaceRecognizer): 已初始化的FaceRecognizer实例
            num_frames (int): 每个视频片段要加载的连续帧数
            target_img_size (tuple): 目标空间大小(H, W)（不使用VAE时相关）
            use_vae_latent (bool): Vt是否应该是VAE潜在空间特征
            vae_encoder_fn (callable, optional): 编码图像到VAE潜在空间的函数
            image_size_for_vae (tuple): VAE编码前的目标图像大小
        """
        super().__init__()
        
        # 验证输入参数
        if video_list_file is None and data_root is None:
            raise ValueError("必须提供video_list_file或data_root中的至少一个")
        
        if face_recognizer is None:
            raise ValueError("必须提供face_recognizer实例")
            
        self.face_recognizer = face_recognizer
        self.num_frames = num_frames
        self.target_img_size = target_img_size
        self.use_vae_latent = use_vae_latent
        self.vae_encoder_fn = vae_encoder_fn
        self.image_size_for_vae = image_size_for_vae
        
        if self.use_vae_latent and self.vae_encoder_fn is None:
            raise ValueError("use_vae_latent为True时必须提供vae_encoder_fn")

        # --- 初始化视频文件列表 ---
        self.video_files = []
        
        # 1. 从视频列表文件加载 (优先)
        if video_list_file:
            print(f"从文件加载视频列表: {video_list_file}")
            try:
                with open(video_list_file, 'r') as f:
                    for line in f:
                        video_path = line.strip()
                        if os.path.exists(video_path):
                            self.video_files.append(Path(video_path))
                        else:
                            logging.warning(f"视频文件不存在: {video_path}")
            except Exception as e:
                raise RuntimeError(f"无法加载视频列表文件: {e}")
        
        # 2. 如果未从文件加载或列表为空，则扫描目录
        if not self.video_files and data_root:
            self.data_root = Path(data_root)
            print(f"扫描目录获取视频文件: {self.data_root}")
            
            # 原始的目录扫描逻辑
            for identity_dir in self.data_root.iterdir():
                if identity_dir.is_dir():
                    for clip_dir in identity_dir.iterdir():
                        if clip_dir.is_dir():
                            for video_file in clip_dir.glob('*.mkv'):
                                self.video_files.append(video_file)
                            for video_file in clip_dir.glob('*.mp4'):
                                self.video_files.append(video_file)
        
        print(f"找到{len(self.video_files)}个视频文件。")
        if not self.video_files:
            raise FileNotFoundError(f"未找到视频文件。检查路径和数据集结构。")

    def __len__(self):
        return len(self.video_files)

    def _preprocess_face_for_recognition(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """使用统一的人脸对齐方法处理人脸"""
        # 直接使用face_recognizer中的detect_and_align方法，保持与提取特征时相同的对齐流程
        aligned_face = self.face_recognizer.detect_and_align(frame_bgr)
        
        if aligned_face is None:
            return None
        
        # 确保图像在[-1,1]范围，这是许多神经网络的预期输入范围
        if aligned_face.dtype == np.uint8:
            normalized_face = (aligned_face.astype(np.float32) / 127.5) - 1.0
        else:
            # 如果已经是浮点类型，假设已经在0-1范围，转换到-1到1
            normalized_face = (aligned_face * 2.0) - 1.0
            
        return normalized_face

    def _load_random_face_for_frid(self):
        """Loads a random face from a different video for frid."""
        while True:
            random_video_path = random.choice(self.video_files)
            cap = cv2.VideoCapture(str(random_video_path))
            if not cap.isOpened(): continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 1: continue
            random_frame_idx = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                preprocessed_face = self._preprocess_face_for_recognition(frame)
                if preprocessed_face is not None:
                     # Need BGR for embedding model usually
                     frid_embedding = self.face_recognizer.get_embedding(preprocessed_face if self.face_recognizer.model_name != 'buffalo_l' else cv2.cvtColor(preprocessed_face, cv2.COLOR_RGB2BGR) ) # Adjust based on recognizer input needs
                     if frid_embedding is not None:
                         return frid_embedding
            # Retry if failed

    def __getitem__(self, index):
        video_path = self.video_files[index]

        # --- Load Video Frames ---
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            # Return dummy data or try next index? Best to raise error or handle properly.
            # For simplicity, retry with another random video
            return self.__getitem__(random.randint(0, len(self)-1))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            cap.release()
            logging.warning(f"Video {video_path} has only {total_frames} frames, less than required {self.num_frames}. Skipping.")
            return self.__getitem__(random.randint(0, len(self)-1)) # Retry

        start_frame = random.randint(0, total_frames - self.num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        vt_frames_bgr = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                logging.error(f"Failed to read frame from {video_path} during sequence loading.")
                return self.__getitem__(random.randint(0, len(self)-1)) # Retry
            vt_frames_bgr.append(frame)
        
        # --- 额外加载一个参考帧V' (用于L_attr计算) ---
        # 设置帧指针到随机位置
        reference_frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_idx)
        ret, v_prime_frame = cap.read()
        cap.release()
        
        if not ret:
            logging.error(f"Failed to read reference frame from {video_path}.")
            return self.__getitem__(random.randint(0, len(self)-1)) # Retry
        
        # --- 处理参考帧V' ---
        # 使用与Vt相同的处理流程
        if self.use_vae_latent:
            if self.vae_encoder_fn:
                # 调整大小为VAE所需尺寸
                v_prime_resized = cv2.resize(v_prime_frame, (self.image_size_for_vae[1], self.image_size_for_vae[0]))
                # BGR转RGB (VAE通常期望RGB输入)
                v_prime_rgb = cv2.cvtColor(v_prime_resized, cv2.COLOR_BGR2RGB)
                # 归一化到[-1, 1]
                v_prime_normalized = (v_prime_rgb.astype(np.float32) / 127.5) - 1.0
                # HWC to CHW (PyTorch格式)
                v_prime_chw = np.transpose(v_prime_normalized, (2, 0, 1))
                v_prime_tensor = torch.from_numpy(v_prime_chw).unsqueeze(0)  # 添加批次维度为单个图像
                
                # 使用VAE编码
                v_prime_latent = self.vae_encoder_fn(v_prime_tensor)
            else:
                raise RuntimeError("use_vae_latent is True, but vae_encoder_fn is not provided.")
        else:
            # 像素空间处理
            v_prime_resized = cv2.resize(v_prime_frame, (self.target_img_size[1], self.target_img_size[0]))
            v_prime_normalized = (v_prime_resized.astype(np.float32) / 127.5) - 1.0
            v_prime_latent = torch.from_numpy(v_prime_normalized).permute(2, 0, 1)  # HWC to CHW

        # --- Preprocess Vt ---
        if self.use_vae_latent:
            if self.vae_encoder_fn:
                # 处理帧以准备VAE编码
                vae_ready_frames = []
                for frame in vt_frames_bgr:
                    # 调整大小为VAE所需尺寸
                    frame_resized = cv2.resize(frame, (self.image_size_for_vae[1], self.image_size_for_vae[0]))
                    # BGR转RGB (VAE通常期望RGB输入)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    # 归一化到[-1, 1]
                    frame_normalized = (frame_rgb.astype(np.float32) / 127.5) - 1.0
                    # HWC to CHW (PyTorch格式)
                    frame_chw = np.transpose(frame_normalized, (2, 0, 1))
                    vae_ready_frames.append(frame_chw)
                
                # 转换为批量张量
                frames_tensor = torch.from_numpy(np.stack(vae_ready_frames))
                
                # 使用传入的VAE编码函数
                vt = self.vae_encoder_fn(frames_tensor)
            else:
                raise RuntimeError("use_vae_latent is True, but vae_encoder_fn is not provided.")
        else:
            # Preprocess for pixel space (resize, normalize, permute channels)
            processed_frames = []
            for frame in vt_frames_bgr:
                frame_resized = cv2.resize(frame, (self.target_img_size[1], self.target_img_size[0])) # W, H format for resize
                frame_normalized = (frame_resized.astype(np.float32) / 127.5) - 1.0 # To [-1, 1]
                frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1) # HWC to CHW
                processed_frames.append(frame_tensor)
            vt = torch.stack(processed_frames) # Shape (N, 3, H, W) - Treat N as Batch for now? Or keep N? Need to decide.


        # --- Get fgid (from a random frame in the clip) ---
        random_frame_for_fgid = random.choice(vt_frames_bgr)
        preprocessed_face_fgid = self._preprocess_face_for_recognition(random_frame_for_fgid)
        if preprocessed_face_fgid is None:
            logging.warning(f"Could not preprocess face for fgid from {video_path}. Retrying.")
            return self.__getitem__(random.randint(0, len(self)-1)) # Retry

        # Insightface app.get expects BGR uint8 usually
        fgid = self.face_recognizer.get_embedding( ( (preprocessed_face_fgid + 1.0) * 127.5 ).astype(np.uint8) ) # Convert back for insightface app
        if fgid is None:
            logging.warning(f"Could not extract fgid from {video_path}. Retrying.")
            return self.__getitem__(random.randint(0, len(self)-1)) # Retry
        fgid = torch.from_numpy(fgid).float()


        # --- Choose frid and is_same_identity ---
        if random.random() < 0.5:
            # Use same identity
            frid = fgid.clone()
            is_same_identity = torch.tensor(True)
        else:
            # Use different identity (load from another random video)
            frid_numpy = self._load_random_face_for_frid()
            if frid_numpy is None:
                 logging.warning(f"Could not load random face for frid. Retrying.")
                 return self.__getitem__(random.randint(0, len(self)-1)) # Retry
            frid = torch.from_numpy(frid_numpy).float()
            is_same_identity = torch.tensor(False)


        # --- Prepare f_rid for Decoder (Placeholder - needs decision) ---
        # Option 1: Repeat fgid/frid (B, EmbedDim) -> (B, SeqLen, EmbedDim)
        # Option 2: Project fgid/frid to (B, 1, kv_channels)
        # For now, just return the raw 512-dim vector. Handling will be in training loop or model.
        # --- Decide on Vt format ---
        # If Vt is (N, C, H, W), how to handle batching?
        # Option A: Return the sequence, collate_fn handles stacking into (B, N, C, H, W)
        # Option B: Treat N as batch dim for now? (B=N, C, H, W). Simpler start. Let's use B.
        if not self.use_vae_latent:
             # If pixel space, vt is (N, 3, H, W). Assume N is batch for simplicity now.
             pass # vt already (N, 3, H, W)
        else:
             # If latent space, vt is (N, 4, H_lat, W_lat). Assume N is batch.
             pass # vt already (N, 4, H_lat, W_lat)

        # Ensure fgid and frid match batch size N
        fgid = fgid.unsqueeze(0).repeat(self.num_frames, 1) # (N, 512)
        frid = frid.unsqueeze(0).repeat(self.num_frames, 1) # (N, 512)
        is_same_identity = is_same_identity.unsqueeze(0).repeat(self.num_frames, 1) # (N, 1)


        return {
            "vt": vt,               # Shape (N, C, H, W) - N treated as Batch
            "fgid": fgid,           # Shape (N, 512)
            "frid": frid,           # Shape (N, 512)
            "is_same_identity": is_same_identity, # Shape (N, 1)
            "v_prime_latent": v_prime_latent     # Shape (1, C, H, W) for VAE or (C, H, W) for pixel space
        }

