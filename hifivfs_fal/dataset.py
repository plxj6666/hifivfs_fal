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
from .utils.face_recognition import FaceRecognizer
from hifivfs_fal.utils.vae_utils import extract_gid_from_latent, project_and_prepare_frid, decode_with_vae, convert_tensor_to_cv2_images

# 假设你有一个用于人脸检测和对齐的工具函数或类
# from .utils.face_utils import detect_align_face # Placeholder

# 假设你有一个 VAE 模型 (如果使用隐空间)
# from some_vae_library import VAEEncoder, VAEDecoder # Placeholder

class FALDataset(Dataset):
    def __init__(self, data_root: str, face_recognizer: FaceRecognizer, 
                 num_frames: int = 16, target_img_size: tuple = (80, 80), 
                 use_vae_latent: bool = False, vae_encoder_fn = None,
                 image_size_for_vae: tuple = (512, 512)):
        """
        Args:
            data_root (str): Path to the root directory of the dataset (e.g., VoxCeleb2).
            face_recognizer (FaceRecognizer): Initialized FaceRecognizer instance.
            num_frames (int): Number of consecutive frames to load for Vt.
            target_img_size (tuple): Target spatial size (H, W) for Vt (relevant if not using VAE).
            use_vae_latent (bool): Whether Vt should be VAE latent space features.
            vae_encoder_fn (callable, optional): Function to encode images to VAE latent space.
                                               Required if use_vae_latent is True.
            image_size_for_vae (tuple): Target size for images before VAE encoding.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.face_recognizer = face_recognizer
        self.num_frames = num_frames
        self.target_img_size = target_img_size
        self.use_vae_latent = use_vae_latent
        self.vae_encoder_fn = vae_encoder_fn
        self.image_size_for_vae = image_size_for_vae
        
        if self.use_vae_latent and self.vae_encoder_fn is None:
            raise ValueError("vae_encoder_fn must be provided when use_vae_latent is True")

        # --- Scan for video files ---
        self.video_files = []
        print(f"Scanning for video files in {self.data_root}...")
        # Example for VoxCeleb2 structure (id/clip_id/video.mkv or .mp4)
        for identity_dir in self.data_root.iterdir():
            if identity_dir.is_dir():
                for clip_dir in identity_dir.iterdir():
                    if clip_dir.is_dir():
                        for video_file in clip_dir.glob('*.mkv'):
                            self.video_files.append(video_file)
                        for video_file in clip_dir.glob('*.mp4'):
                            self.video_files.append(video_file)
        print(f"Found {len(self.video_files)} video files.")
        if not self.video_files:
            raise FileNotFoundError(f"No video files found in {data_root}. Check the path and dataset structure.")

        # --- Optional: Pre-extracted face pool for faster random frid sampling ---
        # self.random_face_pool = self._build_face_pool() # Could be time consuming


    def __len__(self):
        return len(self.video_files)

    def _preprocess_face_for_recognition(self, frame_bgr: np.ndarray) -> np.ndarray | None:
    # """使用人脸检测和对齐工具处理人脸"""
    # 使用112x112大小，因为大多数人脸识别模型都使用这个尺寸
        aligned_face = detect_align_face(frame_bgr, target_size=(112, 112), mode='arcface')
        if aligned_face is None:
            return None
        
        # 归一化到[-1,1]范围
        normalized_face = (aligned_face.astype(np.float32) / 127.5) - 1.0
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
        cap.release()
        # List of N frames, each (H, W, 3) BGR uint8

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
            "is_same_identity": is_same_identity # Shape (N, 1)
        }


# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing FALDataset ---")
    # !!! IMPORTANT: Replace with the actual path to your downloaded VoxCeleb2 dataset !!!
    # Example: '/path/to/datasets/voxceleb2/dev/aac'
    DATASET_ROOT = "/path/to/your/voxceleb2_or_similar_dataset" # CHANGE THIS

    if not os.path.exists(DATASET_ROOT) or DATASET_ROOT == "/path/to/your/voxceleb2_or_similar_dataset":
         print(f"ERROR: Dataset root '{DATASET_ROOT}' not found or not configured.")
         print("Please download VoxCeleb2 (or similar) and update DATASET_ROOT in dataset.py")
    else:
        try:
            print("Initializing FaceRecognizer...")
            # Use CPU explicitly if testing locally without GPU maybe faster init
            # face_recognizer = FaceRecognizer(providers=['CPUExecutionProvider'])
            face_recognizer = FaceRecognizer() # Use default (try CUDA)
            print("FaceRecognizer initialized.")

            print("Initializing FALDataset (using pixel space for now)...")
            # Set use_vae_latent=False if you don't have a VAE ready
            # Set target_img_size appropriately if not using VAE
            dataset = FALDataset(data_root=DATASET_ROOT,
                                 face_recognizer=face_recognizer,
                                 num_frames=8, # Use smaller N for faster testing
                                 target_img_size=(128, 128), # Example size for pixel space
                                 use_vae_latent=False) # Set True if you have VAE and adapt code
            print(f"Dataset size: {len(dataset)}")

            if len(dataset) > 0:
                print("\nGetting a sample from the dataset...")
                # Get one sample
                sample = dataset[0] # Get the first sample (index 0)

                print("\n--- Sample Content ---")
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        print(f"'{key}': Tensor shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"'{key}': {value}")

                # Basic checks
                assert 'vt' in sample
                assert 'fgid' in sample
                assert 'frid' in sample
                assert 'is_same_identity' in sample
                assert sample['vt'].shape[0] == 8 # Matches num_frames
                assert sample['fgid'].shape == (8, 512)
                assert sample['frid'].shape == (8, 512)
                assert sample['is_same_identity'].shape == (8, 1)

                print("\n--- DataLoader Test ---")
                # Test with DataLoader
                dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0) # num_workers=0 for easier debugging
                print(f"DataLoader created with batch_size=2")

                try:
                    batch = next(iter(dataloader))
                    print("\n--- Batch Content ---")
                    for key, value in batch.items():
                         if isinstance(value, torch.Tensor):
                            # Note: DataLoader stacks the first dimension, so shape becomes (B, N, ...)
                            print(f"'{key}': Tensor shape={value.shape}, dtype={value.dtype}")
                         else:
                            print(f"'{key}': {value}")
                     # Expected shapes:
                     # vt: (B, N, C, H, W)
                     # fgid: (B, N, 512)
                     # frid: (B, N, 512)
                     # is_same_identity: (B, N, 1)
                    print("Successfully retrieved one batch.")
                except Exception as e:
                     print(f"ERROR: Failed to get batch from DataLoader: {e}")
                     print("Check dataset implementation and data integrity.")


            else:
                print("Dataset is empty. Check video scanning logic and data path.")

        except Exception as e:
            print(f"\nAn error occurred during dataset testing: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Dataset Test Finished ---")