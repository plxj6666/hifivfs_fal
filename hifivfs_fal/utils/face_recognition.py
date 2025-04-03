# face_recognition.py

import numpy as np
import cv2
import logging
import os
from typing import Optional, List, Tuple, Union, Dict
import torch # 导入 torch 以使用 @torch.no_grad()

# 外部依赖 (确保已安装 deepface 及其依赖)
try:
    from deepface.modules import modeling, preprocessing
    from deepface.commons import package_utils, folder_utils
    from deepface.commons.logger import Logger
    # 使用DeepFace自己的Logger
    logger = Logger() 
except ImportError:
    raise ImportError("无法导入DeepFace模块。请运行 'pip install deepface' 并确保其依赖项已安装。")

# 内部依赖 (来自你的项目)
try:
    # 假设你的MediaPipe对齐工具在这个路径
    from hifivfs_fal.utils.detect_align_face import detect_align_face 
except ImportError:
    raise ImportError("无法导入 hifivfs_fal.utils.detect_align_face。请确保路径正确。")

# --- Helper Functions ---

def _get_embedding_size(model_name: str) -> int:
    """根据模型名称获取特征向量维度"""
    model_params = {
        "VGG-Face": 4096, 
        "Facenet": 128,
        "Facenet512": 512,
        "OpenFace": 128,
        "DeepFace": 4096,
        "DeepID": 160, 
        "ArcFace": 512,
        "Dlib": 128, 
        "SFace": 128,
        "GhostFaceNet": 512,
    }
    size = model_params.get(model_name)
    if size is None:
        logger.warn(f"模型 '{model_name}' 的Embedding size未知，将使用默认值 512。")
        return 512
    return size

def _get_default_normalization(model_name: str) -> str:
    """根据模型名称获取推荐的归一化方法"""
    normalization_map = {
        "VGG-Face": "VGGFace",
        "Facenet": "Facenet",
        "Facenet512": "Facenet", 
        "OpenFace": "raw", 
        "DeepFace": "base",
        "DeepID": "base", 
        "ArcFace": "ArcFace",
        "Dlib": "raw", 
        "SFace": "SFace",
        "GhostFaceNet": "GhostFaceNet",
    }
    norm = normalization_map.get(model_name, "base")
    logger.info(f"模型 '{model_name}' 将使用 '{norm}' 归一化方法。")
    return norm

# --- Main Class ---

class DeepFaceRecognizer:
    """
    使用DeepFace进行人脸识别特征提取，并强制使用外部MediaPipe进行对齐。
    优化了特征提取效率，避免了临时文件。
    """

    def __init__(self, 
                 model_name: str = "Facenet512", 
                 detector_backend: str = "retinaface", 
                 normalization: Optional[str] = None):
        self.model_name = model_name
        self.detector_backend = detector_backend 
        self.normalization = normalization if normalization else _get_default_normalization(model_name)
        self.embedding_size = _get_embedding_size(model_name)
        
        self.feature_extractor = None
        self.target_size = None # (height, width)
        self.initialized = False

        self._initialize_model()

    def _initialize_model(self):
        try:
            logger.info(f"正在初始化DeepFace特征提取器: {self.model_name}")
            self.feature_extractor = modeling.build_model(
                task="facial_recognition", 
                model_name=self.model_name
            ) 
            
            # 尝试获取输入尺寸 (可能不完整，只包含 H, W)
            raw_shape = self.feature_extractor.input_shape
            determined_size = None
            if isinstance(raw_shape, tuple) and len(raw_shape) >= 2:
                 if isinstance(raw_shape[0], int) and isinstance(raw_shape[1], int) and raw_shape[0] > 0 and raw_shape[1] > 0:
                      determined_size = (raw_shape[0], raw_shape[1]) # (height, width)
                 elif len(raw_shape) >= 3 and isinstance(raw_shape[1], int) and isinstance(raw_shape[2], int) and raw_shape[1] > 0 and raw_shape[2] > 0:
                      determined_size = (raw_shape[1], raw_shape[2]) # (height, width)
                 # ... (可以添加更多检查，如从config获取) ...

            if determined_size:
                 self.target_size = determined_size
            else:
                 # 如果无法确定，使用常见默认值并警告
                 default_size = (160, 160) if "Facenet" in self.model_name else (112, 112)
                 logger.warn(f"无法自动确定模型 '{self.model_name}' 的输入尺寸 (从input_shape获取为{raw_shape})，将使用默认值: {default_size}")
                 self.target_size = default_size
            
            logger.info(f"模型 '{self.model_name}' 加载成功。预期输入尺寸 (H, W): {self.target_size}")
            self.initialized = True

        except Exception as e:
            logger.error(f"初始化DeepFace模型 '{self.model_name}' 失败: {e}", exc_info=True)
            self.initialized = False

    def detect_and_align(self, image_bgr: np.ndarray, target_size=(112, 112)) -> Optional[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0:
            logger.warn("detect_and_align 收到空图像。")
            return None
        try:
            if image_bgr.dtype != np.uint8:
                 # ... (类型转换逻辑，同上一版本) ...
                 if np.issubdtype(image_bgr.dtype, np.floating):
                      if image_bgr.min() >= 0 and image_bgr.max() <= 1.0:
                           image_bgr = (image_bgr * 255.0).astype(np.uint8)
                      elif image_bgr.min() >= -1.0 and image_bgr.max() <= 1.0:
                           image_bgr = ((image_bgr + 1.0) / 2.0 * 255.0).astype(np.uint8)
                      else: 
                           image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
                 else:
                      logger.error(f"无法处理类型为 {image_bgr.dtype} 的输入图像。")
                      return None

            logger.debug(f"使用MediaPipe进行人脸对齐，目标尺寸: {target_size}")
            aligned_face = detect_align_face(image_bgr, target_size=target_size) 
            if aligned_face is not None:
                if aligned_face.dtype != np.uint8:
                     logger.warn(f"MediaPipe对齐返回类型 {aligned_face.dtype}，强制转换为 uint8。")
                     aligned_face = np.clip(aligned_face, 0, 255).astype(np.uint8)
                logger.debug("MediaPipe人脸对齐成功。")
                return aligned_face
            else:
                logger.debug("MediaPipe未检测到或对齐人脸失败。")
                return None
        except Exception as e:
            logger.error(f"调用MediaPipe人脸检测与对齐时出错: {e}", exc_info=True)
            return None

    # 使用 torch.no_grad() 明确告知不需要计算梯度，可以节省显存
    # 注意: 这只在底层模型是 PyTorch 模型时有效。如果模型是 TensorFlow/Keras，
    # 它不会产生影响，但也不会报错。
    @torch.no_grad() 
    def get_embedding(self, aligned_face_bgr_uint8: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.feature_extractor is None:
            logger.error("特征提取器未初始化。")
            return None
        if aligned_face_bgr_uint8 is None or aligned_face_bgr_uint8.size == 0:
            logger.warn("get_embedding 收到空图像。")
            return None
        if aligned_face_bgr_uint8.dtype != np.uint8:
             logger.error(f"输入 get_embedding 的图像必须是 BGR uint8，但收到 {aligned_face_bgr_uint8.dtype}。")
             return None

        try:
            # 1. 调整大小到模型所需尺寸 (H, W)
            img_resized = preprocessing.resize_image(
                img=aligned_face_bgr_uint8,
                target_size=(self.target_size[1], self.target_size[0]) # 传入 (W, H)
            )
            logger.debug(f"图像已调整大小至模型输入尺寸: {img_resized.shape}") # (H, W, C)

            # 2. 归一化输入
            img_normalized = preprocessing.normalize_input(
                img=img_resized, 
                normalization=self.normalization
            )
            # **预期 normalize_input 返回 (1, H, W, C)**
            logger.debug(f"图像已归一化，方法: {self.normalization}, 形状: {img_normalized.shape}") 

            # 3. **关键修改**: 检查维度并准备模型输入
            #    现在期望 normalize_input 返回 4D 数组 (1, H, W, C)
            if not isinstance(img_normalized, np.ndarray) or img_normalized.ndim != 4:
                logger.error(f"归一化步骤未产生预期的4D NumPy数组 (1, H, W, C)。实际形状: {img_normalized.shape}, 类型: {type(img_normalized)}")
                # 尝试修复：如果它是 3D (H, W, C)，则添加批次维度
                if isinstance(img_normalized, np.ndarray) and img_normalized.ndim == 3:
                     logger.warn("归一化返回了 3D 数组，将手动添加批次维度。")
                     img_expanded = np.expand_dims(img_normalized, axis=0)
                else:
                     return None # 无法处理其他情况
            else:
                 # 如果已经是 4D，直接使用
                 img_expanded = img_normalized 
            
            logger.debug(f"为模型准备的输入形状: {img_expanded.shape}") # 预期 (1, H, W, C)

            # 4. 模型前向传播
            embedding_raw = self.feature_extractor.forward(img_expanded)
                 
            if isinstance(embedding_raw, list): 
                embedding_raw = np.array(embedding_raw)
                     
            embedding = embedding_raw.flatten() 
                 
            # 5. L2 归一化
            norm = np.linalg.norm(embedding)
            if norm == 0:
                logger.warn("提取的特征向量范数为零。")
                return None
            embedding_normalized = embedding / norm
            
            logger.debug(f"成功提取并归一化特征向量，形状: {embedding_normalized.shape}")
            return embedding_normalized.astype(np.float32) 

        except Exception as e:
            logger.error(f"直接特征提取失败。输入对齐图像形状: {aligned_face_bgr_uint8.shape}, 调整后形状: {img_resized.shape if 'img_resized' in locals() else 'N/A'}, 归一化后形状: {img_normalized.shape if 'img_normalized' in locals() else 'N/A'}. 错误: {e}", exc_info=True)
            return None

    def extract_identity(self, image_bgr: np.ndarray, align_target_size=(112, 112)) -> Optional[np.ndarray]:
        if not self.initialized:
            logger.error("DeepFaceRecognizer 未初始化。")
            return None
        logger.debug("开始身份提取流程...")
        aligned_face = self.detect_and_align(image_bgr, target_size=align_target_size)
        if aligned_face is None:
            logger.warn("MediaPipe人脸对齐失败，无法提取身份特征。")
            return None
        embedding = self.get_embedding(aligned_face)
        if embedding is None:
            logger.warn("从对齐后的人脸提取特征失败。")
            return None
        logger.info("身份特征提取成功。")
        return embedding

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        if embedding1 is None or embedding2 is None:
            logger.warn("计算相似度时输入向量为None。")
            return -2.0
        if embedding1.ndim > 1: embedding1 = embedding1.flatten()
        if embedding2.ndim > 1: embedding2 = embedding2.flatten()
        if embedding1.shape != embedding2.shape:
             logger.error(f"无法计算相似度：向量形状不匹配 {embedding1.shape} vs {embedding2.shape}")
             return -2.0
        try:
            similarity = np.dot(embedding1, embedding2)
            similarity = np.clip(similarity, -1.0, 1.0)
            return float(similarity)
        except Exception as e:
            logger.error(f"计算相似度时出错: {e}", exc_info=True)
            return -2.0

# --- 示例用法 (保持不变) ---
if __name__ == '__main__':
    # 配置日志级别
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ... (加载图像和测试逻辑同上一版本) ...
    # --- 测试初始化 ---
    try:
        recognizer = DeepFaceRecognizer(model_name="Facenet512") 
        if not recognizer.initialized:
            logger.error("识别器初始化失败，退出测试。")
            exit()
    except Exception as e:
        logger.error(f"初始化过程中发生错误: {e}")
        exit()

    # --- 加载示例图像 ---
    image_path1 = '/root/HiFiVFS/data/mayun1.jpg' # 替换为你自己的路径
    image_path2 = '/root/HiFiVFS/data/mayun1.jpg' # 替换为你自己的路径
    image_path3 = '/root/HiFiVFS/samples/vox2_640/sample_epoch951_step9500.png' # 替换为你自己的路径

    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    img3 = cv2.imread(image_path3)

    if img1 is None or img2 is None or img3 is None:
        logger.error("未能加载所有测试图像，请检查路径。")
        logger.info("使用黑色图像进行流程测试...")
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        img3 = np.zeros((256, 256, 3), dtype=np.uint8)

    # --- 测试身份提取 ---
    logger.info(f"\n--- 提取图像1 ({os.path.basename(image_path1)}) 的身份特征 ---")
    embedding1 = recognizer.extract_identity(img1)
    if embedding1 is not None: logger.info(f"图像1 特征提取成功，维度: {embedding1.shape}")
    else: logger.error("图像1 特征提取失败。")

    logger.info(f"\n--- 提取图像2 ({os.path.basename(image_path2)}) 的身份特征 ---")
    embedding2 = recognizer.extract_identity(img2)
    if embedding2 is not None: logger.info(f"图像2 特征提取成功，维度: {embedding2.shape}")
    else: logger.error("图像2 特征提取失败。")
    
    logger.info(f"\n--- 提取图像3 ({os.path.basename(image_path3)}) 的身份特征 ---")
    embedding3 = recognizer.extract_identity(img3)
    if embedding3 is not None: logger.info(f"图像3 特征提取成功，维度: {embedding3.shape}")
    else: logger.error("图像3 特征提取失败。")

    # --- 测试相似度计算 ---
    if embedding1 is not None and embedding2 is not None:
        similarity_1_2 = recognizer.compute_similarity(embedding1, embedding2)
        logger.info(f"\n--- 图像1 vs 图像2 相似度 ---")
        logger.info(f"相似度: {similarity_1_2:.4f}")

    if embedding1 is not None and embedding3 is not None:
        similarity_1_3 = recognizer.compute_similarity(embedding1, embedding3)
        logger.info(f"\n--- 图像1 vs 图像3 相似度 ---")
        logger.info(f"相似度: {similarity_1_3:.4f}")

    # --- 测试直接对对齐人脸提取特征 ---
    logger.info("\n--- 测试对已对齐人脸提取特征 ---")
    aligned_face3 = recognizer.detect_and_align(img3)
    if aligned_face3 is not None:
         logger.info("图像1 对齐成功，尝试直接提取特征...")
         embedding3_direct = recognizer.get_embedding(aligned_face3)
         if embedding3_direct is not None:
              logger.info(f"直接从对齐人脸提取特征成功，维度: {embedding3_direct.shape}")
              if embedding1 is not None:
                   diff = np.linalg.norm(embedding1 - embedding3_direct)
                   sim_check = recognizer.compute_similarity(embedding3, embedding3_direct)
                   logger.info(f" extract_identity vs get_embedding(aligned) 差异 (L2范数): {diff:.6f}, 相似度: {sim_check:.6f}")
         else: logger.error("直接从对齐人脸提取特征失败。")
    else: logger.warning("图像1 对齐失败，跳过直接提取测试。")