# face_recognition.py

import numpy as np
import cv2
import logging # 使用标准库 logging
import os
from typing import Optional, List, Tuple, Union, Dict
import torch 
import traceback # 导入 traceback

# **** 配置 logging ****
# 获取当前模块的 logger
# 注意：如果在 __main__ 中已经配置了 basicConfig，这里获取的 logger 会继承那个配置
# 如果没有，需要在这里配置，或者在调用此模块前配置
logger = logging.getLogger(__name__) 
# 如果直接运行此文件，需要设置基本配置
# if __name__ == "__main__":
#      logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# 外部依赖
try:
    import tensorflow as tf
    from deepface.modules import modeling, preprocessing
    from deepface.commons import package_utils, folder_utils
    # 不再使用 DeepFace 的 Logger
except ImportError:
    raise ImportError("需要安装 TensorFlow 和 DeepFace。请运行 'pip install tensorflow deepface'")

# 内部依赖
try:
    from hifivfs_fal.utils.detect_align_face import detect_align_face 
except ImportError:
    # 尝试相对导入，以防在 utils 目录内运行
    try:
         from .detect_align_face import detect_align_face
    except ImportError:
         raise ImportError("无法导入 hifivfs_fal.utils.detect_align_face 或 detect_align_face。请确保路径正确。")

# --- Helper Functions ---
def _get_embedding_size(model_name: str) -> int:
    """根据模型名称获取特征向量维度"""
    model_params = {"VGG-Face": 4096, "Facenet": 128, "Facenet512": 512, "OpenFace": 128, "DeepFace": 4096, "DeepID": 160, "ArcFace": 512, "Dlib": 128, "SFace": 128, "GhostFaceNet": 512,}
    size = model_params.get(model_name)
    if size is None: logging.warning(f"模型 '{model_name}' 的Embedding size未知，将使用默认值 512。"); return 512
    return size

def _get_default_normalization(model_name: str) -> str:
    """根据模型名称获取推荐的归一化方法"""
    normalization_map = {"VGG-Face": "VGGFace", "Facenet": "Facenet", "Facenet512": "Facenet", "OpenFace": "raw", "DeepFace": "base", "DeepID": "base", "ArcFace": "ArcFace", "Dlib": "raw", "SFace": "SFace", "GhostFaceNet": "GhostFaceNet",}
    norm = normalization_map.get(model_name, "base")
    logging.info(f"模型 '{model_name}' 将使用 '{norm}' 归一化方法。") # 可以保留 info 级别
    return norm

# --- Main Class ---
class DeepFaceRecognizer:
    """
    使用DeepFace进行人脸识别特征提取，并强制使用外部MediaPipe进行对齐。
    增加了提取详细中间层特征 (fdid) 的功能。
    """
    def __init__(self, 
                 model_name: str = "Facenet512", 
                 detector_backend: str = "retinaface", 
                 normalization: Optional[str] = None,
                 # **** 注意：fdid_layer_name 现在在 _initialize_model 中自动确定 ****
                 # fdid_layer_name: Optional[str] = None): # 不再需要作为参数传入
                 ):
        self.model_name = model_name
        self.detector_backend = detector_backend 
        self.normalization = normalization if normalization else _get_default_normalization(model_name)
        self.embedding_size = _get_embedding_size(model_name)
        
        self.feature_extractor = None # 用于提取 fgid
        self.fdid_extractor = None    # **** 用于提取 fdid ****
        self.target_size = None 
        self.initialized = False
        self.fdid_layer_name = None # **** 将在初始化时确定 ****

        # 获取 logger 实例
        self.logger = logging.getLogger(self.__class__.__name__) 

        self._initialize_model()

    def _initialize_model(self):
        try:
            self.logger.info(f"正在初始化DeepFace特征提取器: {self.model_name}")
            # 1. 加载主模型 (用于 fgid)
            self.feature_extractor = modeling.build_model(task="facial_recognition", model_name=self.model_name) 
            
            if not hasattr(self.feature_extractor, 'model') or not isinstance(self.feature_extractor.model, tf.keras.Model):
                 self.logger.error(f"DeepFace feature_extractor 不含 Keras 模型。类型: {type(self.feature_extractor)}")
                 self.initialized = True; self.logger.warning("只能提供 fgid 功能。"); return 

            keras_model = self.feature_extractor.model
            self.logger.info(f"成功获取底层 Keras 模型: {type(keras_model)}")
            
            # 打印摘要以供参考（如果需要调试可以取消注释）
            # print("\n" + "=" * 30 + "\nKeras Model Summary:\n" + "=" * 30)
            # try:
            #      summary_lines = []; keras_model.summary(print_fn=lambda x: summary_lines.append(x))
            #      print("\n".join(summary_lines))
            # except Exception as e_summary: self.logger.error(f"打印模型摘要时出错: {e_summary}")
            # print("=" * 30 + "\nEnd of Summary\n" + "=" * 30 + "\n")

            # 2. 确定输入尺寸 (逻辑不变)
            raw_shape = keras_model.input_shape 
            determined_size = None
            if isinstance(raw_shape, tuple) and len(raw_shape) >= 3: 
                 if isinstance(raw_shape[1], int) and isinstance(raw_shape[2], int): determined_size = (raw_shape[1], raw_shape[2]) 
                 elif len(raw_shape) >= 4 and isinstance(raw_shape[2], int) and isinstance(raw_shape[3], int): determined_size = (raw_shape[2], raw_shape[3]) 
            if determined_size: self.target_size = determined_size
            else: default_size = (160, 160) if "Facenet" in self.model_name else (112, 112); self.logger.warning(f"无法确定输入尺寸 (shape: {raw_shape})，使用默认值: {default_size}"); self.target_size = default_size
            self.logger.info(f"模型 '{self.model_name}' 加载成功。预期输入尺寸 (H, W): {self.target_size}")

            # 3. **** 创建 fdid 提取器 ****
            target_layer = None
            # **** 更新 potential_layer_names - 基于你的模型摘要 ****
            potential_layer_names = [
                 'add_20',              # 最后一个 Add 层 (Block8_6 输出) - 优先尝试
                 'Block8_6_Activation', # 最后一个激活层 - 备选
                 'Mixed_7a',           # Block8 之前的混合层 - 备选
                 'Block17_10_Activation' # Block17 最后一个块的输出 - 备选 (8x8)
            ]
            self.logger.info(f"尝试自动查找 fdid 层，候选: {potential_layer_names}")
            for name in potential_layer_names:
                 try:
                      target_layer = keras_model.get_layer(name=name) # 使用 name= 关键字参数
                      self.fdid_layer_name = name 
                      self.logger.info(f"自动找到可能的 fdid 层: '{name}'")
                      break
                 except ValueError: 
                      self.logger.debug(f"层 '{name}' 未找到，尝试下一个...")
                      continue 

            if target_layer:
                try:
                    self.fdid_extractor = tf.keras.Model(inputs=keras_model.input, outputs=target_layer.output, name=f"{self.model_name}_fdid_extractor")
                    # 预热 fdid_extractor (可选但推荐)
                    dummy_input = np.zeros((1, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
                    _ = self.fdid_extractor(dummy_input, training=False) # 使用 training=False
                    self.logger.info(f"成功创建 fdid 提取器，输出层: '{self.fdid_layer_name}', 输出形状: {target_layer.output_shape}")
                    self.initialized = True # 标记完全初始化成功
                except Exception as e_fdid:
                    self.logger.error(f"创建 fdid 提取器失败: {e_fdid}", exc_info=True); self.fdid_extractor = None; self.logger.warning("只能提供 fgid 功能。"); self.initialized = True 
            else:
                self.logger.error(f"未能找到合适的中间层来提取 fdid。请检查模型结构或更新 potential_layer_names。")
                self.logger.warning("只能提供 fgid 功能。")
                self.initialized = True # 主模型仍然可用

        except Exception as e:
            self.logger.error(f"初始化DeepFace模型 '{self.model_name}' 失败: {e}", exc_info=True)
            self.initialized = False

    # detect_and_align 方法使用标准 logging
    def detect_and_align(self, image_bgr: np.ndarray, target_size=(112, 112)) -> Optional[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0: self.logger.warning("detect_and_align 收到空图像。"); return None
        try:
            if image_bgr.dtype != np.uint8:
                 if np.issubdtype(image_bgr.dtype, np.floating):
                      if image_bgr.min() >= 0 and image_bgr.max() <= 1.0: image_bgr = (image_bgr * 255.0).astype(np.uint8)
                      elif image_bgr.min() >= -1.0 and image_bgr.max() <= 1.0: image_bgr = ((image_bgr + 1.0) / 2.0 * 255.0).astype(np.uint8)
                      else: image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
                 else: self.logger.error(f"无法处理类型为 {image_bgr.dtype} 的输入图像。"); return None
            self.logger.debug(f"使用MediaPipe进行人脸对齐，目标尺寸: {target_size}")
            aligned_face = detect_align_face(image_bgr, target_size=target_size) 
            if aligned_face is not None:
                if aligned_face.dtype != np.uint8: aligned_face = np.clip(aligned_face, 0, 255).astype(np.uint8)
                self.logger.debug("MediaPipe人脸对齐成功。"); return aligned_face
            else: self.logger.debug("MediaPipe对齐失败。"); return None
        except Exception as e: self.logger.error(f"调用MediaPipe对齐时出错: {e}", exc_info=True); return None

    # get_embedding (fgid) 方法使用标准 logging
    @torch.no_grad() 
    def get_embedding(self, aligned_face_bgr_uint8: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.feature_extractor is None: self.logger.error("特征提取器未初始化。"); return None
        if aligned_face_bgr_uint8 is None or aligned_face_bgr_uint8.size == 0: self.logger.warning("get_embedding 收到空图像。"); return None
        if aligned_face_bgr_uint8.dtype != np.uint8: self.logger.error(f"输入 get_embedding 必须是 BGR uint8"); return None
        try:
            img_resized = preprocessing.resize_image(img=aligned_face_bgr_uint8, target_size=(self.target_size[1], self.target_size[0]))
            img_normalized = preprocessing.normalize_input(img=img_resized, normalization=self.normalization)
            if not isinstance(img_normalized, np.ndarray) or img_normalized.ndim != 4:
                 if isinstance(img_normalized, np.ndarray) and img_normalized.ndim == 3: img_expanded = np.expand_dims(img_normalized, axis=0)
                 else: self.logger.error(f"归一化未产生预期4D数组(fgid)。Shape: {img_normalized.shape}"); return None
            else: img_expanded = img_normalized 
            embedding_raw = self.feature_extractor.forward(img_expanded)
            if isinstance(embedding_raw, list): embedding_raw = np.array(embedding_raw)
            embedding = embedding_raw.flatten() 
            norm = np.linalg.norm(embedding); 
            if norm == 0: self.logger.warning("fgid 范数为零。"); return None
            embedding_normalized = embedding / norm
            self.logger.debug(f"成功提取 fgid，形状: {embedding_normalized.shape}")
            return embedding_normalized.astype(np.float32) 
        except Exception as e: self.logger.error(f"提取 fgid 失败: {e}", exc_info=True); return None

    # **** 新增：提取详细特征 fdid 的方法 ****
    @torch.no_grad() 
    def get_detailed_embedding(self, aligned_face_bgr_uint8: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialized or self.fdid_extractor is None:
            self.logger.error("详细特征提取器 (fdid_extractor) 未初始化或创建失败。")
            return None
        if aligned_face_bgr_uint8 is None or aligned_face_bgr_uint8.size == 0: self.logger.warning("get_detailed_embedding 收到空图像。"); return None
        if aligned_face_bgr_uint8.dtype != np.uint8: self.logger.error(f"输入 get_detailed_embedding 必须是 BGR uint8"); return None
        try:
            img_resized = preprocessing.resize_image(img=aligned_face_bgr_uint8, target_size=(self.target_size[1], self.target_size[0]))
            img_normalized = preprocessing.normalize_input(img=img_resized, normalization=self.normalization)
            if not isinstance(img_normalized, np.ndarray) or img_normalized.ndim != 4:
                if isinstance(img_normalized, np.ndarray) and img_normalized.ndim == 3: img_expanded = np.expand_dims(img_normalized, axis=0)
                else: self.logger.error(f"归一化未产生预期4D数组(fdid)。Shape: {img_normalized.shape}"); return None
            else: img_expanded = img_normalized 
            self.logger.debug(f"为 fdid 提取器准备的输入形状: {img_expanded.shape}")
            # **** 使用 Keras 模型的调用方式 ****
            fdid_raw = self.fdid_extractor(img_expanded, training=False) 
            if isinstance(fdid_raw, tf.Tensor): fdid_np = fdid_raw.numpy()
            elif isinstance(fdid_raw, np.ndarray): fdid_np = fdid_raw
            else: self.logger.error(f"fdid_extractor 返回了未知类型: {type(fdid_raw)}"); return None
            if fdid_np.shape[0] == 1:
                 fdid_map = fdid_np[0] 
                 self.logger.debug(f"成功提取 fdid 特征图，形状: {fdid_map.shape}"); return fdid_map.astype(np.float32)
            else: self.logger.error(f"fdid_extractor 返回的批次大小不为 1: {fdid_np.shape}"); return None
        except Exception as e: self.logger.error(f"提取 fdid 失败: {e}", exc_info=True); return None

    # **** 修改：extract_identity 返回 fgid 和 fdid ****
    def extract_identity(self, image_bgr: np.ndarray, align_target_size=(112, 112)) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从原始图像中提取身份特征（先对齐，再提取）。
        总是尝试同时返回 fgid 和 fdid。

        Args:
            image_bgr (np.ndarray): BGR格式的原始图像。
            align_target_size (tuple): MediaPipe对齐的目标尺寸 (height, width)。

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (fgid, fdid) 元组。
                 fgid 是 (embedding_size,) 的 1D 数组。
                 fdid 是 (H', W', C') 的 3D 数组。
                 如果任一提取失败，对应项为 None。
        """
        if not self.initialized:
            self.logger.error("DeepFaceRecognizer 未初始化。")
            return (None, None)
            
        self.logger.debug(f"开始身份提取流程 (fgid & fdid)...")
        
        # 1. 使用MediaPipe对齐
        aligned_face = self.detect_and_align(image_bgr, target_size=align_target_size)
        if aligned_face is None:
            self.logger.warning("MediaPipe人脸对齐失败，无法提取身份特征。")
            return (None, None)
        
        # 2. 提取全局特征 fgid
        fgid = self.get_embedding(aligned_face)
        if fgid is None:
            self.logger.warning("从对齐后的人脸提取 fgid 失败。")
            # 即使 fgid 失败，也尝试提取 fdid

        # 3. 提取详细特征 fdid
        fdid = self.get_detailed_embedding(aligned_face)
        if fdid is None:
            self.logger.warning("从对齐后的人脸提取 fdid 失败。")
            # 即使 fdid 失败，如果 fgid 成功了，仍然返回 fgid

        self.logger.info(f"身份特征提取完成。fgid_shape: {fgid.shape if fgid is not None else 'None'}, fdid_shape: {fdid.shape if fdid is not None else 'None'}")
        return fgid, fdid # 总是返回包含两个结果的元组

    # compute_similarity 方法保持不变
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        # ... (代码同上) ...
        if embedding1 is None or embedding2 is None: self.logger.warning("计算相似度时输入向量为None。"); return -2.0
        if embedding1.ndim > 1: embedding1 = embedding1.flatten()
        if embedding2.ndim > 1: embedding2 = embedding2.flatten()
        if embedding1.shape != embedding2.shape: self.logger.error(f"无法计算相似度：形状不匹配"); return -2.0
        try:
            similarity = np.dot(embedding1, embedding2); similarity = np.clip(similarity, -1.0, 1.0); return float(similarity)
        except Exception as e: self.logger.error(f"计算相似度时出错: {e}", exc_info=True); return -2.0

# --- 示例用法 (更新以测试新 extract_identity) ---
if __name__ == '__main__':
    # **** 配置标准 logging ****
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # 初始化时不再需要传入 fdid_layer_name，让它自动查找
        recognizer = DeepFaceRecognizer(model_name="Facenet512") 
        if not recognizer.initialized: logging.error("识别器初始化失败"); exit()
    except Exception as e: logging.error(f"初始化错误: {e}"); exit()

    # --- 加载示例图像 ---
    image_path1 = '/root/HiFiVFS/data/mayun1.jpg' 
    image_path2 = '/root/HiFiVFS/data/mayun1.jpg' 
    image_path3 = '/root/HiFiVFS/samples/vox2_640/sample_epoch951_step9500.png' # 使用你的样本图像
    img1 = cv2.imread(image_path1); img2 = cv2.imread(image_path2); img3 = cv2.imread(image_path3)
    if img1 is None or img2 is None or img3 is None: logging.error("未能加载图像"); exit()

    # --- 测试身份提取 (同时获取 fgid 和 fdid) ---
    logging.info(f"\n--- 提取图像1 (同时获取 fgid 和 fdid) ---")
    fgid1, fdid1 = recognizer.extract_identity(img1) # 默认 return_detailed=False，需要修改调用或方法
    # **** 修改调用方式 ****
    fgid1, fdid1 = recognizer.extract_identity(img1) # extract_identity 现在总是返回元组

    if fgid1 is not None: logging.info(f"图像1 fgid 提取成功，维度: {fgid1.shape}")
    else: logging.error("图像1 fgid 提取失败。")
    if fdid1 is not None: logging.info(f"图像1 fdid 提取成功，形状: {fdid1.shape}") 
    else: logging.error("图像1 fdid 提取失败。")

    logging.info(f"\n--- 提取图像3 (同时获取 fgid 和 fdid) ---")
    fgid3, fdid3 = recognizer.extract_identity(img3) # 总是返回元组
    if fgid3 is not None: logging.info(f"图像3 fgid 提取成功，维度: {fgid3.shape}")
    else: logging.error("图像3 fgid 提取失败。")
    if fdid3 is not None: logging.info(f"图像3 fdid 提取成功，形状: {fdid3.shape}") 
    else: logging.error("图像3 fdid 提取失败。")

    # --- 测试相似度计算 (使用 fgid) ---
    if fgid1 is not None and fgid3 is not None:
        similarity_1_3 = recognizer.compute_similarity(fgid1, fgid3)
        logging.info(f"\n--- 图像1 vs 图像3 fgid 相似度 ---")
        logging.info(f"相似度: {similarity_1_3:.4f}")

    # --- 测试直接调用 get_detailed_embedding ---
    # (这部分可以保留用于验证)
    logging.info("\n--- 测试直接对对齐人脸提取 fdid ---")
    aligned_face1 = recognizer.detect_and_align(img1)
    if aligned_face1 is not None:
         logging.info("图像1 对齐成功，尝试直接提取 fdid...")
         fdid1_direct = recognizer.get_detailed_embedding(aligned_face1)
         if fdid1_direct is not None:
              logging.info(f"直接从对齐人脸提取 fdid 成功，形状: {fdid1_direct.shape}")
              if fdid1 is not None:
                   diff_norm = np.linalg.norm(fdid1 - fdid1_direct)
                   logging.info(f" extract_identity vs get_detailed_embedding(aligned) fdid 差异 (L2范数): {diff_norm:.6f}")
                   # 由于浮点数精度问题，atol 可以设得宽松一些
                   assert np.allclose(fdid1, fdid1_direct, atol=1e-4), "Direct fdid extraction differs!" 
         else: logging.error("直接从对齐人脸提取 fdid 失败。")
    else: logging.warning("图像1 对齐失败，跳过直接 fdid 提取测试。")