import numpy as np
import cv2
import logging
import os
from typing import Optional, List, Tuple, Union
import requests
import tempfile
import time

# 配置日志
logger = logging.getLogger(__name__)

class DeepFaceRecognizer:
    """
    基于DeepFace的人脸识别器
    提供人脸检测、对齐和特征提取功能
    统一使用MediaPipe进行人脸对齐
    """
    
    def __init__(self, model_name: str = "Facenet512", detector_backend: str = "retinaface"):
        """初始化DeepFace人脸识别器
        
        Args:
            model_name: 特征提取模型，可选: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace"
            detector_backend: 特征提取时使用的检测器，可选: "opencv", "ssd", "mtcnn", "retinaface"
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.model = None
        self.detector = None
        self.initialized = False
        self.embedding_size = 512 if model_name == "Facenet512" else 128
        
        # 初始化DeepFace
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化DeepFace模型"""
        try:
            # 仅在使用时导入，避免不必要的依赖
            from deepface import DeepFace
            
            logger.info(f"尝试初始化DeepFace，使用模型: {self.model_name}")
            
            # 预加载模型，以避免第一次调用的延迟
            try:
                logger.info("预加载DeepFace模型...")
                # 正确加载人脸识别模型
                DeepFace.build_model(model_name=self.model_name)
                logger.info(f"DeepFace {self.model_name} 模型加载成功")
                
                # 不预加载检测器，让它在第一次使用时自动加载
                valid_backends = ["opencv", "ssd", "mtcnn", "retinaface", "mediapipe", "yolov8"]
                if self.detector_backend not in valid_backends:
                    logger.warning(f"不支持的检测器后端: {self.detector_backend}，将使用默认检测器")
                    self.detector_backend = "opencv"  # 降级到最简单的检测器
                
                logger.info(f"将在首次调用时加载 {self.detector_backend} 人脸检测器")
                self.initialized = True
            except Exception as e:
                logger.error(f"DeepFace模型加载失败: {e}")
                raise
            
        except ImportError:
            logger.error("无法导入DeepFace库。请安装: pip install deepface")
            logger.info("您可能还需要安装: pip install tensorflow retina-face mtcnn")
            self.initialized = False
        except Exception as e:
            logger.error(f"DeepFace初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.initialized = False
    
    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """从人脸图像提取特征向量
        
        Args:
            face_img: BGR格式的人脸图像
            
        Returns:
            特征向量，如果失败则返回None
        """
        if not self.initialized:
            logger.error("DeepFace未初始化")
            return None
        
        if face_img is None:
            logger.error("输入人脸图像为None")
            return None
        
        try:
            from deepface import DeepFace
            
            # 确保图像类型正确
            if face_img.dtype != np.uint8:
                if face_img.max() <= 1.0:
                    face_img = (face_img * 255).astype(np.uint8)
                else:
                    face_img = face_img.astype(np.uint8)
            
            # 使用临时文件保存图像（DeepFace API需要文件路径）
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, face_img)
            
            # 使用DeepFace提取特征向量
            try:
                embedding_obj = DeepFace.represent(
                    img_path=temp_path, 
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=True,  # 改回和原来一样的参数
                    align=True  # 让DeepFace再次对齐，保持一致性
                )
                
                # 清理临时文件
                os.remove(temp_path)
                
                if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                    embedding = np.array(embedding_obj[0]["embedding"])
                    # 归一化特征向量
                    embedding = embedding / np.linalg.norm(embedding)
                    logger.info(f"成功提取特征，形状: {embedding.shape}")
                    return embedding
                else:
                    logger.warning("DeepFace未返回有效的特征向量")
                    return None
                
            except Exception as e:
                logger.error(f"DeepFace特征提取失败: {e}")
                
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return None
                
        except Exception as e:
            logger.error(f"特征提取错误: {e}")
            return None

    def detect_and_align(self, image: np.ndarray) -> Optional[np.ndarray]:
        """从原始图像中检测和对齐人脸，专门使用MediaPipe
        
        Args:
            image: BGR格式图像
            
        Returns:
            对齐后的人脸图像，如果失败则返回None
        """
        try:
            # 导入MediaPipe人脸对齐模块
            from hifivfs_fal.utils.detect_align_face import detect_align_face
            
            # 确保图像类型正确
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            logger.info(f"使用MediaPipe进行人脸对齐，输入图像: dtype={image.dtype}, shape={image.shape}")
            
            # 使用MediaPipe对齐
            aligned_face = detect_align_face(image, target_size=(112, 112))
            
            if aligned_face is not None:
                logger.info("MediaPipe人脸对齐成功")
                return aligned_face
            else:
                logger.warning("MediaPipe未检测到人脸")
                return None
        except Exception as e:
            logger.error(f"人脸检测与对齐失败: {e}")
            return None
    
    def extract_identity(self, image: np.ndarray) -> Optional[np.ndarray]:
        """从图像中提取身份特征，包括检测、对齐和特征提取
        
        Args:
            image: BGR格式图像
            
        Returns:
            身份特征向量，如果失败则返回None
        """
        if not self.initialized:
            logger.error("DeepFace未初始化")
            return None
        
        try:
            # 首先使用MediaPipe进行人脸对齐
            aligned_face = self.detect_and_align(image)
            
            if aligned_face is None:
                logger.warning("MediaPipe人脸对齐失败，尝试直接提取特征")
                # 确保图像类型正确
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # 使用DeepFace的represent功能直接提取特征
                from deepface import DeepFace
                
                # 使用临时文件保存图像
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, image)
                
                try:
                    embedding_obj = DeepFace.represent(
                        img_path=temp_path, 
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=True,  # 强制检测
                        align=True  # 需要对齐
                    )
                    
                    # 清理临时文件
                    os.remove(temp_path)
                    
                    if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                        embedding = np.array(embedding_obj[0]["embedding"])
                        # 归一化特征向量
                        embedding = embedding / np.linalg.norm(embedding)
                        return embedding
                    else:
                        logger.warning("DeepFace未返回有效的特征向量")
                        return None
                    
                except Exception as e:
                    logger.error(f"DeepFace身份提取失败: {e}")
                    
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    return None
            else:
                # 如果对齐成功，使用对齐后的人脸提取特征
                return self.get_embedding(aligned_face)
                
        except Exception as e:
            logger.error(f"身份提取失败: {e}")
            return None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个特征向量之间的余弦相似度
        
        Args:
            embedding1: 第一个特征向量
            embedding2: 第二个特征向量
            
        Returns:
            相似度分数，范围[-1, 1]
        """
        if embedding1 is None or embedding2 is None:
            return -2.0
            
        try:
            # 确保是一维向量
            if embedding1.ndim > 1:
                embedding1 = embedding1.flatten()
            if embedding2.ndim > 1:
                embedding2 = embedding2.flatten()
                
            # 确保向量已经归一化
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if abs(norm1 - 1.0) > 1e-4:
                embedding1 = embedding1 / norm1
                
            if abs(norm2 - 1.0) > 1e-4:
                embedding2 = embedding2 / norm2
                
            # 计算余弦相似度
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return -2.0


# --- 测试代码 ---
if __name__ == "__main__":
    print("--- 测试人脸识别器 ---")
    
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 测试图像路径
    test_image_path = "/root/HiFiVFS/data/test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"创建测试图像: {test_image_path}")
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, dummy_img)
    
    # 初始化人脸识别器
    try:
        print("初始化DeepFace人脸识别器...")
        recognizer = DeepFaceRecognizer()
        
        if not recognizer.initialized:
            print("DeepFace识别器初始化失败")
        else:
            print("DeepFace识别器初始化成功")
    except Exception as e:
        print(f"初始化错误: {e}")
    
    # 测试特征提取
    if os.path.exists(test_image_path):
        print(f"\n加载测试图像: {test_image_path}")
        img_bgr = cv2.imread(test_image_path)
        
        if img_bgr is None:
            print(f"加载图像失败: {test_image_path}")
        else:
            print("图像加载成功")
            print("提取特征...")
            
            # 提取特征
            start_time = time.time()
            embedding = recognizer.extract_identity(img_bgr)
            elapsed = time.time() - start_time
            
            if embedding is not None:
                print(f"\n--- 特征提取成功 (耗时: {elapsed:.2f}秒) ---")
                print(f"特征类型: {type(embedding)}")
                print(f"特征形状: {embedding.shape}")
                print(f"特征范数: {np.linalg.norm(embedding):.4f}")
                print(f"特征片段: {embedding[:5]}...")
            else:
                print("\n--- 特征提取失败 ---")
                
                # 尝试检测和对齐
                print("尝试人脸检测和对齐...")
                aligned_face = recognizer.detect_and_align(img_bgr)
                
                if aligned_face is not None:
                    print("人脸对齐成功，尝试提取特征...")
                    embedding = recognizer.get_embedding(aligned_face)
                    
                    if embedding is not None:
                        print("\n--- 对齐后特征提取成功 ---")
                        print(f"特征形状: {embedding.shape}")
                    else:
                        print("对齐后特征提取仍然失败")
                else:
                    print("人脸对齐失败")
    
    print("\n--- 测试完成 ---")