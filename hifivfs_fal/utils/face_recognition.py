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
    """
    
    def __init__(self, model_name: str = "Facenet512", detector_backend: str = "retinaface"):
        """初始化DeepFace人脸识别器
        
        Args:
            model_name: 特征提取模型，可选: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace"
            detector_backend: 人脸检测器，可选: "opencv", "ssd", "mtcnn", "retinaface"
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
            # 这会触发模型下载（如果需要）
            try:
                logger.info("预加载DeepFace模型...")
                # 正确加载人脸识别模型
                DeepFace.build_model(model_name=self.model_name)
                logger.info(f"DeepFace {self.model_name} 模型加载成功")
                
                # 不要尝试预加载检测器，让它在第一次使用时自动加载
                # 确认检测器类型是有效的
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
                    enforce_detection=False,  # 不强制检测，处理已对齐的人脸
                    align=True  # 允许对齐
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
    
    def detect_and_align_fallback(self, image: np.ndarray) -> Optional[np.ndarray]:
        """当DeepFace对齐失败时使用MediaPipe作为备选对齐方法"""
        try:
            # 导入MediaPipe人脸对齐模块
            from hifivfs_fal.utils.detect_align_face import detect_align_face
            
            # 再次确保图像类型正确
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            logger.info("使用MediaPipe备选方法进行人脸对齐")
            # 使用MediaPipe对齐
            aligned_face = detect_align_face(image, target_size=(112, 112))
            
            if aligned_face is not None:
                logger.info("MediaPipe备选对齐成功")
                return aligned_face
            else:
                logger.warning("MediaPipe备选对齐也失败了")
                return None
        except Exception as e:
            logger.error(f"备选对齐方法失败: {e}")
            return None

    def detect_and_align(self, image: np.ndarray) -> Optional[np.ndarray]:
        """从原始图像中检测和对齐人脸"""
        if not self.initialized:
            logger.error("DeepFace未初始化")
            return None
        
        try:
            from deepface import DeepFace
            
            # 打印更多调试信息
            logger.info(f"输入图像: dtype={image.dtype}, shape={image.shape}, 范围=[{image.min():.4f}, {image.max():.4f}]")
            
            # 改进图像类型转换逻辑
            if image.dtype == np.float64:  # CV_64F类型
                logger.info(f"检测到float64类型图像，转换为uint8")
                # 确保数据在0-255范围或0-1范围
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            elif image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 使用临时文件保存图像
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                success = cv2.imwrite(temp_path, image)
                if not success:
                    logger.error(f"保存临时图像失败: {temp_path}")
                    return self.detect_and_align_fallback(image)
            
            # 使用DeepFace检测并预处理人脸
            try:
                faces = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=self.detector_backend,
                    enforce_detection=True,
                    align=True
                )
                
                # 清理临时文件
                os.remove(temp_path)
                
                if isinstance(faces, list) and len(faces) > 0:
                    # 获取置信度最高的人脸
                    best_face = max(faces, key=lambda x: x.get("confidence", 0))
                    face_img = best_face["face"]
                    
                    # 调整到112x112大小
                    face_img = cv2.resize(face_img, (112, 112))
                    
                    # 转换为BGR（如果需要）
                    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    
                    return face_img
                else:
                    logger.warning("DeepFace未检测到人脸，尝试备选方法")
                    return self.detect_and_align_fallback(image)
                
            except Exception as e:
                logger.error(f"DeepFace人脸提取失败: {e}")
                
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # 使用备选方法
                return self.detect_and_align_fallback(image)
                
        except Exception as e:
            logger.error(f"人脸检测与对齐失败: {e}")
            return self.detect_and_align_fallback(image)
    
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
            # 确保图像类型正确
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                    
            from deepface import DeepFace
            
            # 使用临时文件保存图像
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image)
            
            # 直接使用DeepFace的represent功能提取特征
            try:
                embedding_obj = DeepFace.represent(
                    img_path=temp_path, 
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=True,
                    align=True
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


class DlibFaceRecognizer:
    """
    基于Dlib的人脸识别器
    提供人脸检测、对齐和特征提取功能
    """
    
    def __init__(self):
        """初始化Dlib人脸识别器"""
        self.initialized = False
        self.face_detector = None
        self.shape_predictor = None
        self.face_recognizer = None
        self.embedding_size = 128  # Dlib人脸模型输出128维特征
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化Dlib模型"""
        try:
            # 仅在使用时导入dlib
            import dlib
            
            # 模型路径
            shape_predictor_path = os.path.join(os.path.expanduser('~'), '.dlib', 'shape_predictor_68_face_landmarks.dat')
            face_recognition_model_path = os.path.join(os.path.expanduser('~'), '.dlib', 'dlib_face_recognition_resnet_model_v1.dat')
            
            # 创建目录
            os.makedirs(os.path.dirname(shape_predictor_path), exist_ok=True)
            
            # 下载模型文件（如果不存在）
            if not os.path.exists(shape_predictor_path):
                logger.info("下载Dlib人脸关键点模型...")
                self._download_file(
                    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                    shape_predictor_path + ".bz2"
                )
                # 解压
                import bz2
                with bz2.open(shape_predictor_path + ".bz2", "rb") as f_in:
                    with open(shape_predictor_path, "wb") as f_out:
                        f_out.write(f_in.read())
                os.remove(shape_predictor_path + ".bz2")
                logger.info("人脸关键点模型下载完成")
            
            if not os.path.exists(face_recognition_model_path):
                logger.info("下载Dlib人脸识别模型...")
                self._download_file(
                    "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
                    face_recognition_model_path + ".bz2"
                )
                # 解压
                import bz2
                with bz2.open(face_recognition_model_path + ".bz2", "rb") as f_in:
                    with open(face_recognition_model_path, "wb") as f_out:
                        f_out.write(f_in.read())
                os.remove(face_recognition_model_path + ".bz2")
                logger.info("人脸识别模型下载完成")
            
            # 加载模型
            self.face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
            self.face_recognizer = dlib.face_recognition_model_v1(face_recognition_model_path)
            
            logger.info("Dlib模型加载成功")
            self.initialized = True
            
        except ImportError:
            logger.error("无法导入Dlib库。请安装: pip install dlib")
            self.initialized = False
        except Exception as e:
            logger.error(f"Dlib初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.initialized = False
    
    def _download_file(self, url, dest_path):
        """下载文件到指定路径"""
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # 打印进度
                            done = int(50 * downloaded / total_size)
                            if total_size > 0:
                                logger.info(f"\r下载进度: [{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} 字节")
            
            return True
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False
    
    def detect_and_align(self, image: np.ndarray) -> Optional[np.ndarray]:
        """从原始图像中检测和对齐人脸
        
        Args:
            image: BGR格式图像
            
        Returns:
            对齐后的人脸图像，如果失败则返回None
        """
        if not self.initialized:
            logger.error("Dlib未初始化")
            return None
        
        try:
            import dlib
            
            # 确保图像类型正确
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 转换为RGB (Dlib使用RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            faces = self.face_detector(rgb_image, 1)
            
            if len(faces) == 0:
                logger.warning("未检测到人脸")
                return None
            
            # 获取最大的人脸
            face_rect = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # 获取人脸关键点
            shape = self.shape_predictor(rgb_image, face_rect)
            
            # 对齐人脸
            face_chip = dlib.get_face_chip(rgb_image, shape, size=112)
            
            # 转换回BGR
            face_chip_bgr = cv2.cvtColor(face_chip, cv2.COLOR_RGB2BGR)
            
            return face_chip_bgr
            
        except Exception as e:
            logger.error(f"人脸检测与对齐失败: {e}")
            return None
    
    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """从人脸图像提取特征向量
        
        Args:
            face_img: BGR格式的人脸图像
            
        Returns:
            特征向量，如果失败则返回None
        """
        if not self.initialized:
            logger.error("Dlib未初始化")
            return None
        
        if face_img is None:
            logger.error("输入人脸图像为None")
            return None
        
        try:
            import dlib
            
            # 确保图像类型正确
            if face_img.dtype != np.uint8:
                if face_img.max() <= 1.0:
                    face_img = (face_img * 255).astype(np.uint8)
                else:
                    face_img = face_img.astype(np.uint8)
            
            # 转换为RGB (Dlib使用RGB)
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # 如果是已对齐的人脸，直接提取特征
            if rgb_face.shape[0] == 112 and rgb_face.shape[1] == 112:
                embedding = np.array(self.face_recognizer.compute_face_descriptor(rgb_face))
            else:
                # 检测人脸
                faces = self.face_detector(rgb_face, 1)
                
                if len(faces) == 0:
                    logger.warning("未检测到人脸")
                    return None
                
                # 获取最大的人脸
                face_rect = max(faces, key=lambda rect: rect.width() * rect.height())
                
                # 获取人脸关键点
                shape = self.shape_predictor(rgb_face, face_rect)
                
                # 提取特征
                embedding = np.array(self.face_recognizer.compute_face_descriptor(rgb_face, shape))
            
            # 归一化特征向量
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"特征提取错误: {e}")
            return None
    
    def extract_identity(self, image: np.ndarray) -> Optional[np.ndarray]:
        """从图像中提取身份特征，包括检测、对齐和特征提取
        
        Args:
            image: BGR格式图像
            
        Returns:
            身份特征向量，如果失败则返回None
        """
        if not self.initialized:
            logger.error("Dlib未初始化")
            return None
        
        try:
            # 确保图像类型正确
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 转换为RGB (Dlib使用RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            import dlib
            
            # 检测人脸
            faces = self.face_detector(rgb_image, 1)
            
            if len(faces) == 0:
                logger.warning("未检测到人脸")
                return None
            
            # 获取最大的人脸
            face_rect = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # 获取人脸关键点
            shape = self.shape_predictor(rgb_image, face_rect)
            
            # 提取特征
            embedding = np.array(self.face_recognizer.compute_face_descriptor(rgb_image, shape))
            
            # 归一化特征向量
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
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


class DummyFaceRecognizer:
    """备用人脸识别器，用于测试和调试"""
    
    def __init__(self):
        """初始化备用人脸识别器"""
        self.embedding_size = 512
        self.initialized = True
        logger.warning("使用备用人脸识别器 - 将生成随机特征")
    
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """生成一个随机特征向量
        
        Args:
            face_img: 人脸图像
            
        Returns:
            随机特征向量
        """
        if face_img is None:
            return None
            
        # 使用图像的平均值作为随机种子
        if hasattr(face_img, 'mean'):
            seed = int(face_img.mean() * 1000) % 10000
        else:
            seed = 42
            
        # 生成随机向量
        np.random.seed(seed)
        embedding = np.random.normal(0, 0.1, 512)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def detect_and_align(self, image: np.ndarray) -> np.ndarray:
        """模拟人脸检测和对齐
        
        Args:
            image: 输入图像
            
        Returns:
            裁剪和调整大小的图像
        """
        if image is None:
            return None
            
        # 简单裁剪中心区域
        h, w = image.shape[:2]
        
        # 裁剪中心区域并调整大小
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        
        crop = image[y1:y1+size, x1:x1+size]
        resized = cv2.resize(crop, (112, 112))
        
        return resized
    
    def extract_identity(self, image: np.ndarray) -> np.ndarray:
        """从图像中提取模拟身份特征
        
        Args:
            image: 输入图像
            
        Returns:
            随机特征向量
        """
        return self.get_embedding(image)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个特征向量之间的模拟相似度
        
        Args:
            embedding1: 第一个特征向量
            embedding2: 第二个特征向量
            
        Returns:
            相似度分数
        """
        if embedding1 is None or embedding2 is None:
            return -2.0
            
        # 简单的余弦相似度
        return float(np.dot(embedding1, embedding2))


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
    
    # 选择使用哪个识别器
    recognizer_type = "deepface"  # 'deepface', 'dlib' 或 'dummy'
    
    # 初始化人脸识别器
    try:
        print(f"初始化{recognizer_type}人脸识别器...")
        
        if recognizer_type == "deepface":
            recognizer = DeepFaceRecognizer()
        elif recognizer_type == "dlib":
            recognizer = DlibFaceRecognizer()
        else:
            recognizer = DummyFaceRecognizer()
        
        if not hasattr(recognizer, 'initialized') or not recognizer.initialized:
            print(f"{recognizer_type}识别器初始化失败，使用备用识别器")
            recognizer = DummyFaceRecognizer()
        else:
            print(f"{recognizer_type}识别器初始化成功")
    except Exception as e:
        print(f"初始化错误: {e}")
        print("使用备用识别器")
        recognizer = DummyFaceRecognizer()
    
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