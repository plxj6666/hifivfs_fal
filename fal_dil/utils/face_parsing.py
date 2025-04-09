import cv2
import numpy as np
import logging
import mediapipe as mp
from typing import Optional
import torch

logger = logging.getLogger(__name__)

class FaceParser:
    """面部解析类，使用MediaPipe Face Mesh"""
    
    def __init__(self, device="cuda", model_path=None):
        self.device = device
        self.initialized = False
        self.mp_face_mesh = mp.solutions.face_mesh
        self.initialize()
    
    def initialize(self, model_path=None):
        """初始化MediaPipe Face Mesh"""
        try:
            # MediaPipe不需要外部模型文件
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True, 
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.initialized = True
            logger.info("已初始化MediaPipe Face Mesh解析模型")
        except Exception as e:
            logger.error(f"初始化MediaPipe Face Mesh失败: {e}")
            self.initialized = False
    
    def parse(self, img_bgr):
        """解析面部区域，增强处理各种输入格式"""
        # 处理输入格式和精度
        if torch.is_tensor(img_bgr):
            # 处理Tensor输入
            if img_bgr.ndim == 4 and img_bgr.shape[0] == 1:  # (1,C,H,W)
                img_bgr = img_bgr.squeeze(0)  # (C,H,W)
                
            if img_bgr.ndim == 3:  # (C,H,W)
                # Tensor格式从(C,H,W)转为(H,W,C)
                img_bgr = img_bgr.permute(1, 2, 0).cpu().numpy()
                
            # 处理值范围
            if img_bgr.max() <= 1.0:
                img_bgr = (img_bgr * 255).astype(np.uint8)
            else:
                img_bgr = img_bgr.astype(np.uint8)
                
            # 确保BGR顺序
            if img_bgr.shape[2] == 3:  # RGB->BGR
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    
        if not self.initialized:
            logger.warning("MediaPipe Face Mesh未初始化，尝试初始化...")
            self.initialize()
            if not self.initialized:
                logger.error("MediaPipe Face Mesh初始化失败")
                # 返回全白遮罩（没有要替换的区域）
                return np.ones((img_bgr.shape[0], img_bgr.shape[1], 1), dtype=np.float32)
        
        # 预处理
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 使用MediaPipe处理
        results = self.face_mesh.process(img_rgb)
        
        # 创建全白遮罩（默认全部保留）
        mask = np.ones((h, w), dtype=np.float32)
        
        if results.multi_face_landmarks:
            # 获取第一个检测到的人脸
            face_landmarks = results.multi_face_landmarks[0]
            
            # 定义面部区域索引（与之前相同）
            left_eye = list(range(33, 42)) + list(range(160, 173))
            right_eye = list(range(263, 273)) + list(range(362, 373))
            left_eyebrow = list(range(46, 53)) + list(range(285, 295))
            right_eyebrow = list(range(276, 284)) + list(range(295, 305))
            nose = list(range(168, 194))
            lips = list(range(0, 17)) + list(range(61, 68)) + list(range(291, 308))
            
            # 创建面部轮廓的完整多边形
            # 添加面部轮廓点
            face_contour = list(range(0, 17)) + list(range(17, 78))
            
            # 创建多边形顶点列表
            polygons = []
            
            # 1. 首先添加面部轮廓作为一个整体
            face_points = []
            for idx in face_contour:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    pt = (int(lm.x * w), int(lm.y * h))
                    face_points.append(pt)
            
            if len(face_points) > 2:
                polygons.append(np.array(face_points, dtype=np.int32))
                
            # 2. 可选：添加各个面部特征的多边形（眼睛、鼻子等）
            for region in [left_eye, right_eye, left_eyebrow, right_eyebrow, nose, lips]:
                region_points = []
                for idx in region:
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        pt = (int(lm.x * w), int(lm.y * h))
                        region_points.append(pt)
                if len(region_points) > 2:
                    polygons.append(np.array(region_points, dtype=np.int32))
            
            # 填充多边形区域为黑色（0值）- 表示要替换的区域
            cv2.fillPoly(mask, polygons, 0.0)
            
            # 稍微膨胀黑色区域，确保完全覆盖面部
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)  # 使用erode扩大黑色区域
            
        # 添加通道维度
        mask = mask[..., np.newaxis]
        return mask

    def download_model(self, save_path):
        """
        MediaPipe不需要下载额外模型，此方法仅保留API兼容性
        """
        logger.info("MediaPipe Face Mesh不需要下载额外模型文件")
        return True