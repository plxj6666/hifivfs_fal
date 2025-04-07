# hifivfs_fal/utils/detect_align_face.py

import cv2
import numpy as np
import mediapipe as mp
import logging
import math
import os

# 获取日志记录器
logger = logging.getLogger(__name__)
# 配置日志级别（如果尚未在别处配置）
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MediaPipe Face Mesh Landmark Indices (for 468 landmarks) ---
# 参考: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# 这些索引通常比较稳定
LMK_LEFT_EYE_OUTER_CORNER = 33
LMK_LEFT_EYE_INNER_CORNER = 133 # Alternative: 246 ?
LMK_RIGHT_EYE_OUTER_CORNER = 263
LMK_RIGHT_EYE_INNER_CORNER = 362 # Alternative: 466 ?
LMK_NOSE_TIP = 1
LMK_MOUTH_LEFT_CORNER = 61
LMK_MOUTH_RIGHT_CORNER = 291

# 眼睛中心点（通过平均获得更稳定的估计）
LMK_LEFT_EYE_CENTER_PTS = [33, 160, 158, 133, 153, 144] # 外角, 上中, 下中, 内角, 下缘, 上缘
LMK_RIGHT_EYE_CENTER_PTS = [263, 387, 385, 362, 380, 373] # 外角, 上中, 下中, 内角, 下缘, 上缘

# 眉毛点有时也用于对齐
LMK_LEFT_EYEBROW_OUTER = 70
LMK_RIGHT_EYEBROW_OUTER = 300


class MediapipeFaceAligner:
    """使用MediaPipe Face Mesh进行人脸检测、关键点提取和对齐"""

    def __init__(self,
                 static_image_mode=True,
                 max_num_faces=1,
                 refine_landmarks=True, # 设为 True 可以获取虹膜点 (468-477)，但基础对齐通常不需要
                 min_detection_confidence=0.4, # 稍微提高一点默认值，减少误检
                 min_tracking_confidence=0.4):
        """初始化MediaPipe Face Mesh"""
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # 延迟初始化 face_mesh，避免在不需要时加载
        self._face_mesh = None
        logger.info(f"MediaPipeFaceAligner configured (will initialize on first use).")

    def _initialize_mesh(self):
        """Lazy initialization of FaceMesh."""
        if self._face_mesh is None:
            logger.info("Initializing MediaPipe Face Mesh...")
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=self.static_image_mode,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine_landmarks,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                logger.info("MediaPipe Face Mesh initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe Face Mesh: {e}", exc_info=True)
                raise RuntimeError("Could not initialize MediaPipe Face Mesh") from e

    @property
    def face_mesh(self):
        """Getter for the FaceMesh instance, initializes if needed."""
        self._initialize_mesh()
        return self._face_mesh

    def detect_landmarks(self, img_rgb: np.ndarray) -> list | None:
        """
        检测图像中的人脸关键点 (返回第一个检测到的人脸的原始像素坐标)。
        增加了对检测结果的检查。
        """
        if img_rgb is None or img_rgb.size == 0:
            logger.warning("Input image is empty.")
            return None

        original_h, original_w = img_rgb.shape[:2]
        logger.debug(f"Detecting landmarks on image of size: {original_w}x{original_h}")

        try:
            # Process the image
            results = self.face_mesh.process(img_rgb)
        except Exception as e:
            logger.error(f"MediaPipe face_mesh.process failed: {e}", exc_info=True)
            return None

        if not results or not results.multi_face_landmarks:
            logger.debug("No face landmarks detected by MediaPipe.")
            return None

        # 只处理第一个检测到的人脸
        face_landmarks = results.multi_face_landmarks[0]
        
        # 检查landmark数量是否符合预期
        num_landmarks = len(face_landmarks.landmark)
        expected_min = 468
        expected_max = 478 if self.refine_landmarks else 468
        if not (expected_min <= num_landmarks <= expected_max):
             logger.warning(f"Unexpected number of landmarks detected: {num_landmarks}. Expected between {expected_min} and {expected_max}.")
             # 即使数量不对，也尝试继续，但可能下游会失败
             # return None # 或者在这里直接失败

        landmarks_px = []
        for i, landmark in enumerate(face_landmarks.landmark):
            # 检查 landmark 坐标是否有效
            if not (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1):
                # logger.warning(f"Invalid relative coordinate for landmark {i}: ({landmark.x}, {landmark.y}). Skipping face.")
                # return None # 如果坐标无效，可能整个检测结果都有问题
                # 暂时只记录，并尝试使用裁剪后的值
                px = np.clip(landmark.x * original_w, 0, original_w - 1)
                py = np.clip(landmark.y * original_h, 0, original_h - 1)
                logger.warning(f"Clipped invalid relative coordinate for landmark {i}: ({landmark.x}, {landmark.y}) -> ({px}, {py})")
            else:
                px = landmark.x * original_w
                py = landmark.y * original_h
            landmarks_px.append((px, py))

        logger.debug(f"Detected {len(landmarks_px)} landmarks.")
        return landmarks_px # 返回原始图像尺寸下的像素坐标列表 [(x1, y1), (x2, y2), ...]

    def _get_stable_landmarks_for_alignment(self, all_landmarks_px: list) -> np.ndarray | None:
        """
        从所有关键点中提取用于对齐的5个稳定点。
        使用平均策略提高稳定性。
        """
        if not all_landmarks_px or len(all_landmarks_px) < 468:
             logger.warning(f"Not enough landmarks provided for stable extraction: {len(all_landmarks_px)}")
             return None
        
        try:
            # --- 策略1: 使用平均眼睛中心点 ---
            left_eye_center = np.mean([all_landmarks_px[i] for i in LMK_LEFT_EYE_CENTER_PTS], axis=0)
            right_eye_center = np.mean([all_landmarks_px[i] for i in LMK_RIGHT_EYE_CENTER_PTS], axis=0)
            
            # --- 策略2: 使用内外眼角 (可能选一组) ---
            # lm_left_eye = all_landmarks_px[LMK_LEFT_EYE_INNER_CORNER]
            # lm_right_eye = all_landmarks_px[LMK_RIGHT_EYE_INNER_CORNER]
            # lm_left_eye = all_landmarks_px[LMK_LEFT_EYE_OUTER_CORNER]
            # lm_right_eye = all_landmarks_px[LMK_RIGHT_EYE_OUTER_CORNER]

            # 选择策略1 (平均中心点)
            lm_left_eye = left_eye_center
            lm_right_eye = right_eye_center

            lm_nose = np.array(all_landmarks_px[LMK_NOSE_TIP])
            lm_mouth_left = np.array(all_landmarks_px[LMK_MOUTH_LEFT_CORNER])
            lm_mouth_right = np.array(all_landmarks_px[LMK_MOUTH_RIGHT_CORNER])

            dst_points = np.array([
                lm_left_eye, lm_right_eye, lm_nose, lm_mouth_left, lm_mouth_right
            ], dtype=np.float32)

            # 检查是否有 NaN 或 Inf 值
            if not np.all(np.isfinite(dst_points)):
                logger.error("NaN or Inf found in selected landmark points.")
                return None

            return dst_points

        except IndexError as e:
            logger.error(f"Failed to get required landmarks using indices: {e}. Landmark count: {len(all_landmarks_px)}")
            return None
        except Exception as e:
            logger.error(f"Error during stable landmark extraction: {e}", exc_info=True)
            return None

    def align_face_arcface(self, img_bgr: np.ndarray, target_size=(112, 112)) -> np.ndarray | None:
        """
        使用MediaPipe关键点进行ArcFace标准对齐。
        改进了关键点选择和错误处理。

        Args:
            img_bgr (np.ndarray): BGR格式的输入图像。
            target_size (tuple): 目标输出尺寸 (高度, 宽度)。

        Returns:
            np.ndarray | None: 对齐后的人脸图像 (BGR), 如果失败则返回None。
        """
        if img_bgr is None or img_bgr.size == 0:
            logger.warning("Input image for alignment is empty.")
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # 检测关键点
        all_landmarks_px = self.detect_landmarks(img_rgb)
        if all_landmarks_px is None:
            logger.warning("Landmark detection failed, cannot align face.")
            return None

        # 提取用于对齐的5个稳定关键点
        dst_landmarks = self._get_stable_landmarks_for_alignment(all_landmarks_px)
        if dst_landmarks is None:
            logger.warning("Failed to extract stable landmarks for alignment.")
            return None
        
        logger.debug(f"Using landmarks for alignment: {dst_landmarks.tolist()}")

        # --- ArcFace 标准参考点 (112x112 空间) ---
        # 这些是广泛使用的参考点，确保它们与你的 Eid 模型训练时使用的对齐方式一致
        src_ref_points = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # 根据目标尺寸缩放参考点
        target_h, target_w = target_size
        src_scaled_points = src_ref_points.copy()
        src_scaled_points[:, 0] *= target_w / 112.0
        src_scaled_points[:, 1] *= target_h / 112.0

        # 计算仿射变换矩阵 (Similarity Transform)
        # 使用 estimateAffinePartial2D 更适合估计相似变换（旋转、缩放、平移）
        # 它返回一个 2x3 的矩阵 M
        try:
            # method=cv2.LMEDS 或 cv2.RANSAC 都可以增加鲁棒性
            # RANSAC 通常对离群点更鲁棒
            M, inliers = cv2.estimateAffinePartial2D(dst_landmarks, src_scaled_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            if M is None:
                logger.warning("estimateAffinePartial2D failed (returned None). Trying estimateAffine2D as fallback.")
                # Fallback: estimateAffine2D 估计完整的仿射变换，可能对非刚性形变更敏感
                M, inliers = cv2.estimateAffine2D(dst_landmarks, src_scaled_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M is None:
                    logger.error("Both estimateAffinePartial2D and estimateAffine2D failed.")
                    return None
            
            num_inliers = np.sum(inliers)
            logger.debug(f"Affine transform estimated with {num_inliers} inliers.")
            if num_inliers < 3: # 如果内点太少，变换可能不可靠
                 logger.warning(f"Transform estimation has only {num_inliers} inliers, result might be unstable.")
                 # return None # 可以选择在这里失败

        except cv2.error as e:
             logger.error(f"OpenCV error during Affine Transform estimation: {e}", exc_info=True)
             return None
        except Exception as e:
             logger.error(f"Unexpected error during Affine Transform estimation: {e}", exc_info=True)
             return None

        logger.debug(f"Estimated Affine Matrix M:\n{M}")

        # 应用仿射变换
        # warpAffine 的 dsize 参数是 (宽度, 高度)
        try:
            aligned_face = cv2.warpAffine(img_bgr, M, (target_w, target_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        except cv2.error as e:
            logger.error(f"OpenCV error during warpAffine: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during warpAffine: {e}", exc_info=True)
            return None
            
        logger.info(f"Face aligned successfully to {target_w}x{target_h}.")
        return aligned_face

    def close(self):
        """关闭MediaPipe Face Mesh资源"""
        if self._face_mesh is not None:
            logger.info("Closing MediaPipe Face Mesh...")
            try:
                self._face_mesh.close()
                self._face_mesh = None
                logger.info("MediaPipe Face Mesh closed.")
            except Exception as e:
                logger.error(f"Error closing MediaPipe Face Mesh: {e}", exc_info=True)


# --- 单例模式获取 Aligner ---
_aligner_instance = None

def get_mediapipe_face_aligner_instance(**kwargs):
    """
    获取单例的MediaPipe Face Aligner实例。
    允许在第一次获取时传递初始化参数。
    """
    global _aligner_instance
    if _aligner_instance is None:
        logger.info("Creating global MediaPipeFaceAligner instance...")
        # 可以从配置或参数传入 min_detection_confidence 等
        conf = kwargs.get('min_detection_confidence', 0.4)
        track_conf = kwargs.get('min_tracking_confidence', 0.4)
        refine = kwargs.get('refine_landmarks', True)
        static_mode = kwargs.get('static_image_mode', True)
        max_faces = kwargs.get('max_num_faces', 1)

        _aligner_instance = MediapipeFaceAligner(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine,
            min_detection_confidence=conf,
            min_tracking_confidence=track_conf
        )
    return _aligner_instance


# --- 简化的便捷函数 ---
def detect_align_face(frame_bgr: np.ndarray, target_size=(112, 112)) -> np.ndarray | None:
    """
    使用全局 Aligner 实例进行人脸检测与对齐。

    Args:
        frame_bgr (np.ndarray): 输入 BGR 图像。
        target_size (tuple): 目标对齐尺寸 (高度, 宽度)。

    Returns:
        np.ndarray | None: 对齐后的人脸图像 (BGR), 或 None 如果失败。
    """
    if frame_bgr is None or frame_bgr.size == 0:
        logger.warning("detect_align_face received an empty frame.")
        return None

    # 获取（或创建）全局 aligner 实例
    # 如果需要特定配置，可以在首次调用时传入参数
    aligner = get_mediapipe_face_aligner_instance() 

    # 直接调用对齐方法
    aligned_face = aligner.align_face_arcface(frame_bgr, target_size=target_size)

    if aligned_face is None:
        logger.warning("Face alignment failed for the input frame.")
        return None

    return aligned_face

