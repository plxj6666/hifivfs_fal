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
# logging.basicConfig(level=logging.INFO)

class MediapipeFaceAligner:
    """使用MediaPipe Face Mesh进行人脸检测、关键点提取和对齐"""

    def __init__(self,
                 static_image_mode=True, # 处理静态图片还是视频流
                 max_num_faces=1,        # 最多检测人脸数量
                 refine_landmarks=True,  # 细化眼部和唇部关键点
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """初始化MediaPipe Face Mesh"""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # 定义用于ArcFace对齐的MediaPipe关键点索引
        # 这些索引需要根据MediaPipe Face Mesh的468/478点模型确定
        # 这是一个常见的映射，但可能需要根据版本微调：
        # 参考：https://github.com/google/mediapipe/issues/1615#issuecomment-752534379
        # 或 https://github.com/serengil/retinaface/blob/master/retinaface/commons/align.py
        # 索引可能基于0开始
        self.ARCFAE_LMK_INDICES = {
            "left_eye": 33,   # 左眼内角? (或者需要平均几个点?) 常见的可能是 133, 173, 157, 158, 159, 160, 161, 246
            "right_eye": 263, # 右眼内角? (或者需要平均几个点?) 常见的可能是 362, 398, 384, 385, 386, 387, 388, 466
            "nose": 1,      # 鼻尖
            "mouth_left": 61, # 左嘴角
            "mouth_right": 291 # 右嘴角
        }
        # 可能需要调整上述索引以获得最佳效果，或使用眼睛瞳孔点：
        # Left eye pupil: 468, Right eye pupil: 473 (如果 refine_landmarks=True)


    def detect_landmarks(self, img_rgb: np.ndarray) -> list | None:
        """检测图像中的人脸关键点"""
        # 原始图像尺寸
        original_h, original_w = img_rgb.shape[:2]
        
        # 检查图像是否过大，如果是则缩小
        max_dim = 1024  # 最大尺寸阈值
        scale_factor = 1.0
        
        if original_h > max_dim or original_w > max_dim:
            scale_factor = min(max_dim / original_h, max_dim / original_w)
            new_h, new_w = int(original_h * scale_factor), int(original_w * scale_factor)
            img_rgb_resized = cv2.resize(img_rgb, (new_w, new_h))
            logger.info(f"缩小图像进行检测: {original_h}x{original_w} -> {new_h}x{new_w}")
            img_rgb_for_detection = img_rgb_resized
        else:
            img_rgb_for_detection = img_rgb
        
        # 在适当尺寸的图像上运行MediaPipe
        results = self.face_mesh.process(img_rgb_for_detection)
        if not results.multi_face_landmarks:
            return None

        # 只返回第一个检测到的人脸的关键点
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        
        # 检测后的图像尺寸
        detect_h, detect_w = img_rgb_for_detection.shape[:2]
        
        for landmark in face_landmarks.landmark:
            # 先获取相对于检测图像的像素坐标
            px, py = landmark.x * detect_w, landmark.y * detect_h
            
            # 如果进行了缩放，将坐标映射回原始图像
            if scale_factor != 1.0:
                px, py = px / scale_factor, py / scale_factor
                
            landmarks.append((px, py))

        if len(landmarks) != 468 and len(landmarks) != 478:
            logger.warning(f"Expected 468 or 478 landmarks, but got {len(landmarks)}")
            if not landmarks:
                return None

        return landmarks  # 返回原始图像尺寸下的坐标

    def align_face_arcface(self, img_bgr: np.ndarray, target_size=(112, 112)) -> np.ndarray | None:
        """
        使用MediaPipe关键点进行ArcFace标准对齐

        Args:
            img_bgr (np.ndarray): BGR格式的输入图像
            target_size (tuple): 目标输出尺寸 (高度, 宽度)

        Returns:
            np.ndarray | None: 对齐后的人脸图像 (BGR), 如果失败则返回None
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]
        all_landmarks_px = self.detect_landmarks(img_rgb)

        if all_landmarks_px is None:
            logger.warning("No landmarks detected by MediaPipe.")
            return None

        # --- 提取用于ArcFace对齐的5个关键点 ---
        # !!! 这里的索引选择非常关键，需要仔细验证 !!!
        # 使用像素坐标
        try:
            # 使用之前定义的索引
            # lm_left_eye = all_landmarks_px[self.ARCFAE_LMK_INDICES["left_eye"]]
            # lm_right_eye = all_landmarks_px[self.ARCFAE_LMK_INDICES["right_eye"]]
            # lm_nose = all_landmarks_px[self.ARCFAE_LMK_INDICES["nose"]]
            # lm_mouth_left = all_landmarks_px[self.ARCFAE_LMK_INDICES["mouth_left"]]
            # lm_mouth_right = all_landmarks_px[self.ARCFAE_LMK_INDICES["mouth_right"]]

            # 另一种常见的5点选择（需要验证索引的准确性！）
            # 例如，基于 https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py 的思路但不直接用它的点
            # 尝试取眼睛瞳孔（如果可用且准确）或眼睛中心
            # 瞳孔点（如果 refine_landmarks=True）
            if len(all_landmarks_px) > 468: # 假设 refine_landmarks=True 且返回了虹膜点
                 left_eye_idx = 468 + 5 -1 # 左虹膜中心 approx
                 right_eye_idx = 473 + 5 -1 # 右虹膜中心 approx
                 lm_left_eye = all_landmarks_px[left_eye_idx]
                 lm_right_eye = all_landmarks_px[right_eye_idx]
            else: # 否则取眼睛的某个稳定点，例如眼角
                lm_left_eye = all_landmarks_px[33] # 左眼内角?
                lm_right_eye = all_landmarks_px[263]# 右眼内角?

            lm_nose = all_landmarks_px[1]      # 鼻尖
            lm_mouth_left = all_landmarks_px[61] # 左嘴角
            lm_mouth_right = all_landmarks_px[291]# 右嘴角

            dst = np.array([
                lm_left_eye, lm_right_eye, lm_nose, lm_mouth_left, lm_mouth_right
            ], dtype=np.float32)

        except IndexError as e:
            logger.error(f"Failed to get required landmarks using indices: {e}. Landmark count: {len(all_landmarks_px)}")
            return None


        # --- ArcFace 参考点 (与之前代码相同) ---
        src = np.array([
            [30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
            [33.5493, 92.3655], [62.7299, 92.2041]
        ], dtype=np.float32)

        # 根据目标尺寸缩放参考点
        src[:, 0] *= target_size[1] / 112.0
        src[:, 1] *= target_size[0] / 112.0

        # 计算变换矩阵
        try:
             # 使用 estimateAffinePartial2D 可能更鲁棒
            # M = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0] # LMEDS 更鲁棒?
             # 或者使用 getAffineTransform (如果只有3对点，但我们有5对，用 estimate更好)
             # 考虑到 dst 可能因为检测误差不完全符合仿射变换，estimateAffinePartial2D 通常更好
            tform = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC)[0]
            if tform is None:
                 logger.warning("Estimate Affine Transform failed.")
                 # Fallback using estimateAffine2D?
                 tform = cv2.estimateAffine2D(dst, src)[0]
                 if tform is None:
                     logger.error("Estimate Affine Transform failed completely.")
                     return None
            M = tform

        except cv2.error as e:
             logger.error(f"OpenCV error during Affine Transform estimation: {e}")
             return None


        # 应用变换
        aligned_face = cv2.warpAffine(img_bgr, M, (target_size[1], target_size[0]), borderValue=0.0) # (width, height) for warpAffine size

        return aligned_face

    def close(self):
        """关闭MediaPipe Face Mesh资源"""
        self.face_mesh.close()


# --- 便捷函数，现在使用 MediaPipe ---
def detect_align_face(frame_bgr, target_size=(112, 112), return_landmarks=False):
    """
    使用MediaPipe检测和对齐图像中的人脸（便捷函数）

    Args:
        frame_bgr (numpy.ndarray): BGR格式的输入图像
        target_size (tuple): 目标输出尺寸 (高度, 宽度)
        return_landmarks (bool): 是否同时返回所选的5个关键点

    Returns:
        tuple: (aligned_face, five_landmarks) 如果return_landmarks=True
               aligned_face 如果return_landmarks=False
               如果没有检测或对齐失败，则返回(None, None)或None
    """
    # 使用单例模式确保只创建一个实例
    if not hasattr(detect_align_face, "aligner_instance"):
        logger.info("Initializing MediaPipe Face Aligner...")
        detect_align_face.aligner_instance = MediapipeFaceAligner()
        logger.info("MediaPipe Face Aligner initialized.")

    aligner = detect_align_face.aligner_instance

    # 执行对齐（内部会先做检测）
    aligned_face = aligner.align_face_arcface(frame_bgr, target_size)

    if aligned_face is None:
        return (None, None) if return_landmarks else None

    if return_landmarks:
        # 如果需要返回关键点，需要再次检测或修改 align_face_arcface 返回关键点
        # 为了简单起见，我们再次检测（效率稍低）
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        all_landmarks_px = aligner.detect_landmarks(img_rgb)
        if all_landmarks_px is None:
             five_landmarks = None
        else:
             # 提取用于对齐的5个点（代码与 align_face_arcface 中一致）
            try:
                if len(all_landmarks_px) > 468:
                    left_eye_idx, right_eye_idx = 468 + 5 -1, 473 + 5 -1
                    lm_left_eye = all_landmarks_px[left_eye_idx]
                    lm_right_eye = all_landmarks_px[right_eye_idx]
                else:
                    lm_left_eye = all_landmarks_px[33]
                    lm_right_eye = all_landmarks_px[263]
                lm_nose = all_landmarks_px[1]
                lm_mouth_left = all_landmarks_px[61]
                lm_mouth_right = all_landmarks_px[291]
                five_landmarks = np.array([
                    lm_left_eye, lm_right_eye, lm_nose, lm_mouth_left, lm_mouth_right
                ], dtype=np.float32)
            except IndexError:
                five_landmarks = None

        return aligned_face, five_landmarks
    else:
        return aligned_face


# --- Example Usage / Testing ---
if __name__ == '__main__':
    print("--- Testing MediaPipe Face Alignment ---")
    # --- !!! CHANGE THIS to your test image path !!! ---
    test_image_path = "/root/HiFiVFS/data/test_image.jpg"

    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at '{test_image_path}'")
    else:
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"ERROR: Failed to load image '{test_image_path}'")
        else:
            print(f"Original image shape: {img.shape}")

            # --- Test ArcFace Alignment ---
            print("\nTesting ArcFace Alignment (112x112)...")
            aligned_face_arc, landmarks = detect_align_face(img, target_size=(112, 112), return_landmarks=True)

            if aligned_face_arc is not None:
                print("ArcFace alignment successful.")
                print(f"Aligned face shape: {aligned_face_arc.shape}")
                cv2.imwrite("aligned_face_arcface.jpg", aligned_face_arc)
                print("Saved aligned face to 'aligned_face_arcface.jpg'")
                if landmarks is not None:
                    print(f"Returned 5 landmarks shape: {landmarks.shape}")
                    # Optional: Draw landmarks on original image for verification
                    img_draw = img.copy()
                    for (x, y) in landmarks.astype(int):
                         cv2.circle(img_draw, (x, y), 2, (0, 255, 0), -1)
                    cv2.imwrite("original_with_landmarks.jpg", img_draw)
                    print("Saved original image with 5 landmarks to 'original_with_landmarks.jpg'")

            else:
                print("ArcFace alignment failed.")

            # --- Optional: Test FFHQ-like Alignment (if needed) ---
            # print("\nTesting FFHQ-like Alignment (256x256)...")
            # aligned_face_ffhq = detect_align_face(img, target_size=(256, 256), mode='ffhq') # Need to implement 'ffhq' mode using mediapipe landmarks if desired
            # if aligned_face_ffhq is not None:
            #     print("FFHQ-like alignment successful.")
            #     print(f"Aligned face shape: {aligned_face_ffhq.shape}")
            #     cv2.imwrite("aligned_face_ffhq.jpg", aligned_face_ffhq)
            #     print("Saved aligned face to 'aligned_face_ffhq.jpg'")
            # else:
            #     print("FFHQ-like alignment failed.")

    print("\n--- MediaPipe Test Finished ---")