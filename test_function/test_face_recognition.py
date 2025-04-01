# 创建 test_face_recognition.py
import cv2
import numpy as np
import os
from hifivfs_fal.utils.detect_align_face import detect_align_face
from hifivfs_fal.utils.face_recognition import FaceRecognizer

# 初始化人脸识别器
face_recognizer = FaceRecognizer()

# 加载测试图像
image_path1 = "/root/HiFiVFS/data/test_image.jpg"
image_path2 = "/root/HiFiVFS/data/test_image2.jpg"  # 理想情况下，同一人的另一张照片

# 处理第一张图像
img1 = cv2.imread(image_path1)
aligned_face1 = detect_align_face(img1, target_size=(112, 112))
if aligned_face1 is None:
    print("图像1人脸对齐失败")
    exit(1)

embedding1 = face_recognizer.get_embedding(aligned_face1)
print(f"身份特征1形状: {embedding1.shape}")  # 应该是(512,)

# 如果有第二张图像，计算相似度
if image_path2 and os.path.exists(image_path2):
    img2 = cv2.imread(image_path2)
    aligned_face2 = detect_align_face(img2, target_size=(112, 112))
    if aligned_face2 is not None:
        embedding2 = face_recognizer.get_embedding(aligned_face2)
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"两张图像的身份相似度: {similarity:.4f}")
        # 相似度>0.5通常表示同一个人，>0.7表示高度相似