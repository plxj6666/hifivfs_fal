# 测试DeepFace人脸识别功能
import cv2
import numpy as np
import os
import time
import logging
from hifivfs_fal.utils.face_recognition import DeepFaceRecognizer

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("face_recognition_test")

# 创建输出目录
output_dir = '/root/HiFiVFS/data/face_test_output'
os.makedirs(output_dir, exist_ok=True)

# 初始化人脸识别器(使用DeepFace)
print("正在初始化DeepFace人脸识别器...")
try:
    # 尝试使用DeepFace
    recognizer = DeepFaceRecognizer(model_name="Facenet512", detector_backend="retinaface")
    
    if not recognizer.initialized:
        print("DeepFace初始化失败，切换到备用识别器")
    else:
        print("DeepFace初始化成功！")
except Exception as e:
    print(f"初始化错误: {e}")

# 读取测试图像
print("\n开始读取测试图像...")
img1_path = '/root/HiFiVFS/data/test1.png'
img2_path = '/root/HiFiVFS/data/mayun2.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None:
    print(f"无法读取图像: {img1_path}")
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)  # 创建空白图像
else:
    print(f"成功读取图像1: 尺寸={img1.shape}")
    
if img2 is None:
    print(f"无法读取图像: {img2_path}")
    img2 = np.zeros((300, 300, 3), dtype=np.uint8)  # 创建空白图像
else:
    print(f"成功读取图像2: 尺寸={img2.shape}")

# 检测和对齐人脸
print("\n正在进行人脸检测和对齐...")
aligned1 = recognizer.detect_and_align(img1)
aligned2 = recognizer.detect_and_align(img2)

# 保存对齐结果
if aligned1 is not None:
    print("成功对齐图像1的人脸")
    cv2.imwrite(os.path.join(output_dir, 'aligned_mayun1.jpg'), aligned1)
else:
    print("无法对齐图像1的人脸")
    
if aligned2 is not None:
    print("成功对齐图像2的人脸")
    cv2.imwrite(os.path.join(output_dir, 'aligned_mayun2.jpg'), aligned2)
else:
    print("无法对齐图像2的人脸")

# 提取身份特征
print("\n正在提取身份特征...")
start_time = time.time()
emb1 = recognizer.extract_identity(img1)
time1 = time.time() - start_time
print(f"提取特征1耗时: {time1:.2f}秒")

start_time = time.time()
emb2 = recognizer.extract_identity(img2)
time2 = time.time() - start_time
print(f"提取特征2耗时: {time2:.2f}秒")

# 计算相似度
if emb1 is not None and emb2 is not None:
    print("\n特征提取成功，计算相似度...")
    similarity = recognizer.compute_similarity(emb1, emb2)
    print(f"相似度: {similarity:.4f}")
    
    # 判断是否为同一人
    threshold = 0.5
    is_same = similarity > threshold
    print(f"是否为同一人(阈值={threshold}): {is_same}")
    
    # 打印特征向量信息
    print(f"\n特征1形状: {emb1.shape}, 范数: {np.linalg.norm(emb1):.4f}")
    print(f"特征1前5个值: {emb1[:5]}")
    
    print(f"\n特征2形状: {emb2.shape}, 范数: {np.linalg.norm(emb2):.4f}")
    print(f"特征2前5个值: {emb2[:5]}")
else:
    print("\n特征提取失败")
    if emb1 is None:
        print("图像1特征提取失败")
    if emb2 is None:
        print("图像2特征提取失败")

print("\n测试完成!")
# 打印模型类型和参数
print(f"使用的识别器类型: {type(recognizer).__name__}")
if hasattr(recognizer, 'model_name'):
    print(f"模型名称: {recognizer.model_name}")
if hasattr(recognizer, 'detector_backend'):
    print(f"检测器后端: {recognizer.detector_backend}")
print(f"特征维度: {recognizer.embedding_size}")