import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("test_face_parsing")

# 添加项目根目录到sys.path
project_root = Path("/root/HiFiVFS")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 创建必要的目录
os.makedirs(project_root / "models" / "face_parsing", exist_ok=True)
os.makedirs(project_root / "tests" / "results", exist_ok=True)

# 添加一个下载方法，当模型不存在时自动下载
def download_model(self, save_path):
    """下载BiSeNet模型文件"""
    import gdown
    
    logger.info(f"正在下载BiSeNet模型到 {save_path}...")
    # 这是一个公开的BiSeNet模型文件链接，您可能需要替换为实际可用的链接
    url = "https://drive.google.com/uc?id=1YkvIBHDPX1UPh6DR-YKKKXlZzXqbSjHT"
    gdown.download(url, str(save_path), quiet=False)
    
    if os.path.exists(save_path):
        logger.info(f"模型下载成功: {save_path}")
        return True
    else:
        logger.error(f"模型下载失败")
        return False
    
def test_face_parsing():
    """测试面部解析功能"""
    # 导入FaceParser类(在测试函数内导入，确保路径已添加)
    try:
        from fal_dil.utils.face_parsing import FaceParser
    except ImportError as e:
        logger.error(f"无法导入FaceParser: {e}")
        # 检查是否需要创建face_parsing.py文件
        face_parsing_path = project_root / "fal_dil" / "utils" / "face_parsing.py"
        if not face_parsing_path.exists():
            logger.error(f"文件 {face_parsing_path} 不存在，请先创建该文件")
        return False
    
    # 检查ONNX模型是否存在
    model_path = project_root / "models" / "face_parsing" / "bisenet.onnx"
    if not model_path.exists():
        logger.warning(f"模型文件 {model_path} 不存在。尝试初始化FaceParser时将提示下载或指定模型路径。")
    
    # 创建FaceParser实例，允许使用默认路径
    try:
        parser = FaceParser()
        logger.info("成功创建FaceParser实例")
    except Exception as e:
        logger.error(f"无法初始化FaceParser: {e}")
        return False
    
    # 找到一张包含人脸的测试图像
    test_image_path = project_root / "tests" / "test_data" / "face_sample.jpg"
    if not test_image_path.exists():
        # 如果没有特定的测试图像，尝试找到一个视频并提取第一帧
        video_paths = list(project_root.glob("data/**/*.mp4")) + list(project_root.glob("data/**/*.avi"))
        if not video_paths:
            logger.error("未找到测试图像或视频。请提供包含人脸的测试图像。")
            # 如果没有图像，尝试生成简单的测试图像
            test_image = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.circle(test_image, (256, 256), 200, (255, 255, 255), -1)  # 画一个简单的圆形
            cv2.circle(test_image, (200, 200), 30, (0, 0, 0), -1)  # 左眼
            cv2.circle(test_image, (300, 200), 30, (0, 0, 0), -1)  # 右眼
            cv2.ellipse(test_image, (250, 300), (50, 20), 0, 0, 180, (0, 0, 0), -1)  # 嘴巴
            logger.info("已创建简单的测试图像")
        else:
            # 从视频中读取第一帧作为测试图像
            cap = cv2.VideoCapture(str(video_paths[0]))
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"无法从视频{video_paths[0]}读取帧。")
                return False
            
            test_image = frame
            logger.info(f"已从视频 {video_paths[0]} 提取第一帧作为测试图像")
    else:
        # 加载测试图像
        test_image = cv2.imread(str(test_image_path))
        logger.info(f"已加载测试图像 {test_image_path}")
    
    # 确保测试图像存在
    if test_image is None:
        logger.error("无法加载测试图像。")
        return False
    
    # 调用parse方法生成面部遮罩
    try:
        face_mask = parser.parse(test_image)
        logger.info(f"成功生成面部遮罩，形状: {face_mask.shape}")
    except Exception as e:
        logger.error(f"无法生成面部遮罩: {e}")
        return False
    
    # 可视化结果
    try:
        plt.figure(figsize=(12, 6))
        
        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis("off")
        
        # 显示生成的面部遮罩
        plt.subplot(1, 2, 2)
        plt.imshow(face_mask[:, :, 0], cmap='gray')
        plt.title("面部遮罩")
        plt.axis("off")
        
        # 保存结果
        result_path = project_root / "tests" / "results" / "face_mask_result.png"
        plt.savefig(str(result_path))
        logger.info(f"结果已保存到 {result_path}")
        
        # 同时保存原始图像和面部遮罩作为单独的文件
        cv2.imwrite(str(project_root / "tests" / "results" / "original_image.png"), test_image)
        cv2.imwrite(str(project_root / "tests" / "results" / "face_mask.png"), face_mask * 255)
        
        plt.show()
        return True
    except Exception as e:
        logger.error(f"可视化结果时出错: {e}")
        return False

if __name__ == "__main__":
    success = test_face_parsing()
    if success:
        print("✅ 测试成功完成!")
    else:
        print("❌ 测试失败，请查看日志获取详细信息。")