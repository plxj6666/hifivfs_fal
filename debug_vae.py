import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoencoderKL
from tqdm import tqdm
import time

# 导入所需模块
from hifivfs_fal.utils.face_recognition import DeepFaceRecognizer
from hifivfs_fal.utils.detect_align_face import detect_align_face

# 配置日志
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_video")

def process_video_frames(video_path, output_dir, frame_interval=10, max_frames=20):
    """
    处理视频帧并测试各个处理环节的成功率
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出结果保存目录
        frame_interval: 每隔多少帧提取一帧
        max_frames: 最多提取多少帧
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return
    
    # 视频信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧")
    
    # 加载VAE模型
    logger.info("加载VAE模型...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")
    vae_scale_factor = 0.18215
    
    # 初始化人脸识别器
    logger.info("初始化人脸识别器...")
    face_recognizer = DeepFaceRecognizer(model_name="Facenet512", detector_backend="retinaface")
    if not face_recognizer.initialized:
        logger.error("人脸识别器初始化失败")
        return
    
    # 创建结果表格HTML
    html_content = """
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid black; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            img { max-width: 200px; max-height: 200px; }
            .success { color: green; }
            .failure { color: red; }
        </style>
    </head>
    <body>
        <h1>视频帧人脸检测与VAE处理测试</h1>
        <table>
            <tr>
                <th>帧索引</th>
                <th>原始帧</th>
                <th>MediaPipe对齐</th>
                <th>DeepFace特征</th>
                <th>VAE尺寸调整(640x640)</th>
                <th>VAE编码-解码</th>
                <th>解码后帧检测</th>
                <th>解码后对齐</th>
                <th>解码后特征</th>
                <th>不同尺寸测试(224x224)</th>
            </tr>
    """
    
    # 统计成功率
    stats = {
        "total_frames": 0,
        "mediapipe_align_success": 0,
        "deepface_feature_success": 0,
        "vae_decoded_align_success": 0,
        "vae_decoded_feature_success": 0,
        "small_size_align_success": 0
    }
    
    # 处理视频帧
    frame_index = 0
    processed_count = 0
    
    with tqdm(total=min(max_frames, frame_count // frame_interval + 1)) as pbar:
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % frame_interval == 0:
                logger.info(f"处理帧 {frame_index}")
                stats["total_frames"] += 1
                
                # 保存原始帧
                original_path = output_path / f"frame_{frame_index:04d}_original.jpg"
                cv2.imwrite(str(original_path), frame)
                
                # 1. 测试MediaPipe人脸对齐
                mediapipe_start = time.time()
                aligned_face = detect_align_face(frame, target_size=(112, 112))
                mediapipe_time = time.time() - mediapipe_start
                
                mediapipe_status = "失败"
                aligned_path = None
                if aligned_face is not None:
                    stats["mediapipe_align_success"] += 1
                    mediapipe_status = "成功"
                    aligned_path = output_path / f"frame_{frame_index:04d}_aligned.jpg"
                    cv2.imwrite(str(aligned_path), aligned_face)
                
                # 2. 测试DeepFace特征提取
                feature_start = time.time()
                features = face_recognizer.extract_identity(frame)
                feature_time = time.time() - feature_start
                
                feature_status = "失败"
                if features is not None and not np.all(features == 0):
                    stats["deepface_feature_success"] += 1
                    feature_status = "成功"
                
                # 3. VAE编码-解码测试
                # 调整大小到640x640
                frame_resized = cv2.resize(frame, (640, 640))
                resized_path = output_path / f"frame_{frame_index:04d}_resized.jpg"
                cv2.imwrite(str(resized_path), frame_resized)
                
                # 转换为张量
                img_tensor = torch.from_numpy(frame_resized).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to("cuda")
                
                # VAE编码-解码
                with torch.no_grad():
                    latents = vae.encode(img_tensor).latent_dist.sample() * vae_scale_factor
                    logger.info(f"潜在表示形状: {latents.shape}")  # 应该是 [1, 4, 80, 80]
                    
                    # 解码回像素空间
                    decoded = vae.decode(latents / vae_scale_factor).sample
                    decoded_img = decoded[0].permute(1, 2, 0).cpu().numpy()
                    decoded_img = np.clip(decoded_img, 0, 1)
                    decoded_img = (decoded_img * 255).astype(np.uint8)
                
                # 保存解码后的图像
                decoded_path = output_path / f"frame_{frame_index:04d}_decoded.jpg"
                cv2.imwrite(str(decoded_path), cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR))
                
                # 4. 解码后的帧检测测试
                decoded_bgr = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR)
                
                # 检测是否有面部特征
                face_present = "是"
                has_face_features = False
                if np.mean(cv2.Canny(decoded_bgr, 100, 200)) > 5.0:  # 简单边缘检测
                    has_face_features = True
                    face_present = "是"
                    
                # 对解码帧进行人脸对齐
                decoded_aligned = detect_align_face(decoded_bgr, target_size=(112, 112))
                decoded_align_status = "失败"
                decoded_aligned_path = None
                if decoded_aligned is not None:
                    stats["vae_decoded_align_success"] += 1
                    decoded_align_status = "成功"
                    decoded_aligned_path = output_path / f"frame_{frame_index:04d}_decoded_aligned.jpg"
                    cv2.imwrite(str(decoded_aligned_path), decoded_aligned)
                
                # 解码帧特征提取
                decoded_features = face_recognizer.extract_identity(decoded_bgr)
                decoded_feature_status = "失败"
                if decoded_features is not None and not np.all(decoded_features == 0):
                    stats["vae_decoded_feature_success"] += 1
                    decoded_feature_status = "成功"
                
                # 5. 不同尺寸测试 - 224x224
                small_frame = cv2.resize(frame, (224, 224))
                small_path = output_path / f"frame_{frame_index:04d}_small.jpg"
                cv2.imwrite(str(small_path), small_frame)
                
                small_aligned = detect_align_face(small_frame, target_size=(112, 112))
                small_align_status = "失败"
                if small_aligned is not None:
                    stats["small_size_align_success"] += 1
                    small_align_status = "成功"
                    small_aligned_path = output_path / f"frame_{frame_index:04d}_small_aligned.jpg"
                    cv2.imwrite(str(small_aligned_path), small_aligned)
                
                # 添加到HTML表格
                html_content += f"""
                <tr>
                    <td>{frame_index}</td>
                    <td><img src="{original_path.name}" /></td>
                    <td class="{'success' if aligned_face is not None else 'failure'}">
                        {mediapipe_status}<br/>({mediapipe_time:.2f}秒)
                        {f'<br/><img src="{aligned_path.name}" />' if aligned_path else ''}
                    </td>
                    <td class="{'success' if features is not None and not np.all(features == 0) else 'failure'}">
                        {feature_status}<br/>({feature_time:.2f}秒)
                        {f'<br/>前5个值: {features[:5]}' if features is not None and not np.all(features == 0) else ''}
                    </td>
                    <td><img src="{resized_path.name}" /></td>
                    <td><img src="{decoded_path.name}" /></td>
                    <td>{face_present}</td>
                    <td class="{'success' if decoded_aligned is not None else 'failure'}">
                        {decoded_align_status}
                        {f'<br/><img src="{decoded_aligned_path.name}" />' if decoded_aligned_path else ''}
                    </td>
                    <td class="{'success' if decoded_features is not None and not np.all(decoded_features == 0) else 'failure'}">
                        {decoded_feature_status}
                    </td>
                    <td class="{'success' if small_aligned is not None else 'failure'}">
                        {small_align_status}
                        <br/><img src="{small_path.name}" />
                    </td>
                </tr>
                """
                
                processed_count += 1
                pbar.update(1)
            
            frame_index += 1
            if frame_index >= frame_count:
                break
    
    # 关闭视频
    cap.release()
    
    # 计算成功率
    for key in stats:
        if key != "total_frames" and stats["total_frames"] > 0:
            success_rate = stats[key] / stats["total_frames"] * 100
            logger.info(f"{key}: {stats[key]}/{stats['total_frames']} = {success_rate:.2f}%")
    
    # 完成HTML表格并保存
    html_content += """
        </table>
        <h2>统计结果</h2>
        <table>
            <tr>
                <th>测试项目</th>
                <th>成功次数</th>
                <th>总帧数</th>
                <th>成功率</th>
            </tr>
    """
    
    for key in stats:
        if key != "total_frames":
            success_rate = stats[key] / stats["total_frames"] * 100
            html_content += f"""
            <tr>
                <td>{key}</td>
                <td>{stats[key]}</td>
                <td>{stats["total_frames"]}</td>
                <td>{success_rate:.2f}%</td>
            </tr>
            """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path / "results.html", "w") as f:
        f.write(html_content)
    
    logger.info(f"结果保存到 {output_path / 'results.html'}")
    
    # 生成总结报告
    summary = f"""
    视频处理总结报告
    ================
    视频路径: {video_path}
    分辨率: {width}x{height}
    帧率: {fps}fps
    总帧数: {frame_count}
    
    测试帧数: {stats['total_frames']}
    
    成功率:
    - MediaPipe人脸对齐: {stats['mediapipe_align_success']}/{stats['total_frames']} = {stats['mediapipe_align_success']/stats['total_frames']*100:.2f}%
    - DeepFace特征提取: {stats['deepface_feature_success']}/{stats['total_frames']} = {stats['deepface_feature_success']/stats['total_frames']*100:.2f}%
    - VAE解码后对齐: {stats['vae_decoded_align_success']}/{stats['total_frames']} = {stats['vae_decoded_align_success']/stats['total_frames']*100:.2f}%
    - VAE解码后特征: {stats['vae_decoded_feature_success']}/{stats['total_frames']} = {stats['vae_decoded_feature_success']/stats['total_frames']*100:.2f}%
    - 小尺寸(224x224)对齐: {stats['small_size_align_success']}/{stats['total_frames']} = {stats['small_size_align_success']/stats['total_frames']*100:.2f}%
    
    主要发现:
    1. 原始帧人脸检测成功率高于VAE解码后的帧
    2. 小尺寸(224x224)下的检测成功率与原始尺寸相比: {'更高' if stats['small_size_align_success'] > stats['mediapipe_align_success'] else '更低'}
    3. VAE解码可能导致面部细节丢失，影响人脸检测
    
    建议:
    1. {'降低人脸检测置信度阈值' if stats['mediapipe_align_success'] < stats['total_frames'] else '保持当前人脸检测参数'}
    2. {'考虑使用更小的输入尺寸(224x224)' if stats['small_size_align_success'] > stats['mediapipe_align_success'] else '保持当前输入尺寸'}
    3. 确保特征提取错误不会中断训练流程
    """
    
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary)
    
    logger.info(f"总结报告保存到 {output_path / 'summary.txt'}")


if __name__ == "__main__":
    # 设置测试视频路径
    video_path = "/root/HiFiVFS/data/VOX2/dev/id00020/_djPgQPgjPs/00229.mp4"
    output_dir = "/root/HiFiVFS/debug_results"
    
    process_video_frames(
        video_path=video_path,
        output_dir=output_dir,
        frame_interval=30,  # 每30帧提取一帧
        max_frames=10       # 最多处理10帧
    )