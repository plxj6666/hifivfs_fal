import tensorflow as tf
from deepface.modules import modeling
import numpy as np
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelExplorer")

def explore_facenet512():
    """探索Facenet512模型的层结构，找到适合作为fdid的层"""
    logger.info("正在加载Facenet512模型...")
    
    # 加载模型
    model = modeling.build_model(task="facial_recognition", model_name="Facenet512")
    keras_model = model.model
    
    logger.info(f"模型加载成功，类型: {type(keras_model)}")
    
    # 获取所有层
    all_layers = keras_model.layers
    logger.info(f"模型共有 {len(all_layers)} 个层")
    
    # 分析所有层
    res_blocks = []
    potential_fdid_layers = []
    
    # 创建一个假的输入来测试各层输出
    dummy_input = np.zeros((1, 160, 160, 3), dtype=np.float32)
    
    # 首先查找所有可能的Res-Block相关层
    for i, layer in enumerate(all_layers):
        layer_class = layer.__class__.__name__
        layer_name = layer.name
        
        # 检查是否为Res-Block或可能的fdid层
        if "Block" in layer_name or "Mixed" in layer_name or "add" in layer_name:
            try:
                # 创建提取特定层输出的模型
                temp_model = tf.keras.Model(inputs=keras_model.input, outputs=layer.output)
                
                # 运行模型获取输出
                output = temp_model(dummy_input, training=False)
                
                # 获取实际输出形状
                if isinstance(output, tf.Tensor):
                    actual_shape = output.shape
                else:
                    actual_shape = output.numpy().shape
                
                logger.info(f"[{i}] {layer_name} ({layer_class}) - 输出形状: {actual_shape}")
                
                # 记录Res-Block层
                if "Block" in layer_name:
                    res_blocks.append((i, layer_name, layer_class, actual_shape))
                
                # 检查是否可能是我们需要的fdid层 (特征图形状接近7x7)
                if len(actual_shape) == 4:  # (batch, h, w, c)
                    h, w, c = actual_shape[1], actual_shape[2], actual_shape[3]
                    tokens = h * w
                    
                    # 论文提到fdid应该有49个token (大约是7x7特征图)
                    if 9 <= tokens <= 100:  # 允许一定范围
                        potential_fdid_layers.append((i, layer_name, layer_class, actual_shape))
            except Exception as e:
                logger.error(f"分析层 '{layer_name}' 时出错: {e}")
    
    # 打印潜在的fdid层
    logger.info("\n--- 潜在的fdid层候选 ---")
    for i, name, cls, shape in potential_fdid_layers:
        h, w, c = shape[1], shape[2], shape[3]
        tokens = h * w
        logger.info(f"[{i}] {name} ({cls}) - 形状: {shape} - 空间大小: ({h}x{w}) - Tokens: {tokens} - 通道数: {c}")
    
    # 推荐最佳的fdid层
    logger.info("\n===== 推荐的fdid层 =====")
    if potential_fdid_layers:
        # 优先选择Block8相关层，因为它们通常是最后的Res-Block
        block8_layers = [l for l in potential_fdid_layers if "Block8" in l[1]]
        if block8_layers:
            best = block8_layers[-1]  # 最后一个Block8层
        else:
            best = potential_fdid_layers[-1]  # 否则取最后一个潜在层
            
        i, name, cls, shape = best
        h, w, c = shape[1], shape[2], shape[3]
        tokens = h * w
        
        logger.info(f"推荐使用: {name} ({cls})")
        logger.info(f"  - 输出形状: {shape}")
        logger.info(f"  - 空间大小: {h}x{w} = {tokens} tokens")
        logger.info(f"  - 通道数: {c}")
        logger.info(f"  - 加载代码: target_layer = keras_model.get_layer(name='{name}')")
    else:
        logger.info("未找到合适的fdid层")

if __name__ == "__main__":
    explore_facenet512()