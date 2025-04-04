# hifivfs_fal/models/decoder.py

import torch
import torch.nn as nn
import logging # 使用 logging
from typing import Optional
import traceback
try:
    # 假设 CrossAttentionBlock 在同一目录下或可导入
    from .blocks import ResBlock, CrossAttentionBlock 
except ImportError:
    # 尝试相对导入
    try:
         from blocks import ResBlock, CrossAttentionBlock
    except ImportError:
         raise ImportError("无法导入 blocks.py 中的 ResBlock 或 CrossAttentionBlock")

logger = logging.getLogger(__name__)

class Decoder(nn.Module):
    """
    解码器，将属性特征 f_attr 解码回 VAE 潜变量空间。
    增加了对详细身份 token (tdid) 的支持，通过 Cross-Attention 注入。
    """
    def __init__(self, 
                 # frid_channels 不再需要，因为 projector 的输出维度由 unet_cross_attn_dim 决定
                 # frid_channels=1280, 
                 # allow_frid_projection 也不再需要，投影由外部 FridProjector 完成
                 # allow_frid_projection=True, 
                 # **** 新增：tdid 的嵌入维度 ****
                 cross_attention_dim: int = 768, # 对应 tdid 和处理后 frid 的维度
                 # **** 新增：tdid 的 token 数量 ****
                 num_tdid_tokens: int = 9 # 例如 3x3 特征图
                 ):
        """
        初始化 Decoder。

        Args:
            cross_attention_dim (int): 交叉注意力期望的 Key/Value 嵌入维度。
                                       tdid 和 frid (投影后) 都应匹配此维度。
            num_tdid_tokens (int): tdid token 的数量 (例如 H * W)。
        """
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tdid_tokens = num_tdid_tokens
        self.total_context_tokens = 1 + num_tdid_tokens # 1 for frid + num_tdid_tokens

        # 移除内部的 frid_projection，假设 f_rid 输入时已经是 (B, 1, cross_attention_dim)
        # self.frid_projection = ... 

        # --- Stage 1 (Input: f_attr 1280x20x20, frid_context, tdid_context) ---
        ca1_heads = 16 # 可以根据 cross_attention_dim 调整
        self.stage1_res1 = ResBlock(1280)
        # **** 修改 CrossAttentionBlock 以接收拼接后的上下文 ****
        # kv_channels 现在是 cross_attention_dim
        self.stage1_ca1 = CrossAttentionBlock(query_channels=1280, kv_channels=self.cross_attention_dim, num_heads=ca1_heads)
        self.stage1_res2 = ResBlock(1280)
        self.stage1_ca2 = CrossAttentionBlock(query_channels=1280, kv_channels=self.cross_attention_dim, num_heads=ca1_heads)
        # Output: 1280x20x20

        # --- Upsample 1 (20x20 -> 40x40) ---
        self.upsample1 = nn.ConvTranspose2d(1280, 640, kernel_size=4, stride=2, padding=1)

        # --- Stage 2 (Input: 640x40x40, frid_context, tdid_context) ---
        ca2_heads = 8 # 可以根据 cross_attention_dim 调整
        self.stage2_res1 = ResBlock(640)
        # **** 修改 CrossAttentionBlock ****
        self.stage2_ca1 = CrossAttentionBlock(query_channels=640, kv_channels=self.cross_attention_dim, num_heads=ca2_heads)
        self.stage2_res2 = ResBlock(640)
        self.stage2_ca2 = CrossAttentionBlock(query_channels=640, kv_channels=self.cross_attention_dim, num_heads=ca2_heads)
        # Output: 640x40x40

        # --- Upsample 2 (40x40 -> 80x80) ---
        self.upsample2 = nn.ConvTranspose2d(640, 320, kernel_size=4, stride=2, padding=1)

        # --- Stage 3 (Input: 320x80x80) ---
        # 最后一阶段通常不加 Attention，专注于空间重建
        self.stage3_res1 = ResBlock(320)
        self.stage3_res2 = ResBlock(320)
        # Output: 320x80x80

        # --- Final Convolution ---
        # 输出通道数为 4，对应 VAE latent 的通道数 (通常是 4)
        self.final_conv = nn.Conv2d(320, 4, kernel_size=3, stride=1, padding=1) # 使用 3x3 卷积可能比 1x1 好

        logger.info(f"Decoder 初始化: CrossAttn Dim={cross_attention_dim}, Total Context Tokens={self.total_context_tokens}")

    # **** 修改 forward 方法以接收 tdid_context ****
    def forward(self, 
                f_attr: torch.Tensor, 
                frid_context: Optional[torch.Tensor] = None, # 来自 FridProjector (B, 1, C)
                tdid_context: Optional[torch.Tensor] = None  # 来自 DIT (B, N_tokens, C)
                ) -> torch.Tensor:
        """
        Decoder 前向传播。

        Args:
            f_attr (torch.Tensor): 属性特征图，形状 (B, 1280, H_attr, W_attr)。
            frid_context (Optional[torch.Tensor]): 来自 FridProjector 的全局参考身份上下文，
                                                   形状 (B, 1, cross_attention_dim)。
            tdid_context (Optional[torch.Tensor]): 来自 DIT 的详细身份 token 上下文，
                                                   形状 (B, num_tdid_tokens, cross_attention_dim)。

        Returns:
            torch.Tensor: 重建的潜变量，形状 (B, 4, H_out, W_out)。
        """
        
        # --- 准备交叉注意力上下文 ---
        combined_context = None
        expected_context_dim = self.cross_attention_dim

        # 处理 frid_context
        if frid_context is not None:
            if frid_context.shape[-1] != expected_context_dim:
                 logger.error(f"frid_context 维度错误! 期望 {expected_context_dim}, 收到 {frid_context.shape[-1]}")
                 # 可以选择报错或忽略 frid_context
                 frid_context = None 
            elif frid_context.ndim != 3 or frid_context.shape[1] != 1:
                 logger.error(f"frid_context 形状错误! 期望 (B, 1, {expected_context_dim}), 收到 {frid_context.shape}")
                 frid_context = None

        # 处理 tdid_context
        if tdid_context is not None:
            if tdid_context.shape[-1] != expected_context_dim:
                 logger.error(f"tdid_context 维度错误! 期望 {expected_context_dim}, 收到 {tdid_context.shape[-1]}")
                 tdid_context = None
            elif tdid_context.ndim != 3 or tdid_context.shape[1] != self.num_tdid_tokens:
                 logger.error(f"tdid_context 形状错误! 期望 (B, {self.num_tdid_tokens}, {expected_context_dim}), 收到 {tdid_context.shape}")
                 tdid_context = None
                 
        # 拼接上下文
        if frid_context is not None and tdid_context is not None:
            # 确保在同一设备
            if frid_context.device != tdid_context.device:
                 logger.warning("frid_context 和 tdid_context 不在同一设备，将 tdid 移至 frid 设备")
                 tdid_context = tdid_context.to(frid_context.device)
            # 沿序列长度维度 (dim=1) 拼接
            combined_context = torch.cat((frid_context, tdid_context), dim=1) # (B, 1 + N_tokens, C)
            logger.debug(f"使用拼接后的上下文，形状: {combined_context.shape}")
        elif frid_context is not None:
            combined_context = frid_context
            logger.debug("只使用 frid_context")
        elif tdid_context is not None:
            combined_context = tdid_context
            logger.debug("只使用 tdid_context")
        else:
            # 如果两者都不可用，交叉注意力层将无法工作或需要特殊处理
            logger.warning("没有提供 frid_context 或 tdid_context，交叉注意力将无法使用有效上下文！")
            # 可以选择创建一个零张量或让 CrossAttentionBlock 内部处理 None
            # combined_context = None

        # --- Decoder 网络 ---
        x = f_attr # (B, 1280, 20, 20) - 假设输入是 20x20

        # --- Stage 1 ---
        x = self.stage1_res1(x)
        if combined_context is not None: 
             x = self.stage1_ca1(x, combined_context) # **** 使用拼接后的上下文 ****
        x = self.stage1_res2(x)
        if combined_context is not None:
             x = self.stage1_ca2(x, combined_context) # **** 使用拼接后的上下文 ****
        logger.debug(f"Stage 1 输出形状: {x.shape}") # (B, 1280, 20, 20)

        # Upsample 1
        x = self.upsample1(x) 
        logger.debug(f"Upsample 1 输出形状: {x.shape}") # (B, 640, 40, 40)

        # --- Stage 2 ---
        x = self.stage2_res1(x)
        if combined_context is not None:
             x = self.stage2_ca1(x, combined_context) # **** 使用拼接后的上下文 ****
        x = self.stage2_res2(x)
        if combined_context is not None:
             x = self.stage2_ca2(x, combined_context) # **** 使用拼接后的上下文 ****
        logger.debug(f"Stage 2 输出形状: {x.shape}") # (B, 640, 40, 40)

        # Upsample 2
        x = self.upsample2(x) 
        logger.debug(f"Upsample 2 输出形状: {x.shape}") # (B, 320, 80, 80)

        # Stage 3
        x = self.stage3_res1(x)
        x = self.stage3_res2(x) 
        logger.debug(f"Stage 3 输出形状: {x.shape}") # (B, 320, 80, 80)

        # Final Convolution
        vt_prime = self.final_conv(x) 
        logger.debug(f"Final Conv 输出形状: {vt_prime.shape}") # (B, 4, 80, 80) 

        return vt_prime

# --- 测试代码 (更新以测试 tdid 输入) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # 设置日志级别为 DEBUG
    
    print("\n--- Testing Decoder with DIL Context ---")
    
    # 假设参数
    batch_size = 2
    attr_channels = 1280
    attr_h, attr_w = 20, 20
    
    cross_attn_dim = 768 # UNet/Decoder 交叉注意力维度
    num_tdid_tok = 9     # 3x3 特征图
    
    # 创建假的输入
    dummy_f_attr = torch.randn(batch_size, attr_channels, attr_h, attr_w)
    dummy_frid_context = torch.randn(batch_size, 1, cross_attn_dim) # 来自 FridProjector
    dummy_tdid_context = torch.randn(batch_size, num_tdid_tok, cross_attn_dim) # 来自 DIT
    
    # 初始化 Decoder
    try:
        decoder = Decoder(
            cross_attention_dim=cross_attn_dim, 
            num_tdid_tokens=num_tdid_tok
        )
        print(f"Decoder 模型:\n{decoder}")

        # 测试前向传播
        vt_prime = decoder(dummy_f_attr, dummy_frid_context, dummy_tdid_context)
        
        print(f"\nInput f_attr shape: {dummy_f_attr.shape}")
        print(f"Input frid_context shape: {dummy_frid_context.shape}")
        print(f"Input tdid_context shape: {dummy_tdid_context.shape}")
        print(f"Output V't shape: {vt_prime.shape}") 
        
        # 验证输出形状
        expected_output_shape = (batch_size, 4, 80, 80) # 假设 VAE latent 是 4x80x80
        assert vt_prime.shape == expected_output_shape, f"V't shape mismatch! Expected {expected_output_shape}, Got {vt_prime.shape}"
        print("形状验证通过!")

        # 测试只传入 tdid
        print("\n--- Testing with only tdid_context ---")
        vt_prime_only_tdid = decoder(dummy_f_attr, None, dummy_tdid_context)
        assert vt_prime_only_tdid.shape == expected_output_shape
        print("只使用 tdid 测试通过!")
        
        # 测试只传入 frid
        print("\n--- Testing with only frid_context ---")
        vt_prime_only_frid = decoder(dummy_f_attr, dummy_frid_context, None)
        assert vt_prime_only_frid.shape == expected_output_shape
        print("只使用 frid 测试通过!")

        # 测试都不传入 (应该会看到警告)
        print("\n--- Testing with no context (expect warning) ---")
        vt_prime_no_context = decoder(dummy_f_attr, None, None)
        assert vt_prime_no_context.shape == expected_output_shape
        print("不使用 context 测试通过 (请检查是否有警告信息)!")

    except Exception as e:
         print(f"测试失败: {e}")
         traceback.print_exc()