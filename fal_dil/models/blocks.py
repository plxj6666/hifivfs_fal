# fal_dil/models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- 新增导入 ---
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import logging
logger = logging.getLogger(__name__)
# ---

class ResBlock(nn.Module):
    # --- 修改 __init__ 添加 use_checkpoint ---
    def __init__(self, channels, use_checkpoint: bool = False): # <<<--- 添加参数
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.use_checkpoint = use_checkpoint # <<<--- 保存参数
        if self.use_checkpoint:
             logger.debug("ResBlock initialized with checkpointing ENABLED.")

    def _forward_impl(self, x):
        # 将原 forward 逻辑放入内部方法
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

    def forward(self, x):
        # --- 修改 forward 以使用 checkpoint ---
        if self.use_checkpoint and self.training: # 只在训练时使用 checkpoint
            # 使用 checkpoint 包裹主要计算
            # 注意：需要确保 _forward_impl 的输入和输出都是 Tensor 或 Tuple/List of Tensors
            return grad_checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            # 非训练或不使用 checkpoint 时直接调用
            return self._forward_impl(x)
        # ---


class SelfAttentionBlock(nn.Module):
    # --- 修改 __init__ 添加 use_checkpoint ---
    def __init__(self, channels, num_heads=8, use_checkpoint: bool = False): # <<<--- 添加参数
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.compress_ratio = 4
        self.compress = nn.MaxPool2d(kernel_size=self.compress_ratio, stride=self.compress_ratio)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        self.ln2 = nn.LayerNorm(channels)
        self.use_checkpoint = use_checkpoint # <<<--- 保存参数
        if self.use_checkpoint:
             logger.debug("SelfAttentionBlock initialized with checkpointing ENABLED.")

    def _forward_impl(self, x):
        # 将原 forward 逻辑放入内部方法
        B, C, H, W = x.shape
        identity = x
        x_small = self.compress(x)
        H_small, W_small = H // self.compress_ratio, W // self.compress_ratio
        x_reshaped = x_small.view(B, C, H_small * W_small).permute(0, 2, 1)
        x_norm = self.ln(x_reshaped)
        # --- Multi-Head Attention ---
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)
        # ---
        x_res1 = x_reshaped + attn_output
        x_norm2 = self.ln2(x_res1)
        # --- Feedforward ---
        ff_output = self.ff(x_norm2)
        # ---
        x_res2 = x_res1 + ff_output
        x_small_out = x_res2.permute(0, 2, 1).view(B, C, H_small, W_small)
        x_upsampled = F.interpolate(x_small_out, size=(H, W), mode='bilinear', align_corners=False)
        output = identity + x_upsampled
        return output

    def forward(self, x):
        # --- 修改 forward 以使用 checkpoint ---
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, query_channels, kv_channels, num_heads=8):
        super().__init__()
        self.query_channels = query_channels
        self.kv_channels = kv_channels
        self.num_heads = num_heads

        # 确保通道数能被头数整除
        assert query_channels % num_heads == 0, "query_channels must be divisible by num_heads"
        # 注意：实际应用中 K, V 的特征维度可能需要先投影到和 Q 一样的维度
        # 这里我们假设 kv_channels 经过处理后维度与 query_channels 相同
        # 或者 MultiheadAttention 可以处理不同的维度，只要 embed_dim 设置正确
        # 这里我们简单假设它们最后都是 query_channels
        self.mha = nn.MultiheadAttention(embed_dim=query_channels, kdim=kv_channels, vdim=kv_channels, num_heads=num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(query_channels)
        self.ln_kv = nn.LayerNorm(kv_channels) # 对 K, V 输入也进行归一化
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(query_channels, query_channels * 4),
            nn.ReLU(),
            nn.Linear(query_channels * 4, query_channels)
        )
        self.ln2 = nn.LayerNorm(query_channels)

    def forward(self, x, context):
        # x (Query) 的形状: (B, C_q, H, W)
        # context (Key, Value) 的形状: (B, SeqLen_kv, C_kv)
        # 注意：context (f_rid) 如何变成这个形状需要设计，这里先假设它已经是这个形状
        B, C_q, H, W = x.shape
        identity = x # 保存原始输入

        # 1. Reshape Query: (B, C_q, H, W) -> (B, H*W, C_q)
        x_reshaped = x.view(B, C_q, H * W).permute(0, 2, 1) # (B, H*W, C_q)

        # 2. Layer Normalization
        q_norm = self.ln_q(x_reshaped)
        kv_norm = self.ln_kv(context) # 对 context 进行归一化

        # 3. Multi-Head Cross-Attention
        # Query = q_norm, Key = kv_norm, Value = kv_norm
        attn_output, _ = self.mha(q_norm, kv_norm, kv_norm) # Output: (B, H*W, C_q)

        # 4. Residual Connection 1
        x_res1 = x_reshaped + attn_output

        # 5. Feedforward Network part
        x_norm2 = self.ln2(x_res1)
        ff_output = self.ff(x_norm2)

         # 6. Residual Connection 2
        x_res2 = x_res1 + ff_output # (B, H*W, C_q)

        # 7. Reshape back: (B, H*W, C_q) -> (B, C_q, H, W)
        output = x_res2.permute(0, 2, 1).view(B, C_q, H, W)

        return output


if __name__ == "__main__":
    # 测试一下 ResBlock
    print("--- Testing ResBlock ---")
    dummy_input_res = torch.randn(1, 320, 80, 80) # 假设 batch=1, channels=320, H=80, W=80
    res_block = ResBlock(channels=320)
    output_res = res_block(dummy_input_res)
    print(f"Input shape: {dummy_input_res.shape}")
    print(f"Output shape: {output_res.shape}") # 输出形状应该和输入一致
    print("-" * 20)


    # 测试一下 SelfAttentionBlock
    print("--- Testing SelfAttentionBlock ---")
    dummy_input_sa = torch.randn(1, 320, 80, 80) # 假设 batch=1, channels=320, H=80, W=80
    # 注意力头的数量需要能整除通道数
    sa_block = SelfAttentionBlock(channels=320, num_heads=8)
    output_sa = sa_block(dummy_input_sa)
    print(f"Input shape: {dummy_input_sa.shape}")
    print(f"Output shape: {output_sa.shape}") # 输出形状应该和输入一致
    print("-" * 20)


    # 测试一下 CrossAttentionBlock
    print("--- Testing CrossAttentionBlock ---")
    dummy_input_ca_x = torch.randn(1, 640, 40, 40) # Decoder 中的一个特征图 Query (B, Cq, H, W)
    # f_rid 假设是 (B, SeqLen, Ckv)，例如 (1, 49, 1280)，这个维度需要匹配
    # 这里假设 f_rid 已经被处理成 Key/Value 需要的形状和维度 Ckv
    # 假设 Ckv 也是 1280
    dummy_context_frid = torch.randn(1, 49, 1280) # (B, SeqLen_kv, Ckv)
    # 交叉注意力需要 Q 和 K/V 的通道数，这里假设 K/V 也是 1280
    # 注意：在实际应用中，MultiheadAttention 可能需要 Q 和 K/V 的维度一致，
    # 或者可以指定 kdim 和 vdim。这里我们假设它们最终作用在 640 维上
    # 因此可能需要一个线性层将 context 映射到 640 维或者将 x 映射到 1280 维
    # 为了简单，我们先假设 K/V 的维度可以直接被 MHA 使用，且输出维度为 query_channels (640)
    # 真实情况：一般会将 context 投影到和 query 相同的维度
    # 我们这里简化，假设 kdim=vdim=1280, embed_dim=640 (query_channels)
    ca_block = CrossAttentionBlock(query_channels=640, kv_channels=1280, num_heads=8)

    # 运行前向传播
    output_ca = ca_block(dummy_input_ca_x, dummy_context_frid)
    print(f"Input x shape: {dummy_input_ca_x.shape}")
    print(f"Input context shape: {dummy_context_frid.shape}")
    print(f"Output shape: {output_ca.shape}") # 输出形状应该和输入 x 一致
    print("-" * 20)

