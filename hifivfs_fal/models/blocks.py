import torch
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 定义残差块的两个卷积层
        # 这里保持通道数不变，使用 3x3 卷积核，并用 padding=1 保持 H, W 不变
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU() # 使用 ReLU 作为激活函数
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 输入 x
        identity = x
        # 通过两个卷积层和激活函数
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        # 跳跃连接：将输出与原始输入相加
        out += identity
        # 再次通过激活函数（有些设计会放在跳跃连接之前）
        out = self.relu(out)
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # PyTorch 的 MultiheadAttention
        # embed_dim 必须能被 num_heads 整除
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        # 通常在 Attention 前后会加 Layer Normalization
        self.ln = nn.LayerNorm(channels)
        # Feedforward network (optional but common)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels)
        )
        self.ln2 = nn.LayerNorm(channels)


    def forward(self, x):
        # x 的形状: (B, C, H, W)
        B, C, H, W = x.shape
        identity = x # 保存原始输入用于残差连接

        # 1. Reshape for Attention: (B, C, H, W) -> (B, H*W, C)
        # H*W 相当于序列长度
        x_reshaped = x.view(B, C, H * W).permute(0, 2, 1) # (B, H*W, C)

        # 2. Layer Normalization
        x_norm = self.ln(x_reshaped)

        # 3. Multi-Head Attention
        # Q, K, V 都来自 x_norm
        attn_output, _ = self.mha(x_norm, x_norm, x_norm) # Output: (B, H*W, C)

        # 4. Residual Connection 1
        x_res1 = x_reshaped + attn_output

        # 5. Feedforward Network part
        x_norm2 = self.ln2(x_res1)
        ff_output = self.ff(x_norm2)

        # 6. Residual Connection 2
        x_res2 = x_res1 + ff_output # (B, H*W, C)

        # 7. Reshape back: (B, H*W, C) -> (B, C, H, W)
        output = x_res2.permute(0, 2, 1).view(B, C, H, W)

        return output


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

