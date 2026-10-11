import torch
import torch.nn as nn

# 假设这些文件都放在 src/omniweave/ 目录下，相对导入
from .operators import SimpleSSM, LinearAttention
from .router import BudgetAwareRouter
from .memweave import MemWeave

class WeaveBlock(nn.Module):
    """
    OmniWeave 的核心基础模块。
    组合了：算子池 (SSM / 线性注意力) + 硬件感知路由 + 三级记忆。
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 1. 归一化层
        self.norm = nn.LayerNorm(d_model)
        
        # 2. 算子池（低硬件版只保留两个核心算子）
        self.ssm = SimpleSSM(d_model)
        self.linear_attn = LinearAttention(d_model)
        
        # 3. 轻量级路由器（0: SSM, 1: LinearAttention）
        self.router = BudgetAwareRouter(d_model, num_operators=2)
        
        # 4. 三级记忆系统
        self.mem = MemWeave(d_model)

    def forward(self, x, latency_budget=1.0, memory_budget=1.0, mem_mode="fast"):
        """
        x: [batch, seq_len, d_model]
        latency_budget / memory_budget: 硬件预算，用于指导路由
        mem_mode: "fast" | "warm" | "cold"，决定用哪级记忆
        """
        residual = x
        x_norm = self.norm(x)
        
        # --- 1. 路由决策 ---
        route_weights = self.router(x_norm, latency_budget, memory_budget) # [B, 2]
        
        # --- 2. 算子计算 ---
        # 无论是训练（软混合）还是推理（One-Hot 硬路由），统一用权重计算，
        # 避免遍历 batch 逐个判断，最大化 GPU/NPU 利用率。
        out_ssm = self.ssm(x_norm)
        out_attn = self.linear_attn(x_norm)
        
        # 广播乘法：route_weights[:, 0:1] 形状是 [B, 1, 1]，自动广播
        out = route_weights[:, 0:1] * out_ssm + route_weights[:, 1:2] * out_attn
        
        # --- 3. 记忆融合 ---
        mem_out = self.mem(x_norm, mode=mem_mode)
        if mem_out is not None:
            # 如果记忆输出存在，直接与算子输出相加
            # 注意：fast memory 输出形状可能是 [B, 1, D]，会自动广播到 [B, L, D]
            out = out + mem_out
            
        # --- 4. 残差连接 ---
        return residual + out