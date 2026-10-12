import torch
import torch.nn as nn
import torch.nn.functional as F

class FastMemory(nn.Module):
    """快记忆：SSM 状态，使用因果卷积保留时序结构"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 使用深度可分离因果卷积，作为极简 SSM
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, reset=False):
        # x: [B, L, D] -> [B, D, L]
        x_t = x.transpose(1, 2)
        # 因果卷积，截断未来信息
        out = self.conv(x_t)[:, :, :-2] 
        out = out.transpose(1, 2) # 恢复 [B, L, D]
        return self.proj(out)

class WarmMemory(nn.Module):
    """温记忆：线性注意力状态矩阵，保留时序"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, reset=False):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # 简化版因果线性注意力：使用累积和
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        # 计算前缀和，模拟线性注意力状态累积
        k_cumsum = torch.cumsum(k, dim=1)
        out = q * k_cumsum
        return self.proj(out)

class ColdMemory(nn.Module):
    """冷记忆：压缩 KV 检索，存在 CPU 或外部"""
    def __init__(self, max_size=1024):
        super().__init__()
        self.max_size = max_size
        self.buffer = []

    def write(self, x):
        self.buffer.append(x.detach().cpu())
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def retrieve(self, query):
        if not self.buffer: 
            return None
        stacked = torch.cat(self.buffer, dim=0).to(query.device)
        return stacked

class MemWeave(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fast = FastMemory(d_model)
        self.warm = WarmMemory(d_model)
        self.cold = ColdMemory()

    def forward(self, x, mode="fast", reset=False):
        if mode == "fast":
            return self.fast(x, reset=reset)
        elif mode == "warm":
            return self.warm(x, reset=reset)
        elif mode == "cold":
            return self.cold.retrieve(x)
        else:
            return x