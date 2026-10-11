import torch
import torch.nn as nn

class FastMemory(nn.Module):
    """快记忆：SSM 状态，恒定内存，极省显存，适合端侧"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.state = None  # 延迟初始化，避免固定 batch size

    def forward(self, x, reset=False):
        batch_size = x.size(0)
        if reset or self.state is None:
            self.state = torch.zeros(batch_size, self.d_model, device=x.device)
        # 简化版状态更新（实际可用 Selective Scan）
        x_mean = x.mean(dim=1)  # [B, D]
        self.state = self.state * 0.9 + x_mean * 0.1
        return self.state.unsqueeze(1)  # [B, 1, D]

class WarmMemory(nn.Module):
    """温记忆：线性注意力状态矩阵，用于工作记忆"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.state_kv = None

    def forward(self, x, reset=False):
        batch_size = x.size(0)
        if reset or self.state_kv is None:
            self.state_kv = torch.zeros(batch_size, self.d_model, self.d_model, device=x.device)
        
        # 低秩更新，模拟线性注意力
        k = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        v = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        self.state_kv = self.state_kv * 0.95 + torch.bmm(k.transpose(1, 2), v) * 0.05
        return torch.bmm(x, self.state_kv)  # [B, L, D]

class ColdMemory(nn.Module):
    """冷记忆：压缩 KV 检索，存在 CPU 或外部，不占 NPU 内存"""
    def __init__(self, max_size=1024):
        super().__init__()
        self.max_size = max_size
        self.buffer = []

    def write(self, x):
        # 剥离梯度，转到 CPU，避免撑爆显存
        self.buffer.append(x.detach().cpu())
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def retrieve(self, query):
        if not self.buffer: 
            return None
        # 实际项目中这里接 FAISS 或向量数据库
        stacked = torch.cat(self.buffer, dim=1).to(query.device)
        return stacked

class MemWeave(nn.Module):
    """三级记忆统一接口"""
    def __init__(self, d_model):
        super().__init__()
        self.fast = FastMemory(d_model)
        self.warm = WarmMemory(d_model)
        self.cold = ColdMemory()

    def forward(self, x, mode="fast"):
        if mode == "fast":
            return self.fast(x)
        elif mode == "warm":
            return self.warm(x)
        elif mode == "cold":
            return self.cold.retrieve(x)
        else:
            return x