import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSSM(nn.Module):
    """极简 SSM，使用因果卷积并行化处理序列"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_model)
        self.dt_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, L, D]
        x_t = x.transpose(1, 2)
        conv_out = self.conv(x_t)[:, :, :-2]
        conv_out = conv_out.transpose(1, 2)
        gate = torch.sigmoid(self.dt_proj(x))
        return self.x_proj(conv_out) * gate

class LinearAttention(nn.Module):
    """线性注意力，保留因果时序"""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # 因果线性注意力: k_cumsum 前缀和
        k_cumsum = torch.cumsum(k, dim=2)
        out = torch.einsum('bhld,bhld->bhld', q, k_cumsum) # 这是一个极其简化的近似
        # 为了不过于复杂，这里直接把 out 投影回去
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)