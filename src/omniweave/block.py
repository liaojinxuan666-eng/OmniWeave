import torch
import torch.nn as nn

from .operators import SimpleSSM, LinearAttention
from .router import BudgetAwareRouter
from .memweave import MemWeave

class WeaveBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SimpleSSM(d_model)
        self.linear_attn = LinearAttention(d_model)
        self.router = BudgetAwareRouter(d_model, num_operators=2)
        self.mem = MemWeave(d_model)

    def forward(self, x, latency_budget=1.0, memory_budget=1.0, mem_mode="fast"):
        residual = x
        x_norm = self.norm(x)
        
        # 1. 路由决策
        route_weights = self.router(x_norm, latency_budget, memory_budget) # [B, 2]
        
        # 2. 计算负载均衡损失
        aux_loss = self.router.compute_load_balancing_loss(route_weights)
        
        # 3. 算子计算
        out_ssm = self.ssm(x_norm)
        out_attn = self.linear_attn(x_norm)
        
        # 修复：添加 unsqueeze(-1) 保证张量广播正确
        w_ssm = route_weights[:, 0:1].unsqueeze(-1)   # [B, 1, 1]
        w_attn = route_weights[:, 1:2].unsqueeze(-1)  # [B, 1, 1]
        
        out = w_ssm * out_ssm + w_attn * out_attn
        
        # 4. 记忆融合
        mem_out = self.mem(x_norm, mode=mem_mode)
        if mem_out is not None:
            out = out + mem_out
            
        return residual + out, aux_loss