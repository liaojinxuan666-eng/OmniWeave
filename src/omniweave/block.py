import torch
import torch.nn as nn

from .operators import SimpleSSM, LinearAttention
from .router import BudgetAwareRouter
from .memweave import MemWeave

class WeaveBlock(nn.Module):
    def __init__(self, d_model: int, num_slots: int = 16):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SimpleSSM(d_model)
        self.linear_attn = LinearAttention(d_model)
        self.router = BudgetAwareRouter(d_model, num_operators=2)
        self.mem = MemWeave(d_model, num_slots=num_slots)

    def forward(self, x, latency_budget=1.0, memory_budget=1.0,
                mem_mode="dynamic", reset_memory=False):
        residual = x
        x_norm = self.norm(x)

        # 1. 路由
        route_weights = self.router(x_norm, latency_budget, memory_budget)  # [B, 2]
        aux_loss = self.router.compute_load_balancing_loss(route_weights)

        # 2. 算子计算
        out_ssm = self.ssm(x_norm)
        out_attn = self.linear_attn(x_norm)

        w_ssm = route_weights[:, 0:1].unsqueeze(-1)   # [B,1,1]
        w_attn = route_weights[:, 1:2].unsqueeze(-1)  # [B,1,1]
        out = w_ssm * out_ssm + w_attn * out_attn

        # 3. 记忆融合
        mem_out = self.mem(x_norm, mode=mem_mode, reset=reset_memory)
        if mem_out is not None:
            out = out + mem_out

        return residual + out, aux_loss