import torch.nn as nn
from .operators import SimpleSSM, LinearAttention
from .router import BudgetAwareRouter

class WeaveBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SimpleSSM(d_model)
        self.linear_attn = LinearAttention(d_model)
        self.router = BudgetAwareRouter(d_model, num_operators=2)

    def forward(self, x, latency_budget=1.0, memory_budget=1.0):
        residual = x
        x_norm = self.norm(x)
        
        # 路由决定走哪个算子
        route_weights = self.router(x_norm, latency_budget, memory_budget)
        
        # 此处简化：按 batch 平均权重，真实场景可能按 token 路由
        if self.training:
            # 软混合
            out = route_weights[:, 0:1] * self.ssm(x_norm) + \
                  route_weights[:, 1:2] * self.linear_attn(x_norm)
        else:
            # 硬路由（端侧极致省算力）
            if route_weights[0, 0] > 0.5:
                out = self.ssm(x_norm)
            else:
                out = self.linear_attn(x_norm)
                
        return residual + out