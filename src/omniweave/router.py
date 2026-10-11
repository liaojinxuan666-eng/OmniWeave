import torch
import torch.nn as nn
import torch.nn.functional as F

class BudgetAwareRouter(nn.Module):
    """根据输入特征和硬件预算，硬路由到某个算子"""
    def __init__(self, d_model, num_operators=2):
        super().__init__()
        # 轻量级分类头，只有 1 层
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, 64), # +2 代表当前的 延迟预算 和 内存预算
            nn.ReLU(),
            nn.Linear(64, num_operators)
        )
        self.num_operators = num_operators

    def forward(self, x, latency_budget, memory_budget):
        # x: [batch, seq_len, d_model]
        # 取平均特征来做路由决策
        pooled = x.mean(dim=1) # [batch, d_model]
        budget = torch.tensor([[latency_budget, memory_budget]], device=x.device).expand(pooled.size(0), -1)
        logits = self.gate(torch.cat([pooled, budget], dim=-1))
        
        # 推理时使用硬路由 (Top-1)
        if not self.training:
            idx = torch.argmax(logits, dim=-1)
            return F.one_hot(idx, num_classes=self.num_operators).float()
        # 训练时使用软路由
        return F.softmax(logits / 1.0, dim=-1)
        
    def compute_load_balancing_loss(route_weights):
    """防止路由坍缩：鼓励不同算子被均匀使用"""
    # route_weights: [batch_size, num_operators]
    # 计算每个算子在当前 batch 中被选中的概率均值
    mean_weights = route_weights.mean(dim=0) # [num_operators]
    # 理想情况下，概率分布应该接近均匀分布
    target = torch.ones_like(mean_weights) / mean_weights.size(0)
    loss = torch.sum((mean_weights - target) ** 2)
        return loss