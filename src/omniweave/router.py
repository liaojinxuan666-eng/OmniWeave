import torch
import torch.nn as nn
import torch.nn.functional as F

class BudgetAwareRouter(nn.Module):
    """
    硬件预算感知路由器。
    训练时：输出软权重 (Softmax)，允许梯度回传。
    推理时：输出 One-Hot 硬权重 (Top-1)，极致省算力。
    """
    def __init__(self, d_model, num_operators=2):
        super().__init__()
        # 极简的 MLP 路由，1 层就够，防止增加端侧延迟
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, 64), # +2 代表输入 延迟预算 和 内存预算
            nn.ReLU(),
            nn.Linear(64, num_operators)
        )
        self.num_operators = num_operators

    def forward(self, x, latency_budget=1.0, memory_budget=1.0):
        # x: [batch, seq_len, d_model]
        # 池化：取序列平均值，代表整个句子的状态
        pooled = x.mean(dim=1) # [batch, d_model]
        
        # 拼接硬件预算
        budget = torch.tensor(
            [[latency_budget, memory_budget]], 
            device=x.device, 
            dtype=x.dtype
        ).expand(pooled.size(0), -1)
        
        logits = self.gate(torch.cat([pooled, budget], dim=-1))
        
        if self.training:
            # 训练时：软路由 (Softmax)
            return F.softmax(logits / 1.0, dim=-1)
        else:
            # 推理时：硬路由 (Top-1)
            idx = torch.argmax(logits, dim=-1)
            return F.one_hot(idx, num_classes=self.num_operators).float()

    def compute_load_balancing_loss(self, route_weights):
        """
        负载均衡损失：防止路由坍缩。
        鼓励模型在训练时，不要把所有的概率都押注在某一个算子上。
        来源：Switch Transformer 论文 (https://arxiv.org/abs/2101.03961)
        """
        # route_weights: [batch_size, num_operators]
        # 1. 计算每个算子在当前 batch 中被分配到的平均概率
        mean_weights = route_weights.mean(dim=0) # [num_operators]
        
        # 2. 计算每个算子被选中的比例（硬选择比例）
        # 这里用软概率近似，因为硬选择不可导
        # 理想情况下，每个算子的平均概率应该接近 1/num_operators
        target = torch.ones_like(mean_weights) / self.num_operators
        
        # 3. 均方误差损失
        loss = torch.sum((mean_weights - target) ** 2)
        return loss