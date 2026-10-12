import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicMemorySlots(nn.Module):
    """
    动态可学习记忆槽。
    - 读取：根据输入对槽位做注意力，softmax 加权求和。
    - 写入：门控决定哪些槽位被更新、哪些被遗忘。
    - 写入使用 no_grad 隔离梯度，防止跨 batch 累积。
    - slots 参数的梯度来自读取路径（可微），写入路径只改变数值（不传梯度）。
    """
    def __init__(self, d_model, num_slots=16):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model

        # 记忆槽本体：可学习参数
        self.slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)

        # 读取门控：每个 token 对每个槽位的注意力
        self.read_gate = nn.Linear(d_model, num_slots)
        # 写入门控：控制是否写入
        self.write_gate = nn.Linear(d_model, num_slots)
        # 遗忘门控：控制哪些槽位被遗忘
        self.forget_gate = nn.Linear(d_model, num_slots)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, reset=False):
        B, L, D = x.shape

        # ---------- 读取 ----------
        read_logits = self.read_gate(x)                    # [B, L, num_slots]
        read_weights = F.softmax(read_logits, dim=-1)
        read_out = torch.einsum('bls,sd->bld', read_weights, self.slots)
        out = self.out_proj(read_out)

        # ---------- 写入（仅训练时更新） ----------
        if self.training:
            with torch.no_grad():
                # 每个 token 的写/忘门控
                w_write = torch.sigmoid(self.write_gate(x))    # [B, L, num_slots]
                w_forget = torch.sigmoid(self.forget_gate(x))  # [B, L, num_slots]

                # 序列级聚合：取平均，避免逐 token 更新抖动太大
                w_write_mean = w_write.mean(dim=1)             # [B, num_slots]
                w_forget_mean = w_forget.mean(dim=1)           # [B, num_slots]

                # 要写入的内容：按写权重对 x 加权
                weighted_x = torch.einsum('bls,bld->bsd', w_write, x)
                norm = w_write.sum(dim=1, keepdim=True) + 1e-8
                write_content = (weighted_x / norm).mean(dim=0)  # [num_slots, D]

                # 跨 batch 求平均的门控（也可以只取第一个样本，视需求）
                w_write_batch = w_write_mean.mean(dim=0)          # [num_slots]
                w_forget_batch = w_forget_mean.mean(dim=0)        # [num_slots]

                # 更新记忆槽：new = (1 - forget) * old + write * content
                self.slots.data = (
                    (1 - w_forget_batch.unsqueeze(-1)) * self.slots.data
                    + w_write_batch.unsqueeze(-1) * write_content
                )

        return out


class FastMemory(nn.Module):
    """快记忆：保留因果时序的深度可分离因果卷积"""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, reset=False):
        x_t = x.transpose(1, 2)
        out = self.conv(x_t)[:, :, :-2]
        out = out.transpose(1, 2)
        return self.proj(out)


class FastMemory(nn.Module):
    def __init__(self, d_model):
        super().__init__()  # 注意这里：super() 后面跟括号和点
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, reset=False):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        k_cumsum = torch.cumsum(k, dim=1)
        out = q * k_cumsum
        return self.proj(out)


class ColdMemory(nn.Module):
    """冷记忆：外置 CPU 缓冲，不占显存"""
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
    """
    统一记忆接口。
    mode: "dynamic"（默认，可学习槽位）| "fast" | "warm" | "cold"
    """
    def __init__(self, d_model, num_slots=16):
        super().__init__()
        self.dynamic = DynamicMemorySlots(d_model, num_slots=num_slots)
        self.fast = FastMemory(d_model)
        self.warm = WarmMemory(d_model)
        self.cold = ColdMemory()

    def forward(self, x, mode="dynamic", reset=False):
        if mode == "dynamic":
            return self.dynamic(x)
        elif mode == "fast":
            return self.fast(x, reset=reset)
        elif mode == "warm":
            return self.warm(x, reset=reset)
        elif mode == "cold":
            return self.cold.retrieve(x)
        else:
            return x