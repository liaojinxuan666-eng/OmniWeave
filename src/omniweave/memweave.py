import torch
import torch.nn as nn

class FastMemory(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.state = None

    def forward(self, x, reset=False):
        batch_size = x.size(0)
        if reset or self.state is None:
            self.state = torch.zeros(batch_size, self.d_model, device=x.device)
        x_mean = x.mean(dim=1)
        # 关键修复：更新状态后立刻 detach()，切断计算图
        self.state = (self.state * 0.9 + x_mean * 0.1).detach()
        return self.state.unsqueeze(1)

class WarmMemory(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.state_kv = None

    def forward(self, x, reset=False):
        batch_size = x.size(0)
        if reset or self.state_kv is None:
            self.state_kv = torch.zeros(batch_size, self.d_model, self.d_model, device=x.device)
        
        k = x.mean(dim=1, keepdim=True)
        v = x.mean(dim=1, keepdim=True)
        # 关键修复：同样加上 detach()
        self.state_kv = (self.state_kv * 0.95 + torch.bmm(k.transpose(1, 2), v) * 0.05).detach()
        return torch.bmm(x, self.state_kv)

class ColdMemory(nn.Module):
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
        stacked = torch.cat(self.buffer, dim=1).to(query.device)
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