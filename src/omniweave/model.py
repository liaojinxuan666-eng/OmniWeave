import torch
import torch.nn as nn

from .block import WeaveBlock

class OmniWeaveModel(nn.Module):
    """
    OmniWeave 顶层模型。
    组装：Token 嵌入 + 位置嵌入 + 多层 WeaveBlock + 输出头。
    支持硬件预算传入，指导每一层的路由决策。
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        num_layers: int = 8,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # 1. 输入层
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # 2. 主干：堆叠 WeaveBlock
        self.blocks = nn.ModuleList([
            WeaveBlock(d_model) for _ in range(num_layers)
        ])

        # 3. 输出层
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 4. 权重共享（可选，节省参数）
        self.lm_head.weight = self.token_embed.weight

        # 5. 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids,
        latency_budget: float = 1.0,
        memory_budget: float = 1.0,
        mem_mode: str = "fast",
    ):
        """
        input_ids: [batch, seq_len]
        返回 logits: [batch, seq_len, vocab_size]
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 1. 位置索引
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        # 2. 嵌入相加
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # 3. 逐层通过 WeaveBlock
        for block in self.blocks:
            x = block(
                x,
                latency_budget=latency_budget,
                memory_budget=memory_budget,
                mem_mode=mem_mode,
            )

        # 4. 输出
        x = self.norm_out(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)