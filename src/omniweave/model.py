import torch
import torch.nn as nn

from .block import WeaveBlock

class OmniWeaveModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, num_layers=8,
                 max_seq_len=2048, num_slots=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            WeaveBlock(d_model, num_slots=num_slots) for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 判断 module 是否是线性层
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # 判断 module 是否是嵌入层
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, latency_budget=1.0, memory_budget=1.0,
                mem_mode="dynamic", reset_memory=False):
        B, L = input_ids.shape
        device = input_ids.device
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        x = self.token_embed(input_ids) + self.pos_embed(positions)

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x, latency_budget, memory_budget, mem_mode, reset_memory)
            total_aux_loss += aux_loss

        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))