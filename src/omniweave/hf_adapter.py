import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .memweave import MemWeave

class OmniWeave Core(nn.Module):
    """
    OmniWeave 架构适配器。
    核心功能：挂载外部开源基座模型（冻结），注入 MemWeave 记忆槽，实现“神不知鬼不觉”的知识融合。
    """
    def __init__(self, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", num_slots=64):
        super().__init__()
        print(f"正在加载基座模型: {base_model_name} ...")
        # 1. 加载基座模型和分词器
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 2. 冻结基座模型的所有参数（不搞模型训练，只搞架构推理）
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.base_model.config.hidden_size
        
        # 3. 实例化我们的核心架构：动态记忆槽
        # 这里的 MemWeave 是我们之前写好的，为了适配 Qwen 的 hidden_size
        self.memweave = MemWeave(d_model=self.hidden_size, num_slots=num_slots).to(self.base_model.device)

    def forward(self, input_ids, use_memory=True):
        """
        重写前向传播，在基座模型输出特征后，强行融合记忆槽的潜意识。
        """
        # 1. 基座模型前向传播，获取隐藏状态（hidden_states）
        outputs = self.base_model(input_ids, output_hidden_states=True)
        
        # 获取最后一层的隐藏状态 (shape: [batch, seq_len, hidden_size])
        last_hidden_state = outputs.hidden_states[-1]
        
        # 2. 架构介入：注入动态记忆
        if use_memory:
            # 记忆槽读取并融合
            mem_out = self.memweave(last_hidden_state, mode="dynamic", reset=False)
            # 残差连接：将记忆潜意识直接叠加到模型当前的认知上
            fused_hidden_state = last_hidden_state + mem_out
        else:
            fused_hidden_state = last_hidden_state

        # 3. 通过基座模型的输出头（lm_head）生成最终的 token 概率
        logits = self.base_model.lm_head(fused_hidden_state)
        
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, use_memory=True):
        """
        极简的推理生成函数，对比有无记忆槽的差异。
        """
        self.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        input_ids = inputs.input_ids
        
        for _ in range(max_new_tokens):
            # 截断上下文，防止爆显存
            curr_input_ids = input_ids[:, -512:] 
            logits = self.forward(curr_input_ids, use_memory=use_memory)
            
            # 取最后一个词的 logits
            next_token_logits = logits[0, -1, :]
            # 采样生成
            probs = torch.softmax(next_token_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # 遇到 eos 停止
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)