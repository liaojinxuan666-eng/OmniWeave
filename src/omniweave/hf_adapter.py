import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .memweave import MemWeave

class OmniWeaveCore(nn.Module):  # 注意这里：去掉了空格
    """
    OmniWeave 架构适配器。
    核心功能：挂载外部开源基座模型（冻结），注入 MemWeave 记忆槽。
    """
    def __init__(self, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", num_slots=64):
        super().__init__()
        print(f"正在加载基座模型: {base_model_name} ...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 冻结基座模型的所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.base_model.config.hidden_size
        
        # 实例化我们的核心架构：动态记忆槽
        self.memweave = MemWeave(d_model=self.hidden_size, num_slots=num_slots).to(self.base_model.device)

    def forward(self, input_ids, use_memory=True):
        outputs = self.base_model(input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        if use_memory:
            mem_out = self.memweave(last_hidden_state, mode="dynamic", reset=False)
            fused_hidden_state = last_hidden_state + mem_out
        else:
            fused_hidden_state = last_hidden_state

        logits = self.base_model.lm_head(fused_hidden_state)
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, use_memory=True):
        self.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        input_ids = inputs.input_ids
        
        for _ in range(max_new_tokens):
            curr_input_ids = input_ids[:, -512:] 
            logits = self.forward(curr_input_ids, use_memory=use_memory)
            
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)