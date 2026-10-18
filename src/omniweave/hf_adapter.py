import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .memweave import MemWeave

class OmniWeaveCore(nn.Module):
    def __init__(self, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", num_slots=64):
        super().__init__()
        print(f"正在加载基座模型: {base_model_name} ...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.base_model.config.hidden_size
        # 强制转换为 half 精度
        self.memweave = MemWeave(d_model=self.hidden_size, num_slots=num_slots).half().to(self.base_model.device)

    def forward(self, input_ids, use_memory=True):
        outputs = self.base_model(input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        if use_memory:
            mem_out = self.memweave(last_hidden_state, mode="dynamic", reset=False)
            # 修复点 1：强制将 mem_out 转换回与 hidden_state 相同的精度，防止加法报错
            fused_hidden_state = last_hidden_state + mem_out.to(last_hidden_state.dtype)
        else:
            fused_hidden_state = last_hidden_state

        # 最后过 lm_head
        logits = self.base_model.lm_head(fused_hidden_state)
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, use_memory=True):
        self.eval()
        
        # 修复点 2：分步进行 to device，避免 BatchEncoding 直接 .to() 报错
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.base_model.device)
        
        for _ in range(max_new_tokens):
            # 截断过长的上下文
            curr_input_ids = input_ids[:, -1024:] 
            
            logits = self.forward(curr_input_ids, use_memory=use_memory)
            
            # 提取最后一个词的概率
            next_token_logits = logits[0, -1, :].float() # 转为 float32 再 softmax 更稳定
            
            probs = torch.softmax(next_token_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)