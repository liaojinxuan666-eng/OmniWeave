import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.omniweave.model import OmniWeaveModel

# ================= 配置 =================
BATCH_SIZE = 32
SEQ_LEN = 128
VOCAB_SIZE = 256      # 字节级，0-255
D_MODEL = 256
NUM_LAYERS = 8
EPOCHS = 3
LR = 3e-4
LAMBDA_BALANCE = 0.01  # 负载均衡损失的权重
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ================= 模拟数据集 =================
class DummyByteDataset(Dataset):
    def __init__(self, size=10000, seq_len=SEQ_LEN):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(0, VOCAB_SIZE, (self.seq_len,))
        y = torch.roll(x, shifts=-1)  # 预测下一个字节
        return x, y

train_dataset = DummyByteDataset(size=10000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ================= 模型初始化 =================
model = OmniWeaveModel(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    max_seq_len=SEQ_LEN + 10
).to(DEVICE)

print(f"模型参数量: {model.count_parameters() / 1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

# ================= 训练循环 =================
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    total_task_loss = 0
    total_aux_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        
        # 前向传播（加入 reset_memory=True）
        logits, aux_loss = model(x, latency_budget=1.0, memory_budget=1.0, mem_mode="fast", reset_memory=True)
        
        # 计算任务损失
        task_loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            y.view(-1)
        )
        
        # 总损失 = 任务损失 + 负载均衡损失
        loss = task_loss + LAMBDA_BALANCE * aux_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止路由训练爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_aux_loss += aux_loss.item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "task": f"{task_loss.item():.3f}",
            "aux": f"{aux_loss.item():.4f}"
        })

    avg_loss = total_loss / len(train_loader)
    avg_task = total_task_loss / len(train_loader)
    avg_aux = total_aux_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Task: {avg_task:.4f} | Aux: {avg_aux:.6f}")

print("训练完成！")

# ================= 推理测试 =================
model.eval()
with torch.no_grad():
    test_input = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN)).to(DEVICE)
    test_logits, test_aux = model(test_input, latency_budget=0.5, memory_budget=0.5, mem_mode="warm", reset_memory=True)
    print(f"推理模式输出形状: {test_logits.shape}")
    print(f"推理模式辅助损失: {test_aux.item():.6f}")