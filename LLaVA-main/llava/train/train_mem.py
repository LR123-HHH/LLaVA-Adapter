
import torch
# 执行内存清理
torch.cuda.empty_cache()
# 限制内存使用（可选）
torch.cuda.set_per_process_memory_fraction(0.8)


from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="sdpa")
