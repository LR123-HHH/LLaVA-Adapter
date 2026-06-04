# convert_lora_to_full.py
import torch
import argparse
from peft import PeftModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

parser = argparse.ArgumentParser()
parser.add_argument("--model-base", type=str, required=True)
parser.add_argument("--lora-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
args = parser.parse_args()

# 加载基础模型
model_name = get_model_name_from_path(args.model_base)
_, base_model, _, _ = load_pretrained_model(args.model_base, None, model_name)

# 加载LoRA模型
print(f"加载LoRA权重: {args.lora_path}")
model = PeftModel.from_pretrained(base_model, args.lora_path)

# 合并权重
print("合并LoRA权重到基础模型...")
merged_model = model.merge_and_unload()

# 保存完整模型
print(f"保存完整模型到: {args.output_path}")
merged_model.save_pretrained(args.output_path)

print("转换完成！")