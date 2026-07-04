from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
import torch
import argparse
parser = argparse.ArgumentParser(description="Merge Lora Model")
parser.add_argument("--base_model",type=str)
parser.add_argument("--peft_model",type=str)
parser.add_argument("--merge_model_name",type=str)
args = parser.parse_args()

base_model_name = args.base_model
peft_model_name = args.peft_model

merge_model_name = args.merge_model_name
tokenizer = AutoTokenizer.from_pretrained(peft_model_name)
# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(base_model_name,dtype = torch.float16,device_map="auto")
# 加载适配器
peft_model = PeftModel.from_pretrained(base_model,model_id=peft_model_name,dtype=torch.float16,device_map="auto")
# 将适配器合并到基础模型，避免在推理时需要进行额外的计算量，从而导致延迟
merged_model = peft_model.merge_and_unload()
# 调用save_pretrained方法保存合并后的模型到本地
merged_model.save_pretrained(merge_model_name)
tokenizer.save_pretrained(merge_model_name)
