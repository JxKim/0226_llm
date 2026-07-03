from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1、加载基座模型
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")


# 2、记载适配器
peft_model = PeftModel.from_pretrained(model=model,model_id="finetuned/05_peft_lora_demo/checkpoint-1000")


# 3、将适配器和基座模型合并
model = peft_model.merge_and_unload()

model.save_pretrained("finetuned/05_peft_merged")
tokenizer.save_pretrained("finetuned/05_peft_merged")