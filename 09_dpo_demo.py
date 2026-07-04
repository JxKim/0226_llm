from typing import Any
import torch
from datasets import load_dataset,DatasetDict,Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig

model = AutoModelForCausalLM.from_pretrained("./model/Qwen3-0.6B")


def last_assistant_message(messages: Any) -> dict[str, str]:
    if isinstance(messages, str):
        return {"role": "assistant", "content": messages}

    for message in reversed(messages):
        if message.get("role") == "assistant":
            return {"role": "assistant", "content": str(message.get("content", ""))}

    raise ValueError("Could not find an assistant message in chosen/rejected.")



def normalize_dpo_record(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """
    将数据集改造成DPOTrainer所需要的类型（Type）:Preference
    {
    "prompt":[{"role":"user","content":"xxxxx"}],
    "chosen":[{"role":"assistant","content":"xxxxx"}],
    "rejected":[{"role":"assistant","content":"xxxxx"}]
    }
    """
    prompt = record["prompt"]
    prompt_messages = [{"role": "user", "content": str(prompt)}]

    return {
        "prompt": prompt_messages,
        "chosen": [last_assistant_message(record["chosen"])],
        "rejected": [last_assistant_message(record["rejected"])],
    }



train_dataset :list[Dataset]= load_dataset("./data/ultrafeedback_binarized",split=["train_prefs","test_prefs"])
train_data = DatasetDict({"train":train_dataset[0],"test":train_dataset[1]})

mapped_train_dataset = train_data.map(normalize_dpo_record,remove_columns=train_data["train"].column_names,)


if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}


tokenizer = AutoTokenizer.from_pretrained("./model/Qwen3-0.6B")

peft_config = LoraConfig(
    r= 16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

model = get_peft_model(model,peft_config)


import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/09_dpo_demo"
dpo_config = DPOConfig(
    output_dir="./finetuned/09_dpo_demo",
    beta=0.1,
    max_length=2048,
    max_steps=1000,
    max_prompt_length=1024,
    num_train_epochs=1,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps=32,
    warmup_steps=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=30,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps = 100,
    report_to="tensorboard",
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=mapped_train_dataset["train"],
    eval_dataset = mapped_train_dataset["test"],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./finetuned/09_dpo_demo")
