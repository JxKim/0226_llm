from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-8B")
# 1、加载数据集
train_data = load_dataset("json",data_files={"train":"data/keywords_data_train.jsonl","test":"data/keywords_data_test.jsonl"})


# 2、处理数据集的格式
from typing  import List,Dict
def convert_data_format(examples:Dict[str, List]):
    """
    将数据集处理出 {"messages":[xx]}
    """
    coversation:List[List] = examples["conversation"]
    all_examples_messages:List = []

    for example in coversation:
        message_list = []
        example_message:Dict[str,str]=example[0]
        message_list.append({"role":"user","content":example_message["human"]})
        message_list.append({"role":"assistant","content":example_message["assistant"]})
        all_examples_messages.append(message_list)

    return {"messages":all_examples_messages}

mapped_train_data = train_data.map(convert_data_format,batched=True,remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])


# 3、构造SFTConfig实例
from trl.trainer.sft_config import SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/09_accelerate"
sft_config = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=1000,
    num_train_epochs=1,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    # warmup_ratio=0.1,
    warmup_steps=0.1,
    eval_strategy="steps",
    eval_steps=200,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    output_dir="./finetuned/09_accelerate",
    bf16=True,
    gradient_checkpointing=True,
    activation_offloading=False,
    max_length=700,
    assistant_only_loss=True,
    chat_template_path="./chat_template.jinja"
)

# 4、构造一个SFTTrainer的实例
from trl.trainer.sft_trainer import SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=mapped_train_data["train"],
    eval_dataset=mapped_train_data["test"],
    processing_class=tokenizer
)

trainer.train()
trainer.save_model("finetuned/09_accelerate")