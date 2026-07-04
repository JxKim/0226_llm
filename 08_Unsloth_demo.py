# 1、优化一：先引入Unsloth，UnSloth会将trl,transformers,peft库进行优化（patch，打补丁）
from unsloth import FastLanguageModel
from trl.trainer.sft_trainer import SFTTrainer
from transformers import AutoTokenizer


model,tokenizer = FastLanguageModel.from_pretrained(
    # 2、优化二：当我们仅传入模型在huggingface的model_id以及load_in_4bit/load_in_8bit等，unsloth会下载它自己量化好之后的模型
    model_name="./model/Qwen3-8B",
    load_in_4bit=True,
    # 为了让unsloth直接下载原模型，需要使用use_exact_model_name这个参数
    use_exact_model_name=True,
    # 为了让unsloth加载本地模型，需要使用local_files_only参数
    local_files_only = True
)

model = FastLanguageModel.get_peft_model(
    model=model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
)

from datasets import load_dataset
train_data = load_dataset("json",data_files={"train":"data/psychology_data.jsonl"})
# 将数据集切分成训练集和验证集
train_data = train_data["train"].train_test_split(test_size=0.1)


# 处理数据集的格式
from typing  import List,Dict
def convert_data_format(examples:Dict[str, List]):
    """
    将数据集处理出 {"messages":[xx]}
    """
    coversation:List[List] = examples["conversation"]
    all_examples_texts:List = []

    for example in coversation:
        message_list = []
        example_message:Dict[str,str]=example[0]
        message_list.append({"role":"user","content":example_message["human"]})
        message_list.append({"role":"assistant","content":example_message["assistant"]})
        # 3、优化三：可以自己在map当中调用原生的qwen3的chat_template方法
        result = tokenizer.apply_chat_template(message_list,tokenize=False,add_generation_prompt = False)
        all_examples_texts.append(result)
    return {"text":all_examples_texts}


mapped_train_data = train_data.map(convert_data_format,batched=True,remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])

from trl.trainer.sft_config import SFTConfig
import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "logs/08_Unsloth_demo"
sft_config = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=1000,
    num_train_epochs=1,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    # 注意：LoRA微调的学习率，一般会比全参微调的学习率更高，高一个数量级
    learning_rate=3e-4,
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
    output_dir="./finetuned/08_Unsloth_demo",
    bf16=True,
    gradient_checkpointing=True,
    activation_offloading=False,
    max_length=700,
    # assistant_only_loss=True,
    # chat_template_path="./chat_template.jinja"
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

from unsloth.chat_templates import train_on_responses_only

# 优化点4：直接通过chat template的结构，来去找到assistant 回答损失，不依赖jinja模板的特殊 %generation%键
trainer = train_on_responses_only(
    trainer=trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"
)

trainer.train()
trainer.save_model("finetuned/08_Unsloth_demo")

# 优化点5：直接通过Unsloth的model的save_pretrained_merged方法，来合并并且保存模型和tokenizer
model.save_pretrained_merged("./finetuned/Qwen3-8B-SFT-unsloth-merged", tokenizer, save_method="merged_16bit")





