from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base/")


from typing import Dict,List
def get_train_data():
    """
    加载数据，调用chat template方法，得到分词之后的input_ids；后面在train主循环当中会调用该方法，获取训练数据
    """
    from datasets import load_dataset

    train_data = load_dataset("data/ultrachat_200k")["train_sft"]
    result_list = []
    for i in range(200):
        message_list:List = train_data[i]["messages"]
        message_list.insert(0,{"role":"system","content":"you are a helpfule assistant"})
        result: Dict[str, List]= tokenizer.apply_chat_template(message_list,tokenize=True)
        result_list.append(result["input_ids"])

    return result_list


from transformers import PreTrainedTokenizerFast
import torch
def create_answer_mask(input_ids,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从input_ids当中找出assistant回答的部分，然后输出一个与input_ids相同shape的mask，
    后续将其与pad_mask进行逻辑与操作，得到最终的mask，用以计算损失
    """

    
    # 构建answer mask，输入的input_ids为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和input_ids相同shape的全0矩阵
    answer_mask = torch.zeros_like(input_ids)

    # 2. 遍历input_ids中的每一条数据，查找assistant回答的部分，将其设置为1
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    # input_ids.shape: batch_size, seq_len
    for idx,ids in enumerate(input_ids):
        # 获取到所有的eos_position
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()
        print(eos_position)
        # 排除第一个eos_position: 第一个对应的是system prompt
        eos_position = eos_position[1:]
        # 解析获得user_ends和assistant_ends
        user_ends,assistant_ends = _parse_conversation_turns(eos_position)
        # 设置answer mask
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)   
    
    # 结果返回:
    return answer_mask

def _parse_conversation_turns(eos_positions:List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取（具体索引位置需要加一，因为eos_token_id后面还有一个\n换行符），每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[11,21]
        assistant_ends:[16,26]
    """

    use_ends = [pos+1 for pos in eos_positions[::2]]
    assistant_ends = [pos+1 for pos in eos_positions[1::2]]

    return use_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么user_turns:[11,21]，assistant_ends:[16,26]

    user_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    assistant_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    要想获取到assistant的回答的起始位置，就需要再跳过\n,<|im_start|>,assistant 这三个token，所以需要加3.
    要想获取到assistant的回答的结束位置，就需要往前跳一个<|im_end|>，所以需要减1.
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    for user_end,assistant_end in zip(user_ends,assistant_ends):
        answer_start = user_end + 3
        answer_end = assistant_end - 1
        mask[answer_start:answer_end] = 1

def compute_loss(output_logits,labels,assistant_mask):
    """
    SFT计算损失的函数
    args:
        output_logits: 模型前向传播得到的结果，shape: [batch_size, seq_len, vocab_size]
        labels: 真实答案标签，shape:[batch_size, seq_len]
        assistant_mask: assistant回答的掩码，shape:[batch_size, seq_len]
    """
    
    # 1、对output_logits先做softmax，得到对数概率分布
    # shape: batch_size, seq_len, vocab_size
    log_probs = torch.log_softmax(output_logits,dim=-1)
    # 2、取得模型生成答案中真实标签的概率
    # label_log_probs.shape: batch_size, seq_len
    label_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
        ).squeeze(-1)
    
    negative_label_log_probs = label_log_probs * (-1)
    
    # 做assistant answer 掩码
    masked_label_log_probs = negative_label_log_probs * assistant_mask

    # 当前批次的平均loss
    loss = masked_label_log_probs.sum() / assistant_mask.sum()

    return loss

    

    # 3、将对数概率加起来，除以token总数，得到平均loss，return loss就可以了


from re import L

from torch import detach_copy
from transformers import AutoModelForCausalLM
from torch.optim.adamw import AdamW
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
@dataclass
class SFTConfig:

    lr:float = 1e-5
    batch_size:int = 4
    warmup_ratio:float = 0.1
    log_dir : str = "logs/01_sft_demo"
    log_iter:int = 100
    save_dir : str = "finetuned/01_sft_demo"

def cosine_lr_decay(batch,total_batch,lr,warmup_ratio):
    """
    基于cosine学习率衰减  / 学习率调度器

    """
    warmup_batch = total_batch * warmup_ratio 

    if batch < warmup_batch:
        return batch * lr / warmup_batch
    else:
        progress = (batch - warmup_batch) / (total_batch - warmup_batch)
        # 衰减：0.5 * (1+cos(pi * progress)) 衰减，从1到0
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        return decay * lr



def train(sft_config:SFTConfig):
    # 1、准备模型，数据，优化器
    model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B-Base/")
    model.train()
    model.to("cuda")
    train_data:List[List] = get_train_data()
    optimizer = AdamW(model.parameters(),lr=sft_config.lr)
    total_batch = (len(train_data) + sft_config.batch_size -1) // sft_config.batch_size
    writer = SummaryWriter(log_dir=sft_config.log_dir)
    total_loss_list = []
    progress_bar = tqdm(total=total_batch)
    for batch in range(total_batch):

        # 1、准备张量，input_ids和labels
        current_batch_data = train_data[batch * sft_config.batch_size : (batch+1) * sft_config.batch_size]

        current_batch_max_len = max([len(data) for data in current_batch_data])
        
        for data in current_batch_data:
            padding_length = current_batch_max_len - len(data)
            data.extend([tokenizer.pad_token_id] * padding_length)

        # shape: batch_size, seq_len
        batch_data_tensor = torch.tensor(current_batch_data,dtype=torch.long).to("cuda")
        input_ids = batch_data_tensor[:,:-1]
        labels = batch_data_tensor[:,1:]
        assistant_mask = create_answer_mask(input_ids=input_ids,tokenizer=tokenizer)

        #2、前向传播
        output_logits = model(input_ids).logits

        # 3、计算损失，反向传播，算得梯度
        
        loss = compute_loss(output_logits,labels,assistant_mask)
        total_loss_list.append(loss.item())
        loss.backward()

        # 4、使用优化器更新参数
        current_lr = cosine_lr_decay(batch,total_batch,sft_config.lr,sft_config.warmup_ratio)
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("train/lr",scalar_value=current_lr, global_step=batch)
        should_log = batch % sft_config.log_iter == 0 or batch== total_batch-1
        progress_bar.update(1)
        progress_bar.set_postfix(lr=f"{current_lr:.2f}",loss = f"{loss.item():.4f}")
        if should_log:
            """
            记录一下损失
            """
            loss_list = total_loss_list[-sft_config.log_iter:]
            average_loss = sum(loss_list) / len(loss_list) 
            writer.add_scalar("train/loss",scalar_value=average_loss, global_step=batch)

    return model, tokenizer

def save_model_tokenizer(model, tokenizer,save_dir):

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)



if __name__ == "__main__":
    sft_config = SFTConfig()
    model,tokenizer = train(sft_config)
    save_model_tokenizer(model,tokenizer,sft_config.save_dir)

        



