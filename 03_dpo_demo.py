from transformers import AutoTokenizer
from transformers.models.qwen3 import Qwen3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("finetuned/01_sft_demo")


from typing import Dict,List
def get_train_data(dpo_config):
    """
    加载数据，调用chat template方法，得到分词之后的input_ids；后面在train主循环当中会调用该方法，获取训练数据
    """
    from datasets import load_dataset

    train_data = load_dataset("data/ultrafeedback_binarized")["train_prefs"]
    chosen_result_list = []
    rejected_result_list = []

    for i in range(dpo_config.train_data_size):
        chosen_message_list:List = train_data[i]["chosen"]
        chosen_message_list.insert(0,{"role":"system","content":"you are a helpfule assistant"})

        chosen_result: Dict[str, List]= tokenizer.apply_chat_template(chosen_message_list,tokenize=True,truncation=True,max_length =2500)
        chosen_result_list.append(chosen_result["input_ids"])

        rejected_message_list:List = train_data[i]["rejected"]
        rejected_message_list.insert(0,{"role":"system","content":"you are a helpfule assistant"})

        rejected_result: Dict[str, List]= tokenizer.apply_chat_template(rejected_message_list,tokenize=True,truncation=True,max_length =2500)
        rejected_result_list.append(rejected_result["input_ids"])

    return chosen_result_list,rejected_result_list


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
    if num_user_turns == num_assistant_turns:
        for user_end,assistant_end in zip(user_ends,assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

    elif num_user_turns == num_assistant_turns + 1:
        for user_end,assistant_end in zip(user_ends[:-1],assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1] 
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1

def _compute_log_prob(output_logits, labels, assistant_mask):
    """
    用于计算样本的对数概率
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

    
    # 做assistant answer 掩码
    masked_label_log_probs = label_log_probs * assistant_mask

    # 对样本长度做归一化: 将每个样本的所有log_probs加起来，除以样本中，有效token的数量，就得到了归一化之后的对数概率
    average_log_prob = masked_label_log_probs.sum(dim = -1) / assistant_mask.sum(dim = -1)

    return average_log_prob

def compute_loss(chosen_log_prob, rejected_log_prob, reference_chosen_log_prob, reference_rejected_log_prob, beta):
    """
    DPO计算损失的函数
    args:
        chosen_log_prob: 当前模型输出chosen的对数概率，shape: (batch_size,)
        rejected_log_prob：当前模型输出rejected的对数概率，shape(batch_size,)
        reference_chosen_log_prob: 参考模型输出chosen的对数概率，shape: (batch_size,)
        reference_rejected_log_prob：参考模型输出rejected的对数概率，shape: (batch_size,)
        beta: 用于控制模型区分chosen和rejected的强度
    """
    
    margin = (chosen_log_prob - rejected_log_prob) - (reference_chosen_log_prob - reference_rejected_log_prob)
    # loss.shape:(batch_size, )
    loss =   -torch.nn.functional.logsigmoid( beta * margin)

    loss = loss.sum() / len(loss)

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
class DPOConfig:

    lr:float = 1e-6
    batch_size:int = 3
    warmup_ratio:float = 0.1
    log_dir : str = "logs/01_sft_demo"
    log_iter:int = 100
    save_dir : str = "finetuned/02_dpo_demo"
    train_data_size:int = 5000
    beta:float = 0.1

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



def train(dpo_config:DPOConfig):
    # 1、准备模型，数据，优化器
    model = AutoModelForCausalLM.from_pretrained("finetuned/01_sft_demo")
    ref_model = AutoModelForCausalLM.from_pretrained("finetuned/01_sft_demo")
    model.train()
    ref_model.eval()
    model.to("cuda")
    ref_model.to("cuda")
    chosen_train_data, rejected_train_data= get_train_data(dpo_config)
    optimizer = AdamW(model.parameters(),lr=dpo_config.lr)
    total_batch = (len(chosen_train_data) + dpo_config.batch_size -1) // dpo_config.batch_size
    writer = SummaryWriter(log_dir=dpo_config.log_dir)
    total_loss_list = []
    progress_bar = tqdm(total=total_batch)
    for batch in range(total_batch):

        # 1、准备张量，chosen_input_ids和chosen_labels、chosen_assistant_mask, rejected_input_ids和rejected_labels,rejected_assistant_mask
        current_chosen_batch_data = chosen_train_data[batch * dpo_config.batch_size : (batch+1) * dpo_config.batch_size]

        current_chosen_max_len = max([len(data) for data in current_chosen_batch_data])
        
        for data in current_chosen_batch_data:
            padding_length = current_chosen_max_len - len(data)
            data.extend([tokenizer.pad_token_id] * padding_length)

        # shape: batch_size, seq_len
        chosen_batch_data_tensor = torch.tensor(current_chosen_batch_data,dtype=torch.long).to("cuda")
        chosen_input_ids = chosen_batch_data_tensor[:,:-1]
        chosen_labels = chosen_batch_data_tensor[:,1:]
        chosen_assistant_mask = create_answer_mask(input_ids=chosen_input_ids,tokenizer=tokenizer)


        current_rjected_batch_data = rejected_train_data[batch * dpo_config.batch_size : (batch+1) * dpo_config.batch_size]

        current_rejected_max_len = max([len(data) for data in current_rjected_batch_data])
        
        for data in current_rjected_batch_data:
            padding_length = current_rejected_max_len - len(data)
            data.extend([tokenizer.pad_token_id] * padding_length)

        # shape: batch_size, seq_len
        rejected_batch_data_tensor = torch.tensor(current_rjected_batch_data,dtype=torch.long).to("cuda")
        rejected_input_ids = rejected_batch_data_tensor[:,:-1]
        rejected_labels = rejected_batch_data_tensor[:,1:]
        rejected_assistant_mask = create_answer_mask(input_ids=rejected_input_ids,tokenizer=tokenizer)



        #2、前向传播
        # logis.shape: batch_size, seq_len, vocab_size
        chosen_output_logits = model(chosen_input_ids).logits
        rejected_output_logits = model(rejected_input_ids).logits

        with torch.no_grad():
            reference_chosen_output_logits = ref_model(chosen_input_ids).logits
            reference_rejected_output_logits = ref_model(rejected_input_ids).logits
            

        # 3、计算损失，反向传播，算得梯度
        
        chosen_log_prob = _compute_log_prob(chosen_output_logits, chosen_labels,chosen_assistant_mask)
        rejected_log_prob = _compute_log_prob(rejected_output_logits, rejected_labels,rejected_assistant_mask)

        reference_chosen_log_prob = _compute_log_prob(reference_chosen_output_logits, chosen_labels,chosen_assistant_mask)
        reference_rejected_log_prob = _compute_log_prob(reference_rejected_output_logits, rejected_labels,rejected_assistant_mask)

        loss = compute_loss(chosen_log_prob,rejected_log_prob, reference_chosen_log_prob, reference_rejected_log_prob,dpo_config.beta)
        total_loss_list.append(loss.item())
        loss.backward()

        # 4、使用优化器更新参数
        current_lr = cosine_lr_decay(batch,total_batch,dpo_config.lr,dpo_config.warmup_ratio)
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("train/lr",scalar_value=current_lr, global_step=batch)
        should_log = batch % dpo_config.log_iter == 0 or batch== total_batch-1
        progress_bar.update(1)
        progress_bar.set_postfix(lr=f"{current_lr:.2e}",loss = f"{loss.item():.4f}")
        if should_log:
            """
            记录一下损失
            """
            loss_list = total_loss_list[-dpo_config.log_iter:]
            average_loss = sum(loss_list) / len(loss_list) 
            writer.add_scalar("train/loss",scalar_value=average_loss, global_step=batch)

    return model, tokenizer

def save_model_tokenizer(model, tokenizer,save_dir):

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)



if __name__ == "__main__":
    dpo_config = DPOConfig()
    model,tokenizer = train(dpo_config)
    save_model_tokenizer(model,tokenizer,dpo_config.save_dir)

        



