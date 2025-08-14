# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

# %%
# 3. Imports
import torch
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import transformers
from datasets import Dataset


# %%


# %%
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)

# %%
from transformers import TrainingArguments
print(TrainingArguments.__module__)

# %%
from peft import LoraConfig
from trl import SFTTrainer

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# %%
# # 4. Load and Prepare Data (Replace this with your dataset path)
# data = [
#     {
#         "instruction": "Classify the type of adverse event as Death, Injury, or Device Malfunction.",
#         "input": "A Mynxgrip vascular closure device failed to deploy properly...",
#         "output": "Device Malfunction"
#     },
#     {
#         "instruction": "Classify the type of adverse event as Death, Injury, or Device Malfunction.",
#         "input": "The patient died due to hemorrhagic shock following surgery.",
#         "output": "Death"
#     },
#     {
#         "instruction": "Classify the type of adverse event as Death, Injury, or Device Malfunction.",
#         "input": "Patient suffered a severe allergic reaction but recovered.",
#         "output": "Injury"
#     }
# ]
# df = pd.DataFrame(data)

# %%
# df

# %%
train_df = pd.read_json('phase1_train.jsonl', lines=True)
val_df=pd.read_json("phase1_val.jsonl",lines=True)

# %%
train_df.head(3)

# %%
val_df.head(3)

# %%
# 5. Prompt Functions
def generate_prompt(row):
    return f"""{row['instruction']}

Event: {row['input']}

Answer: {row['output']}"""

# %%
train_df['text'] = train_df.apply(generate_prompt, axis=1)

# %%
val_df["text"] = val_df.apply(generate_prompt, axis=1)


# %%
val_df.head(1)

# %%
train_df.head(1)

# %%
train_data = Dataset.from_pandas(train_df[["text"]])
eval_data = Dataset.from_pandas(val_df[["text"]])

# %%
train_data

# %%
model_name = "microsoft/Phi-3-mini-4k-instruct"

# %%
compute_dtype = getattr(torch, "float16")

# %%


# 6. Load Model and Tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 2048  # 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, max_seq_length=max_seq_length)
tokenizer.pad_token = tokenizer.eos_token




# %%
print(model.device)

# %%
from accelerate import infer_auto_device_map
device_map = infer_auto_device_map(model)
print(device_map)

# %%
# 7. Setup PEFT (LoRA) Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)


# %%

# 8. TrainingArguments — GPU Efficient
training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=4,
    per_device_train_batch_size=1,               # smallest to fit memory
    gradient_accumulation_steps=4,               # simulate batch size = 4
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    gradient_checkpointing=True,                 # to saves memory at cost of compute with 6gb vram 
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    eval_strategy ="epoch",
)



# %%
def formatting_func(example):
    return example["text"]

# %%
train_data

# %%
def formatting_func(example):
    return example["text"]

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
    formatting_func=formatting_func,
    peft_config=peft_config,
    args=training_arguments
)

# 10. Train Model
trainer.train()
trainer.model.save_pretrained("trained-model")


