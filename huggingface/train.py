import math
import torch
import torch.nn.functional as F
from lion import Lion
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    BertConfig,
    BertForMaskedLM, 
    Trainer, 
    TrainingArguments
)


# Import training configs
from configs import SEED, TRAINING_CONFIGS
config = TRAINING_CONFIGS['bert-wikitext']
tokenizer_name = config['tokenizer_name']
path = config['dataset_path']
name = config['dataset_name']


tokenized_datasets = load_from_disk(f'./save/{path}/{name}/datasets/')
tokenizer = PreTrainedTokenizerFast(
    # TODO: make sure these are set for MASKED models
    # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
    # sep_token="",
    # cls_token="",
    # mask_token="",
    tokenizer_file=f'./save/{path}/{name}/tokenizer/tokenizer.json',
)
print(tokenizer.sep_token, tokenizer.cls_token, tokenizer.mask_token)

config = BertConfig(vocab_size=len(tokenizer))
model  = BertForMaskedLM(config) # model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)

def compute_custom_metric(pred):
    logits = torch.from_numpy(pred.predictions)
    labels = torch.from_numpy(pred.label_ids)
    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
    return {'perplexity': math.exp(loss), 'calculated_loss': loss}

training_args = TrainingArguments(
    output_dir='./bert/output/',
    evaluation_strategy = 'epoch',
    # learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # warmup_steps=500,
    # weight_decay=0.01,
    logging_dir='./bert/logs/',
    seed=SEED,
    fp16=True,
    eval_accumulation_steps=50
)

optimizer = Lion(model.parameters())

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_custom_metric,
    optimizers=(optimizer, None)
)
trainer.train()