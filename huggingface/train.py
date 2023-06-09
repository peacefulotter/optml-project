import math
import torch
import torch.nn.functional as F
from lion import Lion
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    BertConfig,
    BertForMaskedLM, 
    Trainer, 
    TrainingArguments
)

seed = 42

tokenized_datasets = load_from_disk('./datasets/wikitext/wikitext-103-raw-v1')
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

config = BertConfig(vocab_size=len(tokenizer))
model = BertForMaskedLM(config)
# model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)
# optim = Lion
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
    seed=seed,
    fp16=True,
    eval_accumulation_steps=50,
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