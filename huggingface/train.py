from datasets import load_from_disk
from transformers import (
    CONFIG_MAPPING,
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    AutoModelWithLMHead, 
    Trainer, 
    TrainingArguments
)

max_seq_length = 512

tokenized_datasets = load_from_disk('./datasets/wikitext/wikitext-103-raw-v1')
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

config = CONFIG_MAPPING['bert']()
model = AutoModelWithLMHead.from_config(config)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)
# model.resize_token_embeddings(len(tokenizer))
# optim = Lion
training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()