from transformers import Trainer
from transformers import TrainingArguments

model = AutoModelWithLMHead.from_config(config)
# model.resize_token_embeddings(len(tokenizer))
optim = Lion
training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)