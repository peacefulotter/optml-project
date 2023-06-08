from transformers import Trainer, TrainingArguments
from transformers import BertConfig, BertTokenizer, BertModel

# Initializing a BERT bert-base-uncased style configuration
configuration = BertConfig()

# Initializering BERT Tokenizer
tokenizer = BertTokenizer(vocab_file="")

# Initializing a model (with random weights) from the bert-base-uncased style configuration
model = BertModel(configuration)

# Accessing the model configuration
configuration = model.config
print(configuration)


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