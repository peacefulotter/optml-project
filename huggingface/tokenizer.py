from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

# Import configs
from configs import TRAINING_CONFIGS
config = TRAINING_CONFIGS['bert-wikitext']
tokenizer_name = config['tokenizer_name']
path = config['dataset_path']
name = config['dataset_name']

# Load dataset
max_seq_length = 512 # model dependent
raw_datasets = load_dataset(path, name)
column_names = list(raw_datasets["train"].features) # Evaluation: column_names = list(raw_datasets["validation"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000][text_column_name]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

# Use pretrained tokenizer and train it on new corpus
training_corpus = get_training_corpus()
old_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
# TODO: replace 52000 with correct vocab_size, len(old_tokenizer)?
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=52000)

def tokenize_function(examples):
    return tokenizer(
        examples[text_column_name], 
        truncation=True, 
        max_length=max_seq_length, 
        return_special_tokens_mask=True
    )

# Tokenize dataset
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on every text in dataset",
)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# Group in batch the tokenized dataset
tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=8,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {max_seq_length}",
)

tokenizer.save(f'./save/{path}/{name}/tokenizer/tokenizer.json')
tokenized_datasets.save_to_disk(f'./save/{path}/{name}/datasets/')