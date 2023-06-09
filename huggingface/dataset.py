from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer_name = 'bert-base-cased'
path = 'wikitext' # 'openwebtext'
name = 'wikitext-103-raw-v1'



max_seq_length = 512 # model dependent
raw_datasets = load_dataset(path, name)
column_names = list(raw_datasets["train"].features)
# Evaluation: column_names = list(raw_datasets["validation"].features)
text_column_name = "text" if "text" in column_names else column_names[0]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def tokenize_function(examples):
    return tokenizer(
        examples[text_column_name], 
        truncation=True, 
        max_length=max_seq_length, 
        return_special_tokens_mask=True
    )

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

tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=8,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {max_seq_length}",
)

tokenized_datasets.save_to_disk(f'./datasets/{path}/{name}')