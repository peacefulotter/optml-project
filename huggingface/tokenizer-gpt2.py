from datasets import load_dataset
from configs import MODEL_CONFIGS, DATASET_CONFIGS
from transformers import AutoTokenizer
import sys
from transformers import pipeline


if __name__ == '__main__':
    def _exit():
        print(f"""
            Usage: tokenizer.py <model> <dataset>
            Models: {MODEL_CONFIGS.keys()}
            Datasets: {DATASET_CONFIGS.keys()}
        """)
        sys.exit(1)

    if len(sys.argv) != 3:
        _exit()

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    if model_name not in MODEL_CONFIGS.keys() or dataset_name not in DATASET_CONFIGS.keys():
        _exit()

    model_config = MODEL_CONFIGS[model_name]
    dataset_config = DATASET_CONFIGS[dataset_name]
    tokenizer_name = model_config['tokenizer_name']
    max_seq_length = model_config['max_seq_length']
    mlm = model_config['mlm']
    path = dataset_config['dataset_path']
    name = dataset_config['dataset_name']

    train_dataset = load_dataset(path, name, split="train")
    validation_dataset = load_dataset(path, name, split="validation")

    #dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

    batch_size = 512
    all_texts = [train_dataset[i : i + batch_size]["text"] for i in range(0, len(train_dataset), batch_size)]


    def batch_iterator():
        for i in range(0, len(train_dataset), batch_size):
            yield train_dataset[i : i + batch_size]["text"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=25000)

    train_encoding = tokenizer(train_dataset["text"], padding=True, truncation=True, max_length=1024, return_tensors='pt')
    eval_encoding = tokenizer(validation_dataset["text"], padding=True, truncation=True, max_length=1024, return_tensors='pt')


    new_tokenizer.save_pretrained(f'./save/{path}/{name}/tokenizer/{model_name}/')
    example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    encoding = tokenizer(example)
    print(encoding.tokens())
    print(encoding.word_ids())

    # train_encoding.save_to_disk(f'./save/{path}/{name}/datasets/{model_name}/')
    # eval_encoding.save_to_disk(f'./save/{path}/{name}/datasets/{model_name}/')