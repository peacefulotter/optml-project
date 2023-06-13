import sys
import json
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    GPT2LMHeadModel,
    pipeline,
    set_seed
)
from configs import (
    SEED, 
    MODEL_CONFIGS, 
    DATASET_CONFIGS,
    OPTIMIZER_CONFIGS
)

IGNORE_INDEX = -100
PAD_TOKEN = '[PAD]'


def test(model_name, dataset_name, optimizer_name, lr=None):

    model_config = MODEL_CONFIGS[model_name]
    dataset_config = DATASET_CONFIGS[dataset_name]
    optimizer_config = OPTIMIZER_CONFIGS[optimizer_name]
    # tokenizer_name = model_config['tokenizer_name']
    path = dataset_config['dataset_path']
    name = dataset_config['dataset_name']

    with open(f'./save/{path}/{name}/tokenizer/{model_name}/special_tokens_map.json') as f:
        special_tokens = json.load(f)
    
    tokenized_datasets = load_from_disk(f'./save/{path}/{name}/datasets/{model_name}/')
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f'./save/{path}/{name}/tokenizer/{model_name}/tokenizer.json',
        verbose=False,
        **special_tokens
    )
    tokenizer.pad_token = PAD_TOKEN

    model = model_config['model'](tokenizer)
    lr = lr if lr is not None else optimizer_config['default-lr']
    
    dataset = tokenized_datasets['test']

    load_path = './save/gpt2/output/AdamW/' # f'./save/{model_name}/{name}/{path}/{optimizer_name}/{lr}/{time_now}'
    model = GPT2LMHeadModel.from_pretrained(load_path)
    set_seed(42)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', device='cuda:0', max_length=30, num_return_sequences=5)
    sentence = 'Paris is the capital of '
    res = pipe(sentence)
    print(res)

    generator = pipeline('text-generation', model='gpt2')
    res = generator(sentence, max_length=30, num_return_sequences=5)
    print(res)

if __name__ == "__main__":
    def _exit():
        print(f"""
            Usage: train.py <model> <dataset> <optimizer>
            Models: {MODEL_CONFIGS.keys()}
            Datasets: {DATASET_CONFIGS.keys()}
            Optimizers: {OPTIMIZER_CONFIGS.keys()}
            lr: float > 0
        """)
        sys.exit(1)

    l = len(sys.argv)
    if l < 4:
        _exit()

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    optimizer_name = sys.argv[3]
    lr = sys.argv[4] if l >= 5 else None
    if (
        model_name not in MODEL_CONFIGS.keys() or 
        dataset_name not in DATASET_CONFIGS.keys() or 
        optimizer_name not in OPTIMIZER_CONFIGS.keys()
    ):
        _exit()

    test(model_name, dataset_name, optimizer_name, lr)
       

    