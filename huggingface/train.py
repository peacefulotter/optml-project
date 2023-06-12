import json
import math
import wandb
import torch
import gc, os, sys
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
from configs import (
    SEED, 
    MODEL_CONFIGS, 
    DATASET_CONFIGS,
    OPTIMIZER_CONFIGS
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

def compute_metric(tokenizer):
    def inner(pred):
        logits = torch.from_numpy(pred.predictions)
        labels = torch.from_numpy(pred.label_ids)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
        pp = math.exp(loss)
        wandb.log({'train/perplexity': pp, 'train/calculated_loss': loss}, commit=False)
        return {'train/perplexity': pp, 'calculated_loss': loss}
    return inner

def train(model_name, dataset_name, optimizer_name, lr=None):
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
        sep_token=special_tokens['sep_token'],
        cls_token=special_tokens['cls_token'],
        mask_token=special_tokens['mask_token'],
        unk_token=special_tokens['unk_token'],
        pad_token=special_tokens['pad_token'],
        tokenizer_file=f'./save/{path}/{name}/tokenizer/{model_name}/tokenizer.json',
    )

    model = model_config['model'](tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./{model_name}/output/',
        logging_dir=f'./{model_name}/logs/',
        evaluation_strategy = 'steps',
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        seed=SEED,
        bf16=True,
        bf16_full_eval=True
    )

    lr = lr if lr is not None else optimizer_config['default-lr']
    optimizer: Optimizer = optimizer_config['build'](model, lr=lr)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric(tokenizer),
        optimizers=(optimizer, None)
    )

    torch.cuda.empty_cache()
    gc.collect()

    trainer.train()
    trainer.save_model(f"./{model_name}/output/{optimizer.__class__.__name__}")

    eval_results = trainer.evaluate()
    print(f"{optimizer.__class__.__name__} {optimizer.defaults.lr} - results: {eval_results}")


def train_on_all(model_name, dataset_name):
    for optim_name, optimizer in OPTIMIZER_CONFIGS.items():
        for lr in optimizer['lrs']:
            train(model_name, dataset_name, optim_name, lr=lr)


if __name__ == "__main__":
    def _exit():
        print(f"""
            Usage: train.py <model> <dataset> <optimizer> \n
            Models: {MODEL_CONFIGS.keys()}\n
            Datasets: {DATASET_CONFIGS.keys()}\n
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

    train(model_name, dataset_name, optimizer_name, lr)
       

    