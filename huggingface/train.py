import json
import math
import wandb
import torch
import torch.nn.functional as F
from optimizers import Lion, Sophia, SignSGD
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
from torch.optim import AdamW
# Import training configs
from configs import SEED, MODEL_CONFIGS, DATASET_CONFIGS, OPTIMIZER_CONFIGS
from datetime import datetime
import gc, os, sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

def compute_custom_metric(pred):
    logits = torch.from_numpy(pred.predictions)
    labels = torch.from_numpy(pred.label_ids)
    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
    pp = math.exp(loss)
    wandb.log({'train/perplexity': pp, 'train/calculated_loss': loss}, commit=False)
    return {'train/perplexity': pp, 'calculated_loss': loss}

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage : train.py <model> <dataset> <optimizer> \nModels : 't5','bert','gpt2' \nDatasets : 'wikitext'\nOptimizers: 'adam', 'lion', 'sophia', 'signSGD'")
        sys.exit(1)
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    optimizer_name = sys.argv[3]
    if (
        model_name not in MODEL_CONFIGS.keys() or 
        dataset_name not in DATASET_CONFIGS.keys() or 
        optimizer_name not in OPTIMIZER_CONFIGS.keys()
    ):
        print(f"""
            Usage: train.py <model> <dataset> <optimizer> \n
            Models: {MODEL_CONFIGS.keys()}\n
            Datasets: {DATASET_CONFIGS.keys()}\n
            Optimizers: {OPTIMIZER_CONFIGS.keys()}
        """)
        sys.exit(1)

    model_config = MODEL_CONFIGS[model_name]
    dataset_config = DATASET_CONFIGS[dataset_name]
    tokenizer_name = model_config['tokenizer_name']
    path = dataset_config['dataset_path']
    name = dataset_config['dataset_name']

    with open(f'./save/{path}/{name}/tokenizer/{model_name}/special_tokens_map.json') as f:
        special_tokens = json.load(f)
    tokenized_datasets = load_from_disk(f'./save/{path}/{name}/datasets/{model_name}/')
    tokenizer = PreTrainedTokenizerFast(
        # TODO: make sure these are set for MASKED models
        # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
        sep_token=special_tokens['sep_token'],
        cls_token=special_tokens['cls_token'],
        mask_token=special_tokens['mask_token'],
        unk_token=special_tokens['unk_token'],
        pad_token=special_tokens['pad_token'],
        tokenizer_file=f'./save/{path}/{name}/tokenizer/{model_name}/tokenizer.json',
    )
    print(tokenizer.sep_token, tokenizer.cls_token, tokenizer.mask_token, tokenizer.unk_token, tokenizer.pad_token)

    model = model_config['model'](tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    training_args = TrainingArguments(
        output_dir=f'./{model_name}/output/',
        logging_dir=f'./{model_name}/logs/',
        evaluation_strategy = 'epoch',
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        seed=SEED,
        bf16=True,
        bf16_full_eval=True
    )

    # TODO: learning_rate=?? optim dependant
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

    torch.cuda.empty_cache()
    gc.collect()

    # wandb.init(name="BERT on Wikitext "+ datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    # wandb.define_metric('train/perplexity')
    # wandb.define_metric('train/calculated_loss')  

    trainer.train()
    trainer.save_model(f"./{model_name}/output/{optimizer.__class__.__name__}")
    # evaluate the model
    eval_results = trainer.evaluate()
    #print eval results + name of optimizer
    print(f"{optimizer.__class__.__name__} results: {eval_results}")