import json
import math
import wandb
import torch
import gc, sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from datasets import load_from_disk
from huggingface_hub import login
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from configs import (
    SEED, 
    MODEL_CONFIGS, 
    DATASET_CONFIGS,
    OPTIMIZER_CONFIGS
)
from collections.abc import Mapping
from datetime import datetime as dt
import evaluate
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

IGNORE_INDEX = -100
PAD_TOKEN = '[PAD]'

perplexity = evaluate.load("perplexity", module_type="metric") # str[]
exact_match = evaluate.load("exact_match") # str[]
accuracy = evaluate.load("accuracy") # int32[] 
rouge = evaluate.load('rouge') # str[]
bleu = evaluate.load("bleu") # str[]

def _prepare_input(data):
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": 'cuda:0'}
        return data.to(**kwargs)
    return data

def get_metrics_callback(dataset_tr, dataset_ev, data_collator):
    batch_size = 32
    def build_dataloader(dataset):
        return DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=8), len(dataset)

    dataloader_tr = build_dataloader(dataset_tr)
    dataloader_ev = build_dataloader(dataset_ev)

    def on_evaluate(training: bool, **kwargs):
        name = 'training' if training else 'eval'
        dataloader, len_dataset = dataloader_tr if training else dataloader_ev
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']

        predictions = torch.empty((len_dataset, 512), dtype=torch.int32, device='cuda:0')
        references = torch.empty((len_dataset, 512), dtype=torch.int32, device='cuda:0')

        for step, inputs in enumerate(dataloader):
            inputs = _prepare_input(inputs)
            outputs = model(**inputs)
            logits = outputs['logits']
            pred_ids = torch.argmax(logits, dim=-1)
            labels_ids = inputs['labels']
            start = step * batch_size
            end = start + pred_ids.shape[0]
            predictions[start:end] = pred_ids 
            references[start:end] = labels_ids 
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == IGNORE_INDEX] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        pred_ids_flat = pred_ids.view(-1)
        labels_ids_flat = labels_ids.view(-1)

        res = {
            **rouge.compute(predictions=pred_str, references=label_str),
            **accuracy.compute(predictions=pred_ids_flat, references=labels_ids_flat),
            **exact_match.compute(predictions=pred_str, references=label_str),
            'bleu': bleu.compute(predictions=pred_str, references=[[seq] for seq in label_str])['bleu'],
            'perplexity': perplexity.compute(predictions=pred_str, model_id='gpt2')['mean_perplexity']
        }
        res = { f'{name}/{k}': v for k, v in res.items() }
        print(res)
        wandb.log( res, commit=False )
    
    class MetricsCallback(TrainerCallback):
        @torch.no_grad()
        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            # _evaluate(True, **kwargs)
            on_evaluate(False, **kwargs)
            
    return MetricsCallback

class LogCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
        # logs = state.log_history
        # if 'loss' in logs[-1].keys():
        #     loss = logs[-1]['loss']
        #     pp = math.exp(loss)
        #     wandb.log({'train/perplexity': pp}, commit=False)




def compute_metrics(tokenizer):
    def inner(pred):  
        pred_ids = torch.from_numpy(pred.predictions[0])
        labels_ids = torch.from_numpy(pred.label_ids)
        pred_ids.to('cuda:0')
        labels_ids.to('cuda:0')

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == IGNORE_INDEX] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        pred_ids_flat = pred_ids.view(-1)
        labels_ids_flat = labels_ids.view(-1)

        res = {
            **bleu.compute(predictions=pred_str, references=[[seq] for seq in label_str]),
            **rouge.compute(predictions=pred_str, references=label_str),
            **accuracy.compute(predictions=pred_ids_flat, references=labels_ids_flat),
            **exact_match.compute(predictions=pred_str, references=label_str),
            'perplexity': perplexity.compute(predictions=pred_str, model_id='gpt2')['mean_perplexity']
        }
        print(res)
        return res
    return inner

# def preprocess_logits_for_metrics(logits, labels):
#     """
#     Original Trainer may have a memory leak. 
#     This is a workaround to avoid storing too many tensors that are not needed.
#     """
#     pred_ids = torch.argmax(logits, dim=-1)
#     return pred_ids, labels

def train(model_name, dataset_name, optimizer_name, lr=None):
    # login()

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

    mlm = model_config['mlm']
    model = model_config['model'](tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
    lr = lr if lr is not None else optimizer_config['default-lr']
    optimizer: Optimizer = optimizer_config['build'](model, lr=lr)
    
    time_now = dt.now().strftime("%m/%d/%Y, %H:%M:%S")
    save_path = f'./save/{model_name}/{name}/{path}/{optimizer_name}/{lr}/{time_now}'

    training_args = TrainingArguments(
        output_dir=f'{save_path}/output/',
        logging_dir=f'{save_path}/logs/',
        evaluation_strategy = 'steps',
        gradient_accumulation_steps=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        seed=SEED,
        bf16=True,
        bf16_full_eval=True,
        # push_to_hub=True,
        eval_steps=100,
        run_name=f"{model_name}-{dataset_name}-{optimizer_name}-{lr}-{time_now}",
        # report_to='none'
    )

    MetricCB = get_metrics_callback(
        tokenized_datasets["train"], 
        tokenized_datasets["validation"], 
        data_collator
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
        callbacks=[MetricCB, LogCallback],
        # compute_metrics=compute_metrics(tokenizer),
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    torch.cuda.empty_cache()
    gc.collect()

    trainer.train()
    trainer.save_model(f"{save_path}/model/")
    # trainer.push_to_hub()

    eval_results = trainer.evaluate()
    print(f"{optimizer_name} {optimizer.defaults['lr']} - results: {eval_results}")


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

    train(model_name, dataset_name, optimizer_name, lr)
       

    