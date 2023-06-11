from transformers import (
    BertConfig,
    BertForMaskedLM, 
    T5Config,
    # T5ForMaskedLM
)

def get_bert_model(**kwargs):
    def cb(tokenizer):
        config = BertConfig(
            vocab_size=len(tokenizer),
            **kwargs
        )
        return BertForMaskedLM(config)
    return cb

def get_t5_model(**kwargs):
    def cb(tokenizer):
        config = T5Config(
            vocab_size=len(tokenizer),
            **kwargs
        )
        raise NotImplementedError()
        # return T5ForMaskedLM(config)
    return cb

def not_implemented():
    raise NotImplementedError()

SEED = 42

OPTIMIZER_CONFIGS = {
    'adam': { },
    'lion': { },
    'sophia': { },
    'signsgd': { },
}

DATASET_CONFIGS = {
    'wikitext': {
        'dataset_path': 'wikitext',
        'dataset_name': 'wikitext-103-raw-v1'
    }
}

MODEL_CONFIGS = {
    'mini-bert': {
        'max_seq_length': 512,
        'tokenizer_name': 'bert-base-cased',
        'model': get_bert_model(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=3072
        )
    },
    'bert': {
        'max_seq_length': 512,
        'tokenizer_name': 'bert-base-cased',
        'model': get_bert_model()
    },
    'gpt2': {
        'max_seq_length': 512, # TODO: check max_seq_length
        'tokenizer_name': 'gpt2',
        'model': not_implemented # gpt2
    },
    't5': {
        'max_seq_length': 512, # TODO: check max_seq_length
        'tokenizer_name': 't5-small',
        'model': get_t5_model # t5
    }
}

