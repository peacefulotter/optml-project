
SEED = 42

TRAINING_CONFIGS = {
    'bert-wikitext': {
        'max_seq_length': 512,
        'tokenizer_name': 'bert-base-cased',
        'dataset_path': 'wikitext',
        'dataset_name': 'wikitext-103-raw-v1',
        'model' : 'bert'
        # 'learning_rates': np.linspace(1e-4, 1e-5)
    },
    'gpt2-wikitext':{
        'max_seq_length':512,
        'tokenizer_name': 'gpt2',
        'dataset_path': 'wikitext',
        'dataset_name': 'wikitext-103-raw-v1',
        'model' : 'gpt2'
    },
    't5-wikitext': {
        'max_seq_length':512,
        'tokenizer_name':'t5-small',
        'dataset_path':'wikitext',
        'dataset_name': 'wikitext-103-raw-v1',
        'model' : 't5'
    }
}