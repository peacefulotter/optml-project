
SEED = 42

TRAINING_CONFIGS = {
    'bert-wikitext': {
        'tokenizer_name': 'bert-base-cased',
        'dataset_path': 'wikitext', # 'openwebtext'
        'dataset_name': 'wikitext-103-raw-v1',
        # 'learning_rates': np.linspace(1e-4, 1e-5)
    }
}