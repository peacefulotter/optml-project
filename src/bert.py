from transformers import BertConfig, BertModel

# Initializing a BERT bert-base-uncased style configuration

configuration = BertConfig()

# Initializing a model (with random weights) from the bert-base-uncased style configuration

model = BertModel(configuration)

# Accessing the model configuration

configuration = model.config

print(configuration)
print(model)