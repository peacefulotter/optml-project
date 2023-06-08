

from transformers import AutoTokenizer, BertModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

print(inputs)
print(outputs)
print(last_hidden_states)