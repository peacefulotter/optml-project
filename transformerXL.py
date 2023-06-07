import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import numericalize_tokens_from_iterator

from transformer_xl_model import TransformerXLModel  # Custom Transformer-XL model implementation


class TransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads):
        super(TransformerXLModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerXL(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs, input_positions, input_masks):
        embedded = self.embedding(inputs)
        output = self.transformer(embedded, input_positions, input_masks)
        output = self.fc(output)
        return output


# Hyperparameters
embedding_dim = 512
hidden_dim = 512
num_layers = 6
num_heads = 8
batch_size = 16
lr = 0.001
epochs = 10

# Dataset preparation
tokenizer = get_tokenizer('basic_english')
train_dataset, valid_dataset, test_dataset = WikiText103()

vocab = build_vocab_from_iterator(map(tokenizer, train_dataset))
vocab.set_default_index(vocab['<unk>'])

train_data = numericalize_tokens_from_iterator(map(tokenizer, train_dataset), vocab)
valid_data = numericalize_tokens_from_iterator(map(tokenizer, valid_dataset), vocab)
test_data = numericalize_tokens_from_iterator(map(tokenizer, test_dataset), vocab)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Model initialization
model = TransformerXLModel(len(vocab), embedding_dim, hidden_dim, num_layers, num_heads)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in train_loader:
        optimizer.zero_grad()

        inputs = batch[:-1]
        targets = batch[1:]

        # Generate positional encodings
        input_masks = (inputs != vocab['<pad>']).unsqueeze(-2)
        input_positions = torch.arange(inputs.size(1), dtype=torch.long, device=inputs.device)
        input_positions = input_positions.unsqueeze(0).expand_as(inputs)

        output = model(inputs, input_positions, input_masks)
        output = output.view(-1, len(vocab))
        targets = targets.view(-1)

        loss = criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    print(f"Epoch: {epoch}, Training Loss: {avg_loss}")

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_tokens = 0

        for batch in valid_loader:
            inputs = batch[:-1]
            targets = batch[1:]

            input_masks = (inputs != vocab['<pad>']).unsqueeze(-2)
            input_positions = torch.arange(inputs.size(1), dtype=torch.long, device=inputs.device)
            input_positions = input_positions.unsqueeze(0).expand_as(inputs)

            output = model(inputs, input_positions, input_masks)
            output = output.view(-1, len(vocab))
            targets = targets.view(-1)

            loss = criterion(output, targets)

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens
        print(f"Epoch: {epoch}, Validation Loss: {avg_loss}")
