import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Define the LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, _) = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden)

# Returns timestamped filename like "train_log_2025-05-10_0314.txt"
def generate_log_filename(prefix="log"):
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M')}.txt"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Config
embed_dim = 100
hidden_dim = 128
output_dim = 2
batch_size = 64
num_epochs = 5

# Prepare tokenizer and vocab
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

print("Building vocabulary...")
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# Prepare data pipeline
label_map = {'pos': 1, 'neg': 0}

def collate_batch(batch):
    text_list, label_list = [], []
    for label, text in batch:
        processed_text = torch.tensor(vocab(tokenizer(text)), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(label_map[label])
    return pad_sequence(text_list, batch_first=True), torch.tensor(label_list, dtype=torch.int64)

train_iter = IMDB(split='train')
train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# Initialize model, optimizer, loss
model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training model...")
training_start = time.perf_counter()

log_filename = generate_log_filename("train_log")
with open(log_filename, 'w') as log_file:
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0

        for batch_idx, (data_batch, targets) in enumerate(train_loader):
            data_batch = data_batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(data_batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

        epoch_end = time.perf_counter()
        epoch_duration = epoch_end - epoch_start
        print(f"Epoch {epoch} average loss: {epoch_loss / (batch_idx + 1):.4f}, duration: {epoch_duration:.2f} seconds")
        log_file.write(f"Epoch {epoch}: {epoch_duration:.2f} sec\n")

    training_end = time.perf_counter()
    total_duration = training_end - training_start
    print(f"Total training time: {total_duration:.2f} seconds")
    log_file.write(f"Total training time: {total_duration:.2f} seconds\n")
    log_file.write(f"Average epoch duration: {total_duration / num_epochs:.2f} seconds\n")

print(f"Training log saved to {log_filename}")

# Save model
torch.save(model.state_dict(), "sentiment_lstm_preloaded.pth")
print("Training complete. Model saved to sentiment_lstm_preloaded.pth")
