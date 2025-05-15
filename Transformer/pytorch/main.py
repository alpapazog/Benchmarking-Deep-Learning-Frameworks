import torch
import time
from datetime import datetime
import os

# Positional Encoding (same as C++)
def positional_encoding(seq_len, embed_dim, device):
    pe = torch.zeros(seq_len, embed_dim, device=device)
    position = torch.arange(0, seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(1)  # [seq_len, 1, embed_dim]

# TransformerEncoder and TransformerClassifier (same as C++)
class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
        self.feedforward1 = torch.nn.Linear(embed_dim, ff_hidden_dim)
        self.feedforward2 = torch.nn.Linear(ff_hidden_dim, embed_dim)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feedforward2(torch.relu(self.feedforward1(x)))
        x = self.norm2(x + ff_output)
        return x

class TransformerClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_heads, ff_hidden_dim, num_classes, device):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, ff_hidden_dim)
        self.classifier = torch.nn.Linear(embed_dim, num_classes)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.device = device

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        pe = positional_encoding(self.seq_len, self.embed_dim, self.device)
        x = x + pe
        x = self.encoder(x)
        x = x.mean(0)  # [batch_size, embed_dim]
        logits = self.classifier(x)
        return logits

# Utility for log file
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M')}.txt"

# Setup
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset (same as C++)
print("Loading dataset...")
data = torch.load("./data/dataset.pt", map_location=device)
inputs = data['inputs'].to(device)
labels = data['labels'].to(device)

# Config
vocab_size = 100684
seq_len = inputs.size(1)
embed_dim = 512
num_heads = 8
ff_hidden_dim = 2048
num_classes = 2
batch_size = 64
num_epochs = 5
num_samples = inputs.size(0)
num_batches = (num_samples + batch_size - 1) // batch_size

# Build model
model = TransformerClassifier(vocab_size, embed_dim, seq_len, num_heads, ff_hidden_dim, num_classes, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
log_filename = generate_log_filename("train_log")
log_file = open(log_filename, 'w')

training_start = time.perf_counter()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.perf_counter()
    model.train()
    total_loss = 0.0

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_inputs = inputs[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        optimizer.zero_grad()
        logits = model(batch_inputs)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch + 1) % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch + 1}/{num_batches}], Loss: {loss.item():.4f}")

    epoch_end = time.perf_counter()
    print(f"[Epoch {epoch}] Average Loss: {total_loss / num_batches:.4f}, duration: {epoch_end - epoch_start:.2f} seconds")
    log_file.write(f"Epoch {epoch}: {epoch_end - epoch_start:.2f} sec\n")

training_end = time.perf_counter()
print(f"Total training time: {training_end - training_start:.2f} seconds")
log_file.write(f"Total training time: {training_end - training_start:.2f} seconds\n")
log_file.write(f"Average epoch duration: {(training_end - training_start) / num_epochs:.2f} seconds\n")
log_file.close()

torch.save(model.state_dict(), "transformer_imdb_preloaded.pth")
print("Training complete. Model saved to transformer_imdb_preloaded.pth")
