import torch
import time
from datetime import datetime
import os

# Positional Encoding
def positional_encoding(seq_len, embed_dim, device):
    pe = torch.zeros(seq_len, embed_dim, device=device)
    position = torch.arange(0, seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(1)  # [seq_len, 1, embed_dim]

# Transformer Classes (same as training)
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
        x = self.embedding(input_ids)
        x = x.transpose(0, 1)
        pe = positional_encoding(self.seq_len, self.embed_dim, self.device)
        x = x + pe
        x = self.encoder(x)
        x = x.mean(0)
        logits = self.classifier(x)
        return logits

# Logging utility
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M')}.txt"

# Setup
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
print("Loading dataset...")
data = torch.load("./data/dataset.pt", map_location=device)
inputs = data['inputs'].to(device)

# Config (match training)
vocab_size = 100684
seq_len = inputs.size(1)
embed_dim = 512
num_heads = 8
ff_hidden_dim = 2048
num_classes = 2
batch_size = 64
num_samples = inputs.size(0)
num_batches = (num_samples + batch_size - 1) // batch_size

# Load model
model = TransformerClassifier(vocab_size, embed_dim, seq_len, num_heads, ff_hidden_dim, num_classes, device).to(device)
model.load_state_dict(torch.load("transformer_imdb_preloaded.pth", map_location=device))
model.eval()

# Inference benchmarking
torch.cuda.synchronize() if device.type == 'cuda' else None
start_time = time.perf_counter()

with torch.no_grad():
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_inputs = inputs[start_idx:end_idx]
        logits = model(batch_inputs)

torch.cuda.synchronize() if device.type == 'cuda' else None
end_time = time.perf_counter()

# Metrics
total_inference_time = end_time - start_time
throughput = num_samples / total_inference_time
avg_batch_time = (total_inference_time / num_batches) * 1000  # ms/batch
model_size_mb = os.path.getsize("transformer_imdb_preloaded.pth") / (1024 * 1024)
param_count = sum(p.numel() for p in model.parameters())

print(f"Total Inference Time: {total_inference_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} samples/sec")
print(f"Average Inference Time per Batch: {avg_batch_time:.2f} ms/batch")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Parameter count: {param_count}")

if device.type == 'cuda':
    print(f"Peak GPU memory (MB): {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f}")

# Optional: log to file
log_filename = generate_log_filename("inference_log")
with open(log_filename, 'w') as f:
    f.write(f"Total Inference Time: {total_inference_time:.4f} seconds\n")
    f.write(f"Throughput: {throughput:.2f} samples/sec\n")
    f.write(f"Average Inference Time per Batch: {avg_batch_time:.2f} ms/batch\n")
    f.write(f"Model size: {model_size_mb:.2f} MB\n")
    f.write(f"Parameter count: {param_count}\n")
    if device.type == 'cuda':
        f.write(f"Peak GPU memory (MB): {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f}\n")

print(f"Inference log saved to {log_filename}")
