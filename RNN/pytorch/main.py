import torch
import torch.nn as nn
import time
from datetime import datetime

# SentimentLSTM matching your saved model
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

# Generate timestamped log filename
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M')}.txt"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Config (must match training)
vocab_size = 100684
embed_dim = 100
hidden_dim = 128
output_dim = 2
batch_size = 64

# Load model and state dict
model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load('sentiment_lstm_preloaded.pth', map_location=device))
model.eval()

# Load test dataset (same format as training dataset)
print("Loading test dataset...")
data = torch.load('./data/dataset_test.pt', map_location=device)
inputs = data['inputs'].to(device)
labels = data['labels'].to(device)

# Inference loop with manual batching
num_samples = inputs.size(0)
num_batches = (num_samples + batch_size - 1) // batch_size
total_correct = 0

with torch.no_grad():
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for batch_idx in range(num_batches):
        batch_inputs = inputs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_labels = labels[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        outputs = model(batch_inputs)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == batch_labels).sum().item()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()

# Accuracy and timing
accuracy = 100.0 * total_correct / num_samples
print(f'Accuracy: {accuracy:.2f}%')
print(f'Inference time: {end_time - start_time:.4f} seconds')

# Log results
log_filename = generate_log_filename('test_log')
with open(log_filename, 'w') as log_file:
    log_file.write(f'Accuracy: {accuracy:.2f}%\n')
    log_file.write(f'Total Inference Time: {end_time - start_time:.4f} seconds\n')

print(f'Test log saved to {log_filename}')
