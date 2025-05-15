import torch
import time
from datetime import datetime

def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M')}.txt"

# Config
N = 1519
T = 1383
n_mels = 80
vocab_size = 28
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset from TorchScript extra_files
extra_files = {
    "features": "",
    "targets": "",
    "input_lengths": "",
    "target_lengths": ""
}

_ = torch.jit.load("data/librispeech_400min_dataset_model.pt", map_location=device, _extra_files=extra_files)

features = torch.frombuffer(extra_files["features"], dtype=torch.float32).view(N, T, n_mels).clone().to(device)

# Define model (must match training)
class AcousticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_mels, 128)
        self.fc2 = torch.nn.Linear(128, vocab_size + 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AcousticModel().to(device)
model.load_state_dict(torch.load("ctc_model_preloaded.pth", map_location=device))
model.eval()

# Inference only loop (like your C++)
num_batches = (N + batch_size - 1) // batch_size

if device.type == 'cuda':
    torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        batch_features = features[start:end]
        logits = model(batch_features)
        logits = logits.transpose(0, 1).log_softmax(2)
        # No loss, no accuracy — just forward pass

if device.type == 'cuda':
    torch.cuda.synchronize()
end_time = time.perf_counter()

# Logging
duration = end_time - start_time
print(f"Total Inference Time: {duration:.4f} seconds")

log_filename = generate_log_filename("ctc_inference_log")
with open(log_filename, 'w') as log_file:
    log_file.write(f"Total Inference Time: {duration:.4f} seconds\n")

print(f"Inference log saved to {log_filename}")
