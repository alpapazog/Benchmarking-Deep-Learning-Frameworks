import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime

def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M')}.txt"

# Config
N = 1519
T = 1383
n_mels = 80
target_seq_len = 320
vocab_size = 28
batch_size = 8
num_epochs = 5

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
targets = torch.frombuffer(extra_files["targets"], dtype=torch.int64).view(N, target_seq_len).clone().to(device)
input_lengths = torch.frombuffer(extra_files["input_lengths"], dtype=torch.int64).view(N).clone().to(device)
target_lengths = torch.frombuffer(extra_files["target_lengths"], dtype=torch.int64).view(N).clone().to(device)


# Define model
class AcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_mels, 128)
        self.fc2 = nn.Linear(128, vocab_size + 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AcousticModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
ctc_loss = nn.CTCLoss(zero_infinity=True)

log_filename = generate_log_filename("train_log")
log_file = open(log_filename, 'w')

print("Starting training...")
start_time = time.perf_counter()

for epoch in range(num_epochs):
    epoch_start = time.perf_counter()
    model.train()
    epoch_loss = 0.0
    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)

        batch_features = features[start:end]
        batch_targets = targets[start:end]
        batch_input_lengths = input_lengths[start:end]
        batch_target_lengths = target_lengths[start:end]

        logits = model(batch_features)  # [batch, time, vocab+1]
        logits = logits.transpose(0, 1).log_softmax(2)  # [time, batch, vocab+1]

        loss = ctc_loss(logits, batch_targets, batch_input_lengths, batch_target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{num_batches}], Loss: {loss.item()}")

    epoch_end = time.perf_counter()
    print(f"Epoch {epoch + 1} average loss: {epoch_loss / num_batches:.4f}, duration: {epoch_end - epoch_start:.2f} seconds")
    log_file.write(f"Epoch {epoch + 1}: {epoch_end - epoch_start:.2f} sec\n")

total_time = time.perf_counter() - start_time
print(f"Total training time: {total_time:.2f} seconds")
log_file.write(f"Total training time: {total_time:.2f} seconds\n")
log_file.write(f"Average epoch duration: {total_time / num_epochs:.2f} seconds\n")
log_file.close()

torch.save(model.state_dict(), "ctc_model_preloaded.pth")
print("Training complete. Model saved to ctc_model_preloaded.pth")
