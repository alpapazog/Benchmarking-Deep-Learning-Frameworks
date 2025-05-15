import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from datetime import datetime

# Enable cuDNN autotuner for better performance with fixed-size inputs
torch.backends.cudnn.benchmark = True

# Generate timestamped log filename
def generate_log_filename(prefix='log'):
    now = datetime.now()
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H%M')}.txt"

# CNN model identical to your C++
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Setup device
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")
print("Device: ", device)
# Load entire dataset to GPU memory
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
all_data = torch.stack([data[0] for data in dataset]).to(device)
all_targets = torch.tensor([data[1] for data in dataset], device=device)

# Model, optimizer, loss
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Settings
batch_size = 64
num_epochs = 5
num_batches = all_data.size(0) // batch_size

# Logging
log_filename = generate_log_filename('train_log')
log_file = open(log_filename, 'w')

# Training loop
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()
    model.train()
    perm = torch.randperm(all_data.size(0), device=device)  # Shuffle indices
    for batch_idx in range(num_batches):
        idx = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        data = all_data[idx]
        target = all_targets[idx]

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item()}')

    epoch_end = time.time()
    epoch_duration = epoch_end - epoch_start
    print(f'Epoch {epoch} duration: {epoch_duration:.2f} seconds')
    log_file.write(f'Epoch {epoch}: {epoch_duration:.2f} sec\n')

total_duration = time.time() - start_time
print(f'Total training time: {total_duration:.2f} seconds')
log_file.write(f'Total training time: {total_duration:.2f} seconds\n')
log_file.write(f'Average epoch duration: {total_duration / num_epochs:.2f} seconds\n')
log_file.close()

# Save model
torch.save(model.state_dict(), 'mnist_cnn_preloaded.pth')
print('Model saved to mnist_cnn_preloaded.pth')
