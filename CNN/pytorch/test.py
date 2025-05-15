import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy
import time

# Define CNN model
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# Load model
model = Net().to(device)
model.load_state_dict(torch.load('mnist_cnn_preloaded.pth', map_location=device))
model.eval()

# Load test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inference loop with timing
correct = 0
total = 0

# Warm-up (optional, good practice for GPU)
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        _ = model(data)
        break  # Only first batch

# Start timing
if device.type == 'cuda':
    torch.cuda.synchronize()
start_time = time.perf_counter()

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

if device.type == 'cuda':
    torch.cuda.synchronize()
end_time = time.perf_counter()

# Results
accuracy = 100.0 * correct / total
print(f'Inference Accuracy: {accuracy:.2f}%')
print(f'Total Inference Time: {end_time - start_time:.4f} seconds')
print(f'Avg Inference Time per Sample: {(end_time - start_time)/total:.6f} seconds')
