from torchvision.datasets import MNIST

# Force re-processing from existing raw .gz files
MNIST(root="data", train=True, download=True)
MNIST(root="data", train=False, download=True)
