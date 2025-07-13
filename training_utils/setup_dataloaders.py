from setup_env import device, dtype, logger

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from custom_layers.cnn import ConvolutionalNeuralNetwork

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root='./mnist', 
    train=True, 
    download=True, 
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
)

test_dataset = datasets.MNIST(
    root='./mnist', 
    train=False, 
    download=True,
     transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model, loss function, optimizer
cnn = ConvolutionalNeuralNetwork().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
