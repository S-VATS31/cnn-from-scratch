from setup_env import device

import math

import torch

def train_cnn(model, train_loader, criterion, optimizer):
    """Train looping for CNN.

    Args:
        model (nn.Module): Model to be trained.
        train_loader (DataLoader): Loader to be iterated through during training.
        criterion (nn.CrossEntropyLoss): Loss function to be minimized.
        optimizer (optim.Adam): Optimization algorithm to minimize loss.

    Returns:
        float: Calculated by dividing total loss by number of samples.
        float: Calculated by dividing total correct predictions by number of samples.
    """
    model.train() # Turn off dropout
    
    # Initialization
    train_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        # Ensure on targets, labels on device
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss and backward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate total loss
        train_loss += loss.item() * images.size(0)

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += images.size(0)

    # Calculate average loss and accuracy
    avg_loss = train_loss / total_samples
    accuracy = total_correct / total_samples
    perplexity = math.exp(avg_loss)
    return avg_loss, accuracy, perplexity
