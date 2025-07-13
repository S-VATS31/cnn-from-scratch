from setup_env import device, dtype, logger

import torch

def test_cnn(model, test_loader, criterion):
    """
    Testing loop for CNN.

    Args:
        model (ConvolutionalNeuralNetwork): Model to be tested.
        test_loader (DataLoader): Loader to be iterated through during testing.

    Returns:
        accuracy (float): Calculated by dividing total correct predictions by number of samples.
    """
    model.eval() # Turn off training

    # Initialization
    test_loss = 0.0
    total_correct = 0
    total_samples = 0
    # Turn off gradient computation
    with torch.no_grad():
        for images, labels in test_loader:
            # Ensure images, labels on same device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute test loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
        
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    # Calculate accuracy
    avg_loss = test_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy