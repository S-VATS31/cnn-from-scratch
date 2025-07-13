from setup_env import device, dtype

import torch
import torch.nn as nn

class BatchNorm2D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        eps: float = 1e-6, 
        momentum: float = 0.1
    ):
        """Custom BatchNorm2D with minimal PyTorch interference.

        Args:
            in_channels (int): Number of channels in the input tensor, x.
            eps (float): Small constant to avoid division by 0.
            momentum (float): Determines how fast running stats update.
                Larger momentum means stats are updated quicker and focus more on recent stats.
                Lower momentum means stats are updated slower and focus more around past stats. 
        """
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.momentum = momentum

        # Define learnable parameters - gamma and beta
        self.gamma = nn.Parameter(torch.ones(in_channels)) # Scaling factor
        self.beta = nn.Parameter(torch.zeros(in_channels)) # Shifting factor

        # Running stats
        self.register_buffer("running_mean", torch.zeros(in_channels))
        self.register_buffer("running_variance", torch.ones(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the BatchNorm2D layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height_in, width_in]

        Returns:
            torch.Tensor: Normalized tensor, previous shape applies here.
        """
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            # Compute mean and variance over batch_size, height_in, width_in
            if self.training:
                mean = x.mean(dim=[0, 2, 3])
                variance = x.var(dim=[0, 2, 3], unbiased=False)

                # Update running stats
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * variance
            else:
                mean = self.running_mean
                variance = self.running_variance

            # Apply normalization to channels dimension
            x = (x - mean[None, :, None, None]) / (torch.sqrt(variance[None, :, None, None]) + self.eps)

            # Apply scaling and shifting to channels dimension
            x = self.gamma[None, :, None, None] * x + self.beta[None, :, None, None]

            return x
        