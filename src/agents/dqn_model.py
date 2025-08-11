import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model architecture.

    This model uses a convolutional neural network to process game screen
    frames and a fully connected layer to output Q-values for each action.
    """
    def __init__(self, input_shape, num_actions):
        """
        Initializes the DQN model.

        Args:
            input_shape (tuple): Shape of the input state (e.g., (4, 84, 84) for stacked frames).
                                 input_shape[0] is the number of channels (stacked frames).
            num_actions (int): Number of possible actions the agent can take.
        """
        super(DQN, self).__init__()
        # Convolutional layers for feature extraction from game frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # Input: (N, C, H, W) -> Output: (N, 32, H', W')
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Output: (N, 64, H'', W'')
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # Output: (N, 64, H''', W''')
            nn.ReLU()
        )
        
        # Calculate the size of the output from the convolutional layers
        conv_out_size = self._get_conv_out(input_shape)
        
        # Fully connected layers to map convolutional features to Q-values
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), # Input: flattened conv output -> Output: 512 features
            nn.ReLU(),
            nn.Linear(512, num_actions) # Input: 512 features -> Output: Q-values for each action
        )

    def _get_conv_out(self, shape):
        """
        Calculates the size of the output from the convolutional layers.

        Args:
            shape (tuple): The input shape to the convolutional layers.

        Returns:
            int: The flattened size of the convolutional output.
        """
        # Create a dummy tensor to pass through the conv layers to determine output size
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor representing the game state.

        Returns:
            torch.Tensor: The output tensor containing Q-values for each action.
        """
        # Normalize pixel values to be between 0 and 1
        x = x.float() / 255.0
        # Pass through convolutional layers and flatten the output
        conv_out = self.conv(x).view(x.size()[0], -1)
        # Pass through fully connected layers to get Q-values
        return self.fc(conv_out)