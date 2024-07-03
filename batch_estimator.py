import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DEFINING THE TARGET DQN AND THE COPY NETWORK.

The batch_estimator.py file defines a neural network model called Estimator,
which is designed to work in reinforcement learning settings,
particularly in Deep Q-Learning (DQN).

The model comprises three linear layers with intermediate sigmoid activations
and is capable of combining input states with actions to produce predictions.

The model can be configured as a target network,
where its parameters are frozen to prevent updates during training.

This architecture is typically used to estimate Q-values in reinforcement learning,
where the input state and action are combined to predict the expected reward.
The flexibility to freeze the model's parameters makes it suitable
for use as a stable target network in DQN algorithms.

SUMMARY:

1. The Estimator class defines a neural network model with three linear layers and sigmoid activations for intermediate layers.
2. The model can be configured as a target network in DQN, where its parameters are frozen.
3. It is likely used for tasks where inputs are combined with additional actions to predict some value (common in reinforcement learning scenarios).
"""

class Estimator(nn.Module): # Defines a neural network model by extending nn.Module.

    def __init__(self, classifier_state_length, is_target_dqn, bias_average):
        
        """
        The constructor for the Estimator class. It initializes the neural network layers and sets up the model.

        - classifier_state_length: Input dimension size.
        - is_target_dqn: Boolean to check if the network is a target DQN (Deep Q-Network).
        - bias_average: Value to initialize the bias of the final layer.
        """
        
        super(Estimator, self).__init__()

        # Define the layers.

        # Fully connected layer from input dimension to 10 neurons.
        # A linear layer that takes input of size classifier_state_length and maps it to a 10-dimensional space.
        self.fc1 = nn.Linear(classifier_state_length, 10)

        # Fully connected layer from 11 (10 + 1 for action input) to 5 neurons.
        # A linear layer that takes the 10-dimensional output from fc1 concatenated with an action input (of size 1) and maps it to a 5-dimensional space.
        self.fc2 = nn.Linear(10 + 1, 5)

        # Fully connected layer from 5 neurons to 1 output neuron.
        # A linear layer that maps the 5-dimensional output from fc2 to a single value (likely a prediction or Q-value).
        self.fc3 = nn.Linear(5, 1)

        # Initialize bias for the final layer.
        # The bias of the final layer (fc3) is initialized to a constant value (bias_average).
        # Sets the bias of fc3 to a constant value bias_average.
        nn.init.constant_(self.fc3.bias, bias_average)

        # If it's a target DQN, we freeze the layers' parameters to prevent training.
        # If is_target_dqn is True, it prevents the parameters of the network from being updated during training.
        # If is_target_dqn is True, the parameters of the network are frozen (i.e., they do not update during training).
        if is_target_dqn:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, classifier_input, action_input):

        """
        1. Defines the forward pass of the network.

            - classifier_input: Input to the network.
            - action_input: Additional input concatenated to the output of fc1.

        2. Applies sigmoid activation function to the outputs of fc1 and fc2.

        3. Concatenates action_input to the output of fc1 along the feature dimension.

        4. Computes the final prediction using fc3.
        """

        x = torch.sigmoid(self.fc1(classifier_input))
        x = torch.cat((x, action_input), dim=1)  # Concatenate along the feature dimension.
        x = torch.sigmoid(self.fc2(x))
        predictions = self.fc3(x)
        return predictions