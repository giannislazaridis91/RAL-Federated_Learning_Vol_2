import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cpu")
LEARNING_RATE = 0.0001

class Network(nn.Module):
    
    def __init__(self, state_length=30, action_length=3, learning_rate=LEARNING_RATE):
        
        # The super call delegates the function call to the parent class,
        # which is nn.Module in your case.
        # This is needed to initialize the nn.Module properly.
        # Instead of super(Network, self).__init__(),
        # super().__init__() can also be used.
        super(Network, self).__init__()
        
        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        # Applies a linear transformation to the incoming data: y=x*A^T + b
        # Parameters:
        # in_features (int):      The size of each input sample.
        # out_features (int):     The size of each output sample.
        # bias (bool):            If set to False, the layer will not learn an additive bias.
        #                         Default: True
        #
        # Example:
        #    >>> m = nn.Linear(20, 30)
        #    >>> input = torch.randn(128, 20)
        #    >>> output = m(input)
        #    >>> print(output.size())
        #    torch.Size([128, 30])
        self.fc1 = nn.Linear(state_length, 10)
        self.fc2 = nn.Linear(10 + action_length, 5)
        self.predictions = nn.Linear(5, 1)

        # Adam is an optimization algorithm that can be used instead of
        # the classical stochastic gradient descent procedure
        # to update network weights iterative based in training data.
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, x_classifier, x_datapoint):
        x = F.sigmoid(self.fc1(x_classifier))
        x = torch.cat((x, x_datapoint), 1)
        x = F.sigmoid(self.fc2(x))
        predictions = self.predictions(x)
        return predictions