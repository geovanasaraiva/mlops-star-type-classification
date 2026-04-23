import torch
import torch.nn as nn

#Implementation of a Multi-Layer Perceptron (MLP) using PyTorch.
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        dropout: float = 0.0
    ):
        super().__init__()

        #List that stores the network layers.
        layers = []
        #Previous layer size.
        prev_size = input_size

        #Building the hidden layers.
        for h in hidden_sizes:
            #Linear layer (fully connected).
            layers.append(nn.Linear(prev_size, h))
            #Nonlinear activation function.
            layers.append(nn.ReLU())

            #Dropout to reduce overfitting (optional).
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            #Update the size of the next entry.
            prev_size = h

        #Output layer.
        layers.append(nn.Linear(prev_size, output_size))

        #It packages all layers into a sequential model.
        self.model = nn.Sequential(*layers)

    #Defines the data flow in the network (forward pass).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

"""
Parameters:
- input_size: number of input features
- hidden_sizes: list with the number of neurons in each hidden layer
- output_size: number of neurons in the output layer
- dropout: dropout rate applied between layers (regularization)
"""