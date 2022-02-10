from typing import List, Tuple, Union

import torch
from torch import Size, Tensor, nn


class FCEmbedding(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 2, num_hiddens: int = 20):
        """Fully-connected multi-layer neural network to be used as embedding network.

        Args:
            input_dim: Dimensionality of input that will be passed to the embedding net.
            num_layers: Number of layers of the embedding network.
            num_hiddens: Number of hidden units in each layer of the embedding network.
        """
        super().__init__()
        layers = [nn.Linear(input_dim, num_hiddens), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hiddens, num_hiddens))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.

        Args:
            x: Input tensor (batch_size, num_features)

        Returns:
            Network output (batch_size, num_features).
        """
        x = self.net(x)
        return x
