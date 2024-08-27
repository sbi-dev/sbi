# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from torch import Tensor, nn


class FCEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 20,
        num_layers: int = 2,
        num_hiddens: int = 50,
    ):
        """Fully-connected multi-layer neural network to be used as embedding network.

        Args:
            input_dim: Dimensionality of input that will be passed to the embedding net.
            output_dim: Dimensionality of the output.
            num_layers: Number of layers of the embedding network. (Minimum of 2).
            num_hiddens: Number of hidden units in each layer of the embedding network.
        """
        super().__init__()
        layers = [nn.Linear(input_dim, num_hiddens), nn.ReLU()]
        # first and last layer is defined by the input and output dimension.
        # therefor the "number of hidden layeres" is num_layers-2
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(num_hiddens, num_hiddens))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_hiddens, output_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.

        Args:
            x: Input tensor (batch_size, num_features)

        Returns:
            Network output (batch_size, num_features).
        """
        return self.net(x)
