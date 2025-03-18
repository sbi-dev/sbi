from typing import Callable

import lru
import torch
from torch import Tensor, nn


class LRUEmbedding(nn.Module):
    """Embedding network backed by a Linear Recurrent Unit (LRU).

    As suggested in https://github.com/forgi86/sysid-pytorch-lru, we use GLU
    activation function after each layer.

    See also:
        https://arxiv.org/pdf/2303.06349
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 10,
        hidden_dim: int = 20,
        num_layers: int = 2,
        aggregation_fn: Callable[[Tensor], Tensor] = torch.mean,
    ):
        """Fully-connected multi-layer neural network to be used as embedding network.

        Args:
            input_dim: Dimensionality of input that will be passed to the embedding net.
            output_dim: Dimensionality of the output.
            hidden_dim: Number of hidden units in each layer of the embedding network.
            num_layers: Number of layers of the embedding network. (Minimum of 2).
        """
        super().__init__()

        assert aggregation_fn in (torch.mean, torch.sum), (
            "Only torch.mean and torch.sum are supported as aggregation_fn"
        )
        self.aggregation_fn = aggregation_fn

        # The first layer is defined by the observations' input dimension.
        layers = [
            lru.linear.LRU(
                in_features=input_dim,
                out_features=hidden_dim,
                state_features=hidden_dim,
            ),
            nn.GLU(),
        ]

        # The number of hidden layers is num_layers - 2.
        for _ in range(num_layers - 2):
            layers.append(
                lru.linear.LRU(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    state_features=hidden_dim,
                )
            )
            layers.append(nn.GLU())

        # The layer is defined by the observations' output dimension.
        layers.append(
            lru.linear.LRU(
                in_features=hidden_dim,
                out_features=output_dim,
                state_features=hidden_dim,
            )
        )
        layers.append(nn.GLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Embed a batch of 2-dim observations, e.g. a multi-dimensional
        time series.

        Args:
            x: Input tensor (batch_size, len_sequence, num_features)

        Returns:
            Network output (batch_size, output_dim).
        """
        x_scan = self.net(x)  # (batch_size, len_sequence, output_dim)

        # Aggregate the features of the time dimension.
        x_embed = self.aggregation_fn(x_scan, dim=1)

        return x_embed
