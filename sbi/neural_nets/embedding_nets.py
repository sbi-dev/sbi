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


class CNNEmbedding(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_fc: int = 2, num_hiddens: int = 120
    ):
        """multi-layer (C)NN
            first 2 layers are convolutional, followed by fully connected layers.
            Performing 1d convolution and max pooling with preset configs.
        Args:
            input_dim: Dimensionality of input.
            num_conv: Number of con layers.
            num_fc: Number fully connected layer, min of 2
            num_hiddens: number of hidden dims in fc layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hiddens = num_hiddens

        # configs for conv layers
        out_channels_cnn_1 = 10
        out_channels_cnn_2 = 16
        kernel_size = 5
        pool_size = 4

        # construct conv-pool subnet
        pool = nn.MaxPool1d(pool_size)
        conv_layers = [
            nn.Conv1d(1, out_channels_cnn_1, kernel_size, padding="same"),
            nn.ReLU(),
            pool,
        ]
        conv_layers.append(
            nn.Conv1d(
                out_channels_cnn_1, out_channels_cnn_2, kernel_size, padding="same"
            )
        )
        conv_layers.append(nn.ReLU())
        conv_layers.append(pool)
        self.conv_subnet = nn.Sequential(*conv_layers)

        # construct fully connected layers
        fc_layers = [
            nn.Linear(
                out_channels_cnn_2 * (int(input_dim / out_channels_cnn_2)), num_hiddens
            ),
            nn.ReLU(),
        ]
        for _ in range(num_fc - 2):
            fc_layers.append(nn.Linear(num_hiddens, num_hiddens))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(num_hiddens, output_dim))
        fc_layers.append(nn.ReLU())

        self.fc_subnet = nn.Sequential(*fc_layers)

    def forward(self, x):
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Network output (batch_size, output_dim).
        """
        x = self.conv_subnet(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        embedding = self.fc_subnet(x)

        return embedding


class FCEmbedding_permutation_inv(nn.Module):
    """Permutation invariant embedding network.
    takes as input a tensor with (batch, permutation_dim, input_dim)
    and outputs (batch,output_dim).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc: int = 2,
        first_layer_dim: int = 500,
        num_hiddens: int = 120,
    ):
        """permutation invariant multi-layer NN
            fully connected layers takes (batch,permutation_dim, input_dim) as input.
            takes the mean over permutation_dim.
        Args:
            input_dim: Dimensionality of input.
            num_fc: Number fully connected layer, min of 2
            first_layer_dim: dimension of the first layer, the one which is taken the mean over.
            num_hiddens: number of hidden dims in fc
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.first_layer_dim = first_layer_dim
        self.num_hiddens = num_hiddens

        # layer on which the mean is taken from
        self.first_layer = nn.Sequential(
            *[nn.Linear(input_dim, first_layer_dim), nn.ReLU()]
        )

        # construct fully connected layers
        fc_layers = [nn.Linear(first_layer_dim, num_hiddens), nn.ReLU()]
        for _ in range(num_fc - 2):
            fc_layers.append(nn.Linear(num_hiddens, num_hiddens))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(num_hiddens, output_dim))
        fc_layers.append(nn.ReLU())

        self.fc_subnet = nn.Sequential(*fc_layers)

    def forward(self, x):
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, permutation_dim, input_dim)
        Returns:
            Network output (batch_size, output_dim).
        """
        _, permutation_dim, _ = x.shape

        e = self.first_layer(x[:, 0])
        for i, item in enumerate(torch.transpose(x[:, 1:], 0, 1)):
            e = e + self.first_layer(item)

        e = e / permutation_dim

        embedding = self.fc_subnet(e)

        return embedding
