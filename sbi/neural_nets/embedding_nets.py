from typing import List, Tuple, Union

import torch
from torch import Size, Tensor, nn


class FCEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 20,
        num_layers: int = 2,
        num_hiddens: int = 20,
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


class CNNEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fully_connected: int = 2,
        num_hiddens: int = 120,
        out_channels_cnn_1: int = 10,
        out_channels_cnn_2: int = 16,
        kernel_size: int = 5,
        pool_size=4,
    ):
        """Multi-layer (C)NN
            First two layers are convolutional, followed by fully connected layers.
            Performing 1d convolution and max pooling with preset configs.

        Args:
            input_dim: Dimensionality of input.
            num_conv: Number of convolutional layers.
            num_fully_connected: Number fully connected layer, minimum of 2.
            num_hiddens: Number of hidden dimensions in fully-connected layers.
            out_channels_cnn_1: Number of oputput channels for the first convolutional
                layer.
            out_channels_cnn_2: Number of oputput channels for the second
                convolutional layer.
            kernel_size: Kernel size for both convolutional layers.
            pool_size: pool size for MaxPool1d operation after the convolutional
                layers.

            Remark: The implementation of the convolutional layers was not tested
            rigourously. While it works for the default configuration parameters it
            might cause shape conflicts fot badly chosen parameters.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hiddens = num_hiddens

        # construct convolutional-pooling subnet
        pool = nn.MaxPool1d(pool_size)
        conv_layers = [
            nn.Conv1d(1, out_channels_cnn_1, kernel_size, padding="same"),
            nn.ReLU(),
            pool,
            nn.Conv1d(
                out_channels_cnn_1, out_channels_cnn_2, kernel_size, padding="same"
            ),
            nn.ReLU(),
            pool,
        ]
        self.conv_subnet = nn.Sequential(*conv_layers)

        # construct fully connected layers
        input_dim_fc = out_channels_cnn_2 * (int(input_dim / out_channels_cnn_2))

        self.fc_subnet = FCEmbedding(
            input_dim=input_dim_fc,
            output_dim=output_dim,
            num_layers=num_fully_connected,
            num_hiddens=num_hiddens,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Network output (batch_size, output_dim).
        """
        x = self.conv_subnet(x.unsqueeze(1))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        embedding = self.fc_subnet(x)

        return embedding


class PermutationInvariantEmbedding(nn.Module):
    """Permutation invariant embedding network.

    Takes as input a tensor with (batch, permutation_dim, input_dim)
    and outputs (batch, output_dim).
    """

    def __init__(
        self,
        single_trial_net: nn.Module,
        input_dim: int,
        output_dim: int,
        num_fully_connected: int = 2,
        num_hiddens: int = 20,
        combining_operation: str = "mean",
    ):
        """Permutation invariant multi-layer NN.
            The single_trial_net is applied to each "trial" of the input
            and is combined by the combining_operation (mean or sum).

        Args:
            single_trial_net: Network to process one trial, the combining_operation is
                applied to its output. Taskes as input (batch, input_dim).
                Remark: This network should be large enough as it acts on all (iid)
                inputs seperatley and needs enough capacity to process the information
                of all inputs.
            input_dim: Dimensionality of input to the fully connected layers
                (output_dimension of single_trial_net).
            output_dim: Dimensionality of the output.
            num_fully_connected: Number of fully connected layer, minimum of 2.
            num_hiddens: Number of hidden dimensions in fully-connected layers.
            combining_operation: How to combine the permutational dimensions, one of
                'mean' or 'sum'.
        """
        super().__init__()
        self.single_trial_net = single_trial_net
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hiddens = num_hiddens
        self.combining_operation = combining_operation

        if combining_operation not in ["sum", "mean"]:
            raise ValueError("combining_operation must be in ['sum', 'mean'].")

        # construct fully connected layers
        self.fc_subnet = FCEmbedding(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_fully_connected,
            num_hiddens=num_hiddens,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, permutation_dim, input_dim)
        Returns:
            Network output (batch_size, output_dim).
        """
        batch, permutation_dim, _ = x.shape

        iid_embeddings = self.single_trial_net(
            x.view(batch * permutation_dim, -1)
        ).view(batch, permutation_dim, -1)

        if self.combining_operation == "mean":
            e = iid_embeddings.mean(dim=1)
        elif self.combining_operation == "sum":
            e = iid_embeddings.sum(dim=1)
        else:
            raise ValueError("combining_operation must be in ['sum', 'mean'].")

        embedding = self.fc_subnet(e)

        return embedding
