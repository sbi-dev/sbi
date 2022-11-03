# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import List, Tuple, Union

import torch
from torch import Tensor, nn


class FCEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 20,
        num_layers: int = 2,
        num_hiddens: int = 40,
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


def calculate_filter_output_size(input_size, padding, dilation, kernel, stride) -> int:
    """Returns output size of a filter given filter arguments.

    Uses formulas from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.
    """

    return int(
        (int(input_size) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1)
        / int(stride)
        + 1
    )


def get_new_cnn_output_size(
    input_shape: Tuple,
    conv_layer: Union[nn.Conv1d, nn.Conv2d],
    pool: Union[nn.MaxPool1d, nn.MaxPool2d],
) -> Union[Tuple[int], Tuple[int, int]]:
    """Returns new output size after applying a given convolution and pooling.

    Args:
        input_shape: tup.
        conv_layer: applied convolutional layers
        pool: applied pooling layer

    Returns:
        new output dimension of the cnn layer.

    """
    assert isinstance(input_shape, Tuple), "input shape must be Tuple."
    assert 0 < len(input_shape) < 3, "input shape must be 1 or 2d."
    assert isinstance(conv_layer.padding, Tuple), "conv layer attributes must be Tuple."
    assert isinstance(pool.padding, int), "pool layer attributes must be integers."

    out_after_conv = [
        calculate_filter_output_size(
            input_shape[i],
            conv_layer.padding[i],
            conv_layer.dilation[i],
            conv_layer.kernel_size[i],
            conv_layer.stride[i],
        )
        for i in range(len(input_shape))
    ]
    out_after_pool = [
        calculate_filter_output_size(
            out_after_conv[i],
            pool.padding,
            pool.dilation,
            pool.kernel_size,
            pool.stride,
        )
        for i in range(len(input_shape))
    ]
    return tuple(out_after_pool)


class CNNEmbedding(nn.Module):
    def __init__(
        self,
        input_shape: Tuple,
        in_channels: int = 1,
        out_channels_per_layer: List = [6, 12],
        num_conv_layers: int = 2,
        num_linear_layers: int = 2,
        num_linear_units: int = 50,
        output_dim: int = 20,
        kernel_size: int = 5,
        pool_kernel_size: int = 2,
    ):
        """Convolutional embedding network.
        First two layers are convolutional, followed by fully connected layers.

        Automatically infers whether to apply 1D or 2D convolution depending on
        input_shape.
        Allows usage of multiple (color) channels by passing in_channels > 1.

        Args:
            input_shape: Dimensionality of input, e.g., (28,) for 1D, (28, 28) for 2D.
            in_channels: Number of image channels, default 1.
            out_channels_per_layer: Number of out convolutional out_channels for each
                layer. Must match the number of layers passed below.
            num_cnn_layers: Number of convolutional layers.
            num_linear_layers: Number fully connected layer.
            num_linear_units: Number of hidden units in fully-connected layers.
            output_dim: Number of output units of the final layer.
            kernel_size: Kernel size for both convolutional layers.
            pool_size: pool size for MaxPool1d operation after the convolutional
                layers.
        """
        super(CNNEmbedding, self).__init__()

        assert isinstance(
            input_shape, Tuple
        ), "input_shape must be a Tuple of size 1 or 2, e.g., (width, [height])."
        assert (
            0 < len(input_shape) < 3
        ), """input_shape must be a Tuple of size 1 or 2, e.g.,
            (width, [height]). Number of input channels are passed separately"""

        use_2d_cnn = len(input_shape) == 2
        conv_module = nn.Conv2d if use_2d_cnn else nn.Conv1d
        pool_module = nn.MaxPool2d if use_2d_cnn else nn.MaxPool1d

        assert (
            len(out_channels_per_layer) == num_conv_layers
        ), "out_channels needs as many entries as num_cnn_layers."

        # define input shape with channel
        self.input_shape = (in_channels, *input_shape)

        # Construct CNN feature extractor.
        cnn_layers = []
        cnn_output_size = input_shape
        stride = 1
        padding = 1
        for ii in range(num_conv_layers):
            # Defining another 2D convolution layer
            conv_layer = conv_module(
                in_channels=in_channels if ii == 0 else out_channels_per_layer[ii - 1],
                out_channels=out_channels_per_layer[ii],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            pool = pool_module(kernel_size=pool_kernel_size)
            cnn_layers += [conv_layer, nn.ReLU(inplace=True), pool]
            # Calculate change of output size of each CNN layer
            cnn_output_size = get_new_cnn_output_size(cnn_output_size, conv_layer, pool)

        self.cnn_subnet = nn.Sequential(*cnn_layers)

        # Construct linear post processing net.
        self.linear_subnet = FCEmbedding(
            input_dim=out_channels_per_layer[-1]
            * torch.prod(torch.tensor(cnn_output_size)),
            output_dim=output_dim,
            num_layers=num_linear_layers,
            num_hiddens=num_linear_units,
        )

    # Defining the forward pass
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        # reshape to account for single channel data.
        x = self.cnn_subnet(x.view(batch_size, *self.input_shape))
        # flatten for linear layers.
        x = x.view(batch_size, -1)
        x = self.linear_subnet(x)
        return x


class PermutationInvariantEmbedding(nn.Module):
    """Permutation invariant embedding network.

    Takes as input a tensor with (batch, permutation_dim, input_dim)
    and outputs (batch, output_dim).

    References:
    Chan et al. (2018): "A likelihood-free inference framework for population genetic
    data using exchangeable neural networks"
    Radev et al. (2020): "BayesFlow: Learning complex stochastic models with invertible
    neural networks"
    """

    def __init__(
        self,
        trial_net: nn.Module,
        trial_net_output_dim: int,
        combining_operation: str = "mean",
        num_layers: int = 2,
        num_hiddens: int = 40,
        output_dim: int = 20,
    ):
        """Permutation invariant multi-layer NN.

        The trial_net is applied to each "trial" of the input
        and is combined by the combining_operation (mean or sum) to construct a
        permutation invariant embedding across iid trials.
        This embedding is embedded again using an additional fully connected net.

        Args:
            trial_net: Network to process one trial. The combining_operation is
                applied to its output. Takes as input (batch, input_dim), where
                input_dim is the dimensionality of a single trial. Produces output
                (batch, latent_dim).
                Remark: This network should be large enough as it acts on all (iid)
                inputs seperatley and needs enough capacity to process the information
                of all inputs.
            trial_net_output_dim: Dimensionality of the output of the trial_net / input
                to the fully connected layers.
            combining_operation: How to combine the permutational dimensions, one of
                'mean' or 'sum'.
            num_layers: Number of fully connected layer, minimum of 2.
            num_hiddens: Number of hidden dimensions in fully-connected layers.
            output_dim: Dimensionality of the output.
        """
        super().__init__()
        self.trial_net = trial_net
        self.combining_operation = combining_operation

        if combining_operation not in ["sum", "mean"]:
            raise ValueError("combining_operation must be in ['sum', 'mean'].")

        # construct fully connected layers
        self.fc_subnet = FCEmbedding(
            input_dim=trial_net_output_dim,
            output_dim=output_dim,
            num_layers=num_layers,
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

        iid_embeddings = self.trial_net(x.view(batch * permutation_dim, -1)).view(
            batch, permutation_dim, -1
        )

        if self.combining_operation == "mean":
            e = iid_embeddings.mean(dim=1)
        elif self.combining_operation == "sum":
            e = iid_embeddings.sum(dim=1)
        else:
            raise ValueError("combining_operation must be in ['sum', 'mean'].")

        embedding = self.fc_subnet(e)

        return embedding
