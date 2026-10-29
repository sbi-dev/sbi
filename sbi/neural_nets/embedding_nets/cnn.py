# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from math import prod
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from sbi.neural_nets.embedding_nets.fully_connected import FCEmbedding


def _validate_cnn_input_shape(x: Tensor, input_shape: Tuple) -> None:
    """Validate a flat or channel-first CNN input shape.

    Args:
        x: Input tensor with a batch dimension.
        input_shape: Expected shape after the batch dimension.

    Raises:
        ValueError: If the trailing input shape is not supported.
    """
    trailing_shape = tuple(x.shape[1:])
    valid_shapes = (input_shape, (prod(input_shape),))
    if input_shape[0] == 1:
        valid_shapes += (input_shape[1:],)
    if trailing_shape not in valid_shapes:
        raise ValueError(
            "Expected flat or channel-first input with trailing shape "
            f"in {valid_shapes}, but got {trailing_shape}."
        )


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
    return tuple(out_after_pool)  # pyright: ignore[reportReturnType]


class CNNEmbedding(nn.Module):
    """Convolutional embedding network with channel-first 1D or 2D inputs."""

    def __init__(
        self,
        input_shape: Tuple,
        in_channels: int = 1,
        out_channels_per_layer: Optional[List] = None,
        num_conv_layers: int = 2,
        num_linear_layers: int = 2,
        num_linear_units: int = 50,
        output_dim: int = 20,
        kernel_size: int = 5,
        pool_kernel_size: int = 2,
    ):
        """Convolutional embedding network.

        Applies `num_conv_layers` blocks of convolution, ReLU and max pooling,
        followed by fully connected layers. A 1D `input_shape` uses `Conv1d` and
        `MaxPool1d`, a 2D `input_shape` uses `Conv2d` and `MaxPool2d`. Each
        convolution (stride 1, padding 1) turns a spatial size `n` into
        `n + 3 - kernel_size`, and each pooling step then divides it by
        `pool_kernel_size`, rounding down.

        Args:
            input_shape: Spatial input shape without batch or channel dimensions, e.g.,
                (28,) for 1D or (28, 28) for 2D. Inputs may be flat or use
                channel-first layout.
            in_channels: Number of input channels, e.g., 3 for RGB images.
            out_channels_per_layer: Number of output channels of each convolutional
                layer, one entry per layer. Defaults to 6 channels in the first
                layer, doubling in each further layer, e.g., `[6, 12]` for two
                layers.
            num_conv_layers: Number of convolutional layers.
            num_linear_layers: Number of fully connected layers.
            num_linear_units: Number of hidden units in the fully connected layers.
            output_dim: Number of output units of the final layer.
            kernel_size: Kernel size of every convolutional layer.
            pool_kernel_size: Kernel size of the max pooling after every
                convolutional layer.
        """
        super(CNNEmbedding, self).__init__()

        if not isinstance(input_shape, tuple):
            raise TypeError(
                "input_shape must be a tuple, e.g., (28,) or (28, 28), but got "
                f"{type(input_shape).__name__}."
            )
        if not 0 < len(input_shape) < 3:
            raise ValueError(
                "input_shape must have length 1 or 2, e.g., (28,) or (28, 28). "
                "Pass the number of channels as in_channels."
            )

        use_2d_cnn = len(input_shape) == 2
        conv_module = nn.Conv2d if use_2d_cnn else nn.Conv1d
        pool_module = nn.MaxPool2d if use_2d_cnn else nn.MaxPool1d

        if num_conv_layers < 1:
            raise ValueError(
                f"num_conv_layers must be at least 1, but got {num_conv_layers}."
            )
        if out_channels_per_layer is None:
            out_channels_per_layer = [6 * 2**ii for ii in range(num_conv_layers)]
        if len(out_channels_per_layer) != num_conv_layers:
            raise ValueError(
                f"out_channels_per_layer has {len(out_channels_per_layer)} entries, "
                f"but num_conv_layers is {num_conv_layers}. Pass one entry per "
                "convolutional layer."
            )

        # define input shape with channel
        self.input_shape = (in_channels, *input_shape)

        # Construct CNN feature extractor.
        cnn_layers = []
        cnn_output_size = input_shape
        stride = 1
        padding = 1

        for ii in range(num_conv_layers):
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

            if not all(size > 0 for size in cnn_output_size):
                fewer_layers_hint = (
                    f", or set num_conv_layers to at most {ii}" if ii > 0 else ""
                )
                raise ValueError(
                    f"CNN output size {cnn_output_size} is not positive at layer "
                    f"{ii + 1}. Use a smaller kernel_size or pool_kernel_size"
                    f"{fewer_layers_hint}."
                )

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
        """Embed a batch of inputs.

        Args:
            x: Inputs of shape `(batch_dim, in_channels, *input_shape)`. A flat
                `(batch_dim, in_channels * prod(input_shape))` is also accepted, as
                is `(batch_dim, *input_shape)` when `in_channels` is 1.

        Returns:
            Embeddings of shape `(batch_dim, output_dim)`.

        Raises:
            ValueError: If the trailing shape of `x` is none of the above.
        """
        batch_size = x.size(0)
        _validate_cnn_input_shape(x, self.input_shape)

        # reshape to account for single channel data.
        x = self.cnn_subnet(x.reshape(batch_size, *self.input_shape))
        # flatten for linear layers.
        x = x.reshape(batch_size, -1)
        x = self.linear_subnet(x)
        return x
