# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import List, Optional, Tuple, Union

from torch import Tensor, nn

from sbi.neural_nets.embedding_nets.cnn import calculate_filter_output_size


def causalConv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int = 1,
    stride: int = 1,
) -> nn.Module:
    """Returns a causal convolution by left padding the input

    Args:
        in_channels: number of input channels
        out_channels: number of output channels wanted
        kernel_size: wanted kernel size
        dilation: dilation to use in the convolution.
        stride: stride to use in the convolution.
            Stride and dilation cannot both be > 1.

    Returns:
        An nn.Sequential object that represents a 1D causal convolution.
    """
    assert not (dilation > 1 and stride > 1), (
        "we don't allow combining stride with dilation."
    )
    padding_size = dilation * (kernel_size - 1)
    padding = nn.ZeroPad1d(padding=(padding_size, 0))
    conv_layer = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=stride,
        padding=0,
    )
    return nn.Sequential(padding, conv_layer)


def WaveNetSRLikeAggregator(
    in_channels: int,
    num_timepoints: int,
    output_dim: int,
    activation: nn.Module = nn.LeakyReLU(inplace=True),
    kernel_sizes: Optional[List] = None,
    intermediate_channel_sizes: Optional[List] = None,
    stride_sizes: Union[int, List] = 1,
) -> nn.Module:
    """
    Creates a non-causal 1D CNN aggregator based on the WaveNet speach recognition task

    By default this function creates an aggregator with two CNN layers,
    after every convolution a maxpooling operation halves the number of timepoints.
    The final CNN will have as many channels as the desired output dimension.
    A global average pooling operation is applied in the end. The dimension of the
    output will thus be (batch_size, output_dim, 1) regardless of the input size.

    Args:
        in_channels: number of channels at input.
        num_timepoints: length of the input.
        output dim: wanted number of features as output.
        activation: activation to apply after the convolution.
        kernel_sizes: (optional) alter the kernel size used and the number of CNN layers
            (through the length of the kernel size vector).
        intermediate_channel_sizes: (optional) alter the intermediate channel sizes
            used, should have length = len(kernel_sizes) - 1.
        stride_sizes = Optional alter the stride used, either a vector of
            len = len(kernel_sizes) or a single integer, in which case the same stride
            is used in every convolution.

    Returns:
        nn.Module object that contains a sequence of CNN and max_pool layer
            and finally a global average pooling layer.
    """
    aggregator_out_shape = (
        in_channels,
        num_timepoints,
    )
    if kernel_sizes is None:
        kernel_sizes = [
            min(9, aggregator_out_shape[-1]),
            min(5, int(aggregator_out_shape[-1] / 2)),
        ]
    if intermediate_channel_sizes is None:
        intermediate_channel_sizes = [64]
    assert len(intermediate_channel_sizes) == len(kernel_sizes) - 1, (
        "Provided kernel size list should be exactly one element longer "
        "than channel size list."
    )
    intermediate_channel_sizes += [output_dim]
    if isinstance(stride_sizes, List):
        assert len(stride_sizes) == len(kernel_sizes), (
            "Provided stride size list should be have the same size as"
            "the kernel size list."
        )
    else:
        stride_sizes = [stride_sizes] * len(kernel_sizes)

    non_causal_layers = []
    for ll in range(len(kernel_sizes)):
        print(aggregator_out_shape)
        conv_layer = nn.Conv1d(
            in_channels=in_channels if ll == 0 else intermediate_channel_sizes[ll - 1],
            out_channels=intermediate_channel_sizes[ll],
            kernel_size=kernel_sizes[ll],
            stride=stride_sizes[ll],
            padding='same',
        )
        maxpool = nn.MaxPool1d(kernel_size=2 if aggregator_out_shape[-1] > 2 else 1)
        non_causal_layers += [conv_layer, activation, maxpool]
        aggregator_out_shape = (
            intermediate_channel_sizes[ll],
            int(
                calculate_filter_output_size(
                    aggregator_out_shape[-1],
                    (kernel_sizes[ll] - 1) / 2,
                    1,
                    kernel_sizes[ll],
                    stride_sizes[ll],
                )
                / 2
            ),
        )
    print(aggregator_out_shape)
    aggregator = nn.Sequential(*non_causal_layers, nn.AdaptiveAvgPool1d(1))
    return aggregator


class CausalCNNEmbedding(nn.Module):
    """Embedding network that uses 1D causal convolutions."""

    def __init__(
        self,
        input_shape: Tuple,
        in_channels: int = 1,
        out_channels_per_layer: Optional[List] = None,
        dilation: Union[str, List] = "exponential_cyclic",
        num_conv_layers: int = 5,
        activation: nn.Module = nn.LeakyReLU(inplace=True),
        pool_kernel_size: int = 160,
        kernel_size: int = 2,
        aggregator: Optional[nn.Module] = None,
        output_dim: int = 20,
    ):
        """Intitialize embedding network that uses 1D causal convolutions.

        This is a simplified version of the architecture introduced for
        the speech recognition task in the WaveNet paper (van den Oord, et al. (2016))

        After several dilated causal convolutions (that maintain the dimensionality
        of the input), an aggregator network is used to bring down the dimensionality.
        You can provide an aggregator network that you deem reasonable for your data.
        If you do not provide an aggregator network yourself, a default aggregator
        is used. This default aggregator is based on the WaveNet paper's description
        of their Speech Recognition Task, and uses non-causal convolutions and pooling
        layers, and global average poolingg to obtain a final low dimensional embedding.

        Args:
            input_shape: Dimensionality of the input e.g. (num_timepoints,),
                currently only 1D is supported.
            in_channels: Number of input channels, default = 1.
            out_channels_per_layer: number of out_channels for each layer, number
                of entries should correspond with num_conv_layers passed below.
                Default = 16 in every convolutional layer.
            dilation: type of dilation to use either one of "none" (dilation = 1
                in every layer), "exponential" (increase dilation by a factor of 2
                every layer), "exponential_cyclic" (as exponential, but reset to 1
                after dilation = 2**9) or pass a list with dilation size per layer.
                By default the cyclic, exponential scheme from WaveNet is used.
            num_conv_layers: the number of causal convolutional layers
            kernel_size: size of the kernels in the causal convolutional layers.
            activation: activation function to use between convolutions,
                default = LeakyReLU.
            pool_kernel_size: pool size to use for the AvgPool1d operation after
                the causal convolutional layers.
            aggregator: aggregation net that reduces the dimensionality of the data
                to a low-dimensional embedding.
            output_dim: number of output units in the final layer when using
                the default aggregation
        """

        super(CausalCNNEmbedding, self).__init__()
        assert isinstance(input_shape, Tuple), (
            "input_shape must be a Tuple of size 1, e.g. (timepoints,)."
        )
        assert len(input_shape) == 1, "Currently only 1D causal CNNs are supported."
        self.input_shape = (in_channels, *input_shape)

        total_timepoints = input_shape[0]
        assert total_timepoints >= pool_kernel_size, (
            "Please ensure that the pool kernel size is not "
            "larger than the number of observed timepoints."
        )
        if isinstance(dilation, str):
            match dilation.lower():
                case "exponential_cyclic":
                    max_dil_exp = 10
                    ## Use dilation scheme as in WaveNet paper
                    dilation_per_layer = [
                        2 ** (i % max_dil_exp) for i in range(num_conv_layers)
                    ]
                case "exponential":
                    dilation_per_layer = [2**i for i in range(num_conv_layers)]
                case "none":
                    dilation_per_layer = [1] * num_conv_layers
                case _:
                    raise ValueError(
                        f"{dilation} is not a valid option, please use \"none\","
                        "\"exponential\",or \"exponential_cyclic\", or pass a list "
                        "of dilation sizes."
                    )
        else:
            assert isinstance(dilation, List), (
                "Please pass dilation size as list or a string option."
            )
            dilation_per_layer = dilation

        assert max(dilation_per_layer) < total_timepoints, (
            "Your maximal dilations size used is larger than the number of "
            "timepoints in your input, please provide a list with smaller dilations."
        )
        if out_channels_per_layer is None:
            out_channels_per_layer = [16] * num_conv_layers

        causal_layers = []
        for ll in range(num_conv_layers):
            causal_layers += [
                causalConv1d(
                    in_channels if ll == 0 else out_channels_per_layer[ll - 1],
                    out_channels_per_layer[ll],
                    kernel_size,
                    dilation_per_layer[ll],
                    1,
                ),
                activation,
            ]

        self.causal_cnns = nn.Sequential(*causal_layers)

        self.pooling_layer = nn.AvgPool1d(kernel_size=pool_kernel_size)

        if aggregator is None:
            aggregator_out_shape = (
                out_channels_per_layer[-1],
                int(total_timepoints / pool_kernel_size),
            )
            assert aggregator_out_shape[-1] > 1, (
                "Your dimensionality is already small,"
                "Please ensure a larger input size or use a custom aggregator."
            )
            aggregator = WaveNetSRLikeAggregator(
                aggregator_out_shape[0],
                aggregator_out_shape[-1],
                output_dim=output_dim,
            )
        self.aggregation = aggregator

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, *self.input_shape)
        x = self.causal_cnns(x)
        x = self.pooling_layer(x)
        x = self.aggregation(x)
        # ensure flattening when aggregator uses global average pooling
        x = x.view(batch_size, -1)
        return x
