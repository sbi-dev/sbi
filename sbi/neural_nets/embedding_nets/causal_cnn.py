# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import List, Optional, Tuple

from torch import Tensor, nn

from sbi.neural_nets.embedding_nets.cnn import calculate_filter_output_size


def CausalCNN_1d_block(in_channels, out_channels, kernel_size, dilation, activation):
    """Returns a block that computes a causal convolution by left padding the input

    Args:
        in_channels: number of input channels
        out_channels: number of output channels wanted
        kernel_size: wanted kernel size
        dilation: dilation to use in the convolution,
        activation: activation function to append to the convolution

    Returns:
        an nn.Sequential object that combines left-padding, 1D convolution
        and the provided activation
    """
    padding_size = dilation * (kernel_size - 1)
    padding = nn.ZeroPad1d(padding=(padding_size, 0))
    conv_layer = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=1,
        padding=0,
    )
    return nn.Sequential(padding, conv_layer, activation)


class CausalCNNEmbedding(nn.Module):
    def __init__(
        self,
        input_shape: Tuple,
        in_channels: int = 1,
        out_channels_per_layer: Optional[List] = None,
        dilation_per_layer: Optional[List] = None,
        num_conv_layers: int = 5,
        activation: nn.Module = nn.LeakyReLU(inplace=True),
        pool_kernel_size: int = 160,
        kernel_size: int = 2,
        aggregator: Optional[nn.Module] = None,
        output_dim: int = 20,
    ):
        """Embedding network that uses 1D causal convolutions
        This is a simplified version of the architecture introduced for
        the speech recognition task in the WaveNet paper (van den Oord, et al. (2016))

        After several dilated causal convolutions (that maintain the dimensionality
        of the input), an aggregator network is used to bring down the dimensionality.
        You can provide an aggregator network that you deem reasonable for your data

        Args:
            input_shape: Dimensionality of the input e.g. (num_timepoints,),
                    currently only 1D is supported.
            in_channels: Number of input channels, default = 1.
            out_channels_per_layer: number of out_channels for each layer, number
                    of entries should correspond with num_conv_layers passed below.
                    Default = 16 in every convolutional layer.
            dilation_per_layer: dilation size per layer, by default the cyclic,
                    exponential scheme from WaveNet is used.
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

        tot_timepoints = input_shape[0]
        assert tot_timepoints >= pool_kernel_size, (
            "Please ensure that the pool kernel size is not "
            "larger than the number of observed timepoints."
        )
        max_dil_exp = 10
        if dilation_per_layer is None:
            ## Use dilation scheme as in WaveNet paper
            dilation_per_layer = [
                2 ** (i % max_dil_exp) for i in range(num_conv_layers)
            ]

        if out_channels_per_layer is None:
            out_channels_per_layer = [16] * num_conv_layers
        self.padding_size = dilation_per_layer * (kernel_size - 1)
        causal_layers = []
        for ll in range(num_conv_layers):
            padding = nn.ZeroPad1d(padding=(self.padding_size[ll], 0))
            conv_layer = nn.Conv1d(
                in_channels=in_channels if ll == 0 else out_channels_per_layer[ll - 1],
                out_channels=out_channels_per_layer[ll],
                kernel_size=kernel_size,
                dilation=dilation_per_layer[ll],
                stride=1,
                padding=0,
            )
            causal_layers += [padding, conv_layer, activation]

        self.causal_cnns = nn.Sequential(*causal_layers)

        self.pooling_layer = nn.AvgPool1d(kernel_size=pool_kernel_size)

        if aggregator is None:
            aggregator_out_shape = (
                out_channels_per_layer[-1],
                int(tot_timepoints / pool_kernel_size),
            )
            assert aggregator_out_shape[-1] > 1, (
                "Your dimensionality is allready small,"
                "Please ensure a larger input size or use a custom aggregator."
            )
            non_causal_kernel_sizes = [
                min(9, aggregator_out_shape[-1]),
                min(5, int(aggregator_out_shape[-1] / 2)),
            ]
            non_causal_channel_sizes = [64, output_dim]
            non_causal_layers = []
            for ll in range(len(non_causal_kernel_sizes)):
                conv_layer = nn.Conv1d(
                    in_channels=out_channels_per_layer[-1]
                    if ll == 0
                    else non_causal_channel_sizes[ll - 1],
                    out_channels=non_causal_channel_sizes[ll],
                    kernel_size=non_causal_kernel_sizes[ll],
                    stride=1,
                    padding='same',
                )
                maxpool = nn.AvgPool1d(
                    kernel_size=2 if aggregator_out_shape[-1] > 2 else 1
                )
                non_causal_layers += [conv_layer, activation, maxpool]
                aggregator_out_shape = (
                    non_causal_channel_sizes[ll],
                    int(
                        calculate_filter_output_size(
                            aggregator_out_shape[-1],
                            1,
                            1,
                            non_causal_kernel_sizes[ll],
                            1,
                        )
                        / 2
                    ),
                )
            aggregator = nn.Sequential(
                *non_causal_layers, nn.AvgPool1d(kernel_size=aggregator_out_shape[-1])
            )
        self.aggregation = aggregator

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, *self.input_shape)
        x = self.causal_cnns(x)
        x = self.pooling_layer(x)
        x = self.aggregation(x)
        # ensure flattening when aggregator uses GAP
        x = x.view(batch_size, -1)
        return x
