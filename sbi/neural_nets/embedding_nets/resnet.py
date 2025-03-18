from functools import partial
from typing import Callable, Type

import torch
import torch.nn as nn


def zero_pad(x: torch.tensor, c_out: int) -> torch.tensor:
    """
    Add additional channels to the input tensor by filling it with zeros.

    Args:
        x:      Input tensor
        c_out:  Number of output channels

    Returns:
        x_aug:      Input tensor with additional channels filled with zeros
    """

    # Check if it is a image like iput
    assert len(x.shape) == 4

    # Check if the number of channels is increased
    assert c_out >= x.shape[1]

    c_in = x.shape[1]
    a = torch.zeros(x.shape[0], c_out - c_in, x.shape[2], x.shape[3], device=x.device)

    x_aug = torch.cat([x, a], dim=1)

    return x_aug


def construct_simple_conv_net(
    c_in: int,
    c_out: int,
    c_hidden: int = 32,
    activation: Type[nn.Module] = nn.ReLU,
    downsample: int = False,
) -> nn.Module:
    """
    Default implementation for a simple convolutional network with 3 layers.

    Args:
        c_in:       Number of input channels
        c_out:      Number of output channels
        c_hidden:   Number of hidden channels
        activation: Constructor for a activation function
        downsample: Apply a strided convolution to halve
                    the height and width of the input

    Returns:
        layers:     A sequential container with the convolutional layers
    """

    # Define the convolutional layers
    stride = 2 if downsample else 1
    layers = nn.Sequential(
        nn.Conv2d(c_in, c_hidden, kernel_size=3, stride=stride, padding=1),
        activation(),
        nn.Conv2d(c_hidden, c_hidden, kernel_size=3, stride=1, padding=1),
        activation(),
        nn.Conv2d(c_hidden, c_out, kernel_size=3, stride=1, padding=1),
    )

    return layers


class ResidualBLockConv(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int | None = None,
        change_c_mode: str = "conv",
        downsample: bool = False,
        construct_mapping: Callable = construct_simple_conv_net,
        construct_mapping_kwargs: dict | None = None,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        Single residual block for image like data.

        Args:
            c_in: Number of input channels.
            c_out: Number of output channels. If None, use c_in.
            change_c_mode: Mode to change the number of channels.
                           Options are "conv" and "zeros". If "conv" is selected,
                           1x1 convolutions are applied to change the number
                           of channels. If "zeros" is selected, the required additional
                           channels are filled with zeros.
            downsample: Apply a strided convolution to halve the height and
                        width of the input.
            construct_mapping: Constructor for the mapping function. The mapping
                               function is applied to the input tensor before the
                               residual connection is added.
            construct_mapping_kwargs: Additional keyword arguments for the mapping
                                      function.
            activation: Constructor for the activation function.
        """

        super(ResidualBLockConv, self).__init__()

        if construct_mapping_kwargs is None:
            construct_mapping_kwargs = {}

        ###############################################################################
        # Change number of channels
        ###############################################################################

        # Preserve the input dimensionality if the output is not specified
        if c_out is None:
            c_out = c_in

        # Preserve the dimensionality of the input and the output
        if c_in == c_out:
            self.residual = nn.Identity()

        # Apply 1x1 convolutions to change the number of channels
        elif (c_in != c_out) and change_c_mode == "conv":
            self.residual = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False)

        # Fill the required additional channels with zeros
        elif (c_in != c_out) and change_c_mode == "zeros":
            # Checkt if there are more output channels than input channels
            if c_in > c_out:
                raise ValueError("c_in channels must be smaller than c_out.")

            self.residual = partial(zero_pad, c_out=c_out)

        ###############################################################################
        # Halve the height and width of the input
        ###############################################################################

        # Apply a strided convolution if the input is downsampled
        if downsample:
            self.downsampling = nn.Conv2d(
                c_out, c_out, kernel_size=3, stride=2, bias=False, padding=1
            )

        # No downsampling
        else:
            self.downsampling = nn.Identity()

        ###############################################################################
        # Define the transformation of the residual block
        ###############################################################################

        self.f = construct_mapping(
            c_in, c_out, downsample=downsample, **construct_mapping_kwargs
        )

        self.activation = activation()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the residual block.

        Args:
            x:      Input tensor

        Returns:
            y:      Output tensor
        """

        # Check if image like data is used
        assert len(x.shape) == 4

        # Compute the skip connection
        r = self.downsampling(self.residual(x))

        # Full transformation
        y = self.activation(self.f(x) + r)

        return y


class ResNet(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int = 20,
        c_hidden_fc: int = 1000,
        n_stages: int = 4,
        blocks_per_stage: list | None = None,
        c_stages: list | None = None,
        activation: Type[nn.Module] = nn.ReLU,
        change_c_mode: str = "conv",
        construct_mapping: Callable = construct_simple_conv_net,
        construct_mapping_kwargs: dict | None = None,
        coupling_block_kwargs: dict | None = None,
    ) -> None:
        """
        Residual neural network mapping image like data to a fixed sized vector.

        Args:
            c_in: Number of input channels.
            c_out: Dimensionality of the embedding vector.
            c_hidden_fc: Number of hidden units in the fully connected layers.
            n_stages: Number of stages in the network.
            blocks_per_stage: Number of residual blocks per stage.
            c_stages: Number of channels per stage.
            activation: Constructor for the activation function
            change_c_mode: Mode to change the number of channels. Options are
                           "conv" and "zeros". If "conv" is selected, 1x1 convolutions
                           are applied to change the number of channels. If "zeros" is
                           selected, the required additional channels are filled with
                           zeros.
            construct_mapping: Constructor for the mapping function. The mapping
                               function is applied to the input tensor before the
                               residual connection is added.
            construct_mapping_kwargs: Additional keyword arguments for the mapping
                                      functions.
            coupling_block_kwargs: Additional keyword arguments for the coupling blocks.
        """
        super().__init__()

        if construct_mapping_kwargs is None:
            construct_mapping_kwargs = {}

        if coupling_block_kwargs is None:
            coupling_block_kwargs = {}

        if blocks_per_stage is None:
            blocks_per_stage = [2, 2, 2, 2]

        if c_stages is None:
            c_stages = [64, 128, 256, 512]

        self.c_hidden_fc = c_hidden_fc
        self.c_out = c_out
        self.activation = activation

        # Intial transformation
        self.initial = nn.Sequential(
            nn.Conv2d(c_in, c_stages[0], kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Initialize the residual blocks

        construct_mapping_kwargs["activation"] = activation

        blocks = nn.ModuleList()

        for i in range(n_stages):
            for j in range(blocks_per_stage[i]):
                # Last block in the stage is a downsampling block which
                # increases the number of channels, except for the final stage
                if (j == blocks_per_stage[i] - 1) and (i != n_stages - 1):
                    downsample = True
                    c_out_ij = c_stages[i + 1]
                else:
                    downsample = False
                    c_out_ij = c_stages[i]

                block_ij = ResidualBLockConv(
                    c_in=c_stages[i],
                    c_out=c_out_ij,
                    downsample=downsample,
                    change_c_mode=change_c_mode,
                    construct_mapping=construct_mapping,
                    construct_mapping_kwargs=construct_mapping_kwargs,
                    activation=activation,
                    **coupling_block_kwargs,
                )

                blocks.append(block_ij)

        self.blocks = nn.Sequential(*blocks)

        # Final transformation
        self.final = None

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the residual network.

        Args:
            x:      Input tensor

        Returns:
            z:      Output tensor
        """

        # Check if image like data is used
        assert len(x.shape) == 4

        # Apply the initial transformation
        x = self.initial(x)

        # Apply the residual blocks
        y = self.blocks(x)

        # On first call intitalize the final transformation
        if self.final is None:
            # Get the shape of the input to the final layer
            c_in_final = int(torch.tensor(y.shape[1:]).prod().item())

            self.final = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                nn.Flatten(),
                nn.Linear(c_in_final, self.c_hidden_fc),
                self.activation(),
                nn.Linear(self.c_hidden_fc, self.c_hidden_fc),
                self.activation(),
                nn.Linear(self.c_hidden_fc, self.c_out),
            )

        # Apply the final transformation
        z = self.final(y)

        return z
