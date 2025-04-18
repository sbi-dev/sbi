from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


def zero_pad(x: Tensor, c_out: int) -> Tensor:
    """
    Add additional channels to the input tensor by padding it with zeros.

    Args:
        x:      Input tensor
        c_out:  Number of output channels

    Returns:
        x_aug:      Input tensor with additional channels filled with zeros
    """

    # Check if it is an image like-input
    if not (len(x.shape) == 4):
        raise ValueError("Only 4D input tensors are supported.")

    # Check if the number of channels is increased
    if c_out <= x.shape[1]:
        raise ValueError("c_out must be larger than c_in to apply zero padding.")

    c_in = x.shape[1]
    a = torch.zeros(x.shape[0], c_out - c_in, x.shape[2], x.shape[3], device=x.device)

    x_aug = torch.cat([x, a], dim=1)

    return x_aug


def construct_simple_conv_net(
    c_in: int,
    c_out: int,
    c_hidden: int,
    activation: Type[nn.Module],
    downsample: bool,
) -> nn.Module:
    """
    Default implementation for a simple convolutional network with 3 layers.

    Args:
        c_in:       Number of input channels
        c_out:      Number of output channels
        c_hidden:   Number of hidden channels
        activation: Constructor for an activation function
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


def construct_simple_fc_net(
    c_in: int, c_out: int, c_hidden: int, activation: Type[nn.Module]
) -> nn.Module:
    """
    Default implementation for a fully connected neural network.

    Args:
        c_in:       Number of input channels
        c_out:      Number of output channels
        c_hidden:   Number of hidden channels
        activation: Constructor for an activation function

    Returns:
        layers:     A sequential container with the fully connected layers
    """

    layers = nn.Sequential(
        nn.Linear(c_in, c_hidden),
        activation(),
        nn.Linear(c_hidden, c_hidden),
        activation(),
        nn.Linear(c_hidden, c_out),
    )

    return layers


class ResNetEmbedding2D(nn.Module):
    """Residual neural network mapping image-like data to a fixed sized vector."""

    def __init__(
        self,
        c_in: int,
        c_out: int = 20,
        c_hidden_fc: int = 1000,
        n_stages: int = 4,
        blocks_per_stage: Optional[List] = None,
        c_stages: Optional[List] = None,
        activation: Type[nn.Module] = nn.ReLU,
        change_c_mode: str = "conv",
        construct_mapping: Callable = construct_simple_conv_net,
        construct_mapping_kwargs: Optional[Dict] = None,
        residual_block_kwargs: Optional[Dict] = None,
    ) -> None:
        """Residual neural network mapping image-like data to a fixed sized vector.

        This network consists of a stacked set of residual blocks. The output of
        a block is given by the input to the block plus the transformed input,
        i.e. there is a skip connection.

        The network is structured in stages, where
        after each stage, the height and the width of the input are halved. Each
        stage can have a different number of residual blocks and a different
        number of channels.

        Image-like input is expected, i.e., a 4D tensor with dimensions
        (batch_size, channels, height, width) or a 3D tensor with dimensions
        (batch_size, height, width). In the case of 3D input the input is internally
        transformed to a 4D tensor with dimensions (batch_size, 1, height, width).

        The image like input data is transformed into a fixed sized vector of
        dimensions [batch_size, c_out].

        By default, convolutional networks are used to model the transformation
        in each of the residual blocks. At the end of the network, a fully
        connected network is applied to map the flattened output of the last
        stage to the fixed sized output vector.

        References:

        He et al. (2015): "Deep Residual Learning for Image Recognition"

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
                               residual connection is added. The default is a simple
                               convolutional network. The function must have the
                               signature
                               `construct_mapping(c_in, c_out, activation, **kwargs)`.
            construct_mapping_kwargs: Additional keyword arguments for the mapping
                                      functions.
            residual_block_kwargs: Additional keyword arguments for the initialization
                                   of the residual blocks.
        """
        super().__init__()

        # Additional keyword arguments for initializing the mapping function
        if construct_mapping_kwargs is None:
            construct_mapping_kwargs = {"c_hidden": 128}

        construct_mapping_kwargs["activation"] = activation

        # Additional keyword arguments for initializing the residual blocks
        if residual_block_kwargs is None:
            residual_block_kwargs = {}

        # Number of residual blocks in each stage
        if blocks_per_stage is None:
            blocks_per_stage = [2, 2, 2, 2]

        # Number of channels in each stage
        if c_stages is None:
            c_stages = [64, 128, 256, 512]

        # Check consistency of the specified network structure
        if not len(blocks_per_stage) == n_stages:
            raise ValueError(
                "The number of stages and the number of specified block must match."
            )

        if not len(c_stages) == n_stages:
            raise ValueError(
                "The number of stages and the number of specified channels must match."
            )

        self.c_hidden_fc = c_hidden_fc
        self.c_out = c_out
        self.activation = activation

        # Intial transformation applied before the residual blocks
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

                construct_mapping_kwargs["downsample"] = downsample

                block_ij = ResidualBLock(
                    is_conv_block=True,
                    c_in=c_stages[i],
                    c_out=c_out_ij,
                    downsample=downsample,
                    change_c_mode=change_c_mode,
                    construct_mapping=construct_mapping,
                    construct_mapping_kwargs=construct_mapping_kwargs,
                    activation=activation,
                    **residual_block_kwargs,
                )

                blocks.append(block_ij)

        self.blocks = nn.Sequential(*blocks)

        # Final transformation is initialized on the first forward pass
        self.final = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the residual network.

        Args:
            x:      Input tensor

        Returns:
            z:      Output tensor
        """

        # Only three dimensions, interpret as one channel and add it
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        if not len(x.shape) == 4:
            raise ValueError("Only 4D input tensors are supported.")

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


class ResNetEmbedding1D(nn.Module):
    """Residual neural network mapping vector-like data to a fixed-size vector."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        construct_mapping: Callable = construct_simple_fc_net,
        construct_mapping_kwargs: Union[Dict, None] = None,
        residual_block_kwargs: Union[Dict, None] = None,
        n_blocks: int = 20,
        c_internal: int = 128,
        c_hidden_final: int = 1000,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """Residual neural network mapping vector-like data to a fixed-size vector.

        This network consists of a stacked set of residual blocks. The output of
        a block is given by the input to the block plus the transformed input,
        i.e., there is a skip connection.

        The input is expected to be vector-like, i.e., a 2D tensor with dimensions
        (batch_size, channels).

        The input data is transformed into a fixed-size vector of dimensions
        [batch_size, c_out].

        By default, fully connected networks are used to model the transformation
        in each of the residual blocks.

        Initially, the input can be mapped to a space of different dimensionality
        than the input. On this space, the residual blocks act. At the end of
        the network, a fully connected network is applied to map the output of the last
        residual block to the fixed-size output vector.

        Args:
            c_in: Input dimensionality.
            c_out: Dimensionality of the embedding vector.
            construct_mapping: Constructor for the mapping function. The mapping
                            function is applied to the input tensor before the
                            residual connection is added. The default is a simple
                            fully connected network. The function must have the
                            signature
                            `construct_mapping(c_in, c_out, activation, **kwargs)`.
            construct_mapping_kwargs: Additional keyword arguments for the mapping
                                      functions.
            residual_block_kwargs: Additional keyword arguments for the initialization
                                   of the residual blocks.
            n_blocks: Number of residual blocks.
            c_internal: Dimensionality of the internal space.
            c_hidden_final: Hidden dimensionality of the final aggregation network.
            activation: Constructor for the activation function.
        """

        super().__init__()

        # Parameteters for the subnetwork initialization
        if construct_mapping_kwargs is None:
            construct_mapping_kwargs = {"c_hidden": 128}

        construct_mapping_kwargs["activation"] = activation

        # Parameters for the coupling block initialization
        if residual_block_kwargs is None:
            residual_block_kwargs = {}

        # Mapping of the input to the dimensionality used internally
        if c_in == c_internal:
            self.initial = nn.Identity()
        else:
            self.initial = nn.Linear(c_in, c_internal)

        # Final aggregation network
        self.final = nn.Sequential(
            nn.Linear(c_internal, c_hidden_final),
            activation(),
            nn.Linear(c_hidden_final, c_hidden_final),
            activation(),
            nn.Linear(c_hidden_final, c_out),
        )

        # Residual blocks
        blocks = nn.ModuleList()

        for _i in range(n_blocks):
            block_i = ResidualBLock(
                c_in=c_internal,
                c_out=c_internal,
                is_conv_block=False,
                construct_mapping=construct_mapping,
                construct_mapping_kwargs=construct_mapping_kwargs,
                activation=activation,
                **residual_block_kwargs,
            )

            blocks.append(block_i)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the residual network.

        Args:
            x:      Input tensor

        Returns:
            z:      Output tensor
        """

        # Check if vector like data is used
        if not len(x.shape) == 2:
            raise ValueError("Only 2D input tensors are supported.")

        # Apply the initial transformation
        x = self.initial(x)

        # Apply the residual blocks
        y = self.blocks(x)

        # Apply the final transformation
        z = self.final(y)

        return z


class ResidualBLock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: Optional[int],
        is_conv_block: bool,
        construct_mapping: Callable,
        construct_mapping_kwargs: Optional[Dict],
        activation: Type[nn.Module],
        downsample: bool = False,
        change_c_mode: str = "conv",
    ) -> None:
        """
        Single residual block used in residual networks.

        Args:
            c_in: Number of input channels.
            c_out: Number of output channels. If None, use c_in.
            change_c_mode: Mode to change the number of channels.
                        Options are "conv" and "zeros". If "conv" is selected,
                        1x1 convolutions are used to adjust the number of channels.
                        If "zeros" is selected, any additional required channels
                        are filled with zeros. Only relevant for image-like data.
            downsample: Apply a strided convolution to halve the height and
                        width of the input. Only relevant for image-like data.
            construct_mapping: Constructor for the mapping function. The mapping
                            function is applied to the input tensor before the
                            residual connection is added.
            construct_mapping_kwargs: Additional keyword arguments for the mapping
                                      function.
            activation: Constructor for the activation function.
        """

        super(ResidualBLock, self).__init__()

        # Additional keyword arguments for the mapping function
        if construct_mapping_kwargs is None:
            construct_mapping_kwargs = {}

        ###############################################################################
        # Change number of channels
        ###############################################################################

        # Preserve the input dimensionality if the output is not specified
        if c_out is None:
            c_out = c_in

        # Preserve the dimensionality of the input and the output of the block
        if c_in == c_out:
            self.residual = nn.Identity()

        # Initialize the transformation of the residual for the case of image-like data
        elif is_conv_block and (c_in != c_out):
            # Apply 1x1 convolutions to change the number of channels
            if change_c_mode == "conv":
                self.residual = nn.Conv2d(
                    c_in, c_out, kernel_size=1, stride=1, bias=False
                )

            # Fill the required additional channels with zeros
            elif change_c_mode == "zeros":
                # Check if there are more output channels than input channels
                if c_in > c_out:
                    raise ValueError("c_in channels must be smaller than c_out.")

                self.residual = partial(zero_pad, c_out=c_out)

            else:
                raise ValueError(f"Invalid change_c_mode {change_c_mode}.")

        # Initialize the transformation of the residual for the case of vector-like data
        else:
            self.residual = nn.Linear(c_in, c_out, bias=False)

        ###############################################################################
        # Halve the height and width of the input for image-like data
        ###############################################################################

        # Apply a strided convolution if the input is downsampled
        if is_conv_block and downsample:
            self.downsampling = nn.Conv2d(
                c_out, c_out, kernel_size=3, stride=2, bias=False, padding=1
            )
            construct_mapping_kwargs["downsample"] = downsample

        # No downsampling
        else:
            self.downsampling = nn.Identity()

        ###############################################################################
        # Define the transformation of the residual block
        ###############################################################################

        self.f = construct_mapping(c_in, c_out, **construct_mapping_kwargs)

        self.final_activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the residual block.

        Args:
            x:      Input tensor

        Returns:
            y:      Output tensor
        """

        # Check if image like data is used
        if not ((len(x.shape) == 4) or (len(x.shape) == 2)):
            raise ValueError("Only 2D images or 1D vectors are supported.")

        # Compute the residual
        r = self.downsampling(self.residual(x))

        # Full transformation
        y = self.final_activation(self.f(x) + r)

        return y
