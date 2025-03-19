# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


# TODO: add code reference
class SpectralConvEmbedding(nn.Module):
    def __init__(
        self,
        n_points: int,
        modes: int,
        n_channels: int = 1,
        conv_channels: int = 8,
        num_layers: int = 3,
        n_positions: Optional[Union[np.ndarray, List[float]]] = None,
    ):
        """SpectralConvEmbedding is a neural network module that applies spectral
        convolution to 1D input data (that can come from multiple channels).
        It uses a series of spectral convolution layers and pointwise
        convolution layers to transform the input tensor.

        Args:
            n_points: Number of points in the 1D input data.
            modes: Number of modes for the spectral convolution,
                at most floor(n_points/2) + 1.
            n_channels: Number of channels in the input data.
            conv_channels: Number of convolution channels (same for in and out).
            num_layers: Number of convolution layers.
            n_positions: Positions for the spectral convolution.
                If not provided, uses equispaced points.

        """
        super().__init__()
        assert modes <= n_points // 2 + 1, (
            "Modes should be at most floor(n_points/2) + 1"
        )

        self.modes = modes
        self.n_positions = n_positions
        self.conv_channels = conv_channels
        self.n_channels = n_channels

        # layers
        self.fc0 = nn.Linear(n_channels, self.conv_channels)
        self.conv_layers = nn.ModuleList([
            SpectralConv1d_SMM(self.conv_channels, self.conv_channels, self.modes)
            for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(self.conv_channels, self.conv_channels, 1)
            for _ in range(num_layers)
        ])
        self.conv_last = SpectralConv1d_SMM(
            self.conv_channels, self.conv_channels, self.modes
        )

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.

        Args:
            x: Input tensor (batch_size, channels * n_points).

        Returns:
            Network output (batch_size, conv_channels * modes).
        """

        assert x.shape[-1] == self.n_channels * self.n_points, (
            f"Input tensor should have shape (batch_size, n_channels * n_points), "
            f"got {x.shape[-1]} instead of {self.n_channels * self.n_points}"
        )
        batch_size = x.shape[0]
        x.reshape(batch_size, self.n_channels, -1)  # (batch_size, channels, n_points)

        x = self.fc0(x.permute(0, 2, 1)).permute(0, 2, 1)

        fourier_transform = VFT(batch_size, self.n_points, self.modes, self.n_positions)

        for conv, w in zip(self.conv_layers, self.w_layers, strict=False):
            x1 = conv(x, fourier_transform)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)

        x_spec = self.conv_last(x, fourier_transform)

        return x_spec.reshape(batch_size, -1)


# class for fully nonequispaced 1d points
class VFT:
    def __init__(self, batch_size, n_points, modes, n_positions):
        self.number_points = n_points
        self.batch_size = batch_size
        self.modes = modes

        if n_positions is not None:
            # only works if positions are the same for all data samples
            new_times = n_positions.repeat(self.batch_size, 1) / n_positions.max()
        else:
            new_times = (torch.arange(self.number_points) / self.number_points).repeat(
                self.batch_size, 1
            )

        self.new_times = new_times * 2 * np.pi

        self.X_ = torch.arange(modes).repeat(self.batch_size, 1)[:, :, None].float()
        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        X_mat = torch.bmm(self.X_, self.new_times[:, None, :])
        forward_mat = torch.exp(-1j * (X_mat))

        inverse_mat = torch.conj(forward_mat.clone()).permute(0, 2, 1)

        return forward_mat, inverse_mat

    def forward(self, data, norm='forward'):
        # data has shape: (batch_size, n_points, conv_channel)
        data_fwd = torch.bmm(self.V_fwd, data)
        if norm == 'forward':
            data_fwd /= self.number_points
        elif norm == 'ortho':
            data_fwd /= np.sqrt(self.number_points)

        return data_fwd

    def inverse(self, data, norm='backward'):
        data_inv = torch.bmm(self.V_inv, data)
        if norm == 'backward':
            data_inv /= self.number_points
        elif norm == 'ortho':
            data_inv /= np.sqrt(self.number_points)

        return data_inv


class SpectralConv1d_SMM(nn.Module):
    """
    A 1D spectral convolutional layer using the Fourier transform.
    This layer applies a learned complex multiplication in the frequency domain.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to multiply,
            at most floor(N/2) + 1.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d_SMM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        """
        Performs complex multiplication in the Fourier domain.

        Args:
            input: Input tensor of shape (batch, in_channel, n_points).
            weights: Weight tensor of shape (in_channel, out_channel, n_points).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channel, n_points).
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, transform):
        """
        Forward pass of the spectral convolution layer.

        Args:
            x: Input tensor of shape (batch, channels, points).
            transform: Fourier transform operator with forward and inverse methods.

        Returns:
            The real part of the transformed output tensor.
        """
        batchsize = x.shape[0]

        x = x.permute(0, 2, 1)

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = transform.forward(x.cfloat(), norm='backward')
        x_ft = x_ft.permute(0, 2, 1)

        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            self.modes,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft = self.compl_mul1d(x_ft, self.weights1)

        # Return to physical space
        x_ft = out_ft.permute(0, 2, 1)
        x = transform.inverse(x_ft, norm='backward')  # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)

        return x.real

        def last_layer(self, x, transform):
            """
            Applies the last layer transformation in the Fourier domain.

            Args:
                x: Input tensor of shape (batch, channels, points).
                transform: Fourier transform operator with forward and inverse methods.

            Returns:
                Transformed output tensor (in Fourier domain).
            """
            x = x.permute(0, 2, 1)

            # Compute Fourier coeffcients up to factor of e^(- something constant)
            x_ft = transform.forward(x.cfloat(), norm='forward')

            x_ft = x_ft.permute(0, 2, 1)
            x_ft = self.compl_mul1d(x_ft, self.weights1)

            return torch.view_as_real(x_ft).reshape(x.shape[0], x.shape[2], -1)
