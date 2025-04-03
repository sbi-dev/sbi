# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# This code is based on on the following three papers:

# Lingsch et al. (2024) FUSE: Fast Unified Simulation and Estimation for PDEs
# (https://proceedings.neurips.cc/paper_files/paper/2024/file/266c0f191b04cbbbe529016d0edc847e-Paper-Conference.pdf)
#
# Lingsch et al. (2024) Beyond Regular Grids: Fourier-Based Neural Operators
# on Arbitrary Domains
# (https://arxiv.org/pdf/2305.19663)

# Li et al. (2021) Fourier Neural Operator for Parametric Partial Differential Equations
# (https://openreview.net/pdf?id=c8P9NQVtmnO)

# and partially adapted from the following repository:
# https://github.com/camlab-ethz/FUSE

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VFT:
    """Class for performing Fourier transformations for non-equally
    and equally spaced 1d grids.

    It provides a function for creating grid-dependent operator V to compute the
    Forward Fourier transform X of data x with X = V*x.
    The inverse Fourier transform can then be computed by x = V_inv*X with
    V_inv = transpose(conjugate(V)).

    Adapted from: Lingsch et al. (2024) Beyond Regular Grids: Fourier-Based
    Neural Operators on Arbitrary Domains

    Args:
        batch_size: Training batch size
        n_points: Number of 1d grid points
        modes: number of Fourier modes that should be used
            (maximal floor(n_points/2) + 1)
        point_positions: Grid point positions of shape (batch_size, n_points).
            If not provided, equispaced points are used. Positions have to be
            normalized with domain length.
    """

    def __init__(
        self,
        batch_size: int,
        n_points: int,
        modes: int,
        point_positions: Optional[Tensor] = None,
    ):
        self.number_points = n_points
        self.batch_size = batch_size
        self.modes = modes

        if point_positions is not None:
            new_times = point_positions[:, None, :]
        else:
            new_times = (
                (torch.arange(self.number_points) / self.number_points).repeat(
                    self.batch_size, 1
                )
            )[:, None, :]

        self.new_times = new_times * 2 * np.pi

        self.X_ = torch.arange(modes).repeat(self.batch_size, 1)[:, :, None].float()
        # V_fwd: (batch, modes, points) V_inf: (batch, points, modes)
        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self) -> Tuple[Tensor, Tensor]:
        """Create matrix operators V and V_inf for forward and backward
        Fourier transformation on arbitrary grids
        """

        X_mat = torch.bmm(self.X_, self.new_times)
        forward_mat = torch.exp(-1j * (X_mat))

        inverse_mat = torch.conj(forward_mat.clone()).permute(0, 2, 1)

        return forward_mat, inverse_mat

    def forward(self, data: Tensor, norm: str = 'forward') -> Tensor:
        """Perform forward Fourier transformation
        Args:
            data: Input data with shape (batch_size, n_points, conv_channel)
        """
        if norm == 'forward':
            data_fwd = torch.bmm(self.V_fwd, data) / self.number_points
        elif norm == 'ortho':
            data_fwd = torch.bmm(self.V_fwd, data) / np.sqrt(self.number_points)
        elif norm == 'backward':
            data_fwd = torch.bmm(self.V_fwd, data)

        return data_fwd  # (batch, modes, conv_channels)

    def inverse(self, data: Tensor, norm: str = 'forward') -> Tensor:
        """Perform inverse Fourier transformation
        Args:
            data: Input data with shape (batch_size, modes, conv_channel)
        """
        if norm == 'backward':
            data_inv = torch.bmm(self.V_inv, data) / self.number_points
        elif norm == 'ortho':
            data_inv = torch.bmm(self.V_inv, data) / np.sqrt(self.number_points)
        elif norm == 'forward':
            data_inv = torch.bmm(self.V_inv, data)

        return data_inv  # (batch, n_points, conv_channels)


class SpectralConv1d_SMM(nn.Module):
    """
    A 1D spectral convolutional layer using the Fourier transform.
    This layer applies a learned complex multiplication in the frequency domain.

    Adapted from:
    - Lingsch et al. (2024) FUSE: Fast Unified Simulation and Estimation for PDEs
    - Li et al. (2021) Fourier Neural Operator for Parametric Partial Differential
                        Equations

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
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: Tensor, weights: Tensor) -> Tensor:
        """
        Performs complex multiplication in the Fourier domain.

        Args:
            input: Input tensor of shape (batch, in_channels, modes).
            weights: Weight tensor of shape (in_channels, out_channels, modes).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, modes).
        """

        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: Tensor, transform: VFT) -> Tensor:
        """
        Forward pass of the spectral convolution layer.

        Args:
            x: Input tensor of shape (batch, n_points, in_channels).
            transform: Fourier transform operator with forward and inverse methods.

        Returns:
            The real part of the transformed output tensor
            with shape (batch, points, out_channels).
        """
        # Compute Fourier coefficients
        x_ft = transform.forward(x.to(torch.complex64), norm='forward')
        x_ft = x_ft.permute(0, 2, 1)
        out_ft = self.compl_mul1d(x_ft, self.weights1)
        x_ft = out_ft.permute(0, 2, 1)

        # Return to physical space
        x = transform.inverse(x_ft, norm='forward')

        return x.real

    def last_layer(self, x: Tensor, transform: VFT) -> Tensor:
        """
        Last convolutional layer returning Fourier coefficients to be used as embedding

        Args:
            x: Input tensor of shape (batch, points, in_channels).
            transform: Fourier transform operator with forward and inverse methods.

        Returns:
            Transformed output tensor of shape (batch, 2*modes, out_channels).
        """

        # Compute Fourier coeffcients
        x_ft = transform.forward(x.to(torch.complex64), norm='forward')
        x_ft = x_ft.permute(0, 2, 1)
        x_ft = self.compl_mul1d(x_ft, self.weights1)  # (batch, conv_channels, modes)
        x_ft = x_ft.permute(0, 2, 1)  # (batch, modes, conv_channels)
        x_ft = torch.view_as_real(x_ft)  # (batch, modes, conv_channels, 2)
        x_ft = x_ft.permute(0, 1, 3, 2)
        x_ft = x_ft.reshape(x.shape[0], 2 * self.modes, self.out_channels)

        return x_ft


class SpectralConvEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        modes: int = 10,
        out_channels: int = 1,
        conv_channels: int = 5,
        num_layers: int = 3,
    ):
        """SpectralConvEmbedding is a neural network module that performs convolution
        in Fourier space for 1D input data (that can have multiple channels).
        It uses a series of spectral convolution layers and pointwise
        convolution layers to transform the input tensor.

        Adapted from: Lingsch et al. (2024) Beyond Regular Grids: Fourier-Based
        Neural Operators on Arbitrary Domains

        Args:
            in_channels: Number of channels in the input data.
            modes: Number of modes considered in the spectral convolution,
                at most floor(n_points/2) + 1.
            out_channels: number of channels for final output.
            conv_channels: Number of going in and out convolutional layer.
            num_layers: Number of convolution layers.

        """
        super().__init__()

        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_channels = conv_channels
        self.num_layers = num_layers

        # Initialize fully connected layer to raise number of
        # input channels to number of convolutional channels
        self.fc0 = nn.Linear(self.in_channels, self.conv_channels)

        # Inititalize layers performing convolution in Fourier space
        self.conv_layers = nn.ModuleList([
            SpectralConv1d_SMM(self.conv_channels, self.conv_channels, self.modes)
            for _ in range(self.num_layers)
        ])

        # Initialize layer performing pointwise convolution
        self.w_layers = nn.ModuleList([
            nn.Conv1d(self.conv_channels, self.conv_channels, 1)
            for _ in range(self.num_layers)
        ])

        # Initialize last convolutional layer with output in Fourier space
        self.conv_last = SpectralConv1d_SMM(
            self.conv_channels, self.conv_channels, self.modes
        )

        # Initialize fully connected layer to reduce number of output channels
        self.fc_last = nn.Linear(self.conv_channels, self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass.

        Args:
            x: 3D input tensor (batch_size, in_channels, n_points) for equi-spaced data
            or 4D tensor (batch_size, 2, in_channels, n_points) for non-equispaced data,
            where we additionally pass the point positions in the second dimension,
            repeating the same point positions for each channel.
            For non-equispaced data, the positions have to be normalized with
            physical domain length.

        Exemplary code:

        # Example for equispaced grid data with batch size of 256, 3 channels and
        # sequence length of 500
        data_equispaced = torch.rand(256, 3, 500)
        embedding_net = SpectralConvEmbedding(modes=15, in_channels=3,
            out_channels=1, conv_channels=5, num_layers=4)
        neural_posterior = posterior_nn(model="nsf", embedding_net=embedding_net)
        inference = SNPE(prior=sbi_prior, density_estimator=neural_posterior)
        _ = inference.append_simulations(theta, data_equispaced)

        # Example for non-equispaced data with batch size of 256, 3 channels and
        # sequence length of 500
        irregular_positions = torch.rand(500) # non-equally spaced positions in [0;1]
        irregular_positions, indices = torch.sort(irregular_positions, 0)
        irregular_positions = irregular_positions.repeat(256, 3, 1)

        random_data = torch.rand(256, 3, 500)

        data_nonequispaced = torch.zeros(256, 2, 3, 500)
        data_nonequispaced[:, 0, :, :] = random_data
        data_nonequispaced[:, 1, :, :] = irregular_positions

        embedding_net = SpectralConvEmbedding(modes=15, in_channels=3, out_channels=1,
            conv_channels=5, num_layers=4)
        neural_posterior = posterior_nn(model="nsf", embedding_net=embedding_net)
        inference = SNPE(prior=sbi_prior, density_estimator=neural_posterior)
        _ = inference.append_simulations(theta, data_nonequispaced)

        Returns:
            Network output (batch_size, out_channels * 2 * modes).
        """
        batch_size = x.shape[0]

        # Check dimension of input data and reshape it
        if x.ndim == 3:
            x = x.permute(0, 2, 1)  # (batch, n_points, in_channels)
            point_positions = None

        elif x.ndim == 4:
            point_positions = x[:, 1, 0, :]
            x = x[:, 0, :, :].permute(0, 2, 1)

        else:
            raise ValueError(
                'Input tensor should be 3D (batch_size, channels, n_points) '
                'or 4D (batch_size, 2, channels, n_points). ',
                f'The tensor that was passed has shape {x.shape}.',
            )

        n_points = x.shape[1]

        assert self.modes <= n_points // 2 + 1, (
            "Modes should be at most floor(n_points/2) + 1"
        )

        x = self.fc0(x)  # (batch_size, n_points, in_channels)

        # Initialize Fourier transform for arbitrarily spaced points
        fourier_transform = VFT(batch_size, n_points, self.modes, point_positions)

        # Send the data through Fourier layers, output in original space
        for conv, w in zip(self.conv_layers, self.w_layers, strict=False):
            x1 = conv(x, fourier_transform)
            x2 = w(x.permute(0, 2, 1))
            x = x1 + x2.permute(0, 2, 1)
            x = F.gelu(x)

        # Send data through last convolutional layer which returns data in Fourier space
        x_spec = self.conv_last.last_layer(
            x, fourier_transform
        )  # (batch, 2*modes, out_channels)

        # Reduce the number of channels with last layer
        x_spec = self.fc_last(x_spec)  # (batch, 2*modes, out_channels)

        return x_spec.reshape(batch_size, -1)
