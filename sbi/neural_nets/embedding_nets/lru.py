from typing import Callable

import lru
import torch
from torch import Tensor, nn
import numpy as np


class LRUEmbedding(nn.Module):
    """Embedding network backed by a Linear Recurrent Unit (LRU).
    See also:
        https://arxiv.org/pdf/2303.06349
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 10,
        state_dim: int = 20,
        hidden_dim: int = 20,
        mlp_dim: int = 80,
        num_layers: int = 2,
        r_min=0.0,
        r_max=1.0,
        max_phase=2 * np.pi,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        """Fully-connected multi-layer neural network to be used as embedding network.

        Args:
            input_dim: Dimensionality of input that will be passed to the embedding net.
            output_dim: Dimensionality of the output.
            hidden_dim: Number of hidden units in each layer of the embedding network.
            num_layers: Number of layers of the embedding network. (Minimum of 2).
        """
        super().__init__()

        # The first layer is defined by the observations' input dimension.
        self.embedding = nn.Linear(input_dim, hidden_dim)

        layers = [
            LRUBlock(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                state_dim=state_dim,
                mlp_dim=mlp_dim,
                r_min=r_min,
                r_max=r_max,
                max_phase=max_phase,
                bidirectional=bidirectional,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]

        # Output layer.
        # TODO: only readout one timestep
        self.output = nn.Linear(hidden_dim, output_dim)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Embed a batch of 2-dim observations, e.g. a multi-dimensional
        time series.

        Args:
            x: Input tensor (batch_size, len_sequence, num_features)

        Returns:
            Network output (batch_size, output_dim).
        """
        x = self.embedding(x)
        x_scan = self.net(x)  # (batch_size, len_sequence, output_dim)

        # Aggregate the features of the time dimension.
        x_embed = self.output(x_scan)

        return x_embed


class LRUBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 10,
        state_dim: int = 20,
        mlp_dim: int = 20,
        r_min=0.0,
        r_max=1.0,
        max_phase=2 * np.pi,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lru = LRU(input_dim, state_dim, r_min, r_max, max_phase)
        self.hidden_projection = nn.Linear(state_dim, mlp_dim)
        self.hidden_nonlinearity = nn.GELU()
        self.output_projection = nn.Linear(mlp_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # TO DO: GLU mixing Layers?

        # run LRU
        x = self.lru(x)

        # state mixing
        y = self.hidden_nonlinearity(self.hidden_projection(x))
        y = self.dropout(y)
        y = self.output_projection(y) + x
        return y


class LRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int = 20,
        r_min=0.1,
        r_max=1.0,
        max_phase=2 * np.pi,
    ):
        super().__init__()

        # between r_min and r_max, with phase in [0, max_phase].
        u1 = torch.rand(size=(state_dim,))
        u2 = torch.rand(size=(state_dim,))
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        # Glorot initialized Input/Output projection matrices
        B_re = torch.randn(size=(state_dim, input_dim)) / np.sqrt(2 * input_dim)
        B_im = torch.randn(size=(state_dim, input_dim)) / np.sqrt(2 * input_dim)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn(size=(input_dim, state_dim)) / np.sqrt(state_dim)
        C_im = torch.randn(size=(input_dim, state_dim)) / np.sqrt(state_dim)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        self.D = nn.Parameter(torch.randn(size=(input_dim,)))

        # Normalization factor
        diag_lambda = torch.exp(
            -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)
        )
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.gamma_log = nn.Parameter(gamma_log)

    def forward(self, input_sequence):
        """Forward pass of the LRU layer. Output y and input_sequence are of shape (L, H)."""

        # Materializing the diagonal of Lambda and projections
        Lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * torch.unsqueeze(
            torch.exp(self.gamma_log), dim=-1
        )
        C = self.C_re + 1j * self.C_im

        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        Lambda_elements = torch.tile(Lambda[None, ...], input_sequence.shape[0], dim=0)
        Bu_elements = torch.vmap(lambda u: B_norm @ u)(input_sequence)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = parallel_scan(binary_operator_diag, elements)  # all x_k
        y = torch.vmap(lambda x: u)


def binary_operator_diag(element_i, element_j):
    # Binary operator for parallel scan of linear recurrence.
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j
