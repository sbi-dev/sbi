import warnings
from math import sqrt
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch._higher_order_ops.associative_scan import associative_scan  # type: ignore


class LRUEmbedding(nn.Module):
    """Embedding network backed by a stack of Linear Recurrent Unit (LRU) blocks.
    This type of embedding is intended to be used with sequential data, e.g.
    time series. So far, the implementation assumes that the sequence length is
    constant for all observations.

    Each LRU block is comprised of an `LRU` instance surrounded by input normalization,
    dropout, nonlinearities, state mixing, and a skip connection.

    See also:
        Orvieto et al. 2023, https://arxiv.org/abs/2303.06349
        Smith, Warrington & Linderman, 2023, https://arxiv.org/abs/2208.04933
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int = 20,
        hidden_dim: int = 40,
        num_blocks: int = 2,
        r_min: float = 0.0,
        r_max: float = 1.0,
        phase_max: float = 2 * np.pi,
        bidirectional: bool = True,
        mode: str = "loop",
        dropout: float = 0.0,
        apply_input_normalization: bool = False,
        aggregate_fcn: Union[str, Callable] = "mean",
    ):
        """
        Args:
            input_dim: Dimensionality of input, i.e., the observation, that is passed
                to the embedding net.
            output_dim: Dimensionality of the output, i.e, the resulting features of
                this embedding model.
            state_dim: Dimensionality of the LRU layers hidden state.
            hidden_dim: Number of hidden units in each layer of the embedding network.
            num_blocks: Number of LRU blocks in the embedding network. Must be >= 1.
            r_min: Minimum distance of the state-dynamics matrix' eigenvalues from
                the origin in the complex plane.
            r_max: Maximum distance of the state-dynamics matrix' eigenvalues from
                the origin in the complex plane. Must be `< 1` to yield a stable system.
            phase_max: Maximum value for the LRUs' phase initialization. When working
                with longer sequences, a smaller value is advised, e.g. `< pi/10`.
            bidirectional: Whether the LRUs should be run bi-directionally. This is
                expected to produce richer features at the cost of doubling the
                (internal) state dimension.
            mode: Whether to run the LRU's forward passes in a for-loop `mode="loop"`
                or using an associative scan `mode="scan"`. The former one is the naive
                implementation while the latter one relies on very recent features of
                PyTorch. Note that at the point of writing this, PyTorch's associative
                scan function does no support a backward pass.
            dropout: Dropout rate applied to the hidden states of the LRU blocks.
                There are two dropout layers in each LRU block, one after the LRU and
                one after the state mixing.
            apply_input_normalization: Whether to apply input normalization to the
                inputs of each LRU Block, see `LRUBlock.forward()`.
            aggregate_fcn: Function to aggregate the sequence of hidden states. Can be
                `"last_step"` to take the last hidden state, `"mean"` to take the mean
        """
        super().__init__()

        # The first layer is defined by the observations' input dimension.
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1.")
        lru_blocks = [
            LRUBlock(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                r_min=r_min,
                r_max=r_max,
                phase_max=phase_max,
                bidirectional=bidirectional,
                mode=mode,
                dropout=dropout,
                apply_input_normalization=apply_input_normalization,
            )
            for _ in range(num_blocks)
        ]
        self.lru_blocks = nn.Sequential(*lru_blocks)

        if aggregate_fcn == "last_step":
            self.aggregation = lambda x: x[:, -1, :]
        elif aggregate_fcn == "mean":
            self.aggregation = lambda x: x.mean(dim=1)
        elif isinstance(aggregate_fcn, str):
            raise ValueError(f"aggretate_func {aggregate_fcn} not implemented")
        else:
            self.aggregation = aggregate_fcn

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: Tensor) -> Tensor:
        """Embed a batch of 2-dim observations, e.g. a multi-dimensional
        time series.

        Args:
            input: Sequential data, i.e., observations, of shape (batch_size,
                len_sequence, input_dim)

        Returns:
            Network output, i.e., features, of shape (batch_size, output_dim).
        """
        output = self.input_layer(input)  # (batch_size, len_sequence, hidden_dim)
        output = self.lru_blocks(output)  # (batch_size, len_sequence, hidden_dim)
        output = self.aggregation(output)  # (batch_size, hidden_dim)
        output = self.output_layer(output)  # (batch_size, output_dim)
        return output


class LRUBlock(nn.Module):
    """Stack of layers surrounding a `LRU`.

    Note:
        The `_LRUBlock` class is intended to be used with a preceding and succeeding
        linear layers which take care of converting the dimensions such that all LRU
        blocks can be stacked without intermediate linear layers.

    See also:
        The `SequenceLayer` class in
        https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        r_min: float,
        r_max: float,
        phase_max: float,
        bidirectional: bool,
        mode: str,
        dropout: float,
        apply_input_normalization: bool,
    ):
        super().__init__()

        if apply_input_normalization:
            self.input_norm = nn.LayerNorm(hidden_dim)
        else:
            self.input_norm = nn.Identity()
        self.lru = LRU(
            input_dim=hidden_dim,  # here, output_dim = input_dim
            state_dim=state_dim,
            r_min=r_min,
            r_max=r_max,
            phase_max=phase_max,
            bidirectional=bidirectional,
            mode=mode,
        )
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = nn.GELU()  # could also use different one here
        self.state_mixing_gate = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through an LRU block, which also contains input
        normalization, dropout, nonlinearities, state mixing, and a skip
        connection.

        Args:
            inputs: Sequential data of shape (batch_size, sequence_length,
                hidden_dim)

        Returns:
            Transformed sequential data of shape (batch_size, sequence_length,
                hidden_dim)
        """
        y = self.input_norm(input)
        y = self.lru(y)
        y = self.nonlinearity(y)
        y = self.dropout(y)
        # apply state mixing gate as in Smith, Warrington & Linderman, 2023
        y = y * self.sigmoid(self.state_mixing_gate(y))
        y = self.dropout(y)
        y = y + input  # skip connection
        return y


class LRU(nn.Module):
    """A single Linear Recurrent Unit (LRU).

    This implementation, just like the others, makes the simplification that
    `output_dim = input_dim`. The benefit is that you can stack multiple LRU
    instances without linear layers in between, and the skip connections `D`
    become element-wise, i.e., `D` is a vector.

    For the bidirectional mode, we flit the inputs and the resulting states/outputs.
    This is in line with how the authors of
    https://arxiv.org/pdf/2202.09729
    are doing it. This is not the same as scipy's `filtfilt` though, as we are not
    flipping the outputs and running the same process again, i.e., here it is done
    sequentially and not in parallel.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        r_min: float,
        r_max: float,
        phase_max: float,
        bidirectional: bool,
        mode: str,
    ):
        super().__init__()

        # Check and store the inputs.
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not isinstance(state_dim, int) or input_dim <= 0:
            raise ValueError("state_dim must be a positive integer.")
        if not (0 <= r_min < r_max <= 1):
            raise ValueError(
                f"Invalid {r_min=} and/or {r_max: float = }. They must suffice "
                "0 <= r_min < r_max <= 1."
            )
        if not (0 <= phase_max <= 2 * torch.pi):
            raise ValueError(
                f"Invalid {phase_max: float = }. I must suffice 0 <= phase_max <= 2 pi."
            )
        if mode not in ("loop", "scan"):
            raise ValueError(f"Invalid {mode=}. Must be 'loop' or 'scan'.")
        if mode == "scan":
            warnings.warn(
                "The scan mode currently does not support a backward pass, "
                "the code will be updated once this is pytorch native",
                stacklevel=2,
            )

        self.state_dim = state_dim
        self.bidirectional = bidirectional
        self.mode = mode
        if bidirectional:
            state_dim = state_dim * 2

        # Sample two independent random variables to initialize lambda between
        # two rings (r_min, r_max) on the complex plane.
        # See section 3.2 in Orvieto et al., 2023
        r_unit_sample = torch.rand(size=(state_dim,))
        r_init = r_unit_sample * (r_max**2 - r_min**2) + r_min**2
        self.log_nu = nn.Parameter(torch.log(-0.5 * torch.log(r_init)))
        theta_init = phase_max * torch.rand(size=(state_dim,))
        self.log_theta = nn.Parameter(torch.log(theta_init))

        # Create the Glorot-initialized projection matrices.
        B_re = torch.randn(size=(state_dim, input_dim)) / sqrt(2 * input_dim)
        B_im = torch.randn(size=(state_dim, input_dim)) / sqrt(2 * input_dim)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn(size=(input_dim, state_dim)) / sqrt(2 * input_dim)
        C_im = torch.randn(size=(input_dim, state_dim)) / sqrt(2 * input_dim)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        self.D = nn.Parameter(torch.randn(size=(input_dim,)))

        # Initialize the normalization factor.
        gamma_log = torch.log(torch.sqrt(1 - self.lambda_abs**2))
        self.log_gamma = nn.Parameter(gamma_log)

    @property
    def lambda_abs(self) -> Tensor:
        """Compute the absolute value of the elements of the diagonal dynamics matrix A,
        parameterized by nu."""
        return torch.exp(-torch.exp(self.log_nu))

    @property
    def lambda_complex(self) -> Tensor:
        """Compute the elements of the diagonal dynamics matrix A, parameterized by nu
        and theta."""
        return torch.exp(-torch.exp(self.log_nu) + 1j * torch.exp(self.log_theta))

    @property
    def gamma(self) -> Tensor:
        """Get the input normalization."""
        return torch.exp(self.log_gamma)

    def forward(
        self, input: Tensor, state: Optional[Tensor] = None, mode: Optional[str] = None
    ) -> Tensor:
        """Run the forward pass.

        Args:
            input: Sequential data of shape (batch_size, sequence_length,
                input_dim)
            state: Initial hidden state of the LRU, if `None`, it will be initialized
                to a complex zero tensor of the expected shape.
            mode: Whether to run the forward pass in a for-loop `"loop"` or using an
                associative scan `"scan"`. The former one is the naive implementation
                while the latter one relies on very recent features of PyTorch. If
                set to `None` (default) the forward pass will use the mode given at
                initialization.

        Return:
            Transformed sequential data of shape (batch_size, sequence_length,
                input_dim)
        """
        # Initialize the hidden state if not given.
        if self.bidirectional:
            expected_state_shape = (input.size(0), self.state_dim * 2)
        else:
            expected_state_shape = (input.size(0), self.state_dim)

        if state is None:
            state = torch.complex(
                torch.zeros(expected_state_shape, device=input.device),
                torch.zeros(expected_state_shape, device=input.device),
            )
        else:
            state = state.to(device=input.device)
            assert state.shape == expected_state_shape, (
                f"Invalid state shape {state.shape}, "  # fmt: skip
                "expected {expected_state_shape}"  # fmt: skip
            )

        # Detemine which mode to run the forward method with.
        mode = mode or self.mode
        match mode:
            case "scan":
                output = self._forward_scan(input, state)
            case "loop":
                output = self._forward_loop(input, state)
        return output

    def _forward_loop(self, input: Tensor, state: Tensor) -> Tensor:
        """Straightforward implementation of the forward pass.

        Args:
            input: Sequential data of shape (batch_size, sequence_length,
                input_dim)
            state: Initial hidden complex state of the LRU of shape
                (batch_size, state_dim)
        Return:
            Transformed sequential data of shape (batch_size, sequence_length,
                input_dim)
        """
        # Normalize the input-facing matrix.
        B_norm = self.B * self.gamma.unsqueeze(dim=-1)

        # Precompute the influence of the inputs one the states. This is faster
        # than many small matrix-vector multiplications within the loop.
        u = input.to(dtype=B_norm.dtype)
        u_times_B_norm = u @ B_norm.T  # complex-values

        if self.bidirectional:
            # If the network is run bidirectionally, we simply repeat the input
            # and flip the new part. This way, we only need to run the for-loop
            # once.
            u_times_B_norm = torch.cat(
                [
                    u_times_B_norm[:, :, : self.state_dim],
                    torch.flip(u_times_B_norm[:, :, self.state_dim :], dims=[1]),
                ],
                dim=-1,
            )

        # Compute the evolution of the internal state over time given the inputs.
        x = []
        x_t = state.clone()
        for u_t in u_times_B_norm.split(1, dim=1):  # dim=1 is the time dimension
            x_t = self.lambda_complex * x_t + u_t.squeeze(1)
            x.append(x_t)
        x = torch.stack(x, dim=1)

        # Reverse the temporal oder of the 2nd block, i.e., 2nd direction.
        if self.bidirectional:
            x = torch.cat(
                [
                    x[:, :, : self.state_dim],
                    torch.flip(x[:, :, self.state_dim :], dims=[1]),
                ],
                dim=-1,
            )

        # Compute the output (for both directions at the same time).
        y = (x @ self.C.mT).real + input * self.D

        return y

    def _forward_scan(self, input: Tensor, state: Tensor) -> Tensor:
        """For details on parallel scan, check discussion in Smith et al (2022).

        Args:
            input: Sequential data of shape (batch_size, sequence_length,
                input_dim)
            state: Initial hidden complex state of the LRU of shape
                (batch_size , state_dim).

        Return:
            Transformed sequential data of shape (batch_size, sequence_length,
                input_dim)
        """
        # Normalize the input-facing matrix.
        B_norm = self.B * self.gamma.unsqueeze(dim=-1)

        # Precompute the influence of the inputs one the states.
        u = input.to(dtype=B_norm.dtype)
        u_times_B_norm = u @ B_norm.T  # complex-values

        u_times_B_norm[:, 0, : self.state_dim] = u_times_B_norm[
            :, 0, : self.state_dim
        ] + (
            self.lambda_complex[: self.state_dim].view(1, -1)
            * state[:, : self.state_dim]
        )
        lambdas = self.lambda_complex.view(1, 1, -1).expand(u_times_B_norm.shape)

        # Create the elements container, and run the scan.
        # The result is the complete sequence x.
        elements = (
            lambdas[:, :, : self.state_dim],
            u_times_B_norm[:, :, : self.state_dim],
        )
        _, x = associative_scan(
            binary_operator_diag, elements, dim=1, combine_mode="generic"
        )

        if self.bidirectional:
            u_times_B_norm[:, -1, self.state_dim :] = u_times_B_norm[
                :, -1, self.state_dim :
            ] + (
                self.lambda_complex[self.state_dim :].view(1, -1)
                * state[:, self.state_dim :]
            )
            elements = (
                lambdas[:, :, self.state_dim :],
                u_times_B_norm[:, :, self.state_dim :],
            )
            _, x_bi = associative_scan(
                binary_operator_diag,
                elements,
                dim=1,
                combine_mode="generic",
                reverse=True,
            )
            x = torch.cat([x, x_bi], dim=-1)

        # Compute the output (for both directions at the same time).
        y = (x @ self.C.mT).real + input * self.D

        return y


def binary_operator_diag(
    element_i: Tuple[Tensor, Tensor],
    element_j: Tuple[Tensor, Tensor],
) -> Tuple[Tensor, Tensor]:
    """Binary operator for parallel scan of linear recurrence.

    Args:
        element_i: tuple containing a_i and bu_i at position i
        element_j: tuple containing a_j and bu_j at position j

    Returns:
        New element containing $(a_j * a_i, a_j * bu_i + bu_j)$.
    """
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j
