from math import sqrt
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch._higher_order_ops.associative_scan import associative_scan
from typing import Tuple
class LRUEmbedding(nn.Module):
    """Embedding network backed by a stack of Linear Recurrent Unit (LRU) layers.

    See also:
        https://arxiv.org/pdf/2303.06349
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int = 20,
        hidden_dim: int = 20,
        num_layers: int = 2,
        r_min=0.0,
        r_max=1.0,
        max_phase=2 * np.pi,
        bidirectional: bool = False,
        dropout: float = 0.0,
        norm: bool = False,
        aggregate_func: [str, callable] = "last_ts",
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
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                r_min=r_min,
                r_max=r_max,
                max_phase=max_phase,
                bidirectional=bidirectional,
                dropout=dropout,
                norm = norm
            )
            for _ in range(num_layers)
        ]
        
        if aggregate_func == "last_ts":
            self.aggregation = lambda x: x[:,-1,:]
        elif aggregate_func == "mean":
            self.aggregation = lambda x: x.mean(dim=1)
        elif aggregate_func == "sum":
            self.aggregation = lambda x: x.mean(dim=1)
        elif isinstance(aggregate_func, str):
            raise ValueError(f"aggretate_func {aggregate_func} not implemented")
        else:
            self.aggregation = aggregate_func

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
        x = self.layers(x)  # (batch_size, len_sequence, output_dim)

        # Pooling
        x = self.aggregation(x)

        # output embedding
        x = self.output(x)
        return x


class LRUBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 20,
        r_min=0.0,
        r_max=1.0,
        max_phase=2 * np.pi,
        bidirectional: bool = False,
        dropout: float = 0.0,
        norm: bool = False,
    ):
        super().__init__()
        # https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py
        self.lru = LRU(hidden_dim, state_dim, r_min, r_max, max_phase,bidirectional)
        self.out1 = nn.Linear(hidden_dim, hidden_dim)
        self.out2 = nn.Linear(hidden_dim, hidden_dim)
        self.GELU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Batch, len_sequence, hidden_dim
        # normalization
        y = self.norm(x)

        # run LRU
        y = self.lru(y)

        # state mixing
        y = self.GELU(y)
        y = self.dropout(y)
        y = self.out1(y) * self.sigmoid(self.out2(y))
        y = self.dropout(y)

        # skip connection
        y = y + x
        return y


class LRU(nn.Module):
    """A single Linear Recurrent Unit (LRU).

    This implementation, just like all other, makes the simplification
    that `output_dim = input_dim`. The benefor is that you can stack
    multiple LRU instances without linear layers inbetween, and the
    skip connections `D` become element-wise, i.e., `D` is a vector.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 2 * torch.pi,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        # between r_min and r_max, with phase in [0, max_phase].
        self.state_dim = state_dim



        if bidirectional:
            state_dim = state_dim * 2


        u1 = torch.rand(size=(state_dim,))
        u2 = torch.rand(size=(state_dim,))
        self.log_nu = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        )
        self.log_theta = nn.Parameter(torch.log(max_phase * u2))

        # Glorot initialized Input/Output projection matrices.
        B_re = torch.randn(size=(state_dim, input_dim)) / sqrt(2 * input_dim)
        B_im = torch.randn(size=(state_dim, input_dim)) / sqrt(2 * input_dim)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn(size=(input_dim, state_dim)) / sqrt(2 * input_dim)
        C_im = torch.randn(size=(input_dim, state_dim)) / sqrt(2 * input_dim)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        self.D = nn.Parameter(
            torch.randn(size=(input_dim,))
        )  # simplified skip connection

        gamma_log = torch.log(torch.sqrt(1 - torch.abs(self.lambda_complex) ** 2))
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
        return torch.exp(self.log_gamma)

    def forward(
        self, input: Tensor, state: Optional[Tensor] = None, mode: str = "loop"
    ) -> Tensor:
        # Initialize the hidden state if not given.
        if self.bidirectional:
            expected_state_shape = (input.size(0), self.state_dim*2)
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
                f"Invalid state shape {state.shape}, expected {expected_state_shape}"
            )

        match mode:
            case "scan":
                y = self._forward_scan(input, state)
            case "loop":
                y = self._forward_loop(input, state)
        return y

    def _forward_loop(self, input: Tensor, state: Tensor) -> Tensor:
        # Input size: (B, L, H)

        B_norm = self.B * self.gamma.unsqueeze(dim=-1)

        states = []
        state_t = state
        uB = input.to(dtype=B_norm.dtype) @ B_norm.T

        if self.bidirectional:
            uB = torch.cat([uB[:,:,:self.state_dim],uB[:,::-1,self.state_dim:]],dim=-1) 

        for u_step in uB.split(1, dim=1):  # dim=1 is the time dimension
            state_t = self.lambda_complex * state_t + u_step.squeeze(1)
            states.append(state_t)
        states = torch.stack(states, dim=1)

        output = (states @ self.C.mT).real + input * self.D

        return output
    
    def _forward_scan(self, input: Tensor, state: Tensor) -> Tensor:
        # Input size: (B, L, H)

        B_norm = self.B * self.gamma.unsqueeze(dim=-1)

        # For details on parallel scan, check discussion in Smith et al (2022).
    
        Bu_elements = input.to(self.B.dtype)@B_norm.T
       
        if state is not None:
            Bu_elements[:, 0, :self.state_dim] = Bu_elements[:, 0, :self.state_dim] + \
                (self.lambda_complex[:self.state_dim].view(1,-1) * state[:,:self.state_dim])
        Lambda_elements = self.lambda_complex.view(1,1,-1).expand(Bu_elements.shape)
        
        elements = (Lambda_elements[:,:,:self.state_dim], Bu_elements[:,:,:self.state_dim])
        _, states = associative_scan(binary_operator_diag, elements,dim=1, combine_mode='generic') # all x_k

        if self.bidirectional:
            if state is not None:
                Bu_elements[:, -1, self.state_dim:] = Bu_elements[:, -1, self.state_dim:] + \
                (self.lambda_complex[self.state_dim:].view(1,-1) * state[:,self.state_dim:])
            elements = (Lambda_elements[:,:,self.state_dim:], torch.flip(Bu_elements[:,:,self.state_dim:],dims=[1]))
            _, inner_states2 = associative_scan(
                binary_operator_diag, elements, dim=1, combine_mode = 'generic', reverse=False
            )
            states = torch.cat([states, inner_states2], dim=-1)
        output = (states @ self.C.mT).real + input * self.D
        
        return output
    

#@torch.jit.script
def binary_operator_diag(element_i: Tuple[torch.Tensor, torch.Tensor], element_j: Tuple[torch.Tensor, torch.Tensor]):
    """Binary operator for parallel scan of linear recurrence.
    Args:
        element_i: tuple containing a_i and bu_i at position i
        element_j: tuple containing a_j and bu_j at position j
    Returns:
        new element ( _out, bu_out )
    """
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    # a_j * a_i, torch.addcmul(bu_j, a_j, bu_i)
    return a_j * a_i, a_j * bu_i + bu_j


    # def forward_scan(self, input_sequence):
    #     """Forward pass of the LRU layer. Output y and input_sequence are of shape (batchsize, length, hidden_dim)."""

    #     Lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))

    #     # Input and output projections
    #     B_norm = (self.B_re + 1j * self.B_im) * torch.unsqueeze(
    #         torch.exp(self.gamma_log), dim=-1
    #     )
    #     C = self.C_re + 1j * self.C_im

    #     # Running the LRU + output projection
    #     # For details on parallel scan, check discussion in Smith et al (2022).
    #     # Lambda_elements = torch.tile(Lambda[None, ...], input_sequence.shape[0], dim=0)
    #     Lambda_elements = Lambda.tile(
    #         input_sequence.shape[1], 1
    #     )  # TODO: check if we need repeat
    #     Bu_elements = input_sequence @ B_norm.T
    #     elements = (Lambda_elements, Bu_elements)
    #     _, inner_states = associative_scan(binary_operator_diag, elements)  # all x_k

    #     if self.bidirectional:
    #         _, inner_states2 = associative_scan(
    #             binary_operator_diag, elements, reverse=True
    #         )
    #         inner_states = torch.cat([inner_states, inner_states2], dim=-1)

    #     y = (inner_states @ C.T).real + input_sequence @ self.D.T
    #     return y


# https://github.com/i404788/s5-pytorch/blob/master/s5/s5_model.py
# https://github.com/state-spaces/mamba/tree/main
# https://github.com/pytorch/pytorch/issues/95408


# def associative_scan(operator: Callable, elems, axis: int = 0, reverse: bool = False):
#     # if not callable(operator):
#     #     raise TypeError("lax.associative_scan: fn argument should be callable.")
#     elems_flat, tree = tree_flatten(elems)

#     if reverse:
#         elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

#     assert axis >= 0 or axis < elems_flat[0].ndim, (
#         "Axis should be within bounds of input"
#     )
#     num_elems = int(elems_flat[0].shape[axis])
#     if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
#         raise ValueError(
#             'Array inputs to associative_scan must have the same '
#             'first dimension. (saw: {})'.format([elem.shape for elem in elems_flat])
#         )

#     scans = _scan(tree, operator, elems_flat, axis)

#     if reverse:
#         scans = [torch.flip(scanned, [axis]) for scanned in scans]

#     return tree_unflatten(scans, tree)


# def combine(tree, operator, a_flat, b_flat):
#     # Lower `fn` to operate on flattened sequences of elems.
#     a = tree_unflatten(a_flat, tree)
#     b = tree_unflatten(b_flat, tree)
#     c = operator(a, b)
#     c_flat, _ = tree_flatten(c)
#     return c_flat


# def _scan(tree, operator, elems, axis: int):
#     """Perform scan on `elems`."""
#     num_elems = elems[0].shape[axis]

#     if num_elems < 2:
#         return elems

#     # Combine adjacent pairs of elements.
#     reduced_elems = combine(
#         tree,
#         operator,
#         [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
#         [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems],
#     )

#     # Recursively compute scan for partially reduced tensors.
#     odd_elems = _scan(tree, operator, reduced_elems, axis)

#     if num_elems % 2 == 0:
#         even_elems = combine(
#             tree,
#             operator,
#             [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
#             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
#         )
#     else:
#         even_elems = combine(
#             tree,
#             operator,
#             odd_elems,
#             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems],
#         )

#     # The first element of a scan is the same as the first element
#     # of the original `elems`.
#     even_elems = [
#         torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
#         if result.shape.numel() > 0 and elem.shape[axis] > 0
#         else result
#         if result.shape.numel() > 0
#         else torch.ops.aten.slice(
#             elem, axis, 0, 1
#         )  # Jax allows/ignores concat with 0-dim, Pytorch does not
#         for (elem, result) in zip(elems, even_elems, strict=False)
#     ]

#     return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))


# @torch.jit.script
# def binary_operator_diag(
#     q_i: tuple[Tensor, Tensor], q_j: tuple[Tensor, Tensor]
# ) -> tuple[Tensor, Tensor]:
#     """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.

#     See also:
#         https://github.com/i404788/s5-pytorch/blob/master/s5/s5_model.py

#     Args:
#         q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
#         q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)

#     Returns:
#         New element (A_out, Bu_out) ``A_j * A_i, A_j * Bu_i + Bu_j``
#     """
#     A_i, Bu_i = q_i
#     A_j, Bu_j = q_j
#     return A_j * A_i, torch.addcmul(Bu_j, A_j, Bu_i)
