# import all needed modules
from typing import List, Optional, Tuple
from warnings import warn

import torch
from torch import Tensor, nn


def get_numel(
    batch_input: Tensor,
    embedding_net: Optional[nn.Module] = None,
    warn_on_1d: bool = False,
) -> int:
    """
    Return number of elements from an embedded batch of inputs.

    Offers option to warn if the embedded input is one-dimensional.

    Args:
        batch_input: Batch of inputs.
        embedding_net: Optional embedding network.
        warn_on_1d: Whether to warn if the output space is one-dimensional.

    Returns:
        Number of elements after optional embedding.

    """
    if embedding_net is None:
        embedding_net = nn.Identity()

    # Make sure the embedding_net is on the same device as the data.
    numel = embedding_net.to(batch_input.device)(batch_input[:1]).numel()
    if numel == 1 and warn_on_1d:
        warn(
            "In one-dimensional output space, this flow is limited to Gaussians",
            stacklevel=2,
        )

    return numel


def check_net_device(
    net: nn.Module, device: str, message: Optional[str] = None
) -> nn.Module:
    """
    Check whether a net is on the desired device and move it there if not.

    Args:
        net: neural network.
        device: desired device.

    Returns:
        Neural network on the desired device.
    """

    if isinstance(net, nn.Identity):
        return net
    if str(next(net.parameters()).device) != device:
        warn(
            message or f"Network is not on the correct device. Moving it to {device}.",
            stacklevel=2,
        )
        return net.to(device)
    else:
        return net


def concatenate_input_and_condition(
    input: Tensor, condition: Tensor
) -> Tuple[Tensor, List[int]]:
    """Expands and repeats the input to match the shape of condition.

    input has shape (sample_dim, batch_dim, *event_shape)
    condition has shape (batch_dim, *event_shape)

    There are scenarios where we need to concatenate input and condition, e.g.,
    in MNLE, where the continuous net needs to be conditioned on both the
    discrete input and the condituours condition. The condition can
    high-dimensional as well, e.g., (batch_dim, 7, 7).
    """

    if input.dim() != 2:
        raise ValueError(
            """To concatenate inputs and condition, input and condition must
            both be of shape (batch, *event_shape)."""
        )
    batch_shape, *event_shape = input.shape
    condition_batch_shape, *condition_event_shape = condition.shape

    assert (
        condition_batch_shape == batch_shape
    ), "Batch shapes of condition and input must match."

    # if the discrete data has fewer event dimensions than condition data, we
    # need to expand it to match the condition data.
    num_dims_condition = len(condition.shape)
    num_dims_to_expand = len(condition_event_shape) - len(event_shape)
    if num_dims_to_expand > 0:
        input_expanded = input.unsqueeze(-1).expand(*input.shape, num_dims_to_expand)
    else:
        input_expanded = input

    # we also need to repeat the discrete data to match the condition
    # data.
    repeat_shape = [1] * len(condition.shape)
    repeat_shape[0] = 1
    for i in range(1, num_dims_condition - 1):
        repeat_shape[i] = condition.shape[i]

    input_repeated = input_expanded.repeat(*repeat_shape)
    # find dimensions where the input will be repeated
    dims_repeated = torch.where(torch.tensor(repeat_shape) > 1)[0].tolist()

    combined_condition = torch.cat((condition, input_repeated), dim=-1)

    return combined_condition, dims_repeated
