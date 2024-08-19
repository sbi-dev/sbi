# import all needed modules
from typing import Optional
from warnings import warn

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
    if str(next(net.parameters()).device) != str(device):
        warn(
            message or f"Network is not on the correct device. Moving it to {device}.",
            stacklevel=2,
        )
        return net.to(device)
    else:
        return net
