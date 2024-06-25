# import all needed modules
from typing import Optional, Tuple
from warnings import warn

from torch import Tensor, nn

from sbi.utils.user_input_checks import check_data_device


def get_numel(
    batch_x: Tensor,
    batch_y: Tensor,
    embedding_net_x: Optional[nn.Module] | None = None,
    embedding_net_y: Optional[nn.Module] | None = None,
    warn_on_1d: bool = False,
) -> Tuple[int, int]:
    """
    Get the number of elements in the input and output space.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        embedding_net_x: Optional embedding network for x.
        embedding_net_y: Optional embedding network for y.
        warn_on_1d: Whether to warn if the output space is one-dimensional.

    Returns:
        Tuple of the number of elements in the input and output space.

    """
    if embedding_net_x is None:
        embedding_net_x = nn.Identity()
    if embedding_net_y is None:
        embedding_net_y = nn.Identity()

    # Infer the output dimensionality of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    # Make sure the embedding_net is on the same device as the data.
    x_numel = embedding_net_x.to(batch_x.device)(batch_x[:1]).numel()
    y_numel = embedding_net_y.to(batch_y.device)(batch_y[:1]).numel()
    if x_numel == 1 and warn_on_1d:
        warn(
            "In one-dimensional output space, this flow is limited to Gaussians",
            stacklevel=2,
        )

    return x_numel, y_numel
