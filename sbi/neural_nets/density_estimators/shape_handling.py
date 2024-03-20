import torch
from torch import Tensor


def reshape_to_batch_iid_event(
    theta_or_x: Tensor, event_shape: torch.Size, batch_is_iid: bool
) -> Tensor:
    """Return theta or x such that its shape is `(batch_dim, iid_dim, event_shape)`.

    Args:
        theta_or_x: The tensor to be reshaped. Can have any of the following shapes:
            - (event)
            - (batch, event)
            - (iid, event)
            - (batch, iid, event)
        event_shape: The shape of a single datapoint (without batch dimension or iid
            dimension).
        batch_is_iid: Used only if `len(theta_or_x) == 2` and `theta_or_x.shape > 1`.
            Defines whether the leading dimension is interpreted as batch dimension or
            as iid dimension.

    Returns:
        A tensor of shape `(batch, iid, event)`.
    """
    # `2` for image data, `3` for video data, ...
    event_shape_dim = len(event_shape)

    trailing_theta_or_x_shape = theta_or_x.shape[-event_shape_dim:]
    leading_theta_or_x_shape = theta_or_x.shape[:-event_shape_dim]
    assert (
        trailing_theta_or_x_shape == event_shape
    ), "The trailing dimensions of `theta_or_x` do not match the `event_shape`."

    if len(leading_theta_or_x_shape) == 0:
        # A single datapoint is passed. Add batch and iid dim artificially.
        return theta_or_x.unsqueeze(0).unsqueeze(0)
    elif len(leading_theta_or_x_shape) == 1:
        # Either a batch dimension or an iid dimension was passed.
        return theta_or_x.unsqueeze(0) if batch_is_iid else theta_or_x.unsqueeze(1)
    elif len(leading_theta_or_x_shape) == 2:
        # Batch dimension and iid dimension were passed.
        return theta_or_x
    else:
        raise ValueError(
            f"`len(leading_theta_or_x_shape) = {leading_theta_or_x_shape} > 2`. "
            f"It is unclear how the additional entries should be interpreted"
        )
