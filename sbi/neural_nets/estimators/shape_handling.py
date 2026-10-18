# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import torch
from torch import Tensor


def reshape_to_sample_batch_event(
    theta_or_x: Tensor, event_shape: torch.Size, leading_is_sample: bool = False
) -> Tensor:
    """Return theta or x s.t. its shape is `(sample_dim, batch_dim, *event_shape)`.

    This follows the conventions used in pytorch distributions:
    https://bochang.me/blog/posts/pytorch-distributions/

    Args:
        theta_or_x: The tensor to be reshaped. Can have any of the following shapes:
            - (event)
            - (batch, event)
            - (sample, event)
            - (sample, batch, event)
        event_shape: The shape of a single datapoint (without batch dimension or sample
            dimension).
        leading_is_sample: Used only if `theta_or_x` has exactly one dimension beyond
            the `event` dims. Defines whether the leading dimension is interpreted as
            batch dimension or as sample dimension.

    Returns:
        A tensor of shape `(sample, batch, event)`.
    """
    # `2` for image data, `3` for video data, ...
    event_shape_dim = len(event_shape)

    trailing_theta_or_x_shape = theta_or_x.shape[-event_shape_dim:]
    leading_theta_or_x_shape = theta_or_x.shape[:-event_shape_dim]
    if trailing_theta_or_x_shape != event_shape:
        raise RuntimeError(
            f"The shape of the input does not match the expected shape. "
            f"Expected trailing dimensions {tuple(event_shape)}, but got "
            f"{tuple(trailing_theta_or_x_shape)}. This can happen when `x_o` has "
            f"more (or fewer) entries than the `x` used during training."
        )

    if len(leading_theta_or_x_shape) == 0:
        # A single datapoint is passed. Add batch and sample dim artificially.
        return theta_or_x.unsqueeze(0).unsqueeze(0)
    elif len(leading_theta_or_x_shape) == 1:
        # Either a batch dimension or an sample dimension was passed.
        return theta_or_x.unsqueeze(1) if leading_is_sample else theta_or_x.unsqueeze(0)
    elif len(leading_theta_or_x_shape) == 2:
        # Batch dimension and sample dimension were passed.
        return theta_or_x if leading_is_sample else theta_or_x.transpose(1, 0)
    else:
        raise ValueError(
            f"`len(leading_theta_or_x_shape) = {leading_theta_or_x_shape} > 2`. "
            f"It is unclear how the additional entries should be interpreted"
        )


def reshape_to_batch_event(theta_or_x: Tensor, event_shape: torch.Size) -> Tensor:
    """Return theta or x s.t. its shape is `(batch_dim, *event_shape)`.

    Args:
        theta_or_x: The tensor to be reshaped. Can have any of the following shapes:
            - (event)
            - (batch, event)
        event_shape: The shape of a single datapoint (without batch dimension or sample
            dimension).

    Returns:
        A tensor of shape `(batch, event)`.
    """
    # Check for degenerate case, it is used by the Simformer
    # when the user requires a full latent tensor, i.e., the user
    # is effectively using the Simformer as a "data generator"
    if event_shape == torch.Size([0]):
        if theta_or_x.numel() == 0:
            return theta_or_x
        else:
            raise ValueError(
                "event_shape is torch.Size([0]) but theta_or_x is not an empty tensor."
            )

    # `2` for image data, `3` for video data, ...
    event_shape_dim = len(event_shape)

    trailing_theta_or_x_shape = theta_or_x.shape[-event_shape_dim:]
    leading_theta_or_x_shape = theta_or_x.shape[:-event_shape_dim]
    if trailing_theta_or_x_shape != event_shape:
        raise RuntimeError(
            f"The shape of the input does not match the expected shape. "
            f"Expected trailing dimensions {tuple(event_shape)}, but got "
            f"{tuple(trailing_theta_or_x_shape)}. This can happen when `x_o` has "
            f"more (or fewer) entries than the `x` used during training."
        )

    if len(leading_theta_or_x_shape) == 0:
        # A single datapoint is passed. Add batch artificially.
        return theta_or_x.unsqueeze(0)
    elif len(leading_theta_or_x_shape) == 1:
        # A batch dimension was passed.
        return theta_or_x
    else:
        raise ValueError(
            f"`len(leading_theta_or_x_shape) = {leading_theta_or_x_shape} > 1`. "
            f"It is unclear how the additional entries should be interpreted"
        )
