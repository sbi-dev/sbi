# import all needed modules
from typing import Optional
from warnings import warn

import nflows.nn.nde.made as made
import numpy as np
import torch
import torch.nn.functional as F
from pyknos.nflows import distributions as distributions_
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


"""
Temporary Patches to fix nflows MADE bug. Remove once upstream bug is fixed.
"""


class MADEWrapper(made.MADE):
    """Implementation of MADE.

    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        output_multiplier=1,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__(
            features + 1,
            hidden_features,
            context_features,
            num_blocks,
            output_multiplier,
            use_residual_blocks,
            random_mask,
            activation,
            dropout_probability,
            use_batch_norm,
        )
        self.output_multiplier = output_multiplier

    def forward(self, inputs, context=None):
        # add dummy input to ensure all dims conditioned on context.
        dummy_input = torch.zeros((inputs.shape[:-1] + (1,)))
        concat_input = torch.cat((dummy_input, inputs), dim=-1)
        outputs = super().forward(concat_input, context)
        # the final layer of MADE produces self.output_multiplier outputs for each
        # input dimension, in order. We only want the outputs corresponding to the
        # real inputs, so we discard the first self.output_multiplier outputs.
        return outputs[..., self.output_multiplier :]


"""
Temporary Patches to fix nflows MADE bug. Remove once upstream bug is fixed.
"""


class MADEMoGWrapper(distributions_.MADEMoG):
    def __init__(
        self,
        features,
        hidden_features,
        context_features,
        num_blocks=2,
        num_mixture_components=1,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        custom_initialization=False,
    ):
        super().__init__(
            features + 1,
            hidden_features,
            context_features,
            num_blocks,
            num_mixture_components,
            use_residual_blocks,
            random_mask,
            activation,
            dropout_probability,
            use_batch_norm,
            custom_initialization,
        )

    def _log_prob(self, inputs, context=None):
        dummy_input = torch.zeros((inputs.shape[:-1] + (1,)))
        concat_inputs = torch.cat((dummy_input, inputs), dim=-1)

        outputs = self._made.forward(concat_inputs, context=context)
        outputs = outputs.reshape(
            *concat_inputs.shape, self._made.num_mixture_components, 3
        )

        logits, means, unconstrained_stds = (
            outputs[..., 0],
            outputs[..., 1],
            outputs[..., 2],
        )
        # remove first dimension of means, unconstrained_stds
        logits = logits[..., 1:, :]
        means = means[..., 1:, :]
        unconstrained_stds = unconstrained_stds[..., 1:, :]

        log_mixture_coefficients = torch.log_softmax(logits, dim=-1)
        stds = F.softplus(unconstrained_stds) + self._made.epsilon

        log_prob = torch.sum(
            torch.logsumexp(
                log_mixture_coefficients
                - 0.5
                * (
                    np.log(2 * np.pi)
                    + 2 * torch.log(stds)
                    + ((inputs[..., None] - means) / stds) ** 2
                ),
                dim=-1,
            ),
            dim=-1,
        )
        return log_prob

    def _sample(self, num_samples, context=None):
        samples = self._made.sample(num_samples, context=context)
        return samples[..., 1:]
