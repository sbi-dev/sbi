# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

# build function for flow matching neural networks
# like in classifier.py, we need to build a network that can z-score the inputs

# import ABC for abstract class
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import zuko
from torch import Tensor
from torch.nn import functional as F
from zuko.nn import MLP as ZukoMLP

from sbi.neural_nets.estimators.flowmatching_estimator import (
    FlowMatchingEstimator,
    VectorFieldNet,
)
from sbi.utils.nn_utils import get_numel
from sbi.utils.sbiutils import (
    standardizing_net,
    z_score_parser,
    z_standardization,
)
from sbi.utils.user_input_checks import check_data_device


def build_mlp_flowmatcher(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: Union[Sequence[int], int] = 64,
    num_layers: int = 5,
    num_freqs: int = 3,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> FlowMatchingEstimator:
    """Builds a flow matching neural network.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_layers: Number of num_layers in the vector field regressor.
        num_freqs: Number of frequencies in the time embeddings.
        embedding_net: Embedding network for batch_y.
        kwargs: Additional keyword arguments passed to the FlowMatchingEstimator.
    """
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # create a list of layers for the regression network; the vector field
    # regressor is a MLP consisting of num_layers of layers with
    # hidden_features neurons each
    if isinstance(hidden_features, int):
        hidden_features = [hidden_features] * num_layers

    # Wrap zuko MLP as VectorFieldNet to ensure uniform forward signature
    vectorfield_net = VectorFieldMLP(
        net=ZukoMLP(
            in_features=x_numel + y_numel + 2 * num_freqs,
            out_features=x_numel,
            hidden_features=hidden_features,
            activation=nn.ELU,
        )
    )

    # input data is only z-scored, not embedded.
    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        t_mean, t_std = z_standardization(batch_x, structured_x)
        z_score_transform = torch.distributions.AffineTransform(
            -t_mean / t_std, 1 / t_std
        )
    else:
        z_score_transform = zuko.transforms.IdentityTransform()

    # pre-pend the z-scoring layer to the embedding net.
    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # create the flow matching estimator, will take care of time embeddings.
    flow_matching_estimator = FlowMatchingEstimator(
        net=vectorfield_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
        zscore_transform_input=z_score_transform,
        embedding_net=embedding_net,
        num_freqs=num_freqs,
        **kwargs,  # e.g., noise_scale
    )

    return flow_matching_estimator


def build_resnet_flowmatcher(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 64,
    num_blocks: int = 5,
    num_freqs: int = 3,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> FlowMatchingEstimator:
    """Builds a flow matching neural network.

    Note: for FMPE, batch_x is theta and batch_y is x, and the embedding_net refers to
    x. Theta is not embedded, only z-scored.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_units: Number of hidden units in the ResNet blocks.
        num_blocks: Number of ResNet blocks in the network.
        num_freqs: Number of frequencies in the time embeddings.
        embedding_net: Embedding network for batch_y.
        kwargs: Additional keyword arguments passed to the FlowMatchingEstimator.
    """
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x)  # theta
    y_numel = get_numel(batch_y, embedding_net=embedding_net)  # x

    # create the ResNet with GLU conditioning network
    # Note, here we condition on theta and time (and not x) using GLUs
    vectorfield_net = ResNetWithGLUConditioning(
        input_dim=y_numel,  # the resnet takes the embedded x as input
        hidden_units=hidden_features,
        num_blocks=num_blocks,
        # we condition on theta and embedded time using GLUs
        condition_dim=(x_numel + 2 * num_freqs),
        output_dim=x_numel,
    )

    # input data is only z-scored, not embedded.
    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        t_mean, t_std = z_standardization(batch_x, structured_x)
        z_score_transform = torch.distributions.AffineTransform(
            -t_mean / t_std, 1 / t_std
        )
    else:
        z_score_transform = zuko.transforms.IdentityTransform()

    # pre-pend the z-scoring layer to the embedding net.
    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # create the flow matching estimator, will take care of time embeddings.
    return FlowMatchingEstimator(
        net=vectorfield_net,
        input_shape=batch_x[0].shape,  # theta
        condition_shape=batch_y[0].shape,  # x
        zscore_transform_input=z_score_transform,
        embedding_net=embedding_net,  # embedding net for x
        num_freqs=num_freqs,  # number of frequencies in time embeddings
        **kwargs,  # e.g., noise_scale
    )


class VectorFieldMLP(VectorFieldNet):
    """MLP for the vector field regressor."""

    def __init__(self, net: ZukoMLP):
        super(VectorFieldMLP, self).__init__()
        self.net = net

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        # concatenate theta, x, and t
        return self.net(torch.cat([theta, x, t], dim=-1))


class ResNetWithGLUConditioning(VectorFieldNet):
    """ResNet with GLU units for additional conditioning.

    Used for flow matching to encode high-dimensional data in the main network and add
    contet of theta and t using GLU units on each layer, as proposed in Dax et al.
    "Flow Matching for Scalable SBI" (2023)", https://arxiv.org/abs/2305.17161.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        num_blocks: int,
        condition_dim: int,
        output_dim: int,
    ):
        super(ResNetWithGLUConditioning, self).__init__()
        self.blocks = nn.ModuleList([
            ResNetBlock(input_dim, hidden_units, condition_dim)
            for _ in range(num_blocks)
        ])
        self.final_fc = nn.Linear(input_dim, output_dim)

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Pass x through the ResNet with GLU units for additional conditioning.

        Assuming FMPE notation: x is the data, we condition on theta and time.
        """
        glu_condition = torch.cat([t, theta], dim=-1)
        out = x
        for block in self.blocks:
            out = block(out, glu_condition)
        out = self.final_fc(out)
        return out


class GLU(nn.Module):
    """Gated Linear Unit (GLU) block to combined input and condition."""

    def __init__(self, input_dim: int, condition_dim: int):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim + condition_dim, input_dim)
        self.gate = nn.Linear(input_dim + condition_dim, input_dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        # x and condition must match in all dimensions except the last one
        assert x.size()[:-1] == condition.size()[:-1]
        combined_input = torch.cat([x, condition], dim=-1)
        return self.fc(combined_input) * torch.sigmoid(self.gate(combined_input))


class ResNetBlock(nn.Module):
    """ResNet block with GLU units for additional conditioning."""

    def __init__(self, input_dim: int, hidden_units: int, condition_dim: int):
        """Initialize ResNet block.

        The block consists of two fully connected layers with GLU units in between.

        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Dimensionality of the hidden layer.
            condition_dim: Dimensionality of the condition.
        """
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.glu1 = GLU(hidden_units, condition_dim)
        self.fc2 = nn.Linear(hidden_units, input_dim)
        self.glu2 = GLU(input_dim, condition_dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Pass x through the ResNet block, condition with GLU units, add residual."""
        residual = x
        out = F.relu(self.fc1(x))
        out = self.glu1(out, condition)
        out = F.relu(self.fc2(out))
        out = self.glu2(out, condition)
        return out + residual
