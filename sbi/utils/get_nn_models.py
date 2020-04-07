from __future__ import annotations

from typing import Optional

import torch
from pyknos.mdn.mdn import MultivariateGaussianMDN
from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from pyknos.nflows.nn.nde import MixtureOfGaussiansMADE
from torch import nn

import sbi.utils as utils
from sbi.utils.torchutils import create_alternating_binary_mask


def posterior_nn(
    model: str,
    prior: torch.distributions.Distribution,
    context: torch.Tensor,
    embedding: Optional[torch.nn.Module] = None,
    hidden_features: int = 50,
    mdn_num_components: int = 20,
    made_num_mixture_components: int = 10,
    made_num_blocks: int = 4,
    flow_num_transforms: int = 5,
) -> torch.nn.Module:
    """Neural posterior density estimator

    Args:
        model: Model, one of maf / mdn / made / nsf
        prior: Prior distribution
        context: Observation
        embedding: Embedding network
        hidden_features: For all, number of hidden features
        mdn_num_components: For MDNs only, number of components
        made_num_mixture_components: For MADEs only, number of mixture components
        made_num_blocks: For MADEs only, number of blocks
        flow_num_transforms: For flows only, number of transforms

    Returns:
        Neural network
    """
    mean, std = (prior.mean, prior.stddev)
    standardizing_transform = transforms.AffineTransform(
        shift=-mean / std, scale=1 / std
    )

    parameter_dim = prior.sample([1]).shape[1]

    context = utils.torchutils.atleast_2d(context)
    observation_dim = torch.tensor([context.shape[1:]])

    if model == "mdn":
        neural_net = MultivariateGaussianMDN(
            features=parameter_dim,
            context_features=observation_dim,
            hidden_features=hidden_features,
            hidden_net=nn.Sequential(
                nn.Linear(observation_dim, hidden_features),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
            ),
            num_components=mdn_num_components,
            custom_initialization=True,
        )

    elif model == "made":
        transform = standardizing_transform
        distribution = distributions_.MADEMoG(
            features=parameter_dim,
            hidden_features=hidden_features,
            context_features=observation_dim,
            num_blocks=made_num_blocks,
            num_mixture_components=made_num_mixture_components,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            custom_initialization=True,
        )
        neural_net = flows.Flow(transform, distribution, embedding)

    elif model == "maf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.MaskedAffineAutoregressiveTransform(
                            features=parameter_dim,
                            hidden_features=hidden_features,
                            context_features=observation_dim,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=torch.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        transforms.RandomPermutation(features=parameter_dim),
                    ]
                )
                for _ in range(flow_num_transforms)
            ]
        )

        transform = transforms.CompositeTransform([standardizing_transform, transform,])

        distribution = distributions_.StandardNormal((parameter_dim,))
        neural_net = flows.Flow(transform, distribution, embedding)

    elif model == "nsf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.PiecewiseRationalQuadraticCouplingTransform(
                            mask=create_alternating_binary_mask(
                                features=parameter_dim, even=(i % 2 == 0)
                            ),
                            transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
                                in_features=in_features,
                                out_features=out_features,
                                hidden_features=hidden_features,
                                context_features=observation_dim,
                                num_blocks=2,
                                activation=torch.relu,
                                dropout_probability=0.0,
                                use_batch_norm=False,
                            ),
                            num_bins=10,
                            tails="linear",
                            tail_bound=3.0,
                            apply_unconditional_transform=False,
                        ),
                        transforms.LULinear(parameter_dim, identity_init=True),
                    ]
                )
                for i in range(flow_num_transforms)
            ]
        )

        transform = transforms.CompositeTransform([standardizing_transform, transform,])

        distribution = distributions_.StandardNormal((parameter_dim,))
        neural_net = flows.Flow(transform, distribution, embedding)

    else:
        raise ValueError

    return neural_net


def likelihood_nn(
    model: str,
    prior: torch.distributions.Distribution,
    context: torch.Tensor,
    embedding: Optional[torch.nn.Module] = None,
    hidden_features: int = 50,
    mdn_num_components: int = 20,
    made_num_mixture_components: int = 10,
    made_num_blocks: int = 4,
    flow_num_transforms: int = 5,
) -> torch.nn.Module:
    """Neural likelihood density estimator

    Args:
        model: Model, one of maf / mdn / made / nsf
        prior: Prior distribution
        context: Observation
        embedding: Embedding network
        hidden_features: For all, number of hidden features
        mdn_num_components: For MDNs only, number of components
        made_num_mixture_components: For MADEs only, number of mixture components
        made_num_blocks: For MADEs only, number of blocks
        flow_num_transforms: For flows only, number of transforms

    Returns:
        Neural network
    """
    parameter_dim = prior.sample([1]).shape[1]
    if context.dim() == 1:
        observation_dim = torch.tensor([context.shape]).item()
    else:
        observation_dim = torch.tensor([context.shape[1]]).item()

    if model == "mdn":
        neural_net = MultivariateGaussianMDN(
            features=observation_dim,
            context_features=parameter_dim,
            hidden_features=hidden_features,
            hidden_net=nn.Sequential(
                nn.Linear(parameter_dim, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(hidden_features, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features),
                nn.BatchNorm1d(hidden_features),
                nn.ReLU(),
            ),
            num_components=mdn_num_components,
            custom_initialization=True,
        )

    elif model == "made":
        neural_net = MixtureOfGaussiansMADE(
            features=observation_dim,
            hidden_features=hidden_features,
            context_features=parameter_dim,
            num_blocks=made_num_blocks,
            num_mixture_components=made_num_mixture_components,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            use_batch_norm=True,
            dropout_probability=0.0,
            custom_initialization=True,
        )

    elif model == "maf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.MaskedAffineAutoregressiveTransform(
                            features=observation_dim,
                            hidden_features=hidden_features,
                            context_features=parameter_dim,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=torch.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        transforms.RandomPermutation(features=observation_dim),
                    ]
                )
                for _ in range(flow_num_transforms)
            ]
        )
        distribution = distributions_.StandardNormal((observation_dim,))
        neural_net = flows.Flow(transform, distribution, embedding)

    elif model == "nsf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.PiecewiseRationalQuadraticCouplingTransform(
                            mask=create_alternating_binary_mask(
                                features=observation_dim, even=(i % 2 == 0)
                            ),
                            transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
                                in_features=in_features,
                                out_features=out_features,
                                hidden_features=hidden_features,
                                context_features=parameter_dim,
                                num_blocks=2,
                                activation=torch.relu,
                                dropout_probability=0.0,
                                use_batch_norm=False,
                            ),
                            num_bins=10,
                            tails="linear",
                            tail_bound=3.0,
                            apply_unconditional_transform=False,
                        ),
                        transforms.LULinear(observation_dim, identity_init=True),
                    ]
                )
                for i in range(flow_num_transforms)
            ]
        )
        distribution = distributions_.StandardNormal((observation_dim,))
        neural_net = flows.Flow(transform, distribution)

    else:
        raise ValueError

    return neural_net


def classifier_nn(
    model,
    prior: torch.distributions.Distribution,
    context: torch.Tensor,
    hidden_features: int = 50,
) -> torch.nn.Module:
    """Neural classifier

    Args:
        model: Model, one of linear / mlp / resnet
        prior: Prior distribution
        context: Observation
        hidden_features: For all, number of hidden features

    Returns:
        Neural network
    """
    parameter_dim = prior.sample([1]).shape[1]
    context = utils.torchutils.atleast_2d(context)
    observation_dim = torch.tensor([context.shape[1:]])

    if model == "linear":
        neural_net = nn.Linear(parameter_dim + observation_dim, 1)

    elif model == "mlp":
        neural_net = nn.Sequential(
            nn.Linear(parameter_dim + observation_dim, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    elif model == "resnet":
        neural_net = nets.ResidualNet(
            in_features=parameter_dim + observation_dim,
            out_features=1,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=2,
            activation=torch.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    else:
        raise ValueError(f"'model' must be one of ['linear', 'mlp', 'resnet'].")

    return neural_net
