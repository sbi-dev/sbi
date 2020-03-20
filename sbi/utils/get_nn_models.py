from sbi.utils.torchutils import create_alternating_binary_mask
from torch import nn
from torch.nn import functional as F

import sbi.utils as utils
from pyknos.nflows import distributions as distributions_
from pyknos.nflows import transforms
from pyknos.nflows.nn import nets
import torch
from pyknos.nflows import flows
from pyknos.mdn.mdn import MultivariateGaussianMDN
from pyknos.nflows.nn.nde import MixtureOfGaussiansMADE


def posterior_nn(
    model, prior, context, embedding=None,
):

    mean, std = (prior.mean, prior.stddev)
    normalizing_transform = transforms.AffineTransform(shift=-mean / std, scale=1 / std)

    parameter_dim = prior.sample([1]).shape[1]

    context = utils.torchutils.atleast_2d(context)
    observation_dim = torch.tensor([context.shape[1:]])

    if model == "mdn":
        raise NameError(
            "MDN needs to be updated with embedding_net to work."
            " Problem is that the input shape will be (num_dim) instead of just an integer value"
        )
        hidden_features = 50
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
            num_components=20,
            custom_initialization=True,
        )

    elif model == "made":
        num_mixture_components = 5
        transform = normalizing_transform
        distribution = distributions_.MADEMoG(
            features=parameter_dim,
            hidden_features=50,
            context_features=observation_dim,
            num_blocks=2,
            num_mixture_components=num_mixture_components,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
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
                            hidden_features=50,
                            context_features=observation_dim,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=F.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        transforms.RandomPermutation(features=parameter_dim),
                    ]
                )
                for _ in range(5)
            ]
        )

        transform = transforms.CompositeTransform([normalizing_transform, transform,])

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
                                hidden_features=50,
                                context_features=observation_dim,
                                num_blocks=2,
                                activation=F.relu,
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
                for i in range(5)
            ]
        )

        distribution = distributions_.StandardNormal((parameter_dim,))
        neural_net = flows.Flow(transform, distribution, embedding)

    else:
        raise ValueError

    return neural_net


def likelihood_nn(
    model, prior, context, embedding=None,
):
    parameter_dim = prior.sample([1]).shape[1]
    if context.dim() == 1:
        observation_dim = torch.tensor([context.shape]).item()
    else:
        observation_dim = torch.tensor([context.shape[1]]).item()

    if model == "mdn":
        hidden_features = 50
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
            num_components=20,
            custom_initialization=True,
        )

    elif model == "made":
        neural_net = MixtureOfGaussiansMADE(
            features=observation_dim,
            hidden_features=50,
            context_features=parameter_dim,
            num_blocks=4,
            num_mixture_components=10,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
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
                            hidden_features=50,
                            context_features=parameter_dim,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=F.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        transforms.RandomPermutation(features=observation_dim),
                    ]
                )
                for _ in range(5)
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
                                hidden_features=50,
                                context_features=parameter_dim,
                                num_blocks=2,
                                activation=F.relu,
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
                for i in range(5)
            ]
        )
        distribution = distributions_.StandardNormal((observation_dim,))
        neural_net = flows.Flow(transform, distribution)

    else:
        raise ValueError

    return neural_net


def classifier_nn(
    model, prior, context,
):
    parameter_dim = prior.sample([1]).shape[1]
    context = utils.torchutils.atleast_2d(context)
    observation_dim = torch.tensor([context.shape[1:]])

    if model == "linear":
        neural_net = nn.Linear(parameter_dim + observation_dim, 1)

    elif model == "mlp":
        hidden_dim = 50
        neural_net = nn.Sequential(
            nn.Linear(parameter_dim + observation_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    elif model == "resnet":
        neural_net = nets.ResidualNet(
            in_features=parameter_dim + observation_dim,
            out_features=1,
            hidden_features=50,
            context_features=None,
            num_blocks=2,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    else:
        raise ValueError(f"'model' must be one of ['linear', 'mlp', 'resnet'].")

    return neural_net
