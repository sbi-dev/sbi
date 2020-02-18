from sbi.utils.torchutils import create_alternating_binary_mask
from torch import nn
from torch.nn import functional as F

from nflows import distributions as distributions_
from nflows import transforms
from nflows.nn import nets
from sbi.inference.snpe.sbi_MDN_posterior import MDNPosterior
from sbi.inference.snpe.sbi_flow_posterior import FlowPosterior


def get_sbi_posterior(
    model,
    embedding,
    parameter_dim,
    observation_dim,
    prior,
    context,
    train_with_mcmc,
    mcmc_method,
):

    mean, std = (prior.mean, prior.stddev)
    normalizing_transform = transforms.AffineTransform(shift=-mean / std, scale=1 / std)

    if model == "mdn":
        hidden_features = 50
        neural_posterior = MDNPosterior(
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
            prior=prior,
            context=context,
            train_with_mcmc=train_with_mcmc,
            mcmc_method=mcmc_method,
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
        neural_posterior = FlowPosterior(
            transform,
            distribution,
            embedding,
            prior=prior,
            context=context,
            train_with_mcmc=train_with_mcmc,
            mcmc_method=mcmc_method,
        )

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
        neural_posterior = FlowPosterior(
            transform,
            distribution,
            embedding,
            prior=prior,
            context=context,
            train_with_mcmc=train_with_mcmc,
            mcmc_method=mcmc_method,
        )

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
        neural_posterior = FlowPosterior(
            transform,
            distribution,
            embedding,
            prior=prior,
            context=context,
            train_with_mcmc=train_with_mcmc,
            mcmc_method=mcmc_method,
        )

    else:
        raise ValueError

    return neural_posterior
