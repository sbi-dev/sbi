# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from functools import partial
from warnings import warn

from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from torch import Tensor, nn, relu, tanh, tensor, uint8

from sbi.utils.sbiutils import (
    standardizing_net,
    standardizing_transform,
    z_score_parser,
)
from sbi.utils.torchutils import create_alternating_binary_mask


def build_made(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
    num_mixture_components: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> nn.Module:
    """Builds MADE p(x|y).

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
        num_mixture_components: Number of mixture components.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for mades and are therefore ignored.

    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    y_numel = embedding_net(batch_y[:1]).numel()

    if x_numel == 1:
        warn(f"In one-dimensional output space, this flow is limited to Gaussians")

    transform = transforms.IdentityTransform()

    z_score_x, structured_x = z_score_parser(z_score_x)
    if z_score_x:
        transform_zx = standardizing_transform(batch_x, structured_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    z_score_y, structured_y = z_score_parser(z_score_y)
    if z_score_y:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    distribution = distributions_.MADEMoG(
        features=x_numel,
        hidden_features=hidden_features,
        context_features=y_numel,
        num_blocks=5,
        num_mixture_components=num_mixture_components,
        use_residual_blocks=True,
        random_mask=False,
        activation=relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        custom_initialization=True,
    )

    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


def build_maf(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> nn.Module:
    """Builds MAF p(x|y).

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
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    y_numel = embedding_net(batch_y[:1]).numel()

    if x_numel == 1:
        warn(f"In one-dimensional output space, this flow is limited to Gaussians")

    transform_list = []
    for _ in range(num_transforms):
        block = [
            transforms.MaskedAffineAutoregressiveTransform(
                features=x_numel,
                hidden_features=hidden_features,
                context_features=y_numel,
                num_blocks=2,
                use_residual_blocks=False,
                random_mask=False,
                activation=tanh,
                dropout_probability=0.0,
                use_batch_norm=True,
            ),
            transforms.RandomPermutation(features=x_numel),
        ]
        transform_list += block

    z_score_x, structured_x = z_score_parser(z_score_x)
    if z_score_x:
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    z_score_y, structured_y = z_score_parser(z_score_y)
    if z_score_y:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


def build_nsf(
    batch_x: Tensor = None,
    batch_y: Tensor = None,
    z_score_x: bool = True,
    z_score_y: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    **kwargs,
) -> nn.Module:
    """Builds NSF p(x|y).

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
        num_transforms: Number of transforms.
        num_bins: Number of bins used for the splines.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    y_numel = embedding_net(batch_y[:1]).numel()

    # If x is just a scalar then use a dummy mask and learn spline parameters using the
    # conditioning variables only.
    if x_numel == 1:

        # Define dummy mask because there is only one dimension.
        def mask_in_layer(_):
            return tensor([1], dtype=uint8)

        # Conditioner ignores the data and uses the conditioning variables only.
        conditioner = partial(
            ContextSplineMap,
            hidden_features=hidden_features,
            context_features=y_numel,
            hidden_layers=hidden_layers_spline_context,
        )

    else:
        # Define mask function to alternate between predicted x-dimensions.
        def mask_in_layer(i):
            return create_alternating_binary_mask(features=x_numel, even=(i % 2 == 0))

        # Use conditional resnet as spline conditioner.
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=y_numel,
            num_blocks=2,
            activation=relu,
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    # Stack spline transforms.
    transform_list = []
    for i in range(num_transforms):
        block = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        # Add LU transform only for high D x. Permutation makes sense only for more than
        # one feature.
        if x_numel > 1:
            block.append(
                transforms.LULinear(x_numel, identity_init=True),
            )
        transform_list += block

    z_score_x, structured_x = z_score_parser(z_score_x)
    if z_score_x:
        # Prepend standardizing transform to nsf transforms.
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    z_score_y, structured_y = z_score_parser(z_score_y)
    if z_score_y:
        # Prepend standardizing transform to y-embedding.
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    distribution = distributions_.StandardNormal((x_numel,))

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


class ContextSplineMap(nn.Module):
    """
    Neural network from `context` to the spline parameters.

    We cannot use the resnet as conditioner to learn each dimension conditioned
    on the other dimensions (because there is only one). Instead, we learn the
    spline parameters directly. In the case of conditinal density estimation,
    we make the spline parameters conditional on the context. This is
    implemented in this class.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        context_features: int,
        hidden_layers: int,
    ):
        """
        Initialize neural network that learns to predict spline parameters.

        Args:
            in_features: Unused since there is no `conditioner` in 1D.
            out_features: Number of spline parameters.
            hidden_features: Number of hidden units.
            context_features: Number of context features.
        """
        super().__init__()
        # `self.hidden_features` is only defined such that nflows can infer
        # a scaling factor for initializations.
        self.hidden_features = hidden_features

        # Use a non-linearity because otherwise, there will be a linear
        # mapping from context features onto distribution parameters.

        # Initialize with input layer.
        layer_list = [nn.Linear(context_features, hidden_features), nn.ReLU()]
        # Add hidden layers.
        layer_list += [
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ] * hidden_layers
        # Add output layer.
        layer_list += [nn.Linear(hidden_features, out_features)]
        self.spline_predictor = nn.Sequential(*layer_list)

    def __call__(self, inputs: Tensor, context: Tensor, *args, **kwargs) -> Tensor:
        """
        Return parameters of the spline given the context.

        Args:
            inputs: Unused. It would usually be the other dimensions, but in
                1D, there are no other dimensions.
            context: Context features.

        Returns:
            Spline parameters.
        """
        return self.spline_predictor(context)
