# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from functools import partial
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
import zuko
from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from pyknos.nflows.transforms.splines import (
    rational_quadratic,  # pyright: ignore[reportAttributeAccessIssue]
)
from torch import Tensor, nn, relu, tanh, tensor, uint8
from torch.distributions import Distribution
from zuko.lazy import Flow, LazyDistribution

from sbi.neural_nets.estimators import NFlowsFlow, ZukoFlow, ZukoUnconditionalFlow
from sbi.utils.nn_utils import MADEMoGWrapper, get_numel
from sbi.utils.sbiutils import (
    biject_transform_zuko,
    mcmc_transform,
    standardizing_net,
    standardizing_transform,
    standardizing_transform_zuko,
    z_score_parser,
)
from sbi.utils.torchutils import create_alternating_binary_mask
from sbi.utils.user_input_checks import check_data_device

nflow_specific_kwargs = ["num_bins", "num_components", "tail_bound"]


def build_made(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: int = 50,
    num_mixture_components: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> NFlowsFlow:
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
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=None)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    transform = transforms.IdentityTransform()

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_zx = standardizing_transform(batch_x, structured_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    embedding_net = _prepare_y_embedding(z_score_y, batch_y, embedding_net)

    distribution = MADEMoGWrapper(
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
    flow = NFlowsFlow(
        neural_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )

    return flow


def build_maf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> NFlowsFlow:
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
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(
        batch_x,
        embedding_net=None,
        warn_on_1d=True,  # warn if output space is 1D.
    )
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    transform_list = []
    for _ in range(num_transforms):
        block = [
            transforms.MaskedAffineAutoregressiveTransform(
                features=x_numel,
                hidden_features=hidden_features,
                context_features=y_numel,
                num_blocks=num_blocks,
                use_residual_blocks=False,
                random_mask=False,
                activation=tanh,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ),
            transforms.RandomPermutation(features=x_numel),
        ]
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    embedding_net = _prepare_y_embedding(z_score_y, batch_y, embedding_net)

    # Combine transforms
    transform = transforms.CompositeTransform(transform_list)

    distribution = get_base_dist(x_numel, **kwargs)
    neural_net = flows.Flow(transform, distribution, embedding_net)
    flow = NFlowsFlow(
        neural_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )

    return flow


def build_maf_rqs(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    num_blocks: int = 2,
    num_bins: int = 10,
    tails: Optional[str] = "linear",
    tail_bound: float = 3.0,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    min_bin_width: float = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    **kwargs,
) -> NFlowsFlow:
    """Builds MAF p(x|y), where the diffeomorphisms are rational-quadratic
    splines (RQS).

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
        num_blocks: number of blocks used for residual net for context embedding.
        num_bins: Number of bins of the RQS.
        tails: Whether to use constrained or unconstrained RQS, can be one of:
            - None: constrained RQS.
            - 'linear': unconstrained RQS (RQS transformation is only
            applied on domain [-B, B], with `linear` tails, outside [-B, B],
            identity transformation is returned).
        tail_bound: RQS transformation is applied on domain [-B, B],
            `tail_bound` is equal to B.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        min_bin_width: Minimum bin width.
        min_bin_height: Minimum bin height.
        min_derivative: Minimum derivative at knot values of bins.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(
        batch_x,
        embedding_net=None,
        warn_on_1d=True,  # warn if output space is 1D.
    )
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    transform_list = []
    for _ in range(num_transforms):
        block = [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=x_numel,
                hidden_features=hidden_features,
                context_features=y_numel,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                num_blocks=num_blocks,
                use_residual_blocks=False,
                random_mask=False,
                activation=tanh,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            ),
            transforms.RandomPermutation(features=x_numel),
        ]
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    embedding_net = _prepare_y_embedding(z_score_y, batch_y, embedding_net)

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = get_base_dist(x_numel, **kwargs)
    neural_net = flows.Flow(transform, distribution, embedding_net)
    flow = NFlowsFlow(
        neural_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )

    return flow


def build_nsf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> NFlowsFlow:
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
        tail_bound: tail bound for each spline.
        hidden_layers_spline_context: number of hidden layers of the spline context net
            for one-dimensional x.
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=None)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # Define mask function to alternate between predicted x-dimensions.
    def mask_in_layer(i):
        return create_alternating_binary_mask(features=x_numel, even=(i % 2 == 0))

    # If x is just a scalar then use a dummy mask and learn spline parameters using the
    # conditioning variables only.
    if x_numel == 1:
        # Conditioner ignores the data and uses the conditioning variables only.
        conditioner = partial(
            ContextSplineMap,
            hidden_features=hidden_features,
            context_features=y_numel,
            hidden_layers=hidden_layers_spline_context,
        )
    else:
        # Use conditional resnet as spline conditioner.
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=y_numel,
            num_blocks=num_blocks,
            activation=relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    # Stack spline transforms.
    transform_list = []
    for i in range(num_transforms):
        block: List[transforms.Transform] = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i) if x_numel > 1 else tensor([1], dtype=uint8),
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

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        # Prepend standardizing transform to nsf transforms.
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    embedding_net = _prepare_y_embedding(z_score_y, batch_y, embedding_net)

    distribution = get_base_dist(x_numel, **kwargs)

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)
    neural_net = flows.Flow(transform, distribution, embedding_net)
    flow = NFlowsFlow(
        neural_net, input_shape=batch_x[0].shape, condition_shape=batch_y[0].shape
    )

    return flow


def build_zuko_nice(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    randmask: bool = False,
    **kwargs,
) -> ZukoFlow:
    """
    Build a Non-linear Independent Components Estimation (NICE) flow.

    Affine transformations are used by default, instead of the additive transformations
    used by Dinh et al. (2014) originally.

    References:
        | NICE: Non-linear Independent Components Estimation (Dinh et al., 2014)
        | https://arxiv.org/abs/1410.8516

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        randmask: Whether to use random masks in the flow. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "NICE"
    additional_kwargs = {"randmask": randmask, **kwargs}
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_maf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    randperm: bool = False,
    **kwargs,
) -> ZukoFlow:
    """
    Build a Masked Autoregressive Flow (MAF).

    References:
        | Masked Autoregressive Flow for Density Estimation (Papamakarios et al., 2017)
        | https://arxiv.org/abs/1705.07057

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        randperm: Whether to use random permutations in the flow. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "MAF"
    additional_kwargs = {"randperm": randperm, **kwargs}
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_nsf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    num_bins: int = 8,
    **kwargs,
) -> ZukoFlow:
    """
    Build a Neural Spline Flow (NSF) with monotonic rational-quadratic spline
    transformations.

    By default, transformations are fully autoregressive. Coupling transformations
    can be obtained by setting :py:`passes=2`.

    Warning:
        Spline transformations are defined over the domain :math:`[-5, 5]`. Any feature
        outside of this domain is not transformed. It is recommended to standardize
        features (zero mean, unit variance) before training.

    References:
        | Neural Spline Flows (Durkan et al., 2019)
        | https://arxiv.org/abs/1906.04032

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        num_bins: The number of bins in the spline transformations. Defaults to 8.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "NSF"
    additional_kwargs = {"bins": num_bins, **kwargs}
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_ncsf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    num_bins: int = 8,
    **kwargs,
) -> ZukoFlow:
    r"""
    Build a Neural Circular Spline Flow (NCSF).

    Circular spline transformations are obtained by composing circular domain shifts
    with regular spline transformations. Features are assumed to lie in the half-open
    interval :math:`[-\pi, \pi[`.

    References:
        | Normalizing Flows on Tori and Spheres (Rezende et al., 2020)
        | https://arxiv.org/abs/2002.02428

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        num_bins: The number of bins in the spline transformations. Defaults to 8.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "NCSF"
    additional_kwargs = {"bins": num_bins, **kwargs}
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_sospf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    degree: int = 4,
    polynomials: int = 3,
    **kwargs,
) -> ZukoFlow:
    """
    Build a Sum-of-Squares Polynomial Flow (SOSPF).

    References:
        | Sum-of-Squares Polynomial Flow (Jaini et al., 2019)
        | https://arxiv.org/abs/1905.02325

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        degree: The degree of the polynomials. Defaults to 4.
        polynomials: The number of polynomials. Defaults to 3.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "SOSPF"
    additional_kwargs = {"degree": degree, "polynomials": polynomials, **kwargs}
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_naf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    randperm: bool = False,
    signal: int = 16,
    **kwargs,
) -> ZukoFlow:
    """
    Build a Neural Autoregressive Flow (NAF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Neural Autoregressive Flows (Huang et al., 2018)
        | https://arxiv.org/abs/1804.00779

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        randperm: Whether features are randomly permuted between transformations or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transformations.
        signal: The number of signal features of the monotonic network.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "NAF"
    additional_kwargs = {
        "randperm": randperm,
        "signal": signal,
        # "network": network,
        **kwargs,
    }
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_unaf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    randperm: bool = False,
    signal: int = 16,
    **kwargs,
) -> ZukoFlow:
    """
    Build an Unconstrained Neural Autoregressive Flow (UNAF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Unconstrained Monotonic Neural Networks (Wehenkel et al., 2019)
        | https://arxiv.org/abs/1908.05164

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        randperm: Whether features are randomly permuted between transformations or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transformations.
        signal: The number of signal features of the monotonic network.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "UNAF"
    additional_kwargs = {
        "randperm": randperm,
        "signal": signal,
        # "network": network,
        **kwargs,
    }
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_cnf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
) -> ZukoFlow:
    """
    Build a Continuous Normalizing Flow (CNF) with a free-form Jacobian transformation.

    References:
        | Neural Ordinary Differential Equations (Chen el al., 2018)
        | https://arxiv.org/abs/1806.07366

        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible
        | Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "CNF"
    additional_kwargs = {**kwargs}
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_gf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 3,
    embedding_net: nn.Module = nn.Identity(),
    components: int = 8,
    **kwargs,
) -> ZukoFlow:
    """
    Build a Gaussianization Flow (GF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Gaussianization Flows (Meng et al., 2020)
        | https://arxiv.org/abs/2003.01941

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        components: The number of components in the Gaussian mixture model.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "GF"
    additional_kwargs = {"components": components, **kwargs}
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_bpf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 3,
    embedding_net: nn.Module = nn.Identity(),
    degree: int = 16,
    **kwargs,
) -> ZukoFlow:
    """
    Build a Bernstein polynomial flow (BPF).

    Warning:
        Invertibility is only guaranteed for features within the interval :math:`[-10,
        10]`. It is recommended to standardize features (zero mean, unit variance)
        before training.

    References:
        | Short-Term Density Forecasting of Low-Voltage Load using
        | Bernstein-Polynomial Normalizing Flows (Arpogaus et al., 2022)
        | https://arxiv.org/abs/2204.13939

    Arguments:
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
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        degree: The degree :math:`M` of the Bernstein polynomial.
        **kwargs: Additional keyword arguments to pass to the flow constructor.
    """
    which_nf = "BPF"
    additional_kwargs = {"degree": degree, **kwargs}
    flow = build_zuko_flow(
        which_nf,
        batch_x,
        batch_y,
        z_score_x,
        z_score_y,
        hidden_features,
        num_transforms,
        embedding_net,
        **additional_kwargs,
    )

    return flow


def build_zuko_flow(
    which_nf: str,
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ] = "independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    x_dist: Optional[Distribution] = None,
    **kwargs,
) -> ZukoFlow:
    """
    Fundamental building blocks to build a Zuko normalizing flow model.

    The following cases are considered in the if statements down below:

    z_score_x is `independent, `structured` or None, in which case we just use
        the normal standardizing transform.
    z_score_x is `transform_to_unconstrained`, in this case, we check if `x_dist` is
        provided and has a support property. If `x_dist` is not valid (i.e. None
        or has no support property), we raise an error.

    Args:
        which_nf (str): The type of normalizing flow to build.
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
            - `transform_to_unconstrained`: Transforms to
            an unbounded space if bounds from `x_dist` are given.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        embedding_net: The embedding network to use. Defaults to nn.Identity().
        x_dist: The distribution over x, used to determine the bounds for the
            unconstrained transformation.
            - In Neural Posterior Estimation (NPE), `x_dist` typically corresponds
            to the prior over x (e.g., a `BoxUniform`).
            - For Neural Likelihood Estimation (NLE) or Neural Ratio Estimation (NRE),
            `x_dist` may instead be a user-specified distribution. However, make sure
            all the data lies within the support of the distribution if you want to
            use the `transform_to_unconstrained` option for NLE and NRE.
        **kwargs: Additional keyword arguments to pass to the flow constructor.

    Returns:
        ZukoFlow: The constructed Zuko normalizing flow model.
    """
    check_data_device(batch_x, batch_y)
    x_numel = get_numel(batch_x, embedding_net=None)
    y_numel = get_numel(batch_y, embedding_net=embedding_net)

    # Keep only zuko kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in nflow_specific_kwargs}

    if isinstance(hidden_features, int):
        hidden_features = [hidden_features] * num_transforms

    # Get base transforms from specified flow
    base, base_transforms = _get_base_and_transforms(
        which_nf, x_numel, y_numel, hidden_features, num_transforms, **kwargs
    )

    # Get x transforms (z-score or logit transform)
    x_transforms = _prepare_x_transforms(z_score_x, batch_x, x_dist)

    # Combine all transforms
    transforms = x_transforms + base_transforms

    # Maybe add y-z-scoring via embedding network
    embedding_net = _prepare_y_embedding(z_score_y, batch_y, embedding_net)

    # Create final neural network
    neural_net = zuko.flows.Flow(transforms, base)

    flow = ZukoFlow(
        neural_net,
        embedding_net,
        input_shape=batch_x[0].shape,
        condition_shape=batch_y[0].shape,
    )

    return flow


def _get_base_and_transforms(
    which_nf: str,
    x_numel: int,
    y_numel: int,
    hidden_features: Sequence[int],
    num_transforms: int,
    **kwargs,
) -> Tuple[LazyDistribution, tuple]:
    """
    Build the base zuko flow and extract its transforms.

    Args:
        which_nf: The type of normalizing flow to build.
        x_numel: Number of elements in x.
        y_numel: Number of elements in y.
        hidden_features: Hidden features as a sequence.
        num_transforms: Number of transforms.
        **kwargs: Additional arguments for flow constructor.

    Returns:
        tuple of flow base and its transforms.
    """
    build_nf = getattr(zuko.flows, which_nf)

    if which_nf == "CNF":
        flow: Flow = build_nf(
            features=x_numel, context=y_numel, hidden_features=hidden_features, **kwargs
        )
        # CNF has a single continuous transform
        base_transforms = (flow.transform,)
    else:
        flow: Flow = build_nf(
            features=x_numel,
            context=y_numel,
            hidden_features=hidden_features,
            transforms=num_transforms,
            **kwargs,
        )
        # Regular flows have multiple discrete transforms
        base_transforms = tuple(flow.transform.transforms)

    return flow.base, base_transforms


def _prepare_x_transforms(
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ],
    batch_x: Tensor,
    x_dist: Optional[Distribution],
) -> tuple:
    """
    Prepare transforms to prepend for x processing.

    Args:
        z_score_x: Type of x preprocessing.
        batch_x: Batch of x data.
        x_dist: Distribution for unconstrained transformation.

    Returns:
        Tuple of transforms to prepend (empty tuple if no preprocessing).
    """
    transforms = ()
    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x == "transform_to_unconstrained":
        if x_dist is None:
            raise ValueError(
                "Transformation to unconstrained space requires a distribution "
                "provided through `x_dist`."
            )
        if not hasattr(x_dist, "support"):
            raise ValueError(
                "`x_dist` requires a `.support` attribute for"
                "an unconstrained transformation."
            )
        transform_to_unconstrained = biject_transform_zuko(mcmc_transform(x_dist))
        transforms = (transform_to_unconstrained,)
    elif z_score_x_bool:
        z_score_transform = standardizing_transform_zuko(batch_x, structured_x)
        transforms = (z_score_transform,)

    return transforms


def build_zuko_unconditional_flow(
    which_nf: str,
    batch_x: Tensor,
    z_score_x: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ],
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    **kwargs,
) -> ZukoUnconditionalFlow:
    """
    Fundamental building blocks to build a Zuko normalizing flow model.

    Args:
        which_nf (str): The type of normalizing flow to build.
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        hidden_features: The number of hidden features in the flow. Defaults to 50.
        num_transforms: The number of transformations in the flow. Defaults to 5.
        **kwargs: Additional keyword arguments to pass to the flow constructor.

    Returns:
        ZukoUnconditionalFlow: The constructed Zuko normalizing flow model.
    """

    # check_data_device(batch_x)
    x_numel = get_numel(batch_x, embedding_net=None)

    # keep only zuko kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in nflow_specific_kwargs}

    if isinstance(hidden_features, int):
        hidden_features = [hidden_features] * num_transforms

    base, base_transforms = _get_base_and_transforms(
        which_nf, x_numel, 0, hidden_features, num_transforms, **kwargs
    )

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        # TODO: Check whether first base transform, then z-score is correct (it's the
        # other way around in the conditional flows).
        transforms = (
            *base_transforms,
            standardizing_transform_zuko(batch_x, structured_x),
        )
    else:
        transforms = base_transforms

    # Combine transforms.
    neural_net = zuko.flows.Flow(transforms, base)

    flow = ZukoUnconditionalFlow(
        neural_net,
        input_shape=batch_x[0].shape,
    )

    return flow


def _prepare_y_embedding(
    z_score_y: Literal[
        "none", "independent", "structured", "transform_to_unconstrained"
    ],
    batch_y: Tensor,
    embedding_net: nn.Module,
) -> nn.Module:
    """
    Prepend the embedding network for y, adding z-scoring if needed.

    Args:
        z_score_y: Type of y preprocessing.
        batch_y: Batch of y data.
        embedding_net: Original embedding network.

    Returns:
        Modified embedding network.
    """
    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        return nn.Sequential(standardizing_net(batch_y, structured_y), embedding_net)
    return embedding_net


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


def get_base_dist(
    num_dims: int, dtype: torch.dtype = torch.float32, **kwargs
) -> distributions_.Distribution:
    """Returns the base distribution for a flow with given float dtype."""

    base = distributions_.StandardNormal((num_dims,))
    base._log_z = base._log_z.to(dtype)
    return base


def build_zuko_vi_flow(
    event_shape: torch.Size,
    link_transform: Optional["zuko.transforms.Transform"] = None,
    flow_type: str = "nsf",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    **kwargs,
) -> Flow:
    """Build an unconditional Zuko normalizing flow for variational inference.

    This function creates a Zuko flow suitable for use with VI training. The flow
    maps from a simple base distribution (standard normal) to a more complex
    distribution that approximates the posterior.

    Args:
        event_shape: Shape of the events generated by the distribution. For 1D
            parameters, this is typically torch.Size([dim]).
        link_transform: Optional bijective transform that constrains samples to
            a specific support (e.g., transforms to positive reals or bounded
            intervals). Applied as the final transform in the flow.
        flow_type: The type of normalizing flow to build. Supported options:
            - "nsf": Neural Spline Flow (default, flexible and expressive)
            - "maf": Masked Autoregressive Flow (fast density evaluation)
            - "gaussian": Full covariance Gaussian (single affine transform)
            - "gaussian_diag": Diagonal covariance Gaussian
        hidden_features: The number of hidden features in each transform layer.
            Can be an int (same for all layers) or sequence. Defaults to 50.
        num_transforms: The number of transform layers. Defaults to 5. Ignored
            for gaussian and gaussian_diag flow types.
        **kwargs: Additional keyword arguments passed to the Zuko flow constructor.
            Common options include `randperm` (bool) for permutation between layers.

    Returns:
        A Zuko Flow object that can be used for VI training. The flow has
        `log_prob()` and `rsample()` methods through its distribution interface.

    Raises:
        ValueError: If an unsupported flow_type is specified.

    Example:
        >>> import torch
        >>> from sbi.neural_nets.net_builders.flow import build_zuko_vi_flow
        >>> flow = build_zuko_vi_flow(
        ...     event_shape=torch.Size([2]),
        ...     flow_type="nsf",
        ...     num_transforms=3,
        ... )
        >>> dist = flow()
        >>> samples = dist.rsample((100,))  # Shape: (100, 2)
    """
    # Convert event_shape to number of features
    if len(event_shape) != 1:
        raise ValueError(
            f"event_shape must be 1D, got {event_shape}. "
            "Multi-dimensional event shapes are not yet supported for VI flows."
        )
    features = event_shape[0]

    # Handle hidden_features as sequence
    if isinstance(hidden_features, int):
        hidden_features_list = [hidden_features] * num_transforms
    else:
        hidden_features_list = list(hidden_features)

    # Keep only zuko-compatible kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in nflow_specific_kwargs}

    # Build the base flow based on flow_type
    flow_type_lower = flow_type.lower()

    if flow_type_lower == "nsf":
        base_flow = zuko.flows.NSF(
            features=features,
            context=0,  # Unconditional
            hidden_features=hidden_features_list,
            transforms=num_transforms,
            **kwargs,
        )
    elif flow_type_lower == "maf":
        base_flow = zuko.flows.MAF(
            features=features,
            context=0,
            hidden_features=hidden_features_list,
            transforms=num_transforms,
            **kwargs,
        )
    elif flow_type_lower in ("gaussian", "gaussian_diag"):
        # For Gaussian distributions, we create a simple affine transform
        # The base is a standard normal, and we learn location and scale
        base_flow = _build_zuko_gaussian_flow(
            features=features,
            diagonal=(flow_type_lower == "gaussian_diag"),
        )
    else:
        supported = ["nsf", "maf", "gaussian", "gaussian_diag"]
        raise ValueError(
            f"Unsupported flow_type '{flow_type}'. Supported types: {supported}"
        )

    # Apply link transform if provided
    if link_transform is not None:
        # Compose the base flow transforms with the link transform
        # The link transform maps from unconstrained space to the target support
        transforms = [*base_flow.transform.transforms, link_transform]
        neural_net = zuko.flows.Flow(transforms, base_flow.base)  # type: ignore[arg-type]
        return neural_net

    return base_flow


def _build_zuko_gaussian_flow(
    features: int,
    diagonal: bool = False,
) -> Flow:
    """Build a simple Gaussian flow (affine transform on standard normal).

    Args:
        features: Number of features/dimensions.
        diagonal: If True, use diagonal covariance (scale). If False, use
            full covariance (lower triangular).

    Returns:
        A Zuko Flow representing a Gaussian distribution.
    """
    # Create learnable location and scale parameters
    loc = torch.nn.Parameter(torch.zeros(features))

    if diagonal:
        # Diagonal scale (log-space for positivity)
        log_scale = torch.nn.Parameter(torch.zeros(features))

        class DiagAffineTransform(zuko.transforms.Transform):
            """Diagonal affine transform: y = loc + exp(log_scale) * x."""

            domain = zuko.transforms.constraints.real_vector
            codomain = zuko.transforms.constraints.real_vector
            bijective = True

            def __init__(self, loc: torch.nn.Parameter, log_scale: torch.nn.Parameter):
                super().__init__()
                self.loc = loc
                self.log_scale = log_scale

            def _call(self, x: torch.Tensor) -> torch.Tensor:
                return self.loc + torch.exp(self.log_scale) * x

            def _inverse(self, y: torch.Tensor) -> torch.Tensor:
                return (y - self.loc) / torch.exp(self.log_scale)

            def log_abs_det_jacobian(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                return self.log_scale.sum()

        transform = DiagAffineTransform(loc, log_scale)

    else:
        # Lower triangular scale matrix (with positive diagonal via exp)
        # We parameterize: L = tril(raw_L) with diag(L) = exp(diag(raw_L))
        tril_indices = torch.tril_indices(features, features)
        num_tril = len(tril_indices[0])
        raw_tril = torch.nn.Parameter(torch.zeros(num_tril))

        class TriLAffineTransform(zuko.transforms.Transform):
            """Lower triangular affine transform: y = loc + L @ x."""

            domain = zuko.transforms.constraints.real_vector
            codomain = zuko.transforms.constraints.real_vector
            bijective = True

            def __init__(
                self,
                loc: torch.nn.Parameter,
                raw_tril: torch.nn.Parameter,
                tril_indices: torch.Tensor,
                features: int,
            ):
                super().__init__()
                self.loc = loc
                self.raw_tril = raw_tril
                # Store indices directly (not as buffer since Transform isn't nn.Module)
                self._tril_indices = tril_indices
                self._features = features

            def _get_scale_tril(self) -> torch.Tensor:
                """Construct the lower triangular matrix from parameters."""
                L = torch.zeros(
                    self._features,
                    self._features,
                    device=self.raw_tril.device,
                    dtype=self.raw_tril.dtype,
                )
                L[self._tril_indices[0], self._tril_indices[1]] = self.raw_tril
                # Make diagonal positive via exp
                diag_indices = torch.arange(self._features, device=L.device)
                L[diag_indices, diag_indices] = torch.exp(L[diag_indices, diag_indices])
                return L

            def _call(self, x: torch.Tensor) -> torch.Tensor:
                L = self._get_scale_tril()
                return self.loc + x @ L.T

            def _inverse(self, y: torch.Tensor) -> torch.Tensor:
                L = self._get_scale_tril()
                return torch.linalg.solve_triangular(
                    L, (y - self.loc).unsqueeze(-1), upper=False
                ).squeeze(-1)

            def log_abs_det_jacobian(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> torch.Tensor:
                L = self._get_scale_tril()
                # Log det of lower triangular = sum of log of diagonal
                return torch.log(torch.diag(L)).sum()

        transform = TriLAffineTransform(loc, raw_tril, tril_indices, features)

    # Standard normal base distribution
    base = zuko.distributions.DiagNormal(torch.zeros(features), torch.ones(features))

    return zuko.flows.Flow([transform], base)  # type: ignore[arg-type]
