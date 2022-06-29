from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from pyro.distributions import transforms
from pyro.nn import AutoRegressiveNN, DenseNN
from torch import nn
from torch.distributions import Distribution, Independent, Normal

from sbi.samplers.vi.vi_utils import filter_kwrags_for_func, get_modules, get_parameters
from sbi.types import TorchTransform

# Supported transforms and flows are registered here i.e. associated with a name

_TRANSFORMS = {}
_TRANSFORMS_INITS = {}
_FLOW_BUILDERS = {}


def register_transform(
    cls: Optional[TorchTransform] = None,
    name: Optional[str] = None,
    inits: Callable = lambda *args, **kwargs: (args, kwargs),
) -> Callable:
    """Decorator to register a learnable transformation.


    Args:
        cls: Class to register
        name: Name of the transform.
        inits: Function that provides initial args and kwargs.


    """

    def _register(cls):
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _TRANSFORMS:
            raise ValueError(f"The transform {cls_name} is already registered")
        else:
            _TRANSFORMS[cls_name] = cls
            _TRANSFORMS_INITS[cls_name] = inits
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_all_transforms() -> List[str]:
    """Returns all registered transforms.

    Returns:
        List[str]: List of names of all transforms.

    """
    return list(_TRANSFORMS.keys())


def get_transform(name: str, dim: int, device: str = "cpu", **kwargs) -> TorchTransform:
    """Returns an initialized transformation



    Args:
        name: Name of the transform, must be one of [affine_diag,
            affine_tril, affine_coupling, affine_autoregressive, spline_coupling,
            spline_autoregressive].
        dim: Input dimension.
        device: Device on which everythink is initialized.
        kwargs: All associated parameters which will be passed through.

    Returns:
        Transform: Invertible transformation.

    """
    name = name.lower()
    transform = _TRANSFORMS[name]
    overwritable_kwargs = filter_kwrags_for_func(transform.__init__, kwargs)
    args, default_kwargs = _TRANSFORMS_INITS[name](dim, device=device, **kwargs)
    kwargs = {**default_kwargs, **overwritable_kwargs}
    return _TRANSFORMS[name](*args, **kwargs)


def register_flow_builder(
    cls: Optional[Callable] = None, name: Optional[str] = None
) -> Callable:
    """Registers a function that builds a normalizing flow.

    Args:
        cls: Builder that is registered.
        name: Name of the builder.


    """

    def _register(cls):
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _FLOW_BUILDERS:
            raise ValueError(f"The flow {cls_name} is not registered as default.")
        else:
            _FLOW_BUILDERS[cls_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_default_flows() -> List[str]:
    """Returns names of all registered flow builders.

    Returns:
        List[str]: List of names.

    """
    return list(_FLOW_BUILDERS.keys())


def get_flow_builder(
    name: str,
    **kwargs,
) -> Callable:
    """Returns an normalizing flow, by instantiating the flow build with all arguments.
    For details within the keyword arguments we refer to the actual builder class. Some
    common arguments are listed here.

    Args:
        name: Name of the flow.
        kwargs: Hyperparameters for the flow.
            num_transforms: Number of normalizing flows that are concatenated.
            permute: Permute dimension after each layer. This may helpfull for
                autoregressive or coupling nets.
            batch_norm: Perform batch normalization.
            base_dist: Base distribution. If `None` then a standard Gaussian is used.
            hidden_dims: The dimensionality of the hidden units per layer. Given as a
                list of integers.

    Returns:
        Callable: A function that if called returns a initialized flow.

    """

    def build_fn(
        event_shape: torch.Size, link_flow: TorchTransform, device: str = "cpu"
    ):
        return _FLOW_BUILDERS[name](event_shape, link_flow, device=device, **kwargs)

    build_fn.__doc__ = _FLOW_BUILDERS[name].__doc__

    return build_fn


# Initialization functions.


def init_affine_autoregressive(dim: int, device: str = "cpu", **kwargs):
    """Provides the default initial arguments for an affine autoregressive transform."""
    hidden_dims = kwargs.pop("hidden_dims", [3 * dim + 5, 3 * dim + 5])
    skip_connections = kwargs.pop("skip_connections", False)
    nonlinearity = kwargs.pop("nonlinearity", nn.ReLU())
    arn = AutoRegressiveNN(
        dim, hidden_dims, nonlinearity=nonlinearity, skip_connections=skip_connections
    ).to(device)
    return [arn], {"log_scale_min_clip": -3.0}


def init_spline_autoregressive(dim: int, device: str = "cpu", **kwargs):
    """Provides the default initial arguments for an spline autoregressive transform."""
    hidden_dims = kwargs.pop("hidden_dims", [3 * dim + 5, 3 * dim + 5])
    skip_connections = kwargs.pop("skip_connections", False)
    nonlinearity = kwargs.pop("nonlinearity", nn.ReLU())
    count_bins = kwargs.get("count_bins", 10)
    order = kwargs.get("order", "linear")
    bound = kwargs.get("bound", 10)
    if order == "linear":
        param_dims = [count_bins, count_bins, (count_bins - 1), count_bins]
    else:
        param_dims = [count_bins, count_bins, (count_bins - 1)]
    neural_net = AutoRegressiveNN(
        dim,
        hidden_dims,
        param_dims=param_dims,
        skip_connections=skip_connections,
        nonlinearity=nonlinearity,
    ).to(device)
    return [dim, neural_net], {"count_bins": count_bins, "bound": bound, "order": order}


def init_affine_coupling(dim: int, device: str = "cpu", **kwargs):
    """Provides the default initial arguments for an affine autoregressive transform."""
    assert dim > 1, "In 1d this would be equivalent to affine flows, use them."
    nonlinearity = kwargs.pop("nonlinearity", nn.ReLU())
    split_dim = kwargs.get("split_dim", dim // 2)
    hidden_dims = kwargs.pop("hidden_dims", [5 * dim + 20, 5 * dim + 20])
    arn = DenseNN(split_dim, hidden_dims, nonlinearity=nonlinearity).to(device)
    return [split_dim, arn], {"log_scale_min_clip": -3.0}


def init_spline_coupling(dim: int, device: str = "cpu", **kwargs):
    """Intitialize a spline coupling transform, by providing necessary args and
    kwargs."""
    assert dim > 1, "In 1d this would be equivalent to affine flows, use them."
    split_dim = kwargs.get("split_dim", dim // 2)
    hidden_dims = kwargs.pop("hidden_dims", [5 * dim + 30, 5 * dim + 30])
    nonlinearity = kwargs.pop("nonlinearity", nn.ReLU())
    count_bins = kwargs.get("count_bins", 15)
    order = kwargs.get("order", "linear")
    bound = kwargs.get("bound", 10)
    if order == "linear":
        param_dims = [
            (dim - split_dim) * count_bins,
            (dim - split_dim) * count_bins,
            (dim - split_dim) * (count_bins - 1),
            (dim - split_dim) * count_bins,
        ]
    else:
        param_dims = [
            (dim - split_dim) * count_bins,
            (dim - split_dim) * count_bins,
            (dim - split_dim) * (count_bins - 1),
        ]
    neural_net = DenseNN(
        split_dim, hidden_dims, param_dims, nonlinearity=nonlinearity
    ).to(device)
    return [dim, split_dim, neural_net], {
        "count_bins": count_bins,
        "bound": bound,
        "order": order,
    }


# Register these directly from pyro

register_transform(
    transforms.AffineAutoregressive,
    "affine_autoregressive",
    inits=init_affine_autoregressive,
)
register_transform(
    transforms.SplineAutoregressive,
    "spline_autoregressive",
    inits=init_spline_autoregressive,
)

register_transform(
    transforms.AffineCoupling, "affine_coupling", inits=init_affine_coupling
)

register_transform(
    transforms.SplineCoupling, "spline_coupling", inits=init_spline_coupling
)


# Register these very simple transforms.


@register_transform(
    name="affine_diag",
    inits=lambda dim, device="cpu", **kwargs: (
        [],
        {
            "loc": torch.zeros(dim, device=device),
            "scale": torch.ones(dim, device=device),
        },
    ),
)
class AffineTransform(transforms.AffineTransform):
    """Trainable version of an Affine transform. This can be used to get diagonal
    Gaussian approximations."""

    __doc__ += transforms.AffineTransform.__doc__

    def parameters(self):
        self.loc.requires_grad_(True)
        self.scale.requires_grad_(True)
        yield self.loc
        yield self.scale

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return AffineTransform(self.loc, self.scale, cache_size=cache_size)


@register_transform(
    name="affine_tril",
    inits=lambda dim, device="cpu", **kwargs: (
        [],
        {
            "loc": torch.zeros(dim, device=device),
            "scale_tril": torch.eye(dim, device=device),
        },
    ),
)
class LowerCholeskyAffine(transforms.LowerCholeskyAffine):
    """Trainable version of a Lower Cholesky Affine transform. This can be used to get
    full Gaussian approximations."""

    __doc__ += transforms.LowerCholeskyAffine.__doc__

    def parameters(self):
        self.loc.requires_grad_(True)
        self.scale_tril.requires_grad_(True)
        yield self.loc
        yield self.scale_tril

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return LowerCholeskyAffine(self.loc, self.scale_tril, cache_size=cache_size)

    def log_abs_det_jacobian(self, x, y):
        """This modification allows batched scale_tril matrices."""
        return self.log_abs_jacobian_diag(x, y).sum(-1)

    def log_abs_jacobian_diag(self, x, y):
        """This returns the full diagonal which is necessary to compute conditionals"""
        dim = self.scale_tril.dim()
        return torch.diagonal(self.scale_tril, dim1=dim - 2, dim2=dim - 1).log()


# Now build Normalizing flows


class TransformedDistribution(torch.distributions.TransformedDistribution):
    """This is TransformedDistribution with the capability to return parameters."""

    assert __doc__ is not None
    assert torch.distributions.TransformedDistribution.__doc__ is not None
    __doc__ += torch.distributions.TransformedDistribution.__doc__

    def parameters(self):
        if hasattr(self.base_dist, "parameters"):
            yield from self.base_dist.parameters()  # type: ignore
        for t in self.transforms:
            yield from get_parameters(t)

    def modules(self):
        if hasattr(self.base_dist, "modules"):
            yield from self.base_dist.modules()  # type: ignore
        for t in self.transforms:
            yield from get_modules(t)


def build_flow(
    event_shape: torch.Size,
    link_flow: transforms.Transform,
    num_transforms: int = 5,
    transform: str = "affine_autoregressive",
    permute: bool = True,
    batch_norm: bool = False,
    base_dist: Optional[Distribution] = None,
    device: str = "cpu",
    **kwargs,
) -> TransformedDistribution:
    """Generates a Transformed Distribution where the base_dist is transformed by
       num_transforms bijective transforms of specified type.

    Args:
        event_shape: Shape of the events generated by the distribution.
        link_flow: Links to a specific support .
            num_transforms: Number of normalizing flows that are concatenated.
        transform: The type of normalizing flow. Should be one of [affine_diag,
            affine_tril, affine_coupling, affine_autoregressive, spline_coupling,
            spline_autoregressive].
        permute: Permute dimension after each layer. This may helpfull for
            autoregressive or coupling nets.
        batch_norm: Perform batch normalization.
        base_dist: Base distribution. If `None` then a standard Gaussian is used.
        device: Device on which we build everythink.
        kwargs: Hyperparameters are added here.
    Returns:
        TransformedDistribution

    """
    # Some transforms increase dimension by decreasing the degrees of freedom e.g.
    # SoftMax.
    # `unsqueeze(0)` because the `link_flow` requires a batch dimension if the prior is
    # a `MultipleIndependent`.
    additional_dim = (
        len(link_flow(torch.zeros(event_shape, device=device).unsqueeze(0))[0])
        - torch.tensor(event_shape, device=device).item()
    )
    event_shape = torch.Size(
        (torch.tensor(event_shape, device=device) - additional_dim).tolist()
    )
    # Base distribution is standard normal if not specified
    if base_dist is None:
        base_dist = Independent(
            Normal(
                torch.zeros(event_shape, device=device),
                torch.ones(event_shape, device=device),
            ),
            1,
        )
    # Generate normalizing flow
    if isinstance(event_shape, int):
        dim = event_shape
    elif isinstance(event_shape, Iterable):
        dim = event_shape[-1]
    else:
        raise ValueError("The eventshape must either be an Integer or a Iterable.")

    flows = []
    for i in range(num_transforms):
        flows.append(
            get_transform(transform, dim, device=device, **kwargs).with_cache()
        )
        if permute and i < num_transforms - 1:
            permutation = torch.randperm(dim, device=device)
            flows.append(transforms.Permute(permutation))
        if batch_norm and i < num_transforms - 1:
            bn = transforms.BatchNorm(dim).to(device)
            flows.append(bn)
    flows.append(link_flow.with_cache())
    dist = TransformedDistribution(base_dist, flows)
    return dist


@register_flow_builder(name="gaussian_diag")
def gaussian_diag_flow_builder(
    event_shape: torch.Size, link_flow: TorchTransform, device: str = "cpu", **kwargs
):
    """Generates a Gaussian distribution with diagonal covariance.

    Args:
        event_shape: Shape of the events generated by the distribution.
        link_flow: Links to a specific support .
        kwargs: Hyperparameters are added here.
            loc: Initial location.
            scale: Initial triangular matrix.

    Returns:
        TransformedDistribution

    """
    if "transform" in kwargs:
        kwargs.pop("transform")
    if "base_dist" in kwargs:
        kwargs.pop("base_dist")
    if "num_transforms" in kwargs:
        kwargs.pop("num_transforms")
    return build_flow(
        event_shape,
        link_flow,
        device=device,
        transform="affine_diag",
        num_transforms=1,
        shuffle=False,
        **kwargs,
    )


@register_flow_builder(name="gaussian")
def gaussian_flow_builder(
    event_shape: torch.Size, link_flow: TorchTransform, device: str = "cpu", **kwargs
) -> TransformedDistribution:
    """Generates a Gaussian distribution.

    Args:
        event_shape: Shape of the events generated by the distribution.
        link_flow: Links to a specific support .
        device: Device on which to build.
        kwargs: Hyperparameters are added here.
            loc: Initial location.
            scale_tril: Initial triangular matrix.

    Returns:
        TransformedDistribution

    """
    if "transform" in kwargs:
        kwargs.pop("transform")
    if "base_dist" in kwargs:
        kwargs.pop("base_dist")
    if "num_transforms" in kwargs:
        kwargs.pop("num_transforms")
    return build_flow(
        event_shape,
        link_flow,
        device=device,
        transform="affine_tril",
        shuffle=False,
        num_transforms=1,
        **kwargs,
    )


@register_flow_builder(name="maf")
def masked_autoregressive_flow_builder(
    event_shape: torch.Size, link_flow: TorchTransform, device: str = "cpu", **kwargs
) -> TransformedDistribution:
    """Generates a masked autoregressive flow

    Args:
        event_shape: Shape of the events generated by the distribution.
        link_flow: Links to a specific support.
        device: Device on which to build.
        num_transforms: Number of normalizing flows that are concatenated.
        permute: Permute dimension after each layer. This may helpfull for
            autoregressive or coupling nets.
        batch_norm: Perform batch normalization.
        base_dist: Base distribution. If `None` then a standard Gaussian is used.
        kwargs: Hyperparameters are added here.
            hidden_dims: The dimensionality of the hidden units per layer.
            skip_connections: Whether to add skip connections from the input to the
                output.
            nonlinearity: The nonlinearity to use in the feedforward network such as
                torch.nn.ReLU().
            log_scale_min_clip: The minimum value for clipping the log(scale) from
                the autoregressive NN
            log_scale_max_clip: The maximum value for clipping the log(scale) from
                the autoregressive NN
            sigmoid_bias: A term to add the logit of the input when using the stable
                tranform.
            stable: When true, uses the alternative "stable" version of the transform.
                Yet this version is also less expressive.

    Returns:
        TransformedDistribution

    """
    if "transform" in kwargs:
        kwargs.pop("transform")
    return build_flow(
        event_shape,
        link_flow,
        transform="affine_autoregressive",
        device=device,
        **kwargs,
    )


@register_flow_builder(name="nsf")
def spline_autoregressive_flow_builder(
    event_shape: torch.Size, link_flow: TorchTransform, device: str = "cpu", **kwargs
) -> TransformedDistribution:
    """Generates an autoregressive neural spline flow.

    Args:
        event_shape: Shape of the events generated by the distribution.
        link_flow: Links to a specific support .
        num_transforms: Number of normalizing flows that are concatenated.
        permute: Permute dimension after each layer. This may helpfull for
            autoregressive or coupling nets.
        batch_norm: Perform batch normalization.
        base_dist: Base distribution. If `None` then a standard Gaussian is used.
        kwargs: Hyperparameters are added here.
            hidden_dims: The dimensionality of the hidden units per layer.
            skip_connections: Whether to add skip connections from the input to the
                output.
            nonlinearity: The nonlinearity to use in the feedforward network such as
                torch.nn.ReLU().
            count_bins: The number of segments comprising the spline.
            bound: The quantity `K` determining the bounding box.
            order: One of [`linear`, `quadratic`] specifying the order of the spline.

    Returns:
        TransformedDistribution

    """
    if "transform" in kwargs:
        kwargs.pop("transform")
    return build_flow(
        event_shape,
        link_flow,
        transform="spline_autoregressive",
        device=device,
        **kwargs,
    )


@register_flow_builder(name="mcf")
def coupling_flow_builder(
    event_shape: torch.Size, link_flow: TorchTransform, device: str = "cpu", **kwargs
) -> TransformedDistribution:
    """Generates a affine coupling flow.

    Args:
        event_shape: Shape of the events generated by the distribution.
        link_flow: Links to a specific support.
        num_transforms: Number of normalizing flows that are concatenated.
        permute: Permute dimension after each layer. This may helpfull for
            autoregressive or coupling nets.
        batch_norm: Perform batch normalization.
        base_dist: Base distribution. If `None` then a standard Gaussian is used.
        kwargs: Hyperparameters are added here.
            hidden_dims: The dimensionality of the hidden units per layer.
            skip_connections: Whether to add skip connections from the input to the
                output.
            nonlinearity: The nonlinearity to use in the feedforward network such as
                torch.nn.ReLU().
            log_scale_min_clip: The minimum value for clipping the log(scale) from
                the autoregressive NN
            log_scale_max_clip: The maximum value for clipping the log(scale) from
                the autoregressive NN
            split_dim : The dimension to split the input on for the coupling transform.

    Returns:
        TransformedDistribution

    """
    if "transform" in kwargs:
        kwargs.pop("transform")
    return build_flow(
        event_shape, link_flow, device=device, transform="affine_coupling", **kwargs
    )


@register_flow_builder(name="scf")
def spline_coupling_flow_builder(
    event_shape: torch.Size, link_flow: TorchTransform, device: str = "cpu", **kwargs
) -> TransformedDistribution:
    """Generates an spline coupling flow. Implementation is based on [1], we do not
    implement affine transformations using LU decomposition as proposed in [2].

    Args:
        event_shape: Shape of the events generated by the distribution.
        link_flow: Links to a specific support .
        num_transforms: Number of normalizing flows that are concatenated.
        permute: Permute dimension after each layer. This may helpfull for
            autoregressive or coupling nets.
        batch_norm: Perform batch normalization.
        base_dist: Base distribution. If `None` then a standard Gaussian is used.
        kwargs: Hyperparameters are added here.
            hidden_dims: The dimensionality of the hidden units per layer.
            nonlinearity: The nonlinearity to use in the feedforward network such as
                torch.nn.ReLU().
            count_bins: The number of segments comprising the spline.
            bound: The quantity `K` determining the bounding box.
            order: One of [`linear`, `quadratic`] specifying the order of the spline.
            split_dim : The dimension to split the input on for the coupling transform.

    Returns:
        TransformedDistribution

    References:
        [1] Invertible Generative Modeling using Linear Rational Splines, Hadi M.
            Dolatabadi, Sarah Erfani, Christopher Leckie, 2020,
            https://arxiv.org/pdf/2001.05168.pdf.
        [2] Neural Spline Flows, Conor Durkan, Artur Bekasov, Iain Murray, George
            Papamakarios, 2019, https://arxiv.org/pdf/1906.04032.pdf.


    """
    if "transform" in kwargs:
        kwargs.pop("transform")
    return build_flow(
        event_shape, link_flow, device=device, transform="spline_coupling", **kwargs
    )
