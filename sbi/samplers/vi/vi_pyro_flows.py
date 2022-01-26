import torch
from torch import nn
from typing import Iterable, Callable, Optional
from torch.distributions import Distribution, Normal, Independent


from pyro.distributions import transforms
from pyro.nn import AutoRegressiveNN, DenseNN

from .vi_utils import get_modules, get_parameters, filter_kwrags_for_func

# Supported transforms and flows are registered here i.e. associated with a name

_TRANSFORMS = {}
_TRANSFORMS_INITS = {}
_FLOW_BUILDERS = {}


def register_transform(
    cls: Optional[object] = None,
    name: Optional[str] = None,
    inits: Callable = lambda *args, **kwargs: {},
):
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


def get_transform(name: str, dim, **kwargs):
    name = name.lower()
    transform = _TRANSFORMS[name]
    overwritable_kwargs = filter_kwrags_for_func(transform.__init__, kwargs)
    args, default_kwargs = _TRANSFORMS_INITS[name](dim, **kwargs)
    kwargs = {**default_kwargs, **overwritable_kwargs}
    return _TRANSFORMS[name](*args, **kwargs)


def register_flow_builder(cls=None, name=None):
    def _register(cls):
        if name is None:
            cls_name = cls.__name__
        else:
            cls_name = name
        if cls_name in _FLOW_BUILDERS:
            raise ValueError(f"The transform {cls_name} is already registered")
        else:
            _FLOW_BUILDERS[cls_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_flow_builder(name, event_shape, link_flow, **kwargs):
    builder = _FLOW_BUILDERS[name]
    return builder(event_shape, link_flow, **kwargs)


# Autoregressive transforms


def init_affine_autoregressive(dim, **kwargs):
    hidden_dims = kwargs.pop("hidden_dims", [5 * dim + 5])
    arn = AutoRegressiveNN(dim, hidden_dims)
    return [arn], {"log_scale_min_clip": -3.0}


def init_spline_autoregressive(dim, **kwargs):
    hidden_dims = kwargs.pop("hidden_dims", [dim * 10, dim * 10])
    count_bins = 10
    order = "linear"
    bound = 3
    if order == "linear":
        param_dims = [count_bins, count_bins, (count_bins - 1), count_bins]
    else:
        param_dims = [count_bins, count_bins, (count_bins - 1)]
    nn = AutoRegressiveNN(dim, hidden_dims, param_dims)
    return [dim, nn], {"count_bins": count_bins, "bound": bound, "order": order}


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


class DenseNN(DenseNN):
    """More powerfull dense net compared to the pyro implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [
            nn.Sequential(
                nn.Linear(self.input_dim + self.context_dim, self.hidden_dims[0]),
                nn.ReLU(),
            )
        ]
        for hidden_dim in self.hidden_dims:
            layers += [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())]

        layers += [
            nn.Sequential(nn.Linear(self.hidden_dims[-1], self.output_multiplier))
        ]
        self.layers = nn.ModuleList(layers)


def init_affine_coupling(dim, **kwargs):
    assert dim > 1, "In 1d this would be equivalent to affine flows, use them!"
    split_dim = dim // 2
    hidden_dims = kwargs.pop("hidden_dims", [5 * dim + 5, 5 * dim + 5])
    arn = DenseNN(split_dim, hidden_dims)
    return [split_dim, arn], {"log_scale_min_clip": -3.0}


def init_spline_coupling(dim, **kwargs):
    assert dim > 1, "In 1d this would be equivalent to affine flows, use them!"
    hidden_dims = kwargs.pop("hidden_dims", [dim * 10, dim * 10])
    split_dim = dim // 2
    count_bins = 10
    order = "linear"
    bound = 10
    if order == "linear":
        param_dims = [count_bins, count_bins, (count_bins - 1), count_bins]
    else:
        param_dims = [count_bins, count_bins, (count_bins - 1)]
    nn = DenseNN(split_dim, hidden_dims, param_dims)
    return [dim, split_dim, nn], {
        "count_bins": count_bins,
        "bound": bound,
        "order": order,
    }


register_transform(
    transforms.AffineCoupling, "affine_coupling", inits=init_affine_coupling
)

register_transform(
    transforms.SplineCoupling, "spline_coupling", inits=init_spline_coupling
)


@register_transform(
    name="affine_diag",
    inits=lambda dim, **kwargs: (
        [],
        {"loc": torch.zeros(dim), "scale": torch.ones(dim)},
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

    def log_abs_jacobian_diag(self, x, y):
        return self.scale


@register_transform(
    name="affine_tril",
    inits=lambda dim, **kwargs: (
        [],
        {"loc": torch.zeros(dim), "scale_tril": torch.eye(dim)},
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
    """This is TransformedDistribution with the capability to return parameters!"""

    __doc__ += torch.distributions.TransformedDistribution.__doc__

    def parameters(self):
        for t in self.transforms:
            yield from get_parameters(t)

    def modules(self):
        for t in self.transforms:
            yield from get_modules(t)


def build_flow(
    event_shape: torch.Size,
    link_flow: transforms.Transform,
    num_flows: int = 5,
    transform: str = "affine_autoregressive",
    permute: bool = True,
    batch_norm: bool = False,
    base_dist: Distribution = None,
    **kwargs,
) -> TransformedDistribution:
    f"""Generates a Transformed Distribution where the base_dist is transformed by
       num_flows normalizing flows of specified type.



    Args:
        event_shape: Dimension of the events generated by the distribution.
        link_flow: Links to a specific support .
        num_flows: Number of normalizing flows that are concatenated.
        type: The type of normalizing flow. Should be one of {_TRANSFORMS.keys()}
        permute: Permute dimension after each layer. This may helpfull for
        autoregressive or coupling nets.
        batch_norm: Perform batch normalization.
        base_dist: Base distribution.
        kwargs
    Returns:
        TransformedDistribution

    """
    # Some transforms increase dimension by decreasing the degrees of freedom e.g.
    # SoftMax.
    additional_dim = len(link_flow(torch.zeros(event_shape))) - torch.tensor(
        event_shape
    )
    event_shape = torch.Size(torch.tensor(event_shape) - additional_dim)
    # Base distribution is standard normal if not specified
    if base_dist is None:
        base_dist = Independent(
            Normal(torch.zeros(event_shape), torch.ones(event_shape)),
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
    for i in range(num_flows):
        flows.append(get_transform(transform, dim, **kwargs).with_cache())
        if permute and i < num_flows - 1:
            flows.append(transforms.permute(dim).with_cache())
        if batch_norm and i < num_flows - 1:
            flows.append(transforms.batchnorm(dim))
    flows.append(link_flow.with_cache())
    dist = TransformedDistribution(base_dist, flows)
    return dist


@register_flow_builder(name="gaussian_diag")
def gaussian_diag_flow_builder(event_shape, link_flow, **kwargs):
    if "transform" in kwargs:
        kwargs.pop("transform")
    if "num_flows" in kwargs:
        kwargs.pop("num_flows")
    return build_flow(
        event_shape,
        link_flow,
        transform="affine_autoregressive",
        num_flows=1,
        shuffle=False,
        **kwargs,
    )


@register_flow_builder(name="gaussian")
def gaussian_flow_builder(event_shape, link_flow, **kwargs):
    if "transform" in kwargs:
        kwargs.pop("transform")
    if "num_flows" in kwargs:
        kwargs.pop("num_flows")
    return build_flow(
        event_shape,
        link_flow,
        transform="affine_tril",
        shuffle=False,
        num_flows=1,
        **kwargs,
    )


@register_flow_builder(name="maf")
def masked_autoregressive_flow_builder(event_shape, link_flow, **kwargs):
    if "transform" in kwargs:
        kwargs.pop("transform")
    return build_flow(
        event_shape, link_flow, transform="affine_autoregressive", **kwargs
    )


@register_flow_builder(name="nsf")
def spline_autoregressive_flow_builder(event_shape, link_flow, **kwargs):
    if "transform" in kwargs:
        kwargs.pop("transform")
    return build_flow(
        event_shape, link_flow, transform="spline_autoregressive", **kwargs
    )


@register_flow_builder(name="mcf")
def coupling_flow_builder(event_shape, link_flow, **kwargs):
    if "transform" in kwargs:
        kwargs.pop("transform")
    return build_flow(event_shape, link_flow, transform="affine_coupling", **kwargs)


@register_flow_builder(name="scf")
def spline_coupling_flow_builder(event_shape, link_flow, **kwargs):
    if "transform" in kwargs:
        kwargs.pop("transform")
    return build_flow(event_shape, link_flow, transform="spline_coupling", **kwargs)
