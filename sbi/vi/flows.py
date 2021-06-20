from pyro.distributions.transforms.spline_coupling import SplineCoupling
import torch
from torch import nn
from torch.distributions import Distribution, Normal, Independent, transforms
from torch.distributions.constraints import Constraint, real
from torch.distributions.transforms import Transform, ComposeTransform
from torch.distributions import biject_to

from typing import Optional, Iterable

from scipy.optimize import fsolve

from pyro.distributions import transforms
from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.nn import AutoRegressiveNN, DenseNN

TYPES = [
    "planar",
    "radial",
    "sylvester",
    "affine_diag",
    "affine_coupling",
    "affine_autoregressive",
    "neural_autoregressive",
    "block_autoregressive",
    "spline",
    "spline_coupling",
    "spline_autoregressive",
]


def get_parameters(t: Transform):
    """ Recursive helper function to determine all possible parameters """
    if hasattr(t, "parameters"):
        yield from t.parameters()
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_parameters(part)
    else:
        pass


def get_modules(t: Transform):
    """ Recursive helper function to determine all modules """
    if isinstance(t, nn.Module):
        yield t
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_modules(part)
    else:
        pass


def _inverse(self, y):
    """ Numerical root finding algorithm to evaluate the log probability if the inverse
    is analytically not tractable. """
    with torch.no_grad():
        shape = y.shape

        def f(x):
            x = torch.from_numpy(x).reshape(shape).float()
            return (self(x) - y).flatten().numpy()

        x = torch.tensor(fsolve(f, np.zeros(shape).flatten(), xtol=1e-5)).float()
    x = x.reshape(shape)

    return x


def _inverse_batched(self, y, batch_size=20):
    """ Batched inverse. For large batches of data it is more efficient to process it in
   small batches. """
    shape = y.shape
    xs = []
    y = y.reshape(-1, y.shape[-1])
    for i in range(batch_size, len(y) + batch_size, batch_size):
        y_i = y[i - batch_size : i, :]
        x_i = _inverse(self, y_i)
        xs.append(x_i)
    x = torch.stack(xs).reshape(shape)

    return x


class TransformedDistribution(torch.distributions.TransformedDistribution):
    """ This is TransformedDistribution with the capability to return parameters!"""

    def parameters(self):
        for t in self.transforms:
            yield from get_parameters(t)

    def modules(self):
        for t in self.transforms:
            yield from get_modules(t)


class AffineTransform(transforms.AffineTransform):
    """ Trainable version of an Affine transform. This can be used to get diagonal
    gaussian approximation """

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


class LowerCholeskyAffine(transforms.LowerCholeskyAffine):
    """ Trainable version of a Lower Cholesky Affine transform. This can be used to get
full Gaussian approximations."""

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
        """ This modification allows batched scale_tril matrices. """
        return self.log_abs_jacobian_diag(x, y).sum(-1)

    def log_abs_jacobian_diag(self, x, y):
        """ This returns the full diagonal which is necessary to compute conditionals """
        dim = self.scale_tril.dim()
        return torch.diagonal(self.scale_tril, dim1=dim - 2, dim2=dim - 1).log()


class AffineAutoregressive(transforms.AffineAutoregressive):
    """ Modification that also returns the jacobian diagonal. """

    def log_abs_jacobian_diag(self, x, y):
        """
        Calculates the diagonal of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            self(x)

        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        elif not self.stable:
            _, log_scale = self.arn(x)
            log_scale = clamp_preserve_gradients(
                log_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )
        else:
            _, logit_scale = self.arn(x)
            log_scale = self.logsigmoid(logit_scale + self.sigmoid_bias)
        return log_scale


class SplineAutoregressive(transforms.SplineAutoregressive):
    """ Modification that also returns the jacobian diagonal. """

    def log_abs_jacobian_diag(self, x, y):
        """
        Calculates the diagonal of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            self(x)

        return self._cache_log_detJ


def build_flow(
    event_shape: torch.Size,
    support: Constraint = real,
    num_flows: int = 5,
    type: str = "affine_autoregressive",
    permute: bool = True,
    batch_norm: bool = False,
    base_dist: Distribution = None,
    **kwargs,
) -> TransformedDistribution:
    f"""Generates a Transformed Distribution where the base_dist is transformed by
       num_flows normalizing flows of specified type.
    
    
    
    Args:
        event_shape: Dimension of the events generated by the distribution.
        support: The support of the distribution.
        num_flows: Number of normalizing flows that are concatenated.
        type: The type of normalizing flow. Should be one of {TYPES}
        permute: Permute dimension after each layer. This may helpfull for
        autoregressive or coupling nets.
        batch_norm: Perform batch normalization.
        base_dist: Base distribution.
        kwargs
    Returns:
        TransformedDistribution
    
    """

    # Base distribution is standard normal if not specified
    if base_dist is None:
        base_dist = Independent(
            Normal(torch.zeros(event_shape), torch.ones(event_shape)), 1,
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
        flows.append(flow_block(dim, type, **kwargs).with_cache())
        if permute and i < num_flows - 1:
            flows.append(transforms.permute(dim).with_cache())
        if batch_norm and i < num_flows - 1:
            flows.append(transforms.batchnorm(dim))
    link_flow = biject_to(support)
    flows.append(link_flow.with_cache())
    dist = TransformedDistribution(base_dist, flows)
    return dist


class DenseNN(DenseNN):
    """ More powerfull dense net compared to the pyro implementation """

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


def flow_block(dim, type, **kwargs):
    r""" Gives pyro flow of specified type.
    Args:
        dim: Event shape of input. 
        type: Type, should be one of 
    
    Returns:
        pyro.distributions.transform: Transform object of specified type
    
    """
    if type.lower() == "planar":
        flow = transforms.planar(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "radial":
        flow = transforms.radial(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "sylvester":
        flow = transforms.sylvester(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "affine_diag":
        flow = AffineTransform(torch.zeros(dim), torch.ones(dim))
    elif type.lower() == "affine_tril":
        flow = LowerCholeskyAffine(torch.zeros(dim), torch.eye(dim))
    elif type.lower() == "affine_coupling":
        flow = transforms.affine_coupling(dim, **kwargs)
    elif type.lower() == "affine_autoregressive":
        inverse = kwargs.get("inverse", False)  # IAF or MAF
        hidden_dims = kwargs.get("hidden_dims", None)
        if hidden_dims is None:
            hidden_dims = [5 * dim + 5]
        arn = AutoRegressiveNN(dim, hidden_dims)
        flow = AffineAutoregressive(arn, log_scale_min_clip=-3.0)
        if inverse:
            flow = flow.inv()
    elif type.lower() == "neural_autoregressive":
        flow = transforms.neural_autoregressive(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "block_autoregressive":
        flow = transforms.block_autoregressive(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "spline":
        flow = transforms.spline(dim, **kwargs)
    elif type.lower() == "spline_coupling":
        split_dim = kwargs.get("split_dim", dim // 2)
        hidden_dims = kwargs.get("hidden_dims", [dim * 20, dim * 20])
        count_bins = kwargs.get("count_bins", 8)
        order = kwargs.get("order", "linear")
        bound = kwargs.get("bound", 3.0)
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
        nn = DenseNN(split_dim, hidden_dims, param_dims)
        flow = transforms.SplineCoupling(
            dim, split_dim, nn, count_bins, bound=bound, order=order
        )
    elif type.lower() == "spline_autoregressive":
        inverse = kwargs.get("inverse", False)
        hidden_dims = kwargs.get("hidden_dims", [dim * 10, dim * 10])
        count_bins = kwargs.get("count_bins", 8)
        order = kwargs.get("order", "linear")
        bound = kwargs.get("bound", 3.0)
        if order == "linear":
            param_dims = [count_bins, count_bins, (count_bins - 1), count_bins]
        else:
            param_dims = [count_bins, count_bins, (count_bins - 1)]
        nn = AutoRegressiveNN(dim, hidden_dims, param_dims)
        flow = transforms.SplineAutoregressive(
            dim, nn, count_bins, bound=bound, order=order
        )
        if inverse:
            flow = flow.inv()
    else:
        raise NotImplementedError()
    return flow
