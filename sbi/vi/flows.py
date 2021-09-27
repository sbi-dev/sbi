import torch
import numpy as np
from torch import nn
from torch.distributions import Distribution, Normal, Independent
from torch.distributions.constraints import Constraint, real
from torch.distributions.transforms import Transform, ComposeTransform
from torch.distributions import biject_to

from typing import Optional, Iterable

from scipy.optimize import fsolve

from pyro.distributions import transforms, constraints, TransformModule
from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.nn import AutoRegressiveNN, DenseNN

from torchdiffeq import odeint_adjoint, odeint

TYPES = [
    "planar",
    "radial",
    "sylvester",
    "polynomial",
    "affine_diag",
    "affine_coupling",
    "affine_autoregressive",
    "neural_autoregressive",
    "block_autoregressive",
    "spline",
    "spline_coupling",
    "spline_autoregressive",
    "neural_ode",
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


def _inverse(self, y, xtol=1e-4):
    """ Numerical root finding algorithm to evaluate the log probability if the inverse
    is analytically not tractable. """
    with torch.no_grad():
        shape = tuple(y.shape)

        def f(x):
            x = torch.from_numpy(x).reshape(shape).float()
            return (self(x) - y).flatten().numpy()

        x = torch.tensor(
            fsolve(f, np.random.randn(*shape).flatten(), xtol=xtol)
        ).float()
    x = x.reshape(shape)

    return x


def _inverse_batched(self, y, batch_size=20):
    """ Batched inverse. For large batches of data it is more efficient to process it in
   small batches, because a single 'hard' point can slow down all other."""
    shape = y.shape
    batch_size = min(batch_size, shape[0])
    xs = []
    y = y.reshape(-1, shape[-1])
    for i in range(0, shape[0], batch_size):
        y_i = y[i : i + batch_size, :]
        x_i = _inverse(self, y_i)
        xs.append(x_i)
    x = torch.vstack(xs).reshape(shape)
    # For batched we need another forward pass to get correct determinant in cache
    self(y)
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
    link_flow = biject_to(support)
    additional_dim = len(link_flow(torch.zeros(event_shape))) - torch.tensor(
        event_shape
    )
    event_shape = torch.Size(torch.tensor(event_shape) - additional_dim)
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


def trace_df_dz(f, z):
    trace = 0.0
    for i in range(z.shape[-1]):
        trace += (
            torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0]
            .contiguous()[:, i]
            .contiguous()
        )

    return trace.contiguous()


class ODEnet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=20, num_layers=2, normalization=nn.Identity
    ):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.ELU())
        self.input_embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU())
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
                for _ in range(num_layers)
            ]
        )
        self.out_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, t, x):
        h = self.input_embed(x)
        h = h + self.time_embed(t.reshape(-1, 1))
        for layer in self.layers:
            h = layer(h)
        out = self.out_layer(h)
        return out


class NeuralODETransform(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = +1

    def __init__(
        self,
        ODEnet,
        T=1.0,
        t0=0.0,
        atol=1e-2,
        rtol=1e-2,
        solver="dopri5",
        options=dict(),
        adjoint=False,
    ):
        """This is a continous normalizing flow, which use a neural ODE transform.
        
        
        
        Args:
            ODEnet: An network that inputs time and x and outputs dx.
            T: End time of the ODE (default T=1)
            t0: Start time of the ODE (default t0=0)
            atol: Absolute tolerance of the ODE solver
            rtol: Relative tolerance of the ODE solver
            solver: The ODE solver (one implemented in torchdiffeq)
            options: Further options of odeint
            adjoint: If the adjoint method should be used to compute gradients.
        
        """
        super().__init__(cache_size=1)
        self.net = ODEnet
        self.T = float(T)
        self.t0 = float(t0)
        self.atol = atol
        self.rtol = rtol
        self.solver = solver
        self.adjoint = adjoint
        self.options = options

    def _call(self, x):
        logp_diff_t0 = torch.zeros(x.shape[0], 1)
        if not self.adjoint:
            y, logP_diff_T = odeint(
                self.ode_func,
                (x, logp_diff_t0),
                torch.tensor([self.t0, self.T]),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.options,
            )
        else:
            y, logP_diff_T = odeint_adjoint(
                self.ode_func,
                (x, logp_diff_t0),
                torch.tensor([self.t0, self.T]),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                adjoint_params=self.net.parameters(),
                options=self.options,
            )
        self._cached_logP_diff_T = -logP_diff_T[-1].flatten()
        return y[-1]

    def _inverse(self, y):
        logp_diff_t0 = torch.zeros(y.shape[0], 1)
        if not self.adjoint:
            x, logP_diff_T = odeint(
                self.ode_func,
                (y, logp_diff_t0),
                torch.tensor([self.T, self.t0]),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.options,
            )
        else:
            x, logP_diff_T = odeint_adjoint(
                self.ode_func,
                (y, logp_diff_t0),
                torch.tensor([self.T, self.t0]),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                adjoint_params=self.net.parameters(),
                options=self.options,
            )
        self._cached_logP_diff_T = logP_diff_T[-1].flatten()
        return x[-1]

    def ode_func(self, t, states):
        z = states[0]
        logp_z = states[1]
        batch_size = z.shape[0]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt = self.net(t, z)
            dlogp_z_dt = -trace_df_dz(dz_dt, z)
        return (dz_dt, dlogp_z_dt)

    def log_abs_det_jacobian(self, x, y):
        return self._cached_logP_diff_T


def neural_ode_transform(dim, **kwargs):
    hidden_dim = 5 * dim + 10
    net = ODEnet(dim, hidden_dim=hidden_dim)
    t = NeuralODETransform(net, **kwargs)
    return t


def flow_block(dim, type, **kwargs):
    r""" Gives pyro flow of specified type.
    Args:
        dim: Event shape of input. 
        type: Type, should be one of 
    
    Returns:
        pyro.distributions.transform: Transform object of specified type
    
    """
    inverse = kwargs.pop("inverse", False)
    if type.lower() == "planar":
        flow = transforms.planar(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "radial":
        flow = transforms.radial(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "sylvester":
        flow = transforms.sylvester(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "polynomial":
        flow = transforms.polynomial(dim, **kwargs)
    elif type.lower() == "affine_diag":
        # This is equivalent to a gaussian with diagonal covariance (up to support link transforms)
        flow = AffineTransform(torch.zeros(dim), torch.ones(dim))
    elif type.lower() == "affine_tril":
        # This is equivalent to a gaussian with full covariance (up to support link transforms)
        flow = LowerCholeskyAffine(torch.zeros(dim), torch.eye(dim))
    elif type.lower() == "affine_coupling":
        flow = transforms.affine_coupling(dim, **kwargs)
    elif type.lower() == "affine_autoregressive":
        hidden_dims = kwargs.get("hidden_dims", None)
        if hidden_dims is None:
            hidden_dims = [5 * dim + 5]
        arn = AutoRegressiveNN(dim, hidden_dims)
        flow = AffineAutoregressive(arn, log_scale_min_clip=-3.0)
    elif type.lower() == "neural_autoregressive":
        flow = transforms.neural_autoregressive(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "block_autoregressive":
        flow = transforms.block_autoregressive(dim, **kwargs)
        flow._inverse = lambda x: _inverse_batched(flow, x)
    elif type.lower() == "spline":
        flow = transforms.spline(dim, **kwargs)
    elif type.lower() == "spline_coupling":
        # Linear or quadratic rational splines
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
        # Linear or quadratic rational spline transform
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
    elif type.lower() == "neural_ode":
        flow = neural_ode_transform(dim, **kwargs)
    else:
        raise NotImplementedError()

    if inverse:
        # That is relevant for e.g. Autoregressive Flows or Flows where the inverse is
        # numerically tractable.
        flow = flow.inv
    return flow
