# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

import torch
from torch import Tensor, nn
from torch.distributions import (
    Distribution,
    Independent,
    MultivariateNormal,
    Normal,
    TransformedDistribution,
)
from torch.distributions.transforms import ComposeTransform, IndependentTransform
from torch.nn import Module

from sbi.neural_nets.estimators.zuko_flow import ZukoUnconditionalFlow
from sbi.sbi_types import Shape, TorchTransform, VariationalDistribution


class TransformedZukoFlow(nn.Module):
    """Wrapper for Zuko flows that applies a link transform to samples.

    This wrapper ensures that:
    1. Samples from the flow (in unconstrained space) are transformed to constrained
       space via link_transform
    2. log_prob accounts for the Jacobian of the transformation

    The underlying Zuko flow operates in unconstrained space, but this wrapper
    provides an interface where samples and log_probs are in constrained space
    (matching the prior's support).
    """

    def __init__(
        self,
        flow: ZukoUnconditionalFlow,
        link_transform: TorchTransform,
    ):
        """Initialize the transformed flow wrapper.

        Args:
            flow: The underlying Zuko unconditional flow (operates in unconstrained
                space).
            link_transform: Transform from unconstrained to constrained space.
                link_transform.forward maps unconstrained -> constrained.
                link_transform.inv maps constrained -> unconstrained.
        """
        super().__init__()
        self._flow = flow
        self._link_transform = link_transform

    @property
    def net(self):
        """Access the underlying flow's network (for compatibility)."""
        return self._flow.net

    def parameters(self):
        """Return the parameters of the underlying flow."""
        return self._flow.parameters()

    def sample(self, sample_shape: Shape) -> Tensor:
        """Sample from the flow and transform to constrained space.

        Args:
            sample_shape: Shape of samples to generate.

        Returns:
            Samples in constrained space with shape (*sample_shape, event_dim).
        """
        # Sample in unconstrained space
        unconstrained_samples = self._flow.sample(sample_shape)
        # Transform to constrained space
        constrained_samples = self._link_transform(unconstrained_samples)
        assert isinstance(constrained_samples, Tensor)  # Type narrowing for pyright
        return constrained_samples

    def log_prob(self, theta: Tensor) -> Tensor:
        """Compute log probability of samples in constrained space.

        Uses change of variables: log p(θ) = log q(z) + log|det(dz/dθ)|
        where z = link_transform.inv(θ) and q is the flow's distribution.

        Args:
            theta: Samples in constrained space.

        Returns:
            Log probabilities with shape (*batch_shape,).
        """
        # Transform to unconstrained space
        z = self._link_transform.inv(theta)
        assert isinstance(z, Tensor)  # Type narrowing for pyright
        # Get flow log prob in unconstrained space
        log_prob_z = self._flow.log_prob(z)
        # Add Jacobian correction for the inverse transform
        # log_abs_det_jacobian gives log|det(dz/dθ)|
        log_det_jacobian = self._link_transform.inv.log_abs_det_jacobian(theta, z)
        # Some transforms (e.g. identity) return per-dimension Jacobians,
        # while IndependentTransform returns summed Jacobians. Sum if needed.
        if log_det_jacobian.dim() > log_prob_z.dim():
            log_det_jacobian = log_det_jacobian.sum(dim=-1)
        return log_prob_z + log_det_jacobian

    def sample_and_log_prob(self, sample_shape: Shape) -> tuple[Tensor, Tensor]:
        """Sample from the flow and compute log probabilities efficiently.

        Args:
            sample_shape: Shape of samples to generate.

        Returns:
            Tuple of (samples, log_probs) where samples are in constrained space.
        """
        # Sample in unconstrained space and get log prob
        z, log_prob_z = self._flow.sample_and_log_prob(torch.Size(sample_shape))
        # Transform to constrained space
        theta = self._link_transform(z)
        assert isinstance(theta, Tensor)  # Type narrowing for pyright
        # Subtract Jacobian for forward transform (we want log p(θ) not log q(z))
        # log p(θ) = log q(z) - log|det(dθ/dz)| = log q(z) + log|det(dz/dθ)|
        log_det_jacobian = self._link_transform.log_abs_det_jacobian(z, theta)
        # Some transforms (e.g. identity) return per-dimension Jacobians,
        # while IndependentTransform returns summed Jacobians. Sum if needed.
        if log_det_jacobian.dim() > log_prob_z.dim():
            log_det_jacobian = log_det_jacobian.sum(dim=-1)
        log_prob_theta = log_prob_z - log_det_jacobian
        return theta, log_prob_theta


class LearnableGaussian(nn.Module):
    """Learnable Gaussian distribution for variational inference.

    A simple parametric variational family with learnable mean and covariance.
    Supports both full covariance (gaussian) and diagonal covariance (gaussian_diag).
    """

    def __init__(
        self,
        dim: int,
        full_covariance: bool = True,
        link_transform: Optional[TorchTransform] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the learnable Gaussian.

        Args:
            dim: Dimensionality of the distribution.
            full_covariance: If True, use full covariance matrix. If False, use
                diagonal covariance (faster, fewer parameters).
            link_transform: Optional transform to apply to samples. Maps from
                unconstrained to constrained space (matching prior support).
            device: Device to create parameters on.
        """
        super().__init__()
        self._dim = dim
        self._full_cov = full_covariance
        self._link_transform = link_transform

        # Learnable parameters - create on correct device from the start
        self.loc = nn.Parameter(torch.zeros(dim, device=device))
        if full_covariance:
            # Lower triangular matrix for Cholesky parameterization
            self.scale_tril = nn.Parameter(torch.eye(dim, device=device))
        else:
            # Log scale for numerical stability
            self.log_scale = nn.Parameter(torch.zeros(dim, device=device))

    def _base_dist(self) -> Distribution:
        """Get the base Gaussian distribution with current parameters."""
        if self._full_cov:
            return MultivariateNormal(self.loc, scale_tril=self.scale_tril)
        return Independent(Normal(self.loc, self.log_scale.exp()), 1)

    def sample(self, sample_shape: Shape) -> Tensor:
        """Sample from the distribution.

        Args:
            sample_shape: Shape of samples to generate.

        Returns:
            Samples with shape (*sample_shape, dim).
        """
        # Use sample() not rsample() - this is for inference, not training
        samples = self._base_dist().sample(sample_shape)
        if self._link_transform is not None:
            samples = self._link_transform(samples)
            assert isinstance(samples, Tensor)  # Type narrowing for pyright
        return samples

    def log_prob(self, theta: Tensor) -> Tensor:
        """Compute log probability.

        Args:
            theta: Points at which to evaluate log probability.

        Returns:
            Log probabilities with shape (*batch_shape,).
        """
        if self._link_transform is not None:
            # Transform to unconstrained space
            z = self._link_transform.inv(theta)
            assert isinstance(z, Tensor)  # Type narrowing for pyright
            log_prob_z = self._base_dist().log_prob(z)
            # Add Jacobian correction
            log_det = self._link_transform.inv.log_abs_det_jacobian(theta, z)
            if log_det.dim() > log_prob_z.dim():
                log_det = log_det.sum(dim=-1)
            return log_prob_z + log_det
        return self._base_dist().log_prob(theta)

    def sample_and_log_prob(self, sample_shape: Shape) -> tuple[Tensor, Tensor]:
        """Sample and compute log probability efficiently.

        Args:
            sample_shape: Shape of samples to generate.

        Returns:
            Tuple of (samples, log_probs).
        """
        dist = self._base_dist()
        z = dist.rsample(sample_shape)
        log_prob_z = dist.log_prob(z)

        if self._link_transform is not None:
            theta = self._link_transform(z)
            assert isinstance(theta, Tensor)  # Type narrowing for pyright
            # Adjust log_prob for the transformation
            log_det = self._link_transform.log_abs_det_jacobian(z, theta)
            if log_det.dim() > log_prob_z.dim():
                log_det = log_det.sum(dim=-1)
            return theta, log_prob_z - log_det
        return z, log_prob_z


def filter_kwrags_for_func(f: Callable, kwargs: Dict) -> Dict:
    """This function will filter a dictionary of possible arguments for arguments the
    function can use.

    Args:
        f: Function for which kwargs are filtered
        kwargs: Possible kwargs for function

    Returns:
        dict: Subset of kwargs, which the function f can take as arguments.

    """
    args = f.__code__.co_varnames
    new_kwargs = dict([(key, val) for key, val in kwargs.items() if key in args])
    return new_kwargs


def get_parameters(t: Union[TorchTransform, Module]) -> Iterable:
    """Recursive helper function which can be used to extract parameters from
    TransformedDistributions.

    Args:
        t: A TorchTransform object, which is scanned for the "parameters" attribute.

    Yields:
        Iterator[Iterable]: Generator of parameters.
    """
    if hasattr(t, "parameters"):
        yield from t.parameters()  # type: ignore
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_parameters(part)
    elif isinstance(t, IndependentTransform):
        yield from get_parameters(t.base_transform)
    else:
        pass


def get_modules(t: Union[TorchTransform, Module]) -> Iterable:
    """Recursive helper function which can be used to extract modules from
    TransformedDistributions.

    Args:
        t: A TorchTransform object, which is scanned for the "modules" attribute.

    Yields:
        Iterator[Iterable]: Generator of Modules
    """
    if isinstance(t, Module):
        yield t
    elif isinstance(t, ComposeTransform):
        for part in t.parts:
            yield from get_modules(part)
    elif isinstance(t, IndependentTransform):
        yield from get_modules(t.base_transform)
    else:
        pass


def check_parameters_modules_attribute(q: VariationalDistribution) -> None:
    """Checks a parameterized distribution object for valid `parameters` and `modules`.

    Args:
        q: Distribution object
    """

    if not hasattr(q, "parameters"):
        raise ValueError(
            """The variational distribution requires an `parameters` attribute, which
            returns an iterable of parameters"""
        )
    else:
        assert isinstance(q.parameters, Callable), "The parameters must be callable"
        parameters = q.parameters()
        assert isinstance(parameters, Iterable), (
            "The parameters return value must be iterable"
        )
        trainable = 0
        for p in parameters:
            assert isinstance(p, torch.Tensor)
            if p.requires_grad:
                trainable += 1
        assert (
            trainable > 0
        ), """Nothing to train, atleast one of the parameters must have an enabled
            gradient."""
    if not hasattr(q, "modules"):
        raise ValueError(
            """The variational distribution requires an modules attribute, which returns
            an iterable of parameters."""
        )
    else:
        assert isinstance(q.modules, Callable), "The parameters must be callable"
        modules = q.modules()
        assert isinstance(modules, Iterable), (
            "The parameters return value must be iterable"
        )
        for m in modules:
            assert isinstance(m, Module), (
                "The modules must contain PyTorch Module objects"
            )


def check_sample_shape_and_support(q: Distribution, prior: Distribution) -> None:
    """Checks the samples shape and support between variational distribution and the
    prior. Especially it checks if the shapes match and that the support between q and
    the prior matches (a property which holds for the true posterior in any case).

    Args:
        q: Variational distribution which is checked
        prior: Prior to check certain attributes which should be satisfied.

    """
    assert q.event_shape == prior.event_shape, (
        "The event shape of q must match that of the prior"
    )
    assert q.batch_shape == prior.batch_shape, (
        "The batch sahpe of q must match that of the prior"
    )

    sample_shape = torch.Size((1000,))
    samples = q.sample(sample_shape)
    samples_prior = prior.sample(sample_shape).to(samples.device)
    try:
        _ = prior.support
        has_support = True
    except (NotImplementedError, AttributeError):
        has_support = False
    if has_support:
        assert all(
            prior.support.check(samples)  # type: ignore
        ), "The support of q must match that of the prior"
    assert samples.shape == samples_prior.shape, (
        "sample_shape and event_shape or batch_shape do not match."
    )
    assert torch.isfinite(q.log_prob(samples_prior)).all(), (
        "Invalid values in logprob on prior samples."
    )
    assert torch.isfinite(prior.log_prob(samples)).all(), (
        "Invalid values in logprob on q samples."
    )


def check_variational_distribution(q: Distribution, prior: Distribution) -> None:
    """Runs all basic checks such the q is `valid`.

    Args:
        q: Variational distribution which is checked
        prior: Prior to check certain attributes which should be satisfied.

    """
    check_parameters_modules_attribute(q)
    check_sample_shape_and_support(q, prior)


def add_parameters_module_attributes(
    q: Distribution, parameters: Callable, modules: Callable
):
    """Sets the attribute `parameters` and `modules` on q

    Args:
        q: Variational distribution which is checked
        parameters: A function which returns an iterable of leaf tensors.
        modules: A function which returns an iterable of nn.Module


    """
    q.parameters = parameters  # type: ignore
    q.modules = modules  # type: ignore


def add_parameter_attributes_to_transformed_distribution(
    q: VariationalDistribution,
) -> None:
    """A function that will add `parameters` and `modules` to q automatically, if q is a
    TransformedDistribution.

    Args:
        q: Variational distribution instances TransformedDistribution.


    """

    def parameters():
        """Returns the parameters of the distribution."""
        if hasattr(q.base_dist, "parameters"):
            yield from q.base_dist.parameters()
        for t in q.transforms:
            yield from get_parameters(t)

    def modules():
        """Returns the modules of the distribution."""
        if hasattr(q.base_dist, "modules"):
            yield from q.base_dist.modules()
        for t in q.transforms:
            yield from get_modules(t)

    add_parameters_module_attributes(q, parameters, modules)


def adapt_variational_distribution(
    q: VariationalDistribution,
    prior: Distribution,
    link_transform: Callable,
    parameters: Optional[Iterable] = None,
    modules: Optional[Iterable] = None,
) -> Distribution:
    """This will adapt a distribution to be compatible with DivergenceOptimizers.
    Especially it will make sure that the distribution has parameters and that it
    satisfies obvious contraints which a posterior must satisfy i.e. the support must be
    equal to that of the prior.

    Args:
        q: Variational distribution.
        prior: Prior distribution
        theta_transform: Theta transformation.
        parameters: List of parameters.
        modules: List of modules.

    Returns:
        TransformedDistribution: Compatible variational distribution.

    """
    if parameters is None:
        parameters = []
    if modules is None:
        modules = []

    # Extract user define parameters
    def parameters_fn():
        """Returns the parameters of the distribution."""
        return parameters

    def modules_fn():
        """Returns the modules of the distribution."""
        return modules

    if isinstance(q, TransformedDistribution):
        if parameters == [] or modules_fn == []:
            add_parameter_attributes_to_transformed_distribution(q)
        else:
            add_parameters_module_attributes(q, parameters_fn, modules_fn)
        if hasattr(prior, "support") and q.support != prior.support:
            q = TransformedDistribution(q.base_dist, q.transforms + [link_transform])
    else:
        if hasattr(prior, "support") and q.support != prior.support:
            q = TransformedDistribution(q, [link_transform])
        add_parameters_module_attributes(q, parameters_fn, modules_fn)

    return q


def _base_recursor(
    obj: object,
    parent: Optional[object] = None,
    key: Optional[str] = None,
    check: Callable[..., bool] = lambda obj: False,
    action: Callable[..., object] = lambda obj: obj,
):
    """This functions is a recursive function that traverses classes i.e. Distributions
    and checks any encountered object according to `check`. If a check evaluates to
    True, then an action is applied as specified in `action`. We use it e.g. to
    move tensors to a given device.

    Args:
        obj: An object which serves as root of the traversal.
        parent: The previously traversed object.
        key: The name of the previously traversed object
        check: A function that inputs a object which is currently investigates and
            outputs a boolean. If the check evalues to True, then `action` is applied.
        action: A function that specifies an operation on an object and return an
            modified version.
    """
    if isinstance(obj, Module) and check(obj):
        action(obj)
    elif isinstance(obj, (Dict, OrderedDict)):
        for k, o in obj.items():
            if check(o):
                obj[k] = action(o)
            else:
                _base_recursor(o, parent=obj, key=k, check=check, action=action)
    elif hasattr(obj, "__dict__"):
        for k, o in obj.__dict__.items():
            if check(o):
                setattr(obj, k, action(o))
            else:
                _base_recursor(o, parent=obj, key=k, check=check, action=action)
    elif isinstance(obj, (List, Tuple, Generator)):
        new_obj = []
        for o in obj:
            if check(o):
                new_obj.append(action(o))
            else:
                _base_recursor(o, check=check, action=action)
                new_obj.append(o)
        if parent is not None and key is not None:
            setattr(parent, key, type(obj)(new_obj))  # type: ignore
    else:
        return


def detach_all_non_leaf_tensors(obj: object) -> None:
    """This detaches all non leaf tensors, which especially is required if one wants to
    create a deepcopy of the object. This is because PyTorch does not support the
    deepcopy protocol on non-leaf tensors.

    Args:
        obj: An object which is traversed for non_leaf tensors.

    """

    def check(o):
        return isinstance(o, Tensor) and o.requires_grad and not o.is_leaf

    def action(o):
        return o.detach()

    with torch.no_grad():
        _base_recursor(obj, check=check, action=action)


def make_object_deepcopy_compatible(obj: object):
    """This function overwrites the `__deepcopy__` attribute. This is required if e.g.
    the object contains non leaf PyTorch tensors.

    Args:
        obj: An object where a corresponding `__deepcopy__` attributed is added.

    """

    def __deepcopy__(memo):
        detach_all_non_leaf_tensors(obj)
        cls = obj.__class__
        result = cls.__new__(cls)
        memo[id(obj)] = result
        for k, v in obj.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    obj.__deepcopy__ = __deepcopy__  # type: ignore
    return obj
