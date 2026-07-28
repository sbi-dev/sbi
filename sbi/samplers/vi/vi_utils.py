# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import (
    Callable,
    Dict,
    Iterable,
    Optional,
    TypeVar,
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
from torch.nn import Module

from sbi.neural_nets.estimators.zuko_flow import ZukoUnconditionalFlow
from sbi.sbi_types import Shape, TorchTransform, VariationalDistribution
from sbi.utils.torchutils import _base_recursor

_CopyableT = TypeVar("_CopyableT")


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


def filter_kwargs_for_func(f: Callable, kwargs: Dict) -> Dict:
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
        assert isinstance(q.parameters, Callable), "The parameters must be callable"  # type: ignore[union-attr]
        parameters = q.parameters()  # type: ignore[union-attr]
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
        assert isinstance(q.modules, Callable), "The parameters must be callable"  # type: ignore[union-attr]
        modules = q.modules()  # type: ignore[union-attr]
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


class AdaptedVariationalDistribution(Distribution):
    """Wraps a user-supplied variational distribution for the `DivergenceOptimizer`s.

    Makes the support match the prior's, and defines `parameters` and `modules` as
    methods on the class so that the distribution can be pickled.

    Must offer the same methods as a `TransformedDistribution` and no more: it is not an
    `nn.Module` and defines neither `to` nor `sample_and_log_prob`, because
    `DivergenceOptimizer` checks for all three and behaves differently if they exist.
    """

    arg_constraints: Dict = {}

    def __init__(
        self,
        q: Distribution,
        prior: Distribution,
        link_transform: Callable,
        parameters: Optional[Iterable] = None,
        modules: Optional[Iterable] = None,
    ) -> None:
        """
        Args:
            q: The user's variational distribution.
            prior: Prior, used only to compare supports.
            link_transform: Applied when `q`'s support differs from the prior's.
            parameters: Trainable tensors, for a `q` that does not expose them itself.
                Required if the tensors live anywhere other than `q` or, for a
                `TransformedDistribution`, its base distribution.
            modules: Modules, for a `q` that does not expose them itself.

        Raises:
            ValueError: If no trainable tensors can be found, or if some sit somewhere
                sbi does not look and were not passed as `parameters`.
        """
        self._user_parameters = list(parameters) if parameters is not None else []
        self._user_modules = list(modules) if modules is not None else []

        self._source = q
        if not self._user_parameters and not hasattr(q, "parameters"):
            # A `TransformedDistribution` holds its trainable tensors on the base.
            base = getattr(q, "base_dist", None)
            if base is None or not hasattr(base, "parameters"):
                raise ValueError(
                    "The variational distribution has no parameters to optimize. Pass "
                    "the trainable tensors as `parameters` (and any modules as "
                    "`modules`)."
                )
            if any(hasattr(t, "parameters") for t in getattr(q, "transforms", [])):
                raise ValueError(
                    "The variational distribution keeps trainable tensors in its "
                    "transforms, which sbi does not collect on its own. Pass all of "
                    "them as `parameters` (and any modules as `modules`), so that none "
                    "are silently left out of training."
                )
            self._source = base

        if hasattr(prior, "support") and q.support != prior.support:
            if isinstance(q, TransformedDistribution):
                q = TransformedDistribution(
                    q.base_dist, list(q.transforms) + [link_transform]
                )
            else:
                q = TransformedDistribution(q, [link_transform])
        self._q = q

        super().__init__(
            batch_shape=q.batch_shape,
            event_shape=q.event_shape,
            validate_args=False,
        )

    def parameters(self) -> Iterable:
        """Yield the trainable tensors, as given or as `q` exposes them."""
        if self._user_parameters:
            return iter(self._user_parameters)
        return self._source.parameters()  # type: ignore[attr-defined]

    def modules(self) -> Iterable:
        """Yield the modules, as given or as `q` exposes them."""
        if self._user_modules:
            return iter(self._user_modules)
        if hasattr(self._source, "modules"):
            return self._source.modules()  # type: ignore[attr-defined]
        return iter(())

    @property
    def support(self):  # type: ignore[override]
        return self._q.support

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        return self._q.has_rsample

    def sample(self, sample_shape: Shape = torch.Size()) -> Tensor:
        return self._q.sample(sample_shape)

    def rsample(self, sample_shape: Shape = torch.Size()) -> Tensor:
        return self._q.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return self._q.log_prob(value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._q})"


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


def detach_and_deepcopy(obj: _CopyableT) -> _CopyableT:
    """Deep-copy `obj`, detaching any non-leaf tensors it caches first.

    Only needed for `torch.distributions` objects, whose constructors call `.expand()`
    and cache non-leaf tensors that `deepcopy` refuses. `nn.Module`-based families hold
    only leaf parameters and copy fine.

    Args:
        obj: Object to copy.

    Returns:
        An independent copy of `obj`.
    """
    detach_all_non_leaf_tensors(obj)
    return deepcopy(obj)
