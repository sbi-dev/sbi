# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, Tuple, Union

import pyro.distributions
import torch
import torch.distributions.constraints as constraints
from pyro import poutine as poutine
from torch.distributions import biject_to

from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.neural_nets.ratio_estimators import RatioEstimator


def get_transforms(model: Callable, *model_args: Any, **model_kwargs: Any):
    """Get automatic transforms to unbounded space

    Args:
        model: Pyro model
        model_args: Arguments passed to model
        model_args: Keyword arguments passed to model

    Example:
        ```python
        def prior():
            return pyro.sample("theta", pyro.distributions.Uniform(0., 1.))

        transform_to_unbounded = get_transforms(prior)["theta"]
        ```
    """
    transforms = {}

    model_trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)

    for name, node in model_trace.iter_stochastic_nodes():
        if "fn" in node:
            fn = node["fn"]
            transforms[name] = biject_to(fn.support).inv

    return transforms


class EstimatorDistribution(pyro.distributions.TorchDistribution):
    """
    Base class for a conditioned `sbi` estimator wrapped as a Pyro distribution.
    """

    def __init__(self, estimator: Any, condition: torch.Tensor):
        self._estimator = estimator
        condition_shape, event_shape = self._get_condition_and_event_shapes()
        self._check_condition_shape(condition)
        self._condition = condition
        super().__init__(
            batch_shape=condition.shape[: -len(condition_shape)],
            event_shape=event_shape,
        )

    def _get_condition_and_event_shapes(self) -> Tuple[torch.Size, torch.Size]:
        raise NotImplementedError

    @property
    def support(self) -> constraints.Constraint:
        # Note: this will currently be wrong for estimators with discrete or mixed
        # support (e.g. `MixedDensityEstimator`). Torch constraints currently don't
        # support mixed support, and support is primarily used for choosing
        # unconstraining transforms for priors.
        # see pytorch/pytorch#149718
        num_input_dims = len(self.event_shape)
        if num_input_dims == 0:
            # currently this branch isn't reachable by any of the estimators, but we
            # include it to be safe
            return constraints.real
        else:
            return constraints.independent(constraints.real, num_input_dims)

    @property
    def estimator(self):
        """The conditioned estimator."""
        return self._estimator

    @property
    def condition(self):
        """The conditioning parameter."""
        return self._condition

    def _check_condition_shape(self, condition: torch.Tensor):
        """Check that the shape of `condition` is compatible with the estimator."""
        condition_shape_expected, _ = self._get_condition_and_event_shapes()
        if len(condition.shape) < len(condition_shape_expected):
            raise ValueError(
                "Dimensionality of condition is too small and does not match the "
                f"expected dimensionality {len(condition_shape_expected)}. "
                "It should be compatible with condition_shape "
                f"{condition_shape_expected}."
            )
        else:
            condition_shape = condition.shape[-len(condition_shape_expected) :]
            if condition_shape != condition_shape_expected:
                raise ValueError(
                    f"Condition shape {condition_shape} is not compatible with "
                    f"estimator condition shape {condition_shape_expected}"
                )


class ConditionalDensityEstimatorDistribution(EstimatorDistribution):
    """
    A conditioned `sbi` estimator wrapped as a Pyro distribution.
    """

    def __init__(self, estimator: ConditionalDensityEstimator, condition: torch.Tensor):
        """
        Condition an `sbi` estimator and wrap it as a Pyro distribution.

        Args:
            estimator: A trained conditional estimator
            condition: The conditioning parameter for the estimator.
        """
        super().__init__(estimator, condition)
        self._condition_reshaped = self.condition.reshape(
            -1, *self.estimator.condition_shape
        )

    def _get_condition_and_event_shapes(self) -> Tuple[torch.Size, torch.Size]:
        return self.estimator.condition_shape, self.estimator.input_shape

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of `x`."""
        # ConditionalDensityEstimator expects a batch shape of (batch_dim,)
        # and sample shape of (sample_dim,)
        sample_shape = x.shape[: -len(self.event_shape) - len(self.batch_shape)]
        x = x.reshape(-1, self.batch_shape.numel(), *self.event_shape)
        lp = self.estimator.log_prob(x, condition=self._condition_reshaped)
        # reshape (sample_dim, batch_dim,) -> (*sample_shape, *batch_shape)
        return lp.reshape(*sample_shape, *self.batch_shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Generate samples from the conditioned estimator."""
        draws = self.estimator.sample(sample_shape, condition=self._condition_reshaped)
        # reshape (batch_dim,) -> batch_shape
        return draws.reshape(*sample_shape, *self.batch_shape, *self.event_shape)

    def expand(
        self, batch_shape: torch.Size, _instance=None
    ) -> "ConditionalDensityEstimatorDistribution":
        """Expand the batch shape of the distribution."""
        # because batch shape of the distribution is entirely determined by the
        # condition, we only need to expand the condition
        condition = self.condition.expand(*batch_shape, *self.estimator.condition_shape)
        return ConditionalDensityEstimatorDistribution(self.estimator, condition)


class RatioEstimatorDistribution(EstimatorDistribution):
    """
    A conditioned `sbi` ratio estimator wrapped as a Pyro distribution.
    """

    def __init__(self, estimator: RatioEstimator, condition: torch.Tensor):
        """
        Condition an `sbi` ratio estimator and wrap it as a Pyro distribution.

        Args:
            estimator: A trained ratio estimator
            condition: The conditioning parameter for the estimator.
        """
        super().__init__(estimator, condition)

    def _get_condition_and_event_shapes(self) -> Tuple[torch.Size, torch.Size]:
        return self.estimator.theta_shape, self.estimator.x_shape

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of `x`."""
        # RatioEstimator expects condition and x to have the same leading shape
        sample_shape = x.shape[: -len(self.event_shape) - len(self.batch_shape)]
        condition_shape, _ = self._get_condition_and_event_shapes()
        condition = self.condition.expand(
            *sample_shape, *self.batch_shape, *condition_shape
        )
        lp = self.estimator.unnormalized_log_ratio(theta=condition, x=x)
        return lp

    def expand(
        self, batch_shape: torch.Size, _instance=None
    ) -> "RatioEstimatorDistribution":
        """Expand the batch shape of the distribution."""
        # because batch shape of the distribution is entirely determined by the
        # condition, we only need to expand the condition
        condition = self.condition.expand(*batch_shape, *self.estimator.theta_shape)
        return RatioEstimatorDistribution(self.estimator, condition)


def to_pyro_distribution(
    estimator: Union[ConditionalDensityEstimator, RatioEstimator],
    condition: torch.Tensor,
) -> EstimatorDistribution:
    """Wrap a supported `sbi` estimator as a Pyro distribution.

    Args:
        estimator: A trained `sbi` estimator. Either a `ConditionalDensityEstimator` or
            a `RatioEstimator`.
        condition: The conditioning parameter for the estimator.

    Returns:
        A Pyro distribution wrapping the estimator.

    Raises:
        ValueError: If `estimator` is not a `ConditionalDensityEstimator` or
            `RatioEstimator`.

    Note:
        If `estimator` input has discrete or mixed support (e.g.
        `MixedDensityEstimator`), then the resulting distribution is only suitable as a
        likelihood in a Pyro model. Only distributions with continuous support may be
        used as priors in a Pyro model.
    """
    if isinstance(estimator, ConditionalDensityEstimator):
        return ConditionalDensityEstimatorDistribution(estimator, condition)
    elif isinstance(estimator, RatioEstimator):
        return RatioEstimatorDistribution(estimator, condition)
    else:
        raise ValueError(
            f"Unsupported estimator type: {type(estimator)}. "
            "Supported types are ConditionalDensityEstimator and RatioEstimator."
        )
