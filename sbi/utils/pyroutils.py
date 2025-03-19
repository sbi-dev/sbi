# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from abc import abstractmethod
from typing import Any, Callable, Tuple

import pyro.distributions
import torch
import torch.distributions.constraints as constraints
from pyro import poutine as poutine
from torch.distributions import biject_to

from sbi.neural_nets.estimators.base import ConditionalEstimator


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

    @property
    def support(self):
        # TODO: check if we can infer a more specific support from the estimator
        return constraints.real  # Assume continuous values

    def __init__(self, estimator: Any, condition: torch.Tensor):
        self._estimator = estimator
        self._condition_shape, event_shape = self.get_condition_and_event_shapes(
            estimator
        )
        self._check_condition_shape(condition)
        self._condition = condition
        self._condition_reshaped = condition.reshape(-1, *self._condition_shape)
        super().__init__(
            batch_shape=condition.shape[: -len(self._condition_shape)],
            event_shape=event_shape,
        )

    @abstractmethod
    def get_condition_and_event_shapes(
        self, estimator: Any
    ) -> Tuple[torch.Size, torch.Size]:
        pass

    @abstractmethod
    def _estimator_log_prob(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """The log probability method of the estimator."""
        pass

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
        if len(condition.shape) < len(self._condition_shape):
            raise ValueError(
                "Dimensionality of condition is too small and does not match the "
                f"expected dimensionality {len(self._condition_shape)}. "
                "It should be compatible with condition_shape "
                f"{self._condition_shape}."
            )
        else:
            condition_shape = condition.shape[-len(self._condition_shape) :]
            if condition_shape != self._condition_shape:
                raise ValueError(
                    f"Condition shape {condition_shape} is not compatible with "
                    f"estimator condition shape {self._condition_shape}"
                )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Generate samples from the conditioned estimator."""
        if not hasattr(self.estimator, "sample"):
            raise NotImplementedError("Sampling is not implemented for this estimator.")
        draws = self.estimator.sample(sample_shape, condition=self._condition_reshaped)
        # reshape (batch_dim,) -> batch_shape
        return draws.reshape(*sample_shape, *self.batch_shape, *self.event_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of x given theta."""
        # ConditionalEstimator expects a batch shape of (batch_dim,)
        # and sample shape of (sample_dim,)
        sample_shape = x.shape[: -len(self.event_shape) - len(self.batch_shape)]
        x = x.reshape(-1, self.batch_shape.numel(), *self.event_shape)
        lp = self._estimator_log_prob(x, self._condition_reshaped)
        # reshape (sample_dim, batch_dim,) -> (*sample_shape, *batch_shape)
        return lp.reshape(*sample_shape, *self.batch_shape)

    def expand(self, batch_shape: torch.Size, _instance=None):
        """Expand the batch shape of the distribution."""
        # because batch shape of the distribution is entirely determined by the
        # condition, we only need to expand the condition
        condition = self.condition.expand(*batch_shape, *self._condition_shape)
        return type(self)(self.estimator, condition)


class ConditionedEstimatorDistribution(EstimatorDistribution):
    """
    A conditioned `sbi` estimator wrapped as a Pyro distribution.
    """

    def get_condition_and_event_shapes(
        self,
        estimator: ConditionalEstimator,
    ) -> Tuple[torch.Size, torch.Size]:
        return estimator.condition_shape, estimator.input_shape

    def __init__(self, estimator: ConditionalEstimator, condition: torch.Tensor):
        """
        Condition an `sbi` estimator and wrap it as a Pyro distribution.

        Args:
            estimator: A trained conditional estimator
            condition: The conditioning parameter for the estimator.
        """
        super().__init__(estimator, condition)

    @property
    def condition_shape(self):
        return self.estimator.condition_shape

    def _estimator_log_prob(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.estimator.log_prob(x, condition=condition)
