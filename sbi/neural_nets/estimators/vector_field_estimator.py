from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.base import ConditionalEstimator


class ConditionalVectorFieldEstimator(ConditionalEstimator):
    r"""Base class for vector field (e.g., score and ODE flow) estimators.

    The vector field estimator class is a wrapper around neural networks that allows to
    evaluate the `vector_field`, and provide the `loss` of $\theta,x$ pairs. Here
    $\theta$ would be the `input` and $x$ would be the `condition`.

    Note:
        We assume that the input to the density estimator is a tensor of shape
        (sample_dim, batch_dim, *input_shape), where input_shape is the dimensionality
        of the input. The condition is a tensor of shape (batch_dim, *condition_shape),
        where condition_shape is the shape of the condition tensor.

    """

    # When implementing custom estimators,
    # the following properties should be set:

    # Whether the score is defined for this estimator.
    # Required for gradient-based methods.
    # It should be set to True only if score is implemented.
    SCORE_DEFINED: bool = True

    # Whether the SDE functions - score, drift and diffusion -
    # are defined for this estimator.
    # Required for SDE sampling.
    SDE_DEFINED: bool = True

    # Whether the marginals are defined for this estimator.
    # Required for iid methods.
    # It should be set to True only if mean_t_fn and std_fn are implemented.
    MARGINALS_DEFINED: bool = True

    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> None:
        r"""Base class for vector field estimators.

        Args:
            net: Neural network.
            input_shape: Shape of the input.
            condition_shape: Shape of the condition. If not provided, it will assume a
                            1D input.
            t_min: Minimum time for the vector field estimator.
            t_max: Maximum time for the vector field estimator.
        """
        super().__init__(input_shape, condition_shape)
        self.net = net

        # We assume that the time range is the same for ODE and SDE.
        self.t_min = t_min
        self.t_max = t_max

    @abstractmethod
    def forward(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Forward pass of the score estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """

    # -------------------------- BASE DISTRIBUTION METHODS --------------------------

    # We assume that the base distribution is a Gaussian distribution
    # and that it is the same for ODE and SDE.

    @property
    def mean_base(self) -> Tensor:
        r"""Mean of the base distribution (the initial noise at time t=T)."""
        return torch.zeros(1, *self.input_shape)

    @property
    def std_base(self) -> Tensor:
        r"""Standard deviation of the base distribution
        (the initial noise at time t=T)."""
        return torch.ones(1, *self.input_shape)

    # -------------------------- ODE METHODS --------------------------

    @abstractmethod
    def ode_fn(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        r"""ODE flow function of the vector field estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """

    # -------------------------- SDE METHODS --------------------------

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        r"""Score function of the vector field estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError("This method should be implemented by sub-classes.")

    def mean_t_fn(self, times: Tensor) -> Tensor:
        r"""Conditional mean function, E[xt|x0], specifying the "mean factor" at a given
        time, which is always multiplied by x0 to get the mean of the noise distribution
        , i.e., p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)).

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError

    def std_fn(self, times: Tensor) -> Tensor:
        r"""Standard deviation function of the noise distribution at a given time,

        i.e., p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)).

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Drift function of the vector field estimator.

        Args:
            input: variable whose distribution is estimated.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.

        """
        raise NotImplementedError

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Diffusion function of the vector field estimator.

        Args:
            input: variable whose distribution is estimated.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError
