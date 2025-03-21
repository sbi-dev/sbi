from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.base import ConditionalEstimator


class ConditionalVectorFieldEstimator(ConditionalEstimator):
    r"""Base class for vector field estimators. That primarily includes
    score-based and flow matching models.

    The vector field estimator class is a wrapper around neural networks that allows to
    evaluate the `vector_field`, and provide the `loss` of $\theta,x_o$ pairs. Here
    $\theta$ would be the `input` and $x_o$ would be the `condition`.

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
        mean_base: float = 0.0,
        std_base: float = 1.0,
    ) -> None:
        r"""Base class for vector field estimators.

        Args:
            net: Neural network.
            input_shape: Shape of the input.
            condition_shape: Shape of the condition. If not provided, it will assume a
                            1D input.
            t_min: Minimum time for the vector field estimator.
            t_max: Maximum time for the vector field estimator.
            mean_base: Mean of the base distribution.
            std_base: Standard deviation of the base distribution.
        """
        super().__init__(input_shape, condition_shape)
        self.net = net

        # We assume that the time range is the same for ODE and SDE.
        self.t_min = t_min
        self.t_max = t_max

        # We assume that the base distribution is a Gaussian distribution
        # and that it is the same for ODE and SDE.
        # We store the mean and std of the base distribution in buffers
        # to transfer them to the device automatically when the model is moved.
        self.register_buffer(
            "_mean_base", torch.empty(1, *self.input_shape).fill_(mean_base)
        )
        self.register_buffer(
            "_std_base", torch.empty(1, *self.input_shape).fill_(std_base)
        )

    @abstractmethod
    def forward(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Forward pass of the score estimator.

        Args:
            input: Input variable :math:`\theta_t`.
            condition: Conditioning variable :math:`x_o`.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """

    # -------------------------- BASE DISTRIBUTION METHODS --------------------------

    # We assume that the base distribution is a Gaussian distribution
    # and that it is the same for ODE and SDE.

    @property
    def mean_base(self) -> Tensor:
        r"""Mean of the base distribution (the initial noise at time t=T)."""
        return self._mean_base

    @property
    def std_base(self) -> Tensor:
        r"""Standard deviation of the base distribution
        (the initial noise at time t=T)."""
        return self._std_base

    # -------------------------- ODE METHODS --------------------------

    @abstractmethod
    def ode_fn(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        r"""ODE flow function :math:`v(\theta_t, t, x_o)` of the vector field estimator.

        The target distribution can be sampled from by solving the following ODE:

        .. math::
            d\theta_t = v(\theta_t, t; x_o) dt

        with initial :math:`\theta_1` sampled from the base distribution.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """

    # -------------------------- SDE METHODS --------------------------

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        r"""Time-dependent score function

        .. math::
            s(t, \theta_t; x_o) = \nabla_{\theta_t} \log p(\theta_t | x_o)

        Args:
            input: Input parameters :math:`\theta_t`.
            condition: Conditioning variable :math:`x_o`.
            t: Time.

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError("This method should be implemented by sub-classes.")

    def mean_t_fn(self, times: Tensor) -> Tensor:
        r"""Linear coefficient mean_t of the perturbation kernel expectation
        :math:`\mu_t(t) = E[\theta_t | \theta_0] = \text{mean_t}(t) \cdot \theta_0`
        specifying the "mean factor" at a given time, which is always multiplied by
        :math:`\theta_0` to get the mean of the noise distribution, i.e.,
        :math:`p(\theta_t | \theta_0) = N(\theta_t;
                \text{mean_t}(t)*\theta_0, \text{std_t}(t)).`

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError

    def std_fn(self, times: Tensor) -> Tensor:
        r"""Standard deviation function std_t(t) of the perturbation kernel at a given
            time,

        .. math::
            p(\theta_t | \theta_0) = N(\theta_t; \text{mean_t}(t) \cdot
            \theta_0, \text{std_t}(t)^2).

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Drift function :math:`f(t)` of the vector field estimator.

        The drift function :math:`f(t)` and diffusion function :math:`\g(t)`
        enable SDE sampling:

        .. math::
            d\theta_t = [f(t) - g(t)^2 \nabla_{\theta_t} \log p(\theta_t | x_o)]dt
              + \g(t) dW_t

        where :math:`dW_t` is the Wiener process.


        Args:
            input: input parameters :math:`\theta_t`.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.

        """
        raise NotImplementedError

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Diffusion function :math:`\g(t)` of the vector field estimator.

        The drift function :math:`f(t)` and diffusion function :math:`\g(t)`
        enable SDE sampling:

        .. math::
            d\theta_t = [f(t) - g(t)^2 \nabla_{\theta_t} \log p(\theta_t | x_o)]dt
              + \g(t) dW_t

        where :math:`dW_t` is the Wiener process.

        Args:
            input: input parameters :math:`\theta_t`.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method should be implemented by sub-classes.
        """
        raise NotImplementedError
