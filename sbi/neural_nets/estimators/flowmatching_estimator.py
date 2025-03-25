from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.base import ConditionalVectorFieldEstimator


# abstract class to ensure forward signature for flow matching networks
class VectorFieldNet(nn.Module, ABC):
    @abstractmethod
    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor: ...


class FlowMatchingEstimator(ConditionalVectorFieldEstimator):
    r"""
    Rectified flow matching estimator class that estimates the conditional vector field,
    :math:`v(\theta_t, t; x_o) = \mathbb{E}[\theta_1 - \theta_0 | \theta_t, x_o = x_o]`

    This estimator implements the flow matching approach where the vector field is
    learned by matching the flow between the base and target distributions. The vector
    field represents the instantaneous change in the distribution at time t.

    References
    ----------
    .. [1] Liu, X., Gong, C., & Liu, Q. (2023).
           "Flow Straight and Fast: Learning to Generate and Transfer Data with
           Rectified Flow"
           *International Conference on Learning Representations (ICLR)*
           https://arxiv.org/abs/2209.03003

    .. [2] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023).
           "Flow Matching for Generative Modeling"
           *International Conference on Learning Representations (ICLR)*
           https://arxiv.org/abs/2210.02747

    .. [3] Singh, S., & Fischer, I. (2024).
           "Stochastic Sampling from Deterministic Flow Models"
           https://arxiv.org/abs/2410.02217
    """

    # Whether the score is defined for this estimator.
    # Required for gradient-based methods.
    SCORE_DEFINED: bool = True
    # Whether the SDE functions - score, drift and diffusion -
    # are defined for this estimator.
    SDE_DEFINED: bool = True
    # Whether the marginals are defined for this estimator.
    # Required for iid methods.
    MARGINALS_DEFINED: bool = True

    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        num_freqs: int = 3,  # This is ignored and will be removed in PR #1501
        noise_scale: float = 1e-3,
        zscore_transform_input=None,  # This is ignored and will be removed in PR #1501
        **kwargs,
    ) -> None:
        r"""Creates a vector field estimator for Flow Matching.

        Args:
            net: Neural network that estimates the vector field.
            input_shape: Shape of the input :math:`\theta`.
            condition_shape: Shape of the condition :math:`x_o`.
            embedding_net: Embedding network for the condition.
            num_freqs: Number of frequencies to use for the positional time encoding.
                This is ignored and will be removed.
            noise_scale: Scale of the noise added to the vector field
                (:math:`\sigma_{min}` in [2]_).
            zscore_transform_input: Whether to z-score the input.
                This is ignored and will be removed.
        """

        super().__init__(
            net=net, input_shape=input_shape, condition_shape=condition_shape
        )

        self.num_freqs = num_freqs  # This will be removed in PR #1501
        self.noise_scale = noise_scale
        self._embedding_net = (
            embedding_net if embedding_net is not None else nn.Identity()
        )

    @property
    def embedding_net(self):
        return self._embedding_net

    def forward(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass of the FlowMatchingEstimator.

        Args:
            input: The input tensor.
            condition: The condition tensor.
            t: The time tensor.

        Returns:
            The estimated vector field.
        """
        # temporal fix that will be removed when the nn builders are updated
        t = self._get_temporal_t_shape_fix(t)

        batch_shape = torch.broadcast_shapes(
            input.shape[: -len(self.input_shape)],
            condition.shape[: -len(self.condition_shape)],
        )

        input = torch.broadcast_to(input, batch_shape + self.input_shape)
        condition = torch.broadcast_to(condition, batch_shape + self.condition_shape)
        t = torch.broadcast_to(t, batch_shape + t.shape[1:])

        # the network expects 2D input, so we flatten the input if necessary
        # and remember the original shape
        target_shape = input.shape
        input = input.reshape(-1, input.shape[-1])
        condition = condition.reshape(-1, condition.shape[-1])
        t = t.reshape(-1, t.shape[-1])

        # embed the input and condition
        embedded_condition = self._embedding_net(condition)

        # call the network to get the estimated vector field
        v = self.net(theta=input, x=embedded_condition, t=t)

        # reshape to the original shape
        v = v.reshape(*target_shape)
        return v

    def loss(
        self, input: Tensor, condition: Tensor, times: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        r"""Return the loss for training the density estimator.

        More precisely, we compute the conditional flow matching loss with naive optimal
        trajectories as described in the original paper [1]_:

        .. math::
            \mathbb{E}_{\theta_0 \sim p_{data}, t \sim \text{Uniform}[0, 1]},
            \theta_t = t \cdot \theta_1 + (1 - t) \cdot \theta_0,
            \left[ \| v(\theta_t, t; x_o = x_o) - (\theta_1 - \theta_0) \|^2 \right]

        where :math:`v(\theta_t, t; x_o)` is the vector field estimated by the neural
        network (see Equation 1 in [1]_ with added conditioning on :math:`x_o`. The
        notation is changed to match the standard SBI notation: :math:`\theta_0 = x_0`
        and :math:`\theta_1 = x_1`).

        The loss is computed as the mean squared error between the vector field:

        .. math::
            L(\theta_0, \theta_1, t, x_o) = \| v(\theta_t, t; x_o) -
            (\theta_1 - \theta_0) \|^2

            where

            .. math::
                \theta_1 \sim p_{base}
                \theta_t = t \cdot \theta_1 + (1 - t) \cdot \theta_0

            Additionally, the small noise :math:`\sigma_{min}` is added to the
            vector field as per [2]_ to address numerical issues at small times.

        Args:
            theta: Parameters (:math:`\theta_0`).
            x: Observed data (:math:`x_o`).
            times: Time steps to compute the loss at.
                Optional, will sample from [0, 1] if not provided.

        Returns:
            Loss value.
        """
        # randomly sample the time steps to compare the vector field at
        # different time steps
        if times is None:
            times = torch.rand(input.shape[:-1], device=input.device, dtype=input.dtype)
        times_ = times[..., None]

        # sample from probability path at time t
        # TODO: Change to notation from Lipman et al. or Tong et al.
        theta_1 = torch.randn_like(input)
        theta_t = (1 - times_) * input + (times_ + self.noise_scale) * theta_1

        # compute vector field at the sampled time steps
        vector_field = theta_1 - input

        # compute the mean squared error between the vector fields
        return torch.mean(
            (self.forward(theta_t, condition, times) - vector_field) ** 2, dim=-1
        )

    def ode_fn(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        r"""
        ODE flow function :math:`v(\theta_t, t, x_o)` of the vector field estimator.

        The target distribution can be sampled from by solving the following ODE:

        .. math::
            d\theta_t = v(\theta_t, t; x_o) dt

        with initial :math:`\theta_1` sampled from the base distribution.
        Here :math:`v(\theta_t, t; x_o)` is the vector field estimated by the
        flow matching neural network (see Equation 1 in [1]_ with added
        conditioning on :math:`x_o`).

        Args:
            input: :math:`\theta_t`.
            condition: Conditioning variable :math:`x_o`.
            times: Time :math:`t`.

        Returns:
            Estimated vector field :math:`v(\theta_t, t; x_o)`.
            The shape is the same as the input.
        """
        return self.forward(input, condition, times)

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        r"""Score function of the vector field estimator.

        The score function is calculated based on [3]_ (see Equation 13):

        .. math::
            \nabla_{\theta_t} \log p(\theta_t | x_o) =
            (- (1 - t) v(\theta_t, t; x_o) - \theta_0 ) / t

        Taking into account the noise scale :math:`\sigma_{min}`, the score function is
        :math:`\nabla_{\theta_t} \log p(\theta_t | x_o) =
            (- (1 - t) v(\theta_t, t; x_o) - \theta_0 ) / (t + \sigma_{min})`.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Returns:
            Score function of the vector field estimator.
        """
        v = self(input, condition, t)
        score = (-(1 - t) * v - input) / (t + self.noise_scale)
        return score

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Drift function for the flow matching estimator.

        The drift function is calculated based on [3]_ (see Equation 7):

        .. math::
            f(t) = - \theta_t / (1 - t)

        The drift function :math:`f(t)` and diffusion function :math:`\g(t)`
        enable SDE sampling:

        .. math::
            d\theta_t = [f(t) - g(t)^2 \nabla_{\theta_t} \log p(\theta_t | x_o)]dt
            + \g(t) dW_t

        where :math:`dW_t` is the Wiener process.

        Args:
            input: Parameters :math:`\theta_t`.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        # analytical f(t) does not depend on noise_scale and is undefined at t = 1.
        return -input / torch.maximum(1 - times, torch.tensor(1e-6).to(input))

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Diffusion function for the flow matching estimator.

        The diffusion function is calculated based on [3]_ (see Equation 7):

        .. math::
            \g(t) = \sqrt{2t / (1 - t)}

        Taking into account the noise scale :math:`\sigma_{min}`, the diffusion
        function becomes:

        .. math::
            \g(t) = \sqrt{2(t + \sigma_{min}) / (1 - t)}


        Args:
            input: Parameters :math:`\theta_t`.
            times: SDE time variable in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        # analytical g(t) is undefined at t = 1.
        return torch.sqrt(
            2
            * (times + self.noise_scale)
            / torch.maximum(1 - times, torch.tensor(1e-6).to(times))
        )

    def mean_t_fn(self, times: Tensor) -> Tensor:
        r"""Linear coefficient of the perturbation kernel expectation
        :math:`\mu_t(t) = E[\theta_t | \theta_0]` for the flow matching estimator.

        The perturbation kernel for rectified flows with Gaussian base distribution is:

        .. math::
            N(\theta_t; \mu_t(t), \sigma_t(t)^2) ,
            \mu_t(t) = (1 - t) \cdot \theta_0 + t \cdot \mu_{base}
            \sigma_t(t) = t \cdot \sigma_{base}

        So far, the implementation of iid methods assumes that the mean_base
        :math:`\mu_{base}` is 0. Therefore, the linear coefficient of the perturbation
        kernel mean is simply :math:`1 - t`.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Mean function at a given time.
        """
        mean_t = 1 - times
        for _ in range(len(self.input_shape)):
            mean_t = mean_t.unsqueeze(-1)
        return mean_t

    def std_fn(self, times: Tensor) -> Tensor:
        r"""Standard deviation of the perturbation kernel :math:`\sigma_t(t)`
        for the flow matching estimator.

        The perturbation kernel for rectified flows with Gaussian base distribution is:

        .. math::
            N(\theta_t; \mu_t(t), \sigma_t(t)^2) ,
            \mu_t(t) = (1 - t) \cdot \theta_0 + t \cdot \mu_{base}
            \sigma_t(t) = t \cdot \sigma_{base}

        Taking into account the noise scale :math:`\sigma_{min}`, the standard deviation
        becomes:

        .. math::
            \sigma_t(t) = (t + \sigma_{min}) \cdot \sigma_{base}

        Note that in the current implementation, the base distribution is Gaussian with
        zero mean and unit variance.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Standard deviation at a given time.
        """
        std_t = times + self.noise_scale
        for _ in range(len(self.input_shape)):
            std_t = std_t.unsqueeze(-1)
        return std_t

    # this method will be removed in PR #1501
    def _get_temporal_t_shape_fix(self, t: Tensor) -> Tensor:
        """
        This is a hack that allows us to use
        the old nn builders that assume positional embedding of time
        inside the forward method, resulting in a shape of (..., num_freqs * 2).
        """
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t[..., None]
        if t.shape[-1] == 1:
            t = t.expand(*t.shape[:-1], self.num_freqs * 2)
        return t
