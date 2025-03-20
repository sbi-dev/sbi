import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
)
from sbi.utils.vector_field_utils import VectorFieldNet


class FlowMatchingEstimator(ConditionalDensityEstimator):
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
        # embed the input and condition
        embedded_condition = self._embedding_net(condition)
        zscored_input = self.zscore_transform_input(input)

        # broadcast to match shapes of theta, x, and t
        zscored_input, embedded_condition = broadcast(
            zscored_input,  # type: ignore
            embedded_condition,
            ignore=1,
        )

        # return the estimated vector field
        return self.net(theta=zscored_input, x_emb_cond=embedded_condition, t=t)

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

        # the flow will apply and take into account input zscoring.
        input_reshaped = reshape_to_batch_event(input, input.shape[2:])
        condition = reshape_to_batch_event(condition, condition.shape[2:])
        embedded_condition = self._embedding_net(condition)
        log_probs = self.flow(condition=embedded_condition).log_prob(input_reshaped)
        log_probs = log_probs.reshape(input.shape[0], input.shape[1])
        return log_probs

    def sample(self, sample_shape: torch.Size, condition: Tensor, **kwargs) -> Tensor:
        batch_size_ = condition.shape[0]

        embedded_condition = self._embedding_net(condition)
        samples = self.flow(condition=embedded_condition).sample(sample_shape)

        samples = torch.reshape(samples, (sample_shape[0], batch_size_, -1))
        return samples

    def sample_and_log_prob(
        self, sample_shape: torch.Size, condition: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        embedded_condition = self._embedding_net(condition)
        samples, log_probs = self.flow(
            condition=embedded_condition
        ).rsample_and_log_prob(sample_shape)

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

        # Define a wrapper function that properly handles dimensions during sampling
        def vector_field_fn(t, input):
            # When sampling, Zuko adds a sample dimension at the front
            # During sampling, input shape will be [num_samples, batch_size, dim]
            # But during training, it's just [batch_size, dim]
            orig_shape = input.shape
            if len(orig_shape) == 3:  # When sampling (has sample dimension)
                # Reshape to merge the first two dims for processing
                num_samples, batch_size = orig_shape[0], orig_shape[1]
                flat_input = input.reshape(num_samples * batch_size, -1)

                # Repeat condition to match the flattened input
                expanded_condition = condition.repeat(num_samples, 1)

                # Call forward with flattened tensors
                t_expanded = t.expand(flat_input.shape[0])
                vector_field = self.forward(flat_input, expanded_condition, t_expanded)

                # Reshape back to original shape
                return vector_field.reshape(orig_shape)
            else:
                # During training or evaluation, dimensions are as expected
                return self.forward(input, condition, t)

        transform = zuko.transforms.ComposedTransform(
            FreeFormJacobianTransform(
                f=vector_field_fn,
                t0=condition.new_tensor(0.0),
                t1=condition.new_tensor(1.0),
                phi=(condition, *self.net.parameters()),
            ),
            self.zscore_transform_input,
        )

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
