import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.base import (
    ConditionalVectorFieldEstimator,
    MaskedConditionalVectorFieldEstimator,
)
from sbi.utils.vector_field_utils import MaskedVectorFieldNet, VectorFieldNet


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
        noise_scale: float = 1e-3,
        **kwargs,
    ) -> None:
        r"""Creates a vector field estimator for Flow Matching.

        Args:
            net: Neural network that estimates the vector field.
            input_shape: Shape of the input :math:`\theta`.
            condition_shape: Shape of the condition :math:`x_o`.
            embedding_net: Embedding network for the condition.
            noise_scale: Scale of the noise added to the vector field
                (:math:`\sigma_{min}` in [2]_).
            zscore_transform_input: Whether to z-score the input.
                This is ignored and will be removed.
        """

        if "num_freqs" in kwargs:
            del kwargs["num_freqs"]
            warnings.warn(
                "num_freqs is deprecated and will be removed in the future. "
                "Please use the positional_encoding_net instead.",
                stacklevel=2,
            )

        super().__init__(
            net=net,
            input_shape=input_shape,
            condition_shape=condition_shape,
            embedding_net=embedding_net,
        )
        self.noise_scale = noise_scale

    def forward(self, input: Tensor, condition: Tensor, time: Tensor) -> Tensor:
        """Forward pass of the FlowMatchingEstimator.

        Args:
            input: Inputs to evaluate the vector field on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.
            condition: Conditions of shape
                `(batch_dim_condition, *event_shape_condition)`.
            time: Time variable in [0,1] of shape
                `(batch_dim_time, *event_shape_time)`.

        Returns:
            The estimated vector field.
        """
        # Continue with standard processing (broadcast shapes etc.)
        batch_shape_input = input.shape[: -len(self.input_shape)]
        batch_shape_cond = condition.shape[: -len(self.condition_shape)]
        batch_shape = torch.broadcast_shapes(
            batch_shape_input,
            batch_shape_cond,
        )

        # embed the conditioning variable
        condition_emb = self._embedding_net(condition)

        input = torch.broadcast_to(input, batch_shape + self.input_shape)
        condition_emb = torch.broadcast_to(
            condition_emb, batch_shape + condition_emb.shape[len(batch_shape_cond) :]
        )
        time = torch.broadcast_to(time, batch_shape)

        # NOTE: To simplify use of external networks, we will flatten the tensors
        # batch_shape to a single batch dimension.
        input = input.reshape(-1, *input.shape[len(batch_shape) :])
        condition_emb = condition_emb.reshape(
            -1, *condition_emb.shape[len(batch_shape) :]
        )
        time = time.reshape(-1)

        # call the network to get the estimated vector field
        v = self.net(input, condition_emb, time)
        v = v.reshape(*batch_shape + self.input_shape)

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


class MaskedFlowMatchingEstimator(MaskedConditionalVectorFieldEstimator):
    r"""
    Rectified flow matching estimator class that estimates the conditional vector field,
    :math:`v(\text{inputs}_t, t; x_o) = \mathbb{E}[\text{inputs}_1 - \text{inputs}_0 |
        \text{inputs}_t, x_o = x_o]`

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
        net: MaskedVectorFieldNet,
        input_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        noise_scale: float = 1e-3,
    ) -> None:
        r"""Creates a masked vector field estimator for Flow Matching.

        Args:
            net: Neural network that estimates the vector field.
            input_shape: Shape of the input.
            embedding_net: Embedding network for the condition.
            noise_scale: Scale of the noise added to the vector field
                (:math:`\sigma_{min}` in [2]_).
            zscore_transform_input: Whether to z-score the input.
                This is ignored and will be removed.
        """

        super().__init__(
            net=net,
            input_shape=input_shape,
            embedding_net=embedding_net,
        )
        self.noise_scale = noise_scale

    def forward(
        self,
        input: Tensor,
        time: Tensor,
        condition_mask: Tensor,
        edge_mask: Optional[Tensor],
    ) -> Tensor:
        """Forward pass of the FlowMatchingEstimator.

        Args:
            input: Inputs to evaluate the vector field on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.
            time: Time variable in [0,1] of shape
                `(batch_dim_time, *event_shape_time)`.
            condition_mask: A boolean mask indicating the role of each variable.
                Expected shape: `(batch_size, num_variables)`.

                - `True` (or `1`): The variable at this position is observed and its
                    features will be used for conditioning.
                - `False` (or `0`): The variable at this position is latent and its
                    features are subject to inference.
            edge_mask: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among variables.
                Expected shape: `(batch_size, num_variables, num_variables)`.

                - `True` (or `1`): An edge exists from the row variable to the column
                    variable.
                - `False` (or `0`): No edge exists between these variables.
                - if None, it will be equivalent to a full attention (i.e., full ones)
                    mask, we suggest you to use None instead of ones
                    to save memory resources

        Returns:
            The estimated vector field.
        """
        # Continue with standard processing (broadcast shapes etc.)
        batch_shape = input.shape[: -len(self.input_shape)]

        input = torch.broadcast_to(input, batch_shape + self.input_shape)
        time = torch.broadcast_to(time, batch_shape)

        # NOTE: To simplify use of external networks, we will flatten the tensors
        # batch_shape to a single batch dimension.
        input = input.reshape(-1, *input.shape[len(batch_shape) :])
        time = time.reshape(-1)

        # call the network to get the estimated vector field
        v = self.net(input, time, condition_mask, edge_mask)
        v = v.reshape(*batch_shape + self.input_shape)

        return v

    def loss(
        self,
        input: Tensor,
        condition_mask: Tensor,
        edge_mask: Optional[Tensor],
        times: Optional[Tensor] = None,
        rebalance_loss: bool = True,
        **kwargs,
    ) -> Tensor:
        r"""Return the loss for training the density estimator.

        More precisely, we compute the conditional flow matching loss with naive optimal
        trajectories as described in the original paper [1]_:

        .. math::
            \mathbb{E}_{\text{inputs}_0 \sim p_{data}, t \sim \text{Uniform}[0, 1]},
            \text{inputs}_t = t \cdot \text{inputs}_1 + (1 - t) \cdot \text{inputs}_0,
            \left[ \| v(\text{inputs}_t, t; x_o = x_o) - (\text{inputs}_1 -
                \text{inputs}_0) \|^2 \right]

        where :math:`v(\text{inputs}_t, t; x_o)` is the vector field
            estimated by the neural
        network (see Equation 1 in [1]_ with added conditioning on :math:`x_o`. The
        notation is changed to match the standard SBI notation:
            :math:`\text{inputs}_0 = x_0` and :math:`\text{inputs}_1 = x_1`).

        The loss is computed as the mean squared error between the vector field:

        .. math::
            L(\text{inputs}_0, \text{inputs}_1, t, x_o) = \| v(\text{inputs}_t, t; x_o)
                - (\text{inputs}_1 - \text{inputs}_0) \|^2

            where

            .. math::
                \text{inputs}_1 \sim p_{base}
                \text{inputs}_t = t \cdot \text{inputs}_1 + (1 - t) \cdot
                    \text{inputs}_0

            Additionally, the small noise :math:`\sigma_{min}` is added to the
            vector field as per [2]_ to address numerical issues at small times.

        Args:
            input: Original data
                Shape: [B, T, F]
            times: SDE time variable in [t_min, t_max]. Uniformly sampled if None.
            condition_mask: A boolean mask indicating the role of each variable.
                Expected shape: `(batch_size, num_variables)`.

                - `True` (or `1`): The variable at this position is observed and its
                    features will be used for conditioning.
                - `False` (or `0`): The variable at this position is latent and its
                    features are subject to inference.
            edge_mask: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among variables.
                Expected shape: `(batch_size, num_variables, num_variables)`.

                - `True` (or `1`): An edge exists from the row variable to the column
                    variable.
                - `False` (or `0`): No edge exists between these variables.
                - if None, it will be equivalent to a full attention (i.e., full ones)
                    mask, we suggest you to use None instead of ones
                    to save memory resources
            rebalance_loss: If True, the loss for each batch item is divided by the
                number of latent (unobserved) variables in that item. This is useful
                when the number of latent variables varies across the batch, as it
                prevents items with more latent variables from dominating the loss.

        Returns:
            Loss value.
        """
        # randomly sample the time steps to compare the vector field at
        # different time steps
        if times is None:
            times = torch.rand(input.shape[:-2], device=input.device, dtype=input.dtype)
        times_ = times[..., None, None]

        # sample from probability path at time t
        input_1 = torch.randn_like(input)
        input_t = (1 - times_) * input + (times_ + self.noise_scale) * input_1

        # compute vector field at the sampled time steps
        vector_field_target = input_1 - input

        # Get the predicted vector field from the network
        # Pass masks to the forward method
        predicted_vector_field = self.forward(input_t, times, condition_mask, edge_mask)

        # compute the mean squared error between the vector fields
        loss = (predicted_vector_field - vector_field_target) ** 2.0

        # Apply condition mask: loss is only computed for latent variables
        # condition_mask is (..., T), expand to (..., T, F) to match loss shape
        condition_mask_expanded = condition_mask.bool().unsqueeze(-1).expand_as(loss)
        loss = torch.where(condition_mask_expanded, torch.zeros_like(loss), loss)

        # Sum over T and F dimensions to get loss per batch item (shape (...,))
        loss = loss.sum(dim=(-2, -1))

        # Rebalance loss if desired (similar to MaskedConditionalScoreEstimator)
        if rebalance_loss:
            # Count number of unobserved (latent) elements per batch
            # (~condition_mask) computes the boolean negation
            num_elements = (~condition_mask.bool()).sum(dim=-1).clamp(min=1)
            loss = loss / num_elements

        return loss

    def ode_fn(
        self,
        input: Tensor,
        times: Tensor,
        condition_mask: Tensor,
        edge_mask: Tensor,
    ) -> Tensor:
        r"""
        ODE flow function :math:`v(\text{inputs}_t, t, x_o)` of the vector field
            estimator.

        The target distribution can be sampled from by solving the following ODE:

        .. math::
            d\text{inputs}_t = v(\text{inputs}_t, t; x_o) dt

        with initial :math:`\text{inputs}_1` sampled from the base distribution.
        Here :math:`v(\text{inputs}_t, t; x_o)` is the vector field estimated by the
        flow matching neural network (see Equation 1 in [1]_ with added
        conditioning on :math:`x_o`).

        Args:
            input: :math:`\text{inputs}_t`.
            times: Time :math:`t`.
            condition_mask: A boolean mask indicating the role of each variable.
                Expected shape: `(batch_size, num_variables)`.

                - `True` (or `1`): The variable at this position is observed and its
                    features will be used for conditioning.
                - `False` (or `0`): The variable at this position is latent and its
                    features are subject to inference.
            edge_mask: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among variables.
                Expected shape: `(batch_size, num_variables, num_variables)`.

                - `True` (or `1`): An edge exists from the row variable to the column
                    variable.
                - `False` (or `0`): No edge exists between these variables.
                - if None, it will be equivalent to a full attention (i.e., full ones)
                    mask, we suggest you to use None instead of ones
                    to save memory resources

        Returns:
            Estimated vector field :math:`v(\text{inputs}_t, t; x_o)`.
            The shape is the same as the input.
        """
        return self.forward(input, times, condition_mask, edge_mask)

    def score(
        self, input: Tensor, t: Tensor, condition_mask: Tensor, edge_mask: Tensor
    ) -> Tensor:
        r"""Score function of the vector field estimator.

        The score function is calculated based on [3]_ (see Equation 13):

        .. math::
            \nabla_{\text{inputs}_t} \log p(\text{inputs}_t | x_o) =
            (- (1 - t) v(\text{inputs}_t, t; x_o) - \text{inputs}_0 ) / t

        Taking into account the noise scale :math:`\sigma_{min}`, the score function is
        :math:`\nabla_{\text{inputs}_t} \log p(\text{inputs}_t | x_o) =
            (- (1 - t) v(\text{inputs}_t, t; x_o) -
                \text{inputs}_0 ) / (t + \sigma_{min})`.

        Args:
            input: variable whose distribution is estimated.
            t: Time.
            condition_mask: A boolean mask indicating the role of each variable.
                Expected shape: `(batch_size, num_variables)`.

                - `True` (or `1`): The variable at this position is observed and its
                    features will be used for conditioning.
                - `False` (or `0`): The variable at this position is latent and its
                    features are subject to inference.
            edge_mask: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among variables.
                Expected shape: `(batch_size, num_variables, num_variables)`.

                - `True` (or `1`): An edge exists from the row variable to the column
                    variable.
                - `False` (or `0`): No edge exists between these variables.
                - if None, it will be equivalent to a full attention (i.e., full ones)
                    mask, we suggest you to use None instead of ones
                    to save memory resources

        Returns:
            Score function of the vector field estimator.
        """
        v = self(input, t, condition_mask, edge_mask)
        score = (-(1 - t) * v - input) / (t + self.noise_scale)
        return score

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Drift function for the flow matching estimator.

        The drift function is calculated based on [3]_ (see Equation 7):

        .. math::
            f(t) = - \text{inputs}_t / (1 - t)

        The drift function :math:`f(t)` and diffusion function :math:`\g(t)`
        enable SDE sampling:

        .. math::
            d\text{inputs}_t = [f(t) - g(t)^2 \nabla_{\text{inputs}_t}
                \log p(\text{inputs}_t | x_o)]dt
            + \g(t) dW_t

        where :math:`dW_t` is the Wiener process.

        Args:
            input: Parameters :math:`\text{inputs}_t`.
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
            input: Parameters :math:`\text{inputs}_t`.
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
        :math:`\mu_t(t) = E[\text{inputs}_t | \text{inputs}_0]` for the flow matching
            estimator.

        The perturbation kernel for rectified flows with Gaussian base distribution is:

        .. math::
            N(\text{inputs}_t; \mu_t(t), \sigma_t(t)^2) ,
            \mu_t(t) = (1 - t) \cdot \text{inputs}_0 + t \cdot \mu_{base}
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
            N(\text{inputs}_t; \mu_t(t), \sigma_t(t)^2) ,
            \mu_t(t) = (1 - t) \cdot \text{inputs}_0 + t \cdot \mu_{base}
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
