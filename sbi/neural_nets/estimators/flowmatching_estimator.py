# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from sbi.neural_nets.estimators.base import ConditionalVectorFieldEstimator
from sbi.utils.vector_field_utils import VectorFieldNet

# Type aliases for configuration options
ZScoreMethod = Literal["true_marginal", "initial_pr_formula"]
GaussianBaselineMethod = Union[bool, Literal["velocity", "position", "position_raw"]]


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
        mean_1: Union[Tensor, float] = 0.0,
        std_1: Union[Tensor, float] = 1.0,
        gaussian_baseline: GaussianBaselineMethod = False,
        z_score_method: ZScoreMethod = "true_marginal",
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
            mean_1: Mean of the data distribution (used for time-dependent z-scoring).
            std_1: Std of the data distribution (used for time-dependent z-scoring).
            gaussian_baseline: Controls the Gaussian baseline velocity method.
                - False: No baseline (network learns full velocity)
                - True or "velocity": Use correct velocity formula derived from
                  Bayes' rule: v = factor * (x - μ_true) - mean
                - "position": Use position-based formula (for comparison):
                  v = t*x + (1-t)*E[θ_data|θ_t] - x (NOT recommended, fails on
                  shifted data)
                - "position_raw": Same as "position" but without velocity normalization
            z_score_method: Method for time-dependent input z-scoring.
                - "true_marginal": Use true marginal statistics μ_t = (1-t)*mean,
                  σ_t² = (1-t)²*std² + t². Keeps E[normalized input] = 0 for all t.
                - "initial_pr_formula": Use initial PR formula μ_t = t*mean,
                  σ_t² = t²*std² + (1-t)². Causes E[normalized input] to vary
                  wildly with t. (NOT recommended, for comparison only)
        """

        if "num_freqs" in kwargs:
            del kwargs["num_freqs"]
            warnings.warn(
                "num_freqs is deprecated and will be removed in the future. "
                "Please use the positional_encoding_net instead.",
                FutureWarning,
                stacklevel=2,
            )

        super().__init__(
            net=net,
            input_shape=input_shape,
            condition_shape=condition_shape,
            embedding_net=embedding_net,
        )
        self.noise_scale = noise_scale

        # Normalize gaussian_baseline to string or False
        if gaussian_baseline is True:
            self.gaussian_baseline = "velocity"
        elif gaussian_baseline is False:
            self.gaussian_baseline = False
        else:
            self.gaussian_baseline = gaussian_baseline

        self.z_score_method = z_score_method

        # Register z-scoring parameters as buffers
        mean_1_tensor = torch.as_tensor(mean_1).expand(input_shape).clone()
        std_1_tensor = torch.as_tensor(std_1).expand(input_shape).clone()
        self.register_buffer("mean_1", mean_1_tensor)
        self.register_buffer("std_1", std_1_tensor)

    def _get_time_dependent_stats(self, time: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute time-dependent mean and std for z-scoring.

        The method used depends on self.z_score_method:

        - "true_marginal": Uses the true marginal statistics of the interpolation
          θ_t = (1-t) * θ_data + t * θ_noise:
              μ_z(t) = (1-t) * mean_data
              σ_z(t) = sqrt((1-t)² * std_data² + t²)
          This ensures E[(θ_t - μ_z(t))/σ_z(t)] = 0 for all t.

        - "pr_convention": Uses PR's original convention:
              μ_z(t) = t * mean_data
              σ_z(t) = sqrt(t² * std_data² + (1-t)²)
          This causes E[normalized input] to vary with t.

        Args:
            time: Time tensor of shape (batch,)

        Returns:
            mu_t: Time-dependent mean for z-scoring, shape (batch, *input_shape)
            std_t: Time-dependent std for z-scoring, shape (batch, *input_shape)
        """
        t = time.view(-1, *([1] * len(self.input_shape)))
        mean = self.mean_1.view(1, *self.input_shape)
        std = self.std_1.view(1, *self.input_shape)

        if self.z_score_method == "true_marginal":
            mu_t = (1 - t) * mean
            var_t = ((1 - t) * std) ** 2 + t**2 + 1e-6
        elif self.z_score_method == "initial_pr_formula":
            mu_t = t * mean
            var_t = (t * std) ** 2 + (1 - t) ** 2 + 1e-6
        else:
            raise ValueError(f"Unknown z_score_method: {self.z_score_method}")

        std_t = torch.sqrt(var_t)
        return mu_t, std_t

    def _get_velocity_normalization(self) -> tuple[Tensor, Tensor]:
        """Get normalization statistics for the velocity target.

        The velocity v = θ_noise - θ_data is constant over time and has:
            E[v] = E[θ_noise] - E[θ_data] = 0 - mean = -mean
            Var[v] = Var[θ_noise] + Var[θ_data] = 1 + std²

        Returns:
            v_mean: Mean of velocity, shape (1, *input_shape)
            v_std: Std of velocity, shape (1, *input_shape)
        """
        mean = self.mean_1.view(1, *self.input_shape)
        std = self.std_1.view(1, *self.input_shape)
        v_mean = -mean
        v_std = torch.sqrt(1 + std**2)
        return v_mean, v_std

    def _get_gaussian_baseline(self, input: Tensor, time: Tensor) -> Tensor:
        """Compute the analytical baseline velocity based on self.gaussian_baseline.

        Dispatches to the appropriate method based on the gaussian_baseline setting.

        Args:
            input: Current position θ_t, shape (batch, *input_shape)
            time: Time tensor of shape (batch,)

        Returns:
            v_affine: Analytical baseline velocity, shape (batch, *input_shape)
        """
        if self.gaussian_baseline in (True, "velocity"):
            return self._compute_velocity_baseline(input, time)
        elif self.gaussian_baseline in ("position", "position_raw"):
            return self._compute_position_baseline(input, time)
        else:
            raise ValueError(
                f"Unknown gaussian_baseline method: {self.gaussian_baseline}"
            )

    def _compute_velocity_baseline(self, input: Tensor, time: Tensor) -> Tensor:
        """Compute the analytical marginal velocity assuming Gaussian data.

        For the interpolation θ_t = (1-t)*θ_data + t*θ_noise where
        θ_data ~ N(mean, std²) and θ_noise ~ N(0,1), the marginal velocity is:

            v(x, t) = E[θ_noise - θ_data | θ_t = x]
                    = factor * (x - μ_true) - mean

        where:
            μ_true = (1-t) * mean           (true marginal mean)
            var_true = (1-t)² * std² + t²   (true marginal variance)
            factor = (t - (1-t) * std²) / var_true

        This is derived using Bayes' rule for jointly Gaussian variables.

        Args:
            input: Current position θ_t, shape (batch, *input_shape)
            time: Time tensor of shape (batch,)

        Returns:
            v_affine: Analytical Gaussian velocity, shape (batch, *input_shape)
        """
        t = time.view(-1, *([1] * len(self.input_shape)))
        one_minus_t = 1 - t
        mean = self.mean_1.view(1, *self.input_shape)
        std = self.std_1.view(1, *self.input_shape)

        # True marginal statistics (not z-scoring stats!)
        mu_true = one_minus_t * mean
        var_true = (one_minus_t * std) ** 2 + t**2 + 1e-6

        # Analytical velocity: v = factor * (x - μ_true) - mean
        factor = (t - one_minus_t * std**2) / var_true
        v_affine = factor * (input - mu_true) - mean

        return v_affine

    def _compute_position_baseline(self, input: Tensor, time: Tensor) -> Tensor:
        """Compute position-based baseline formula (for comparison).

        Position-based formula from score-based diffusion:
            v_affine = t * x + (1-t) * E[θ_data | θ_t = x] - x

        where E[θ_data | θ_t = x] is the posterior mean of θ_data given θ_t.

        Args:
            input: Current position θ_t, shape (batch, *input_shape)
            time: Time tensor of shape (batch,)

        Returns:
            v_affine: baseline, shape (batch, *input_shape)
        """
        t = time.view(-1, *([1] * len(self.input_shape)))
        one_minus_t = 1 - t
        mean = self.mean_1.view(1, *self.input_shape)
        std = self.std_1.view(1, *self.input_shape)

        # Compute E[θ_data | θ_t = x] using Bayes' rule for Gaussians
        # θ_t = (1-t)*θ_data + t*θ_noise
        # E[θ_data | θ_t] = mean + cov(θ_data, θ_t) / var(θ_t) * (θ_t - E[θ_t])
        #                 = mean + (1-t)*std² / var_t * (x - (1-t)*mean)
        mu_true = one_minus_t * mean
        var_true = (one_minus_t * std) ** 2 + t**2 + 1e-6

        # Posterior mean of θ_data given θ_t
        cov_data_t = one_minus_t * std**2
        x1_hat = mean + cov_data_t / var_true * (input - mu_true)

        # position interpolation
        v_affine = t * input + one_minus_t * x1_hat - input

        return v_affine

    def forward(self, input: Tensor, condition: Tensor, time: Tensor) -> Tensor:
        """Forward pass of the FlowMatchingEstimator.

        Returns velocity in ORIGINAL SPACE for ODE integration.

        Args:
            input: Inputs to evaluate the vector field on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.
            condition: Conditions of shape
                `(batch_dim_condition, *event_shape_condition)`.
            time: Time variable in [0,1] of shape
                `(batch_dim_time, *event_shape_time)`.

        Returns:
            The estimated vector field in original space.
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

        # Time-dependent z-scoring of input
        mu_t, std_t = self._get_time_dependent_stats(time)
        input_norm = (input - mu_t) / std_t

        # Network forward pass (outputs in normalized velocity space)
        v_out = self.net(input_norm, condition_emb, time)

        # Get velocity normalization stats
        v_mean, v_std = self._get_velocity_normalization()

        if self.gaussian_baseline:
            v_affine = self._get_gaussian_baseline(input, time)
            if self.gaussian_baseline == "position_raw":
                # No velocity normalization - network outputs raw residual
                v = v_affine + v_out
            else:
                # Analytical Gaussian velocity + un-normalized network residual.
                # Note: v_affine already includes the velocity mean (-mean), so we
                # only scale the network output by v_std, not shift by v_mean.
                v = v_affine + v_out * v_std
        else:
            # Un-normalize velocity: v = v_out * v_std + v_mean
            v = v_out * v_std + v_mean

        v = v.reshape(*batch_shape + self.input_shape)

        return v

    def loss(
        self, input: Tensor, condition: Tensor, times: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        r"""Return the loss for training the density estimator.

        Computes loss in NORMALIZED SPACE for stable training with uniform
        gradient magnitudes across all time steps.

        More precisely, we compute the conditional flow matching loss with naive optimal
        trajectories as described in the original paper [1]_:

        .. math::
            \mathbb{E}_{\theta_0 \sim p_{data}, t \sim \text{Uniform}[0, 1]},
            \theta_t = t \cdot \theta_1 + (1 - t) \cdot \theta_0,
            \left[ \| v(\theta_t, t; x_o = x_o) - (\theta_1 - \theta_0) \|^2 \right]

        Args:
            input: Parameters (:math:`\theta_0`).
            condition: Observed data (:math:`x_o`).
            times: Time steps to compute the loss at.
                Optional, will sample from [0, 1] if not provided.

        Returns:
            Loss value.
        """
        # Randomly sample time steps
        if times is None:
            times = torch.rand(input.shape[:-1], device=input.device, dtype=input.dtype)
        times_ = times[..., None]

        # Sample from probability path at time t
        # θ_t = (1-t) * θ_data + t * θ_noise, where θ_noise ~ N(0,1)
        theta_1 = torch.randn_like(input)
        theta_t = (1 - times_) * input + (times_ + self.noise_scale) * theta_1

        # Target vector field in original space
        vector_field = theta_1 - input

        # Embed condition
        condition_emb = self._embedding_net(condition)

        # Compute time-dependent z-scoring stats for input
        times_flat = times.reshape(-1)
        mu_t, std_t = self._get_time_dependent_stats(times_flat)

        # Get velocity normalization stats (constant, not time-dependent)
        v_mean, v_std = self._get_velocity_normalization()

        # Reshape for broadcasting
        theta_t_flat = theta_t.reshape(-1, *self.input_shape)
        condition_emb_flat = condition_emb.reshape(-1, condition_emb.shape[-1])
        vector_field_flat = vector_field.reshape(-1, *self.input_shape)

        # Normalize input for network using true marginal stats
        theta_t_norm = (theta_t_flat - mu_t) / std_t

        # Network predicts in normalized velocity space
        v_out = self.net(theta_t_norm, condition_emb_flat, times_flat)

        if self.gaussian_baseline:
            # Compute analytical baseline in original space
            v_affine = self._get_gaussian_baseline(theta_t_flat, times_flat)
            if self.gaussian_baseline == "position_raw":
                # No velocity normalization - raw residual loss
                residual_target = vector_field_flat - v_affine
                loss = torch.mean((v_out - residual_target) ** 2, dim=-1)
            else:
                # Normalize the residual target. Note: v_affine already includes the
                # velocity mean (-mean), so the residual (v - v_affine) has E[r] = 0.
                # We only divide by v_std for scale normalization.
                residual_target = (vector_field_flat - v_affine) / v_std
                # Loss in normalized space
                loss = torch.mean((v_out - residual_target) ** 2, dim=-1)
        else:
            # Normalize the velocity target: v_norm = (v - v_mean) / v_std
            target_norm = (vector_field_flat - v_mean) / v_std
            # Loss in normalized space
            loss = torch.mean((v_out - target_norm) ** 2, dim=-1)

        return loss.reshape(input.shape[:-1])

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

    def drift_fn(
        self, input: Tensor, times: Tensor, effective_t_max: float = 0.99
    ) -> Tensor:
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
            effective_t_max: Upper bound on time to avoid numerical issues at t=1.
                This effectively prevents an explosion of the SDE in the beginning.
                Note that this does not affect the ODE sampling, which always uses
                times in [0,1].

        Returns:
            Drift function at a given time.
        """
        # analytical f(t) does not depend on noise_scale and is undefined at t = 1.
        return -input / torch.maximum(
            1 - times, torch.tensor(1 - effective_t_max).to(input)
        )

    def diffusion_fn(
        self, input: Tensor, times: Tensor, effective_t_max: float = 0.99
    ) -> Tensor:
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
            effective_t_max: Upper bound on time to avoid numerical issues at t=1.
                This effectively prevents an explosion of the SDE in the beginning.
                Note that this does not affect the ODE sampling, which always uses
                times in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        # analytical g(t) is undefined at t = 1.
        return torch.sqrt(
            2
            * (times + self.noise_scale)
            / torch.maximum(1 - times, torch.tensor(1 - effective_t_max).to(times))
        )

    def mean_t_fn(self, times: Tensor, effective_t_max: float = 0.99) -> Tensor:
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
            effective_t_max: Upper bound on time to avoid numerical issues at t=1.
                This prevents singularity at t=1 in the mean function (mean_t=0.).
                NOTE: This did affect the IID sampling as the analytical denoising
                moments run into issues (as mean_t=0) effectively makes it pure
                noise and equations are not well defined anymore. Alternatively
                we could also adapt the analytical denoising equations in
                `utils/vector_field_utils.py` to account for this case.

        Returns:
            Mean function at a given time.
        """
        times = torch.clamp(times, max=effective_t_max)
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
