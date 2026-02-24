# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
import warnings
from typing import Callable, Literal, Optional, Union

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators.base import ConditionalVectorFieldEstimator
from sbi.utils.vector_field_utils import VectorFieldNet


class ConditionalScoreEstimator(ConditionalVectorFieldEstimator):
    r"""Score matching for score-based generative models (e.g., denoising diffusion).
    The estimator neural network (this class) learns the score function, i.e., gradient
    of the conditional probability density with respect to the input, which can be used
    to generate samples from the target distribution by solving the SDE starting from
    the base (Gaussian) distribution.

    We assume the following SDE:
                        dx = A(t)xdt + B(t)dW,
    where A(t) and B(t) are the drift and diffusion functions, respectively, and dW is
    a Wiener process. This will lead to marginal distribution of the form:
                        p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)),
    where mean_t(t) and std_t(t) are the conditional mean and standard deviation at a
    given time t, respectively.

    References
    ----------
    .. [1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
           & Poole, B. (2020).
           "Score-based generative modeling through stochastic differential equations"
           *Advances in Neural Information Processing Systems*
           https://arxiv.org/abs/2011.13456

    .. [2] Ho, J., Jain, A., & Abbeel, P. (2020).
           "Denoising diffusion probabilistic models"
           *Advances in Neural Information Processing Systems, 33, 6840-6851*
           https://arxiv.org/abs/2006.11239

    .. [3] Song, Y., & Ermon, S. (2019).
           "Generative modeling by estimating gradients of the data distribution"
           *Advances in Neural Information Processing Systems, 32*
           https://arxiv.org/abs/1907.05600

    NOTE: This will follow the "noise matching" approach, we could also train a
    "denoising" network aiming to predict the original input given the noised input. We
    can still approx. the score by Tweedie's formula, but training might be easier.
    """

    # Whether the score is defined for this estimator.
    # Required for gradient-based methods.
    SCORE_DEFINED: bool = True
    # Whether the SDE functions - score, drift and diffusion -
    #  are defined for this estimator.
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
        weight_fn: Union[str, Callable] = "max_likelihood",
        beta_min: float = 0.01,
        beta_max: float = 10.0,
        mean_0: Union[Tensor, float] = 0.0,
        std_0: Union[Tensor, float] = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        r"""Score estimator class that estimates the
        conditional score function, i.e.,
        gradient of the density p(xt|x0).

        Args:
            net: Score estimator neural network with call signature: input, condition,
                and time (in [0,1])].
            input_shape: Shape of the input, i.e., the parameters.
            condition_shape: Shape of the conditioning variable.
            embedding_net: Network to embed the conditioning variable before passing it
                to the score network. If None, the identity function is used.
            weight_fn: Function to compute the weights over time. Can be one of the
                following:
                - "identity": constant weights (1.),
                - "max_likelihood": weights proportional to the diffusion function, or
                - a custom function that returns a Callable.
            mean_0: Starting mean of the target distribution.
            std_0: Starting standard deviation of the target distribution.
            t_min: Minimum time for diffusion (0 can be numerically unstable).
            t_max: Maximum time for diffusion.
        """

        # Starting mean and std of the target distribution (otherwise assumes 0,1).
        # This will be used to precondition the score network to improve training.
        if not isinstance(mean_0, Tensor):
            mean_0 = torch.tensor([mean_0])
        if not isinstance(std_0, Tensor):
            std_0 = torch.tensor([std_0])

        super().__init__(
            net,
            input_shape,
            condition_shape,
            mean_base=0.0,  # Will be updated after initialization
            std_base=1.0,  # Will be updated after initialization
            embedding_net=embedding_net,
            t_min=t_min,
            t_max=t_max,
        )

        # Min/max values for noise variance beta
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Set lambdas (variance weights) function.
        self._set_weight_fn(weight_fn)
        self.register_buffer("mean_0", mean_0.clone().detach())
        self.register_buffer("std_0", std_0.clone().detach())

        # Now that input_shape and mean_0, std_0 is set, we can compute the proper mean
        # and std for the "base" distribution.
        # Create t on the correct device to avoid CPU/GPU mismatch
        t_tensor = torch.as_tensor([t_max], device=self.mean_0.device)
        mean_t = self.approx_marginal_mean(t_tensor)
        std_t = self.approx_marginal_std(t_tensor)
        mean_t = torch.broadcast_to(mean_t, (1, *input_shape))
        std_t = torch.broadcast_to(std_t, (1, *input_shape))

        # Update the base distribution parameters
        self._mean_base = mean_t
        self._std_base = std_t

    def forward(self, input: Tensor, condition: Tensor, time: Tensor) -> Tensor:
        r"""Forward pass of the score estimator
        network to compute the conditional score
        at a given time.

        Args:
            input: Inputs to evaluate the score estimator on of shape
                    `(sample_dim_input, batch_dim_input, *event_shape_input)`.
            condition: Conditions of shape
                `(batch_dim_condition, *event_shape_condition)`.
            time: Time variable in [0,1] of shape
                `(batch_dim_time, *event_shape_time)`.

        Returns:
            Score (gradient of the density) at a given time, matches input shape.
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

        # Time dependent mean and std of the target distribution to z-score the input
        # and to approximate the score at the end of the diffusion.
        mean = self.approx_marginal_mean(time)
        std = self.approx_marginal_std(time)

        # As input to the neural net we want to have something that changes proportianl
        # to how the scores change
        time_enc = self.std_fn(time)

        # Time dependent z-scoring! Keeps input at similar scales
        input_enc = (input - mean) / std

        # Approximate score becoming exact for t -> t_max, "skip connection"
        score_gaussian = (input - mean) / std**2

        # Score prediction by the network
        # NOTE: To simplify usage of external networks, we will flatten the tensors
        # batch_shape to a single batch dimension.
        input_enc = input_enc.reshape(-1, *input_enc.shape[len(batch_shape) :])
        condition_emb = condition_emb.reshape(
            -1, *condition_emb.shape[len(batch_shape) :]
        )
        time_enc = time_enc.reshape(-1)
        score_pred = self.net(input_enc, condition_emb, time_enc)
        score_pred = score_pred.reshape(*batch_shape, *score_pred.shape[1:])

        # Output pre-conditioned score
        # The learnable part will be largly scaled at the beginning of the diffusion
        # and the gaussian part (where it should end up) will dominate at the end of
        # the diffusion.
        scale = self.mean_t_fn(time) / self.std_fn(time)
        output_score = -scale * score_pred - score_gaussian

        return output_score

    def score(self, input: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        """Score function of the score estimator.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Returns:
            Score function value.
        """
        return self(input=input, condition=condition, time=t)

    def loss(
        self,
        input: Tensor,
        condition: Tensor,
        times: Optional[Tensor] = None,
        control_variate=True,
        control_variate_threshold=0.3,
    ) -> Tensor:
        r"""Defines the denoising score matching loss (e.g., from Song et al., ICLR
        2021). A random diffusion time is sampled from [0,1], and the network is trained
        to predict thescore of the true conditional distribution given the noised input,
        which is equivalent to predicting the (scaled) Gaussian noise added to the
        input.

        Args:
            input: Input variable i.e. theta.
            condition: Conditioning variable.
            times: SDE time variable in [t_min, t_max]. If None, sampled via
                train_schedule() (which may be overridden by subclasses).
            control_variate: Whether to use a control variate to reduce the variance of
                the stochastic loss estimator.
            control_variate_threshold: Threshold for the control variate. If the std
                exceeds this threshold, the control variate is not used. This is because
                the control variate assumes a Taylor expansion of the score around the
                mean, which is not valid for large std.

        Returns:
            MSE between target score and network output, scaled by the weight function.

        """
        # Sample times from the Markov chain, use batch dimension
        if times is None:
            times = self.train_schedule(input.shape[0])
        times = times.to(input.device)

        # Sample noise.
        eps = torch.randn_like(input)

        # Compute mean and standard deviation.
        mean = self.mean_fn(input, times)
        std = self.std_fn(times)

        # Get noised input, i.e., p(xt|x0).
        input_noised = mean + std * eps

        # Compute true cond. score: -(noised_input - mean) / (std**2).
        score_target = -eps / std

        # Predict score from noised input and diffusion time.
        score_pred = self.forward(input_noised, condition, times)

        # Compute weights over time.
        weights = self.weight_fn(times)

        # Compute MSE loss between network output and true score.
        loss = torch.sum((score_pred - score_target) ** 2.0, dim=-1)

        # For times -> 0 this loss has high variance a standard method to reduce the
        # variance is to use a control variate i.e. a term that has zero expectation but
        # is strongly correlated with our objective.
        # Such a term can be derived by performing a 0 th order taylor expansion score
        # network around the mean (https://arxiv.org/pdf/2101.03288 for details).
        # NOTE: As it is a taylor expansion it will only work well for small std.

        if control_variate:
            D = input.shape[-1]
            score_mean_pred = self.forward(mean, condition, times)
            s = torch.squeeze(std, -1)

            # Loss terms that depend on eps
            term1 = 2 / s * torch.sum(eps * score_mean_pred, dim=-1)
            term2 = torch.sum(eps**2, dim=-1) / s**2
            # This term is the analytical expectation of the above term
            term3 = D / s**2

            control_variate = term3 - term1 - term2

            control_variate = torch.where(
                s < control_variate_threshold, control_variate, 0.0
            )

            loss = loss + control_variate

        return weights * loss

    def approx_marginal_mean(self, times: Tensor) -> Tensor:
        r"""Approximate the marginal mean of the target distribution at a given time.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Approximate marginal mean at a given time.
        """
        return self.mean_t_fn(times) * self.mean_0

    def approx_marginal_std(self, times: Tensor) -> Tensor:
        r"""Approximate the marginal standard deviation of the target distribution at a
        given time.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Approximate marginal standard deviation at a given time.
        """
        var = self.mean_t_fn(times) ** 2 * self.std_0**2 + self.std_fn(times) ** 2
        return torch.sqrt(var)

    def mean_t_fn(self, times: Tensor) -> Tensor:
        r"""Conditional mean function, E[xt|x0], specifying the "mean factor" at a given
        time, which is always multiplied by x0 to get the mean of the noise distribution
        , i.e., p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)).

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method is implemented in each individual SDE
            classes.
        """
        raise NotImplementedError

    def mean_fn(self, x0: Tensor, times: Tensor) -> Tensor:
        r"""Mean function of the SDE, which just multiplies the specific "mean factor"
        by the original input x0, to get the mean of the noise distribution, i.e.,
        p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)).

        Args:
            x0: Initial input data.
            times: SDE time variable in [0,1].

        Returns:
            Mean of the noise distribution at a given time.
        """
        return self.mean_t_fn(times) * x0

    def std_fn(self, times: Tensor) -> Tensor:
        r"""Standard deviation function of the noise distribution at a given time,

        i.e., p(xt|x0) = N(xt; mean_t(t)*x0, std_t(t)).

        Args:
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method is implemented in each individual SDE
            classes.
        """
        raise NotImplementedError

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Drift function, f(x,t), of the SDE described by dx = f(x,t)dt + g(x,t)dW.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method is implemented in each individual SDE
            classes.
        """
        raise NotImplementedError

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        r"""Diffusion function, g(x,t), of the SDE described by
                              dx = f(x,t)dt + g(x,t)dW.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Raises:
            NotImplementedError: This method is implemented in each individual SDE
            classes.
        """
        raise NotImplementedError

    def noise_schedule(self, times: Tensor) -> Tensor:
        """
        Map time to noise magnitude (beta for VP/SubVP, sigma for VE).

        This base implementation returns a linear beta schedule suitable for
        VP/SubVP SDEs where times are expected in [0, 1]. Subclasses
        (e.g., VEScoreEstimator) override for different schedules.

        Args:
            times: SDE times in [0, 1] for VP/SubVP, or [t_min, t_max] for VE.

        Returns:
            Beta (or sigma) schedule values at the given times.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * times

    def train_schedule(
        self,
        num_samples: int,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> Tensor:
        """
        Return diffusion times for training. Samples uniformly in [t_min, t_max].

        Can be overridden by subclasses (e.g., VEScoreEstimator supports lognormal).

        Args:
            num_samples: Number of time samples (typically batch size).
            t_min: Minimum time value. Defaults to self.t_min.
            t_max: Maximum time value. Defaults to self.t_max.

        Returns:
            Tensor of random times in [t_min, t_max].
        """
        t_min = self.t_min if t_min is None else t_min
        t_max = self.t_max if t_max is None else t_max

        return (
            torch.rand(num_samples, device=self._mean_base.device) * (t_max - t_min)
            + t_min
        )

    def solve_schedule(
        self,
        num_steps: int,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> Tensor:
        """
        Return a deterministic monotonic time grid for evaluation/solving.

        Can be overridden by subclasses (e.g., VEScoreEstimator supports power_law).

        Args:
            num_steps: Number of discretization steps.
            t_min: Minimum time value. Defaults to self.t_min.
            t_max: Maximum time value. Defaults to self.t_max.

        Returns:
            Tensor of shape (num_steps,) with times from t_max to t_min.
        """
        t_min = self.t_min if t_min is None else t_min
        t_max = self.t_max if t_max is None else t_max

        return torch.linspace(t_max, t_min, num_steps, device=self._mean_base.device)

    def _set_weight_fn(self, weight_fn: Union[str, Callable]):
        """Set the weight function.

        Args:
            weight_fn: Function to compute the weights over time. Can be one of the
            following:
                - "identity": constant weights (1.),
                - "max_likelihood": weights proportional to the diffusion function, or
                - a custom function that returns a Callable.
        """
        if weight_fn == "identity":
            self.weight_fn = self._identity_weight_fn
        elif weight_fn == "max_likelihood":
            self.weight_fn = self._max_likelihood_weight_fn
        elif weight_fn == "variance":
            self.weight_fn = self._variance_weight_fn
        elif callable(weight_fn):
            self.weight_fn = weight_fn
        else:
            raise ValueError(f"Weight function {weight_fn} not recognized.")

    def _identity_weight_fn(self, times):
        """Return ones for any time t."""
        return 1

    def _max_likelihood_weight_fn(self, times):
        """Return weights proportional to the diffusion function."""
        return self.diffusion_fn(torch.ones((1,), device=times.device), times) ** 2

    def _variance_weight_fn(self, times):
        """Return weights as the variance."""
        return self.std_fn(times) ** 2

    def ode_fn(self, input: Tensor, condition: Tensor, times: Tensor) -> Tensor:
        """ODE flow function of the score estimator.

        For reference, see Equation 13 in [1]_.

        Args:
            input: variable whose distribution is estimated.
            condition: Conditioning variable.
            t: Time.

        Returns:
            ODE flow function value at a given time.
        """
        score = self.score(input=input, condition=condition, t=times)
        f = self.drift_fn(input, times)
        g = self.diffusion_fn(input, times)
        v = f - 0.5 * g**2 * score
        return v


class VPScoreEstimator(ConditionalScoreEstimator):
    """Class for score estimators with variance preserving SDEs (i.e., DDPM).

    The SDE defining the diffusion process is characterized by the following hyper-
    parameters.

    Args:
        beta_min: This defines the smallest noise rate (i.e. how much the input is
            "scaled" down in contrast to how much noise is added) in the subVPSDE.
            Ideally, this would be 0, but score matching losses as employed in most
            diffusion models can become unstable if beta_min is zero. A small positive
            value is chosen to stabilize training (the smaller, the closer to an ODE;
            the larger, the easier to train but the noisier the resulting samples).
        beta_max: This defines the largest noise rate in the variance-preserving SDE.
            It sets the maximum variance introduced by the diffusion process; when
            integrated over [0, T], the marginal distribution at time T should
            approximate N(0, I).

    NOTE: Together with t_min and t_max they ultimatively define the loss function.
        Changing these might also require changing t_min and t_max to find a good
        tradeoff between bias and variance.
    """

    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        weight_fn: Union[str, Callable] = "max_likelihood",
        beta_min: float = 0.01,
        beta_max: float = 10.0,
        mean_0: Union[Tensor, float] = 0.0,
        std_0: Union[Tensor, float] = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        super().__init__(
            net,
            input_shape,
            condition_shape,
            embedding_net=embedding_net,
            weight_fn=weight_fn,
            mean_0=mean_0,
            std_0=std_0,
            beta_min=beta_min,
            beta_max=beta_max,
            t_min=t_min,
            t_max=t_max,
        )

    def mean_t_fn(self, times: Tensor) -> Tensor:
        """Conditional mean function for variance preserving SDEs.
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Conditional mean at a given time.
        """
        phi = torch.exp(
            -0.25 * times**2.0 * (self.beta_max - self.beta_min)
            - 0.5 * times * self.beta_min
        )
        for _ in range(len(self.input_shape)):
            phi = phi.unsqueeze(-1)
        return phi

    def std_fn(self, times: Tensor) -> Tensor:
        """Standard deviation function for variance preserving SDEs.
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Standard deviation at a given time.
        """
        std = 1.0 - torch.exp(
            -0.5 * times**2.0 * (self.beta_max - self.beta_min) - times * self.beta_min
        )
        for _ in range(len(self.input_shape)):
            std = std.unsqueeze(-1)
        return torch.sqrt(std)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Drift function for variance preserving SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        phi = -0.5 * self.noise_schedule(times)
        while len(phi.shape) < len(input.shape):
            phi = phi.unsqueeze(-1)
        return phi * input

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Diffusion function for variance preserving SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        g = torch.sqrt(self.noise_schedule(times))
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        return g


class SubVPScoreEstimator(ConditionalScoreEstimator):
    """Class for score estimators with sub-variance preserving SDEs.

    The SDE defining the diffusion process is characterized by the following hyper-
    parameters.

    Args:
        beta_min: This defines the smallest noise rate (i.e. how much the input is
            "scaled" down in contrast to how much noise is added) in the subVPSDE.
            Ideally, this would be 0, but score matching losses as employed in most
            diffusion models can become unstable if beta_min is zero. A small positive
            value is chosen to stabilize training (the smaller, the closer to an ODE;
            the larger, the easier to train but the noisier the resulting samples).
        beta_max: This defines the largest noise rate in the variance-preserving SDE.
            It sets the maximum variance introduced by the diffusion process; when
            integrated over [0, T], the marginal distribution at time T should
            approximate N(0, I).

    NOTE: Together with t_min and t_max they ultimatively define the loss function.
        Changing these might also require changing t_min and t_max to find a good
        tradeoff between bias and variance.
    """

    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        weight_fn: Union[str, Callable] = "max_likelihood",
        beta_min: float = 0.01,
        beta_max: float = 10.0,
        mean_0: float = 0.0,
        std_0: float = 1.0,
        t_min: float = 1e-2,
        t_max: float = 1.0,
    ) -> None:
        super().__init__(
            net,
            input_shape,
            condition_shape,
            embedding_net=embedding_net,
            weight_fn=weight_fn,
            beta_min=beta_min,
            beta_max=beta_max,
            mean_0=mean_0,
            std_0=std_0,
            t_min=t_min,
            t_max=t_max,
        )

    def mean_t_fn(self, times: Tensor) -> Tensor:
        """Conditional mean function for sub-variance preserving SDEs.
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Conditional mean at a given time.
        """
        phi = torch.exp(
            -0.25 * times**2.0 * (self.beta_max - self.beta_min)
            - 0.5 * times * self.beta_min
        )
        for _ in range(len(self.input_shape)):
            phi = phi.unsqueeze(-1)
        return phi

    def std_fn(self, times: Tensor) -> Tensor:
        """Standard deviation function for variance preserving SDEs.
        Args:
            times: SDE time variable in [0,1].

        Returns:
            Standard deviation at a given time.
        """
        std = 1.0 - torch.exp(
            -0.5 * times**2.0 * (self.beta_max - self.beta_min) - times * self.beta_min
        )
        for _ in range(len(self.input_shape)):
            std = std.unsqueeze(-1)
        return std

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Drift function for sub-variance preserving SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        phi = -0.5 * self.noise_schedule(times)

        while len(phi.shape) < len(input.shape):
            phi = phi.unsqueeze(-1)

        return phi * input

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Diffusion function for sub-variance preserving SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        g = torch.sqrt(
            torch.abs(
                self.noise_schedule(times)
                * (
                    1
                    - torch.exp(
                        -2 * self.beta_min * times
                        - (self.beta_max - self.beta_min) * times**2
                    )
                )
            )
        )

        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)

        return g


class VEScoreEstimator(ConditionalScoreEstimator):
    """Class for score estimators with variance exploding SDEs (i.e., NCSN / SMLD).

    The SDE defining the diffusion process is characterized by the following hyper-
    parameters.

    Args:
        sigma_min: Smallest noise level in the diffusion process. Ideally 0, but
            denoising score matching losses have exploding variance at 0, so a small
            positive value is used.
        sigma_max: Final standard deviation after full diffusion. Should be large
            enough that x_T ~ N(0, sigma_max) approximately.
        train_schedule: Time sampling strategy for training. "uniform" samples
            uniformly in [t_min, t_max]. "lognormal" uses log-normal sigma sampling
            per Karras et al. (2022), concentrating on intermediate noise levels.
        solve_schedule: Time discretization for ODE/SDE integration. "uniform" uses
            uniform linspace. "power_law" uses power-law spacing per Karras et al.
            (2022) Eq. 5, concentrating steps near low noise levels.
        lognormal_mean: Mean of log-normal distribution for train_schedule="lognormal".
            Default -1.2 from Karras et al. (2022).
        lognormal_std: Std of log-normal distribution for train_schedule="lognormal".
            Default 1.2 from Karras et al. (2022).
        power_law_exponent: Exponent (rho) for solve_schedule="power_law". Larger
            values concentrate more steps near low noise. Default 7 from Karras et al.

    Note:
        Together with t_min and t_max, these parameters define the loss function.
        Changing them might require adjusting t_min/t_max for optimal bias-variance
        tradeoff.

    References:
        Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based
        Generative Models" https://arxiv.org/abs/2206.00364

    """

    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        weight_fn: Union[str, Callable] = "max_likelihood",
        sigma_min: float = 1e-4,
        sigma_max: float = 10.0,
        mean_0: float = 0.0,
        std_0: float = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        train_schedule: Literal["uniform", "lognormal"] = "uniform",
        solve_schedule: Literal["uniform", "power_law"] = "uniform",
        lognormal_mean: float = -1.2,
        lognormal_std: float = 1.2,
        power_law_exponent: float = 7.0,
    ) -> None:
        # Validate sigma bounds (required for VE SDE math and log computations).
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max ({sigma_max}) must be greater than sigma_min ({sigma_min})"
            )

        # Validate schedule type strings at runtime.
        valid_train_schedules = ("uniform", "lognormal")
        if train_schedule not in valid_train_schedules:
            raise ValueError(
                f"train_schedule must be one of {valid_train_schedules}, "
                f"got '{train_schedule}'"
            )
        valid_solve_schedules = ("uniform", "power_law")
        if solve_schedule not in valid_solve_schedules:
            raise ValueError(
                f"solve_schedule must be one of {valid_solve_schedules}, "
                f"got '{solve_schedule}'"
            )

        # Validate lognormal parameters (only when schedule is used).
        if train_schedule == "lognormal" and lognormal_std <= 0:
            raise ValueError(f"lognormal_std must be positive, got {lognormal_std}")

        # Validate power-law exponent (only when schedule is used).
        if solve_schedule == "power_law" and power_law_exponent <= 0:
            raise ValueError(
                f"power_law_exponent must be positive, got {power_law_exponent}"
            )

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._train_schedule_type = train_schedule
        self._solve_schedule_type = solve_schedule
        # Log-normal distribution parameters from Karras et al. (2022).
        self.lognormal_mean = lognormal_mean
        self.lognormal_std = lognormal_std
        # Power-law exponent controls step concentration near low noise.
        self.power_law_exponent = power_law_exponent
        super().__init__(
            net,
            input_shape,
            condition_shape,
            embedding_net=embedding_net,
            weight_fn=weight_fn,
            mean_0=mean_0,
            std_0=std_0,
            t_min=t_min,
            t_max=t_max,
        )

    def mean_t_fn(self, times: Tensor) -> Tensor:
        """Conditional mean function for variance exploding SDEs, which is always 1.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Conditional mean at a given time.
        """
        phi = torch.ones_like(times, device=times.device)
        for _ in range(len(self.input_shape)):
            phi = phi.unsqueeze(-1)
        return phi

    def std_fn(self, times: Tensor) -> Tensor:
        """Standard deviation function for variance exploding SDEs.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Standard deviation at a given time.
        """
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** times
        for _ in range(len(self.input_shape)):
            std = std.unsqueeze(-1)
        return std

    def noise_schedule(self, times: Tensor) -> Tensor:
        """Geometric sigma schedule for variance exploding SDEs.

        For VE SDEs, the noise schedule is σ(t) = σ_min * (σ_max / σ_min)^t,
        which differs from the linear beta schedule used by VP/SubVP.

        Args:
            times: SDE times, typically in [t_min, t_max].

        Returns:
            Sigma schedule at given times.
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** times

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Drift function for variance exploding SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        return torch.tensor([0.0], device=input.device)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Diffusion function for variance exploding SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        sigma_ratio = self.sigma_max / self.sigma_min
        sigmas = self.noise_schedule(times)
        g = sigmas * math.sqrt(2 * math.log(sigma_ratio))

        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        return g.to(input.device)

    def train_schedule(
        self,
        num_samples: int,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> Tensor:
        """
        Return diffusion times for training.

        When train_schedule="uniform" (default): samples uniformly in [t_min, t_max].
        When train_schedule="lognormal": uses log-normal sigma sampling per
        Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based
        Generative Models", which concentrates training on intermediate noise levels.

        Args:
            num_samples: Number of time samples (typically batch size).
            t_min: Minimum time value. Defaults to self.t_min.
            t_max: Maximum time value. Defaults to self.t_max.

        Returns:
            Tensor of times in [t_min, t_max].

        Raises:
            ValueError: If t_min >= t_max.
        """
        t_min = self.t_min if t_min is None else t_min
        t_max = self.t_max if t_max is None else t_max

        if t_min >= t_max:
            raise ValueError(f"t_min ({t_min}) must be less than t_max ({t_max}).")

        if self._train_schedule_type == "uniform":
            # Uniform sampling (same as base class)
            return (
                torch.rand(num_samples, device=self._mean_base.device) * (t_max - t_min)
                + t_min
            )
        else:  # lognormal
            # Sample sigma from log-normal distribution
            # sigma = exp(P_mean + P_std * z) where z ~ N(0,1)
            log_sigma = self.lognormal_mean + self.lognormal_std * torch.randn(
                num_samples, device=self._mean_base.device
            )

            # Clamp in log-space BEFORE exponentiation to prevent NaN from
            # log(negative) when converting back to time. This is more numerically
            # stable than clamping sigma directly.
            log_sigma_min = math.log(self.sigma_min)
            log_sigma_max = math.log(self.sigma_max)

            # Check if excessive clamping needed (warn if >5% out of bounds).
            out_of_bounds = (
                ((log_sigma < log_sigma_min) | (log_sigma > log_sigma_max)).sum().item()
            )
            if out_of_bounds > num_samples * 0.05:
                warnings.warn(
                    f"Lognormal schedule: {out_of_bounds}/{num_samples} samples "
                    f"({100 * out_of_bounds / num_samples:.1f}%) clamped to "
                    f"[{self.sigma_min}, {self.sigma_max}]. Consider adjusting "
                    f"lognormal_mean={self.lognormal_mean} or "
                    f"lognormal_std={self.lognormal_std}.",
                    UserWarning,
                    stacklevel=2,
                )

            log_sigma_clamped = torch.clamp(log_sigma, log_sigma_min, log_sigma_max)

            # Convert log_sigma to time using VE's geometric relationship
            log_ratio = log_sigma_max - log_sigma_min  # = log(sigma_max / sigma_min)
            times = (log_sigma_clamped - log_sigma_min) / log_ratio

            # Final clamp to handle any remaining edge cases from t_min/t_max bounds
            return torch.clamp(times, t_min, t_max)

    def solve_schedule(
        self,
        num_steps: int,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> Tensor:
        """
        Return a deterministic time grid for ODE/SDE integration.

        When solve_schedule="uniform" (default): uniform linspace from t_max to t_min.
        When solve_schedule="power_law": power-law discretization per Karras et al.
        (2022), Eq. 5, which concentrates steps near low noise levels where fine
        details are resolved.

        Args:
            num_steps: Number of discretization steps.
            t_min: Minimum time value. Defaults to self.t_min.
            t_max: Maximum time value. Defaults to self.t_max.

        Returns:
            Tensor of shape (num_steps,) with times from t_max to t_min.

        Raises:
            ValueError: If t_min >= t_max.
        """
        t_min = self.t_min if t_min is None else t_min
        t_max = self.t_max if t_max is None else t_max

        if t_min >= t_max:
            raise ValueError(f"t_min ({t_min}) must be less than t_max ({t_max}).")

        if self._solve_schedule_type == "uniform":
            # Uniform spacing (same as base class)
            return torch.linspace(
                t_max, t_min, num_steps, device=self._mean_base.device
            )
        else:  # power_law
            # Power-law sigma schedule (Karras et al. 2022, Eq. 5):
            # σ_i = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
            rho = self.power_law_exponent
            rho_inv = 1.0 / rho

            # Compute sigma values using power-law interpolation
            steps = torch.linspace(0, 1, num_steps, device=self._mean_base.device)
            sigma_max_inv_rho = self.sigma_max**rho_inv
            sigma_min_inv_rho = self.sigma_min**rho_inv
            sigmas = (
                sigma_max_inv_rho + steps * (sigma_min_inv_rho - sigma_max_inv_rho)
            ) ** rho

            # Convert sigma to time using VE's geometric relationship
            log_ratio = math.log(self.sigma_max / self.sigma_min)
            times = torch.log(sigmas / self.sigma_min) / log_ratio

            # Ensure exact boundary values (avoid floating-point imprecision)
            times[0] = t_max
            times[-1] = t_min

            return times
