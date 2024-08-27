# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from math import pi
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators.base import ConditionalVectorFieldEstimator


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

    Relevant literature:
    - Score-based generative modeling through SDE: https://arxiv.org/abs/2011.13456
    - Denoising diffusion probabilistic models: https://arxiv.org/abs/2006.11239
    - Noise conditional score networks: https://arxiv.org/abs/1907.05600

    NOTE: This will follow the "noise matching" approach, we could also train a
    "denoising" network aiming to predict the original input given the noised input. We
    can still approx. the score by Tweedie's formula, but training might be easier.
    """

    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "max_likelihood",
        mean_0: Union[Tensor, float] = 0.0,
        std_0: Union[Tensor, float] = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        r"""Score estimator class that estimates the conditional score function, i.e.,
        gradient of the density p(xt|x0).

        Args:
            net: Score estimator neural network with call signature: input, condition,
                and time (in [0,1])].
            condition_shape: Shape of the conditioning variable.
            weight_fn: Function to compute the weights over time. Can be one of the
                following:
                - "identity": constant weights (1.),
                - "max_likelihood": weights proportional to the diffusion function, or
                - a custom function that returns a Callable.

        """
        super().__init__(net, input_shape, condition_shape)

        # Set lambdas (variance weights) function.
        self._set_weight_fn(weight_fn)

        # Min time for diffusion (0 can be numerically unstable).
        self.t_min = t_min
        self.t_max = t_max

        # Starting mean and std of the target distribution (otherwise assumes 0,1).
        # This will be used to precondition the score network to improve training.
        if not isinstance(mean_0, Tensor):
            mean_0 = torch.tensor([mean_0])
        if not isinstance(std_0, Tensor):
            std_0 = torch.tensor([std_0])

        self.register_buffer("mean_0", mean_0.clone().detach())
        self.register_buffer("std_0", std_0.clone().detach())

        # We estimate the mean and std of the source distribution at time t_max.
        mean_t = self.approx_marginal_mean(torch.tensor([t_max]))
        std_t = self.approx_marginal_std(torch.tensor([t_max]))
        self.register_buffer("mean_t", mean_t)
        self.register_buffer("std_t", std_t)

    def forward(self, input: Tensor, condition: Tensor, time: Tensor) -> Tensor:
        r"""Forward pass of the score estimator network to compute the conditional score
        at a given time.

        Args:
            input: Original data, x0. (input_batch_shape, *input_shape)
            condition: Conditioning variable. (condition_batch_shape, *condition_shape)
            times: SDE time variable in [0,1].

        Returns:
            Score (gradient of the density) at a given time, matches input shape.
        """
        batch_shape = torch.broadcast_shapes(
            input.shape[: -len(self.input_shape)],
            condition.shape[: -len(self.condition_shape)],
        )

        input = torch.broadcast_to(input, batch_shape + self.input_shape)
        condition = torch.broadcast_to(condition, batch_shape + self.condition_shape)
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
        score_pred = self.net(input_enc, condition, time_enc)

        # Output pre-conditioned score
        # The learnable part will be largly scaled at the beginning of the diffusion
        # and the gaussian part (where it should end up) will dominate at the end of
        # the diffusion.
        scale = self.mean_t_fn(time) / self.std_fn(time)
        output_score = -scale * score_pred - score_gaussian

        return output_score

    def loss(
        self,
        input: Tensor,
        condition: Tensor,
        times: Optional[Tensor] = None,
        control_variate=True,
        control_variate_threshold=torch.inf,
    ) -> Tensor:
        r"""Defines the denoising score matching loss (e.g., from Song et al., ICLR
        2021). A random diffusion time is sampled from [0,1], and the network is trained
        to predict thescore of the true conditional distribution given the noised input,
        which is equivalent to predicting the (scaled) Gaussian noise added to the
        input.

        Args:
            input: Input variable i.e. theta.
            condition: Conditioning variable.
            times: SDE time variable in [t_min, t_max]. Uniformly sampled if None.
            control_variate: Whether to use a control variate to reduce the variance of
                the stochastic loss estimator.
            control_variate_threshold: Threshold for the control variate. If the std
                exceeds this threshold, the control variate is not used.

        Returns:
            MSE between target score and network output, scaled by the weight function.

        """
        # Sample diffusion times.
        if times is None:
            times = (
                torch.rand(input.shape[0], device=input.device)
                * (self.t_max - self.t_min)
                + self.t_min
            )

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
        vars = self.mean_t_fn(times) ** 2 * self.std_0**2 + self.std_fn(times) ** 2
        return torch.sqrt(vars)

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
            self.weight_fn = lambda times: 1
        elif weight_fn == "max_likelihood":
            self.weight_fn = (
                lambda times: self.diffusion_fn(
                    torch.ones((1,), device=times.device), times
                )
                ** 2
            )
        elif weight_fn == "variance":
            self.weight_fn = lambda times: self.std_fn(times) ** 2
        elif callable(weight_fn):
            self.weight_fn = weight_fn
        else:
            raise ValueError(f"Weight function {weight_fn} not recognized.")


class VPScoreEstimator(ConditionalScoreEstimator):
    """Class for score estimators with variance preserving SDEs (i.e., DDPM)."""

    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "max_likelihood",
        beta_min: float = 0.01,
        beta_max: float = 10.0,
        mean_0: Union[Tensor, float] = 0.0,
        std_0: Union[Tensor, float] = 1.0,
        t_min: float = 1e-5,
        t_max: float = 1.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(
            net,
            input_shape,
            condition_shape,
            mean_0=mean_0,
            std_0=std_0,
            weight_fn=weight_fn,
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

    def _beta_schedule(self, times: Tensor) -> Tensor:
        """Linear beta schedule for mean scaling in variance preserving SDEs.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Beta schedule at a given time.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * times

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Drift function for variance preserving SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        phi = -0.5 * self._beta_schedule(times)
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
        g = torch.sqrt(self._beta_schedule(times))
        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)
        return g


class SubVPScoreEstimator(ConditionalScoreEstimator):
    """Class for score estimators with sub-variance preserving SDEs."""

    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "max_likelihood",
        beta_min: float = 0.01,
        beta_max: float = 10.0,
        mean_0: float = 0.0,
        std_0: float = 1.0,
        t_min: float = 1e-2,
        t_max: float = 1.0,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(
            net,
            input_shape,
            condition_shape,
            weight_fn=weight_fn,
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

    def _beta_schedule(self, times: Tensor) -> Tensor:
        """Linear beta schedule for mean scaling in sub-variance preserving SDEs.
        (Same as for variance preserving SDEs.)

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Beta schedule at a given time.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * times

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Drift function for sub-variance preserving SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Drift function at a given time.
        """
        phi = -0.5 * self._beta_schedule(times)

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
                self._beta_schedule(times)
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
    """Class for score estimators with variance exploding SDEs (i.e., NCSN / SMLD)."""

    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        weight_fn: Union[str, Callable] = "max_likelihood",
        sigma_min: float = 1e-5,
        sigma_max: float = 5.0,
        mean_0: float = 0.0,
        std_0: float = 1.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(
            net,
            input_shape,
            condition_shape,
            weight_fn=weight_fn,
            mean_0=mean_0,
            std_0=std_0,
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

    def _sigma_schedule(self, times: Tensor) -> Tensor:
        """Geometric sigma schedule for variance exploding SDEs.

        Args:
            times: SDE time variable in [0,1].

        Returns:
            Sigma schedule at a given time.
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
        return torch.tensor([0.0])

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        """Diffusion function for variance exploding SDEs.

        Args:
            input: Original data, x0.
            times: SDE time variable in [0,1].

        Returns:
            Diffusion function at a given time.
        """
        g = self._sigma_schedule(times) * math.sqrt(
            (2 * math.log(self.sigma_max / self.sigma_min))
        )

        while len(g.shape) < len(input.shape):
            g = g.unsqueeze(-1)

        return g


class GaussianFourierTimeEmbedding(nn.Module):
    """Gaussian random features for encoding time steps.

    This is to be used as a utility for score-matching."""

    def __init__(self, embed_dim=256, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, times: Tensor):
        times_proj = times[:, None] * self.W[None, :] * 2 * pi
        embedding = torch.cat([torch.sin(times_proj), torch.cos(times_proj)], dim=-1)
        return torch.squeeze(embedding, dim=1)
