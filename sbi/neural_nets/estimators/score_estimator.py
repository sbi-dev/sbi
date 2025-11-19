# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from typing import Callable, Optional, Union

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
            times: SDE time variable in [t_min, t_max]. Uniformly sampled if None.
            control_variate: Whether to use a control variate to reduce the variance of
                the stochastic loss estimator.
            control_variate_threshold: Threshold for the control variate. If the std
                exceeds this threshold, the control variate is not used. This is because
                the control variate assumes a Taylor expansion of the score around the
                mean, which is not valid for large std.

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
        self.beta_min = beta_min
        self.beta_max = beta_max
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
        self.beta_min = beta_min
        self.beta_max = beta_max
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
        phi = -0.5 * self._beta_schedule(times).to(input.device)

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
    """Class for score estimators with variance exploding SDEs (i.e., NCSN / SMLD).

    The SDE defining the diffusion process is characterized by the following hyper-
    parameters.


    Args:
       sigma_min: This defines the smallest "noise" level in the diffusion process. This
           ideally would be 0., but denoising score matching losses as employed in most
           diffusion models are ill-suited for this case as their variance explodes to
           infinity. Hence often a "small" value is chosen (the larger, the easier to
           learn but the "noisier" the end result if not addressed post-hoc).
       sigma_max: This is the final standard deviation after running the full diffusion
           process. Ideally this would approach ∞ such that x0 and xT are truly
           independent; it should be at least chosen such that x_T ~ N(0, sigma_max) at
           least approximately. If p(x0) for example has itself a very large variance,
           then you might have to increase this.

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
        sigma_min: float = 1e-4,
        sigma_max: float = 10.0,
        mean_0: float = 0.0,
        std_0: float = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
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
        return torch.tensor([0.0], device=input.device)

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

        return g.to(input.device)
