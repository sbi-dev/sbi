# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import math
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators.base import (
    ConditionalVectorFieldEstimator,
    MaskedConditionalVectorFieldEstimator,
)
from sbi.utils.vector_field_utils import MaskedVectorFieldNet, VectorFieldNet


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
        mean_t = self.approx_marginal_mean(torch.tensor([t_max]))
        std_t = self.approx_marginal_std(torch.tensor([t_max]))
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


class MaskedConditionalScoreEstimator(MaskedConditionalVectorFieldEstimator):
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
        net: Union[MaskedVectorFieldNet, nn.Module],
        input_shape: torch.Size,
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
            input_shape: Shape of the input.
            embedding_net: Network to embed the conditioning variable before passing it
                to the score network.
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
        if not isinstance(mean_0, Tensor):
            mean_0 = torch.tensor([mean_0])
        if not isinstance(std_0, Tensor):
            std_0 = torch.tensor([std_0])

        super().__init__(
            net,
            input_shape,
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

        # We estimate the mean and std of the source distribution at time t_max.
        mean_t = self.approx_marginal_mean(torch.tensor([t_max]))
        std_t = self.approx_marginal_std(torch.tensor([t_max]))
        mean_t = torch.broadcast_to(mean_t, (1, *input_shape))
        std_t = torch.broadcast_to(std_t, (1, *input_shape))

        # Update the base distribution parameters
        self._mean_base = mean_t
        self._std_base = std_t

    @property
    def embedding_net(self):
        r"""Return the embedding network."""
        return self._embedding_net

    def forward(
        self,
        input: Tensor,
        time: Tensor,
        condition_mask: Tensor,
        edge_mask: Optional[Tensor],
    ) -> Tensor:
        r"""Forward pass of the score estimator
        network to compute the conditional score
        at a given time.

        Args:
            input: Original data, x0.
            time: SDE time variable in [t_min, t_max].
            condition_masks: A boolean mask indicating the role of each node.
                - `True` (or `1`): The node at this position is observed and its
                features will be used for conditioning.
                - `False` (or `0`): The node at this position is latent and its
                parameters are subject to inference.
            edge_masks: A boolean mask defining the adjacency matrix of the directed
                acyclic graph (DAG) representing dependencies among nodes.
                - `True` (or `1`): An edge exists from the row node to the column node.
                - `False` (or `0`): No edge exists between these nodes.

        Returns:
            Score (gradient of the density) at a given time, matches input shape.
        """

        # Compute time-dependent mean and std for z-scoring
        # Handle scalar time (ndim=0) by unsqueezing to [1] for broadcasting
        if time.ndim == 0:
            time = time.unsqueeze(0)
        mean = self.approx_marginal_mean(time)  # [B, 1, F] or broadcastable
        std = self.approx_marginal_std(time)  # [B, 1, F] or broadcastable

        # Ensure mean and std are broadcastable to input shape
        while mean.dim() < input.dim():
            mean = mean.unsqueeze(-1)
        while std.dim() < input.dim():
            std = std.unsqueeze(-1)

        # Z-score the input
        input_enc = (input - mean) / std

        # "Skip connection" Gaussian score
        score_gaussian = (input - mean) / (std**2)

        # Model prediction
        score_pred = self.net(input_enc, time, condition_mask, edge_mask)  # [B, T, F]

        # Output pre-conditioned score (same scaling as in reference)
        scale = self.mean_t_fn(time) / self.std_fn(time)

        # Ensure scale is broadcastable to [B, T, F]
        while scale.dim() < input.dim():
            scale = scale.unsqueeze(-1)

        output_score = -scale * score_pred - score_gaussian

        return output_score

    def score(
        self,
        input: Tensor,
        t: Tensor,
        condition_mask: Tensor,
        edge_mask: Optional[Tensor],
    ) -> Tensor:
        r"""Score function of the score estimator.

        Args:
            input: Original data, x0.
            t: SDE time variable in [t_min, t_max].
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
            Score function value.
        """
        return self(
            input=input, time=t, condition_mask=condition_mask, edge_mask=edge_mask
        )

    def loss(
        self,
        input: Tensor,
        condition_mask: Tensor,
        edge_mask: Optional[Tensor],
        times: Optional[Tensor] = None,
        control_variate: bool = True,
        control_variate_threshold: float = 0.3,
        rebalance_loss: bool = True,
    ) -> Tensor:
        r"""Defines the denoising score matching loss (e.g., from Song et al., ICLR
        2021). A random diffusion time is sampled from [0,1], and the network is trained
        to predict thescore of the true conditional distribution given the noised input,
        which is equivalent to predicting the (scaled) Gaussian noise added to the
        input.

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
            control_variate: Whether to use a control variate to reduce the variance of
                the stochastic loss estimator.
            control_variate_threshold: Threshold for the control variate. If the std
                exceeds this threshold, the control variate is not used. This is because
                the control variate assumes a Taylor expansion of the score around the
                mean, which is not valid for large std.
            rebalance_loss: If True, the loss for each batch item is divided by the
                number of latent (unobserved) variables in that item. This is useful
                when the number of latent variables varies across the batch, as it
                prevents items with more latent variables from dominating the loss.

        Returns:
            MSE between target score and network output, scaled by the weight function.

        """
        device = input.device

        input_shape = input.shape
        B = input_shape[0]  # Batch size
        T = input_shape[1]

        # Sample times if not provided
        if times is None:
            times = torch.rand(B, device=device)
            times = times * (self.t_max - self.t_min)
            times = times + self.t_min  # [B,]

        # Sample noise
        eps = torch.randn_like(input)  # [B, T, F]

        # Compute mean and std for the SDE
        mean_t = self.mean_fn(input, times)  # [B, T, F]
        while mean_t.dim() < input.dim():
            mean_t = mean_t.unsqueeze(-1)
        std_t = self.std_fn(times)  # [B, 1, 1] or [B, 1] or [B]
        while std_t.dim() < input.dim():
            std_t = std_t.unsqueeze(-1)
        std_t = std_t.expand_as(input)  # [B, T, F]

        # Get noised input
        input_noised = mean_t + std_t * eps  # [B, T, F]

        # True score
        score_target = -eps / std_t

        condition_mask = condition_mask.bool()
        # Ensure condition_mask is broadcastable to input shape
        # input: [B, T] or [B, T, F]
        # condition_mask: [T] or [B, T]
        # We want to broadcast condition_mask to [B, T] for masking

        # Always ensure condition_mask is [B, T]
        if condition_mask.dim() == 1:
            # [T] -> [B, T]
            condition_mask_broadcast = condition_mask.unsqueeze(0).expand(B, T)
        elif condition_mask.dim() == 2:
            # [B, T]
            condition_mask_broadcast = condition_mask
        else:
            raise ValueError(
                f"condition_mask has incorrect dimensions: {condition_mask.shape} "
                f"condition_mask should have shape [T] or [B, T], where T is "
                f"the number of nodes and B is the batch size."
            )

        # Now broadcast to input shape
        if input.dim() == 2:
            # [B, T]
            pass
        elif input.dim() == 3:
            # [B, T, F]
            condition_mask_broadcast = condition_mask_broadcast.unsqueeze(-1).expand_as(
                input
            )
        else:
            raise ValueError(
                f"input has incorrect dimensions: {input.shape} "
                f"input should have shape [B, T] or [B, T, F]"
            )

        # Where mask is True (observed), use input; else use input_noised
        input_noised = torch.where(condition_mask_broadcast, input, input_noised)

        if edge_mask is not None:
            edge_mask = edge_mask.bool()
            if edge_mask.dim() == 2:
                # Shape is [T, T], expand to batch dimension
                edge_mask = edge_mask.unsqueeze(0).expand(B, T, T)
            elif edge_mask.dim() == 3:
                # Already correct shape [B, T, T]
                pass
            else:
                raise ValueError(
                    f"edge_mask has incorrect dimensions: {edge_mask.shape}"
                    f"edge_mask should have shape [T, T] or [B, T, T], where T is "
                    f"the number of nodes and B is the batch size."
                )

        # Model prediction
        score_pred = self.forward(
            input_noised,  # [B, T, F]
            times,  # [B,]
            condition_mask,  # [B, T]
            edge_mask,  # [B, T, T]
        )  # [B, T, F]

        # Compute MSE loss, mask out observed entries
        loss = (score_pred - score_target) ** 2.0
        loss = torch.where(condition_mask_broadcast, torch.zeros_like(loss), loss)

        # Since sbi expects loss-per-batch, I sum on both T and F
        if loss.dim() == 3:
            loss = loss.sum(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
        elif loss.dim() == 2:
            loss = loss.sum(dim=-1, keepdim=True)  # [B, 1]
        else:
            raise ValueError(f"Unexpected loss shape: {loss.shape}")

        # For times -> 0 this loss has high variance; a standard method to reduce the
        # variance is to use a control variate, i.e., a term that has zero expectation
        # but is strongly correlated with our objective.
        # Such a term can be derived by performing a 0th order Taylor expansion of the
        # score network around the mean
        # (see https://arxiv.org/pdf/2101.03288 for details).
        # NOTE: As it is a Taylor expansion, it will only work well for small std.

        if control_variate:
            # Compute score at the mean (mean_t), with same masks
            score_mean_pred = self.forward(
                mean_t, times, condition_mask, edge_mask
            )  # [B, T, F]

            # Compute terms for control variate
            # Only apply to unobserved (latent) nodes
            mask_f = ~condition_mask_broadcast.expand_as(eps)  # [B, T, F]

            # D: number of features per node
            D = eps.shape[-1]

            # term 1: 2/s * sum(eps * score_mean_pred) over F, masked
            term1 = (
                2
                / std_t
                * torch.sum(eps * score_mean_pred * mask_f, dim=-1).unsqueeze(-1)
            )
            # term 2: sum(eps^2) over F, masked, divided by s^2
            term2 = torch.sum((eps**2) * mask_f, dim=-1).unsqueeze(-1) / (std_t**2)
            # term 3: D / s^2, but only for unobserved nodes
            term3 = mask_f * (D / (std_t**2))

            # Sum over features, keep [B, T]
            control_variate_term = term3 - term1 - term2

            # Only apply control variate where std is small
            control_variate_term = torch.where(
                std_t < control_variate_threshold,
                control_variate_term,
                torch.zeros_like(control_variate_term),
            )

            # Sum over T and F to match loss shape [B, 1, 1]
            if control_variate_term.dim() == 3:
                control_variate_term = control_variate_term.sum(
                    dim=(-2, -1), keepdim=True
                )
            else:
                control_variate_term = control_variate_term.sum(dim=-1, keepdim=True)

            # Add to loss
            loss = loss + control_variate_term  # [B, 1, 1]

        if rebalance_loss:
            # Count number of unobserved (latent) elements per batch
            num_elements = (~condition_mask).sum(dim=-1, keepdim=True).clamp(min=1)
            loss = (
                loss / num_elements.unsqueeze(-1)
                if loss.dim() == 3
                else loss / num_elements
            )

        # Compute weights
        weights = self.weight_fn(times)
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=input.device, dtype=loss.dtype)

        # Ensure weights are broadcastable to [B, 1, 1]
        while weights.dim() < loss.dim():
            weights = weights.unsqueeze(-1)

        # Scale loss by weights
        loss = weights.clone().detach() * loss

        if loss.dim() == 3:
            loss = loss.squeeze(dim=(-2, -1))
        elif loss.dim() == 2:
            loss = loss.squeeze(dim=-1)
        else:
            raise ValueError(f"Unexpected loss shape: {loss.shape}")

        return loss

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

    def _set_weight_fn(self, weight_fn: Union[str, Callable]):
        r"""Set the weight function.

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

    def ode_fn(
        self,
        input: Tensor,
        times: Tensor,
        condition_mask: Tensor,
        edge_mask: Tensor,
    ) -> Tensor:
        r"""ODE flow function of the score estimator.

        For reference, see Equation 13 in [1]_.

        Args:
            input: variable whose distribution is estimated.
            times: Time.
            condition_mask: Mask indicating which nodes are observed (conditioned on)
                or latent (conditioned off).
            edge_mask: Mask for edges in the DAG, i.e., dependencies between
                variables (nodes).

        Returns:
            ODE flow function value at a given time.
        """
        score = self.score(
            input=input, t=times, condition_mask=condition_mask, edge_mask=edge_mask
        )
        f = self.drift_fn(input, times)
        g = self.diffusion_fn(input, times)
        v = f - 0.5 * g**2 * score
        return v


class VariancePreservingSDE:
    """
    Mixin for variance preserving SDE.

    Expects the child class to define:
        - self.input_shape
        - self.beta_max
        - self.beta_min
    """

    input_shape: torch.Size
    beta_max: float
    beta_min: float

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


class VPScoreEstimator(ConditionalScoreEstimator, VariancePreservingSDE):
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
        return VariancePreservingSDE.mean_t_fn(self, times)

    def std_fn(self, times: Tensor) -> Tensor:
        return VariancePreservingSDE.std_fn(self, times)

    def _beta_schedule(self, times: Tensor) -> Tensor:
        return VariancePreservingSDE._beta_schedule(self, times)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return VariancePreservingSDE.drift_fn(self, input, times)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return VariancePreservingSDE.diffusion_fn(self, input, times)


class MaskedVPScoreEstimator(MaskedConditionalScoreEstimator, VariancePreservingSDE):
    """Class for score estimators with variance preserving SDEs (i.e., DDPM)."""

    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
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
            embedding_net=embedding_net,
            weight_fn=weight_fn,
            mean_0=mean_0,
            std_0=std_0,
            t_min=t_min,
            t_max=t_max,
        )

    def mean_t_fn(self, times: Tensor) -> Tensor:
        return VariancePreservingSDE.mean_t_fn(self, times)

    def std_fn(self, times: Tensor) -> Tensor:
        return VariancePreservingSDE.std_fn(self, times)

    def _beta_schedule(self, times: Tensor) -> Tensor:
        return VariancePreservingSDE._beta_schedule(self, times)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return VariancePreservingSDE.drift_fn(self, input, times)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return VariancePreservingSDE.diffusion_fn(self, input, times)


class SubVariancePreservingSDE:
    """
    Mixin for sub variance preserving SDE.

    Expects the child class to define:
        - self.input_shape
        - self.beta_max
        - self.beta_min
    """

    input_shape: torch.Size
    beta_max: float
    beta_min: float

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


class SubVPScoreEstimator(ConditionalScoreEstimator, SubVariancePreservingSDE):
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
        return SubVariancePreservingSDE.mean_t_fn(self, times)

    def std_fn(self, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE.std_fn(self, times)

    def _beta_schedule(self, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE._beta_schedule(self, times)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE.drift_fn(self, input, times)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE.diffusion_fn(self, input, times)


class MaskedSubVPScoreEstimator(
    MaskedConditionalScoreEstimator, SubVariancePreservingSDE
):
    """Class for score estimators with sub-variance preserving SDEs."""

    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
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
            embedding_net=embedding_net,
            weight_fn=weight_fn,
            mean_0=mean_0,
            std_0=std_0,
            t_min=t_min,
            t_max=t_max,
        )

    def mean_t_fn(self, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE.mean_t_fn(self, times)

    def std_fn(self, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE.std_fn(self, times)

    def _beta_schedule(self, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE._beta_schedule(self, times)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE.drift_fn(self, input, times)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return SubVariancePreservingSDE.diffusion_fn(self, input, times)


class VarianceExplodingSDE:
    """
    Mixin for variance exploding SDE.

    Expects the child class to define:
        - self.input_shape
        - self.sigma_min
        - self.sigma_max
    """

    # Expected attributes at child
    input_shape: torch.Size
    sigma_max: float
    sigma_min: float

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


class VEScoreEstimator(ConditionalScoreEstimator, VarianceExplodingSDE):
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
        return VarianceExplodingSDE.mean_t_fn(self, times)

    def std_fn(self, times: Tensor) -> Tensor:
        return VarianceExplodingSDE.std_fn(self, times)

    def _sigma_schedule(self, times: Tensor) -> Tensor:
        return VarianceExplodingSDE._sigma_schedule(self, times)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return VarianceExplodingSDE.drift_fn(self, input, times)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return VarianceExplodingSDE.diffusion_fn(self, input, times)


class MaskedVEScoreEstimator(MaskedConditionalScoreEstimator, VarianceExplodingSDE):
    def __init__(
        self,
        net: VectorFieldNet,
        input_shape: torch.Size,
        embedding_net: nn.Module = nn.Identity(),
        weight_fn: Union[str, Callable] = "max_likelihood",
        sigma_min: float = 1e-4,
        sigma_max: float = 10.0,
        mean_0: Union[Tensor, float] = 0.0,
        std_0: Union[Tensor, float] = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        super().__init__(
            net,
            input_shape,
            embedding_net=embedding_net,
            weight_fn=weight_fn,
            mean_0=mean_0,
            std_0=std_0,
            t_min=t_min,
            t_max=t_max,
        )

    def mean_t_fn(self, times: Tensor) -> Tensor:
        return VarianceExplodingSDE.mean_t_fn(self, times)

    def std_fn(self, times: Tensor) -> Tensor:
        return VarianceExplodingSDE.std_fn(self, times)

    def _sigma_schedule(self, times: Tensor) -> Tensor:
        return VarianceExplodingSDE._sigma_schedule(self, times)

    def drift_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return VarianceExplodingSDE.drift_fn(self, input, times)

    def diffusion_fn(self, input: Tensor, times: Tensor) -> Tensor:
        return VarianceExplodingSDE.diffusion_fn(self, input, times)
