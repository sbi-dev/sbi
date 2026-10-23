# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional, Union

import torch
from torch import Tensor, nn

from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.sbi_types import Shape

# Training and schedule settings selected by Arruda et al. (2026), Table S2 and
# Appendix C. They are fixed rather than exposed as arguments.
CYCLE_WEIGHT = 2.5
CYCLE_SWAP_PROBABILITY = 0.6
EQUAL_TIMES_FRACTION = 0.7
ADAPTIVE_WEIGHT_POWER = 0.25
ADAPTIVE_WEIGHT_OFFSET = 0.1
TIME_EXPONENT = 0.5
SCHEDULE_RHO = 7.0
SCHEDULE_SIGMA_MIN = 1e-3
# Rows per exact-Jacobian pass in `log_prob`, which bounds its memory.
LOG_PROB_CHUNK_SIZE = 1024


class MeanFlowEstimator(ConditionalDensityEstimator):
    r"""Conditional mean flow with an exact few-step density.

    The network predicts the average velocity $u(z, s, t, x)$ of the flow-matching
    ODE between times $s$ and $t$, where $t=0$ is the data and $t=1$ is standard
    normal noise. One evaluation gives the flow map
    $\Phi(z; s, t, x) = z + (t - s)\,u(z, s, t, x)$. Composing `num_steps` maps
    along a fixed schedule transports the data to the noise, and the density
    follows from the change of variables with the exact Jacobian of each map:

    .. math::
        \log q(z \mid x) = \log \mathcal{N}(z^{(S)}; 0, I)
        + \sum_{i=1}^{S} \log \left|\det \nabla_z G_i(z^{(i-1)})\right|.

    Sampling applies the maps backwards from noise to data. The density and the
    samples describe the same distribution only if the backward maps invert the
    forward maps, which the cycle loss encourages but does not guarantee.

    References:
        Arruda, Hasenauer, Radev. "Conditional Mean Flows: Few-Step Densities for
        Simulation-Based Inference." arXiv:2610.21657, 2026.

        Geng, Deng, Bai, Kolter, He. "Mean Flows for One-step Generative
        Modeling." NeurIPS, 2025.
    """

    def __init__(
        self,
        net: nn.Module,
        input_shape: torch.Size,
        condition_shape: torch.Size,
        embedding_net: Optional[nn.Module] = None,
        mean_input: Union[Tensor, float] = 0.0,
        std_input: Union[Tensor, float] = 1.0,
        num_steps: int = 2,
    ) -> None:
        r"""Initialize the mean flow.

        Args:
            net: Network with signature `net(z, condition_emb, s, t, head)`, where
                `head` is `"mean"` for the average velocity $u(z, s, t, x)$ or
                `"velocity"` for the instantaneous velocity $v(z, s, x)$.
            input_shape: Event shape of the modeled variable, must be 1D.
            condition_shape: Event shape of the condition.
            embedding_net: Embedding network for the condition.
            mean_input: Mean for z-scoring the modeled variable.
            std_input: Standard deviation for z-scoring the modeled variable.
            num_steps: Default number of flow-map steps for `log_prob` and
                `sample`.
        """
        if len(input_shape) != 1:
            raise ValueError(
                f"MeanFlowEstimator requires a 1D input, got shape {input_shape}."
            )
        super().__init__(net, input_shape, condition_shape)
        self._embedding_net = nn.Identity() if embedding_net is None else embedding_net
        dim = input_shape[0]
        self.register_buffer("_mean_input", _as_float_vector(mean_input, dim))
        self.register_buffer("_std_input", _as_float_vector(std_input, dim))
        self.num_steps = num_steps

    @property
    def embedding_net(self) -> nn.Module:
        r"""Return the embedding network of the condition."""
        return self._embedding_net

    @property
    def num_steps(self) -> int:
        r"""Default number of flow-map steps of `log_prob` and `sample`."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        self._num_steps = self._check_num_steps(value)

    def loss(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        r"""Return the mean-flow loss plus the cycle-consistency loss.

        Args:
            input: Inputs of shape `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *condition_event_shape)`.

        Returns:
            Loss of shape `(batch_dim,)`.
        """
        z_0 = self._standardize(input)
        condition_emb = self._embedding_net(condition)
        batch_size, dim = z_0.shape

        def uniform(*shape: int) -> Tensor:
            return torch.rand(shape, device=z_0.device, dtype=z_0.dtype)

        noise = torch.randn_like(z_0)
        s = uniform(batch_size) ** (1 + TIME_EXPONENT)
        t = uniform(batch_size) ** (1 + TIME_EXPONENT)
        t = torch.where(uniform(batch_size) < EQUAL_TIMES_FRACTION, s, t)
        z_s = (1 - s[:, None]) * z_0 + s[:, None] * noise
        target = noise - z_0

        velocity = self.net(z_s, condition_emb, s, s, "velocity")

        # MeanFlow identity: v(z_s, s) = u(z_s, s, t) - (t - s) d/ds u(z_s, s, t),
        # with d/ds = v . grad_z + partial_s at fixed t.
        def average_velocity(z: Tensor, s_: Tensor, t_: Tensor) -> Tensor:
            return self.net(z, condition_emb, s_, t_, "mean")

        u, du_ds = torch.func.jvp(
            average_velocity,
            (z_s, s, t),
            (velocity.detach(), torch.ones_like(s), torch.zeros_like(t)),
        )[:2]
        prediction = u - (t - s)[:, None] * du_ds.detach()
        squared_error = ((prediction - target) ** 2).sum(-1)
        weight = (squared_error.detach() + ADAPTIVE_WEIGHT_OFFSET) ** (
            -ADAPTIVE_WEIGHT_POWER
        )
        velocity_error = ((velocity - target) ** 2).sum(-1) / dim
        mean_flow_loss = weight * squared_error + velocity_error

        # Map s -> t and back, partly under another sample's condition, so that
        # the maps stay invertible away from the training pairs. The cycle gets its
        # own times without the t = s pairs, which would make the round trip exact.
        s_cycle = uniform(batch_size) ** (1 + TIME_EXPONENT)
        t_cycle = uniform(batch_size) ** (1 + TIME_EXPONENT)
        z_cycle = (1 - s_cycle[:, None]) * z_0 + s_cycle[:, None] * noise
        swap = uniform(batch_size) < CYCLE_SWAP_PROBABILITY
        permutation = torch.randperm(batch_size, device=z_0.device)
        cycle_emb = torch.where(
            swap[:, None], condition_emb[permutation], condition_emb
        )
        z_t = self._flow_map(z_cycle, cycle_emb, s_cycle, t_cycle)
        z_back = self._flow_map(z_t, cycle_emb, t_cycle, s_cycle)
        cycle_loss = ((z_back - z_cycle) ** 2).sum(-1) / dim

        return mean_flow_loss + CYCLE_WEIGHT * cycle_loss

    def log_prob(
        self,
        input: Tensor,
        condition: Tensor,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        r"""Return the log density from the exact Jacobians of the flow maps.

        Args:
            input: Inputs of shape `(sample_dim, batch_dim, *input_event_shape)` or
                `(batch_dim, *input_event_shape)`.
            condition: Conditions of shape `(batch_dim, *condition_event_shape)` or
                `(sample_dim, batch_dim, *condition_event_shape)`.
            num_steps: Number of flow-map steps. Defaults to `self.num_steps`.

        Returns:
            Log probabilities of shape `(sample_dim, batch_dim)`.
        """
        num_steps = self._check_num_steps(
            self.num_steps if num_steps is None else num_steps
        )
        input, condition, batch_dim = self._broadcast_and_align(input, condition)
        sample_dim = input.shape[0]
        z = self._standardize(input.reshape(-1, *self.input_shape))
        condition_emb = self._embedding_net(
            condition.reshape(-1, *self.condition_shape)
        )
        log_prob = torch.cat([
            self._log_prob_standardized(z_chunk, emb_chunk, num_steps)
            for z_chunk, emb_chunk in zip(
                z.split(LOG_PROB_CHUNK_SIZE),
                condition_emb.split(LOG_PROB_CHUNK_SIZE),
                strict=True,
            )
        ])
        log_prob = log_prob - torch.log(self._std_input).sum()
        return log_prob.reshape(sample_dim, batch_dim)

    def sample(
        self,
        sample_shape: Shape,
        condition: Tensor,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        r"""Return samples by applying the flow maps from noise to data.

        Args:
            sample_shape: Shape of the samples to return.
            condition: Conditions of shape `(batch_dim, *condition_event_shape)`.
            num_steps: Number of flow-map steps. Defaults to `self.num_steps`.

        Returns:
            Samples of shape `(*sample_shape, batch_dim, *input_event_shape)`.
        """
        num_steps = self._check_num_steps(
            self.num_steps if num_steps is None else num_steps
        )
        sample_shape = torch.Size(sample_shape)
        condition_emb = self._embedding_net(condition)
        batch_dim = condition_emb.shape[0]
        condition_emb = condition_emb.expand(*sample_shape, *condition_emb.shape)
        condition_emb = condition_emb.reshape(-1, condition_emb.shape[-1])

        z = torch.randn(
            condition_emb.shape[0],
            *self.input_shape,
            device=condition_emb.device,
            dtype=condition_emb.dtype,
        )
        times = self._time_schedule(num_steps, z)
        for s, t in zip(times.flip(0)[:-1], times.flip(0)[1:], strict=True):
            z = self._flow_map(z, condition_emb, s, t)

        samples = z * self._std_input + self._mean_input
        return samples.reshape(*sample_shape, batch_dim, *self.input_shape)

    def _log_prob_standardized(
        self, z: Tensor, condition_emb: Tensor, num_steps: int
    ) -> Tensor:
        r"""Return $\log q(z \mid x)$ of standardized `z` of shape `(n, dim)`."""
        times = self._time_schedule(num_steps, z)
        log_abs_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for s, t in zip(times[:-1], times[1:], strict=True):

            def step(z_i: Tensor, emb_i: Tensor, s=s, t=t) -> tuple[Tensor, Tensor]:
                z_next = self._flow_map(z_i[None], emb_i[None], s, t)[0]
                return z_next, z_next

            jacobian, z = torch.func.vmap(torch.func.jacrev(step, has_aux=True))(
                z, condition_emb
            )
            log_abs_det = log_abs_det + torch.linalg.slogdet(jacobian).logabsdet

        base_log_prob = torch.distributions.Normal(
            torch.zeros_like(z), torch.ones_like(z)
        ).log_prob(z)
        return base_log_prob.sum(-1) + log_abs_det

    def _flow_map(
        self, z: Tensor, condition_emb: Tensor, s: Tensor, t: Tensor
    ) -> Tensor:
        r"""Return $\Phi(z; s, t, x) = z + (t - s)\,u(z, s, t, x)$.

        `s` and `t` are scalars or have shape `(batch_dim,)`.
        """
        s = s.expand(z.shape[0])
        t = t.expand(z.shape[0])
        u = self.net(z, condition_emb, s, t, "mean")
        return z + (t - s)[:, None] * u

    def _standardize(self, input: Tensor) -> Tensor:
        return (input - self._mean_input) / self._std_input

    @staticmethod
    def _time_schedule(num_steps: int, reference: Tensor) -> Tensor:
        r"""Return times $0 = t_0 < \dots < t_S = 1$ of the EDM-style schedule.

        $\sigma_i = (\sigma_{\min}^{1/\rho} + \frac{i}{S}(1 - \sigma_{\min}^{1/\rho}))
        ^\rho$ and $t_i = (\sigma_i - \sigma_{\min}) / (1 - \sigma_{\min})$, which
        places more steps near the data.
        """
        fraction = torch.linspace(
            0, 1, num_steps + 1, device=reference.device, dtype=reference.dtype
        )
        sigma_min_root = SCHEDULE_SIGMA_MIN ** (1 / SCHEDULE_RHO)
        sigma = (sigma_min_root + fraction * (1 - sigma_min_root)) ** SCHEDULE_RHO
        times = (sigma - SCHEDULE_SIGMA_MIN) / (1 - SCHEDULE_SIGMA_MIN)
        times[0], times[-1] = 0.0, 1.0
        return times

    @staticmethod
    def _check_num_steps(num_steps: int) -> int:
        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise TypeError(f"num_steps must be an int, got {num_steps!r}.")
        if num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}.")
        return num_steps


def _as_float_vector(value: Union[Tensor, float], dim: int) -> Tensor:
    """Return `value` broadcast to shape `(dim,)`, keeping a floating dtype."""
    value = torch.as_tensor(value)
    if not value.is_floating_point():
        value = value.to(torch.get_default_dtype())
    return value.expand(dim).clone()
