# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.net_builders.flow import build_zuko_flow
from sbi.samplers.vi.vi_utils import move_all_tensor_to_device
from sbi.sbi_types import Shape, TorchTransform
from sbi.utils.torchutils import atleast_2d_float32_tensor


class AmortizedVIPosterior(NeuralPosterior):
    """Amortized Variational Inference posterior with conditional flow q(θ|x).

    Unlike standard VIPosterior which trains an unconditional q(θ) for a single
    observation x_o, this class trains a conditional q(θ|x) that can be used for
    any observation without retraining.

    The training optimizes the ELBO against a potential function (from NLE/NRE):
        ELBO(x) = E_q(θ|x)[log p(θ|x) - log q(θ|x)]

    where p(θ|x) is the unnormalized posterior from the potential function.
    """

    def __init__(
        self,
        potential_fn: BasePotential,
        prior: Distribution,
        q: Union[
            Literal["nsf", "maf", "nice"],
            ConditionalDensityEstimator,
            Callable[..., ConditionalDensityEstimator],
        ] = "nsf",
        theta_transform: Optional[TorchTransform] = None,
        device: Union[str, torch.device] = "cpu",
        num_transforms: int = 2,
        hidden_features: int = 32,
    ):
        """
        Args:
            potential_fn: Potential function from NLE/NRE defining log p(θ|x).
                Must support `set_x(x)` to change the conditioning observation.
            prior: Prior distribution p(θ).
            q: Conditional flow type ("nsf", "maf", "nice") or a custom
                ConditionalDensityEstimator, or a callable that builds one.
            theta_transform: Optional transform for θ. If None, uses identity.
            device: Device for training and sampling.
            num_transforms: Number of transforms in the flow (if using string q).
            hidden_features: Hidden layer size in the flow (if using string q).
        """
        super().__init__(potential_fn, theta_transform, device)

        self._device = device
        self._prior = prior
        self.potential_fn.device = device
        move_all_tensor_to_device(self.potential_fn, device)
        move_all_tensor_to_device(self._prior, device)

        # Store flow configuration for later building
        self._q_type = q
        self._num_transforms = num_transforms
        self._hidden_features = hidden_features

        # Will be set during training
        self._q: Optional[ConditionalDensityEstimator] = None
        self._trained = False

        self._purpose = (
            "It provides amortized variational inference to .sample() from the "
            "posterior for any observation x, and can evaluate log q(θ|x) with "
            ".log_prob()."
        )

    @property
    def q(self) -> Optional[ConditionalDensityEstimator]:
        """The conditional variational distribution q(θ|x)."""
        return self._q

    def _build_q(
        self,
        theta: Tensor,
        x: Tensor,
    ) -> ConditionalDensityEstimator:
        """Build the conditional flow q(θ|x).

        Args:
            theta: Sample of θ values for z-scoring (batch_size, θ_dim).
            x: Sample of x values for z-scoring (batch_size, x_dim).

        Returns:
            Conditional density estimator q(θ|x).
        """
        if isinstance(self._q_type, str):
            # Map to Zuko flow names
            flow_name_map = {"nsf": "NSF", "maf": "MAF", "nice": "NICE"}
            flow_name = flow_name_map.get(self._q_type, self._q_type.upper())

            return build_zuko_flow(
                flow_name,
                batch_x=theta,  # θ is what we model
                batch_y=x,  # x is the condition
                num_transforms=self._num_transforms,
                hidden_features=self._hidden_features,
            )
        elif isinstance(self._q_type, ConditionalDensityEstimator):
            return self._q_type
        elif callable(self._q_type):
            return self._q_type(theta, x)
        else:
            raise ValueError(f"Unknown q type: {type(self._q_type)}")

    def train(
        self,
        theta: Tensor,
        x: Tensor,
        n_particles: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.999,
        max_num_iters: int = 500,
        min_num_iters: int = 10,
        clip_value: float = 5.0,
        batch_size: int = 64,
        show_progress_bar: bool = True,
        check_for_convergence: bool = True,
        convergence_eps: float = 1e-4,
        retrain_from_scratch: bool = False,
    ) -> "AmortizedVIPosterior":
        """Train the conditional flow q(θ|x) by optimizing ELBO.

        Args:
            theta: Training θ values from simulations (num_sims, θ_dim).
                Used for z-scoring when building the flow.
            x: Training x values from simulations (num_sims, x_dim).
                The ELBO is optimized over batches of these x values.
            n_particles: Number of samples to estimate ELBO per x.
            learning_rate: Learning rate for Adam optimizer.
            gamma: Learning rate decay per iteration.
            max_num_iters: Maximum training iterations.
            min_num_iters: Minimum iterations before convergence check.
            clip_value: Gradient clipping threshold.
            batch_size: Number of x values per training batch.
            show_progress_bar: Whether to show progress.
            check_for_convergence: Whether to check for early stopping.
            convergence_eps: Convergence threshold for loss change.
            retrain_from_scratch: If True, rebuild the flow from scratch.

        Returns:
            self for method chaining.
        """
        theta = atleast_2d_float32_tensor(theta).to(self._device)
        x = atleast_2d_float32_tensor(x).to(self._device)

        # Build or rebuild the conditional flow
        if self._q is None or retrain_from_scratch:
            # Use subset for z-scoring
            n_zscore = min(100, len(theta))
            self._q = self._build_q(theta[:n_zscore], x[:n_zscore])
            self._q.to(self._device)

        # Setup optimizer
        optimizer = Adam(self._q.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        self._q.train()

        # Training loop
        if show_progress_bar:
            iters = tqdm(range(max_num_iters), desc="Amortized VI (ELBO)")
        else:
            iters = range(max_num_iters)

        losses = []
        for iteration in iters:
            optimizer.zero_grad()

            # Sample batch of x values
            idx = torch.randint(0, len(x), (batch_size,), device=self._device)
            x_batch = x[idx]

            # Compute ELBO
            loss = self._compute_elbo_loss(x_batch, n_particles)

            loss.backward()
            nn.utils.clip_grad_norm_(self._q.parameters(), clip_value)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            if show_progress_bar:
                assert isinstance(iters, tqdm)
                iters.set_postfix({"loss": f"{loss.item():.3f}"})

            # Check convergence
            if check_for_convergence and iteration > min_num_iters and len(losses) > 50:
                recent_mean = np.mean(losses[-50:])
                older_mean = np.mean(losses[-100:-50])
                if abs(recent_mean - older_mean) < convergence_eps:
                    if show_progress_bar:
                        print(f"\nConverged at iteration {iteration}")
                    break

        self._q.eval()
        self._trained = True

        return self

    def _compute_elbo_loss(self, x_batch: Tensor, n_particles: int) -> Tensor:
        """Compute negative ELBO loss for a batch of x values.

        Args:
            x_batch: Batch of observations (batch_size, x_dim).
            n_particles: Number of θ samples per x.

        Returns:
            Negative ELBO (scalar tensor).
        """
        assert self._q is not None, "q must be built before computing ELBO"
        batch_size = x_batch.shape[0]

        # Get conditional distributions for all x in batch
        embedded_x = self._q._embedding_net(x_batch)
        zuko_dists = self._q.net(embedded_x)

        # Reparameterized samples from q(θ|x)
        # Shape: (n_particles, batch_size, θ_dim)
        theta_samples = zuko_dists.rsample((n_particles,))

        # log q(θ|x) - Shape: (n_particles, batch_size)
        log_q = zuko_dists.log_prob(theta_samples)

        # Evaluate potential log p(θ|x) for each (θ, x) pair
        log_potential_list = []
        for i in range(batch_size):
            x_i = x_batch[i : i + 1]
            theta_i = theta_samples[:, i, :]  # (n_particles, θ_dim)
            self.potential_fn.set_x(x_i)
            log_pot_i = self.potential_fn(theta_i)  # (n_particles,)
            log_potential_list.append(log_pot_i)

        # Shape: (n_particles, batch_size)
        log_potential = torch.stack(log_potential_list, dim=1)

        # ELBO = E_q[log p(θ|x) - log q(θ|x)]
        # Loss = -ELBO
        elbo = (log_potential - log_q).mean()
        return -elbo

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """Sample from the amortized posterior q(θ|x).

        Args:
            sample_shape: Shape of samples to draw.
            x: Observation(s) to condition on. Shape (x_dim,) or (batch, x_dim).
                Required for amortized posterior.
            show_progress_bars: Unused, for API compatibility.

        Returns:
            Samples of shape (*sample_shape, *batch_shape, θ_dim).

        Raises:
            ValueError: If x is not provided or posterior not trained.
        """
        if x is None:
            raise ValueError(
                "AmortizedVIPosterior requires observation x for sampling. "
                "Use posterior.sample(shape, x=x_observed)."
            )

        if not self._trained or self._q is None:
            raise RuntimeError(
                "AmortizedVIPosterior must be trained before sampling. "
                "Call posterior.train(theta, x) first."
            )

        x = atleast_2d_float32_tensor(x).to(self._device)

        # Type narrowing assertion (self._q cannot be None after the check above)
        q = self._q

        with torch.no_grad():
            # samples shape: (*sample_shape, batch_size, θ_dim)
            samples = q.sample(torch.Size(sample_shape), condition=x)

        # If single x was provided (batch_size=1), squeeze that dimension
        if x.shape[0] == 1:
            samples = samples.squeeze(-2)

        return samples

    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        show_progress_bars: bool = False,
        max_sampling_batch_size: int = 10000,
    ) -> Tensor:
        """Sample from posterior for a batch of observations.

        This is efficient for amortized inference as all x values
        are processed in parallel.

        Args:
            sample_shape: Number of samples per observation.
            x: Batch of observations (num_obs, x_dim).
            show_progress_bars: Unused.
            max_sampling_batch_size: Unused (no batching needed for flow sampling).

        Returns:
            Samples of shape (*sample_shape, num_obs, θ_dim).
        """
        return self.sample(sample_shape, x=x, show_progress_bars=show_progress_bars)

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate log probability log q(θ|x).

        Args:
            theta: Parameter values to evaluate. Shape (θ_dim,) or (batch, θ_dim).
            x: Observation to condition on. Shape (x_dim,) or (batch, x_dim).
                Required for amortized posterior.

        Returns:
            Log probabilities of shape matching batch dimensions.
        """
        if x is None:
            raise ValueError(
                "AmortizedVIPosterior requires observation x for log_prob. "
                "Use posterior.log_prob(theta, x=x_observed)."
            )

        if not self._trained or self._q is None:
            raise RuntimeError(
                "AmortizedVIPosterior must be trained before evaluating log_prob. "
                "Call posterior.train(theta, x) first."
            )

        theta = atleast_2d_float32_tensor(theta).to(self._device)
        x = atleast_2d_float32_tensor(x).to(self._device)

        # ZukoFlow.log_prob expects input (sample_dim, batch_dim, event_dim)
        # and condition (batch_dim, event_dim)
        # Broadcast x to match theta batch size if needed
        if x.shape[0] == 1 and theta.shape[0] > 1:
            x = x.expand(theta.shape[0], -1)

        theta_input = theta.unsqueeze(0)  # (1, batch, θ_dim)
        log_prob = self._q.log_prob(theta_input, condition=x)  # (1, batch)

        return log_prob.squeeze(0)  # (batch,)

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: str = "posterior",
        num_init_samples: int = 1000,
        show_progress_bars: bool = False,
        **kwargs,
    ) -> Tensor:
        """Find maximum a posteriori (MAP) estimate.

        Args:
            x: Observation to condition on. Required.
            num_iter: Number of optimization iterations.
            num_to_optimize: Number of initial points to optimize from.
            learning_rate: Learning rate for gradient ascent.
            init_method: Initialization method ("posterior" or "prior").
            num_init_samples: Number of samples for initialization.
            show_progress_bars: Whether to show progress.

        Returns:
            MAP estimate of shape (θ_dim,).
        """
        if x is None:
            raise ValueError("AmortizedVIPosterior.map() requires observation x.")

        x = atleast_2d_float32_tensor(x).to(self._device)

        # Get initial points
        if init_method == "posterior":
            init_samples = self.sample((num_init_samples,), x=x)
        else:
            init_samples = self._prior.sample((num_init_samples,))

        # Evaluate and select best starting points
        log_probs = self.log_prob(init_samples, x=x)
        _, best_indices = torch.topk(log_probs, num_to_optimize)
        theta = init_samples[best_indices].clone().requires_grad_(True)

        # Optimize
        optimizer = Adam([theta], lr=learning_rate)

        for _ in range(num_iter):
            optimizer.zero_grad()
            log_prob = self.log_prob(theta, x=x)
            loss = -log_prob.sum()
            loss.backward()
            optimizer.step()

        # Return best
        final_log_probs = self.log_prob(theta.detach(), x=x)
        best_idx = final_log_probs.argmax()
        return theta[best_idx].detach()

    def to(self, device: Union[str, torch.device]) -> "AmortizedVIPosterior":
        """Move posterior to device."""
        self._device = device
        if self._q is not None:
            self._q.to(device)
        move_all_tensor_to_device(self.potential_fn, device)
        move_all_tensor_to_device(self._prior, device)
        return self
