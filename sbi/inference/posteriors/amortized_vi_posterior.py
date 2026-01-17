# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.factory import ZukoFlowType
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

    Note:
        This class is not thread-safe. The underlying potential function uses
        stateful `set_x()` calls during training and ELBO computation. Do not
        use the same instance across multiple threads concurrently.
    """

    def __init__(
        self,
        potential_fn: BasePotential,
        prior: Distribution,
        flow_type: Union[
            ZukoFlowType,
            ConditionalDensityEstimator,
            Callable[..., ConditionalDensityEstimator],
        ] = ZukoFlowType.NSF,
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
            flow_type: Flow architecture for the variational distribution.
                Use ZukoFlowType.NSF, ZukoFlowType.MAF, or ZukoFlowType.NICE for
                built-in flows, or pass a custom ConditionalDensityEstimator or
                callable. If a callable is provided, it should accept (theta, x)
                training batches.
            theta_transform: Optional transform for θ. If None, uses identity.
            device: Device for training and sampling.
            num_transforms: Number of transforms in the flow (if using ZukoFlowType).
            hidden_features: Hidden layer size in the flow (if using ZukoFlowType).
        """
        super().__init__(potential_fn, theta_transform, device)

        self._device = device
        self._prior = prior
        self.potential_fn.device = device
        move_all_tensor_to_device(self.potential_fn, device)
        move_all_tensor_to_device(self._prior, device)

        # Store flow configuration for later building
        self._flow_type = flow_type
        self._num_transforms = num_transforms
        self._hidden_features = hidden_features

        # Will be set during training
        self._variational_distribution: Optional[ConditionalDensityEstimator] = None
        self._trained = False

        self._purpose = (
            "It provides amortized variational inference to .sample() from the "
            "posterior for any observation x, and can evaluate log q(θ|x) with "
            ".log_prob()."
        )

    @property
    def variational_distribution(self) -> Optional[ConditionalDensityEstimator]:
        """The learned conditional flow approximation q(θ|x) to the posterior p(θ|x).

        This is the distribution optimized during training to minimize the ELBO.
        Returns None if training has not been performed yet.
        """
        return self._variational_distribution

    @property
    def q(self) -> Optional[ConditionalDensityEstimator]:
        """Alias for variational_distribution (standard VI notation)."""
        return self.variational_distribution

    def _build_variational_distribution(
        self,
        theta: Tensor,
        x: Tensor,
    ) -> ConditionalDensityEstimator:
        """Build the conditional flow for the variational distribution.

        Args:
            theta: Sample of θ values for z-scoring (batch_size, θ_dim).
            x: Sample of x values for z-scoring (batch_size, x_dim).

        Returns:
            Conditional density estimator q(θ|x).
        """
        if isinstance(self._flow_type, ZukoFlowType):
            return build_zuko_flow(
                self._flow_type.value.upper(),
                batch_x=theta,  # θ is what we model
                batch_y=x,  # x is the condition
                num_transforms=self._num_transforms,
                hidden_features=self._hidden_features,
            )
        elif isinstance(self._flow_type, ConditionalDensityEstimator):
            return self._flow_type
        elif callable(self._flow_type):
            return self._flow_type(theta, x)
        else:
            raise ValueError(f"Unknown flow_type: {type(self._flow_type)}")

    def train(
        self,
        theta: Tensor,
        x: Tensor,
        n_particles: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.999,
        max_num_iters: int = 500,
        clip_value: float = 5.0,
        batch_size: int = 64,
        validation_fraction: float = 0.1,
        validation_batch_size: Optional[int] = None,
        validation_n_particles: Optional[int] = None,
        stop_after_iters: int = 20,
        show_progress_bar: bool = True,
        retrain_from_scratch: bool = False,
    ) -> "AmortizedVIPosterior":
        """Train the conditional flow q(θ|x) by optimizing ELBO.

        Args:
            theta: Training θ values from simulations (num_sims, θ_dim).
            x: Training x values from simulations (num_sims, x_dim).
            n_particles: Number of samples to estimate ELBO per x.
            learning_rate: Learning rate for Adam optimizer.
            gamma: Learning rate decay per iteration.
            max_num_iters: Maximum training iterations.
            clip_value: Gradient clipping threshold.
            batch_size: Number of x values per training batch.
            validation_fraction: Fraction of data to use for validation.
            validation_batch_size: Batch size for validation loss. Defaults to
                `batch_size`.
            validation_n_particles: Number of particles for validation loss.
                Defaults to `n_particles`.
            stop_after_iters: Stop training after this many iterations without
                improvement in validation loss.
            show_progress_bar: Whether to show progress.
            retrain_from_scratch: If True, rebuild the flow from scratch.

        Returns:
            self for method chaining.
        """
        theta = atleast_2d_float32_tensor(theta).to(self._device)
        x = atleast_2d_float32_tensor(x).to(self._device)

        # Validate inputs
        if theta.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch: theta has {theta.shape[0]} samples, "
                f"x has {x.shape[0]} samples. They must match."
            )
        if len(theta) == 0:
            raise ValueError("Training data cannot be empty.")
        if not torch.isfinite(theta).all():
            raise ValueError("theta contains NaN or Inf values.")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf values.")

        # Validate hyperparameters
        if not 0 < validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {validation_fraction}"
            )
        if n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {n_particles}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if validation_batch_size is None:
            validation_batch_size = batch_size
        if validation_n_particles is None:
            validation_n_particles = n_particles

        if validation_batch_size <= 0:
            raise ValueError(
                f"validation_batch_size must be positive, got {validation_batch_size}"
            )
        if validation_n_particles <= 0:
            raise ValueError(
                f"validation_n_particles must be positive, got {validation_n_particles}"
            )

        # Split into training and validation sets
        num_examples = len(theta)
        num_val = int(validation_fraction * num_examples)
        num_train = num_examples - num_val

        if num_val == 0:
            raise ValueError(
                "Validation set is empty. Increase validation_fraction or provide more "
                "training data."
            )
        if num_train < batch_size:
            raise ValueError(
                f"Training set size ({num_train}) is smaller than batch_size "
                f"({batch_size}). Reduce validation_fraction or batch_size."
            )

        permuted_indices = torch.randperm(num_examples, device=self._device)
        train_indices = permuted_indices[:num_train]
        val_indices = permuted_indices[num_train:]

        theta_train, x_train = theta[train_indices], x[train_indices]
        x_val = x[val_indices]  # Only x needed for validation (θ sampled from q)

        if validation_batch_size < x_val.shape[0]:
            val_batch_indices = torch.randperm(x_val.shape[0], device=self._device)[
                :validation_batch_size
            ]
        else:
            val_batch_indices = None

        # Build or rebuild the conditional flow (z-score on training data only)
        if self._variational_distribution is None or retrain_from_scratch:
            self._variational_distribution = self._build_variational_distribution(
                theta_train, x_train
            )
            self._variational_distribution.to(self._device)

        # Setup optimizer
        optimizer = Adam(self._variational_distribution.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        # Training loop with validation-based early stopping
        best_val_loss = float("inf")
        iters_since_improvement = 0
        best_state_dict = deepcopy(self._variational_distribution.state_dict())

        if show_progress_bar:
            iters = tqdm(range(max_num_iters), desc="Amortized VI (ELBO)")
        else:
            iters = range(max_num_iters)

        for iteration in iters:
            # Training step
            self._variational_distribution.train()
            optimizer.zero_grad()

            # Sample batch from training set
            idx = torch.randint(0, num_train, (batch_size,), device=self._device)
            x_batch = x_train[idx]

            train_loss = self._compute_elbo_loss(x_batch, n_particles)

            if not torch.isfinite(train_loss):
                raise RuntimeError(
                    f"Training loss became non-finite at iteration {iteration}: "
                    f"{train_loss.item()}. This indicates numerical instability. Try:\n"
                    f"  - Reducing learning_rate (currently {learning_rate})\n"
                    f"  - Reducing n_particles (currently {n_particles})\n"
                    f"  - Checking your potential_fn for numerical issues"
                )

            train_loss.backward()
            nn.utils.clip_grad_norm_(
                self._variational_distribution.parameters(), clip_value
            )
            optimizer.step()
            scheduler.step()

            # Compute validation loss
            self._variational_distribution.eval()
            with torch.no_grad():
                if val_batch_indices is None:
                    x_val_batch = x_val
                else:
                    x_val_batch = x_val[val_batch_indices]
                val_loss = self._compute_elbo_loss(
                    x_val_batch, validation_n_particles
                ).item()

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                iters_since_improvement = 0
                best_state_dict = deepcopy(self._variational_distribution.state_dict())
            else:
                iters_since_improvement += 1

            if show_progress_bar:
                assert isinstance(iters, tqdm)
                iters.set_postfix({
                    "train": f"{train_loss.item():.3f}",
                    "val": f"{val_loss:.3f}",
                })

            # Early stopping
            if iters_since_improvement >= stop_after_iters:
                if show_progress_bar:
                    print(f"\nConverged at iteration {iteration}")
                break

        # Restore best model
        self._variational_distribution.load_state_dict(best_state_dict)
        self._variational_distribution.eval()
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
        assert self._variational_distribution is not None, (
            "q must be built before computing ELBO"
        )
        batch_size = x_batch.shape[0]

        # Reparameterized samples from q(θ|x) with their log probabilities
        # Uses public API (sample_and_log_prob uses rsample_and_log_prob internally)
        # theta_samples shape: (n_particles, batch_size, θ_dim)
        # log_q shape: (n_particles, batch_size)
        theta_samples, log_q = self._variational_distribution.sample_and_log_prob(
            torch.Size((n_particles,)), condition=x_batch
        )

        # Vectorized evaluation of potential log p(θ|x) for all (θ, x) pairs
        # Flatten: (n_particles, batch_size, θ_dim) -> (n_particles * batch_size, θ_dim)
        theta_dim = theta_samples.shape[-1]
        theta_flat = theta_samples.reshape(n_particles * batch_size, theta_dim)

        # Repeat x to match: (batch_size, x_dim) -> (n_particles * batch_size, x_dim)
        # Each x[j] is repeated n_particles times to pair with theta[:, j, :]
        x_expanded = x_batch.repeat(n_particles, 1)

        # Set x_o for batched evaluation (x_is_iid=False: each θ paired with its x)
        self.potential_fn.set_x(x_expanded, x_is_iid=False)
        log_potential_flat = self.potential_fn(theta_flat)

        # Reshape: (n_particles * batch_size,) -> (n_particles, batch_size)
        log_potential = log_potential_flat.reshape(n_particles, batch_size)

        # ELBO = E_q[log p(θ|x) - log q(θ|x)]

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
                Uses default x if not provided.
            show_progress_bars: Unused, for API compatibility.

        Returns:
            Samples of shape (*sample_shape, θ_dim) for single observations, or
            (*sample_shape, batch_size, θ_dim) for batched observations.

        Raises:
            RuntimeError: If posterior not trained.
            ValueError: If neither x nor default x is set.
        """
        if not self._trained or self._variational_distribution is None:
            raise RuntimeError(
                "AmortizedVIPosterior must be trained before sampling. "
                "Call posterior.train(theta, x) first."
            )

        x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(self._device)

        # Type narrowing (self._variational_distribution is not None after check above)
        q = self._variational_distribution

        with torch.no_grad():
            # samples shape: (*sample_shape, batch_size, θ_dim)
            samples = q.sample(torch.Size(sample_shape), condition=x)

        # Match base posterior behavior: drop singleton x batch dimension.
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
                Uses default x if not provided.

        Returns:
            Log probabilities with shape according to broadcasting rules:
            - theta (N, d), x (N, d') → (N,) element-wise
            - theta (N, d), x (1, d') → (N,) broadcast x to all theta
            - theta (1, d), x (M, d') → (M,) broadcast theta to all x
            - theta (N, d), x (M, d') where N≠M and N,M>1 → ValueError

        Raises:
            ValueError: If batch dimensions are incompatible.
        """
        if not self._trained or self._variational_distribution is None:
            raise RuntimeError(
                "AmortizedVIPosterior must be trained before evaluating log_prob. "
                "Call posterior.train(theta, x) first."
            )

        theta = atleast_2d_float32_tensor(theta).to(self._device)
        x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(self._device)

        # Validate and broadcast batch dimensions
        batch_theta, batch_x = theta.shape[0], x.shape[0]

        if batch_theta != batch_x:
            if batch_x == 1:
                x = x.expand(batch_theta, -1)
            elif batch_theta == 1:
                theta = theta.expand(batch_x, -1)
            else:
                raise ValueError(
                    f"Incompatible batch sizes: theta has {batch_theta}, x has "
                    f"{batch_x}. Batch sizes must match or one must be 1."
                )

        # ZukoFlow.log_prob expects input (sample_dim, batch_dim, event_dim)
        # and condition (batch_dim, event_dim)
        theta_input = theta.unsqueeze(0)  # (1, batch, θ_dim)
        log_prob = self._variational_distribution.log_prob(
            theta_input, condition=x
        )  # (1, batch)

        return log_prob.squeeze(0)  # (batch,)

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
        **kwargs,
    ) -> Tensor:
        """Find maximum a posteriori (MAP) estimate.

        Args:
            x: Observation to condition on. If provided, sets default x.
            num_iter: Number of optimization iterations.
            num_to_optimize: Number of initial points to optimize from.
            learning_rate: Learning rate for gradient ascent.
            init_method: Initialization method ("posterior", "proposal", "prior").
            num_init_samples: Number of samples for initialization.
            save_best_every: Save best MAP value every N iterations.
            show_progress_bars: Whether to show progress.
            force_update: Whether to force MAP recomputation.

        Returns:
            MAP estimate of shape (θ_dim,).
        """
        if x is not None:
            x = atleast_2d_float32_tensor(x).to(self._device)
            self.set_default_x(x)

        return super().map(
            x=None,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
            force_update=force_update,
        )

    def to(self, device: Union[str, torch.device]) -> "AmortizedVIPosterior":
        """Move posterior to device."""
        self._device = device
        if self._variational_distribution is not None:
            self._variational_distribution.to(device)
        move_all_tensor_to_device(self.potential_fn, device)
        move_all_tensor_to_device(self._prior, device)
        return self
