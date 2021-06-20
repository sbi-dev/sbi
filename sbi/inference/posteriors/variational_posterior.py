from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import torch
from torch.distributions import Distribution
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape, Array
from sbi.utils import del_entries
from sbi.utils.torchutils import (
    ScalarFloat,
    ensure_theta_batched,
    ensure_x_batched,
    atleast_2d_float32_tensor,
)

from sbi.vi.build_q import build_q, build_optimizer
from sbi.vi.divergence_optimizers import DivergenceOptimizer
from sbi.vi.sampling import (
    importance_resampling,
    independent_mh,
    rejection_sampling,
    random_direction_slice_sampler,
    paretto_smoothed_weights,
    clamp_weights,
)

from tqdm import tqdm


class VariationalPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNLE.<br/><br/>
    SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
    `SNLE_Posterior` class performs variational inference to approximate posterior $q(\theta|x) \approx p(\theta|x)$.
    Where $q$ is a distribution of a specific variational family e.g. a Normalizing flow.
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        prior: Distribution,
        x_shape: torch.Size,
        sample_with: str = "mcmc",
        device: str = "cpu",
        flow_paras: dict = {},
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            flow: Flow used for variational family one of: [iaf, planar, radial,
            affine_coupling, spline, spline_autoregressive, spline_coupling]
            device: Training device, e.g., cpu or cuda:0.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__", "flow_paras"))

        super().__init__(**kwargs)
        self._purpose = f"Variational Posterior approximation"
        self._flow_paras = flow_paras

        self._q = build_q(self._prior.event_shape, self._prior.support, **flow_paras)
        self._optimizer = build_optimizer(self, "elbo")
        self._loss = "elbo"

    @property
    def q(self):
        """Variational distribution that will be learned"""
        return self._q

    @q.setter
    def q(self, q: Distribution) -> None:
        """See `set_default_x`."""
        self._set_q(q)

    @property
    def optimizer(self):
        """ Divergence optimizer for variational inference"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: DivergenceOptimizer) -> None:
        """See `set_default_x`."""
        if not isinstance(optimizer, DivergenceOptimizer):
            raise ValueError(
                "This class relies on a DivergenceOptimizer which minimize some divergence."
            )
        self._optimier = optimizer

    def _set_q(self, q: Distribution) -> None:
        """ Sets the distributions and checks some important properties. """
        if not isinstance(q, Distribution):
            raise ValueError(
                "We only support PyTorch distributions, please wrap your distributions as one!"
            )
        self._q = q

    def predictive(self, x: Tensor, method: str = "naive", num_samples: int = 10000):
        """ Estimates the predictive distributions log prob """
        return self.expectation(
            lambda theta: self.net.log_prob(x.repeat(num_samples, 1), context=theta),
            method=method,
            num_samples=num_samples,
        )

    def predictive_sample(self, shape: torch.Size(), **kwargs):
        thetas = self.sample(shape, **kwargs)
        xs = self.net.sample(context=thetas)
        return xs

    def expectation(
        self,
        f: Callable,
        x: Optional[Tensor] = None,
        method: str = "naive",
        num_samples: int = 10000,
    ):
        """Computes the expectation with respect to the posterior E_q[f(X)]

        Args:
        f: Function for which we will compute the expectation
        method: Method either naive (just using the variatioanl posterior), is
        (importance sampling) or psis (pareto smoothed importance sampling).
        """
        samples = self.sample((num_samples,))
        if method == "naive":
            return f(samples).mean(0)
        else:
            with torch.no_grad():
                x_obs = atleast_2d_float32_tensor(self._x_else_default_x(x))
                x_obs = ensure_x_batched(x_obs)
                obs = x_obs.repeat(num_samples, 1)
                logweights = (
                    self.net.log_prob(obs, samples)
                    + self._prior.log_prob(samples)
                    - self._q.log_prob(samples)
                )
                weights = torch.exp(logweights)
            if method == "is":
                pass
            elif method == "psis":
                weights = paretto_smoothed_weights(weights)
            elif method == "clamped":
                weights = clamp_weights(weights)
            else:
                raise NotImplementedError(
                    "We only have the methods naive, is, psis and clamped."
                )
            weights /= weights.sum()
            return torch.sum(f(samples) * weights.unsqueeze(-1), 0)

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,
    ) -> Tensor:
        """
        Returns the log-probability of $q(\theta|x).$

        This corresponds to an normalized variational posterior log-probability.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `(len(θ),)`-shaped log-probability $\log(p(x|\theta) \cdot p(\theta))$.

        """

        theta = ensure_theta_batched(torch.as_tensor(theta))

        # Select and check x to condition on..
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        x = ensure_x_batched(x)
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        with torch.set_grad_enabled(track_gradients):
            return self._q.log_prob(theta.to(self._device))

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        track_gradients: bool = False,
        method: str = "naive",
        method_params: dict = {},
    ) -> Tensor:
        """
        Return samples from variational posterior distribution $q(\theta|x)$.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            track_gradient: Wheather to reparamterize samples which enables to pass
                gradients through it.
            method: A sampling method e.g. using correction methods as SIR, MCMC or ABC techniques.

        Returns:
            Samples from posterior.
        """

        # Select and check x to condition on..
        sample_shape = torch.Size(sample_shape)
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        x = ensure_x_batched(x)
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        if method.lower() == "naive":
            if track_gradients and self._q.has_rsample:
                samples = self._q.rsample(sample_shape)
            else:
                samples = self._q.sample(sample_shape)
        elif method.lower() == "ir":
            samples = importance_resampling(
                sample_shape.numel(), self, x, **method_params
            )
        elif method.lower() == "imh":
            samples = independent_mh(sample_shape.numel(), self, x, **method_params)
        elif method.lower() == "rejection":
            samples = rejection_sampling(sample_shape.numel(), self, x, **method_params)
        elif method.lower() == "slice":
            samples = random_direction_slice_sampler(
                sample_shape.numel(), self, x, **method_params
            )
        else:
            raise NotImplementedError()

        return samples

    def train(
        self,
        x: Optional[Array] = None,
        loss: str = "elbo",
        n_particles: Optional[int] = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.999,
        max_num_iters: Optional[int] = 2000,
        clip_value: Optional[float] = 5.0,
        warm_up_rounds: int = 200,
        retrain_from_scratch: bool = False,
        reset_optimizer: bool = False,
        show_progress_bar: bool = True,
        check_for_convergence: bool = True,
        **kwargs,
    ):
        """This methods trains the variational posterior.
        
        Args:
            x: The observation
            loss: The loss that is minimimzed, default is the ELBO
            n_particles: Number of samples to approximate expectations.
            learning_rate: Learning rate of the optimizer
            gamma: Learning rate decay per iteration
            max_num_iters: Maximum number of iterations
            clip_value: Gradient clipping value
            warm_up_rounds: Initialize the posterior as the prior.
            retrain_from_scratch: Retrain the flow
            resume_training: Resume training the flow
            show_progress_bar: Show the progress bar
        """

        # Init q and the optimizer if necessary
        if retrain_from_scratch:
            self._q = build_q(
                self._prior.event_shape, self._prior.support, **self._flow_paras
            )
            self._optimizer = build_optimizer(
                self,
                loss,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                **kwargs,
            )

        if (
            reset_optimizer
            or self._loss != loss
            or self._optimizer._loss_name != loss
            or self._optimizer.likelihood != self.net
        ):
            self._optimizer = build_optimizer(
                self,
                loss,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                **kwargs,
            )
            self._loss = loss

        # Check context
        if x is None:
            x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(self._device)
        else:
            x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(self._device)

            self._ensure_single_x(x)
            self._ensure_x_consistent_with_default_x(x)
            self.set_default_x(x)

        # Optimize
        self._optimizer.update(locals())
        optimizer = self._optimizer
        optimizer.reset_loss_stats()

        if show_progress_bar:
            iters = tqdm(range(max_num_iters))
        else:
            iters = range(max_num_iters)

        # Warmup before training
        if optimizer.num_step == 0:
            iters.set_description("Warmup phase, this takes some seconds...")
            optimizer.warm_up(warm_up_rounds)

        for _ in iters:
            optimizer.step(x)
            mean_loss, std_loss = optimizer.get_loss_stats()
            # Update progress bar
            if show_progress_bar:
                iters.set_description(
                    f"Loss: {np.round(mean_loss, 2)} Std: {np.round(std_loss, 2)}"
                )
            # Check for convergence
            if check_for_convergence:
                if optimizer.converged():
                    if show_progress_bar:
                        print(f"\nConverged with loss: {np.round(mean_loss, 2)}")
                    break
        if show_progress_bar:
            k = round(float(optimizer.evaluate(x)), 3)
            print(f"Quality Score: {k} (smaller values are good, should be below 1)")
            if k > 1:
                warn(
                    "The quality of the variational posterior seems to be bad, increase the training iterations or consider a different model!"
                )

