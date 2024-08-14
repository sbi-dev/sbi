# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from functools import partial
from typing import Dict, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from zuko.transforms import FreeFormJacobianTransform

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.score_based_potential import (
    score_estimator_based_potential_gradient,
)
from sbi.neural_nets.estimators.score_estimator import ConditionalScoreEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.samplers.score.correctors import Corrector
from sbi.samplers.score.predictors import Predictor
from sbi.samplers.score.score import Diffuser
from sbi.sbi_types import Shape
from sbi.utils import check_prior
from sbi.utils.sbiutils import within_support
from sbi.utils.torchutils import ensure_theta_batched


class ScorePosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x_o)$ with `log_prob()` and `sample()` methods. It samples
    from the diffusion model given the score_estimator and rejects samples that lie
    outside of the prior bounds.

    The posterior is defined by a score estimator and a prior. The score estimator
    provides the gradient of the log-posterior with respect to the parameters. The
    prior is used to reject samples that lie outside of the prior bounds.

    NOTE: The `log_prob()` method is not implemented yet. It will be implemented in a
    future release using the probability flow ODEs.
    """

    def __init__(
        self,
        score_estimator: ConditionalScoreEstimator,
        prior: Distribution,
        max_sampling_batch_size: int = 10_000,
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
        enable_transform: bool = False,  # NOTE: True not supported yet
    ):
        """
        Args:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            score_estimator: The trained neural score estimator.
            max_sampling_batch_size: Batchsize of samples being drawn from
                the proposal at every iteration.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Shape of a single simulator output. If passed, it is used to check
                the shape of the observed data and give a descriptive error.
            enable_transform: Whether to transform parameters to unconstrained space
                during MAP optimization. When False, an identity transform will be
                returned for `theta_transform`.
        """

        check_prior(prior)

        super().__init__(
            device=device,
            x_shape=x_shape,
        )

        potential_fn_gradient, theta_transform = (
            score_estimator_based_potential_gradient(
                score_estimator=score_estimator,
                prior=prior,
                x_o=None,
                enable_transform=enable_transform,
            )
        )

        device = device if device is not None else potential_fn_gradient.device

        self.prior = prior
        self.score_estimator = score_estimator
        self.potential_fn_gradient = potential_fn_gradient
        self.theta_transform = theta_transform

        self.max_sampling_batch_size = max_sampling_batch_size

        self._purpose = """It samples from the diffusion model given the \
            score_estimator."""

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        predictor: Union[str, Predictor] = "euler_maruyama",
        corrector: Optional[Union[str, Corrector]] = None,
        predictor_params: Optional[Dict] = None,
        corrector_params: Optional[Dict] = None,
        steps: int = 500,
        ts: Optional[Tensor] = None,
        max_sampling_batch_size: int = 10_000,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Shape of the samples to be drawn.
            x: Deprecated - use `.set_default_x()` prior to `.sample()`.
            method: Method to use for sampling. Currently, only "euler_maruyma" is
                supported.
            steps: Number of steps to take for the Euler-Maruyama method.
            max_sampling_batch_size: Maximum batch size for sampling.
            sample_with: Deprecated - use `.build_posterior(sample_with=...)` prior to
                `.sample()`.
            show_progress_bars: Whether to show a progress bar during sampling.
        """

        num_samples = torch.Size(sample_shape).numel()

        x = self._x_else_default_x(x)
        x = reshape_to_batch_event(x, self.score_estimator.condition_shape)
        self.potential_fn_gradient.set_x(x)

        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )

        if ts is None:
            T_max = self.score_estimator.T_max
            T_min = self.score_estimator.T_min
            ts = torch.linspace(T_max, T_min, steps)

        diffuser = Diffuser(
            self.potential_fn_gradient, predictor=predictor, corrector=corrector
        )
        max_sampling_batch_size = min(max_sampling_batch_size, num_samples)
        samples = []
        num_iter = num_samples // max_sampling_batch_size
        num_iter = (
            num_iter + 1 if (num_samples % max_sampling_batch_size) != 0 else num_iter
        )
        for _ in range(num_iter):
            samples.append(
                diffuser.run(
                    num_samples=max_sampling_batch_size,
                    ts=ts,
                    show_progress_bars=show_progress_bars,
                )
            )
        samples = torch.cat(samples, dim=0)[:num_samples]

        return samples.reshape(sample_shape + self.score_estimator.input_shape)

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        track_gradients: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-6,
        exact: bool = True,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        Args:
            theta: Parameters $\theta$.
            x: Observed data $x_o$. If None, the default $x_o$ is used.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            atol: Absolute tolerance for the ODE solver.
            rtol: Relative tolerance for the ODE solver.
            exact: Whether to use the exact Jacobian of the transformation or an
                stochastic approximation, which is faster but less accurate.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """

        x = self._x_else_default_x(x)

        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta_density_estimator = reshape_to_sample_batch_event(
            theta, theta.shape[1:], leading_is_sample=True
        )
        x_density_estimator = reshape_to_batch_event(
            x, event_shape=self.score_estimator.condition_shape
        )
        assert (
            x_density_estimator.shape[0] == 1
        ), ".log_prob() supports only `batchsize == 1`."

        self.score_estimator.eval()

        # Compute the base density
        mean_T = self.score_estimator.mean_T
        std_T = self.score_estimator.std_T
        base_density = torch.distributions.Normal(mean_T, std_T)
        for _ in range(len(self.score_estimator.input_shape)):
            base_density = torch.distributions.Independent(base_density, 1)
        # Build the freeform jacobian transformation by probability flow ODEs
        transform = self.build_freeform_jacobian_transform(
            x_density_estimator, atol=atol, rtol=rtol, exact=exact
        )

        with torch.set_grad_enabled(track_gradients):
            eps_samples, logabsdet = transform.inv.call_and_ladj(  # type: ignore
                theta_density_estimator
            )
            base_log_prob = base_density.log_prob(eps_samples)
            log_probs = base_log_prob - logabsdet
            log_probs = log_probs.squeeze(-1)

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                log_probs,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )
            return masked_log_prob

    def sample_batched(
        self,
        sample_shape: torch.Size,
        x: Tensor,
        max_sampling_batch_size: int = 10000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        raise NotImplementedError(
            "Batched sampling is not implemented for ScorePosterior."
        )

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1000,
        num_to_optimize: int = 100,
        learning_rate: float = 1e54,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The MAP is obtained by running gradient
        ascent from a given number of starting positions (samples from the posterior
        with the highest log-probability). After the optimization is done, we select the
        parameter set that has the highest log-probability after the optimization.

         Args:
             x: Deprecated - use `.set_default_x()` prior to `.map()`.
             num_iter: Number of optimization steps that the algorithm takes
                 to find the MAP.
             learning_rate: Learning rate of the optimizer.
             init_method: How to select the starting parameters for the optimization. If
                 it is a string, it can be either [`posterior`, `prior`], which samples
                 the respective distribution `num_init_samples` times. If it is a
                 tensor, the tensor will be used as init locations.
             num_init_samples: Draw this number of samples from the posterior and
                 evaluate the log-probability of all of them.
             num_to_optimize: From the drawn `num_init_samples`, use the
                 `num_to_optimize` with highest log-probability as the initial points
                 for the optimization.
             save_best_every: The best log-probability is computed, saved in the
                 `map`-attribute, and printed every `save_best_every`-th iteration.
                 Computing the best log-probability creates a significant overhead
                 (thus, the default is `10`.)
             show_progress_bars: Whether to show a progressbar during sampling from the
                 posterior.
             force_update: Whether to re-calculate the MAP when x is unchanged and
                 have a cached value.
             log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                 {'norm_posterior': True} for SNPE.

         Returns:
             The MAP estimate.
        """
        return super().map(
            x,
            num_iter,
            num_to_optimize,
            learning_rate,
            init_method,
            num_init_samples,
            save_best_every,
            show_progress_bars,
            force_update,
        )

    def _calculate_map(
        self,
        num_iter: int = 1000,
        num_to_optimize: int = 100,
        learning_rate: float = 1e-5,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """Calculate the maximum a posteriori (MAP) estimate of the posterior.

        Uses gradient ascent to find the MAP estimate of the posterior. The gradient is
        calculated using the score estimator.

        Args:
            num_iter: Number of interations. Defaults to 1000.
            num_to_optimize : Note used (API), just for interface. Defaults to 100.
            learning_rate: Learning rate. Defaults to 1e-5.
            init_method: Initialization of particles. Defaults to "posterior".
            num_init_samples: Not used (API). Defaults to 1000.
            save_best_evey: Not used (API). Defaults to 10.
            show_progress_bars (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: Invalid init method

        Returns:
            Tensor: MAP
        """

        with torch.no_grad():
            if init_method == "posterior":
                inits = self.sample(
                    (num_init_samples,), show_progress_bars=show_progress_bars
                )
            elif init_method == "proposal":
                inits = self.proposal.sample((num_init_samples,))  # type: ignore
            elif isinstance(init_method, Tensor):
                inits = init_method
            else:
                raise ValueError

            self.potential_fn_gradient.set_x(self.default_x)
            gradient_fn = partial(
                self.potential_fn_gradient,
                time=torch.tensor([self.score_estimator.T_min]),
            )

            # Run MAP optimization
            xs = inits.clone()
            for _ in range(num_iter):
                gradient = gradient_fn(xs)
                xs = xs + learning_rate * gradient

            log_prob = self.log_prob(xs)
            best_idx = torch.argmax(log_prob)
            best_theta = xs[best_idx]
            return best_theta

    def build_freeform_jacobian_transform(
        self, x_o: Tensor, atol: float = 1e-5, rtol: float = 1e-6, exact: bool = True
    ):
        # Create a freeform jacobian transformation
        phi = self.score_estimator.parameters()

        def f(t, x):
            score = self.score_estimator(input=x, condition=x_o, time=t)
            f = self.score_estimator.drift_fn(x, t)
            g = self.score_estimator.diffusion_fn(x, t)
            v = f - 0.5 * g**2 * score
            return v

        transform = FreeFormJacobianTransform(
            f=f,
            t0=self.score_estimator.T_min,
            t1=self.score_estimator.T_max,
            phi=phi,
            atol=atol,
            rtol=rtol,
            exact=exact,
        )

        return transform
