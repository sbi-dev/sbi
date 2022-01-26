# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, List, Optional, Union

import torch
from torch.distributions import Distribution

from tqdm import tqdm


from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape, TorchTransform
from sbi.utils.torchutils import (
    atleast_2d_float32_tensor,
    Array,
    Tensor,
    ensure_theta_batched,
)

from sbi.samplers.vi import (
    adapt_and_check_variational_distributions,
    make_sure_nothing_in_cache,
    get_flow_builder,
    get_VI_method,
    get_sampling_method,
    get_quality_metric,
)


class VIPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNLE.<br/><br/>
    SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
    `SNLE_Posterior` class wraps the trained network such that one can directly evaluate
    the unnormalized posterior log probability $p(\theta|x) \propto p(x|\theta) \cdot
    p(\theta)$ and draw samples from the posterior with MCMC.<br/><br/>
    The neural network itself can be accessed via the `.net` attribute.
    """

    def __init__(
        self,
        potential_fn: Callable,
        theta_transform: Optional[TorchTransform] = None,
        q: Union[str, Distribution] = "maf",
        q_kwargs: dict = dict(),
        vi_method: str = "rKL",
        device: str = "cpu",
    ):
        """
        Args:
            potential_fn: Potential function to fit.
            theta_transform: Maps form prior support to unconstrained space
            q: Variational family, either string or distribution object.
            q_kwargs: Arguments for construction q.
            theta_transform: Transform to different parameter space. This is not as
            important as for MCMC methods.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
        """
        super().__init__(potential_fn, theta_transform, device)

        self._device = device
        self._prior = self.potential_fn.prior
        self._q_arg = q
        self._q_kwargs = q_kwargs
        self._optimizer = None

        # In contrast to MCMC we want to project into constrained space!
        theta_transform = theta_transform.inv

        if isinstance(q, Distribution):
            self.set_q(
                adapt_and_check_variational_distributions(q, q_kwargs, self._prior)
            )
        else:
            self.set_q(get_flow_builder(q, self._prior.event_shape, theta_transform))

        self.set_vi_method(vi_method)

        self._purpose = (
            "It provides Variational inference to .sample() from the posterior and "
            "can evaluate the _normalized_ posterior density with .log_prob()."
        )

    @property
    def q(self):
        """Variational posterior distribution object, can be directly accessed and e.g.
        can also be used as proposal for rejection sampling/MCMC based posteriors!
        """
        return self._q

    @q.setter
    def q(
        self,
        q: Distribution,
    ):
        """Sets the variational distribution. If the distribution does not admit access
        through "parameters" and "modules" function, please use set_q if you want to
        explicitly specify the parameters and modules


        Args:
            q: Variational distribution


        """
        self.set_q(q)

    def set_q(
        self,
        q: Distribution,
        parameters: Optional[List] = None,
        modules: Optional[List] = None,
    ):
        """Defines the variational family. You can specify over which
        parameters/modules we optimize. This is required for custom distributions which
        e.g. do not inherit nn.Modules or has the function "parameters" or "modules" to
        give direct access to trainable parameters.



        Args:
            q: Variational distribution.
            parameters: List of parameters associated with the distribution object.
            modules: List of modules associated with the distribution object.


        """
        # Add checks here! TODO And adding parameters capabilities
        self._q = q

    @property
    def vi_method(self):
        """Variational inference method e.g. you can choose different divergence"""
        return self._vi_method

    @vi_method.setter
    def vi_method(self, method: str) -> None:
        """See `set_vi_method`."""
        self.set_vi_method(method)

    def set_vi_method(self, method: str) -> "NeuralPosterior":
        """Sets variational inference method especially which divergence measure to minimize.

        Args:
            method: Method to use.

        Returns:
            `NeuralPosterior` for chainable calls.
        """
        self._vi_method = method
        self._optimizer_base = get_VI_method(method)
        return self

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        method: str = "naive",
        **kwargs,
    ):
        """Samples from the variational posterior distribution.



        Args:
            sample_shape: Shape of samples
            method: Sampling method, alternatively we can debias the approximation by
            using 'sir' (sampling importance resampling).
            kwargs: Additional arguments to ensure backward compatibility. Here you can
            also add parameters for different methods!

        Returns:
            Samples from posterior.
        """
        sampling_function = get_sampling_method(method)
        num_samples = max(torch.prod(torch.tensor(sample_shape)), 1)
        samples = sampling_function(num_samples, self.potential_fn, self.q, **kwargs)
        return samples.reshape((*sample_shape, samples.shape[-1]))

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""Returns the log-probability of theta under the posterior.

        Args:
            theta: Parameters $\theta$.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `len($\theta$)`-shaped log-probability.
        """
        # TODO CHECK FOR TRIANED UNTRAINED AND SO ON...
        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.q.log_prob(theta)

    def train(
        self,
        x: Optional[Array] = None,
        n_particles: Optional[int] = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.999,
        max_num_iters: Optional[int] = 2000,
        min_num_iters: Optional[int] = 10,
        clip_value: Optional[float] = 5.0,
        warm_up_rounds: int = 100,
        retrain_from_scratch: bool = False,
        reset_optimizer: bool = False,
        show_progress_bar: bool = True,
        check_for_convergence: bool = True,
        quality_controll_metric: str = "psis",
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
        if retrain_from_scratch or self._optimizer is None:
            self.set_q(
                get_flow_builder(
                    self._q_arg, self._prior.event_shape, self.theta_transform.inv
                )
            )

            self._optimizer = self._optimizer_base(
                self,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                **kwargs,
            )

        if reset_optimizer or not isinstance(self._optimizer, self._optimizer_base):
            self._optimizer = self._optimizer_base(
                self,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                **kwargs,
            )

        # Check context
        x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(self._device)
        if not self.potential_fn.allow_iid_x:
            self._ensure_single_x(x)

        # Optimize
        self._optimizer.update({**locals(), **kwargs})
        optimizer = self._optimizer
        optimizer.reset_loss_stats()

        if show_progress_bar:
            iters = tqdm(range(max_num_iters))
        else:
            iters = range(max_num_iters)

        # Warmup before training
        if not optimizer.warm_up_was_done:
            if show_progress_bar:
                iters.set_description("Warmup phase, this takes some seconds...")
            optimizer.warm_up(warm_up_rounds)

        for i in iters:
            optimizer.step(x)
            mean_loss, std_loss = optimizer.get_loss_stats()
            # Update progress bar
            if show_progress_bar:
                iters.set_description(
                    f"Loss: {np.round(mean_loss, 2)} Std: {np.round(std_loss, 2)}"
                )
            # Check for convergence
            if check_for_convergence and i > min_num_iters:
                if optimizer.converged():
                    if show_progress_bar:
                        print(f"\nConverged with loss: {np.round(mean_loss, 2)}")
                    break
        if show_progress_bar:
            # TODO ADD QUALTITY CONTROLL
            try:
                quality_control_fn, quality_control_msg = get_quality_metric(
                    quality_controll_metric
                )
                metric = round(float(quality_control_fn(self)), 3)
                print(f"Quality Score: {metric} \n" + quality_control_msg)
            except Exception as e:
                print(
                    f"Quality controll did not work, we reset the variational \
                         posterior,please check your setting! \n Following error occured {e}"
                )
                self.set_q(
                    get_flow_builder(
                        self._q_arg, self._prior.event_shape, self.theta_transform.inv
                    )
                )
                self._optimizer.q = self._q

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "proposal",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self.map_`.
        The MAP is obtained by running gradient ascent from a given number of starting
        positions (samples from the posterior with the highest log-probability). After
        the optimization is done, we select the parameter set that has the highest
        log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
            x: Observed data at which to evaluate the MAP.
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
            show_progress_bars: Whether or not to show a progressbar for sampling from
                the posterior.
            log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                {'norm_posterior': True} for SNPE.

        Returns:
            The MAP estimate.
        """
        self.proposal = self.q
        return super().map(
            x=x,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
        )

    def __deepcopy__(self, *args, **kwargs):
        # Removes tensor with 'required_grad' from any cache as these
        # do not support deepcopy!
        make_sure_nothing_in_cache(self.q)
