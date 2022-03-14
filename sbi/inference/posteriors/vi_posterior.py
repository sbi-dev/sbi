# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from copy import deepcopy
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution
from tqdm.auto import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.samplers.vi import (
    adapt_variational_distribution,
    check_variational_distribution,
    get_flow_builder,
    get_quality_metric,
    get_sampling_method,
    get_VI_method,
    make_object_deepcopy_compatible,
    move_all_tensor_to_device,
)
from sbi.types import (
    PyroTransformedDistribution,
    Shape,
    TorchDistribution,
    TorchTensor,
    TorchTransform,
)
from sbi.utils import mcmc_transform
from sbi.utils.torchutils import atleast_2d_float32_tensor, ensure_theta_batched


class VIPosterior(NeuralPosterior):
    r"""Provides VI (Variational Inference) to sample from the posterior.<br/><br/>
    SNLE or SNRE train neural networks to approximate the likelihood(-ratios).
    `VIPosterior` allows to learn a tractable variational posterior $q(\theta)$ which
    approximates the true posterior $p(\theta|x_o)$. After this second training stage,
    we can produce approximate posterior samples, by just sampling from q with no
    additional cost. For additional information see [1] and [2].<br/><br/>
    References:<br/>
    [1] Variational methods for simulation-based inference, Manuel Glöckler, Michael
    Deistler, Jakob Macke, 2022, https://openreview.net/forum?id=kZ0UYdhqkNY<br/>
    [2] Sequential Neural Posterior and Likelihood Approximation, Samuel Wiqvist, Jes
    Frellsen, Umberto Picchini, 2021, https://arxiv.org/abs/2102.06522
    """

    def __init__(
        self,
        potential_fn: Callable,
        prior: Optional[TorchDistribution] = None,
        q: Union[str, PyroTransformedDistribution, "VIPosterior", Callable] = "maf",
        theta_transform: Optional[TorchTransform] = None,
        vi_method: str = "rKL",
        device: str = "cpu",
        x_shape: Optional[torch.Size] = None,
        parameters: Iterable = [],
        modules: Iterable = [],
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples.
            prior: This is the prior distribution. Note that this is only
                used to check/construct the variational distribution or within some
                quality metrics. Please make sure that this matches with the prior
                within the potential_fn. If `None` is given, we will try to infer it
                from potential_fn or q, if this fails we raise an Error.
            q: Variational distribution, either string, `TransformedDistribution`, or a
                `VIPosterior` object. This specifies a parametric class of distribution
                over which the best possible posterior approximation is searched. For
                string input, we currently support [nsf, scf, maf, mcf, gaussian,
                gaussian_diag]. You can also specify your own variational family by
                passing a pyro `TransformedDistribution`.
                Additionally, we allow a `Callable`, which allows you the pass a
                `builder` function, which if called returns a distribution. This may be
                useful for setting the hyperparameters e.g. `num_transfroms` within the
                `get_flow_builder` method specifying the number of transformations
                within a normalizing flow. If q is already a `VIPosterior`, then the
                arguments will be copied from it (relevant for multi-round training).
            theta_transform: Maps form prior support to unconstrained space. The
                inverse is used here to ensure that the posterior support is equal to
                that of the prior.
            vi_method: This specifies the variational methods which are used to fit q to
                the posterior. We currently support [rKL, fKL, IW, alpha]. Note that
                some of the divergences are `mode seeking` i.e. they underestimate
                variance and collapse on multimodal targets (`rKL`, `alpha` for alpha >
                1) and some are `mass covering` i.e. they overestimate variance but
                typically cover all modes (`fKL`, `IW`, `alpha` for alpha < 1).
            device: Training device, e.g., `cpu`, `cuda` or `cuda:0`. We will ensure
                that all other objects are also on this device.
            x_shape: Shape of a single simulator output. If passed, it is used to check
                the shape of the observed data and give a descriptive error.
            parameters: List of parameters of the variational posterior. This is only
                required for user-defined q i.e. if q does not have a `parameters`
                attribute.
            modules: List of modules of the variational posterior. This is only
                required for user-defined q i.e. if q does not have a `modules`
                attribute.
        """
        super().__init__(potential_fn, theta_transform, device, x_shape=x_shape)

        # Especially the prior may be on another device -> move it...
        self._device = device
        self.potential_fn.device = device
        move_all_tensor_to_device(self.potential_fn, device)

        # Get prior and previous builds
        if prior is not None:
            self._prior = prior
        elif hasattr(self.potential_fn, "prior") and isinstance(
            self.potential_fn.prior, Distribution
        ):
            self._prior = self.potential_fn.prior
        elif isinstance(q, VIPosterior) and isinstance(q._prior, Distribution):
            self._prior = q._prior
        else:
            raise ValueError(
                "We could not find a suitable prior distribution within `potential_fn`"
                "or `q` (if a VIPosterior is given). Please explicitly specify a prior."
            )
        move_all_tensor_to_device(self._prior, device)
        self._optimizer = None

        # In contrast to MCMC we want to project into constrained space.
        if theta_transform is None:
            self.link_transform = mcmc_transform(self._prior).inv
        else:
            self.link_transform = theta_transform.inv

        # This will set the variational distribution and VI method
        self.set_q(q, parameters=parameters, modules=modules)
        self.set_vi_method(vi_method)

        self._purpose = (
            "It provides Variational inference to .sample() from the posterior and "
            "can evaluate the _normalized_ posterior density with .log_prob()."
        )

    @property
    def q(self) -> Distribution:
        """Returns the variational posterior."""
        return self._q

    @q.setter
    def q(
        self,
        q: Union[str, Distribution, "VIPosterior", Callable],
    ) -> None:
        """Sets the variational distribution. If the distribution does not admit access
        through `parameters` and `modules` function, please use `set_q` if you want to
        explicitly specify the parameters and modules.


        Args:
            q: Variational distribution, either string, distribution, or a VIPosterior
                object. This specifies a parametric class of distribution over which
                the best possible posterior approximation is searched. For string input,
                we currently support [nsf, scf, maf, mcf, gaussian, gaussian_diag]. Of
                course, you can also specify your own variational family by passing a
                `parameterized` distribution object i.e. a torch.distributions
                Distribution with methods `parameters` returning an iterable of all
                parameters (you can pass them within the paramters/modules attribute).
                Additionally, we allow a `Callable`, which allows you the pass a
                `builder` function, which if called returns an distribution. This may be
                useful for setting the hyperparameters e.g. `num_transfroms:int` by
                using the `get_flow_builder` method specifying the hyperparameters. If q
                is already a `VIPosterior`, then the arguments will be copied from it
                (relevant for multi-round training).


        """
        self.set_q(q)

    def set_q(
        self,
        q: Union[str, PyroTransformedDistribution, "VIPosterior", Callable],
        parameters: Iterable = [],
        modules: Iterable = [],
    ) -> None:
        """Defines the variational family.

        You can specify over which parameters/modules we optimize. This is required for
        custom distributions which e.g. do not inherit nn.Modules or has the function
        `parameters` or `modules` to give direct access to trainable parameters.
        Further, you can pass a function, which constructs a variational distribution
        if called.

        Args:
            q: Variational distribution, either string, distribution, or a VIPosterior
                object. This specifies a parametric class of distribution over which
                the best possible posterior approximation is searched. For string input,
                we currently support [nsf, scf, maf, mcf, gaussian, gaussian_diag]. Of
                course, you can also specify your own variational family by passing a
                `parameterized` distribution object i.e. a torch.distributions
                Distribution with methods `parameters` returning an iterable of all
                parameters (you can pass them within the paramters/modules attribute).
                Additionally, we allow a `Callable`, which allows you the pass a
                `builder` function, which if called returns an distribution. This may be
                useful for setting the hyperparameters e.g. `num_transfroms:int` by
                using the `get_flow_builder` method specifying the hyperparameters. If q
                is already a `VIPosterior`, then the arguments will be copied from it
                (relevant for multi-round training).
            parameters: List of parameters associated with the distribution object.
            modules: List of modules associated with the distribution object.

        """
        self._q_arg = q
        if isinstance(q, Distribution):
            q = adapt_variational_distribution(
                q,
                self._prior,
                self.link_transform,
                parameters=parameters,
                modules=modules,
            )
            make_object_deepcopy_compatible(q)
            self_custom_q_init_cache = deepcopy(q)
            self._q_build_fn = lambda *args, **kwargs: self_custom_q_init_cache
            self._trained_on = None
        elif isinstance(q, str) or isinstance(q, Callable):
            if isinstance(q, str):
                self._q_build_fn = get_flow_builder(q)
            else:
                self._q_build_fn = q

            q = self._q_build_fn(
                self._prior.event_shape,
                self.link_transform,
                device=self._device,
            )
            make_object_deepcopy_compatible(q)
            self._trained_on = None
        elif isinstance(q, VIPosterior):
            self._q_build_fn = q._q_build_fn
            self._trained_on = q._trained_on
            self.vi_method = q.vi_method  # type: ignore
            self._device = q._device
            self._prior = q._prior
            self._x = q._x
            self._q_arg = q._q_arg
            make_object_deepcopy_compatible(q.q)
            q = deepcopy(q.q)
        move_all_tensor_to_device(q, self._device)
        assert isinstance(
            q, Distribution
        ), "Something went wrong when initializing the variational distribution. Please create an issue on github https://github.com/mackelab/sbi/issues"
        check_variational_distribution(q, self._prior)
        self._q = q

    @property
    def vi_method(self) -> str:
        """Variational inference method e.g. one of [rKL, fKL, IW, alpha]."""
        return self._vi_method

    @vi_method.setter
    def vi_method(self, method: str) -> None:
        """See `set_vi_method`."""
        self.set_vi_method(method)

    def set_vi_method(self, method: str) -> "VIPosterior":
        """Sets variational inference method.

        Args:
            method: One of [rKL, fKL, IW, alpha].

        Returns:
            `VIPosterior` for chainable calls.
        """
        self._vi_method = method
        self._optimizer_builder = get_VI_method(method)
        return self

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        method: str = "naive",
        **kwargs,
    ) -> Tensor:
        """Samples from the variational posterior distribution.

        Args:
            sample_shape: Shape of samples
            method: Sampling method, alternatively we can debias the approximation by
                using simple and efficient sampling schemes. We support one of [naive,
                sir].
            kwargs: Hyperparameters for the sampling methods.
                naive: Just samples from q, no parameters.
                sir: Performs sampling importance resampling.
                    `K`: Number of importance samples
                    `num_samples_batch`: How many samples are drawn in parallel (For
                        large K you may have to decrease this due to memory limitation).


        Returns:
            Samples from posterior.
        """
        x = self._x_else_default_x(x)
        if self._trained_on is None or (x != self._trained_on).all():
            raise AttributeError(
                f"The variational posterior was not fit using observation {x}."
                "Please train."
            )

        self.potential_fn.set_x(x)
        sampling_function = get_sampling_method(method)
        num_samples = torch.Size(sample_shape).numel()
        samples = sampling_function(num_samples, self.potential_fn, self.q, **kwargs)
        return samples.reshape((*sample_shape, samples.shape[-1]))

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        track_gradients: bool = False,
    ) -> Tensor:
        r"""Returns the log-probability of theta under the variational posterior.

        Args:
            theta: Parameters
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis but increases memory
                consumption.

        Returns:
            `len($\theta$)`-shaped log-probability.
        """
        x = self._x_else_default_x(x)
        if self._trained_on is None or (x != self._trained_on).all():
            raise AttributeError(
                f"The variational posterior was not fit using observation {x}.\
                     Please train."
            )
        with torch.set_grad_enabled(track_gradients):
            theta = ensure_theta_batched(torch.as_tensor(theta))
            return self.q.log_prob(theta)

    def train(
        self,
        x: Optional[TorchTensor] = None,
        n_particles: int = 256,
        learning_rate: float = 1e-3,
        gamma: float = 0.999,
        max_num_iters: int = 2000,
        min_num_iters: int = 10,
        clip_value: float = 10.0,
        warm_up_rounds: int = 100,
        retrain_from_scratch: bool = False,
        reset_optimizer: bool = False,
        show_progress_bar: bool = True,
        check_for_convergence: bool = True,
        quality_control: bool = True,
        quality_control_metric: str = "psis",
        **kwargs,
    ) -> "VIPosterior":
        """This method trains the variational posterior.

        Args:
            x: The observation.
            n_particles: Number of samples to approximate expectations within the
                variational bounds. The larger the more accurate are gradient
                estimates, but the computational cost per iteration increases.
            learning_rate: Learning rate of the optimizer.
            gamma: Learning rate decay per iteration. We use an exponential decay
                scheduler.
            max_num_iters: Maximum number of iterations.
            min_num_iters: Minimum number of iterations.
            clip_value: Gradient clipping value, decreasing may help if you see invalid
                values.
            warm_up_rounds: Initialize the posterior as the prior.
            retrain_from_scratch: Retrain the variational distributions from scratch.
            reset_optimizer: Reset the divergence optimizer
            show_progress_bar: If any progress report should be displayed.
            quality_control: If False quality control is skipped.
            quality_control_metric: Which metric to use for evaluating the quality.
            kwargs: Hyperparameters check corresponding `DivergenceOptimizer` for detail
                eps: Determines sensitivity of convergence check.
                retain_graph: Boolean which decides whether to retain the computation
                    graph. This may be required for some `exotic` user-specified q's.
                optimizer: A PyTorch Optimizer class e.g. Adam or SGD. See
                    `DivergenceOptimizer` for details.
                scheduler: A PyTorch learning rate scheduler. See
                    `DivergenceOptimizer` for details.
                alpha: Only used if vi_method=`alpha`. Determines the alpha divergence.
                K: Only used if vi_method=`IW`. Determines the number of importance
                    weighted particles.
                stick_the_landing: If one should use the STL estimator (only for rKL,
                    IW, alpha).
                dreg: If one should use the DREG estimator (only for rKL, IW, alpha).
                weight_transform: Callable applied to importance weights (only for fKL)
        Returns:
            VIPosterior: `VIPosterior` (can be used to chain calls).
        """
        # Update optimizer with current arguments.
        if self._optimizer is not None:
            self._optimizer.update({**locals(), **kwargs})

        # Init q and the optimizer if necessary
        if retrain_from_scratch:
            self.q = self._q_build_fn()  # type: ignore
            self._optimizer = self._optimizer_builder(
                self.potential_fn,
                self.q,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                prior=self._prior,
                **kwargs,
            )

        if (
            reset_optimizer
            or self._optimizer is None
            or not isinstance(self._optimizer, self._optimizer_builder)
        ):
            self._optimizer = self._optimizer_builder(
                self.potential_fn,
                self.q,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                prior=self._prior,
                **kwargs,
            )

        # Check context
        x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(  # type: ignore
            self._device
        )

        already_trained = self._trained_on is not None and (x == self._trained_on).all()

        # Optimize
        optimizer = self._optimizer
        optimizer.to(self._device)
        optimizer.reset_loss_stats()

        if show_progress_bar:
            iters = tqdm(range(max_num_iters))
        else:
            iters = range(max_num_iters)

        # Warmup before training
        if reset_optimizer or (not optimizer.warm_up_was_done and not already_trained):
            if show_progress_bar:
                iters.set_description(  # type: ignore
                    "Warmup phase, this may take a few seconds..."
                )
            optimizer.warm_up(warm_up_rounds)

        for i in iters:
            optimizer.step(x)
            mean_loss, std_loss = optimizer.get_loss_stats()
            # Update progress bar
            if show_progress_bar:
                assert isinstance(iters, tqdm)
                iters.set_description(  # type: ignore
                    f"Loss: {np.round(float(mean_loss), 2)}"
                    f"Std: {np.round(float(std_loss), 2)}"
                )
            # Check for convergence
            if check_for_convergence and i > min_num_iters:
                if optimizer.converged():
                    if show_progress_bar:
                        print(f"\nConverged with loss: {np.round(float(mean_loss), 2)}")
                    break
        # Training finished:
        self._trained_on = x

        # Evaluate quality
        if quality_control:
            try:
                self.evaluate(quality_control_metric=quality_control_metric)
            except Exception as e:
                print(
                    f"Quality control did not work, we reset the variational \
                        posterior,please check your setting. \
                        \n Following error occured {e}"
                )
                self.train(
                    learning_rate=learning_rate * 0.1,
                    retrain_from_scratch=True,
                    reset_optimizer=True,
                )

        return self

    def evaluate(self, quality_control_metric: str = "psis", N: int = int(5e4)) -> None:
        """This function will evaluate the quality of the variational posterior
        distribution. We currently support two different metrics of type `psis`, which
        checks the quality based on the tails of importance weights (there should not be
        much with a large one), or `prop` which checks the proportionality between q
        and potential_fn.

        NOTE: In our experience `prop` is sensitive to distinguish ``good`` from ``ok``
        whereas `psis` is more sensitive in distinguishing `very bad` from `ok`.

        Args:
            quality_control_metric: The metric of choice, we currently support [psis,
                prop, prop_prior].
            N: Number of samples which is used to evaluate the metric.


        """
        quality_control_fn, quality_control_msg = get_quality_metric(
            quality_control_metric
        )
        metric = round(float(quality_control_fn(self, N=N)), 3)
        print(f"Quality Score: {metric} " + quality_control_msg)

    def map(
        self,
        x: Optional[TorchTensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, TorchTensor] = "proposal",
        num_init_samples: int = 10_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self._map` and
        can be accessed with `self.map()`. The MAP is obtained by running gradient
        ascent from a given number of starting positions (samples from the posterior
        with the highest log-probability). After the optimization is done, we select the
        parameter set that has the highest log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

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
            show_progress_bars: Whether or not to show a progressbar for sampling from
                the posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.
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
            force_update=force_update,
        )
