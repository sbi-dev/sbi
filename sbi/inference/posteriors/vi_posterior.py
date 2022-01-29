# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from copy import deepcopy
from typing import Callable, List, Optional, Union, Dict
from warnings import warn

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
    check_variational_distribution,
    make_sure_nothing_in_cache,
    move_all_tensor_to_device,
    get_flow_builder,
    get_VI_method,
    get_sampling_method,
    get_quality_metric,
    get_default_flows,
    get_default_VI_method,
    get_default_sampling_methods,
    get_sampling_method_parameters_doc,
    docstring_parameter,
)


class VIPosterior(NeuralPosterior):
    r"""Provides VI (Variational Inference) to sample from the posterior.<br/><br/>

    SNLE or SNRE train neural networks to approximate the likelihood(-ratios).
    `VIPosterior` allows to train a tractable variational posterior q(theta) which
    approximates the true posterior p(theta|x_o). After this second training stage, we
    can produce approximate posterior samples, by just sampling from q with no
    additional cost.
    """

    def __init__(
        self,
        potential_fn: Callable,
        q: Union[str, Distribution, NeuralPosterior] = "maf",
        q_kwargs: Dict = dict(),
        theta_transform: Optional[TorchTransform] = None,
        vi_method: str = "rKL",
        device: str = "cpu",
    ):
        f"""
        Args:
            potential_fn: The potential function from which to draw samples.
            q: Variational family, either string, distribution or an VIPosterior
                object. This specifies a parameteric class of distribution over which
                the best possible posterior approximation is searched. For string input
                we currently support {get_default_flows()}. Of course you can also
                specify your own variational family by passing a 'parameterized'
                distribution object i.e. a torch.distributions.Distribution with methods
                'parameters' returning an iterable of all parameters (you can also
                handle this using q_kwargs see below).
            q_kwargs: Arguments for construction q. If q is a string, then this will be
                passed to the 'flow_builder' and thus can be used to specify
                hyperparameters. Examples of arguments are e.g. 'num_flows:int' which
                specify the number of layers but also hyperparameters for pyro
                transforms as e.g. 'hidden_dims:list[int]' to specify number of hidden
                neurons and layers of an neural net used within the flow. If q is a
                distribution, then you can also specify the 'parameters' and 'modules'
                within this dictionary i.e. as a list of tensors or modules. If q is
                already a 'VIPosterior', then most arguments will be copied from it
                (relevant for multi round training).
            theta_transform: Maps form prior support to unconstrained space. In fact the
                inverse is used here to ensure that the posterior support is equal to
                that of the prior!
            vi_method: This specifies the variational methods which is used to fit q to
                the posterior. We currently support {get_default_VI_method()}. Note that
                some of the divergences are 'mode seeking' i.e. they underestimate
                variance and collapse on multimodal targets ('rKL', 'alpha' for alpha >
                1) and some are 'mass covering' i.e. they overestimate variance but
                typically cover all modes ('fKL', 'IW', 'alpha' for alpha < 1).
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
        """
        super().__init__(potential_fn, theta_transform, device)

        self.potential_fn.device = device
        # Especially the prior may be on another device
        move_all_tensor_to_device(self.potential_fn, device)
        self._device = device
        self._prior = self.potential_fn.prior
        self._optimizer = None

        # In contrast to MCMC we want to project into constrained space!
        self.link_transform = theta_transform.inv
        self.set_q(q, q_kwargs)
        self.set_vi_method(vi_method)

        self._trained_on = None

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
        q: Union[str, Distribution, NeuralPosterior],
    ) -> None:
        """Sets the variational distribution. If the distribution does not admit access
        through "parameters" and "modules" function, please use 'set_q' if you want to
        explicitly specify the parameters and modules.


        Args:
            q: Variational distribution


        """
        self.set_q(q)

    def set_q(
        self,
        q: Union[str, Distribution, NeuralPosterior],
        q_kwargs: Dict = {},
    ) -> None:
        """Defines the variational family. You can specify over which parameters/modules
        we optimize. This is required for custom distributions which e.g. do not inherit
        nn.Modules or has the function "parameters" or "modules" to
        give direct access to trainable parameters.



        Args:
            q: Variational distribution.
            parameters: List of parameters associated with the distribution object.
            modules: List of modules associated with the distribution object.


        """
        self._q_arg = q
        self._q_kwargs = q_kwargs
        if isinstance(q, Distribution):
            q = adapt_and_check_variational_distributions(
                q, q_kwargs, self._prior, self.link_transform
            )
        elif isinstance(q, str):
            q = get_flow_builder(
                q,
                self._prior.event_shape,
                self.link_transform,
                device=self._device,
                **q_kwargs,
            )
            # check_variational_distribution(q, self._prior)
        elif isinstance(q, VIPosterior):
            self._trained_on = q._trained_on
            self.vi_method = q.vi_method
            self._device = q._device
            self._prior = q._prior
            q = deepcopy(q.q)
        move_all_tensor_to_device(q, self._device)
        self._q = q

    @property
    @docstring_parameter(get_default_VI_method())
    def vi_method(self):
        """Variational inference method e.g. one of {0}"""
        return self._vi_method

    @vi_method.setter
    def vi_method(self, method: str) -> None:
        """See `set_vi_method`."""
        self.set_vi_method(method)

    @docstring_parameter(get_default_VI_method())
    def set_vi_method(self, method: str) -> "NeuralPosterior":
        """Sets variational inference method.

        Args:
            method: One of {0}

        Returns:
            `NeuralPosterior` for chainable calls.
        """
        self._vi_method = method
        self._optimizer_base = get_VI_method(method)
        return self

    @docstring_parameter(
        get_default_sampling_methods(),
        "".join(
            [
                "\n\t\t" + get_sampling_method_parameters_doc(n).replace("\n", "\n\t\t")
                for n in get_default_sampling_methods()
            ]
        ),
    )
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
                using simple and efficient sampling schemes. We support one of {0}.
            kwargs: Hyperparameters for the sampling methods. {1}


        Returns:
            Samples from posterior.
        """
        x = self._x_else_default_x(x)
        if self._trained_on is None or (x != self._trained_on).all():
            warn(
                f"The variational posterior was not fit using observation {x}.\
                     Please train!"
            )

        self.potential_fn.set_x(x)
        sampling_function = get_sampling_method(method)
        num_samples = torch.Size(sample_shape).numel()
        samples = sampling_function(num_samples, self.potential_fn, self.q, **kwargs)
        return samples.reshape((*sample_shape, samples.shape[-1]))

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""Returns the log-probability of theta under the variational posterior.

        Args:
            theta: Parameters
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `len($\theta$)`-shaped log-probability.
        """
        x = self._x_else_default_x(x)
        if self._trained_on is None or (x != self._trained_on).all():
            warn(
                f"The variational posterior was not fit using observation {x}.\
                     Please train!"
            )
        with torch.set_grad_enabled(track_gradients):
            theta = ensure_theta_batched(torch.as_tensor(theta))
            return self.q.log_prob(theta)

    def train(
        self,
        x: Optional[Array] = None,
        n_particles: Optional[int] = 256,
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
    ) -> NeuralPosterior:
        """This methods trains the variational posterior.

        Args:
            x: The observation.
            n_particles: Number of samples to approximate expectations within the
                variational bounds. The larger the more accurate are gradient
                estimates, but the computational cost per iteration increases.
            learning_rate: Learning rate of the optimizer.
            gamma: Learning rate decay per iteration. We use a exponential decay
                scheduler.
            max_num_iters: Maximum number of iterations.
            min_num_iters: Minimum number of iterations.
            clip_value: Gradient clipping value, decrease may help if you see invalid
                values.
            warm_up_rounds: Initialize the posterior as the prior.
            retrain_from_scratch: Retrain the variational distributions from scratch.
            reset_optimizer: Reset the divergence optimizer
            show_progress_bar: If any progress report should be displayed.
            quality_controll_metric: Which metric to use for evaluate the quality.
            kwargs: Hyperparameters
                retain_graph: Boolean which decides weather to retain the computation
                    graph. This may be required for some 'exotic' user-specified q's.
        Returns:
            NeuralPosterior: The VIPosterior (can be used to chain calls).
        """

        # Init q and the optimizer if necessary
        if retrain_from_scratch or self._optimizer is None:
            self.set_q(self._q_arg, self._q_kwargs)
            self._optimizer = self._optimizer_base(
                self.potential_fn,
                self.q,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                **kwargs,
            )

        if reset_optimizer or not isinstance(self._optimizer, self._optimizer_base):
            self._optimizer = self._optimizer_base(
                self.potential_fn,
                self.q,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                **kwargs,
            )

        # Check context
        x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(self._device)
        already_trained = self._trained_on is not None and (x == self._trained_on).all()
        self._trained_on = x
        if not self.potential_fn.allow_iid_x:
            self._ensure_single_x(x)

        # Optimize
        self._optimizer.update({**locals(), **kwargs})
        optimizer = self._optimizer
        optimizer.to(self._device)
        optimizer.reset_loss_stats()

        if show_progress_bar:
            iters = tqdm(range(max_num_iters))
        else:
            iters = range(max_num_iters)

        # Warmup before training
        if not optimizer.warm_up_was_done and not already_trained or reset_optimizer:
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
            try:
                quality_control_fn, quality_control_msg = get_quality_metric(
                    quality_controll_metric
                )
                metric = round(float(quality_control_fn(self)), 3)
                print(f"Quality Score: {metric} " + quality_control_msg)
            except Exception as e:
                print(
                    f"Quality controll did not work, we reset the variational \
                         posterior,please check your setting! \n Following error occured {e}"
                )
                self.train(
                    learning_rate=learning_rate * 0.1,
                    retrain_from_scratch=True,
                    reset_optimizer=True,
                )
        return self

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "proposal",
        num_init_samples: int = 10_000,
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

    def __deepcopy__(self, memo):
        # Removes tensor with 'required_grad' from any cache as these
        # do not support deepcopy!
        make_sure_nothing_in_cache(self.q)
        if self._optimizer:
            make_sure_nothing_in_cache(self._optimizer.q)
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
