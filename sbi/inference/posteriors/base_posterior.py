# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from copy import deepcopy
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import Tensor
from torch import multiprocessing as mp
from torch import nn

from sbi import utils as utils
from sbi.mcmc import (
    IterateParameters,
    Slice,
    SliceSampler,
    SliceSamplerVectorized,
    prior_init,
    sir,
)
from sbi.types import Array, Shape
from sbi.user_input.user_input_checks import process_x
from sbi.utils.torchutils import (
    ScalarFloat,
    atleast_2d_float32_tensor,
    ensure_theta_batched,
)


class NeuralPosterior(ABC):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods.<br/><br/>
    All inference methods in sbi train a neural network which is then used to obtain
    the posterior distribution. The `NeuralPosterior` class wraps the trained network
    such that one can directly evaluate the (unnormalized) log probability and draw
    samples from the posterior. The neural network itself can be accessed via the `.net`
    attribute.
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        prior,
        x_shape: torch.Size,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling. Init strategies may have their own keywords
                which can also be set from `mcmc_parameters`.
            device: Training device, e.g., cpu or cuda.
        """
        if method_family in ("snpe", "snle", "snre_a", "snre_b"):
            self._method_family = method_family
        else:
            raise ValueError("Method family unsupported.")

        self.net = neural_net

        self.set_mcmc_method(mcmc_method)
        self.set_mcmc_parameters(mcmc_parameters)

        self._leakage_density_correction_factor = None  # Correction factor for SNPE.
        self._mcmc_init_params = None
        self._num_trained_rounds = 0
        self._prior = prior
        self._x = None
        self._x_shape = x_shape
        self._device = device

    @property
    def default_x(self) -> Optional[Tensor]:
        """Return default x used by `.sample(), .log_prob` as conditioning context."""
        return self._x

    @default_x.setter
    def default_x(self, x: Tensor) -> None:
        """See `set_default_x`."""
        self.set_default_x(x)

    def set_default_x(self, x: Tensor) -> "NeuralPosterior":
        """Set new default x for `.sample(), .log_prob` to use as conditioning context.

        This is a pure convenience to avoid having to repeatedly specify `x` in calls to
        `.sample()` and `.log_prob()` - only θ needs to be passed.

        This convenience is particularly useful when the posterior is focused, i.e.
        has been trained over multiple rounds to be accurate in the vicinity of a
        particular `x=x_o` (you can check if your posterior object is focused by
        printing it).

        NOTE: this method is chainable, i.e. will return the NeuralPosterior object so
        that calls like `posterior.set_default_x(my_x).sample(mytheta)` are possible.

        Args:
            x: The default observation to set for the posterior $p(theta|x)$.

        Returns:
            `NeuralPosterior` that will use a default `x` when not explicitly passed.
        """
        processed_x = process_x(x, self._x_shape)
        self._x = processed_x

        return self

    @property
    def mcmc_method(self) -> str:
        """Returns MCMC method."""
        return self._mcmc_method

    @mcmc_method.setter
    def mcmc_method(self, method: str) -> None:
        """See `set_mcmc_method`."""
        self.set_mcmc_method(method)

    def set_mcmc_method(self, method: str) -> "NeuralPosterior":
        """Sets sampling method to for MCMC and returns `NeuralPosterior`.

        Args:
            method: Method to use.

        Returns:
            `NeuralPosterior` for chainable calls.
        """
        self._mcmc_method = method
        return self

    @property
    def mcmc_parameters(self) -> dict:
        """Returns MCMC parameters."""
        if self._mcmc_parameters is None:
            return {}
        else:
            return self._mcmc_parameters

    @mcmc_parameters.setter
    def mcmc_parameters(self, parameters: Dict[str, Any]) -> None:
        """See `set_mcmc_parameters`."""
        self.set_mcmc_parameters(parameters)

    def set_mcmc_parameters(self, parameters: Dict[str, Any]) -> "NeuralPosterior":
        """Sets parameters for MCMC and returns `NeuralPosterior`.

        Args:
            parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.

        Returns:
            `NeuralPosterior` for chainable calls.
        """
        self._mcmc_parameters = parameters
        return self

    @abstractmethod
    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,
    ) -> Tensor:
        """See child classes for docstring."""
        pass

    @abstractmethod
    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """See child classes for docstring."""
        pass

    def copy_hyperparameters_from(self, posterior: "NeuralPosterior"):
        """
        Copies the hyperparameters from a given posterior to `self`.

        The hyperparameters that are copied are:

        - Sampling parameters (MCMC for all methods, rejection sampling for SNPE).
        - `default_x` at which to evaluate the posterior.

        Args:
            posterior: Posterior that the hyperparameters are copied from.
        
        Returns: Posterior object with the same hyperparameters as the passed posterior.
            This makes the call chainable:
            `posterior = infer.build_posterior().copy_hyperparameters_from(proposal)`
        """

        assert isinstance(
            posterior, NeuralPosterior
        ), "`copy_state_from` must be a `NeuralPosterior`."

        self.set_mcmc_method(posterior._mcmc_method)
        self.set_mcmc_parameters(posterior._mcmc_parameters)
        self.set_default_x(posterior.default_x)
        self._mcmc_init_params = posterior._mcmc_init_params
        if hasattr(self, "_sample_with_mcmc"):
            self.set_sample_with_mcmc(posterior._sample_with_mcmc)
        if hasattr(self, "_rejection_sampling_parameters"):
            self.set_rejection_sampling_parameters(
                posterior._rejection_sampling_parameters
            )

        return self

    def _prepare_theta_and_x_for_log_prob_(
        self, theta: Tensor, x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns $\theta$ and $x$ in shape that can be used by posterior.log_prob().

        Checks shapes of $\theta$ and $x$ and then repeats $x$ as often as there were
        batch elements in $\theta$.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided, fall
                back onto an `x_o` if previously provided for multi-round training, or
                to another default if set later for convenience, see `.set_default_x()`.

        Returns:
            ($\theta$, $x$) with the same batch dimension, where $x$ is repeated as
            often as there were batch elements in $\theta$ originally.
        """

        theta = ensure_theta_batched(torch.as_tensor(theta))

        # Select and check x to condition on.
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        # Repeat `x` in case of evaluation on multiple `theta`. This is needed below in
        # when calling nflows in order to have matching shapes of theta and context x
        # at neural network evaluation time.
        x = self._match_x_with_theta_batch_shape(x, theta)

        return theta, x

    def _prepare_for_sample(
        self,
        x: Tensor,
        sample_shape: Optional[Tensor],
        mcmc_method: Optional[str],
        mcmc_parameters: Optional[Dict[str, Any]],
    ) -> Tuple[Tensor, int, str, Dict[str, Any]]:
        r"""
        Return checked and (potentially default) values to sample from the posterior.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x_o` if previously provided for multiround training, or
                to a set default (see `set_default_x()` method).
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.

        Returns: Single (potentially default) $x$ with batch dimension; an integer
            number of samples; a (potentially default) mcmc method; and (potentially
            default) mcmc parameters.
        """

        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)
        num_samples = torch.Size(sample_shape).numel()

        mcmc_method = mcmc_method if mcmc_method is not None else self.mcmc_method
        mcmc_parameters = (
            mcmc_parameters if mcmc_parameters is not None else self.mcmc_parameters
        )

        # Move x to current device.
        x = x.to(self._device)

        return x, num_samples, mcmc_method, mcmc_parameters

    def _sample_posterior_mcmc(
        self,
        num_samples: int,
        potential_fn: Callable,
        init_fn: Optional[Callable] = None,
        mcmc_method: str = "slice_np",
        thin: int = 10,
        warmup_steps: int = 20,
        num_chains: Optional[int] = 1,
        show_progress_bars: bool = True,
        **kwargs,
    ) -> Tensor:
        r"""
        Return MCMC samples from posterior $p(\theta|x)$.

        This function is used in any case by SNLE and SNRE, but can also be used by SNPE
        in order to deal with strong leakage. Depending on the inference method, a
        different potential function for the MCMC sampler is required.

        Args:
            num_samples: Desired number of samples.
            potential_fn: Potential function used for MCMC sampling.
            init_fn: Initialisation function for each chain. When called without
                arguments, it will return a single batched parameter set.
            mcmc_method: Sampling method. Currently defaults to `slice_np` for a custom
                numpy implementation of slice sampling; select `hmc`, `nuts` or `slice`
                for Pyro-based sampling.
            thin: Thinning factor for the chain, e.g. for `thin=3` only every third
                sample will be returned, until a total of `num_samples`.
            warmup_steps: Initial number of samples to discard.
            num_chains: Whether to sample in parallel. If None, use all but one CPU.
            show_progress_bars: Whether to show a progressbar during sampling.
            kwargs: Absorbs passed but unused arguments. E.g. in
                `DirectPosterior.sample()` we pass `mcmc_parameters` which might
                contain also entries for the mcmc initialization, which are not used
                here.

        Returns:
            Tensor of shape (num_samples, shape_of_single_theta).
        """

        initial_params = torch.cat([init_fn() for _ in range(num_chains)])

        track_gradients = mcmc_method in ("hmc", "nuts")
        with torch.set_grad_enabled(track_gradients):
            if mcmc_method in ("slice_np", "slice_np_vectorized"):
                samples = self._slice_np_mcmc(
                    num_samples=num_samples,
                    potential_function=potential_fn,
                    initial_params=initial_params,
                    thin=thin,
                    warmup_steps=warmup_steps,
                    vectorized=(mcmc_method == "slice_np_vectorized"),
                    show_progress_bars=show_progress_bars,
                )
            elif mcmc_method in ("hmc", "nuts", "slice"):
                samples = self._pyro_mcmc(
                    num_samples=num_samples,
                    potential_function=potential_fn,
                    initial_params=initial_params,
                    mcmc_method=mcmc_method,
                    thin=thin,
                    warmup_steps=warmup_steps,
                    num_chains=num_chains,
                    show_progress_bars=show_progress_bars,
                ).detach()
            else:
                raise NameError

        return samples

    def _slice_np_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        initial_params: Tensor,
        thin: int,
        warmup_steps: int,
        vectorized: bool = False,
        show_progress_bars: bool = True,
    ) -> Tensor:
        """
        Custom implementation of slice sampling using Numpy.

        Args:
            num_samples: Desired number of samples.
            potential_function: A callable **class**.
            initial_params: Initial parameters for MCMC chain.
            thin: Thinning (subsampling) factor.
            warmup_steps: Initial number of samples to discard.
            vectorized: Whether to use a vectorized implementation of
                the Slice sampler (still experimental).
            show_progress_bars: Whether to show a progressbar during sampling;
                can only be turned off for vectorized sampler.

        Returns: Tensor of shape (num_samples, shape_of_single_theta).
        """
        num_chains = initial_params.shape[0]
        dim_samples = initial_params.shape[1]

        if not vectorized:  # Sample all chains sequentially
            all_samples = []
            for c in range(num_chains):
                posterior_sampler = SliceSampler(
                    utils.tensor2numpy(initial_params[c, :]).reshape(-1),
                    lp_f=potential_function,
                    thin=thin,
                    verbose=show_progress_bars,
                )
                if warmup_steps > 0:
                    posterior_sampler.gen(int(warmup_steps))
                all_samples.append(
                    posterior_sampler.gen(ceil(num_samples / num_chains))
                )
            all_samples = np.stack(all_samples).astype(np.float32)
            samples = torch.from_numpy(all_samples)  # chains x samples x dim
        else:  # Sample all chains at the same time
            posterior_sampler = SliceSamplerVectorized(
                init_params=utils.tensor2numpy(initial_params),
                log_prob_fn=potential_function,
                num_chains=num_chains,
                verbose=show_progress_bars,
            )
            warmup_ = warmup_steps * thin
            num_samples_ = ceil((num_samples * thin) / num_chains)
            samples = posterior_sampler.run(warmup_ + num_samples_)
            samples = samples[:, warmup_:, :]  # discard warmup steps
            samples = samples[:, ::thin, :]  # thin chains
            samples = torch.from_numpy(samples)  # chains x samples x dim

        # Save sample as potential next init (if init_strategy == 'latest_sample').
        self._mcmc_init_params = samples[:, -1, :].reshape(num_chains, dim_samples)

        samples = samples.reshape(-1, dim_samples)[:num_samples, :]
        assert samples.shape[0] == num_samples

        return samples.type(torch.float32)

    def _pyro_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        initial_params: Tensor,
        mcmc_method: str = "slice",
        thin: int = 10,
        warmup_steps: int = 200,
        num_chains: Optional[int] = 1,
        show_progress_bars: bool = True,
    ):
        r"""Return samples obtained using Pyro HMC, NUTS for slice kernels.

        Args:
            num_samples: Desired number of samples.
            potential_function: A callable **class**. A class, but not a function,
                is picklable for Pyro MCMC to use it across chains in parallel,
                even when the potential function requires evaluating a neural network.
            mcmc_method: One of `hmc`, `nuts` or `slice`.
            thin: Thinning (subsampling) factor.
            warmup_steps: Initial number of samples to discard.
            num_chains: Whether to sample in parallel. If None, use all but one CPU.
            show_progress_bars: Whether to show a progressbar during sampling.

        Returns: Tensor of shape (num_samples, shape_of_single_theta).
        """
        num_chains = mp.cpu_count - 1 if num_chains is None else num_chains

        kernels = dict(slice=Slice, hmc=HMC, nuts=NUTS)

        sampler = MCMC(
            kernel=kernels[mcmc_method](potential_fn=potential_function),
            num_samples=(thin * num_samples) // num_chains + num_chains,
            warmup_steps=warmup_steps,
            initial_params={"": initial_params},
            num_chains=num_chains,
            mp_context="fork",
            disable_progbar=not show_progress_bars,
            transforms={},
        )
        sampler.run()
        samples = next(iter(sampler.get_samples().values())).reshape(
            -1, initial_params.shape[1]  # .shape[1] = dim of theta
        )

        samples = samples[::thin][:num_samples]
        assert samples.shape[0] == num_samples

        return samples

    def sample_conditional(
        self,
        potential_fn_provider: Callable,
        sample_shape: Shape,
        condition: Tensor,
        dims_to_sample: List[int],
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""
        Return samples from conditional posterior $p(\theta_i|\theta_j, x)$.

        In this function, we do not sample from the full posterior, but instead only
        from a few parameter dimensions while the other parameter dimensions are kept
        fixed at values specified in `condition`.

        Samples are obtained with MCMC.

        Args:
            potential_fn_provider: Returns the potential function for the unconditional
                posterior.
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            condition: Parameter set that all dimensions not specified in
                `dims_to_sample` will be fixed to. Should contain dim_theta elements,
                i.e. it could e.g. be a sample from the posterior distribution.
                The entries at all `dims_to_sample` will be ignored.
            dims_to_sample: Which dimensions to sample from. The dimensions not
                specified in `dims_to_sample` will be fixed to values given in
                `condition`.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x_o` if previously provided for multiround training, or
                to a set default (see `set_default_x()` method).
            show_progress_bars: Whether to show sampling progress monitor.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.

        Returns:
            Samples from conditional posterior.
        """

        x, num_samples, mcmc_method, mcmc_parameters = self._prepare_for_sample(
            x, sample_shape, mcmc_method, mcmc_parameters
        )

        self.net.eval()

        cond_potential_fn_provider = ConditionalPotentialFunctionProvider(
            potential_fn_provider, condition, dims_to_sample
        )

        samples = self._sample_posterior_mcmc(
            num_samples=num_samples,
            potential_fn=cond_potential_fn_provider(
                self._prior, self.net, x, mcmc_method
            ),
            init_fn=self._build_mcmc_init_fn(
                # Restrict prior to sample only free dimensions.
                RestrictedPriorForConditional(self._prior, dims_to_sample),
                cond_potential_fn_provider(self._prior, self.net, x, "slice_np"),
                **mcmc_parameters,
            ),
            mcmc_method=mcmc_method,
            condition=condition,
            dims_to_sample=dims_to_sample,
            show_progress_bars=show_progress_bars,
            **mcmc_parameters,
        )

        self.net.train(True)

        return samples.reshape((*sample_shape, -1))

    def _build_mcmc_init_fn(
        self,
        prior: Any,
        potential_fn: Callable,
        init_strategy: str = "prior",
        **kwargs,
    ) -> Callable:
        """
        Return function that, when called, creates an initial parameter set for MCMC.

        Args:
            prior: Prior distribution.
            potential_fn: Potential function that the candidate samples are weighted
                with.
            init_strategy: Specifies the initialization method. Either of
                [`prior`|`sir`|`latest_sample`].
            kwargs: Passed on to init function. This way, init specific keywords can
                be set through `mcmc_parameters`. Unused arguments should be absorbed.

        Returns: Initialization function.
        """
        if init_strategy == "prior":
            return lambda: prior_init(prior, **kwargs)
        elif init_strategy == "sir":
            return lambda: sir(prior, potential_fn, **kwargs)
        elif init_strategy == "latest_sample":
            latest_sample = IterateParameters(self._mcmc_init_params, **kwargs)
            return latest_sample
        else:
            raise NotImplementedError

    def _x_else_default_x(self, x: Optional[Array]) -> Array:
        if x is not None:
            return x
        elif self.default_x is None:
            raise ValueError(
                "Context `x` needed when a default has not been set."
                "If you'd like to have a default, use the `.set_default_x()` method."
            )
        else:
            return self.default_x

    def _ensure_x_consistent_with_default_x(self, x: Tensor) -> None:
        """Check consistency with the shape of `self.default_x` (unless it's None)."""

        # TODO: This is to check the passed x matches the NN input dimensions by
        # comparing to `default_x`, which was checked in user input checks to match the
        # simulator output. Later if we might not have `self.default_x` we might want to
        # compare to the input dimension of `self.net` here.
        if self.default_x is not None:
            assert (
                x.shape == self.default_x.shape
            ), f"""The shape of the passed `x` {x.shape} and must match the shape of `x`
            used during training, {self.default_x.shape}."""

    @staticmethod
    def _ensure_single_x(x: Tensor) -> None:
        """Raise a ValueError if multiple (a batch of) xs are passed."""

        inferred_batch_size, *_ = x.shape

        if inferred_batch_size > 1:

            raise ValueError(
                """The `x` passed to condition the posterior for evaluation or sampling
                has an inferred batch shape larger than one. This is not supported in
                sbi for reasons depending on the scenario:

                    - in case you want to evaluate or sample conditioned on several xs
                    e.g., (p(theta | [x1, x2, x3])), this is not supported yet in sbi.

                    - in case you trained with a single round to do amortized inference
                    and now you want to evaluate or sample a given theta conditioned on
                    several xs, one after the other, e.g, p(theta | x1), p(theta | x2),
                    p(theta| x3): this broadcasting across xs is not supported in sbi.
                    Instead, what you can do it to call posterior.log_prob(theta, xi)
                    multiple times with different xi.

                    - finally, if your observation is multidimensional, e.g., an image,
                    make sure to pass it with a leading batch dimension, e.g., with
                    shape (1, xdim1, xdim2). Beware that the current implementation
                    of sbi might not provide stable support for this and result in
                    shape mismatches.
                """
            )

    @staticmethod
    def _match_x_with_theta_batch_shape(x: Tensor, theta: Tensor) -> Tensor:
        """Return `x` with batch shape matched to that of `theta`.

        This is needed in nflows in order to have matching shapes of theta and context
        `x` when evaluating the neural network.
        """

        # Theta and x are ensured to have a batch dim, get the shape.
        theta_batch_size, *_ = theta.shape
        x_batch_size, *x_shape = x.shape

        assert x_batch_size == 1, "Batch size 1 should be enforced by caller."
        if theta_batch_size > x_batch_size:
            x_matched = x.expand(theta_batch_size, *x_shape)

            # Double check.
            x_matched_batch_size, *x_matched_shape = x_matched.shape
            assert x_matched_batch_size == theta_batch_size
            assert x_matched_shape == x_shape
        else:
            x_matched = x

        return x_matched

    def _get_net_name(self) -> str:
        """
        Return the name of the neural network used for inference.

        For SNRE the net is sequential because it has a standardization net. Hence,
        we only access its last entry.
        """

        try:
            # Why not `isinstance(self.net[0], StandardizeInputs)`? Because
            # `StandardizeInputs` is defined within a function in
            # neural_nets/classifier.py and can not be imported here.
            # TODO: Refactor this into the net's __str__  method.
            if self.net[0].__class__.__name__ == "StandardizeInputs":
                actual_net = self.net[-1]
            else:
                actual_net = self.net
        except TypeError:
            # If self.net is not a sequential, self.net[0] will throw an error.
            actual_net = self.net

        return actual_net.__class__.__name__.lower()

    def __repr__(self):
        desc = f"""{self.__class__.__name__}(
               method_family={self._method_family},
               net=<a {self.net.__class__.__name__}, see `.net` for details>,
               prior={self._prior!r},
               x_shape={self._x_shape!r})
               """
        return desc

    def __str__(self):
        msg = {0: "untrained", 1: "amortized"}

        focused_msg = "multi-round"

        default_x_msg = (
            f" Evaluates and samples by default at x={self.default_x.tolist()!r}."
            if self.default_x is not None
            else ""
        )

        desc = (
            f"Posterior conditional density p(θ|x) "
            f"({msg.get(self._num_trained_rounds, focused_msg)}).{default_x_msg}\n\n"
            f"This {self.__class__.__name__}-object was obtained with a "
            f"{self._method_family.upper()}-class "
            f"method using a {self._get_net_name()}.\n"
            f"{self._purpose}"
        )

        return desc


class ConditionalPotentialFunctionProvider:
    """
    Wraps the potential functions to allow for sampling from the conditional posterior.
    """

    def __init__(
        self,
        potential_fn_provider: Callable,
        condition: Tensor,
        dims_to_sample: List[int],
    ):
        """
        Args:
            potential_fn_provider: Creates potential function of unconditional
                posterior.
            condition: Parameter set that all dimensions not specified in
                `dims_to_sample` will be fixed to. Should contain dim_theta elements,
                i.e. it could e.g. be a sample from the posterior distribution.
                The entries at all `dims_to_sample` will be ignored.
            dims_to_sample: Which dimensions to sample from. The dimensions not
                specified in `dims_to_sample` will be fixed to values given in
                `condition`.
        """

        self.potential_fn_provider = potential_fn_provider
        self.condition = ensure_theta_batched(condition)
        self.dims_to_sample = dims_to_sample

    def __call__(self, prior, net: nn.Module, x: Tensor, mcmc_method: str,) -> Callable:
        """Return potential function.

        Switch on numpy or pyro potential function based on `mcmc_method`.
        """
        # Set prior, net, and x as attributes of unconditional potential_fn_provider.
        _ = self.potential_fn_provider.__call__(prior, net, x, mcmc_method)

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.ndarray) -> ScalarFloat:
        r"""
        Return conditional posterior log-probability or $-\infty$ if outside prior.

        Args:
            theta: Free parameters $\theta_i$, batch dimension 1.

        Returns:
            Conditional posterior log-probability $\log(p(\theta_i|\theta_j, x))$,
            masked outside of prior.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)

        theta_condition = deepcopy(self.condition)
        theta_condition[:, self.dims_to_sample] = theta

        return self.potential_fn_provider.np_potential(
            utils.tensor2numpy(theta_condition)
        )

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""
        Return conditional posterior log-probability or $-\infty$ if outside prior.

        Args:
            theta: Free parameters $\theta_i$ (from pyro sampler).

        Returns:
            Conditional posterior log-probability $\log(p(\theta_i|\theta_j, x))$,
            masked outside of prior.
        """

        theta = next(iter(theta.values()))

        theta_condition = deepcopy(self.condition)
        theta_condition[:, self.dims_to_sample] = theta

        return self.potential_fn_provider.pyro_potential({"": theta_condition})


class RestrictedPriorForConditional:
    """
    Class to restrict a prior to fewer dimensions as needed for conditional sampling.

    The resulting prior samples only from the free dimensions of the conditional.

    This is needed for the the MCMC initialization functions when conditioning.
    For the prior init, we could post-hoc select the relevant dimensions. But
    for SIR, we want to evaluate the `potential_fn` of the conditional
    posterior, which takes only a subset of the full parameter vector theta
    (only the `dims_to_sample`). This subset is provided by `.sample()` from
    this class.
    """

    def __init__(
        self, full_prior: Any, dims_to_sample: List[int],
    ):
        self.full_prior = full_prior
        self.dims_to_sample = dims_to_sample

    def sample(self, *args, **kwargs):
        """
        Sample only from the relevant dimension. Other dimensions are filled in
        by the `ConditionalPotentialFunctionProvider()` during MCMC.
        """
        return self.full_prior.sample(*args, **kwargs)[:, self.dims_to_sample]

    def log_prob(self, *args, **kwargs):
        r"""
        `log_prob` is same as for the full prior, because we usually evaluate
        the $\theta$ under the full joint once we have added the condition.
        """
        return self.full_prior.log_prob(*args, **kwargs)
