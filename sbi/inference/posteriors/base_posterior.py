# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
import torch.distributions.transforms as torch_tf
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
from sbi.utils.sbiutils import (
    check_warn_and_setstate,
    mcmc_transform,
    optimize_potential_fn,
    rejection_sample,
)
from sbi.utils.torchutils import (
    ScalarFloat,
    atleast_2d_float32_tensor,
    ensure_theta_batched,
)
from sbi.utils.user_input_checks import check_for_possibly_batched_x_shape, process_x


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
        sample_with: str,
        mcmc_method: str = "slice_np",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of the simulator data.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection`].
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
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `proposal` as the proposal distribtution.
                `max_sampling_batch_size` as the batchsize of samples being drawn from
                the proposal at every iteration. `num_samples_to_find_max` as the
                number of samples that are used to find the maximum of the
                `potential_fn / proposal` ratio. `num_iter_to_find_max` as the number
                of gradient ascent iterations to find the maximum of that ratio. `m` as
                multiplier to that ratio.
            device: Training device, e.g., cpu or cuda.
        """
        if method_family in ("snpe", "snle", "snre_a", "snre_b"):
            self._method_family = method_family
        else:
            raise ValueError(f"Method family '{method_family}' unsupported.")

        self.net = neural_net

        self.set_mcmc_method(mcmc_method)
        self.set_mcmc_parameters(mcmc_parameters)
        self.set_sample_with(sample_with)
        self.set_rejection_sampling_parameters(rejection_sampling_parameters)

        self._leakage_density_correction_factor = None  # Correction factor for SNPE.
        self._mcmc_init_params = None
        self._num_trained_rounds = 0
        self._prior = prior
        self._x = None
        self._num_iid_trials = None
        self._x_shape = x_shape
        self._device = device
        # Methods capable of handling iid xo.
        self._iid_methods = ["snle", "snre_a", "snre_b"]
        self._allow_iid_x = method_family in self._iid_methods

        if not self._allow_iid_x:
            check_for_possibly_batched_x_shape(self._x_shape)

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
        self._x = process_x(x, self._x_shape, allow_iid_x=self._allow_iid_x).to(
            self._device
        )
        self._num_iid_trials = self._x.shape[0]

        return self

    @property
    def sample_with(self) -> str:
        """
        Return `True` if NeuralPosterior instance should use MCMC in `.sample()`.
        """
        return self._sample_with

    @sample_with.setter
    def sample_with(self, value: str) -> None:
        """See `set_sample_with`."""
        self.set_sample_with(value)

    def set_sample_with(self, sample_with: str) -> "NeuralPosterior":
        """Set the sampling method for the `NeuralPosterior`.

        Args:
            sample_with: The method to sample with.

        Returns:
            `NeuralPosterior` for chainable calls.

        Raises:
            ValueError: on attempt to turn off MCMC sampling for family of methods that
                do not support rejection sampling.
        """
        if sample_with not in ("mcmc", "rejection"):
            raise NameError(
                "The only implemented sampling methods are `mcmc` and `rejection`."
            )
        self._sample_with = sample_with
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

    @property
    def rejection_sampling_parameters(self) -> dict:
        """Returns rejection sampling parameters."""
        if self._rejection_sampling_parameters is None:
            return {}
        else:
            return self._rejection_sampling_parameters

    @rejection_sampling_parameters.setter
    def rejection_sampling_parameters(self, parameters: Dict[str, Any]) -> None:
        """See `set_rejection_sampling_parameters`."""
        self.set_rejection_sampling_parameters(parameters)

    def set_rejection_sampling_parameters(
        self, parameters: Dict[str, Any]
    ) -> "NeuralPosterior":
        """Sets parameters for rejection sampling and returns `NeuralPosterior`.

        Args:
            parameters: Dictonary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `proposal` as the proposal distribtution. `num_samples_to_find_max`
                as the number of samples that are used to find the maximum of the
                `potential_fn / proposal` ratio. `m` as multiplier to that ratio.
                `sampling_batch_size` as the batchsize of samples being drawn from
                the proposal at every iteration.

        Returns:
            `NeuralPosterior for chainable calls.
        """
        self._rejection_sampling_parameters = parameters
        return self

    @abstractmethod
    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
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

    def copy_hyperparameters_from(
        self, posterior: "NeuralPosterior"
    ) -> "NeuralPosterior":
        """
        Copies the hyperparameters from a given posterior to `self`.

        The hyperparameters that are copied are:

        - Sampling parameters (MCMC for all methods, rejection sampling for SNPE).
        - `default_x` at which to evaluate the posterior.

        Args:
            posterior: Posterior that the hyperparameters are copied from.

        Returns:
            Posterior object with the same hyperparameters as the passed posterior.
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
        self, theta: Tensor, x: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns $\theta$ and $x$ in shape that can be used by posterior.log_prob().

        Checks shapes of $\theta$ and $x$ and then repeats $x$ as often as there were
        batch elements in $\theta$.

        Moves $\theta$ and $x$ to the device of the neural net.

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
        if not self._allow_iid_x:
            self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        return theta.to(self._device), x.to(self._device)

    def _prepare_for_sample(
        self, x: Tensor, sample_shape: Optional[Tensor]
    ) -> Tuple[Tensor, int]:
        r"""
        Return checked, reshaped, potentially default values for `x` and `sample_shape`.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x_o` if previously provided for multiround training, or
                to a set default (see `set_default_x()` method).

        Returns: Single (default) $x$ with batch dimension; an integer number of
            samples.
        """

        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        if not self._allow_iid_x:
            self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)
        num_samples = torch.Size(sample_shape).numel()

        # Move x to current device.
        return x.to(self._device), num_samples

    def _potentially_replace_mcmc_parameters(
        self, mcmc_method: Optional[str], mcmc_parameters: Optional[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Return potentially default values to sample from the posterior with MCMC.

        Args:
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.

        Returns: A (default) mcmc method and (potentially
            default) mcmc parameters.
        """
        mcmc_method = mcmc_method if mcmc_method is not None else self.mcmc_method
        mcmc_parameters = (
            mcmc_parameters if mcmc_parameters is not None else self.mcmc_parameters
        )
        return mcmc_method, mcmc_parameters

    def _potentially_replace_rejection_parameters(
        self, rejection_sampling_parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Return potentially default values to rejection sample the posterior.

        Args:
            rejection_sampling_parameters: Dictionary overriding the default
                parameters for rejection sampling. The following parameters are
                supported: `proposal` as the proposal distribtution.
                `num_samples_to_find_max` as the number of samples that are used to
                find the maximum of the `potential_fn / proposal` ratio. `m` as
                multiplier to that ratio. `sampling_batch_size` as the batchsize of
                samples being drawn from the proposal at every iteration.

        Returns: Potentially default rejection sampling parameters.
        """
        rejection_sampling_parameters = (
            rejection_sampling_parameters
            if rejection_sampling_parameters is not None
            else self.rejection_sampling_parameters
        )

        return rejection_sampling_parameters

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

        return samples.type(torch.float32).to(self._device)

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
        sample_with: str = "mcmc",
        show_progress_bars: bool = True,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""
        Return samples from conditional posterior $p(\theta_i|\theta_j, x)$.

        In this function, we do not sample from the full posterior, but instead only
        from a few parameter dimensions while the other parameter dimensions are kept
        fixed at values specified in `condition`.

        Samples are obtained with MCMC or rejection sampling.

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
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection`]. In this method, the value of
                `self.sample_with` will be ignored.
            show_progress_bars: Whether to show sampling progress monitor.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported:
                `thin` to set the thinning factor for the chain.
                `warmup_steps` to set the initial number of samples to discard.
                `num_chains` for the number of chains.
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
                `enable_transform` a bool indicating whether MCMC is performed in
                z-scored (and unconstrained) space.
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `proposal` as the proposal distribtution (default is the prior).
                `max_sampling_batch_size` as the batchsize of samples being drawn from
                the proposal at every iteration.
                `num_samples_to_find_max` as the number of samples that are used to
                find the maximum of the `potential_fn / proposal` ratio.
                `num_iter_to_find_max` as the number of gradient ascent iterations to
                find the maximum of that ratio.
                `m` as multiplier to that ratio.

        Returns:
            Samples from conditional posterior.
        """

        self.net.eval()

        x, num_samples = self._prepare_for_sample(x, sample_shape)

        if sample_with == "mcmc":
            mcmc_method, mcmc_parameters = self._potentially_replace_mcmc_parameters(
                mcmc_method, mcmc_parameters
            )
            transform = mcmc_transform(
                self._prior, device=self._device, **mcmc_parameters
            )
            transform_dims_to_sample = RestrictedTransformForConditional(
                transform, condition, dims_to_sample
            )

            condition = atleast_2d_float32_tensor(condition)

            # Transform the `condition` to unconstrained space.
            transformed_condition = transform(condition)
            cond_potential_fn_provider = ConditionalPotentialFunctionProvider(
                potential_fn_provider, transformed_condition, dims_to_sample
            )

            transformed_samples = self._sample_posterior_mcmc(
                num_samples=num_samples,
                potential_fn=cond_potential_fn_provider(
                    self._prior,
                    self.net,
                    x,
                    mcmc_method,
                    transform,
                ),
                init_fn=self._build_mcmc_init_fn(
                    # Restrict prior to sample only free dimensions.
                    RestrictedPriorForConditional(self._prior, dims_to_sample),
                    cond_potential_fn_provider(
                        self._prior,
                        self.net,
                        x,
                        "slice_np",
                        transform,
                    ),
                    transform=transform_dims_to_sample,
                    **mcmc_parameters,
                ),
                mcmc_method=mcmc_method,
                condition=condition,
                dims_to_sample=dims_to_sample,
                show_progress_bars=show_progress_bars,
                **mcmc_parameters,
            )
            samples = transform_dims_to_sample.inv(transformed_samples)
        elif sample_with == "rejection":
            cond_potential_fn_provider = ConditionalPotentialFunctionProvider(
                potential_fn_provider, condition, dims_to_sample
            )
            rejection_sampling_parameters = (
                self._potentially_replace_rejection_parameters(
                    rejection_sampling_parameters
                )
            )
            if "proposal" not in rejection_sampling_parameters:
                rejection_sampling_parameters[
                    "proposal"
                ] = RestrictedPriorForConditional(self._prior, dims_to_sample)

            samples, _ = rejection_sample(
                potential_fn=cond_potential_fn_provider(
                    self._prior, self.net, x, "rejection"
                ),
                num_samples=num_samples,
                **rejection_sampling_parameters,
            )
        else:
            raise NameError(
                "The only implemented sampling methods are `mcmc` and `rejection`."
            )

        self.net.train(True)

        return samples.reshape((*sample_shape, -1))

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1000,
        learning_rate: float = 0.1,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        num_to_optimize: int = 100,
        save_best_every: int = 10,
        show_progress_bars: bool = True,
        log_prob_kwargs: Dict = {},
    ) -> Tensor:
        """
        Returns the maximum-a-posteriori estimate (MAP).

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
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
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

        warn(
            "This method for obtaining the MAP estimate was introduced recently "
            "(sbi v0.15.0) and has not been tested extensively yet. You might have to "
            "tune the hyperparameters, especially `num_iter` and `learning_rate`. If "
            "you experience problems, please create an issue on Github: "
            "https://github.com/mackelab/sbi/issues"
        )

        if isinstance(init_method, str):
            # Find initial position.
            if init_method == "posterior":
                inits = self.sample(
                    (num_init_samples,), x=x, show_progress_bars=show_progress_bars
                )
            elif init_method == "prior":
                inits = self._prior.sample((num_init_samples,))
            elif isinstance(init_method, Tensor):
                inits = init_method
            else:
                raise NameError(
                    "`init_method` not specified. Use either `posterior` "
                    "or `prior` or provide a tensor."
                )
        else:
            inits = init_method

        def potential_fn(theta):
            return self.log_prob(theta, x=x, track_gradients=True, **log_prob_kwargs)

        interruption_note = """The last estimate of the MAP can be accessed via the
                            `posterior.map_` attribute."""

        self.map_, _ = optimize_potential_fn(
            potential_fn=potential_fn,
            inits=inits,
            dist_specifying_bounds=self._prior,
            num_iter=num_iter,
            learning_rate=learning_rate,
            num_to_optimize=num_to_optimize,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
            interruption_note=interruption_note,
        )

        return self.map_

    def _build_mcmc_init_fn(
        self,
        prior: Any,
        potential_fn: Callable,
        transform: torch_tf.Transform,
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
            return lambda: prior_init(prior, transform=transform, **kwargs)
        elif init_strategy == "sir":
            return lambda: sir(prior, potential_fn, transform=transform, **kwargs)
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
                some sbi methods for reasons depending on the scenario:

                    - in case you want to evaluate or sample conditioned on several xs
                    e.g., (p(theta | [x1, x2, x3])), this is not supported yet except
                    when using likelihood based SNLE.

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
    def _match_theta_and_x_batch_shapes(
        theta: Tensor, x: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""Return $\theta$ and `x` with batch shape matched to each other.

        When `x` is just a single observation it is repeated for all entries in the
        batch of $\theta$s. When there is a batch of multiple `x`, i.e., iid `x`, then
        individual `x` are repeated in the pattern AABBCC and individual $\theta$ are
        repeated in the pattern ABCABC to cover all combinations.

        This is needed in nflows in order to have matching shapes of theta and context
        `x` when evaluating the neural network.

        Args:
            x: (a batch of iid) data
            theta: a batch of parameters

        Returns:
            theta: with shape (theta_batch_size * x_batch_size, *theta_shape)
            x: with shape (theta_batch_size * x_batch_size, *x_shape)
        """

        # Theta and x are ensured to have a batch dim, get the shape.
        theta_batch_size, *theta_shape = theta.shape
        x_batch_size, *x_shape = x.shape

        # Repeat iid trials as AABBCC.
        x_repeated = x.repeat_interleave(theta_batch_size, dim=0)
        # Repeat theta as ABCABC.
        theta_repeated = theta.repeat(x_batch_size, 1)

        # Double check: batch size for log prob evaluation must match.
        assert x_repeated.shape == torch.Size(
            [theta_batch_size * x_batch_size, *x_shape]
        )
        assert theta_repeated.shape == torch.Size(
            [theta_batch_size * x_batch_size, *theta_shape]
        )

        return theta_repeated, x_repeated

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

    def __getstate__(self) -> Dict:
        """
        Returns the state of the object that is supposed to be pickled.

        Returns:
            Dictionary containing the state.
        """
        return self.__dict__

    def __setstate__(self, state_dict: Dict):
        """
        Sets the state when being loaded from pickle.

        For developers: for any new attribute added to `NeuralPosterior`, we have to
        add an entry here using `check_warn_and_setstate()`.

        Args:
            state_dict: State to be restored.
        """

        # In the beginning, the warning message is empty.
        warning_msg = ""

        state_dict, warning_msg = check_warn_and_setstate(
            state_dict, "_device", "cpu", warning_msg
        )

        if state_dict["_x"] is not None:
            state_dict, warning_msg = check_warn_and_setstate(
                state_dict, "_num_iid_trials", state_dict["_x"].shape[0], warning_msg
            )
        else:
            state_dict, warning_msg = check_warn_and_setstate(
                state_dict, "_num_iid_trials", None, warning_msg
            )

        state_dict, warning_msg = check_warn_and_setstate(
            state_dict, "_iid_methods", ["snle", "snre_a", "snre_b"], warning_msg
        )

        state_dict, warning_msg = check_warn_and_setstate(
            state_dict,
            "_allow_iid_x",
            state_dict["_method_family"] in state_dict["_iid_methods"],
            warning_msg,
        )

        state_dict, warning_msg = check_warn_and_setstate(
            state_dict,
            "_sample_with",
            "rejection" if state_dict["_method_family"] == "snpe" else "mcmc",
            warning_msg,
        )

        if warning_msg:
            warning_description = (
                "You had saved the posterior under an older version of `sbi`. To make "
                "the loaded version comply with the version you are using right now, "
                "we had to set the following attributes:"
            )
            warn(warning_description + warning_msg)

        self.__dict__ = state_dict


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

    def __call__(
        self,
        prior,
        net: nn.Module,
        x: Tensor,
        method: str,
        transform: torch_tf.Transform = torch_tf.identity_transform,
    ) -> Callable:
        """Return potential function.

        Switch on numpy or pyro potential function based on `method`.
        """
        # Set prior, net, and x as attributes of unconditional potential_fn_provider.
        _ = self.potential_fn_provider.__call__(prior, net, x, method, transform)
        self.device = next(net.parameters()).device

        if method == "slice":
            return partial(self.pyro_potential, track_gradients=False)
        elif method in ("hmc", "nuts"):
            return partial(self.pyro_potential, track_gradients=True)
        elif "slice_np" in method:
            return partial(self.posterior_potential, track_gradients=False)
        elif method == "rejection":
            return partial(self.posterior_potential, track_gradients=True)
        else:
            NotImplementedError

    def posterior_potential(
        self, theta: np.ndarray, track_gradients: bool = False
    ) -> ScalarFloat:
        r"""
        Return conditional posterior log-probability or $-\infty$ if outside prior.

        The only differences to the `np_potential` is that it tracks the gradients and
        does not return a `numpy` array.

        Args:
            theta: Free parameters $\theta_i$, batch dimension 1.

        Returns:
            Conditional posterior log-probability $\log(p(\theta_i|\theta_j, x))$,
            masked outside of prior.
        """
        theta = ensure_theta_batched(torch.as_tensor(theta, dtype=torch.float32)).to(
            self.device
        )

        theta_condition = deepcopy(self.condition).to(self.device)
        theta_condition = theta_condition.repeat(theta.shape[0], 1)
        theta_condition[:, self.dims_to_sample] = theta

        return self.potential_fn_provider.posterior_potential(
            theta_condition, track_gradients=track_gradients
        )

    def pyro_potential(
        self, theta: Dict[str, Tensor], track_gradients: bool = False
    ) -> Tensor:
        r"""
        Return conditional posterior log-probability or $-\infty$ if outside prior.

        Args:
            theta: Free parameters $\theta_i$ (from pyro sampler).

        Returns:
            Conditional posterior log-probability $\log(p(\theta_i|\theta_j, x))$,
            masked outside of prior.
        """

        theta = next(iter(theta.values()))
        theta = ensure_theta_batched(theta).to(self.device)

        theta_condition = deepcopy(self.condition).to(self.device)
        theta_condition[:, self.dims_to_sample] = theta

        return self.potential_fn_provider.pyro_potential(
            {"": theta_condition}, track_gradients=track_gradients
        )


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

    def __init__(self, full_prior: Any, dims_to_sample: List[int]):
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


class RestrictedTransformForConditional(nn.Module):
    """
    Class to restrict the transform to fewer dimensions for conditional sampling.

    The resulting transform transforms only the free dimensions of the conditional.
    Notably, the `log_abs_det` is computed given all dimensions. However, the
    `log_abs_det` stemming from the fixed dimensions is a constant and drops out during
    MCMC.

    This is needed for the the MCMC initialization functions when conditioning and
    when transforming the samples back into the original theta space after sampling.
    """

    def __init__(
        self,
        transform: torch_tf.Transform,
        condition: Tensor,
        dims_to_sample: List[int],
    ) -> None:
        super().__init__()
        self.transform = transform
        self.condition = ensure_theta_batched(condition)
        self.dims_to_sample = dims_to_sample

    def forward(self, theta: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Transform restricted $\theta$.
        """
        full_theta = self.condition.repeat(theta.shape[0], 1)
        full_theta[:, self.dims_to_sample] = theta
        tf_full_theta = self.transform(full_theta)
        return tf_full_theta[:, self.dims_to_sample]

    def inv(self, theta: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Inverse transform restricted $\theta$.
        """
        full_theta = self.condition.repeat(theta.shape[0], 1)
        full_theta[:, self.dims_to_sample] = theta
        tf_full_theta = self.transform.inv(full_theta)
        return tf_full_theta[:, self.dims_to_sample]

    def log_abs_det_jacobian(self, theta1: Tensor, theta2: Tensor) -> Tensor:
        """
        Return the `log_abs_det_jacobian` of |dtheta1 / dtheta2|.

        The determinant is summed over all dimensions, not just the `dims_to_sample`
        ones.
        """
        full_theta1 = self.condition.repeat(theta1.shape[0], 1)
        full_theta1[:, self.dims_to_sample] = theta1
        full_theta2 = self.condition.repeat(theta2.shape[0], 1)
        full_theta2[:, self.dims_to_sample] = theta2
        log_abs_det = self.transform.log_abs_det_jacobian(full_theta1, full_theta2)
        return log_abs_det
