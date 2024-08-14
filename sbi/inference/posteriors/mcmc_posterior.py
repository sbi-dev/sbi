# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
import warnings
from copy import deepcopy
from functools import partial
from math import ceil
from typing import Any, Callable, Dict, Optional, Union
from warnings import warn

import arviz as az
import torch
import torch.distributions.transforms as torch_tf
from arviz.data import InferenceData
from joblib import Parallel, delayed
from numpy import ndarray
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import Tensor
from torch import multiprocessing as mp
from tqdm.auto import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.estimators.shape_handling import reshape_to_batch_event
from sbi.samplers.mcmc import (
    IterateParameters,
    PyMCSampler,
    SliceSamplerSerial,
    SliceSamplerVectorized,
    proposal_init,
    resample_given_potential_fn,
    sir_init,
)
from sbi.sbi_types import Shape, TorchTransform
from sbi.utils.potentialutils import pyro_potential_wrapper, transformed_potential
from sbi.utils.torchutils import ensure_theta_batched, tensor2numpy


class MCMCPosterior(NeuralPosterior):
    r"""Provides MCMC to sample from the posterior.<br/><br/>
    SNLE or SNRE train neural networks to approximate the likelihood(-ratios).
    `MCMCPosterior` allows to sample from the posterior with MCMC.
    """

    def __init__(
        self,
        potential_fn: Union[Callable, BasePotential],
        proposal: Any,
        theta_transform: Optional[TorchTransform] = None,
        method: str = "slice_np_vectorized",
        thin: int = -1,
        warmup_steps: int = 200,
        num_chains: int = 20,
        init_strategy: str = "resample",
        init_strategy_parameters: Optional[Dict[str, Any]] = None,
        init_strategy_num_candidates: Optional[int] = None,
        num_workers: int = 1,
        mp_context: str = "spawn",
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples. Must be a
                `BasePotential` or a `Callable` which takes `theta` and `x_o` as inputs.
            proposal: Proposal distribution that is used to initialize the MCMC chain.
            theta_transform: Transformation that will be applied during sampling.
                Allows to perform MCMC in unconstrained space.
            method: Method used for MCMC sampling, one of `slice_np`,
                `slice_np_vectorized`, `hmc_pyro`, `nuts_pyro`, `slice_pymc`,
                `hmc_pymc`, `nuts_pymc`. `slice_np` is a custom
                numpy implementation of slice sampling. `slice_np_vectorized` is
                identical to `slice_np`, but if `num_chains>1`, the chains are
                vectorized for `slice_np_vectorized` whereas they are run sequentially
                for `slice_np`. The samplers ending on `_pyro` are using Pyro, and
                likewise the samplers ending on `_pymc` are using PyMC.
            thin: The thinning factor for the chain, default 1 (no thinning).
            warmup_steps: The initial number of samples to discard.
            num_chains: The number of chains. Should generally be at most
                `num_workers - 1`.
            init_strategy: The initialisation strategy for chains; `proposal` will draw
                init locations from `proposal`, whereas `sir` will use Sequential-
                Importance-Resampling (SIR). SIR initially samples
                `init_strategy_num_candidates` from the `proposal`, evaluates all of
                them under the `potential_fn` and `proposal`, and then resamples the
                initial locations with weights proportional to `exp(potential_fn -
                proposal.log_prob`. `resample` is the same as `sir` but
                uses `exp(potential_fn)` as weights.
            init_strategy_parameters: Dictionary of keyword arguments passed to the
                init strategy, e.g., for `init_strategy=sir` this could be
                `num_candidate_samples`, i.e., the number of candidates to find init
                locations (internal default is `1000`), or `device`.
            init_strategy_num_candidates: Number of candidates to find init
                 locations in `init_strategy=sir` (deprecated, use
                 init_strategy_parameters instead).
            num_workers: number of cpu cores used to parallelize mcmc
            mp_context: Multiprocessing start method, either `"fork"` or `"spawn"`
                (default), used by Pyro and PyMC samplers. `"fork"` can be significantly
                faster than `"spawn"` but is only supported on POSIX-based systems
                (e.g. Linux and macOS, not Windows).
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Deprecated, should not be passed.
        """
        if method == "slice":
            warn(
                "The Pyro-based slice sampler is deprecated, and the method `slice` "
                "has been changed to `slice_np`, i.e., the custom "
                "numpy-based slice sampler.",
                DeprecationWarning,
                stacklevel=2,
            )
            method = "slice_np"

        thin = _process_thin_default(thin)

        super().__init__(
            potential_fn,
            theta_transform=theta_transform,
            device=device,
            x_shape=x_shape,
        )

        self.proposal = proposal
        self.method = method
        self.thin = thin
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
        self.init_strategy = init_strategy
        self.init_strategy_parameters = init_strategy_parameters or {}
        self.num_workers = num_workers
        self.mp_context = mp_context
        self._posterior_sampler = None
        # Hardcode parameter name to reduce clutter kwargs.
        self.param_name = "theta"

        if init_strategy_num_candidates is not None:
            warn(
                "Passing `init_strategy_num_candidates` is deprecated as of sbi "
                "v0.19.0. Instead, use e.g., `init_strategy_parameters "
                f"={'num_candidate_samples': 1000}`",
                stacklevel=2,
            )
            self.init_strategy_parameters["num_candidate_samples"] = (
                init_strategy_num_candidates
            )

        self.potential_ = self._prepare_potential(method)

        self._purpose = (
            "It provides MCMC to .sample() from the posterior and "
            "can evaluate the _unnormalized_ posterior density with .log_prob()."
        )

    @property
    def mcmc_method(self) -> str:
        """Returns MCMC method."""
        return self._mcmc_method

    @mcmc_method.setter
    def mcmc_method(self, method: str) -> None:
        """See `set_mcmc_method`."""
        self.set_mcmc_method(method)

    @property
    def posterior_sampler(self):
        """Returns sampler created by `sample`."""
        return self._posterior_sampler

    def set_mcmc_method(self, method: str) -> "NeuralPosterior":
        """Sets sampling method to for MCMC and returns `NeuralPosterior`.

        Args:
            method: Method to use.

        Returns:
            `NeuralPosterior` for chainable calls.
        """
        self._mcmc_method = method
        return self

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
        warn(
            "`.log_prob()` is deprecated for methods that can only evaluate the "
            "log-probability up to a normalizing constant. Use `.potential()` instead.",
            stacklevel=2,
        )
        warn("The log-probability is unnormalized!", stacklevel=2)

        self.potential_fn.set_x(self._x_else_default_x(x))

        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device), track_gradients=track_gradients
        )

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        method: Optional[str] = None,
        thin: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        num_chains: Optional[int] = None,
        init_strategy: Optional[str] = None,
        init_strategy_parameters: Optional[Dict[str, Any]] = None,
        init_strategy_num_candidates: Optional[int] = None,
        mcmc_parameters: Optional[Dict] = None,
        mcmc_method: Optional[str] = None,
        sample_with: Optional[str] = None,
        num_workers: Optional[int] = None,
        mp_context: Optional[str] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$ with MCMC.

        Check the `__init__()` method for a description of all arguments as well as
        their default values.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            mcmc_parameters: Dictionary that is passed only to support the API of
                `sbi` v0.17.2 or older.
            mcmc_method: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. Please use `method` instead.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.
            show_progress_bars: Whether to show sampling progress monitor.

        Returns:
            Samples from posterior.
        """

        self.potential_fn.set_x(self._x_else_default_x(x))

        # Replace arguments that were not passed with their default.
        method = self.method if method is None else method
        thin = self.thin if thin is None else thin
        warmup_steps = self.warmup_steps if warmup_steps is None else warmup_steps
        num_chains = self.num_chains if num_chains is None else num_chains
        init_strategy = self.init_strategy if init_strategy is None else init_strategy
        num_workers = self.num_workers if num_workers is None else num_workers
        mp_context = self.mp_context if mp_context is None else mp_context
        init_strategy_parameters = (
            self.init_strategy_parameters
            if init_strategy_parameters is None
            else init_strategy_parameters
        )
        if init_strategy_num_candidates is not None:
            warn(
                "Passing `init_strategy_num_candidates` is deprecated as of sbi "
                "v0.19.0. Instead, use e.g., "
                f"`init_strategy_parameters={'num_candidate_samples': 1000}`",
                stacklevel=2,
            )
            self.init_strategy_parameters["num_candidate_samples"] = (
                init_strategy_num_candidates
            )
        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                "`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )
        if mcmc_method is not None:
            warn(
                "You passed `mcmc_method` to `.sample()`. As of sbi v0.18.0, this "
                "is deprecated and will be removed in a future release. Use `method` "
                "instead of `mcmc_method`.",
                stacklevel=2,
            )
            method = mcmc_method
        if mcmc_parameters:
            warn(
                "You passed `mcmc_parameters` to `.sample()`. As of sbi v0.18.0, this "
                "is deprecated and will be removed in a future release. Instead, pass "
                "the variable to `.sample()` directly, e.g. "
                "`posterior.sample((1,), num_chains=5)`.",
                stacklevel=2,
            )
        # The following lines are only for backwards compatibility with sbi v0.17.2 or
        # older.
        m_p = mcmc_parameters or {}  # define to shorten the variable name
        method = _maybe_use_dict_entry(method, "mcmc_method", m_p)
        thin = _maybe_use_dict_entry(thin, "thin", m_p)
        warmup_steps = _maybe_use_dict_entry(warmup_steps, "warmup_steps", m_p)
        num_chains = _maybe_use_dict_entry(num_chains, "num_chains", m_p)
        init_strategy = _maybe_use_dict_entry(init_strategy, "init_strategy", m_p)
        self.potential_ = self._prepare_potential(method)  # type: ignore

        initial_params = self._get_initial_params(
            init_strategy,  # type: ignore
            num_chains,  # type: ignore
            num_workers,
            show_progress_bars,
            **init_strategy_parameters,
        )
        num_samples = torch.Size(sample_shape).numel()

        track_gradients = method in ("hmc_pyro", "nuts_pyro", "hmc_pymc", "nuts_pymc")
        with torch.set_grad_enabled(track_gradients):
            if method in ("slice_np", "slice_np_vectorized"):
                transformed_samples = self._slice_np_mcmc(
                    num_samples=num_samples,
                    potential_function=self.potential_,
                    initial_params=initial_params,
                    thin=thin,  # type: ignore
                    warmup_steps=warmup_steps,  # type: ignore
                    vectorized=(method == "slice_np_vectorized"),
                    interchangeable_chains=True,
                    num_workers=num_workers,
                    show_progress_bars=show_progress_bars,
                )
            elif method in ("hmc_pyro", "nuts_pyro"):
                transformed_samples = self._pyro_mcmc(
                    num_samples=num_samples,
                    potential_function=self.potential_,
                    initial_params=initial_params,
                    mcmc_method=method,  # type: ignore
                    thin=thin,  # type: ignore
                    warmup_steps=warmup_steps,  # type: ignore
                    num_chains=num_chains,
                    show_progress_bars=show_progress_bars,
                    mp_context=mp_context,
                )
            elif method in ("hmc_pymc", "nuts_pymc", "slice_pymc"):
                transformed_samples = self._pymc_mcmc(
                    num_samples=num_samples,
                    potential_function=self.potential_,
                    initial_params=initial_params,
                    mcmc_method=method,  # type: ignore
                    thin=thin,  # type: ignore
                    warmup_steps=warmup_steps,  # type: ignore
                    num_chains=num_chains,
                    show_progress_bars=show_progress_bars,
                    mp_context=mp_context,
                )
            else:
                raise NameError(f"The sampling method {method} is not implemented!")

        samples = self.theta_transform.inv(transformed_samples)
        # NOTE: Currently MCMCPosteriors will require a single dimension for the
        # parameter dimension. With recent ConditionalDensity(Ratio) estimators, we
        # can have multiple dimensions for the parameter dimension.
        samples = samples.reshape((*sample_shape, -1))  # type: ignore

        return samples

    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        method: Optional[str] = None,
        thin: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        num_chains: Optional[int] = None,
        init_strategy: Optional[str] = None,
        init_strategy_parameters: Optional[Dict[str, Any]] = None,
        num_workers: Optional[int] = None,
        mp_context: Optional[str] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Given a batch of observations [x_1, ..., x_B] this function samples from
        posteriors $p(\theta|x_1)$, ... ,$p(\theta|x_B)$, in a batched (i.e. vectorized)
        manner.

        Check the `__init__()` method for a description of all arguments as well as
        their default values.

        Args:
            sample_shape: Desired shape of samples that are drawn from the posterior
                given every observation.
            x: A batch of observations, of shape `(batch_dim, event_shape_x)`.
                `batch_dim` corresponds to the number of observations to be
                drawn.
            method: Method used for MCMC sampling, e.g., "slice_np_vectorized".
            thin: The thinning factor for the chain, default 1 (no thinning).
            warmup_steps: The initial number of samples to discard.
            num_chains: The number of chains used for each `x` passed in the batch.
            init_strategy: The initialisation strategy for chains.
            init_strategy_parameters: Dictionary of keyword arguments passed to
                the init strategy.
            num_workers: number of cpu cores used to parallelize initial
                parameter generation and mcmc sampling.
            mp_context: Multiprocessing start method, either `"fork"` or `"spawn"`
            show_progress_bars: Whether to show sampling progress monitor.

        Returns:
            Samples from the posteriors of shape (*sample_shape, B, *input_shape)
        """

        # Replace arguments that were not passed with their default.
        method = self.method if method is None else method
        thin = self.thin if thin is None else thin
        warmup_steps = self.warmup_steps if warmup_steps is None else warmup_steps
        num_chains = self.num_chains if num_chains is None else num_chains
        init_strategy = self.init_strategy if init_strategy is None else init_strategy
        num_workers = self.num_workers if num_workers is None else num_workers
        mp_context = self.mp_context if mp_context is None else mp_context
        init_strategy_parameters = (
            self.init_strategy_parameters
            if init_strategy_parameters is None
            else init_strategy_parameters
        )

        assert (
            method == "slice_np_vectorized"
        ), "Batched sampling only supported for vectorized samplers!"

        # warn if num_chains is larger than num requested samples
        if num_chains > torch.Size(sample_shape).numel():
            warnings.warn(
                "The passed number of MCMC chains is larger than the number of "
                f"requested samples: {num_chains} > {torch.Size(sample_shape).numel()},"
                f" resetting it to {torch.Size(sample_shape).numel()}.",
                stacklevel=2,
            )
            num_chains = torch.Size(sample_shape).numel()

        # custom shape handling to make sure to match the batch size of x and theta
        # without unnecessary combinations.
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        x = reshape_to_batch_event(x, event_shape=x.shape[1:])

        # For batched sampling, we want `num_chains` for each observation in the batch.
        # Here we repeat the observations ABC -> AAABBBCCC, so that the chains are
        # in the order of the observations.
        x_ = x.repeat_interleave(num_chains, dim=0)

        self.potential_fn.set_x(x_, x_is_iid=False)
        self.potential_ = self._prepare_potential(method)  # type: ignore

        # For each observation in the batch, we have num_chains independent chains.
        num_chains_extended = batch_size * num_chains
        if num_chains_extended > 100:
            warnings.warn(
                "Note that for batched sampling, we use num_chains many chains for each"
                " x in the batch. With the given settings, this results in a large "
                f"number large number of chains ({num_chains_extended}), which can be "
                "slow and memory-intensive for vectorized MCMC. Consider reducing the "
                "number of chains.",
                stacklevel=2,
            )
        init_strategy_parameters["num_return_samples"] = num_chains_extended
        initial_params = self._get_initial_params_batched(
            x,
            init_strategy,  # type: ignore
            num_chains,  # type: ignore
            num_workers,
            show_progress_bars,
            **init_strategy_parameters,
        )
        # We need num_samples from each posterior in the batch
        num_samples = torch.Size(sample_shape).numel() * batch_size

        with torch.set_grad_enabled(False):
            transformed_samples = self._slice_np_mcmc(
                num_samples=num_samples,
                potential_function=self.potential_,
                initial_params=initial_params,
                thin=thin,  # type: ignore
                warmup_steps=warmup_steps,  # type: ignore
                vectorized=(method == "slice_np_vectorized"),
                interchangeable_chains=False,
                num_workers=num_workers,
                show_progress_bars=show_progress_bars,
            )

        # (num_chains_extended, samples_per_chain, *input_shape)
        samples_per_chain: Tensor = self.theta_transform.inv(transformed_samples)  # type: ignore
        dim_theta = samples_per_chain.shape[-1]
        # We need to collect samples for each x from the respective chains.
        # However, using samples.reshape(*sample_shape, batch_size, dim_theta)
        # does not combine the samples in the right order, since this mixes
        # samples that belong to different `x`. The following permute is a
        # workaround to reshape the samples in the right order.
        samples_per_x = samples_per_chain.reshape((
            batch_size,
            # We are flattening the sample shape here using -1 because we might have
            # generated more samples than requested (more chains, or multiple of
            # chains not matching sample_shape)
            -1,
            dim_theta,
        )).permute(1, 0, -1)

        # Shape is now (-1, batch_size, dim_theta)
        # We can now select the number of requested samples
        samples = samples_per_x[: torch.Size(sample_shape).numel()]
        # and reshape into (*sample_shape, batch_size, dim_theta)
        samples = samples.reshape((*sample_shape, batch_size, dim_theta))
        return samples

    def _build_mcmc_init_fn(
        self,
        proposal: Any,
        potential_fn: Callable,
        transform: torch_tf.Transform,
        init_strategy: str,
        **kwargs,
    ) -> Callable:
        """Return function that, when called, creates an initial parameter set for MCMC.

        Args:
            proposal: Proposal distribution.
            potential_fn: Potential function that the candidate samples are weighted
                with.
            init_strategy: Specifies the initialization method. Either of
                [`proposal`|`sir`|`resample`|`latest_sample`].
            kwargs: Passed on to init function. This way, init specific keywords can
                be set through `mcmc_parameters`. Unused arguments will be absorbed by
                the intitialization method.

        Returns: Initialization function.
        """
        if init_strategy == "proposal" or init_strategy == "prior":
            if init_strategy == "prior":
                warn(
                    "You set `init_strategy=prior`. As of sbi v0.18.0, this is "
                    "deprecated and it will be removed in a future release. Use "
                    "`init_strategy=proposal` instead.",
                    stacklevel=2,
                )
            return lambda: proposal_init(proposal, transform=transform, **kwargs)
        elif init_strategy == "sir":
            warn(
                "As of sbi v0.19.0, the behavior of the SIR initialization for MCMC "
                "has changed. If you wish to restore the behavior of sbi v0.18.0, set "
                "`init_strategy='resample'.`",
                stacklevel=2,
            )
            return lambda: sir_init(
                proposal, potential_fn, transform=transform, **kwargs
            )
        elif init_strategy == "resample":
            return lambda: resample_given_potential_fn(
                proposal, potential_fn, transform=transform, **kwargs
            )
        elif init_strategy == "latest_sample":
            latest_sample = IterateParameters(self._mcmc_init_params, **kwargs)
            return latest_sample
        else:
            raise NotImplementedError

    def _get_initial_params(
        self,
        init_strategy: str,
        num_chains: int,
        num_workers: int,
        show_progress_bars: bool,
        **kwargs,
    ) -> Tensor:
        """Return initial parameters for MCMC obtained with given init strategy.

        Parallelizes across CPU cores only for resample and SIR.

        Args:
            init_strategy: Specifies the initialization method. Either of
                [`proposal`|`sir`|`resample`|`latest_sample`].
            num_chains: number of MCMC chains, generates initial params for each
            num_workers: number of CPU cores for parallization
            show_progress_bars: whether to show progress bars for SIR init
            kwargs: Passed on to `_build_mcmc_init_fn`.

        Returns:
            Tensor: initial parameters, one for each chain
        """
        # Build init function
        init_fn = self._build_mcmc_init_fn(
            self.proposal,
            self.potential_fn,
            transform=self.theta_transform,
            init_strategy=init_strategy,  # type: ignore
            **kwargs,
        )

        # Parallelize inits for resampling only.
        if num_workers > 1 and (init_strategy == "resample" or init_strategy == "sir"):

            def seeded_init_fn(seed):
                torch.manual_seed(seed)
                return init_fn()

            seeds = torch.randint(high=2**31, size=(num_chains,))

            # Generate initial params parallelized over num_workers.
            initial_params = list(
                tqdm(
                    Parallel(return_as="generator", n_jobs=num_workers)(
                        delayed(seeded_init_fn)(seed) for seed in seeds
                    ),
                    total=len(seeds),
                    desc=f"""Generating {num_chains} MCMC inits with
                            {num_workers} workers.""",
                    disable=not show_progress_bars,
                )
            )
            initial_params = torch.cat(initial_params)  # type: ignore
        else:
            initial_params = torch.cat(
                [init_fn() for _ in range(num_chains)]  # type: ignore
            )
        return initial_params

    def _get_initial_params_batched(
        self,
        x: torch.Tensor,
        init_strategy: str,
        num_chains_per_x: int,
        num_workers: int,
        show_progress_bars: bool,
        **kwargs,
    ) -> Tensor:
        """Return initial parameters for MCMC for a batch of `x`, obtained with given
           init strategy.

        Parallelizes across CPU cores only for resample and SIR.

        Args:
            x: Batch of observations to create different initial parameters for.
            init_strategy: Specifies the initialization method. Either of
                [`proposal`|`sir`|`resample`|`latest_sample`].
            num_chains_per_x: number of MCMC chains for each x, generates initial params
                for each x
            num_workers: number of CPU cores for parallization
            show_progress_bars: whether to show progress bars for SIR init
            kwargs: Passed on to `_build_mcmc_init_fn`.

        Returns:
            Tensor: initial parameters, one for each chain
        """

        potential_ = deepcopy(self.potential_fn)
        initial_params = []
        init_fn = self._build_mcmc_init_fn(
            self.proposal,
            potential_fn=potential_,
            transform=self.theta_transform,
            init_strategy=init_strategy,  # type: ignore
            **kwargs,
        )
        for xi in x:
            # Build init function
            potential_.set_x(xi)

            # Parallelize inits for resampling or sir.
            if num_workers > 1 and (
                init_strategy == "resample" or init_strategy == "sir"
            ):

                def seeded_init_fn(seed):
                    torch.manual_seed(seed)
                    return init_fn()

                seeds = torch.randint(high=2**31, size=(num_chains_per_x,))

                # Generate initial params parallelized over num_workers.
                initial_params = initial_params + list(
                    tqdm(
                        Parallel(return_as="generator", n_jobs=num_workers)(
                            delayed(seeded_init_fn)(seed) for seed in seeds
                        ),
                        total=len(seeds),
                        desc=f"""Generating {num_chains_per_x} MCMC inits with
                                {num_workers} workers.""",
                        disable=not show_progress_bars,
                    )
                )

            else:
                initial_params = initial_params + [
                    init_fn() for _ in range(num_chains_per_x)
                ]  # type: ignore

        initial_params = torch.cat(initial_params)
        return initial_params

    def _slice_np_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        initial_params: Tensor,
        thin: int,
        warmup_steps: int,
        vectorized: bool = False,
        interchangeable_chains=True,
        num_workers: int = 1,
        init_width: Union[float, ndarray] = 0.01,
        show_progress_bars: bool = True,
    ) -> Tensor:
        """Custom implementation of slice sampling using Numpy.

        Args:
            num_samples: Desired number of samples.
            potential_function: A callable **class**.
            initial_params: Initial parameters for MCMC chain.
            thin: Thinning (subsampling) factor, default 1 (no thinning).
            warmup_steps: Initial number of samples to discard.
            vectorized: Whether to use a vectorized implementation of the
                `SliceSampler`.
            interchangeable_chains: Whether chains are interchangeable, i.e., whether
                we can mix samples between chains.
            num_workers: Number of CPU cores to use.
            init_width: Inital width of brackets.
            show_progress_bars: Whether to show a progressbar during sampling;
                can only be turned off for vectorized sampler.

        Returns:
            Tensor of shape (num_samples, shape_of_single_theta).
        """

        num_chains, dim_samples = initial_params.shape

        if not vectorized:
            SliceSamplerMultiChain = SliceSamplerSerial
        else:
            SliceSamplerMultiChain = SliceSamplerVectorized

        def multi_obs_potential(params):
            # Params are of shape (num_chains * num_obs, event).
            all_potentials = potential_function(params)  # Shape: (num_chains, num_obs)
            return all_potentials.flatten()

        posterior_sampler = SliceSamplerMultiChain(
            init_params=tensor2numpy(initial_params),
            log_prob_fn=multi_obs_potential,
            num_chains=num_chains,
            thin=thin,
            verbose=show_progress_bars,
            num_workers=num_workers,
            init_width=init_width,
        )
        warmup_ = warmup_steps * thin
        num_samples_ = ceil((num_samples * thin) / num_chains)
        # Run mcmc including warmup
        samples = posterior_sampler.run(warmup_ + num_samples_)
        samples = samples[:, warmup_steps:, :]  # discard warmup steps
        samples = torch.from_numpy(samples)  # chains x samples x dim

        # Save posterior sampler.
        self._posterior_sampler = posterior_sampler

        # Save sample as potential next init (if init_strategy == 'latest_sample').
        self._mcmc_init_params = samples[:, -1, :].reshape(num_chains, dim_samples)

        # Update: If chains are interchangeable, return concatenated samples. Otherwise
        # return samples per chain.
        if interchangeable_chains:
            # Collect samples from all chains.
            samples = samples.reshape(-1, dim_samples)[:num_samples]

        return samples.type(torch.float32).to(self._device)

    def _pyro_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        initial_params: Tensor,
        mcmc_method: str = "nuts_pyro",
        thin: int = -1,
        warmup_steps: int = 200,
        num_chains: Optional[int] = 1,
        show_progress_bars: bool = True,
        mp_context: str = "spawn",
    ) -> Tensor:
        r"""Return samples obtained using Pyro's HMC or NUTS sampler.

        Args:
            num_samples: Desired number of samples.
            potential_function: A callable **class**. A class, but not a function,
                is picklable for Pyro MCMC to use it across chains in parallel,
                even when the potential function requires evaluating a neural network.
            initial_params: Initial parameters for MCMC chain.
            mcmc_method: Pyro MCMC method to use, either `"hmc_pyro"` or
                `"nuts_pyro"` (default).
            thin: Thinning (subsampling) factor, default 1 (no thinning).
            warmup_steps: Initial number of samples to discard.
            num_chains: Whether to sample in parallel. If None, use all but one CPU.
            show_progress_bars: Whether to show a progressbar during sampling.

        Returns:
            Tensor of shape (num_samples, shape_of_single_theta).
        """
        thin = _process_thin_default(thin)
        num_chains = mp.cpu_count() - 1 if num_chains is None else num_chains
        kernels = dict(hmc_pyro=HMC, nuts_pyro=NUTS)

        sampler = MCMC(
            kernel=kernels[mcmc_method](potential_fn=potential_function),
            num_samples=ceil((thin * num_samples) / num_chains),
            warmup_steps=warmup_steps,
            initial_params={self.param_name: initial_params},
            num_chains=num_chains,
            mp_context=mp_context,
            disable_progbar=not show_progress_bars,
            transforms={},
        )
        sampler.run()
        samples = next(iter(sampler.get_samples().values())).reshape(
            -1,
            initial_params.shape[1],  # .shape[1] = dim of theta
        )

        # Save posterior sampler.
        self._posterior_sampler = sampler

        samples = samples[::thin][:num_samples]

        return samples.detach()

    def _pymc_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        initial_params: Tensor,
        mcmc_method: str = "nuts_pymc",
        thin: int = -1,
        warmup_steps: int = 200,
        num_chains: Optional[int] = 1,
        show_progress_bars: bool = True,
        mp_context: str = "spawn",
    ) -> Tensor:
        r"""Return samples obtained using PyMC's HMC, NUTS or slice samplers.

        Args:
            num_samples: Desired number of samples.
            potential_function: A callable **class**. A class, but not a function,
                is picklable for PyMC MCMC to use it across chains in parallel,
                even when the potential function requires evaluating a neural network.
            initial_params: Initial parameters for MCMC chain.
            mcmc_method: mcmc_method: Pyro MCMC method to use, either `"hmc_pymc"` or
                `"slice_pymc"`, or `"nuts_pymc"` (default).
            thin: Thinning (subsampling) factor, default 1 (no thinning).
            warmup_steps: Initial number of samples to discard.
            num_chains: Whether to sample in parallel. If None, use all but one CPU.
            show_progress_bars: Whether to show a progressbar during sampling.

        Returns:
            Tensor of shape (num_samples, shape_of_single_theta).
        """
        thin = _process_thin_default(thin)
        num_chains = mp.cpu_count() - 1 if num_chains is None else num_chains
        steps = dict(slice_pymc="slice", hmc_pymc="hmc", nuts_pymc="nuts")

        sampler = PyMCSampler(
            potential_fn=potential_function,
            step=steps[mcmc_method],
            initvals=tensor2numpy(initial_params),
            draws=ceil((thin * num_samples) / num_chains),
            tune=warmup_steps,
            chains=num_chains,
            mp_ctx=mp_context,
            progressbar=show_progress_bars,
            param_name=self.param_name,
            device=self._device,
        )
        samples = sampler.run()
        samples = torch.from_numpy(samples).to(dtype=torch.float32, device=self._device)
        samples = samples.reshape(-1, initial_params.shape[1])

        # Save posterior sampler.
        self._posterior_sampler = sampler

        samples = samples[::thin][:num_samples]

        return samples

    def _prepare_potential(self, method: str) -> Callable:
        """Combines potential and transform and takes care of gradients and pyro.

        Args:
            method: Which MCMC method to use.

        Returns:
            A potential function that is ready to be used in MCMC.
        """
        if method in ("hmc_pyro", "nuts_pyro"):
            track_gradients = True
            pyro = True
        elif method in ("hmc_pymc", "nuts_pymc"):
            track_gradients = True
            pyro = False
        elif method in ("slice_np", "slice_np_vectorized", "slice_pymc"):
            track_gradients = False
            pyro = False
        else:
            if "hmc" in method or "nuts" in method:
                warn(
                    "The kwargs 'hmc' and 'nuts' are deprecated. Use 'hmc_pyro', "
                    "'nuts_pyro', 'hmc_pymc', or 'nuts_pymc' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            raise NotImplementedError(f"MCMC method {method} is not implemented.")

        prepared_potential = partial(
            transformed_potential,
            potential_fn=self.potential_fn,
            theta_transform=self.theta_transform,
            device=self._device,
            track_gradients=track_gradients,
        )
        if pyro:
            prepared_potential = partial(
                pyro_potential_wrapper, potential=prepared_potential
            )

        return prepared_potential

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
            show_progress_bars: Whether to show a progressbar during sampling from
                the posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.
            log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                {'norm_posterior': True} for SNPE.

        Returns:
            The MAP estimate.
        """
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

    def get_arviz_inference_data(self) -> InferenceData:
        """Returns arviz InferenceData object constructed most recent samples.

        Note: the InferenceData is constructed using the posterior samples generated in
        most recent call to `.sample(...)`.

        For Pyro and PyMC samplers, InferenceData will contain diagnostics, but for
        sbi slice samplers, only the samples are added.

        Returns:
            inference_data: Arviz InferenceData object.
        """
        assert (
            self._posterior_sampler is not None
        ), """No samples have been generated, call .sample() first."""

        sampler: Union[
            MCMC, SliceSamplerSerial, SliceSamplerVectorized, PyMCSampler
        ] = self._posterior_sampler

        # If Pyro sampler and samples not transformed, use arviz' from_pyro.
        if isinstance(sampler, (HMC, NUTS)) and isinstance(
            self.theta_transform, torch_tf.IndependentTransform
        ):
            inference_data = az.from_pyro(sampler)
        # If PyMC sampler and samples not transformed, get cached InferenceData.
        elif isinstance(sampler, PyMCSampler) and isinstance(
            self.theta_transform, torch_tf.IndependentTransform
        ):
            inference_data = sampler.get_inference_data()

        # otherwise get samples from sampler and transform to original space.
        else:
            transformed_samples = sampler.get_samples(group_by_chain=True)
            # Pyro samplers returns dicts, get values.
            if isinstance(transformed_samples, Dict):
                # popitem gets last items, [1] get the values as tensor.
                transformed_samples = transformed_samples.popitem()[1]
            # Our slice samplers return numpy arrays.
            elif isinstance(transformed_samples, ndarray):
                transformed_samples = torch.from_numpy(transformed_samples).type(
                    torch.float32
                )
            # For MultipleIndependent priors transforms first dim must be batch dim.
            # thus, reshape back and forth to have batch dim in front.
            samples_shape = transformed_samples.shape
            samples = self.theta_transform.inv(  # type: ignore
                transformed_samples.reshape(-1, samples_shape[-1])
            ).reshape(  # type: ignore
                *samples_shape
            )

            inference_data = az.convert_to_inference_data({
                f"{self.param_name}": samples
            })

        return inference_data


def _process_thin_default(thin: int) -> int:
    """
    Check if the user did use the default thinning value and raise a warning if so.

    Args:
        thin: Thinning (subsampling) factor, setting 1 disables thinning.

    Returns:
        The corrected thinning factor.
    """
    if thin == -1:
        thin = 1
        warn(
            "The default value for thinning in MCMC sampling has been changed from "
            "10 to 1. This might cause the results differ from the last benchmark.",
            UserWarning,
            stacklevel=2,
        )

    return thin


def _maybe_use_dict_entry(default: Any, key: str, dict_to_check: Dict) -> Any:
    """Returns `default` if `key` is not in the dict and otherwise the dict entry.

    This method exists only to keep backwards compatibility with `sbi` v0.17.2 or
    older. It allows passing `mcmc_parameters` to `.sample()`.

    Args:
        default: The default value if `key` is not in `dict_to_check`.
        key: The key for which to check in `dict_to_check`.
        dict_to_check: The dictionary to be checked.

    Returns:
        The potentially replaced value.
    """
    attribute = dict_to_check.get(key, default)
    return attribute
