# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from functools import partial
from typing import Any, Callable, Dict, Optional, Union
from warnings import warn

import torch
import torch.distributions.transforms as torch_tf
from joblib import Parallel, delayed
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import Tensor
from torch import multiprocessing as mp
from tqdm.auto import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.samplers.mcmc import (
    IterateParameters,
    Slice,
    proposal_init,
    sir,
    slice_np_parallized,
)
from sbi.simulators.simutils import tqdm_joblib
from sbi.types import Shape, TorchTransform
from sbi.utils import pyro_potential_wrapper, transformed_potential
from sbi.utils.torchutils import ensure_theta_batched


class MCMCPosterior(NeuralPosterior):
    r"""Provides MCMC to sample from the posterior.<br/><br/>
    SNLE or SNRE train neural networks to approximate the likelihood(-ratios).
    `MCMCPosterior` allows to sample from the posterior with MCMC.
    """

    def __init__(
        self,
        potential_fn: Callable,
        proposal: Any,
        theta_transform: Optional[TorchTransform] = None,
        method: str = "slice_np",
        thin: int = 10,
        warmup_steps: int = 10,
        num_chains: int = 1,
        init_strategy: str = "sir",
        init_strategy_num_candidates: int = 1_000,
        num_workers: int = 1,
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples.
            proposal: Proposal distribution that is used to initialize the MCMC chain.
            theta_transform: Transformation that will be applied during sampling.
                Allows to perform MCMC in unconstrained space.
            method: Method used for MCMC sampling, one of `slice_np`,
                `slice_np_vectorized`, `slice`, `hmc`, `nuts`. `slice_np` is a custom
                numpy implementation of slice sampling. `slice_np_vectorized` is
                identical to `slice_np`, but if `num_chains>1`, the chains are
                vectorized for `slice_np_vectorized` whereas they are run sequentially
                for `slice_np`. The samplers `hmc`, `nuts` or `slice` sample with Pyro.
            thin: The thinning factor for the chain.
            warmup_steps: The initial number of samples to discard.
            num_chains: The number of chains.
            init_strategy: The initialisation strategy for chains; `proposal` will draw
                init locations from `proposal`, whereas `sir` will use Sequential-
                Importance-Resampling (SIR). SIR initially samples
                `init_strategy_num_candidates` from the `proposal`, evaluates all of
                them under the `potential_fn`, and then resamples the initial locations
                with weights proportional to the `potential_fn`-value.
            init_strategy_num_candidates: Number of candidates to to find init
                locations in `init_strategy=sir`.
            num_workers: number of cpu cores used to parallelize mcmc
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Shape of a single simulator output. If passed, it is used to check
                the shape of the observed data and give a descriptive error.
        """

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
        self.init_strategy_num_candidates = init_strategy_num_candidates
        self.num_workers = num_workers

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
            """`.log_prob()` is deprecated for methods that can only evaluate the
            log-probability up to a normalizing constant. Use `.potential()` instead."""
        )
        warn("The log-probability is unnormalized!")

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
        init_strategy_num_candidates: Optional[int] = None,
        mcmc_parameters: Dict = {},
        mcmc_method: Optional[str] = None,
        sample_with: Optional[str] = None,
        num_workers: Optional[int] = None,
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
        init_strategy_num_candidates = (
            self.init_strategy_num_candidates
            if init_strategy_num_candidates is None
            else init_strategy_num_candidates
        )
        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )
        if mcmc_method is not None:
            warn(
                "You passed `mcmc_method` to `.sample()`. As of sbi v0.18.0, this "
                "is deprecated and will be removed in a future release. Use `method` "
                "instead of `mcmc_method`."
            )
            method = mcmc_method
        if mcmc_parameters:
            warn(
                "You passed `mcmc_parameters` to `.sample()`. As of sbi v0.18.0, this "
                "is deprecated and will be removed in a future release. Instead, pass "
                "the variable to `.sample()` directly, e.g. "
                "`posterior.sample((1,), num_chains=5)`."
            )
        # The following lines are only for backwards compatibility with sbi v0.17.2 or
        # older.
        m_p = mcmc_parameters  # define to shorten the variable name
        method = _maybe_use_dict_entry(method, "mcmc_method", m_p)
        thin = _maybe_use_dict_entry(thin, "thin", m_p)
        warmup_steps = _maybe_use_dict_entry(warmup_steps, "warmup_steps", m_p)
        num_chains = _maybe_use_dict_entry(num_chains, "num_chains", m_p)
        init_strategy = _maybe_use_dict_entry(init_strategy, "init_strategy", m_p)
        init_strategy_num_candidates = _maybe_use_dict_entry(
            init_strategy_num_candidates, "init_strategy_num_candidates", m_p
        )
        self.potential_ = self._prepare_potential(method)  # type: ignore

        initial_params = self._get_initial_params(
            init_strategy, num_chains, num_workers, show_progress_bars  # type: ignore
        )
        num_samples = torch.Size(sample_shape).numel()

        track_gradients = method in ("hmc", "nuts")
        with torch.set_grad_enabled(track_gradients):
            if method in ("slice_np", "slice_np_vectorized"):
                transformed_samples = self._slice_np_mcmc(
                    num_samples=num_samples,
                    potential_function=self.potential_,
                    initial_params=initial_params,
                    thin=thin,  # type: ignore
                    warmup_steps=warmup_steps,  # type: ignore
                    vectorized=(method == "slice_np_vectorized"),
                    num_workers=num_workers,
                    show_progress_bars=show_progress_bars,
                )
            elif method in ("hmc", "nuts", "slice"):
                transformed_samples = self._pyro_mcmc(
                    num_samples=num_samples,
                    potential_function=self.potential_,
                    initial_params=initial_params,
                    mcmc_method=method,  # type: ignore
                    thin=thin,  # type: ignore
                    warmup_steps=warmup_steps,  # type: ignore
                    num_chains=num_chains,
                    show_progress_bars=show_progress_bars,
                ).detach()
            else:
                raise NameError

        samples = self.theta_transform.inv(transformed_samples)
        return samples.reshape((*sample_shape, -1))  # type: ignore

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
                [`proposal`|`sir`|`latest_sample`].
            kwargs: Passed on to init function. This way, init specific keywords can
                be set through `mcmc_parameters`. Unused arguments should be absorbed.

        Returns: Initialization function.
        """
        if init_strategy == "proposal" or init_strategy == "prior":
            if init_strategy == "prior":
                warn(
                    "You set `init_strategy=prior`. As of sbi v0.18.0, this is "
                    "deprecated and it will be removed in a future release. Use "
                    "`init_strategy=proposal` instead."
                )
            return lambda: proposal_init(proposal, transform=transform, **kwargs)
        elif init_strategy == "sir":
            return lambda: sir(proposal, potential_fn, transform=transform, **kwargs)
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
    ) -> Tensor:
        """Return initial parameters for MCMC obtained with given init strategy.

        Parallelizes across CPU cores only for SIR.

        Args:
            init_strategy: Specifies the initialization method. Either of
                [`proposal`|`sir`|`latest_sample`].
            num_chains: number of MCMC chains, generates initial params for each
            num_workers: number of CPU cores for parallization
            show_progress_bars: whether to show progress bars for SIR init

        Returns:
            Tensor: initial parameters, one for each chain
        """
        # Build init function
        init_fn = self._build_mcmc_init_fn(
            self.proposal,
            self.potential_fn,
            transform=self.theta_transform,
            init_strategy=init_strategy,  # type: ignore
        )

        # Parallelize inits for SIR only.
        if num_workers > 1 and init_strategy == "sir":

            def seeded_init_fn(seed):
                torch.manual_seed(seed)
                return init_fn()

            seeds = torch.randint(high=2**31, size=(num_chains,))

            # Generate initial params parallelized over num_workers.
            with tqdm_joblib(
                tqdm(
                    range(num_chains),  # type: ignore
                    disable=not show_progress_bars,
                    desc=f"""Generating {num_chains} MCMC inits with {num_workers}
                         workers.""",
                    total=num_chains,
                )
            ):
                initial_params = torch.cat(
                    Parallel(n_jobs=num_workers)(
                        delayed(seeded_init_fn)(seed) for seed in seeds
                    )
                )
        else:
            initial_params = torch.cat(
                [init_fn() for _ in range(num_chains)]  # type: ignore
            )

        return initial_params

    def _slice_np_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        initial_params: Tensor,
        thin: int,
        warmup_steps: int,
        vectorized: bool = False,
        num_workers: int = 1,
        show_progress_bars: bool = True,
    ) -> Tensor:
        """Custom implementation of slice sampling using Numpy.

        Args:
            num_samples: Desired number of samples.
            potential_function: A callable **class**.
            initial_params: Initial parameters for MCMC chain.
            thin: Thinning (subsampling) factor.
            warmup_steps: Initial number of samples to discard.
            vectorized: Whether to use a vectorized implementation of
                the Slice sampler (still experimental).
            num_workers: number of CPU cores to use
            seed: seed that will be used to generate sub-seeds for each worker
            show_progress_bars: Whether to show a progressbar during sampling;
                can only be turned off for vectorized sampler.

        Returns: Tensor of shape (num_samples, shape_of_single_theta).
        """

        num_chains, dim_samples = initial_params.shape

        samples = slice_np_parallized(
            potential_function,
            initial_params,
            num_samples,
            thin=thin,
            warmup_steps=warmup_steps,
            vectorized=vectorized,
            num_workers=num_workers,
            show_progress_bars=show_progress_bars,
        )

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
        num_chains = mp.cpu_count() - 1 if num_chains is None else num_chains

        kernels = dict(slice=Slice, hmc=HMC, nuts=NUTS)

        sampler = MCMC(
            kernel=kernels[mcmc_method](potential_fn=potential_function),
            num_samples=(thin * num_samples) // num_chains + num_chains,
            warmup_steps=warmup_steps,
            initial_params={"": initial_params},
            num_chains=num_chains,
            mp_context="spawn",
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

    def _prepare_potential(self, method: str) -> Callable:
        """Combines potential and transform and takes care of gradients and pyro.

        Args:
            method: Which MCMC method to use.

        Returns:
            A potential function that is ready to be used in MCMC.
        """
        if method == "slice":
            track_gradients = False
            pyro = True
        elif method in ("hmc", "nuts"):
            track_gradients = True
            pyro = True
        elif "slice_np" in method:
            track_gradients = False
            pyro = False
        else:
            raise NotImplementedError

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
            show_progress_bars: Whether or not to show a progressbar for sampling from
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
    attribute = default if key not in dict_to_check.keys() else dict_to_check[key]
    return attribute
