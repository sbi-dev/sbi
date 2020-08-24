# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from torch import Tensor, log
from torch import multiprocessing as mp
from torch import nn

from sbi import utils as utils
from sbi.mcmc import Slice, SliceSampler
from sbi.types import Array, Shape
from sbi.user_input.user_input_checks import process_x
from sbi.utils.torchutils import atleast_2d_float32_tensor, ensure_theta_batched


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
        get_potential_function: Optional[Callable] = None,
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
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
            get_potential_function: Callable that returns the potential function used
                for MCMC sampling.
        """
        if method_family in ("snpe", "snle", "snre_a", "snre_b"):
            self._method_family = method_family
        else:
            raise ValueError("Method family unsupported.")

        self.net = neural_net

        self.set_mcmc_method(mcmc_method)
        self.set_mcmc_parameters(mcmc_parameters)

        self._get_potential_function = get_potential_function
        self._leakage_density_correction_factor = None  # Correction factor for SNPE.
        self._mcmc_init_params = None
        self._num_trained_rounds = 0
        self._prior = prior
        self._x = None
        self._x_shape = x_shape

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
            often as there were batch elements in $\theta$ orginally.
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
        return x, num_samples, mcmc_method, mcmc_parameters

    def _sample_posterior_mcmc(
        self,
        num_samples: int,
        x: Tensor,
        mcmc_method: str = "slice_np",
        thin: int = 10,
        warmup_steps: int = 20,
        num_chains: Optional[int] = 1,
        init_strategy: str = "prior",
        init_strategy_num_candidates: int = 10000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""
        Return MCMC samples from posterior $p(\theta|x)$.

        This function is used in any case by SNLE and SNRE, but can also be used by SNPE
        in order to deal with strong leakage. Depending on the inference method, a
        different potential function for the MCMC sampler is required.

        Args:
            num_samples: Desired number of samples.
            x: Conditioning context for posterior $p(\theta|x)$.
            mcmc_method: Sampling method. Currently defaults to `slice_np` for a custom
                numpy implementation of slice sampling; select `hmc`, `nuts` or `slice`
                for Pyro-based sampling.
            thin: Thinning factor for the chain, e.g. for `thin=3` only every third
                sample will be returned, until a total of `num_samples`.
            warmup_steps: Initial number of samples to discard.
            num_chains: Whether to sample in parallel. If None, use all but one CPU.
            init_strategy: Initialisation strategy for chains; `prior` will draw init
                locations from prior, whereas `sir` will use Sequential-Importance-
                Resampling using `init_strategy_num_candidates` to find init
                locations.
            init_strategy_num_candidates: Number of candidate init locations
                when `init_strategy` is `sir`.
            show_progress_bars: Whether to show a progressbar during sampling.

        Returns:
            Tensor of shape (num_samples, shape_of_single_theta).
        """
        # Find init points depending on `init_strategy` if no init is set
        if self._mcmc_init_params is None:
            if init_strategy == "prior":
                self._mcmc_init_params = self._prior.sample((num_chains,)).detach()
            elif init_strategy == "sir":
                self.net.eval()
                init_param_candidates = self._prior.sample(
                    (init_strategy_num_candidates,)
                ).detach()
                potential_function = self._get_potential_function(
                    self._prior, self.net, x, "slice_np"
                )
                log_weights = torch.cat(
                    [
                        potential_function(init_param_candidates[i, :]).detach()
                        for i in range(init_strategy_num_candidates)
                    ]
                )
                probs = np.exp(log_weights.view(-1).numpy().astype(np.float64))
                probs[np.isnan(probs)] = 0.0
                probs[np.isinf(probs)] = 0.0
                probs /= probs.sum()
                idxs = np.random.choice(
                    a=np.arange(init_strategy_num_candidates),
                    size=num_chains,
                    replace=False,
                    p=probs,
                )
                self._mcmc_init_params = init_param_candidates[
                    torch.from_numpy(idxs.astype(int)), :
                ]
                self.net.train(True)
            else:
                raise NotImplementedError

        potential_fn = self._get_potential_function(
            self._prior, self.net, x, mcmc_method
        )
        track_gradients = mcmc_method != "slice" and mcmc_method != "slice_np"
        with torch.set_grad_enabled(track_gradients):
            if mcmc_method == "slice_np":
                samples = self._slice_np_mcmc(
                    num_samples, potential_fn, thin, warmup_steps,
                )
            elif mcmc_method in ("hmc", "nuts", "slice"):
                samples = self._pyro_mcmc(
                    num_samples=num_samples,
                    potential_function=potential_fn,
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
        thin: int,
        warmup_steps: int,
    ) -> Tensor:
        """
        Custom implementation of slice sampling using Numpy.

        Args:
            num_samples: Desired number of samples.
            potential_function: A callable **class**.
            thin: Thinning (subsampling) factor.
            warmup_steps: Initial number of samples to discard.

        Returns: Tensor of shape (num_samples, shape_of_single_theta).
        """
        # Go into eval mode for evaluating during sampling
        self.net.eval()

        num_chains = self._mcmc_init_params.shape[0]
        dim_samples = self._mcmc_init_params.shape[1]

        all_samples = []
        for c in range(num_chains):
            posterior_sampler = SliceSampler(
                utils.tensor2numpy(self._mcmc_init_params[c, :]).reshape(-1),
                lp_f=potential_function,
                thin=thin,
            )
            if warmup_steps > 0:
                posterior_sampler.gen(int(warmup_steps))
            all_samples.append(posterior_sampler.gen(int(num_samples / num_chains)))
        all_samples = np.stack(all_samples).astype(np.float32)

        samples = torch.from_numpy(all_samples)  # chains x samples x dim

        # Final sample will be next init location
        self._mcmc_init_params = samples[:, -1, :].reshape(num_chains, dim_samples)

        samples = samples.reshape(-1, dim_samples)[:num_samples, :]
        assert samples.shape[0] == num_samples

        # Back to training mode
        self.net.train(True)

        return samples.type(torch.float32)

    def _pyro_mcmc(
        self,
        num_samples: int,
        potential_function: Callable,
        mcmc_method: str = "slice",
        thin: int = 10,
        warmup_steps: int = 200,
        num_chains: Optional[int] = 1,
        show_progress_bars: bool = True,
    ):
        r"""Return samples obtained using Pyro HMC, NUTS or slice kernels.

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

        # Go into eval mode for evaluating during sampling
        self.net.eval()

        kernels = dict(slice=Slice, hmc=HMC, nuts=NUTS)

        sampler = MCMC(
            kernel=kernels[mcmc_method](potential_fn=potential_function),
            num_samples=(thin * num_samples) // num_chains + num_chains,
            warmup_steps=warmup_steps,
            initial_params={"": self._mcmc_init_params},
            num_chains=num_chains,
            mp_context="fork",
            disable_progbar=not show_progress_bars,
        )
        sampler.run()
        samples = next(iter(sampler.get_samples().values())).reshape(
            -1, len(self._prior.mean)  # len(prior.mean) = dim of theta
        )

        samples = samples[::thin][:num_samples]
        assert samples.shape[0] == num_samples

        # Back to training mode
        self.net.train(True)

        return samples

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
