# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

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
from warnings import warn

import torch
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import atleast_2d_float32_tensor


class LikelihoodBasedPosterior(NeuralPosterior):
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

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

        self._purpose = (
            f"It provides MCMC to .sample() from the posterior and "
            f"can evaluate the _unnormalized_ posterior density with .log_prob()."
        )

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,
    ) -> Tensor:
        r"""
        Returns the log-probability of $p(x|\theta) \cdot p(\theta).$

        This corresponds to an **unnormalized** posterior log-probability.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided, fall
                back onto an `x_o` if previously provided for multi-round training, or
                to another default if set later for convenience, see `.set_default_x()`.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `(len(θ),)`-shaped log-probability $\log(p(x|\theta) \cdot p(\theta))$.

        """

        # TODO Train exited here, entered after sampling?
        self.net.eval()

        theta, x = self._prepare_theta_and_x_for_log_prob_(theta, x)

        warn(
            "The log probability from SNL is only correct up to a normalizing constant."
        )

        with torch.set_grad_enabled(track_gradients):
            return self.net.log_prob(x, theta) + self._prior.log_prob(theta)

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        sample_with_mcmc: Optional[bool] = None,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""
        Return samples from posterior distribution $p(\theta|x)$ with MCMC.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x_o` if previously provided for multiround training, or
                to a set default (see `set_default_x()` method).
            show_progress_bars: Whether to show sampling progress monitor.
            sample_with_mcmc: Optional parameter to override `self.sample_with_mcmc`.
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
            Samples from posterior.
        """

        x, num_samples, mcmc_method, mcmc_parameters = self._prepare_for_sample(
            x, sample_shape, mcmc_method, mcmc_parameters
        )

        samples = self._sample_posterior_mcmc(
            x=x,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            mcmc_method=mcmc_method,
            **mcmc_parameters,
        )

        return samples.reshape((*sample_shape, -1))
