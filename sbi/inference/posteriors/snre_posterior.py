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

from sbi.inference.posteriors.posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import atleast_2d_float32_tensor


class SnrePosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods.<br/><br/>
    All inference methods in sbi train a neural network which is then used to obtain
    the posterior distribution. The `NeuralPosterior` class wraps the trained network
    such that one can directly evaluate the log probability and draw samples from the
    posterior. The neural network itself can be accessed via the `.net` attribute.
    <br/><br/>
    Specifically, this class offers the following functionality:<br/>
    - Correction of leakage (applicable only to SNPE): If the prior is bounded, the
      posterior resulting from SNPE can generate samples that lie outside of the prior
      support (i.e. the posterior leaks). This class rejects these samples or,
      alternatively, allows to sample from the posterior with MCMC. It also corrects the
      calculation of the log probability such that it compensates for the leakage.<br/>
    - Posterior inference from likelihood (SNL) and likelihood ratio (SRE): SNL and SRE
      learn to approximate the likelihood and likelihood ratio, which in turn can be
      used to generate samples from the posterior. This class provides the needed MCMC
      methods to sample from the posterior and to evaluate the log probability.

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
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`, `hmc`, `nuts`.
                Currently defaults to `slice_np` for a custom numpy implementation of
                slice sampling; select `hmc`, `nuts` or `slice` for Pyro-based sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains, `init_strategy`
                for the initialisation strategy for chains; `prior` will draw init
                locations from prior, whereas `sir` will use Sequential-Importance-
                Resampling using `init_strategy_num_candidates` to find init
                locations.
            get_potential_function: Callable that returns the potential function used
                for MCMC sampling.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def _log_prob_ratio_estimator(self, theta: Tensor, x: Tensor) -> Tensor:
        log_ratio = self.net(torch.cat((theta, x), dim=1)).reshape(-1)
        return log_ratio + self._prior.log_prob(theta)

    def _log_prob_snre_a(self, theta: Tensor, x: Tensor) -> Tensor:
        if self._num_trained_rounds > 1:
            warn(
                "The log-probability from AALR / SNRE-A beyond round 1 is only correct "
                "up to a normalizing constant."
            )
        return self._log_prob_ratio_estimator(theta, x)

    def _log_prob_snre_b(self, theta: Tensor, x: Tensor) -> Tensor:
        warn(
            "The log probability from SNRE_B is only correct up to a normalizing "
            "constant."
        )
        return self._log_prob_ratio_estimator(theta, x)

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
        Return samples from posterior distribution $p(\theta|x)$.

        Samples are obtained either with rejection sampling or MCMC. SNPE can use
        rejection sampling and MCMC (which can help to deal with strong leakage). SNL
        and SRE are restricted to sampling with MCMC.

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
                samples to discard, `num_chains` for the number of chains, `init_strategy`
                for the initialisation strategy for chains; `prior` will draw init
                locations from prior, whereas `sir` will use Sequential-Importance-
                Resampling using `init_strategy_num_candidates` to find init
                locations.

        Returns:
            Samples from posterior.
        """

        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)
        num_samples = torch.Size(sample_shape).numel()

        mcmc_method = mcmc_method if mcmc_method is not None else self.mcmc_method
        mcmc_parameters = (
            mcmc_parameters if mcmc_parameters is not None else self.mcmc_parameters
        )

        samples = self._sample_posterior_mcmc(
            x=x,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            mcmc_method=mcmc_method,
            **mcmc_parameters,
        )

        return samples.reshape((*sample_shape, -1))
