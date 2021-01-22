# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, List, Optional
from warnings import warn

import numpy as np
import torch
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import ScalarFloat, ensure_theta_batched, ensure_x_batched


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
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.
            device: Training device, e.g., cpu or cuda:0.
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
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
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
            # Evaluate on device, move to cpu for comparison with prior.
            return self.net.log_prob(
                x.to(self._device), theta.to(self._device)
            ).cpu() + self._prior.log_prob(theta)

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
                fall back onto `x` passed to `set_default_x()`.
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

        self.net.eval()

        potential_fn_provider = PotentialFunctionProvider()
        samples = self._sample_posterior_mcmc(
            num_samples=num_samples,
            potential_fn=potential_fn_provider(self._prior, self.net, x, mcmc_method),
            init_fn=self._build_mcmc_init_fn(
                self._prior,
                potential_fn_provider(self._prior, self.net, x, "slice_np"),
                **mcmc_parameters,
            ),
            mcmc_method=mcmc_method,
            show_progress_bars=show_progress_bars,
            **mcmc_parameters,
        )

        self.net.train(True)

        return samples.reshape((*sample_shape, -1))

    def sample_conditional(
        self,
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
                fall back onto `x` passed to `set_default_x()`.
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

        return super().sample_conditional(
            PotentialFunctionProvider(),
            sample_shape,
            condition,
            dims_to_sample,
            x,
            show_progress_bars,
            mcmc_method,
            mcmc_parameters,
        )


class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
     Posterior class. When called, it specializes to the potential function appropriate
     to the requested mcmc_method.

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
    most current trained posterior neural net.

    Returns:
        Potential function for use by either numpy or pyro sampler.
    """

    def __call__(
        self, prior, likelihood_nn: nn.Module, x: Tensor, mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on mcmc_method.

        Args:
            prior: Prior distribution that can be evaluated.
            likelihood_nn: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """
        self.likelihood_nn = likelihood_nn
        self.prior = prior
        self.x = x

        if mcmc_method in ("slice", "hmc", "nuts"):
            return self.pyro_potential
        else:
            return self.np_potential

    def np_potential(self, theta: np.array) -> ScalarFloat:
        r"""Return posterior log prob. of theta $p(\theta|x)$"

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability of the theta, $-\infty$ if impossible under prior.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        theta = ensure_theta_batched(theta)
        num_batch = theta.shape[0]
        x = ensure_x_batched(self.x).repeat(num_batch, 1)

        with torch.set_grad_enabled(False):
            # Evaluate on device, move back to cpu for comparison with prior.
            log_likelihood = self.likelihood_nn.log_prob(
                inputs=x, context=theta.to(x.device)
            ).cpu()

        # Notice opposite sign to pyro potential.
        return log_likelihood + self.prior.log_prob(theta)

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return posterior log probability of parameters $p(\theta|x)$.

         Args:
            theta: Parameters $\theta$. The tensor's shape will be
                (1, shape_of_single_theta) if running a single chain or just
                (shape_of_single_theta) for multiple chains.

        Returns:
            The potential $-[\log r(x_o, \theta) + \log p(\theta)]$.
        """

        theta = next(iter(theta.values()))

        # Evaluate on device, move back to cpu for comparison with prior.
        log_likelihood = self.likelihood_nn.log_prob(
            inputs=self.x.reshape(1, -1), context=theta.reshape(1, -1).to(self.x.device)
        ).cpu()

        return -(log_likelihood + self.prior.log_prob(theta))
