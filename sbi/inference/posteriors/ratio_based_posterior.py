# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import torch
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import ScalarFloat, atleast_2d, ensure_theta_batched


class RatioBasedPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNRE.<br/><br/>
    SNRE trains a neural network to approximate likelihood ratios, which in turn can be
    used obtain an unnormalized posterior $p(\theta|x) \propto p(x|\theta) \cdot
    p(\theta)$. The `SNRE_Posterior` class wraps the trained network such that one can
    directly evaluate the unnormalized posterior log-probability $p(\theta|x) \propto
    p(x|\theta) \cdot p(\theta)$ and draw samples from the posterior with
    MCMC. Note that, in the case of single-round SNRE_A / AALR, it is possible to
    evaluate the log-probability of the **normalized** posterior, but sampling still
    requires MCMC.<br/><br/>
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

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""Returns the log-probability of $p(x|\theta) \cdot p(\theta).$

        This corresponds to an **unnormalized** posterior log-probability. Only for
        single-round SNRE_A / AALR, the returned log-probability will correspond to the
        **normalized** log-probability.

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

        self._warn_log_prob_snre()

        # Sum log ratios over x batch of iid trials.
        log_ratio = self._log_ratios_over_trials(
            x.to(self._device),
            theta.to(self._device),
            self.net,
            track_gradients=track_gradients,
        )

        return log_ratio.cpu() + self._prior.log_prob(theta)

    def _warn_log_prob_snre(self) -> None:
        if self._method_family == "snre_a":
            if self._num_trained_rounds > 1:
                warn(
                    "The log-probability from AALR / SNRE-A beyond round 1 is only"
                    " correct up to a normalizing constant."
                )
        elif self._method_family == "snre_b":
            warn(
                "The log probability from SNRE_B is only correct up to a normalizing "
                "constant."
            )

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

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1000,
        learning_rate: float = 1e-2,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 500,
        num_to_optimize: int = 100,
        save_best_every: int = 10,
        show_progress_bars: bool = True,
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

        Args:
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            num_iter: Maximum Number of optimization steps that the algorithm takes
                to find the MAP.
            early_stop_at: If `None`, it will optimize for `max_num_iter` iterations.
                If `float`, the optimization will stop as soon as the steps taken by
                the optimizer are smaller than `early_stop_at` times the standard
                deviation of the initial guesses.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a,
                the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `print_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether or not to show a progressbar for sampling from
                the posterior.

        Returns:
            The MAP estimate.
        """
        return super().map(
            x=x,
            num_iter=num_iter,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            num_to_optimize=num_to_optimize,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
        )

    @property
    def _num_trained_rounds(self) -> int:
        return self._trained_rounds

    @_num_trained_rounds.setter
    def _num_trained_rounds(self, trained_rounds: int) -> None:
        """
        Sets the number of trained rounds and updates the purpose.

        When the number of trained rounds is 1 and the algorithm is SNRE_A, then the
        log_prob will be normalized, as specified in the purpose.

        The reason we made this a property is that the purpose gets updated
        automatically whenever the number of rounds is updated.
        """
        self._trained_rounds = trained_rounds

        normalized_or_not = (
            ""
            if (self._method_family == "snre_a" and self._trained_rounds == 1)
            else "_unnormalized_ "
        )
        self._purpose = (
            f"It provides MCMC to .sample() from the posterior and "
            f"can evaluate the {normalized_or_not}posterior density with .log_prob()."
        )

    @staticmethod
    def _log_ratios_over_trials(
        x: Tensor,
        theta: Tensor,
        net: nn.Module,
        track_gradients: bool = False,
    ) -> Tensor:
        r"""Return log ratios summed over iid trials of `x`.

        Note: `x` can be a batch with batch size larger 1. Batches in x are assumed to
        be iid trials, i.e., data generated based on the same paramters / experimental
        conditions.

        Repeats `x` and $\theta$ to cover all their combinations of batch entries.

        Args:
            x: batch of iid data.
            theta: batch of parameters
            net: neural net representing the classifier to approximate the ratio.
            track_gradients: Whether to track gradients.

        Returns:
            log_ratio_trial_sum: log ratio for each parameter, summed over all
                batch entries (iid trials) in `x`.
        """

        theta_repeated, x_repeated = NeuralPosterior._match_theta_and_x_batch_shapes(
            theta=theta, x=atleast_2d(x)
        )
        assert (
            x_repeated.shape[0] == theta_repeated.shape[0]
        ), "x and theta must match in batch shape."
        assert (
            next(net.parameters()).device == x.device and x.device == theta.device
        ), f"device mismatch: net, x, theta: {next(net.parameters()).device}, {x.device}, {theta.device}."

        # Calculate ratios in one batch.
        with torch.set_grad_enabled(track_gradients):
            log_ratio_trial_batch = net([theta_repeated, x_repeated])
            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_ratio_trial_sum = log_ratio_trial_batch.reshape(x.shape[0], -1).sum(0)

        return log_ratio_trial_sum


class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
    Posterior class. When called, it specializes to the potential function appropriate
    to the requested `mcmc_method`.

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
     most current trained posterior neural net.

    Returns:
        Potential function for use by either numpy or pyro sampler
    """

    def __call__(
        self,
        prior,
        classifier: nn.Module,
        x: Tensor,
        mcmc_method: str,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on `mcmc_method`.

        Args:
            prior: Prior distribution that can be evaluated.
            classifier: Binary classifier approximating the likelihood up to a constant.

            x: Conditioning variable for posterior $p(\theta|x)$.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """

        self.classifier = classifier
        self.prior = prior
        self.device = next(classifier.parameters()).device
        self.x = atleast_2d(x).to(self.device)

        if mcmc_method == "slice":
            return partial(self.pyro_potential, track_gradients=False)
        elif mcmc_method in ("hmc", "nuts"):
            return partial(self.pyro_potential, track_gradients=True)
        else:
            return self.np_potential

    def np_potential(self, theta: np.array) -> ScalarFloat:
        """Return potential for Numpy slice sampler."

        For numpy MCMC samplers this is the unnormalized posterior log prob.

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability of theta.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32)
        theta = ensure_theta_batched(theta)

        log_ratio = RatioBasedPosterior._log_ratios_over_trials(
            self.x, theta.to(self.device), self.classifier, track_gradients=False
        )

        # Notice opposite sign to pyro potential.
        return log_ratio.cpu() + self.prior.log_prob(theta)

    def pyro_potential(
        self, theta: Dict[str, Tensor], track_gradients: bool = False
    ) -> Tensor:
        r"""Return potential for Pyro sampler.

        Note: for Pyro this is the negative unnormalized posterior log prob.

        Args:
            theta: Parameters $\theta$. The tensor's shape will be
             (1, shape_of_single_theta) if running a single chain or just
             (shape_of_single_theta) for multiple chains.

        Returns:
            Potential $-(\log r(x_o, \theta) + \log p(\theta))$.
        """

        theta = next(iter(theta.values()))

        # Theta and x should have shape (1, dim).
        theta = ensure_theta_batched(theta)

        log_ratio = RatioBasedPosterior._log_ratios_over_trials(
            self.x,
            theta.to(self.device),
            self.classifier,
            track_gradients=track_gradients,
        )

        return -(log_ratio.cpu() + self.prior.log_prob(theta))
