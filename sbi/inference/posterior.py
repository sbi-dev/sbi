# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Callable, Optional
from warnings import warn

import numpy as np
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
import torch
from torch import Tensor, log, nn
from torch import multiprocessing as mp
from sbi.types import Shape

from sbi.mcmc import Slice, SliceSampler
from sbi.types import Array
import sbi.utils as utils
from sbi.utils.torchutils import (
    ensure_theta_batched,
    atleast_2d_float32_tensor,
    batched_first_of_batch,
)
from sbi.user_input.user_input_checks import process_x


class NeuralPosterior:
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
        sample_with_mcmc: bool = True,
        mcmc_method: str = "slice_np",
        get_potential_function: Optional[Callable] = None,
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            sample_with_mcmc: Whether to sample with MCMC. Will always be `True` for SRE
                and SNL, but can also be set to `True` for SNPE if MCMC is preferred to
                deal with leakage over rejection sampling.
            mcmc_method: If MCMC sampling is used, specify the method here: either of
                slice_np, slice, hmc, nuts.
            get_potential_function: Callable that returns the potential function used
                for MCMC sampling.
        """

        self.net = neural_net
        self._prior = prior
        # This can be changed via `.set_default_x() below.`
        # TODO: set via default_x directly here? would require process_x to accept None.
        self._x = None
        self._x_o_training_focused_on = None

        self._sample_with_mcmc = sample_with_mcmc
        self._mcmc_method = mcmc_method
        self._mcmc_init_params = None
        self._get_potential_function = get_potential_function
        self._x_shape = x_shape

        if method_family in ("snpe", "snle", "snre_a", "snre_b"):
            self._method_family = method_family
        else:
            raise ValueError("Method family unsupported.")

        self._num_trained_rounds = 0

        # Correction factor for leakage, only applicable to SNPE-family methods.
        self._leakage_density_correction_factor = None

    @property
    def default_x(self) -> Optional[Tensor]:
        """Return default x used by `.sample(), .log_prob` as conditioning context."""
        return self._x

    @default_x.setter
    def default_x(self, x: Tensor) -> None:
        """Set new default x for `.sample(), .log_prob` to use as conditioning context.

        See documentation of `.set_default_x()` for rationale and semantics."""
        processed_x = process_x(x, self._x_shape)
        self._warn_if_posterior_was_focused_on_different_x(processed_x)
        self._x = processed_x

    # When a type is not yet defined, one uses a string representation.
    def set_default_x(self, x: Tensor) -> "NeuralPosterior":
        """
        Return `NeuralPosterior` object with default conditioning context set to `x`.

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
        self._warn_if_posterior_was_focused_on_different_x(processed_x)
        self._x = processed_x

        return self

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior_snpe: bool = True,
        track_gradients: bool = False,
    ) -> Tensor:
        r"""Return posterior log probability  $\log p(\theta|x)$.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided, fall
                back onto an `x_o` if previously provided for multi-round training, or
                to another default if set later for convenience, see `.set_default_x()`.
            norm_posterior_snpe: Whether to enforce a normalized posterior density when
                using SNPE. Renormalization of the posterior is useful when some
                probability falls out or leaks out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set here
                `norm_posterior_snpe=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """

        # TODO Train exited here, entered after sampling?
        self.net.eval()

        theta = ensure_theta_batched(torch.as_tensor(theta))

        # Select and check x to condition on.
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)
        self._warn_if_posterior_was_focused_on_different_x(x)

        # Repeat `x` in case of evaluation on multiple `theta`. This is needed below in
        # when calling nflows in order to have matching shapes of theta and context x
        # at neural network evaluation time.
        x = self._match_x_with_theta_batch_shape(x, theta)

        try:
            log_prob_fn = getattr(self, f"_log_prob_{self._method_family}")
        except AttributeError:
            raise ValueError(f"{self._method_family} cannot evaluate probabilities.")

        with torch.set_grad_enabled(track_gradients):
            if self._method_family == "snpe":
                return log_prob_fn(theta, x, norm_posterior=norm_posterior_snpe)
            else:
                return log_prob_fn(theta, x)

    # TODO: Move _log_prob_X into the respective inference classes (X)?
    # The problem is extensibility. Any third party contributing a method X
    # will need to also add a `_log_prob_X` here, and that is not nice.
    # PLAN: pass an instance of the inference object at Posterior creation,

    def _log_prob_snpe(self, theta: Tensor, x: Tensor, norm_posterior: bool) -> Tensor:
        r"""
        Return posterior log probability $p(\theta|x)$.

        The posterior probability will be only normalized if explicitly requested,
        but it will be always zeroed out (i.e. given -∞ log-prob) outside the prior
        support.
        """

        unnorm_log_prob = self.net.log_prob(theta, x)

        # Force probability to be zero outside prior support.
        is_prior_finite = torch.isfinite(self._prior.log_prob(theta))

        masked_log_prob = torch.where(
            is_prior_finite,
            unnorm_log_prob,
            torch.tensor(float("-inf"), dtype=torch.float32),
        )

        log_factor = (
            log(self.leakage_correction(x=batched_first_of_batch(x)))
            if norm_posterior
            else 0
        )

        return masked_log_prob - log_factor

    def _log_prob_ratio_estimator(self, theta: Tensor, x: Tensor) -> Tensor:
        log_ratio = self.net(torch.cat((theta, x)).reshape(1, -1))
        return log_ratio + self._prior.log_prob(theta)

    def _log_prob_snre_a(self, theta: Tensor, x: Tensor) -> Tensor:
        warn(
            "The log probability from SRE is only correct up to a normalizing constant."
        )
        return self._log_prob_ratio_estimator(theta, x)

    def _log_prob_snre_b(self, theta: Tensor, x: Tensor) -> Tensor:
        if self._num_trained_rounds > 1:
            warn(
                "The log-probability from AALR beyond round 1 is only correct "
                "up to a normalizing constant."
            )
        return self._log_prob_ratio_estimator(theta, x)

    def _log_prob_snle(self, theta: Tensor, x: Tensor) -> Tensor:
        warn(
            "The log probability from SNL is only correct up to a normalizing constant."
        )
        return self.net.log_prob(x, theta) + self._prior.log_prob(theta)

    @torch.no_grad()
    def leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progress_bars: bool = False,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density estimate.

        The factor is estimated from the acceptance probability during rejection
        sampling from the posterior.

        NOTE: This is to avoid re-estimating the acceptance probability from scratch
              whenever `log_prob` is called and `norm_posterior_snpe=True`. Here, it
              is estimated only once for `self.default_x` and saved for later. We
              re-evaluate only whenever a new `x` is passed.

        Arguments:
            x: Conditioning context for posterior $p(\theta|x)$.
            num_rejection_samples: Number of samples used to estimate correction factor.
            force_update: Whether to force a reevaluation of the leakage correction even
                if the context `x` is the same as `self.default_x`. This is useful to
                enforce a new leakage estimate for rounds after the first (2, 3,..).
            show_progress_bars: Whether to show a progress bar during sampling.

        Returns:
            Saved or newly-estimated correction factor (as a scalar `Tensor`).
        """

        def acceptance_at(x: Tensor) -> Tensor:
            return utils.sample_posterior_within_prior(
                self.net, self._prior, x, num_rejection_samples, show_progress_bars
            )[1]

        # Check if the provided x matches the default x (short-circuit on identity).
        is_new_x = self.default_x is None or (
            x is not self.default_x and (x != self.default_x).any()
        )

        not_saved_at_default_x = self._leakage_density_correction_factor is None

        if is_new_x:  # Calculate at x; don't save.
            return acceptance_at(x)
        elif not_saved_at_default_x or force_update:  # Calculate at default_x; save.
            self._leakage_density_correction_factor = acceptance_at(self.default_x)

        return self._leakage_density_correction_factor  # type:ignore

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        **kwargs,
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
            **kwargs: Additional parameters to be passed to the MCMC sampler, such as
                `thin` and `warmup_steps`.

        Returns: Samples from posterior.
        """

        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)
        self._warn_if_posterior_was_focused_on_different_x(x)
        num_samples = torch.Size(sample_shape).numel()

        if self._sample_with_mcmc:
            samples = self._sample_posterior_mcmc(
                x=x,
                num_samples=num_samples,
                mcmc_method=self._mcmc_method,
                show_progress_bars=show_progress_bars,
                **kwargs,
            )
        elif self._method_family == "snpe":
            # Rejection sampling.
            samples, _ = utils.sample_posterior_within_prior(
                self.net,
                self._prior,
                x,
                num_samples=num_samples,
                show_progress_bars=show_progress_bars,
            )
        else:
            raise ValueError(
                "Only SNPE can use rejection sampling. All other"
                "methods require MCMC."
            )

        return samples.reshape((*sample_shape, -1))

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
                self._mcmc_init_params = self._prior.sample((num_chains,))
            elif init_strategy == "sir":
                self.net.eval()
                init_param_candidates = self._prior.sample(
                    (init_strategy_num_candidates,)
                )
                potential_function = self._get_potential_function(
                    self._prior, self.net, x, "slice_np"
                )
                log_weights = torch.cat(
                    [
                        potential_function(init_param_candidates[i, :])
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

    # NOTE: this is done here because NeuralPosterior is created in inference methods
    # at instantiation, while it could and should be created at training time.
    def set_embedding_net(self, embedding_net: nn.Module) -> None:
        """
        Set the `embedding_net` as an attribute of the `neural_net`.

        Args:
            embedding_net: Neural net to encode `x`.
        """
        assert isinstance(embedding_net, nn.Module), (
            "`embedding_net`is not a `nn.Module`. "
            "If you want to use hard-coded summary features, "
            "please simply pass the already encoded summary features as input and pass "
            "`embedding_net=None`."
        )
        self.net._embedding_net = embedding_net

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

    def _warn_if_posterior_was_focused_on_different_x(self, x: Tensor):
        """Warn if user provides an x not equal to the x_o used during inference."""

        if self._num_trained_rounds > 1 and (self._x_o_training_focused_on != x).any():
            warn(
                f"The posterior was trained over multiple rounds focused on a specific"
                f" observation x_o={self._x_o_training_focused_on.tolist()}. The"
                f" observation you provided x={x.tolist()} is not identical to x_o,"
                f" which can lead to poor performance of the inference method."
                f" Consider running inference with `num_rounds==1`, which allows to"
                f" pass any x (i.e. 'amortized' inference)."
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

    def __repr__(self):
        desc = f"""NeuralPosterior(
               method_family={self._method_family},
               net=<a {self.net.__class__.__name__}, see `.net` for details>,
               prior={self._prior!r},
               x_shape={self._x_shape!r})
               """
        return desc

    def __str__(self):
        msg = {0: "untrained", 1: "amortized"}

        focused_msg = (
            f"focused on x_o={self._x_o_training_focused_on.tolist()!r}"
            if self._x_o_training_focused_on is not None
            else ""
        )

        default_x_msg = (
            f" Evaluates and samples by default at x={self.default_x.tolist()!r}"
            if self.default_x is not None
            else ""
        )

        desc = (
            f"Posterior conditional density p(θ|x) "
            f"({msg.get(self._num_trained_rounds, focused_msg)}.){default_x_msg}.\n\n"
            f"This neural posterior was obtained with a "
            f"{self._method_family.upper()}-class "
            f"method using a {self.net.__class__.__name__.lower()}."
        )

        return desc
