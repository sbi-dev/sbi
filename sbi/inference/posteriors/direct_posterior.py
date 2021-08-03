# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from warnings import warn

import numpy as np
import torch
import torch.distributions.transforms as torch_tf
from torch import Tensor, log, nn
from warnings import warn

from sbi import utils as utils
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries, mcmc_transform, rejection_sample, within_support
from sbi.utils.torchutils import (
    atleast_2d,
    batched_first_of_batch,
    ensure_theta_batched,
    atleast_2d_float32_tensor,
)
from sbi.utils.conditional_density import extract_and_transform_mog, condition_mog


class DirectPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNPE.<br/><br/>
    SNPE trains a neural network to directly approximate the posterior distribution.
    However, for bounded priors, the neural network can have leakage: it puts non-zero
    mass in regions where the prior is zero. The `DirectPosterior` class wraps the
    trained network to deal with these cases.<br/><br/>
    Specifically, this class offers the following functionality:<br/>
    - correct the calculation of the log probability such that it compensates for the
      leakage.<br/>
    - reject samples that lie outside of the prior bounds.<br/>
    - alternatively, if leakage is very high (which can happen for multi-round SNPE),
      sample from the posterior with MCMC.<br/><br/>
    The neural network itself can be accessed via the `.net` attribute.
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        prior,
        x_shape: torch.Size,
        sample_with: str = "rejection",
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
            x_shape: Shape of a single simulator output.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection`]. With default parameters, `rejection` samples
                from the posterior estimated by the neural net and rejects only if the
                samples are outside of the prior support.
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
            rejection_sampling_parameters: Dictionary overriding the default parameters
                for rejection sampling. The following parameters are supported:
                `proposal` as the proposal distribtution (default is the trained
                neural net). `max_sampling_batch_size` as the batchsize of samples
                being drawn from the proposal at every iteration.
                `num_samples_to_find_max` as the number of samples that are used to
                find the maximum of the `potential_fn / proposal` ratio.
                `num_iter_to_find_max` as the number of gradient ascent iterations to
                find the maximum of that ratio. `m` as multiplier to that ratio.
            device: Training device, e.g., cpu or cuda:0
        """

        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

        self._purpose = (
            "It allows to .sample() and .log_prob() the posterior and wraps the "
            "output of the .net to avoid leakage into regions with 0 prior probability."
        )

    @property
    def _sample_with_mcmc(self) -> bool:
        """
        Deprecated, will be removed in future versions of `sbi`.

        Return `True` if NeuralPosterior instance should use MCMC in `.sample()`.
        """
        return self._sample_with_mcmc

    @_sample_with_mcmc.setter
    def _sample_with_mcmc(self, value: bool) -> None:
        """
        Deprecated, will be removed in future versions of `sbi`.

        See `set_sample_with_mcmc`."""
        self._set_sample_with_mcmc(value)

    def _set_sample_with_mcmc(self, use_mcmc: bool) -> "NeuralPosterior":
        """
        Deprecated, will be removed in future versions of `sbi`.

        Turns MCMC sampling on or off and returns `NeuralPosterior`.

        Args:
            use_mcmc: Flag to set whether or not MCMC sampling is used.

        Returns:
            `NeuralPosterior` for chainable calls.

        Raises:
            ValueError: on attempt to turn off MCMC sampling for family of methods that
                do not support rejection sampling.
        """
        warn(
            f"You set `sample_with_mcmc={use_mcmc}`. This is deprecated "
            "since `sbi v0.17.0` and will lead to an error in future versions. "
            "Please use `sample_with='mcmc'` instead."
        )
        if use_mcmc:
            self.set_sample_with("mcmc")
        else:
            self.set_sample_with("rejection")
        self._sample_with_mcmc = use_mcmc
        return self

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        r"""
        Returns the log-probability of the posterior $p(\theta|x).$

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            norm_posterior: Whether to enforce a normalized posterior density.
                Renormalization of the posterior is useful when some
                probability falls out or leaks out of the prescribed prior support.
                The normalizing factor is calculated via rejection sampling, so if you
                need speedier but unnormalized log posterior estimates set here
                `norm_posterior=False`. The returned log posterior is set to
                -∞ outside of the prior support regardless of this setting.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            leakage_correction_params: A `dict` of keyword arguments to override the
                default values of `leakage_correction()`. Possible options are:
                `num_rejection_samples`, `force_update`, `show_progress_bars`, and
                `rejection_sampling_batch_size`.
                These parameters only have an effect if `norm_posterior=True`.

        Returns:
            `(len(θ),)`-shaped log posterior probability $\log p(\theta|x)$ for θ in the
            support of the prior, -∞ (corresponding to 0 probability) outside.
        """

        # TODO Train exited here, entered after sampling?
        self.net.eval()

        theta, x = self._prepare_theta_and_x_for_log_prob_(theta, x)
        theta_repeated, x_repeated = self._match_theta_and_x_batch_shapes(theta, x)

        with torch.set_grad_enabled(track_gradients):

            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.net.log_prob(theta_repeated, x_repeated)

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self._prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )

            if leakage_correction_params is None:
                leakage_correction_params = dict()  # use defaults
            log_factor = (
                log(
                    self.leakage_correction(
                        x=batched_first_of_batch(x), **leakage_correction_params
                    )
                )
                if norm_posterior
                else 0
            )

            return masked_log_prob - log_factor

    @torch.no_grad()
    def leakage_correction(
        self,
        x: Tensor,
        num_rejection_samples: int = 10_000,
        force_update: bool = False,
        show_progress_bars: bool = False,
        rejection_sampling_batch_size: int = 10_000,
    ) -> Tensor:
        r"""Return leakage correction factor for a leaky posterior density estimate.

        The factor is estimated from the acceptance probability during rejection
        sampling from the posterior.

        This is to avoid re-estimating the acceptance probability from scratch
        whenever `log_prob` is called and `norm_posterior=True`. Here, it
        is estimated only once for `self.default_x` and saved for later. We
        re-evaluate only whenever a new `x` is passed.

        Arguments:
            x: Conditioning context for posterior $p(\theta|x)$.
            num_rejection_samples: Number of samples used to estimate correction factor.
            force_update: Whether to force a reevaluation of the leakage correction even
                if the context `x` is the same as `self.default_x`. This is useful to
                enforce a new leakage estimate for rounds after the first (2, 3,..).
            show_progress_bars: Whether to show a progress bar during sampling.
            rejection_sampling_batch_size: Batch size for rejection sampling.

        Returns:
            Saved or newly-estimated correction factor (as a scalar `Tensor`).
        """

        def acceptance_at(x: Tensor) -> Tensor:

            return utils.rejection_sample_posterior_within_prior(
                posterior_nn=self.net,
                prior=self._prior,
                x=x.to(self._device),
                num_samples=num_rejection_samples,
                show_progress_bars=show_progress_bars,
                sample_for_correction_factor=True,
                max_sampling_batch_size=rejection_sampling_batch_size,
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
        sample_with: Optional[str] = None,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_with_mcmc: Optional[bool] = None,
    ) -> Tensor:
        r"""
        Return samples from posterior distribution $p(\theta|x)$.

        Samples are obtained either with rejection sampling or MCMC. Rejection sampling
        will be a lot faster if leakage is rather low. If leakage is high (e.g. over
        99%, which can happen in multi-round SNPE), MCMC can be faster than rejection
        sampling.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.
            show_progress_bars: Whether to show sampling progress monitor.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection`]. With default parameters, `rejection` samples
                from the posterior estimated by the neural net and rejects only if the
                samples are outside of the prior support.
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
                `proposal` as the proposal distribtution (default is the trained
                neural net).
                `max_sampling_batch_size` as the batchsize of samples being drawn from
                the proposal at every iteration.
                `num_samples_to_find_max` as the number of samples that are used to
                find the maximum of the `potential_fn / proposal` ratio.
                `num_iter_to_find_max` as the number of gradient ascent iterations to
                find the maximum of that ratio.
                `m` as multiplier to that ratio.
            sample_with_mcmc: Deprecated since `sbi v0.17.0`. Use `sample_with=mcmc`
                instead.

        Returns:
            Samples from posterior.
        """

        if sample_with_mcmc is not None:
            warn(
                f"You set `sample_with_mcmc={sample_with_mcmc}`. This is deprecated "
                "since `sbi v0.17.0` and will lead to an error in future versions. "
                "Please use `sample_with='mcmc'` instead."
            )
            if sample_with_mcmc:
                sample_with = "mcmc"

        self.net.eval()

        sample_with = sample_with if sample_with is not None else self._sample_with

        x, num_samples = self._prepare_for_sample(x, sample_shape)

        potential_fn_provider = PotentialFunctionProvider()
        if sample_with == "mcmc":
            mcmc_method, mcmc_parameters = self._potentially_replace_mcmc_parameters(
                mcmc_method, mcmc_parameters
            )
            transform = mcmc_transform(
                self._prior, device=self._device, **mcmc_parameters
            )
            transformed_samples = self._sample_posterior_mcmc(
                num_samples=num_samples,
                potential_fn=potential_fn_provider(
                    self._prior, self.net, x, mcmc_method, transform
                ),
                init_fn=self._build_mcmc_init_fn(
                    self._prior,
                    potential_fn_provider(
                        self._prior, self.net, x, "slice_np", transform
                    ),
                    transform=transform,
                    **mcmc_parameters,
                ),
                mcmc_method=mcmc_method,
                show_progress_bars=show_progress_bars,
                **mcmc_parameters,
            )
            samples = transform.inv(transformed_samples)
        elif sample_with == "rejection":
            rejection_sampling_parameters = (
                self._potentially_replace_rejection_parameters(
                    rejection_sampling_parameters
                )
            )
            if "proposal" not in rejection_sampling_parameters:
                assert (
                    not self.net.training
                ), "Posterior nn must be in eval mode for sampling."

                # If the user does not explictly pass a `proposal`, we sample from the
                # neural net estimating the posterior and reject only those samples
                # that are outside of the prior support. This can be considered as
                # rejection sampling with a very good proposal.
                samples = utils.rejection_sample_posterior_within_prior(
                    posterior_nn=self.net,
                    prior=self._prior,
                    x=x.to(self._device),
                    num_samples=num_samples,
                    show_progress_bars=show_progress_bars,
                    sample_for_correction_factor=True,
                    **rejection_sampling_parameters,
                )[0]
            else:
                samples, _ = rejection_sample(
                    potential_fn=potential_fn_provider(
                        self._prior, self.net, x, "rejection"
                    ),
                    num_samples=num_samples,
                    show_progress_bars=show_progress_bars,
                    **rejection_sampling_parameters,
                )
        else:
            raise NameError(
                "The only implemented sampling methods are `mcmc` and `rejection`."
            )

        self.net.train(True)

        return samples.reshape((*sample_shape, -1))

    def sample_conditional(
        self,
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
        if not hasattr(self.net, "_distribution"):
            raise NotImplementedError(
                "`sample_conditional` is not implemented for SNPE-A."
            )

        net = self.net
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))

        if type(net._distribution) is mdn:
            condition = atleast_2d_float32_tensor(condition)
            num_samples = torch.Size(sample_shape).numel()

            logits, means, precfs, _ = extract_and_transform_mog(nn=net, context=x)
            logits, means, precfs, _ = condition_mog(
                self._prior, condition, dims_to_sample, logits, means, precfs
            )

            # Currently difficult to integrate `sample_posterior_within_prior`.
            warn(
                "Sampling MoG analytically. "
                "Some of the samples might not be within the prior support!"
            )
            samples = mdn.sample_mog(num_samples, logits, means, precfs)
            return samples.detach().reshape((*sample_shape, -1))

        else:
            return super().sample_conditional(
                PotentialFunctionProvider(),
                sample_shape,
                condition,
                dims_to_sample,
                x,
                sample_with,
                show_progress_bars,
                mcmc_method,
                mcmc_parameters,
                rejection_sampling_parameters,
            )

    def log_prob_conditional(
        self,
        theta: Tensor,
        condition: Tensor,
        dims_to_evaluate: List[int],
        x: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluates the conditional posterior probability of a MDN at a context x for
        a value theta given a condition.

        This function only works for MDN based posteriors, becuase evaluation is done
        analytically. For all other density estimators a `NotImplementedError` will be
        raised!

        Args:
            theta: Parameters $\theta$.
            condition: Parameter set that all dimensions not specified in
                `dims_to_sample` will be fixed to. Should contain dim_theta elements,
                i.e. it could e.g. be a sample from the posterior distribution.
                The entries at all `dims_to_sample` will be ignored.
            dims_to_evaluate: Which dimensions to evaluate the sample for.
                The dimensions not specified in `dims_to_evaluate` will be fixed to values given in `condition`.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x` passed to `set_default_x()`.

        Returns:
            log_prob: `(len(θ),)`-shaped normalized (!) log posterior probability
                $\log p(\theta|x) for θ in the support of the prior, -∞ (corresponding
                to 0 probability) outside.
        """

        if type(self.net._distribution) == mdn:
            logits, means, precfs, sumlogdiag = extract_and_transform_mog(self, x)
            logits, means, precfs, sumlogdiag = condition_mog(
                self._prior, condition, dims_to_evaluate, logits, means, precfs
            )

            batch_size, dim = theta.shape
            prec = precfs.transpose(3, 2) @ precfs

            self.net.eval()  # leakage correction requires eval mode

            if dim != len(dims_to_evaluate):
                X = X[:, dims_to_evaluate]

            # Implementing leakage correction is difficult for conditioned MDNs,
            # because samples from self i.e. the full posterior are used rather
            # then from the new, conditioned posterior.
            warn("Probabilities are not adjusted for leakage.")

            log_prob = mdn.log_prob_mog(
                theta,
                logits.repeat(batch_size, 1),
                means.repeat(batch_size, 1, 1),
                prec.repeat(batch_size, 1, 1, 1),
                sumlogdiag.repeat(batch_size, 1),
            )

            self.net.train(True)
            return log_prob.detach()

        else:
            raise NotImplementedError(
                "This functionality is only available for MDN based posteriors."
            )

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1000,
        learning_rate: float = 1e-2,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
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
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
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
            log_prob_kwargs={"norm_posterior": False},
        )


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
        posterior_nn: nn.Module,
        x: Tensor,
        method: str,
        transform: torch_tf.Transform = torch_tf.identity_transform,
    ) -> Callable:
        """Return potential function.

        Switch on numpy or pyro potential function based on `method`.
        """
        self.posterior_nn = posterior_nn
        self.prior = prior
        self.device = next(posterior_nn.parameters()).device
        self.x = atleast_2d(x).to(self.device)
        self.transform = transform

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
        self, theta: Union[Tensor, np.array], track_gradients: bool = False
    ) -> Tensor:
        """
        Return posterior theta log prob. $p(\theta|x)$, $-\infty$ if outside prior.

        Args:
            theta:  Parameters $\theta$. If a `transform` is applied, `theta` should be
                in transformed space.
        """

        # Device is the same for net and prior.
        transformed_theta = ensure_theta_batched(
            torch.as_tensor(theta, dtype=torch.float32)
        ).to(self.device)
        # Transform `theta` from transformed (i.e. unconstrained) to untransformed
        # space.
        theta = self.transform.inv(transformed_theta)
        log_abs_det = self.transform.log_abs_det_jacobian(theta, transformed_theta)

        theta_repeated, x_repeated = DirectPosterior._match_theta_and_x_batch_shapes(
            theta, self.x
        )

        with torch.set_grad_enabled(track_gradients):

            # Evaluate on device, move back to cpu for comparison with prior.
            posterior_log_prob = self.posterior_nn.log_prob(theta_repeated, x_repeated)
            posterior_log_prob_transformed = posterior_log_prob - log_abs_det

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            posterior_log_prob_transformed = torch.where(
                in_prior_support,
                posterior_log_prob_transformed,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self.device),
            )

        return posterior_log_prob_transformed

    def pyro_potential(
        self, theta: Dict[str, Tensor], track_gradients: bool = False
    ) -> Tensor:
        r"""Return posterior theta log prob. $p(\theta|x)$, $-\infty$ if outside prior."

        Args:
            theta: Parameters $\theta$ (from pyro sampler). If a `transform` is
                applied, `theta` should be in transformed space.

        Returns:
            Negative posterior log probability $p(\theta|x)$, masked outside of prior.
        """

        theta = next(iter(theta.values()))
        return -self.posterior_potential(theta, track_gradients=track_gradients)
