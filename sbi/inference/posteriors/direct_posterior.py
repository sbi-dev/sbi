# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from warnings import warn
from copy import deepcopy

import numpy as np
import torch
import torch.distributions.transforms as torch_tf
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from torch import Tensor, log, nn, optim
from torch.utils import data

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape, TorchModule, Shape
from sbi.utils import del_entries, mcmc_transform, rejection_sample, within_support
from sbi.utils.conditional_density import condition_mog, extract_and_transform_mog
from sbi.analysis.plot import pairplot
from sbi.utils.torchutils import (
    atleast_2d,
    atleast_2d_float32_tensor,
    batched_first_of_batch,
    ensure_theta_batched,
)
from sbi.neural_nets import flow
import matplotlib.pyplot as plt


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
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
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
                log(self.leakage_correction(x=batched_first_of_batch(x), **leakage_correction_params))
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
        is_new_x = self.default_x is None or (x is not self.default_x and (x != self.default_x).any())

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
            mcmc_method, mcmc_parameters = self._potentially_replace_mcmc_parameters(mcmc_method, mcmc_parameters)
            transform = mcmc_transform(self._prior, device=self._device, **mcmc_parameters)
            transformed_samples = self._sample_posterior_mcmc(
                num_samples=num_samples,
                potential_fn=potential_fn_provider(self._prior, self.net, x, mcmc_method, transform),
                init_fn=self._build_mcmc_init_fn(
                    self._prior,
                    potential_fn_provider(self._prior, self.net, x, "slice_np", transform),
                    transform=transform,
                    **mcmc_parameters,
                ),
                mcmc_method=mcmc_method,
                show_progress_bars=show_progress_bars,
                **mcmc_parameters,
            )
            samples = transform.inv(transformed_samples)
        elif sample_with == "rejection":
            rejection_sampling_parameters = self._potentially_replace_rejection_parameters(
                rejection_sampling_parameters
            )
            if "proposal" not in rejection_sampling_parameters:
                assert not self.net.training, "Posterior nn must be in eval mode for sampling."

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
                    potential_fn=potential_fn_provider(self._prior, self.net, x, "rejection"),
                    num_samples=num_samples,
                    show_progress_bars=show_progress_bars,
                    **rejection_sampling_parameters,
                )
        else:
            raise NameError("The only implemented sampling methods are `mcmc` and `rejection`.")

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
            raise NotImplementedError("`sample_conditional` is not implemented for SNPE-A.")

        net = self.net
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))

        if type(net._distribution) is mdn:
            condition = atleast_2d_float32_tensor(condition)
            num_samples = torch.Size(sample_shape).numel()

            logits, means, precfs, _ = extract_and_transform_mog(nn=net, context=x)
            logits, means, precfs, _ = condition_mog(self._prior, condition, dims_to_sample, logits, means, precfs)

            # Currently difficult to integrate `sample_posterior_within_prior`.
            warn("Sampling MoG analytically. " "Some of the samples might not be within the prior support!")
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

    def sample_range(
        self,
        x_range: Tensor,  # range
        x_samples: Tensor,  # xs from p(x)
        sample_shape: Shape = torch.Size(),
        context: Optional[Tensor] = None,
        train_px: bool = False,
        x_flow: Optional[TorchModule] = None,
        visualize_training: int = 0,
        sample_with: Optional[str] = None,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
        sample_with_mcmc: Optional[bool] = None,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        show_train_summary: bool = False,
        visualize_training_interval: int = 0,
        dataloader_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ) -> Tensor:

        r"""
        Return samples from posterior distribution given an observation range, $p(\theta|x_0 < x < x_1)$.

        Samples are obtained either with rejection sampling or MCMC. Rejection sampling
        will be a lot faster if leakage is rather low. If leakage is high (e.g. over
        99%, which can happen in multi-round SNPE), MCMC can be faster than rejection
        sampling.

        Args:
            x_range: Conditioning context for posterior $p(\theta|x0 < x < x1)$. Provide a range (upper,lower) for every dimension (resulting shape: n x 2). Set lower and upper bound to ± infinity if context is provided.
            x_samples: Samples from p(x) provided by first round of SNPE.
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            train_px: Whether to train a density estimator on x_samples to estimate
                p(x) and use it to sample from p(x_0<x<x_1).
            x_flow: Optional argument to pass a specific normalizing flow that is used
                as density estimator.
            show_progress_bars: Whether to show sampling progress monitor.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection`]. With default parameters, `rejection` samples
                from the posterior estimated by the neural net and rejects only if the
                samples are outside of the prior support.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
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
            sample_with_mcmc: Deprecated since `sbi v0.17.0`. Use `sample_with=mcmc`
                instead.

        Returns:
            x_accepted: Accepted observations.
            posterior_samples: Samples from posterior over observation range.
        """
        logging.captureWarnings(True)

        # upper and lower for each dim
        assert x_range.shape[0] == 2, "x_range has to be of form 2 x dim specifying [lower, upper] for each dimension."

        # same dimensions as samples
        # either range for every dim or range + context = same as samples
        assert (
            x_range.shape[1] == x_samples.shape[1]
        ), "Please specify ranges for all dimensions of your samples. Set to ± infinity, if context is provided."

        mask = self._get_mask(x_range)
        range_xs = x_samples[:, mask]
        point_xs = x_samples[:, ~mask] if context is not None else None

        if context is not None:
            assert (
                train_px or (x_flow is not None),
                "Providing context requires a flow to be passed as argument (x_flow) or trained (train_px=True).",
            )
            assert (
                range_xs.shape[1] + context.shape[1] == x_samples.shape[1]
            ), "Please specify context to condition on for all dimensions of your samples. "

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

        if train_px:  # density estimation of p(x)
            if x_flow is None:
                if context is None:
                    print("No context provided, use unconditional MAF.")
                    x_flow = flow.build_uncond_maf(batch_x=range_xs)
                else:
                    print("Context provided, use conditional NSF.")
                    # x_flow = flow.build_maf(batch_x=range_xs, batch_y=point_xs)
                    x_flow = flow.build_nsf(batch_x=range_xs, batch_y=point_xs)

                print("Train density estimator.")
                self.train_density_estimator(
                    x_flow,
                    x=range_xs,
                    context=point_xs,
                    training_batch_size=training_batch_size,
                    learning_rate=learning_rate,
                    validation_fraction=validation_fraction,
                    stop_after_epochs=stop_after_epochs,
                    max_num_epochs=max_num_epochs,
                    clip_max_norm=clip_max_norm,
                    resume_training=resume_training,
                    show_train_summary=show_train_summary,
                    visualize_training_interval=visualize_training,
                    dataloader_kwargs=dataloader_kwargs,
                    device=device,
                )

            # sample from estimated density
            print("Sample from estimated density.")
            x_accepted = torch.empty((0, range_xs.shape[1]))
            while x_accepted.shape[0] < sample_shape[0]:
                with torch.no_grad():
                    # if context is None:
                    flow_samples = x_flow.sample(num_samples=sample_shape[0], context=context).squeeze(
                        dim=0
                    )  # squeeze to remove context-dim
                    flow_samples_acc = flow_samples[
                        torch.all(
                            torch.logical_and(
                                flow_samples >= x_range[:, mask][0, :], flow_samples <= x_range[:, mask][1, :]
                            ),
                            dim=1,
                        )
                    ]
                    # print(f"Sampled {sample_shape[0]}, accepted {flow_samples_acc.shape[0]}.")
                    x_accepted = torch.cat((x_accepted, flow_samples_acc), dim=0)

            if context is not None:
                # merge with provided context in correct order to obtain full samples
                x_full = torch.empty((x_accepted.shape[0], x_samples.shape[1]))
                x_full[:, mask] = x_accepted
                x_full[:, ~mask] = context.repeat(x_accepted.shape[0], 1)
                x_accepted = x_full

        else:  # rejection sampling with provided observations
            # reject observations outside range
            x_accepted = x_samples[
                torch.all(torch.logical_and(x_samples >= x_range[0, :], x_samples <= x_range[1, :]), dim=1)
            ]

        # adapt sampling shape to number of accepted samples
        try:
            sample_shape = (sample_shape[0] // x_accepted.shape[0] + 1,)
            print(
                f"{x_accepted.shape[0]} accepted observations x_i, sampling {sample_shape[0]} from each individual posterior p(theta|x_i)."
            )
        except ZeroDivisionError:
            print("No observations within requested range, train a density estimator by passing 'train_px=True'.")

        # draw samples p(theta|x0 < x < x1)
        posterior_samples = []
        for x in x_accepted:
            sample = self.sample(
                sample_shape,
                x.unsqueeze(dim=0),
                False,  # show_progress_bars
                sample_with,
                mcmc_method,
                mcmc_parameters,
                rejection_sampling_parameters,
                sample_with_mcmc,
            )
            posterior_samples.append(sample)

        posterior_samples = torch.cat(posterior_samples, dim=0)

        return x_accepted, posterior_samples

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
            raise NotImplementedError("This functionality is only available for MDN based posteriors.")

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

    def _get_mask(self, x_range):
        if x_range.type() != "torch.FloatTensor":
            x_range = x_range.float()
        mask = torch.logical_not(torch.logical_and(x_range[0, :] == float("-inf"), x_range[1, :] == float("inf")))
        return mask

    def get_dataloaders(
        self,
        dataset: data.TensorDataset,
        training_batch_size: int = 50,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """Return dataloaders for training and validation.

        Args:
            dataset: holding all theta and x, optionally masks.
            training_batch_size: training arg of inference methods.
            resume_training: Whether the current call is resuming training so that no
                new training and validation indices into the dataset have to be created.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Tuple of dataloaders for training and validation.

        """

        # Get total number of training examples.
        num_examples = len(dataset)

        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        if not resume_training:
            permuted_indices = torch.randperm(num_examples)
            self.train_indices, self.val_indices = (
                permuted_indices[:num_training_examples],
                permuted_indices[num_training_examples:],
            )

        # Create training and validation loaders using a subset sampler.
        # Intentionally use dicts to define the default dataloader args
        # Then, use dataloader_kwargs to override (or add to) any of these defaults
        # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last": True,
            "sampler": data.sampler.SubsetRandomSampler(self.train_indices),
        }
        train_loader_kwargs = (
            dict(train_loader_kwargs, **dataloader_kwargs) if dataloader_kwargs is not None else train_loader_kwargs
        )
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": data.sampler.SubsetRandomSampler(self.val_indices),
        }
        val_loader_kwargs = (
            dict(val_loader_kwargs, **dataloader_kwargs) if dataloader_kwargs is not None else val_loader_kwargs
        )
        train_loader = data.DataLoader(dataset, **train_loader_kwargs)
        val_loader = data.DataLoader(dataset, **val_loader_kwargs)

        return train_loader, val_loader

    def _converged(
        self,
        neural_net,
        _val_log_prob,
        _best_val_log_prob,
        _best_model_state_dict,
        _epochs_since_last_improvement,
        epoch: int,
        stop_after_epochs: int,
    ) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """

        converged = False

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or _val_log_prob > _best_val_log_prob:
            _best_val_log_prob = _val_log_prob
            _epochs_since_last_improvement = 0
            _best_model_state_dict = deepcopy(neural_net.state_dict())
        else:
            _epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if _epochs_since_last_improvement > stop_after_epochs - 1:
            neural_net.load_state_dict(_best_model_state_dict)
            converged = True

        return (
            converged,
            _best_model_state_dict,
            _epochs_since_last_improvement,
            _best_val_log_prob,
        )

    def train_density_estimator(
        self,
        neural_net,
        x,
        context: Tensor = None,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        show_train_summary: bool = False,
        visualize_training_interval: int = 0,
        dataloader_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ):
        """
        Train density estimator to estimate p(x|context) (adapated from snpe_base.py).

        Args:
            neural_net: density estimator
            x: data
            context: data to condition on
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. If None, we
                train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
            device: Device string

        Returns:
            Density estimator that approximates the distribution $p(x|context)$.
        """
        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        _summary = dict(
            median_observation_distances=[],
            epochs=[],
            best_validation_log_probs=[],
            validation_log_probs=[],
            train_log_probs=[],
            epoch_durations_sec=[],
        )

        # Dataset is shared for training and validation loaders.
        if context is None:
            dataset = data.TensorDataset(x)
        else:
            dataset = data.TensorDataset(x, context)

        train_loader, val_loader = self.get_dataloaders(
            dataset,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        neural_net.to(device)

        if not resume_training:
            optimizer = optim.Adam(list(neural_net.parameters()), lr=learning_rate)
            epoch, _val_log_prob = 0, float("-Inf")

        if visualize_training_interval > 0 and x.shape[1] == 2:
            steps = 200
            lx, ly = x.min(dim=0).values.int() - 2
            ux, uy = x.max(dim=0).values.int() + 2
            xline = torch.linspace(lx.item(), ux.item(), steps=steps)
            yline = torch.linspace(ly.item(), uy.item(), steps=steps)
            xgrid, ygrid = torch.meshgrid(xline, yline)
            xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        _best_val_log_prob = float("-inf")
        _best_model_state_dict = None
        _epochs_since_last_improvement = 0
        print("epoch | _val_log_prob | best val_log_prob")
        while epoch <= max_num_epochs:
            # Train for a single epoch.
            neural_net.train()
            train_log_prob_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                optimizer.zero_grad()
                # Get batches on current device.
                if context is None:
                    x_batch = batch[0].to(device)
                    batch_loss = torch.mean(-neural_net.log_prob(x_batch))
                else:
                    x_batch, context_batch = (
                        batch[0].to(device),
                        batch[1].to(device),
                    )
                    batch_loss = torch.mean(-neural_net.log_prob(x_batch, context=context_batch))

                train_log_prob_sum += batch_loss.sum().item()

                batch_loss.backward()
                if clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(neural_net.parameters(), max_norm=clip_max_norm)
                optimizer.step()

            # TODO: move visualization plot.py
            if visualize_training_interval > 0 and x.shape[1] == 2:
                if epoch % visualize_training_interval == 0:
                    with torch.no_grad():
                        if context is None:
                            zgrid = neural_net.log_prob(xyinput).exp().reshape(steps, steps)
                        else:
                            zgrid = neural_net.log_prob(xyinput, context_batch).exp().reshape(steps, steps)

                    plt.scatter(x[:, 0], x[:, 1], alpha=0.5, s=10)
                    plt.contour(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
                    plt.title("epoch {} (loss: {})".format(epoch, batch_loss.sum().item()))
                    plt.show()

            epoch += 1

            # Calculate validation performance.
            neural_net.eval()
            log_prob_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    if context is None:
                        x_batch = batch[0].to(device)
                        # Take negative loss here to get validation log_prob.
                        batch_log_prob = neural_net.log_prob(x_batch)
                    else:
                        x_batch, context_batch = (
                            batch[0].to(device),
                            batch[1].to(device),
                        )
                        # Take negative loss here to get validation log_prob.
                        batch_log_prob = neural_net.log_prob(x_batch, context=context_batch)
                    log_prob_sum += batch_log_prob.sum().item()

            # Take mean over all validation samples.
            _val_log_prob = log_prob_sum / (len(val_loader) * val_loader.batch_size)

            # Log validation log prob for every epoch.
            _summary["validation_log_probs"].append(_val_log_prob)
            _summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            (converged, _best_model_state_dict, _epochs_since_last_improvement, _best_val_log_prob,) = self._converged(
                neural_net,
                _val_log_prob,
                _best_val_log_prob,
                _best_model_state_dict,
                _epochs_since_last_improvement,
                epoch,
                stop_after_epochs,
            )
            if epoch % 10 == 0:
                print(f"{epoch:5} | {_val_log_prob:13.4} | {_best_val_log_prob:17.4}")
            if converged:
                print(f"Converged after {epoch} epochs.")
                break

        # Update summary.
        _summary["epochs"].append(epoch)
        _summary["best_validation_log_probs"].append(_best_val_log_prob)

        if show_train_summary:
            epochs = _summary["epochs"][-1]
            best_validation_log_probs = _summary["best_validation_log_probs"][-1]

            description = f"""
            -------------------------
            ||||| Density Estimator Training Stats|||||:
            -------------------------
            Epochs trained: {epochs}
            Best validation performance: {best_validation_log_probs:.4f}
            -------------------------
            """

            print(description)

        return deepcopy(neural_net)


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

    def posterior_potential(self, theta: Union[Tensor, np.array], track_gradients: bool = False) -> Tensor:
        """
        Return posterior theta log prob. $p(\theta|x)$, $-\infty$ if outside prior.

        Args:
            theta:  Parameters $\theta$. If a `transform` is applied, `theta` should be
                in transformed space.
        """

        # Device is the same for net and prior.
        transformed_theta = ensure_theta_batched(torch.as_tensor(theta, dtype=torch.float32)).to(self.device)
        # Transform `theta` from transformed (i.e. unconstrained) to untransformed
        # space.
        theta = self.transform.inv(transformed_theta)
        log_abs_det = self.transform.log_abs_det_jacobian(theta, transformed_theta)

        theta_repeated, x_repeated = DirectPosterior._match_theta_and_x_batch_shapes(theta, self.x)

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

    def pyro_potential(self, theta: Dict[str, Tensor], track_gradients: bool = False) -> Tensor:
        r"""Return posterior theta log prob. $p(\theta|x)$, $-\infty$ if outside prior."

        Args:
            theta: Parameters $\theta$ (from pyro sampler). If a `transform` is
                applied, `theta` should be in transformed space.

        Returns:
            Negative posterior log probability $p(\theta|x)$, masked outside of prior.
        """

        theta = next(iter(theta.values()))
        return -self.posterior_potential(theta, track_gradients=track_gradients)
