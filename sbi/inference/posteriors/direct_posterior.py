# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from typing import Callable, Optional, Union, Any, Dict, Tuple

import torch
from torch import Tensor, log, nn, optim
from torch.utils import data

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.posterior_based_potential import (
    posterior_estimator_based_potential,
)
from sbi.samplers.rejection.rejection import rejection_sample_posterior_within_prior
from sbi.types import Shape, TorchModule
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes, within_support
from sbi.utils.torchutils import ensure_theta_batched
from sbi.neural_nets import flow

import time
from copy import deepcopy


class DirectPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x_o)$ with `log_prob()` and `sample()` methods, only
    applicable to SNPE.<br/><br/>
    SNPE trains a neural network to directly approximate the posterior distribution.
    However, for bounded priors, the neural network can have leakage: it puts non-zero
    mass in regions where the prior is zero. The `DirectPosterior` class wraps the
    trained network to deal with these cases.<br/><br/>
    Specifically, this class offers the following functionality:<br/>
    - correct the calculation of the log probability such that it compensates for the
      leakage.<br/>
    - reject samples that lie outside of the prior bounds.<br/><br/>
    This class can not be used in combination with SNLE or SNRE.
    """

    def __init__(
        self,
        posterior_estimator: nn.Module,
        prior: Callable,
        max_sampling_batch_size: int = 10_000,
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
    ):
        """
        Args:
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            posterior_estimator: The trained neural posterior.
            x_o: Tensor at which to evaluate the `posterior_estimator`.
            max_sampling_batch_size: Batchsize of samples being drawn from
                the proposal at every iteration.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Shape of a single simulator output. If passed, it is used to check
                the shape of the observed data and give a descriptive error.
        """
        # Because `DirectPosterior` does not take the `potential_fn` as input, it
        # builds it itself. The `potential_fn` and `theta_transform` are used only for
        # obtaining the MAP.
        potential_fn, theta_transform = posterior_estimator_based_potential(
            posterior_estimator, prior, None
        )

        super().__init__(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            device=device,
            x_shape=x_shape,
        )

        self.prior = prior
        self.posterior_estimator = posterior_estimator

        self.max_sampling_batch_size = max_sampling_batch_size
        self._leakage_density_correction_factor = None

        self._purpose = "It samples the posterior network but rejects samples that lie outside of the prior bounds."

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        max_sampling_batch_size: int = 10_000,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
    ):
        r"""Return samples from posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.
            show_progress_bars: Whether to show sampling progress monitor.
        """

        num_samples = torch.Size(sample_shape).numel()
        x = self._x_else_default_x(x)
        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun"
                f"`.build_posterior(sample_with={sample_with}).`"
            )

        samples = rejection_sample_posterior_within_prior(
            posterior_nn=self.posterior_estimator,
            prior=self.prior,
            x=x,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
        )[0]
        return samples

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        norm_posterior: bool = True,
        track_gradients: bool = False,
        leakage_correction_params: Optional[dict] = None,
    ) -> Tensor:
        r"""Returns the log-probability of the posterior $p(\theta|x)$.

        Args:
            theta: Parameters $\theta$.
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
        x = self._x_else_default_x(x)

        # TODO Train exited here, entered after sampling?
        self.posterior_estimator.eval()

        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta_repeated, x_repeated = match_theta_and_x_batch_shapes(theta, x)

        with torch.set_grad_enabled(track_gradients):

            # Evaluate on device, move back to cpu for comparison with prior.
            unnorm_log_prob = self.posterior_estimator.log_prob(
                theta_repeated, context=x_repeated
            )

            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta_repeated)

            masked_log_prob = torch.where(
                in_prior_support,
                unnorm_log_prob,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self._device),
            )

            if leakage_correction_params is None:
                leakage_correction_params = dict()  # use defaults
            log_factor = (
                log(self.leakage_correction(x=x, **leakage_correction_params))
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
            num_rejection_samples: Number of samples used to estimate correction factor.
            show_progress_bars: Whether to show a progress bar during sampling.
            rejection_sampling_batch_size: Batch size for rejection sampling.

        Returns:
            Saved or newly-estimated correction factor (as a scalar `Tensor`).
        """

        def acceptance_at(x: Tensor) -> Tensor:

            return rejection_sample_posterior_within_prior(
                posterior_nn=self.posterior_estimator,
                prior=self.prior,
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

    def sample_range(
        self,
        x_range: Tensor,  # range
        x_samples: Tensor,  # xs from p(x)
        sample_shape: Shape = torch.Size(),
        context: Optional[Tensor] = None,
        train_px: bool = False,
        x_flow: Optional[TorchModule] = None,
        sample_with: Optional[str] = None,
        max_sampling_batch_size: int = 10_000,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Return samples from posterior distribution given an observation range, $p(\theta|x_0 < x < x_1)$.

        Args:
            x_range: Conditioning context for posterior p(theta|x0 < x < x1). Provide interval bounds (upper,lower) for every dimension (resulting shape: d x 2). Set lower and upper bound to ± infinity, if context is provided.
            x_samples: Samples from p(x) provided by first round of SNPE.
            sample_shape: Desired shape of samples that are drawn from posterior. If sample_shape is multidimensional we simply draw `sample_shape.numel()` samples and then reshape into the desired shape.
            context: Provide context (fixed dimensions) when only specifying intervals for a subset of the dimensions of x.
            train_px: Whether to train a density estimator on x_samples to estimate p(x) and use it to sample from p(x_0<x<x_1).
            x_flow: Optional argument to pass a normalizing flow to be used as density estimator.
            show_progress_bars: Whether to show sampling progress monitor.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.

        Returns:
            x_accepted: Accepted observations.
            posterior_samples: Samples from posterior over observation range.
        """

        # upper and lower for each dim
        assert (
            x_range.shape[0] == 2
        ), "x_range has to be of form 2 x dim specifying [lower, upper] for each dimension."

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
                train_px
            ), "Providing context requires a flow to be trained, set train_px=True."

            assert (
                range_xs.shape[1] + context.shape[1] == x_samples.shape[1]
            ), "Please specify either a range or context all dimensions of your samples. "
        else:
            assert (
                range_xs.shape[1] == x_samples.shape[1]
            ), "When not providing context, please specify a range for all dimensions of your samples. "

        if train_px:  # density estimation of p(x)
            if x_flow is None:
                if context is None:
                    print("No context provided, use unconditional MAF.")
                    x_flow = flow.build_uncond_maf(batch_x=range_xs)
                else:
                    print("Context provided, use conditional NSF.")
                    x_flow = flow.build_nsf(batch_x=range_xs, batch_y=point_xs)

                print("Train density estimator.", end="\r")
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
                    dataloader_kwargs=dataloader_kwargs,
                    device=device,
                )

            # sample from estimated density
            print("Sample from estimated density.", end="\r")
            x_accepted = torch.empty((0, range_xs.shape[1]))
            while x_accepted.shape[0] < sample_shape[0]:
                with torch.no_grad():
                    # if context is None:
                    flow_samples = x_flow.sample(
                        num_samples=sample_shape[0], context=context
                    ).squeeze(
                        dim=0
                    )  # squeeze to remove context dim
                    flow_samples_acc = flow_samples[
                        torch.all(
                            torch.logical_and(
                                flow_samples >= x_range[:, mask][0, :],
                                flow_samples <= x_range[:, mask][1, :],
                            ),
                            dim=1,
                        )
                    ]
                    x_accepted = torch.cat((x_accepted, flow_samples_acc), dim=0)

            if context is not None:
                # merge with provided context in correct order
                x_full = torch.empty((x_accepted.shape[0], x_samples.shape[1]))
                x_full[:, mask] = x_accepted
                x_full[:, ~mask] = context.repeat(x_accepted.shape[0], 1)
                x_accepted = x_full

        else:  # rejection sampling with provided observations
            # reject observations outside range
            x_accepted = x_samples[
                torch.all(
                    torch.logical_and(
                        x_samples >= x_range[0, :], x_samples <= x_range[1, :]
                    ),
                    dim=1,
                )
            ]

        # adapt sampling shape to number of accepted samples
        try:
            sample_shape = (sample_shape[0] // x_accepted.shape[0] + 1,)
            print(
                f"{x_accepted.shape[0]} accepted observations x_i, sampling {sample_shape[0]} from each individual posterior p(theta|x_i)."
            )
        except ZeroDivisionError:
            print(
                "No observations within requested range, train a density estimator by passing 'train_px=True'."
            )

        # draw samples p(theta|x0 < x < x1)
        posterior_samples = []
        for x in x_accepted:
            sample = self.sample(
                sample_shape=sample_shape,
                sample_with=sample_with,
                x=x.unsqueeze(dim=0),
                max_sampling_batch_size=max_sampling_batch_size,
                show_progress_bars=False,
            )

            posterior_samples.append(sample)

        posterior_samples = torch.cat(posterior_samples, dim=0)

        return x_accepted, posterior_samples

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self.map_`.
        The MAP is obtained by running gradient ascent from a given number of starting
        positions (samples from the posterior with the highest log-probability). After
        the optimization is done, we select the parameter set that has the highest
        log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
            x: Observed data at which to evaluate the MAP.
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
            log_prob_kwargs={"norm_posterior": False},
        )

    def _get_mask(self, x_range: Tensor) -> Tensor:
        r"""
        Return binary mask based on x_range.

        Args:
            x_range: Conditioning context for posterior p(theta|x0 < x < x1). Provide interval bounds (upper,lower) for every dimension (resulting shape: d x 2). Set lower and upper bound to ± infinity, if context is provided.

        Returns:
            mask: Binary mask, same shape as x_range.
        """

        if x_range.type() != "torch.FloatTensor":
            x_range = x_range.float()
        mask = torch.logical_not(
            torch.logical_and(
                x_range[0, :] == float("-inf"), x_range[1, :] == float("inf")
            )
        )
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
            dict(train_loader_kwargs, **dataloader_kwargs)
            if dataloader_kwargs is not None
            else train_loader_kwargs
        )
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle": False,
            "drop_last": True,
            "sampler": data.sampler.SubsetRandomSampler(self.val_indices),
        }
        val_loader_kwargs = (
            dict(val_loader_kwargs, **dataloader_kwargs)
            if dataloader_kwargs is not None
            else val_loader_kwargs
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

        _best_val_log_prob = float("-inf")
        _best_model_state_dict = None
        _epochs_since_last_improvement = 0

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
                    batch_loss = torch.mean(
                        -neural_net.log_prob(x_batch, context=context_batch)
                    )

                train_log_prob_sum += batch_loss.sum().item()

                batch_loss.backward()
                if clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        neural_net.parameters(), max_norm=clip_max_norm
                    )
                optimizer.step()

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
                        batch_log_prob = neural_net.log_prob(
                            x_batch, context=context_batch
                        )
                    log_prob_sum += batch_log_prob.sum().item()

            # Take mean over all validation samples.
            _val_log_prob = log_prob_sum / (len(val_loader) * val_loader.batch_size)

            # Log validation log prob for every epoch.
            _summary["validation_log_probs"].append(_val_log_prob)
            _summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            (
                converged,
                _best_model_state_dict,
                _epochs_since_last_improvement,
                _best_val_log_prob,
            ) = self._converged(
                neural_net,
                _val_log_prob,
                _best_val_log_prob,
                _best_model_state_dict,
                _epochs_since_last_improvement,
                epoch,
                stop_after_epochs,
            )
            if epoch % 10 == 0:
                print(
                    f"epoch: {epoch:5}, val_log_prob: {_val_log_prob:13.4}, best val_log_prob: {_best_val_log_prob:17.4}",
                    end="\r",
                )
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
