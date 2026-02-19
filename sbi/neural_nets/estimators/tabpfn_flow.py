import math
import warnings
from typing import Literal, Mapping, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from tabpfn import TabPFNClassifier, TabPFNRegressor
from torch import Tensor
from torch.distributions import Distribution


# Suppress specific sklearn warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`BaseEstimator._validate_data` is deprecated",
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="'force_all_finite' was renamed"
)


class NPE_PFN_Core:
    """TabPFN-based simulation-based inference that follows SBI-like interface.

    This class provides similar functionality to SBI's NPE (Neural Posterior Estimation)
    but uses TabPFN as the underlying model.
    """

    def __init__(
        self,
        show_progress_bars: bool = False,
        prior: Optional[Distribution] = None,
        embedding_net: Optional[torch.nn.Module] = None,
        x_shape: Optional[torch.Size] = None,
        regressor_init_kwargs: Mapping = {},
        classifier_init_kwargs: Mapping = {},
    ) -> None:
        """Initialize TabPFN-based inference."""
        self.show_progress_bars = show_progress_bars
        self.prior = prior
        self.regressor_init_kwargs = regressor_init_kwargs
        self.classifier_init_kwargs = classifier_init_kwargs

        self._model = TabPFNRegressor(**self.regressor_init_kwargs)
        self._model_classifier = None
        self.embedding_net = embedding_net
        self.x_shape = x_shape

        # Initialize theta, x for storage of parameters and simulations
        self._theta_train = None
        self._x_train = None

    def __getstate__(self):
        """Prepare the object state for pickling."""
        state = self.__dict__.copy()
        # Remove the model and classifier from the state
        state["_model"] = None
        state["_model_classifier"] = None
        return state

    def __setstate__(self, state):
        """Restore the object state after unpickling."""
        self.__dict__.update(state)
        # Reinitialize the model and classifier
        self._model = TabPFNRegressor(**self.regressor_init_kwargs)
        if self._model_classifier is not None:
            self._model_classifier = DensityRatioWrapper(**self.classifier_init_kwargs)

    def append_simulations(self, theta: Tensor, x: Tensor):
        """Append new simulation outputs to training data."""
        self._theta_train = None
        self._x_train = None
        if self.embedding_net:
            x = x.reshape(-1, *self.x_shape)
            x = self.embedding_net(x)
        self._theta_train = self._validate_theta(theta)
        self._x_train = self._validate_x(x)
        return self

    def get_context(self, x: Tensor):
        """Get context used for observation."""
        return self._theta_train, self._x_train

    def _validate_x(self, x: Tensor):
        """Validate x."""
        if x is None:
            raise NotImplementedError("Setting a default x is not yet supported.")

        x = x.unsqueeze(0) if x.ndim == 1 else x
        assert x.ndim == 2, "x must be a 2D tensor."
        if self._x_train is not None:
            assert (
                x.shape[1] == self._x_train.shape[1]
            ), "The number of features in x must match the training data."
        return x

    def _validate_theta(self, theta: Tensor):
        """Validate theta."""
        theta = theta.unsqueeze(0) if theta.ndim == 1 else theta
        assert theta.ndim == 2, "theta must be a 2D tensor."
        if self._theta_train is not None:
            assert (
                theta.shape[1] == self._theta_train.shape[1]
            ), "The number of features in theta must match the training data."
        return theta

    def _sample(
        self,
        sampling_batch_size: int,
        x: Tensor,
        repeat_x: bool = True,
        with_log_prob: bool = False,
        eps=1e-15,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Sample from the posterior p(theta | x)"""

        if repeat_x:
            samples_batch = x.repeat(sampling_batch_size, 1)
        else:
            samples_batch = x
            sampling_batch_size = x.shape[0]

        # Create joint dataset of observations and parameters
        theta_context, x_context = self.get_context(x)
        joint_data = torch.cat([x_context, theta_context], dim=1)
        dim_x = x_context.shape[1]
        dim_theta = theta_context.shape[1]

        log_probs_batch = torch.zeros(sampling_batch_size) if with_log_prob else None
        # Sequentially predict each parameter dimension
        for param_idx in range(dim_theta):
            # Fit model on joint data up to current parameter
            features_end = dim_x + param_idx
            target_idx = dim_x + param_idx

            self._model.fit(joint_data[:, :features_end], joint_data[:, target_idx])

            # Generate samples for current parameter
            pred_dist = self._model.predict(
                samples_batch, output_type="full", quantiles=[]
            )
            param_samples = pred_dist["criterion"].sample(pred_dist["logits"])

            if with_log_prob:
                dim_log_prob = -pred_dist["criterion"](
                    pred_dist["logits"], param_samples
                )

                dim_log_prob = torch.where(
                    dim_log_prob == float("-inf"),
                    torch.log(torch.tensor(eps)),
                    dim_log_prob,
                )

                log_probs_batch += dim_log_prob

            # Append new parameter samples
            samples_batch = torch.cat([samples_batch, param_samples[:, None]], dim=1)

            # Clear cache to avoid memory issues, otherwise can segmentation fault
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return samples_batch[:, dim_x:], log_probs_batch

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),
        x: Tensor = None,
        max_sampling_batch_size: int = 10_000,
        with_log_prob: bool = False,
        eps=1e-15,
        max_iter_rejection: int | None = None,
        show_progress_bars: bool = False,  # TODO deal with this
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample from the posterior p(theta | x)

        Args:
            sample_shape: Desired shape of samples
            x: Observations to condition on
            max_sampling_batch_size: Maximum batch size for sampling
        """

        # standard input checks
        if self.embedding_net:
            x = x.reshape(-1, *self.x_shape)
            x = self.embedding_net(x)

        x = self._validate_x(x)

        if x.shape[0] > 1:
            raise ValueError(
                ".sample() supports only `batchsize == 1`. If you intend "
                "to sample multiple observations, use `.sample_batched()`. "
            )

        def proposal_fn(max_sampling_batch_size, **kwargs):
            """Generate proposal samples using the original sampling method"""
            return self._sample(
                sampling_batch_size=max_sampling_batch_size,
                x=x,
                repeat_x=True,
                with_log_prob=with_log_prob,
                eps=eps,
            )

        # handles rejection and batching
        ## samples, log_probs, _ar = accept_reject_sample(
        ##     proposal=proposal_fn,
        ##     accept_reject_fn=self._within_support,
        ##     num_samples=torch.Size(sample_shape).numel(),
        ##     show_progress_bars=self.show_progress_bars,
        ##     max_sampling_batch_size=max_sampling_batch_size,
        ##     proposal_sampling_kwargs={},
        ##     max_iter_rejection=max_iter_rejection,
        ## )
        ## if with_log_prob:
        ##     return samples, log_probs
        ## else:
        ##     return samples

    def sample_batched(
        self,
        x: Tensor,
        sample_shape: torch.Size = torch.Size(),
        max_sampling_batch_size: int = 10_000,
    ):
        """Sample from the posterior p(theta | x) in a batched manner.

        Args:
            x: Observations to condition on
            sample_shape: Desired shape of samples
            max_sampling_batch_size: Maximum batch size for sampling
        """
        raise NotImplementedError

    def log_prob(
        self,
        theta: Tensor,
        x: Tensor,
        max_sampling_batch_size: int = 10_000,
        mode="autoregressive",
        eps=1e-15,
        **ratio_kwargs,
    ):
        """Calculate log probability of parameters p(theta | x)

        Args:
            theta: Parameters to evaluate
            x: Observations to condition on
            mode: Method to use for log probability calculation
        """
        if self.embedding_net:
            x = x.reshape(-1, *self.x_shape)
            x = self.embedding_net(x)

        theta = self._validate_theta(theta)
        x = self._validate_x(x)

        log_probs = torch.zeros(theta.shape[0])
        for i in range(0, theta.shape[0], max_sampling_batch_size):
            if mode == "autoregressive":
                log_probs[i : i + max_sampling_batch_size] = (
                    self._autoregressive_log_prob(
                        theta[i : i + max_sampling_batch_size],
                        x,
                        eps=eps,
                    )
                )
            elif mode == "ratio_based":
                log_probs[i : i + max_sampling_batch_size] = self._ratio_based_log_prob(
                    theta[i : i + max_sampling_batch_size],
                    x,
                    eps=eps,
                    **ratio_kwargs,
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")

        return log_probs

    def log_prob_batched(self, theta: Tensor, x: Tensor):
        """Calculate log probability of parameters in a batched manner."""
        # NOTE: Will only support autoregressive log prob
        raise NotImplementedError

    def _autoregressive_log_prob(
        self,
        theta: Tensor,
        x: Tensor = None,
        repeat_x: bool = True,
        eps: float = 1e-15,
    ) -> Tensor:
        """Calculate log probability of parameters p(theta | x)

        Args:
            theta: Parameters to evaluate
            x: Observations to condition on
        """
        # TODO leakage correction is not implemented, can have density outside of prior support

        # NOTE: repeat_x flag crucial for unconditional models
        num_samples = theta.shape[0]
        if repeat_x:
            x_batch = x.repeat(num_samples, 1)
        else:
            x_batch = x

        assert x_batch.shape[0] == num_samples
        test_joint = torch.cat([x_batch, theta], dim=1)

        # Create training joint data (observations and parameters)
        theta_context, x_context = self.get_context(x)
        joint_data = torch.cat([x_context, theta_context], dim=1)
        dim_x = x_context.shape[1]
        dim_theta = theta_context.shape[1]

        # Initialize log probability tensor
        log_prob = torch.zeros(num_samples)

        # Sequentially compute log prob for each parameter dimension
        for param_idx in range(dim_theta):
            # Fit model on joint data up to current parameter
            features_end = dim_x + param_idx
            target_idx = dim_x + param_idx

            self._model.fit(joint_data[:, :features_end], joint_data[:, target_idx])

            # Get prediction distribution
            pred_dist = self._model.predict(
                test_joint[:, :features_end], output_type="full", quantiles=[]
            )

            # Compute log probability for this dimension
            dim_log_prob = -pred_dist["criterion"](
                pred_dist["logits"], test_joint[:, target_idx]
            )

            # Handle -inf values
            dim_log_prob = torch.where(
                dim_log_prob == float("-inf"),
                torch.log(torch.tensor(eps)),
                dim_log_prob,
            )

            # Add to total log probability
            log_prob += dim_log_prob

        return log_prob

    def _ratio_based_log_prob(
        self,
        theta: Tensor,
        x: Tensor = None,
        num_posterior_samples: int = 5000,
        boundary_padding: float = 0.1,
        reuse_estimator_if_possible: bool = True,
        eps: float = 1e-15,
    ) -> Tensor:
        """Calculate log probability of parameters using ratio-based method.

        Args:
            theta: Parameters to evaluate log prob for
            x: Observation to condition on
            num_posterior_samples: Number of posterior samples to generate (should be not more than 5000 under the normal limits)
            boundary_padding: Padding for uniform reference distribution
            eps: Small constant to avoid numerical issues
            reuse_estimator_if_possible: Reuse classifier if obsvervation was seen before.
                This will ignore the `num_posterior_samples` and `boundary_padding` arguments.
        """

        # initialize classifier if not already initialized
        if self._model_classifier is None:
            self._model_classifier = DensityRatioWrapper(**self.classifier_init_kwargs)

        # get actual context, might be different for filtering
        # NOTE: We need to be rather careful here if the sorting from the filter is not fully deterimistic/"stable"
        theta_context, x_context = self.get_context(x)
        if not reuse_estimator_if_possible or self._model_classifier.refit_necessary(
            x,
            x_context,
            theta_context,
            num_posterior_samples,
            boundary_padding,
        ):
            posterior_samples = self.sample(
                sample_shape=torch.Size([num_posterior_samples]), x=x
            )
            self._model_classifier.fit(
                x, posterior_samples, boundary_padding, x_context, theta_context
            )

        log_probs = self._model_classifier.ratio_log_probs(theta, eps)

        return log_probs

    def _get_classifier_bounds(self):
        """Get the bounds of the classifier if they exist."""
        if self._model_classifier is None:
            return None, None
        return (
            self._model_classifier._padded_dim_min,
            self._model_classifier._padded_dim_max,
        )

    def _within_support(self, theta: Tensor) -> Tensor:
        """Check if samples are within prior support.

        First attempts to use the support property of the prior distribution.
        Falls back to checking if log probability is finite if support check
        is not available.

        Args:
            theta: Parameter samples to check

        Returns:
            Tensor of bools indicating whether each sample is within support
        """
        try:
            sample_check = self.prior.support.check(theta)
            if sample_check.shape == theta.shape:
                sample_check = torch.all(sample_check, dim=-1)
            return sample_check
        except (NotImplementedError, AttributeError):
            return torch.isfinite(self.prior.log_prob(theta))


class DensityRatioWrapper:
    """Wrapper class for the density ratio based log probability calculation.
    This enables reuse of the classifier if the observation (and other parameters) are the same, which will be significantly faster.
    """

    def __init__(self, **init_kwargs):
        super().__init__()
        self._classifier = TabPFNClassifier(**init_kwargs)

        self._ratio_log_prob_x = None
        self._num_posterior_samples = None
        self._boundary_padding = None

        self._padded_dim_min = None
        self._padded_dim_max = None
        self._uniform_log_prob = None

    def fit(
        self,
        x: Tensor,
        posterior_samples: Tensor,
        boundary_padding: float,
        x_context: Tensor,
        theta_context: Tensor,
    ):
        """Fit the classifier on the given data."""

        dim_min = posterior_samples.min(dim=0).values
        dim_max = posterior_samples.max(dim=0).values
        dim_length = dim_max - dim_min
        padded_dim_min = dim_min - boundary_padding * dim_length
        padded_dim_max = dim_max + boundary_padding * dim_length
        padded_dim_length = padded_dim_max - padded_dim_min

        uniform_log_prob = -torch.log(padded_dim_length).sum()

        # Generate uniform samples matching the shape of training data
        uniform_samples = (
            torch.rand_like(posterior_samples) * padded_dim_length + padded_dim_min
        )

        # Prepare classifier training data
        num_posterior_samples = posterior_samples.shape[0]
        train_X_classifier = torch.cat([uniform_samples, posterior_samples], dim=0)
        train_y_classifier = torch.cat(
            [torch.zeros(num_posterior_samples), torch.ones(num_posterior_samples)],
            dim=0,
        )

        self._ratio_log_prob_x = x
        self._num_posterior_samples = num_posterior_samples
        self._boundary_padding = boundary_padding
        self._x_context = x_context
        self._theta_context = theta_context

        self._padded_dim_min = padded_dim_min
        self._padded_dim_max = padded_dim_max
        self._uniform_log_prob = uniform_log_prob
        self._classifier.fit(train_X_classifier, train_y_classifier)

    def refit_necessary(
        self,
        x: Tensor,
        x_context: Tensor,
        theta_context: Tensor,
        num_posterior_samples: int,
        boundary_padding: float,
    ):
        """Reuse classifier if possible. Check whether refitting is necessary."""
        return (
            self._ratio_log_prob_x is None
            or not torch.allclose(x, self._ratio_log_prob_x)
            or not x_context.shape == self._x_context.shape
            or not torch.allclose(x_context, self._x_context)
            or not theta_context.shape == self._theta_context.shape
            or not torch.allclose(theta_context, self._theta_context)
            or not num_posterior_samples == self._num_posterior_samples
            or not math.isclose(boundary_padding, self._boundary_padding)
        )

    def ratio_log_probs(self, theta: Tensor, eps=1e-15):
        """Predict probabilities for the given theta."""
        mask = torch.all(
            (theta >= self._padded_dim_min) & (theta <= self._padded_dim_max), dim=1
        )

        log_probs = torch.full(
            (theta.shape[0],),
            self._uniform_log_prob
            + torch.log(torch.tensor(eps))
            - torch.log(torch.tensor(1 + eps)),
        )

        if mask.any():
            classifier_probs = self._classifier.predict_proba(theta[mask])
            log_probs[mask] = (
                self._uniform_log_prob
                + torch.log(torch.tensor(classifier_probs[:, 1] + eps))
                - torch.log(torch.tensor(classifier_probs[:, 0] + eps))
            )

        return log_probs
