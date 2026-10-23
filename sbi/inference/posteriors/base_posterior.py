# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Self,
    Tuple,
    Union,
)
from warnings import warn

import torch
import torch.distributions.transforms as torch_tf
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

if TYPE_CHECKING:
    from sbi.diagnostics.observation import (
        ObservationDiagnostic,
        ReferenceObservationStats,
    )

from sbi.inference.potentials.base_potential import (
    BasePotential,
    CustomPotential,
    CustomPotentialWrapper,
)
from sbi.sbi_types import Array, Shape, TorchTransform
from sbi.utils.sbiutils import (
    gradient_ascent,
    load_with_version,
    nan_tolerant_input_net,
    prior_support_is_bounded,
    save_with_version,
)
from sbi.utils.torchutils import (
    assert_all_finite,
    canonical_device,
    ensure_theta_batched,
    infer_tensor_device,
    process_device,
)
from sbi.utils.user_input_checks import process_x
from sbi.utils.user_input_checks_utils import move_distribution_to_device


class NeuralPosterior:
    r"""Posterior :math:`p(\theta|x)` with `log_prob()` and `sample()` methods.

    All inference methods in sbi train a neural network which is then used to obtain
    the posterior distribution. The `NeuralPosterior` class wraps the trained network
    such that one can directly evaluate the (unnormalized) log probability and draw
    samples from the posterior.
    """

    def __init__(
        self,
        potential_fn: Union[BasePotential, CustomPotential],
        theta_transform: Optional[TorchTransform] = None,
        device: Optional[Union[str, torch.device]] = None,
        x_shape: Optional[torch.Size] = None,
        reference_statistics: Optional[ReferenceObservationStats] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples. Must be a
                `BasePotential` or a `Callable` which takes `theta` and `x_o` as inputs.
            theta_transform: Transformation that will be applied during sampling.
                Allows to perform, e.g. MCMC in unconstrained space.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Deprecated, should not be passed.
            reference_statistics: Optional precomputed reference statistics for
                automated out-of-distribution observation diagnostics.
        """
        if x_shape is not None:
            warn(
                "x_shape is not None. However, passing x_shape to the `Posterior` is "
                "deprecated and will be removed in a future release of `sbi`.",
                stacklevel=2,
            )

        # Wrap custom potential functions to adhere to the `BasePotential` interface.
        if not isinstance(potential_fn, BasePotential):
            potential_device = "cpu" if device is None else device
            potential_fn = CustomPotentialWrapper(
                potential_fn, prior=None, x_o=None, device=potential_device
            )

        self._device = process_device(potential_fn.device if device is None else device)

        self.potential_fn = potential_fn

        if theta_transform is None:
            self.theta_transform = torch_tf.IndependentTransform(
                torch_tf.identity_transform, reinterpreted_batch_ndims=1
            )
        else:
            self.theta_transform = theta_transform

        self._map = None
        self._leakage_cache: Optional[Tuple[Tensor, Tensor]] = None
        self._purpose = ""

        # Retain existing reference statistics if reinstantiating during .to(device)
        self._reference_simulations: Optional[Tensor] = getattr(
            self, "_reference_simulations", None
        )
        self._reference_embedding_net: Optional[nn.Module] = getattr(
            self, "_reference_embedding_net", None
        )
        self._reference_max_samples: int = getattr(
            self, "_reference_max_samples", 10_000
        )
        self._reference_statistics_embedded: Optional[ReferenceObservationStats] = (
            getattr(self, "_reference_statistics_embedded", None)
        )

        existing_ref = getattr(self, "_reference_statistics", None)
        if reference_statistics is not None:
            if getattr(reference_statistics, "is_embedded", False):
                self._reference_statistics_embedded = reference_statistics.to(
                    self._device
                )
                self._reference_statistics = None
            else:
                self._reference_statistics = reference_statistics.to(self._device)
        elif existing_ref is not None:
            self._reference_statistics = existing_ref.to(self._device)
        else:
            self._reference_statistics = None

        # If the sampler interface (#573) is used, the user might have passed `x_o`
        # already to the potential function builder. If so, this `x_o` will be used
        # as default x.
        x_o = self.potential_fn.return_x_o()
        if x_o is not None:
            self._assert_finite_x(x_o)
        self._x = x_o

    def _assert_finite_x(self, x: Tensor) -> None:
        """Raise if `x` contains Inf, or NaN that the net embedding `x` does not
        accept (e.g., a NaN-padding-aware `PermutationInvariantEmbedding`)."""
        x = torch.as_tensor(x)
        net = nan_tolerant_input_net(self.potential_fn.x_embedding_net)
        assert_all_finite(x, "Observed data x_o", allow_nan=net is not None)
        validate_nan_input = getattr(net, "validate_nan_input", None)
        if validate_nan_input is not None and torch.isnan(x).any():
            validate_nan_input(x)

    def potential(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""Evaluates $\theta$ under the potential that is used to sample the posterior.

        The potential is the unnormalized log-probability of $\theta$ under the
        posterior.

        Args:
            theta: Parameters $\theta$.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
        """
        self.potential_fn = self.potential_fn.bind(self._x_else_default_x(x))

        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device), track_gradients=track_gradients
        )

    @abstractmethod
    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        r"""Draw samples from the approximate posterior distribution $p(\theta|x)$.

        Args:
            sample_shape: Shape of samples to draw.
            x: Conditioning observation $x_o$. If not provided, uses the default
                `x` set via `.set_default_x()`.
            show_progress_bars: Whether to show a progress bar during sampling.
            **kwargs: Additional keyword arguments passed to the specific
                posterior's sampling method. See the docstring of the specific
                posterior class for available options.

        Returns:
            Samples from the posterior with shape `(*sample_shape, *theta_shape)`.
        """
        pass

    @abstractmethod
    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        show_progress_bars: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        r"""Draw samples from the posteriors for a batch of different xs.

        Given a batch of observations `[x_1, ..., x_B]`, this method samples from
        posteriors $p(\theta|x_1), \ldots, p(\theta|x_B)$ in a vectorized manner.

        Args:
            sample_shape: Shape of samples to draw for each observation.
            x: Batch of observations with shape `(batch_dim, *event_shape_x)`.
            show_progress_bars: Whether to show a progress bar during sampling.
            **kwargs: Additional keyword arguments passed to the specific
                posterior's sampling method.

        Returns:
            Samples with shape `(*sample_shape, batch_dim, *theta_shape)`.
        """
        pass

    @property
    def default_x(self) -> Optional[Tensor]:
        """Return default x used by `.sample(), .log_prob` as conditioning context."""
        return self._x

    @default_x.setter
    def default_x(self, x: Tensor) -> None:
        """See `set_default_x`."""
        self.set_default_x(x)

    def set_default_x(self, x: Tensor, warn_if_ood: bool = False) -> "NeuralPosterior":
        """Set new default x for `.sample(), .log_prob` to use as conditioning context.

        Reset the MAP stored for the old default x if applicable.

        This is a pure convenience to avoid having to repeatedly specify `x` in calls to
        `.sample()` and `.log_prob()` - only $\theta$ needs to be passed.

        This convenience is particularly useful when the posterior is focused, i.e.
        has been trained over multiple rounds to be accurate in the vicinity of a
        particular `x=x_o` (you can check if your posterior object is focused by
        printing it).

        NOTE: this method is chainable, i.e. will return the NeuralPosterior object so
        that calls like `posterior.set_default_x(my_x).sample(mytheta)` are possible.

        Args:
            x: The default observation to set for the posterior $p(\theta|x)$.
            warn_if_ood: If True, evaluates whether `x` is out-of-distribution
                relative to the reference simulations and issues a `UserWarning` if
                it falls below the threshold. Default is False.

        Returns:
            `NeuralPosterior` that will use a default `x` when not explicitly passed.
        """
        x = process_x(x, x_event_shape=None)
        self._assert_finite_x(x)

        if warn_if_ood:
            diag = self.diagnose_observation(x)
            if diag.warning is not None:
                warn(diag.warning, UserWarning, stacklevel=2)

        self._x = x.to(self._device)
        self._map = None
        return self

    def set_reference_data(
        self,
        x: Tensor,
        embedding_net: Optional[nn.Module] = None,
        max_reference_samples: int = 10_000,
    ) -> Self:
        r"""Set reference simulation data to enable automated OOD diagnostics.

        Reference statistics are constructed lazily upon first diagnostic query
        or warning check to avoid unnecessary covariance and decomposition overhead
        during model construction.

        Args:
            x: Reference simulations with shape `(N, *event_shape)`.
            embedding_net: Optional neural embedding net. If None, resolves from
                `self.potential_fn.x_embedding_net`.
            max_reference_samples: Maximum number of reference samples to retain for
                distance score ranking.

        Returns:
            `NeuralPosterior` (self) for method chaining.
        """
        self._reference_simulations = x
        self._reference_embedding_net = embedding_net
        self._reference_max_samples = max_reference_samples
        self._reference_statistics = None
        self._reference_statistics_embedded = None
        return self

    def set_reference_statistics(self, stats: "ReferenceObservationStats") -> Self:
        """Directly set precomputed reference statistics on the posterior.

        Args:
            stats: Precomputed `ReferenceObservationStats` object.

        Returns:
            `NeuralPosterior` (self) for method chaining.
        """
        if getattr(stats, "is_embedded", False):
            self._reference_statistics_embedded = stats.to(self._device)
        else:
            self._reference_statistics = stats.to(self._device)
        return self

    def _build_reference_statistics(
        self, space: Literal["observation", "embedded"] = "observation"
    ) -> Optional["ReferenceObservationStats"]:
        """Construct reference statistics for the requested space lazily."""
        if self._reference_simulations is None:
            return None

        from sbi.diagnostics.observation import (
            ReferenceObservationStats,
            _get_module_device,
        )

        net = (
            self._reference_embedding_net
            if self._reference_embedding_net is not None
            else getattr(self.potential_fn, "x_embedding_net", None)
        )

        try:
            if space == "embedded":
                if self._reference_statistics_embedded is None:
                    if net is None:
                        raise ValueError(
                            "Cannot construct embedded reference statistics: no "
                            "observation embedding network found on posterior."
                        )
                    target_device = _get_module_device(
                        net, self._reference_simulations.device
                    )
                    orig_mode = net.training
                    try:
                        net.eval()
                        with torch.no_grad():
                            x_emb = net(
                                self._reference_simulations.to(target_device)
                            ).to(self._reference_simulations.device)
                    finally:
                        net.train(orig_mode)

                    self._reference_statistics_embedded = (
                        ReferenceObservationStats.from_simulations(
                            x_emb,
                            embedding_net=net,
                            is_embedded=True,
                            max_reference_samples=self._reference_max_samples,
                        ).to(self._device)
                    )
                return self._reference_statistics_embedded
            else:
                if self._reference_statistics is None:
                    self._reference_statistics = (
                        ReferenceObservationStats.from_simulations(
                            self._reference_simulations,
                            embedding_net=net,
                            is_embedded=False,
                            max_reference_samples=self._reference_max_samples,
                        ).to(self._device)
                    )
                return self._reference_statistics
        except Exception:
            return None

    @property
    def reference_statistics(self) -> Optional["ReferenceObservationStats"]:
        """Reference observation statistics attached to this posterior, if any."""
        if (
            self._reference_statistics is None
            and self._reference_simulations is not None
        ):
            self._build_reference_statistics(space="observation")
        return self._reference_statistics or self._reference_statistics_embedded

    @property
    def has_reference_statistics(self) -> bool:
        """Whether reference observation statistics or simulations are attached."""
        return (
            self._reference_statistics is not None
            or self._reference_statistics_embedded is not None
            or self._reference_simulations is not None
        )

    def diagnose_observation(
        self,
        x: Optional[Tensor] = None,
        *,
        threshold: float = 0.01,
        space: Literal["observation", "embedded"] = "observation",
        x_ref: Optional[Tensor] = None,
    ) -> "ObservationDiagnostic":
        r"""Diagnose whether an observation is supported by the training distribution.

        Args:
            x: Observed data $x_o$. If None, default $x$ (set via `.set_default_x()`)
                is used.
            threshold: Empirical percentile threshold below which the observation is
                flagged as out-of-distribution (default: 0.01, or bottom 1%).
            space: Space in which to perform the diagnostic (`"observation"` or
                `"embedded"`). When `"embedded"`, observations are projected through
                the estimator's embedding net.
            x_ref: Optional explicit reference simulations to compare against. If None,
                uses the reference statistics stored during training or via
                `.set_reference_data()`.

        Returns:
            `ObservationDiagnostic` containing `is_in_distribution`,
            `empirical_percentile`, `mahalanobis_distance`, and `warning`.
        """
        if x is None:
            if self.default_x is None:
                raise ValueError(
                    "Context `x` needed when a default has not been set. "
                    "If you'd like to have a default, use `.set_default_x()`."
                )
            x_to_check = self.default_x
        else:
            x_to_check = x

        embedding_net = (
            getattr(self.potential_fn, "x_embedding_net", None)
            if space == "embedded"
            else None
        )

        from sbi.diagnostics.observation import (
            ObservationDiagnostic,
            diagnose_observation,
        )

        if x_ref is None:
            if space == "embedded":
                stats = self._reference_statistics_embedded
                if stats is None and self._reference_simulations is not None:
                    stats = self._build_reference_statistics(space="embedded")
            else:
                stats = self._reference_statistics
                if stats is None and self._reference_simulations is not None:
                    stats = self._build_reference_statistics(space="observation")

            if stats is None and self._reference_simulations is not None:
                return ObservationDiagnostic(
                    is_in_distribution=None,
                    empirical_percentile=None,
                    mahalanobis_distance=None,
                    warning=(
                        "Reference observation statistics could not be computed from "
                        "stored training simulations."
                    ),
                    threshold=threshold,
                    space=space,
                )
        else:
            stats = None

        return diagnose_observation(
            x_o=x_to_check,
            x_ref=x_ref,
            reference_stats=stats,
            threshold=threshold,
            space=space,
            embedding_net=embedding_net,
        )

    def _move_reference_stats(self, device: Union[str, torch.device]) -> None:
        """Helper to move reference statistics to the specified device."""
        if self._reference_statistics is not None:
            self._reference_statistics = self._reference_statistics.to(device)
        if self._reference_statistics_embedded is not None:
            self._reference_statistics_embedded = (
                self._reference_statistics_embedded.to(device)
            )
        if self._reference_simulations is not None:
            self._reference_simulations = self._reference_simulations.to(device)

    def _x_else_default_x(self, x: Optional[Array]) -> Tensor:
        if x is not None:
            # New x, reset posterior sampler.
            self._posterior_sampler = None
            x = process_x(x, x_event_shape=None)
            self._assert_finite_x(x)

            return x.to(self._device)
        elif self.default_x is None:
            raise ValueError(
                "Context `x` needed when a default has not been set."
                "If you'd like to have a default, use the `.set_default_x()` method."
            )
        else:
            return self.default_x

    def _cached_leakage_factor(
        self,
        x: Tensor,
        prior: Any,
        estimate_fn: Callable[[], Tensor],
        force_update: bool = False,
        use_cache: bool = True,
    ) -> Tensor:
        """Return the leakage correction factor saved for `x`, or estimate it.

        The factor is 1 for priors with unbounded support. Otherwise, one factor is
        saved, for the last `x`, unless `use_cache=False`.
        """
        if prior_support_is_bounded(prior) is False:
            return torch.ones((), device=self._device)
        if self._leakage_cache is not None and use_cache and not force_update:
            cached_x, factor = self._leakage_cache
            if cached_x.shape == x.shape and torch.allclose(
                cached_x, x.to(cached_x.device), rtol=0, atol=0, equal_nan=True
            ):
                return factor
        factor = estimate_fn()
        if use_cache:
            self._leakage_cache = (x.detach().clone(), factor)
        return factor

    def _calculate_map(
        self,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "posterior",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
    ) -> Tensor:
        """Calculates the maximum-a-posteriori estimate (MAP).
        See `map()` method of child classes for docstring.
        """

        if init_method == "posterior":
            inits = self.sample((num_init_samples,))
        elif init_method == "proposal":
            inits = self.proposal.sample((num_init_samples,))  # type: ignore
        elif isinstance(init_method, Tensor):
            inits = init_method
        else:
            raise ValueError

        return gradient_ascent(
            potential_fn=self.potential_fn,
            inits=inits,
            theta_transform=self.theta_transform,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
        )[0]

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
        force_update: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The MAP is obtained by running gradient
        ascent from a given number of starting positions (samples from the posterior
        with the highest log-probability). After the optimization is done, we select the
        parameter set that has the highest log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
            x: Deprecated - use `.set_default_x()` prior to `.map()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a
                tensor, the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether to show a progressbar during sampling from
                the posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.
        Returns:
            The MAP estimate.
        """
        if x is not None:
            raise ValueError(
                "Passing `x` directly to `.map()` has been deprecated."
                "Use `.self_default_x()` to set `x`, and then run `.map()` "
            )

        if self.default_x is None:
            raise ValueError(
                "Default `x` has not been set."
                "To set the default, use the `.set_default_x()` method."
            )

        if self._map is None or force_update:
            self.potential_fn = self.potential_fn.bind(self.default_x)
            self._map = self._calculate_map(
                num_iter=num_iter,
                num_to_optimize=num_to_optimize,
                learning_rate=learning_rate,
                init_method=init_method,
                num_init_samples=num_init_samples,
                save_best_every=save_best_every,
                show_progress_bars=show_progress_bars,
            )
        return self._map

    def __repr__(self):
        desc = f"""{self.__class__.__name__} sampler for potential_fn=<{
            self.potential_fn.__class__.__name__
        }>"""
        return desc

    def __str__(self):
        desc = f"Posterior p(θ|x) of type {self.__class__.__name__}. {self._purpose}"
        return desc

    def save(self, filename: Union[str, Path]) -> None:
        """Save the posterior to a file. Load it with `load()`.

        Args:
            filename: Path to the file.
        """
        save_with_version(self, filename)

    @classmethod
    def load(
        cls,
        filename: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Self:
        """Load a posterior saved with `save()`.

        The file is unpickled, which can execute arbitrary code. Only load trusted
        files.

        Args:
            filename: Path to the file.
            map_location: Device to load the posterior on, e.g. `"cpu"` for a
                posterior saved on a GPU. By default, the device it was saved on.

        Returns:
            The loaded posterior.
        """
        return load_with_version(filename, cls, map_location)

    def __getstate__(self) -> Dict:
        """Returns the state of the object that is supposed to be pickled.

        Returns:
            Dictionary containing the state.
        """
        return self.__dict__

    def __setstate__(self, state_dict: Dict):
        """Sets the state when being loaded from pickle.

        `torch.load(..., map_location=...)` remaps the tensors but not the stored device
        strings. If the restored tensors live on another device than `_device` claims,
        the device strings of the posterior, its potential and its prior are updated.

        Args:
            state_dict: State to be restored.
        """
        # Posteriors pickled by older sbi versions can miss this key.
        state_dict.setdefault("_leakage_cache", None)
        state_dict.setdefault("_reference_statistics", None)
        self.__dict__ = state_dict

        actual_device = infer_tensor_device(self)
        if actual_device is None or canonical_device(actual_device) == canonical_device(
            self._device
        ):
            return

        # The tensors already live on `actual_device`. Only the device strings, and the
        # distributions that cache one, must follow.
        self._device = actual_device
        if hasattr(self, "device"):
            self.device = actual_device
        if self._reference_statistics is not None:
            self._reference_statistics = self._reference_statistics.to(actual_device)
        shared_prior = getattr(self.potential_fn, "prior", None)
        self.potential_fn.to(actual_device)  # type: ignore[union-attr]
        for attr in ("prior", "_prior", "proposal"):
            value = getattr(self, attr, None)
            if value is shared_prior and value is not None:
                # The potential moved this object already. Keep sharing it.
                setattr(self, attr, self.potential_fn.prior)  # type: ignore[union-attr]
            elif isinstance(value, Distribution):
                setattr(self, attr, move_distribution_to_device(value, actual_device))
