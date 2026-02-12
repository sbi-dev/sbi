# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import copy
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm

from sbi.inference.posteriors.base_posterior import NeuralPosterior

if TYPE_CHECKING:
    from sbi.inference.posteriors.posterior_parameters import VIPosteriorParameters
from sbi.inference.potentials.base_potential import BasePotential, CustomPotential
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
from sbi.neural_nets.estimators.zuko_flow import ZukoUnconditionalFlow
from sbi.neural_nets.factory import ZukoFlowType
from sbi.neural_nets.net_builders.flow import (
    build_zuko_flow,
    build_zuko_unconditional_flow,
)
from sbi.samplers.vi.vi_divergence_optimizers import get_VI_method
from sbi.samplers.vi.vi_quality_control import get_quality_metric
from sbi.samplers.vi.vi_utils import (
    LearnableGaussian,
    TransformedZukoFlow,
    adapt_variational_distribution,
    check_variational_distribution,
    make_object_deepcopy_compatible,
)
from sbi.sbi_types import (
    Shape,
    TorchDistribution,
    TorchTensor,
    TorchTransform,
)
from sbi.utils.sbiutils import mcmc_transform
from sbi.utils.torchutils import atleast_2d_float32_tensor, ensure_theta_batched
from sbi.utils.user_input_checks_utils import move_distribution_to_device

# Supported Zuko flow types for VI (lowercase names)
_ZUKO_FLOW_TYPES = {"maf", "nsf", "naf", "unaf", "nice", "sospf"}

# Type for supported variational family strings
VariationalFamily = Literal[
    "maf", "nsf", "naf", "unaf", "nice", "sospf", "gaussian", "gaussian_diag"
]

# Type for the q parameter in VIPosterior
QType = Union[VariationalFamily, Distribution, "VIPosterior", Callable]


class VIPosterior(NeuralPosterior):
    r"""Provides VI (Variational Inference) to sample from the posterior.

    SNLE or SNRE train neural networks to approximate the likelihood (or likelihood
    ratios). ``VIPosterior`` allows learning a tractable variational posterior
    :math:`q(\theta)` which approximates the true posterior
    :math:`p(\theta|x_o)`. After this second training stage, we can produce
    approximate posterior samples by sampling from :math:`q` at no additional cost.

    For additional information, see [1]_ and [2]_.

    References
    ----------

    .. [1] Glöckler, M., Deistler, M., & Macke, J. (2022).
        Variational methods for simulation-based inference.
        https://openreview.net/forum?id=kZ0UYdhqkNY

    .. [2] Wiqvist, S., Frellsen, J., & Picchini, U. (2021).
        Sequential Neural Posterior and Likelihood Approximation.
        https://arxiv.org/abs/2102.06522
    """

    def __init__(
        self,
        potential_fn: Union[BasePotential, CustomPotential],
        prior: Optional[TorchDistribution] = None,  # type: ignore
        q: QType = "maf",
        theta_transform: Optional[TorchTransform] = None,
        vi_method: Literal["rKL", "fKL", "IW", "alpha"] = "rKL",
        device: Union[str, torch.device] = "cpu",
        x_shape: Optional[torch.Size] = None,
        parameters: Optional[Iterable] = None,
        modules: Optional[Iterable] = None,
        num_transforms: int = 5,
        hidden_features: int = 50,
        z_score_theta: Literal["none", "independent", "structured"] = "independent",
        z_score_x: Literal["none", "independent", "structured"] = "independent",
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples. Must be a
                `BasePotential` or a `CustomPotential`.
            prior: This is the prior distribution. Note that this is only
                used to check/construct the variational distribution or within some
                quality metrics. Please make sure that this matches with the prior
                within the potential_fn. If `None` is given, we will try to infer it
                from potential_fn or q, if this fails we raise an Error.
            q: Variational distribution, either string, `Distribution`, or a
                `VIPosterior` object. This specifies a parametric class of distribution
                over which the best possible posterior approximation is searched. For
                string input, we support normalizing flows [maf, nsf, naf, unaf, nice,
                sospf] via Zuko, and Gaussian families [gaussian, gaussian_diag].
                You can also specify your own variational family by passing a
                `torch.distributions.Distribution`. Additionally, we allow a `Callable`
                with signature `(event_shape: torch.Size, link_transform:
                TorchTransform, device: str) -> Distribution` for custom flow
                configurations. The
                callable should return a distribution with `sample()` and `log_prob()`
                methods. If q is already a `VIPosterior`, then the arguments will be
                copied from it (relevant for multi-round training).
            theta_transform: Maps form prior support to unconstrained space. The
                inverse is used here to ensure that the posterior support is equal to
                that of the prior.
            vi_method: This specifies the variational methods which are used to fit q to
                the posterior. We currently support [rKL, fKL, IW, alpha]. Note that
                some of the divergences are `mode seeking` i.e. they underestimate
                variance and collapse on multimodal targets (`rKL`, `alpha` for alpha >
                1) and some are `mass covering` i.e. they overestimate variance but
                typically cover all modes (`fKL`, `IW`, `alpha` for alpha < 1).
            device: Training device, e.g., `cpu`, `cuda` or `cuda:0`. We will ensure
                that all other objects are also on this device.
            x_shape: Deprecated, should not be passed.
            parameters: List of parameters of the variational posterior. This is only
                required for user-defined q i.e. if q does not have a `parameters`
                attribute.
            modules: List of modules of the variational posterior. This is only
                required for user-defined q i.e. if q does not have a `modules`
                attribute.
            num_transforms: Number of transforms in the normalizing flow. Used for
                both single-x VI (when q is a string flow type) and amortized VI.
            hidden_features: Hidden layer size in flow networks. Used for both
                single-x VI and amortized VI.
            z_score_theta: Method for z-scoring θ (parameters). One of "none",
                "independent", "structured". Used for both single-x VI and amortized
                VI. Use "structured" for parameters with correlations.
            z_score_x: Method for z-scoring x (conditioning observation). One of
                "none", "independent", "structured". Only used for amortized VI
                (train_amortized). Use "structured" for structured data like images.
        """
        super().__init__(potential_fn, theta_transform, device, x_shape=x_shape)

        # Especially the prior may be on another device -> move it...
        self._device = device
        self.theta_transform = theta_transform
        self.x_shape = x_shape
        self.potential_fn.device = device
        self.potential_fn.to(device)

        # Get prior and previous builds
        if prior is not None:
            self._prior = prior
        elif hasattr(self.potential_fn, "prior") and isinstance(
            self.potential_fn.prior, Distribution
        ):
            self._prior = self.potential_fn.prior
        elif isinstance(q, VIPosterior) and isinstance(q._prior, Distribution):
            self._prior = q._prior
        else:
            raise ValueError(
                "We could not find a suitable prior distribution within `potential_fn` "
                "or `q` (if a VIPosterior is given). Please explicitly specify a prior."
            )

        self._prior = move_distribution_to_device(self._prior, device)
        self._optimizer = None

        # Mode tracking: None (not trained), "single_x", or "amortized"
        self._mode: Optional[Literal["single_x", "amortized"]] = None

        # Amortized mode: conditional flow q(θ|x)
        self._amortized_q: Optional[ConditionalDensityEstimator] = None

        self._num_transforms: int = num_transforms
        self._hidden_features: int = hidden_features
        self._z_score_theta: Literal["none", "independent", "structured"] = (
            z_score_theta
        )
        self._z_score_x: Literal["none", "independent", "structured"] = z_score_x

        # In contrast to MCMC we want to project into constrained space.
        if theta_transform is None:
            self.link_transform = mcmc_transform(self._prior, device=device).inv
        else:
            self.link_transform = theta_transform.inv

        if parameters is None:
            parameters = []
        if modules is None:
            modules = []
        # This will set the variational distribution and VI method
        self.set_q(
            q,
            parameters=parameters,
            modules=modules,
        )
        self.set_vi_method(vi_method)

        self._purpose = (
            "It provides Variational inference to .sample() from the posterior and "
            "can evaluate the _normalized_ posterior density with .log_prob()."
        )

    def to(self, device: Union[str, torch.device]) -> "VIPosterior":
        """
        Move potential_fn, _prior and x_o to device, and change the device attribute.

        Reinstantiates the posterior and re sets the default x.

        Args:
            device: The device to move the posterior to.

        Returns:
            self for method chaining.
        """
        self.device = device
        self.potential_fn.to(device)  # type: ignore
        self._prior.to(device)  # type: ignore
        if self._x is not None:
            x_o = self._x.to(device)
        self.theta_transform = mcmc_transform(self._prior, device=device)
        super().__init__(
            self.potential_fn, self.theta_transform, device, x_shape=self.x_shape
        )
        # super().__init__ erases the self._x, so we need to set it again
        if self._x is not None:
            self.set_default_x(x_o)

        if self.theta_transform is None:
            self.link_transform = mcmc_transform(self._prior, device=device).inv
        else:
            self.link_transform = self.theta_transform.inv

        return self

    def _build_unconditional_flow(
        self,
        flow_type: str,
        num_transforms: Optional[int] = None,
        hidden_features: Optional[int] = None,
        z_score_theta: Optional[Literal["none", "independent", "structured"]] = None,
    ) -> TransformedZukoFlow:
        """Build a Zuko unconditional flow for variational inference.

        The flow is wrapped with TransformedZukoFlow to handle the transformation
        between unconstrained (flow) space and constrained (prior) space. This ensures
        that samples from the flow match the prior's support and log_prob accounts
        for the Jacobian of the transformation.

        Args:
            flow_type: Type of flow, one of ["maf", "nsf", "naf", "unaf", "nice",
                "sospf"]. For "gaussian" or "gaussian_diag", use LearnableGaussian.
            num_transforms: Number of flow transforms. If None, uses instance default.
            hidden_features: Number of hidden features per layer. If None, uses
                instance default.
            z_score_theta: Method for z-scoring theta (parameters). One of
                "independent", "structured", or "none". If None, uses instance default.
                Use "structured" for parameters with correlations.

        Returns:
            TransformedZukoFlow: The constructed flow wrapped with link_transform.

        Raises:
            ValueError: If flow_type is not supported.
        """
        # Fall back to instance attributes
        if num_transforms is None:
            num_transforms = self._num_transforms
        if hidden_features is None:
            hidden_features = self._hidden_features
        if z_score_theta is None:
            z_score_theta = self._z_score_theta

        if flow_type not in _ZUKO_FLOW_TYPES:
            raise ValueError(
                f"Unknown flow type '{flow_type}'. "
                f"Supported types: {sorted(_ZUKO_FLOW_TYPES)} + "
                f"['gaussian', 'gaussian_diag']."
            )

        zuko_flow_type = flow_type.upper()

        # Get prior dimensionality
        prior_dim = self._prior.event_shape[0] if self._prior.event_shape else 1

        # Warn about 1D limitation
        if prior_dim == 1:
            warnings.warn(
                f"Using {flow_type.upper()} flow for 1D parameter space. "
                f"Normalizing flows may be unstable for 1D VI optimization. "
                f"Consider using q='gaussian' for better results in 1D.",
                UserWarning,
                stacklevel=3,
            )

        # Sample from prior to get batch for dimensionality inference and z-scoring
        # We apply link_transform.inv to map constrained prior samples to unconstrained
        # space (link_transform.forward maps unconstrained -> constrained)
        with torch.no_grad():
            prior_samples = self._prior.sample((1000,))
            batch_theta = self.link_transform.inv(prior_samples)
            assert isinstance(batch_theta, Tensor)  # Type narrowing for pyright

        flow = build_zuko_unconditional_flow(
            which_nf=zuko_flow_type,
            batch_x=batch_theta,
            z_score_x=z_score_theta,  # theta z-scoring passed to Zuko's x parameter
            hidden_features=hidden_features,
            num_transforms=num_transforms,
        )

        # Wrap flow with link_transform to ensure samples are in constrained space
        # The flow operates in unconstrained space, but we want samples/log_probs
        # in constrained space (matching the prior's support)
        transformed_flow = TransformedZukoFlow(
            flow=flow.to(self._device),
            link_transform=self.link_transform,
        )

        return transformed_flow.to(self._device)

    def _build_conditional_flow(
        self,
        theta: Tensor,
        x: Tensor,
        flow_type: Union[ZukoFlowType, str] = ZukoFlowType.NSF,
        num_transforms: int = 2,
        hidden_features: int = 32,
        z_score_theta: Literal["none", "independent", "structured"] = "independent",
        z_score_x: Literal["none", "independent", "structured"] = "independent",
    ) -> ConditionalDensityEstimator:
        """Build a conditional Zuko flow for amortized variational inference.

        Args:
            theta: Sample of θ values for z-scoring (batch_size, θ_dim).
            x: Sample of x values for z-scoring (batch_size, x_dim).
            flow_type: Type of flow. Can be a ZukoFlowType enum or string.
            num_transforms: Number of flow transforms.
            hidden_features: Number of hidden features per layer.
            z_score_theta: Method for z-scoring θ (the parameters being modeled).
                One of "none", "independent", "structured".
            z_score_x: Method for z-scoring x (the conditioning variable).
                One of "none", "independent", "structured". Use "structured" for
                structured data like images.

        Returns:
            ConditionalDensityEstimator: The constructed conditional flow q(θ|x).

        Raises:
            ValueError: If flow_type is not supported.
        """
        # Convert string to ZukoFlowType if needed
        if isinstance(flow_type, str):
            try:
                flow_type = ZukoFlowType[flow_type.upper()]
            except KeyError as e:
                raise ValueError(
                    f"Unknown flow type '{flow_type}'. "
                    f"Supported types: {[t.name for t in ZukoFlowType]}."
                ) from e

        return build_zuko_flow(
            flow_type.value.upper(),
            batch_x=theta,  # θ is what we model
            batch_y=x,  # x is the condition
            z_score_x=z_score_theta,  # z-score for θ (naming mismatch)
            z_score_y=z_score_x,  # z-score for x condition
            num_transforms=num_transforms,
            hidden_features=hidden_features,
        ).to(self._device)

    @property
    def q(
        self,
    ) -> Union[
        Distribution, ZukoUnconditionalFlow, TransformedZukoFlow, LearnableGaussian
    ]:
        """Returns the variational posterior."""
        return self._q

    @q.setter
    def q(self, q: QType) -> None:
        """Sets the variational distribution.

        If the distribution does not admit access through `parameters` and `modules`
        function, please use `set_q` to explicitly specify the parameters and modules.
        """
        self.set_q(q)

    def set_q(
        self,
        q: QType,
        parameters: Optional[Iterable] = None,
        modules: Optional[Iterable] = None,
    ) -> None:
        """Defines the variational family.

        You can specify over which parameters/modules we optimize. This is required for
        custom distributions which e.g. do not inherit nn.Modules or has the function
        `parameters` or `modules` to give direct access to trainable parameters.
        Further, you can pass a function, which constructs a variational distribution
        if called.

        Args:
            q: Variational distribution, either string, distribution, or a VIPosterior
                object. This specifies a parametric class of distribution over which
                the best possible posterior approximation is searched. For string input,
                we support normalizing flows [maf, nsf, naf, unaf, nice, sospf] via
                Zuko, and simple Gaussian families [gaussian, gaussian_diag] via pure
                PyTorch. You can also specify your own variational family by passing a
                `parameterized` distribution object i.e. a torch.distributions
                Distribution with methods `parameters` returning an iterable of all
                parameters (you can pass them within the parameters/modules attribute).
                Additionally, we allow a `Callable` with signature
                `(event_shape: torch.Size, link_transform: TorchTransform, device: str)
                -> Distribution`, which builds a custom distribution. If q is already
                a `VIPosterior`, then the arguments will be copied from it (relevant
                for multi-round training).

                Note: For 1D parameter spaces, normalizing flows may be unstable.
                Consider using `q='gaussian'` for 1D problems.
            parameters: List of parameters associated with the distribution object.
            modules: List of modules associated with the distribution object.

        """
        if parameters is None:
            parameters = []
        if modules is None:
            modules = []
        self._q_arg = (q, parameters, modules)
        _flow_types = (ZukoUnconditionalFlow, TransformedZukoFlow, LearnableGaussian)
        if isinstance(q, _flow_types):
            # Flow/Gaussian passed directly (e.g., from _q_build_fn during retrain)
            make_object_deepcopy_compatible(q)
            self._trained_on = None
        elif isinstance(q, Distribution):
            q = adapt_variational_distribution(
                q,
                self._prior,
                self.link_transform,
                parameters=parameters,
                modules=modules,
            )
            make_object_deepcopy_compatible(q)
            self_custom_q_init_cache = deepcopy(q)
            self._q_build_fn = lambda *args, **kwargs: self_custom_q_init_cache
            self._trained_on = None
            self._zuko_flow_type = None
        elif isinstance(q, (str, Callable)):
            if isinstance(q, str):
                if q in _ZUKO_FLOW_TYPES:
                    q_flow = self._build_unconditional_flow(q)
                    self._zuko_flow_type = q
                    self._q_build_fn = lambda *args, ft=q, **kwargs: (
                        self._build_unconditional_flow(ft)
                    )
                    q = q_flow
                elif q in ("gaussian", "gaussian_diag"):
                    self._zuko_flow_type = None
                    full_cov = q == "gaussian"
                    dim = self._prior.event_shape[0]
                    q_dist = LearnableGaussian(
                        dim=dim,
                        full_covariance=full_cov,
                        link_transform=self.link_transform,
                        device=self._device,
                    )
                    self._q_build_fn = lambda *args, fc=full_cov, d=dim, **kwargs: (
                        LearnableGaussian(
                            dim=d,
                            full_covariance=fc,
                            link_transform=self.link_transform,
                            device=self._device,
                        )
                    )
                    q = q_dist
                else:
                    supported = sorted(_ZUKO_FLOW_TYPES) + ["gaussian", "gaussian_diag"]
                    raise ValueError(
                        f"Unknown variational family '{q}'. "
                        f"Supported options: {supported}"
                    )
            else:
                # Callable provided - use as-is
                self._zuko_flow_type = None
                self._q_build_fn = q
                q = self._q_build_fn(
                    self._prior.event_shape,
                    self.link_transform,
                    device=self._device,
                )
            make_object_deepcopy_compatible(q)
            self._trained_on = None
        elif isinstance(q, VIPosterior):
            self._q_build_fn = q._q_build_fn
            self._trained_on = q._trained_on
            self._mode = getattr(q, "_mode", None)  # Copy mode from source
            self._zuko_flow_type = getattr(q, "_zuko_flow_type", None)
            self.vi_method = q.vi_method  # type: ignore
            self._device = q._device
            self._prior = q._prior
            self._x = q._x
            self._q_arg = q._q_arg
            make_object_deepcopy_compatible(q.q)
            q = deepcopy(q.q)
        # Validate the variational distribution
        if isinstance(q, _flow_types):
            pass  # These are validated during construction
        elif isinstance(q, Distribution):
            check_variational_distribution(q, self._prior)
        else:
            raise ValueError(
                f"Variational distribution must be a Distribution, got {type(q)}. "
                "Please create an issue on github https://github.com/mackelab/sbi/issues"
            )
        self._q = q

    @property
    def vi_method(self) -> str:
        """Variational inference method e.g. one of [rKL, fKL, IW, alpha]."""
        return self._vi_method

    @vi_method.setter
    def vi_method(self, method: str) -> None:
        """See `set_vi_method`."""
        self.set_vi_method(method)

    def set_vi_method(self, method: str) -> "VIPosterior":
        """Sets variational inference method.

        Args:
            method: One of [rKL, fKL, IW, alpha].

        Returns:
            `VIPosterior` for chainable calls.
        """
        self._vi_method = method
        self._optimizer_builder = get_VI_method(method)
        return self

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        show_progress_bars: bool = True,
    ) -> Tensor:
        r"""Draw samples from the variational posterior distribution $p(\theta|x)$.

        For single-x mode (trained via `train()`): samples from q(θ) trained on x_o.
        For amortized mode (trained via `train_amortized()`): samples from q(θ|x).

        Args:
            sample_shape: Desired shape of samples that are drawn from the posterior.
            x: Conditioning observation. In single-x mode, must match trained x_o
                (or be None to use default). In amortized mode, required and can be
                any observation. For batched observations, shape should be
                (batch_size, x_dim).
            show_progress_bars: Unused for `VIPosterior` since sampling from the
                variational distribution is fast. Included for API consistency.

        Returns:
            Samples from posterior with shape (*sample_shape, θ_dim) for single x,
            or (*sample_shape, batch_size, θ_dim) for batched observations in
            amortized mode.

        Raises:
            ValueError: If mode requirements are not met.
        """
        if self._mode == "amortized":
            # Amortized mode: sample from conditional flow q(θ|x)
            x = self._x_else_default_x(x)
            if x is None:
                raise ValueError(
                    "x is required for amortized mode. Provide an observation or "
                    "set a default x with set_default_x()."
                )
            x = atleast_2d_float32_tensor(x).to(self._device)
            assert self._amortized_q is not None
            # samples shape from flow: (*sample_shape, batch_size, θ_dim)
            samples = self._amortized_q.sample(torch.Size(sample_shape), condition=x)
            # Match base posterior behavior: drop singleton x batch dimension
            if x.shape[0] == 1:
                samples = samples.squeeze(-2)
            return samples
        else:
            # Single-x mode: sample from unconditional flow q(θ)
            x = self._x_else_default_x(x)
            if self._trained_on is None or (x != self._trained_on).any():
                raise ValueError(
                    f"The variational posterior was not fit on the specified "
                    f"observation {x}. Please train using posterior.train()."
                )
            samples = self.q.sample(torch.Size(sample_shape))
            return samples.reshape((*sample_shape, samples.shape[-1]))

    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        max_sampling_batch_size: int = 10000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        """Sample from posterior for a batch of observations.

        In amortized mode, this is efficient as all x values are processed in
        parallel through the conditional flow.

        In single-x mode, this raises NotImplementedError since the unconditional
        flow is trained for a specific x_o.

        Args:
            sample_shape: Number of samples per observation.
            x: Batch of observations (num_obs, x_dim).
            max_sampling_batch_size: Unused for amortized mode (no batching needed).
            show_progress_bars: Unused for amortized mode.

        Returns:
            Samples of shape (*sample_shape, num_obs, θ_dim).

        Raises:
            NotImplementedError: If called in single-x mode.
        """
        if self._mode == "amortized":
            # In amortized mode, sample() handles batched x directly
            return self.sample(sample_shape, x=x, show_progress_bars=show_progress_bars)
        else:
            raise NotImplementedError(
                "Batched sampling is not implemented for single-x VI mode. "
                "Use train_amortized() to train an amortized posterior, or "
                "call sample() in a loop: [posterior.sample(shape, x_o) for x_o in x]."
            )

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        track_gradients: bool = False,
    ) -> Tensor:
        r"""Returns the log-probability of theta under the variational posterior.

        For single-x mode: returns log q(θ).
        For amortized mode: returns log q(θ|x).

        Args:
            theta: Parameters to evaluate, shape (batch_theta, θ_dim).
            x: Observation. In single-x mode, must match trained x_o (or be None).
                In amortized mode, required and can be any observation.
                For single x, shape (1, x_dim) or (x_dim,).
                For batched x, shape (batch_x, x_dim).
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis but increases memory
                consumption.

        Returns:
            Log-probability of shape (batch,) where batch is:
            - batch_theta if x has batch size 1 (broadcast x)
            - batch_x if theta has batch size 1 (broadcast theta)
            - batch_theta if batch_theta == batch_x (paired evaluation)

        Raises:
            ValueError: If mode requirements are not met or batch sizes incompatible.
        """
        with torch.set_grad_enabled(track_gradients):
            theta = ensure_theta_batched(torch.as_tensor(theta)).to(self._device)

            if self._mode == "amortized":
                # Amortized mode: evaluate log q(θ|x)
                x = self._x_else_default_x(x)
                if x is None:
                    raise ValueError(
                        "x is required for amortized mode. Provide an observation or "
                        "set a default x with set_default_x()."
                    )
                x = atleast_2d_float32_tensor(x).to(self._device)
                assert self._amortized_q is not None

                # Handle broadcasting between theta and x
                batch_theta = theta.shape[0]
                batch_x = x.shape[0]

                if batch_theta != batch_x:
                    if batch_x == 1:
                        # Broadcast x to match theta
                        x = x.expand(batch_theta, -1)
                    elif batch_theta == 1:
                        # Broadcast theta to match x
                        theta = theta.expand(batch_x, -1)
                    else:
                        raise ValueError(
                            f"Batch sizes of theta ({batch_theta}) and x ({batch_x}) "
                            f"are incompatible. They must be equal, or one must be 1."
                        )

                # ZukoFlow expects input shape (sample_dim, batch_dim, *event_shape)
                # Add sample dimension, compute log_prob, then squeeze back
                theta_with_sample_dim = theta.unsqueeze(0)
                log_probs = self._amortized_q.log_prob(
                    theta_with_sample_dim, condition=x
                )
                return log_probs.squeeze(0)
            else:
                # Single-x mode: evaluate log q(θ)
                x = self._x_else_default_x(x)
                if self._trained_on is None or (x != self._trained_on).any():
                    raise ValueError(
                        f"The variational posterior was not fit on the specified "
                        f"observation {x}. Please train using posterior.train()."
                    )
                return self.q.log_prob(theta)

    def train(
        self,
        x: Optional[TorchTensor] = None,
        n_particles: int = 256,
        learning_rate: float = 1e-3,
        gamma: float = 0.999,
        max_num_iters: int = 2000,
        min_num_iters: int = 10,
        clip_value: float = 10.0,
        warm_up_rounds: int = 100,
        retrain_from_scratch: bool = False,
        reset_optimizer: bool = False,
        show_progress_bar: bool = True,
        check_for_convergence: bool = True,
        quality_control: bool = True,
        quality_control_metric: str = "psis",
        **kwargs,
    ) -> "VIPosterior":
        """This method trains the variational posterior for a single observation.

        Args:
            x: The observation, optional, defaults to self._x.
            n_particles: Number of samples to approximate expectations within the
                variational bounds. The larger the more accurate are gradient
                estimates, but the computational cost per iteration increases.
            learning_rate: Learning rate of the optimizer.
            gamma: Learning rate decay per iteration. We use an exponential decay
                scheduler.
            max_num_iters: Maximum number of iterations.
            min_num_iters: Minimum number of iterations.
            clip_value: Gradient clipping value, decreasing may help if you see invalid
                values.
            warm_up_rounds: Initialize the posterior as the prior.
            retrain_from_scratch: Retrain the variational distributions from scratch.
            reset_optimizer: Reset the divergence optimizer
            show_progress_bar: If any progress report should be displayed.
            quality_control: If False quality control is skipped.
            quality_control_metric: Which metric to use for evaluating the quality.
            kwargs: Hyperparameters check corresponding `DivergenceOptimizer` for detail
                eps: Determines sensitivity of convergence check.
                retain_graph: Boolean which decides whether to retain the computation
                    graph. This may be required for some `exotic` user-specified q's.
                optimizer: A PyTorch Optimizer class e.g. Adam or SGD. See
                    `DivergenceOptimizer` for details.
                scheduler: A PyTorch learning rate scheduler. See
                    `DivergenceOptimizer` for details.
                alpha: Only used if vi_method=`alpha`. Determines the alpha divergence.
                K: Only used if vi_method=`IW`. Determines the number of importance
                    weighted particles.
                stick_the_landing: If one should use the STL estimator (only for rKL,
                    IW, alpha).
                dreg: If one should use the DREG estimator (only for rKL, IW, alpha).
                weight_transform: Callable applied to importance weights (only for fKL)
        Returns:
            VIPosterior: `VIPosterior` (can be used to chain calls).

        Raises:
            ValueError: If hyperparameters are invalid.
        """
        # Validate hyperparameters
        if n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {n_particles}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if max_num_iters <= 0:
            raise ValueError(f"max_num_iters must be positive, got {max_num_iters}")
        if min_num_iters < 0:
            raise ValueError(f"min_num_iters must be non-negative, got {min_num_iters}")
        if clip_value <= 0:
            raise ValueError(f"clip_value must be positive, got {clip_value}")

        # Update optimizer with current arguments.
        if self._optimizer is not None:
            self._optimizer.update({**locals(), **kwargs})

        # Init q and the optimizer if necessary
        if retrain_from_scratch:
            self.q = self._q_build_fn()  # type: ignore
            self._optimizer = self._optimizer_builder(
                self.potential_fn,
                self.q,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                prior=self._prior,
                **kwargs,
            )

        if (
            reset_optimizer
            or self._optimizer is None
            or not isinstance(self._optimizer, self._optimizer_builder)
        ):
            self._optimizer = self._optimizer_builder(
                self.potential_fn,
                self.q,
                lr=learning_rate,
                clip_value=clip_value,
                gamma=gamma,
                n_particles=n_particles,
                prior=self._prior,
                **kwargs,
            )

        # Check context
        x = atleast_2d_float32_tensor(self._x_else_default_x(x)).to(  # type: ignore
            self._device
        )
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf values.")

        already_trained = self._trained_on is not None and (x == self._trained_on).all()

        # Optimize
        optimizer = self._optimizer
        optimizer.to(self._device)
        optimizer.reset_loss_stats()

        if show_progress_bar:
            iters = tqdm(range(max_num_iters))
        else:
            iters = range(max_num_iters)

        # Warmup before training
        if reset_optimizer or (not optimizer.warm_up_was_done and not already_trained):
            if show_progress_bar:
                iters.set_description(  # type: ignore
                    "Warmup phase, this may take a few seconds..."
                )
            optimizer.warm_up(warm_up_rounds)

        for i in iters:
            optimizer.step(x)
            mean_loss, std_loss = optimizer.get_loss_stats()
            # Update progress bar
            if show_progress_bar:
                assert isinstance(iters, tqdm)
                iters.set_description(  # type: ignore
                    f"Loss: {np.round(float(mean_loss), 2)}, "
                    f"Std: {np.round(float(std_loss), 2)}"
                )
            # Check for convergence
            if check_for_convergence and i > min_num_iters and optimizer.converged():
                if show_progress_bar:
                    print(f"\nConverged with loss: {np.round(float(mean_loss), 2)}")
                break
        # Training finished:
        self._trained_on = x
        if self._mode == "amortized":
            warnings.warn(
                "Switching from amortized to single-x mode. "
                "The previously trained amortized model will be discarded.",
                UserWarning,
                stacklevel=2,
            )
            self._amortized_q = None
        self._mode = "single_x"

        # Evaluate quality
        if quality_control:
            try:
                self.evaluate(quality_control_metric=quality_control_metric)
            except Exception as e:
                print(
                    f"Quality control showed a low quality of the variational "
                    f"posterior. We are automatically retraining the variational "
                    f"posterior from scratch with a smaller learning rate. "
                    f"Alternatively, if you want to skip quality control, please "
                    f"retrain with `VIPosterior.train(..., quality_control=False)`. "
                    f"\nThe error that occured is: {e}"
                )
                self.train(
                    learning_rate=learning_rate * 0.1,
                    retrain_from_scratch=True,
                    reset_optimizer=True,
                )

        return self

    def train_amortized(
        self,
        theta: Tensor,
        x: Tensor,
        n_particles: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.999,
        max_num_iters: int = 500,
        clip_value: float = 5.0,
        batch_size: int = 64,
        validation_fraction: float = 0.1,
        validation_batch_size: Optional[int] = None,
        validation_n_particles: Optional[int] = None,
        stop_after_iters: int = 20,
        show_progress_bar: bool = True,
        retrain_from_scratch: bool = False,
        flow_type: Optional[Union[ZukoFlowType, str]] = None,
        num_transforms: Optional[int] = None,
        hidden_features: Optional[int] = None,
        z_score_theta: Optional[Literal["none", "independent", "structured"]] = None,
        z_score_x: Optional[Literal["none", "independent", "structured"]] = None,
        params: Optional["VIPosteriorParameters"] = None,
    ) -> "VIPosterior":
        """Train a conditional flow q(θ|x) for amortized variational inference.

        This allows sampling from q(θ|x) for any observation x without retraining.
        Uses the ELBO (Evidence Lower Bound) objective with early stopping based on
        validation loss.

        Args:
            theta: Training θ values from simulations (num_sims, θ_dim).
            x: Training x values from simulations (num_sims, x_dim).
            n_particles: Number of samples to estimate ELBO per x.
            learning_rate: Learning rate for Adam optimizer.
            gamma: Learning rate decay per iteration.
            max_num_iters: Maximum training iterations.
            clip_value: Gradient clipping threshold.
            batch_size: Number of x values per training batch.
            validation_fraction: Fraction of data to use for validation.
            validation_batch_size: Batch size for validation loss. Defaults to
                `batch_size`.
            validation_n_particles: Number of particles for validation loss.
                Defaults to `n_particles`.
            stop_after_iters: Stop training after this many iterations without
                improvement in validation loss.
            show_progress_bar: Whether to show progress.
            retrain_from_scratch: If True, rebuild the flow from scratch.
            flow_type: Flow architecture for the variational distribution.
                Use ZukoFlowType.NSF, ZukoFlowType.MAF, etc., or a string.
                If None, uses value from params or instance default.
            num_transforms: Number of transforms in the flow. If None, uses value
                from params or instance default.
            hidden_features: Hidden layer size in the flow. If None, uses value
                from params or instance default.
            z_score_theta: Method for z-scoring θ (the parameters being modeled).
                One of "none", "independent", "structured". If None, uses value
                from params or instance default.
            z_score_x: Method for z-scoring x (the conditioning variable).
                One of "none", "independent", "structured". Use "structured" for
                structured data like images with spatial correlations. If None,
                uses value from params or instance default.
            params: Optional VIPosteriorParameters dataclass. Values are used as
                fallbacks when explicit arguments are None. Priority order:
                explicit args > params > instance attributes (from __init__).

        Returns:
            self for method chaining.
        """
        # Resolve parameters: explicit args > params dataclass > instance attrs
        if params is not None:
            # Amortized VI only supports string flow types (not VIPosterior or Callable)
            if not isinstance(params.q, str):
                raise ValueError(
                    "train_amortized() only supports string flow types "
                    f"(e.g., 'nsf', 'maf'), not {type(params.q).__name__}. "
                    "Use set_q() to pass custom distributions for single-x VI."
                )
            if flow_type is None:
                flow_type = params.q
            if num_transforms is None:
                num_transforms = params.num_transforms
            if hidden_features is None:
                hidden_features = params.hidden_features
            if z_score_theta is None:
                z_score_theta = params.z_score_theta
            if z_score_x is None:
                z_score_x = params.z_score_x

        # Fall back to instance attributes (set in __init__ from VIPosteriorParameters)
        if flow_type is None:
            flow_type = ZukoFlowType.NSF
        if num_transforms is None:
            num_transforms = self._num_transforms
        if hidden_features is None:
            hidden_features = self._hidden_features
        if z_score_theta is None:
            z_score_theta = self._z_score_theta
        if z_score_x is None:
            z_score_x = self._z_score_x

        theta = atleast_2d_float32_tensor(theta).to(self._device)
        x = atleast_2d_float32_tensor(x).to(self._device)

        # Validate inputs
        if theta.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch: theta has {theta.shape[0]} samples, "
                f"x has {x.shape[0]} samples. They must match."
            )
        if len(theta) == 0:
            raise ValueError("Training data cannot be empty.")
        if not torch.isfinite(theta).all():
            raise ValueError("theta contains NaN or Inf values.")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf values.")

        # Validate theta dimension matches prior
        prior_event_shape = self._prior.event_shape
        if len(prior_event_shape) > 0:
            expected_theta_dim = prior_event_shape[0]
            if theta.shape[1] != expected_theta_dim:
                raise ValueError(
                    f"theta dimension {theta.shape[1]} does not match prior "
                    f"event shape {expected_theta_dim}."
                )

        # Validate hyperparameters
        if not 0 < validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be in (0, 1), got {validation_fraction}"
            )
        if n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {n_particles}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Validate flow_type early to fail fast
        if isinstance(flow_type, str):
            try:
                flow_type = ZukoFlowType[flow_type.upper()]
            except KeyError:
                raise ValueError(
                    f"Unknown flow type '{flow_type}'. "
                    f"Supported types: {[t.name for t in ZukoFlowType]}."
                ) from None

        if validation_batch_size is None:
            validation_batch_size = batch_size
        if validation_n_particles is None:
            validation_n_particles = n_particles

        if validation_batch_size <= 0:
            raise ValueError(
                f"validation_batch_size must be positive, got {validation_batch_size}"
            )
        if validation_n_particles <= 0:
            raise ValueError(
                f"validation_n_particles must be positive, got {validation_n_particles}"
            )

        # Split into training and validation sets
        num_examples = len(theta)
        num_val = int(validation_fraction * num_examples)
        num_train = num_examples - num_val

        if num_val == 0:
            raise ValueError(
                "Validation set is empty. Increase validation_fraction or provide more "
                "training data."
            )
        if num_train < batch_size:
            raise ValueError(
                f"Training set size ({num_train}) is smaller than batch_size "
                f"({batch_size}). Reduce validation_fraction or batch_size."
            )

        permuted_indices = torch.randperm(num_examples, device=self._device)
        train_indices = permuted_indices[:num_train]
        val_indices = permuted_indices[num_train:]

        theta_train, x_train = theta[train_indices], x[train_indices]
        x_val = x[val_indices]  # Only x needed for validation (θ sampled from q)

        if validation_batch_size < x_val.shape[0]:
            val_batch_indices = torch.randperm(x_val.shape[0], device=self._device)[
                :validation_batch_size
            ]
        else:
            val_batch_indices = None

        # Build or rebuild the conditional flow (z-score on training data only)
        if self._amortized_q is None or retrain_from_scratch:
            self._amortized_q = self._build_conditional_flow(
                theta_train,
                x_train,
                flow_type=flow_type,
                num_transforms=num_transforms,
                hidden_features=hidden_features,
                z_score_theta=z_score_theta,
                z_score_x=z_score_x,
            )

        # Ensure potential_fn is on the correct device for amortized training
        self.potential_fn.to(self._device)

        # Setup optimizer
        optimizer = Adam(self._amortized_q.parameters(), lr=learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        # Training loop with validation-based early stopping
        best_val_loss = float("inf")
        iters_since_improvement = 0
        best_state_dict = deepcopy(self._amortized_q.state_dict())

        if show_progress_bar:
            iters = tqdm(range(max_num_iters), desc="Amortized VI (ELBO)")
        else:
            iters = range(max_num_iters)

        for iteration in iters:
            # Training step
            self._amortized_q.train()
            optimizer.zero_grad()

            # Sample batch from training set
            idx = torch.randint(0, num_train, (batch_size,), device=self._device)
            x_batch = x_train[idx]

            train_loss = self._compute_amortized_elbo_loss(x_batch, n_particles)

            if not torch.isfinite(train_loss):
                raise RuntimeError(
                    f"Training loss became non-finite at iteration {iteration}: "
                    f"{train_loss.item()}. This indicates numerical instability. Try:\n"
                    f"  - Reducing learning_rate (currently {learning_rate})\n"
                    f"  - Reducing n_particles (currently {n_particles})\n"
                    f"  - Checking your potential_fn for numerical issues"
                )

            train_loss.backward()
            nn.utils.clip_grad_norm_(self._amortized_q.parameters(), clip_value)
            optimizer.step()
            scheduler.step()

            # Compute validation loss
            self._amortized_q.eval()
            with torch.no_grad():
                if val_batch_indices is None:
                    x_val_batch = x_val
                else:
                    x_val_batch = x_val[val_batch_indices]
                val_loss = self._compute_amortized_elbo_loss(
                    x_val_batch, validation_n_particles
                ).item()

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                iters_since_improvement = 0
                best_state_dict = deepcopy(self._amortized_q.state_dict())
            else:
                iters_since_improvement += 1

            if show_progress_bar:
                assert isinstance(iters, tqdm)
                iters.set_postfix({
                    "train": f"{train_loss.item():.3f}",
                    "val": f"{val_loss:.3f}",
                })

            # Early stopping
            if iters_since_improvement >= stop_after_iters:
                if show_progress_bar:
                    print(f"\nConverged at iteration {iteration}")
                break

        # Restore best model
        self._amortized_q.load_state_dict(best_state_dict)
        self._amortized_q.eval()
        if self._mode == "single_x":
            warnings.warn(
                "Switching from single-x to amortized mode. "
                "The previously trained single-x model will not be usable.",
                UserWarning,
                stacklevel=2,
            )
        self._mode = "amortized"

        return self

    def _compute_amortized_elbo_loss(self, x_batch: Tensor, n_particles: int) -> Tensor:
        """Compute negative ELBO loss for a batch of x values.

        Args:
            x_batch: Batch of observations (batch_size, x_dim).
            n_particles: Number of θ samples per x.

        Returns:
            Negative ELBO (scalar tensor).
        """
        assert self._amortized_q is not None, "q must be built before computing ELBO"
        batch_size = x_batch.shape[0]

        # Reparameterized samples from q(θ|x) with their log probabilities
        # theta_samples shape: (n_particles, batch_size, θ_dim)
        # log_q shape: (n_particles, batch_size)
        theta_samples, log_q = self._amortized_q.sample_and_log_prob(
            torch.Size((n_particles,)), condition=x_batch
        )

        # Vectorized evaluation of potential log p(θ|x) for all (θ, x) pairs
        # Flatten: (n_particles, batch_size, θ_dim) -> (n_particles * batch_size, θ_dim)
        theta_dim = theta_samples.shape[-1]
        theta_flat = theta_samples.reshape(n_particles * batch_size, theta_dim)

        # Repeat x to match: (batch_size, x_dim) -> (n_particles * batch_size, x_dim)
        # Each x[j] is repeated n_particles times to pair with theta[:, j, :]
        x_expanded = x_batch.repeat(n_particles, 1)

        # Set x_o for batched evaluation (x_is_iid=False: each θ paired with its x)
        self.potential_fn.set_x(x_expanded, x_is_iid=False)
        log_potential_flat = self.potential_fn(theta_flat)

        # Reshape: (n_particles * batch_size,) -> (n_particles, batch_size)
        log_potential = log_potential_flat.reshape(n_particles, batch_size)

        # ELBO = E_q[log p(θ|x) - log q(θ|x)]
        elbo = (log_potential - log_q).mean()
        return -elbo

    def evaluate(self, quality_control_metric: str = "psis", N: int = int(5e4)) -> None:
        """This function will evaluate the quality of the variational posterior
        distribution. We currently support two different metrics of type `psis`, which
        checks the quality based on the tails of importance weights (there should not be
        much with a large one), or `prop` which checks the proportionality between q
        and potential_fn.

        NOTE: In our experience `prop` is sensitive to distinguish ``good`` from ``ok``
        whereas `psis` is more sensitive in distinguishing `very bad` from `ok`.

        Args:
            quality_control_metric: The metric of choice, we currently support [psis,
                prop, prop_prior].
            N: Number of samples which is used to evaluate the metric.
        """
        quality_control_fn, quality_control_msg = get_quality_metric(
            quality_control_metric
        )
        metric = round(float(quality_control_fn(self, N=N)), 3)
        print(f"Quality Score: {metric} " + quality_control_msg)

    def map(
        self,
        x: Optional[TorchTensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, TorchTensor] = "proposal",
        num_init_samples: int = 10_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self._map` and
        can be accessed with `self.map()`. The MAP is obtained by running gradient
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
            show_progress_bars: Whether to show a progressbar during sampling from
                the posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.
            log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                {'norm_posterior': True} for SNPE.

        Returns:
            The MAP estimate.
        """
        self.proposal = self.q
        return super().map(
            x=x,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
            force_update=force_update,
        )

    def __deepcopy__(self, memo: Optional[Dict] = None) -> "VIPosterior":
        """This method is called when using `copy.deepcopy` on the object.

        It defines how the object is copied. We need to overwrite this method, since the
        default implementation does use __getstate__ and __setstate__ which we overwrite
        to enable pickling (and in particular the necessary modifications are
        incompatible deep copying).

        Args:
            memo (Optional[Dict], optional): Deep copy internal memo. Defaults to None.

        Returns:
            VIPosterior: Deep copy of the VIPosterior.
        """
        if memo is None:
            memo = {}

        # Create a new instance of the class
        cls = self.__class__
        result = cls.__new__(cls)
        # Add to memo
        memo[id(self)] = result
        # Copy attributes
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __getstate__(self) -> Dict:
        """This method is called when pickling the object.

        It defines what is pickled. We need to overwrite this method, since some parts
        do not support pickle protocols (e.g. due to local functions).

        Returns:
            Dict: All attributes of the VIPosterior.
        """
        self._optimizer = None
        self.__deepcopy__ = None  # type: ignore
        self._q_build_fn = None
        self._q.__deepcopy__ = None  # type: ignore
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state_dict: Dict):
        """This method is called when unpickling the object.

        Especially, we need to restore the removed attributes and ensure that the object
        e.g. remains deep copy compatible.

        Args:
            state_dict: Given state dictionary, we will restore the object from it.
        """
        self.__dict__ = state_dict
        q = deepcopy(self._q)
        # Restore removed attributes
        self.set_q(*self._q_arg)
        self._q = q
        make_object_deepcopy_compatible(self)
        make_object_deepcopy_compatible(self.q)
        # Handle amortized mode
        if self._mode == "amortized" and self._amortized_q is not None:
            make_object_deepcopy_compatible(self._amortized_q)
