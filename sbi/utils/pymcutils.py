# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Utilities for PyMC integration with SBI neural likelihood estimators.

Notes
-----
- We implement a custom PyTensor Op that returns both the scalar log-likelihood
    and its gradient w.r.t. ``theta`` as two outputs. The gradient output is then
    referenced in ``grad()`` to provide a symbolic backward pass for HMC/NUTS.
- Dtypes are aligned with PyTensor/PyMC ``floatX`` (typically float64), while
    PyTorch evaluation uses the estimator's parameter dtype (typically float32).
    We convert between them at the boundaries.
- **Memory considerations**: The Op stores references to the neural network and
    observation data. For large models or datasets, this may impact memory usage.
- **Op equality**: Ops are considered equal if they have identical observation
    shapes and content (compared via SHA-256 hash digest).
"""

import hashlib
from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import torch
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.variable import TensorVariable

from sbi.neural_nets.estimators import ConditionalDensityEstimator


def _get_estimator_device_dtype(
    estimator: ConditionalDensityEstimator,
) -> tuple[torch.device, torch.dtype]:
    """Get device and dtype from an estimator's parameters.

    Args:
        estimator: A ConditionalDensityEstimator (nn.Module subclass)

    Returns:
        Tuple of (device, dtype) from the estimator's first parameter
    """
    p0 = next(estimator.parameters())  # type: ignore[attr-defined]
    return p0.device, p0.dtype


def _compute_observation_digest(observation: np.ndarray) -> str:
    """Compute SHA-256 digest of observation array for Op equality checking."""
    obs_bytes = np.asarray(observation).tobytes(order="C")
    return hashlib.sha256(obs_bytes).hexdigest()


def _store_outputs(
    output_storage: list[list[np.ndarray]],
    logp: torch.Tensor,
    grad: np.ndarray,
    theta_dtype: np.dtype,
) -> None:
    """Store log-probability and gradient in output storage.

    Args:
        output_storage: PyTensor output storage arrays
        logp: Log probability tensor (scalar)
        grad: Gradient numpy array
        theta_dtype: dtype of input theta for output casting
    """
    out_dtype = theta_dtype if np.issubdtype(theta_dtype, np.floating) else np.float64
    output_storage[0][0] = logp.detach().cpu().numpy().astype(out_dtype)
    output_storage[1][0] = grad.astype(out_dtype)


def _compute_gradients(
    total_logp: torch.Tensor,
    theta_torch: torch.Tensor,
) -> np.ndarray:
    """Compute gradients of log-probability w.r.t. theta.

    Args:
        total_logp: Scalar log probability tensor
        theta_torch: Input theta tensor (with requires_grad=True)

    Returns:
        Gradient as numpy array
    """
    grad_list = torch.autograd.grad(
        outputs=total_logp,
        inputs=theta_torch,
        create_graph=False,
        allow_unused=False,
    )
    return grad_list[0].detach().cpu().numpy()


class NeuralLikelihoodOp(Op):
    """PyTensor Op wrapping a neural likelihood estimator for use with PyMC.

    This Op evaluates log p(x|θ) using a trained ConditionalDensityEstimator
    and provides gradients for HMC/NUTS sampling.

    Supports scalar, 1D, and 2D theta inputs through automatic shape normalization.
    """

    # Input: parameter vector θ; Outputs: scalar log-likelihood and gradient vector
    # We define outputs explicitly in make_node and only use default_output here.
    default_output: int = 0  # Return only log-likelihood by default when called

    __props__ = ("observation_shape", "observation_digest")

    def __init__(
        self,
        estimator: ConditionalDensityEstimator,
        observation: np.ndarray,
    ):
        """Initialize the neural likelihood op.

        Args:
            estimator: Trained ConditionalDensityEstimator from NLE
            observation: Observed data x_o to condition on, shape (n_obs, *event_shape)
        """
        self.estimator = estimator

        # Infer device & dtype from estimator parameters
        self._torch_device, self._torch_dtype = _get_estimator_device_dtype(estimator)

        # Store observation as torch tensor on the estimator's device/dtype
        self.observation = torch.as_tensor(
            observation, dtype=self._torch_dtype, device=self._torch_device
        )
        self.observation_shape = tuple(observation.shape)
        self.observation_digest = _compute_observation_digest(observation)

        # Ensure single observation has an explicit leading n_obs dimension.
        # Shapes after this:
        #   - single observation: (1, *event_shape)
        #   - i.i.d. observations: (n_obs, *event_shape)
        if self.observation.dim() == 1:
            self.observation = self.observation.unsqueeze(0)

    def make_node(self, theta: TensorVariable) -> Apply:
        """Create the Apply node for the computation graph.

        Supports theta with various shapes:
        - Scalar: shape () - single scalar parameter
        - 1D: shape (D,) - single vector parameter
        - 2D: shape (1, D) - single vector parameter with explicit batch dim

        Args:
            theta: Parameter tensor variable

        Returns:
            Apply node with input and output variables
        """
        # Ensure theta is a tensor variable
        theta = pt.as_tensor_variable(theta)
        # Outputs are: scalar log-likelihood and gradient with the same
        # type (dtype/broadcastable) as theta so PyTensor enforces shape rank.
        logp_out = pt.scalar(dtype=theta.dtype)
        grad_out = theta.type()
        return Apply(self, [theta], [logp_out, grad_out])

    def perform(
        self,
        node: Apply,
        inputs: list[np.ndarray],
        output_storage: list[list[np.ndarray]],
    ) -> None:
        """Compute the log-likelihood and its gradients.

        Args:
            node: The Apply node (unused but required by interface)
            inputs: List containing [theta_array]
            output_storage: List of arrays to store outputs [log_prob, gradients]
        """
        self.estimator.eval()

        theta_np = inputs[0]
        original_ndim = int(np.ndim(theta_np))

        # Normalize theta for estimator: target shape is (1, D) for single sample
        # The estimator expects condition shape (batch_dim, *params)
        if original_ndim == 0:
            # Scalar: () -> (1, 1)
            theta_for_est = theta_np.reshape(1, 1)
        elif original_ndim == 1:
            # Vector parameter: (D,) -> (1, D)
            theta_for_est = theta_np[np.newaxis, :]
        elif original_ndim == 2 and theta_np.shape[0] == 1:
            # Already correct shape: (1, D)
            theta_for_est = theta_np
        elif original_ndim == 2 and theta_np.shape[0] > 1:
            raise ValueError(
                f"NeuralLikelihoodOp received theta with batch dimension "
                f"{theta_np.shape[0]}, but expected batch_dim=1 for simple mode. "
                f"If you intended to use a hierarchical model with "
                f"{theta_np.shape[0]} subjects, pass num_subjects={theta_np.shape[0]} "
                f"to neural_likelihood_to_pymc(). Otherwise, check your PyMC prior "
                f"definition - for a {theta_np.shape[1]}D parameter, use:\n"
                f"  theta = pm.Normal('theta', ..., shape={theta_np.shape[1]})"
            )
        else:
            raise ValueError(
                f"NeuralLikelihoodOp expects theta with shape (), (D,), or (1, D), "
                f"got shape {theta_np.shape}. Check your PyMC prior definition."
            )

        # Convert to torch on the estimator's device/dtype with gradient tracking
        theta_torch = torch.as_tensor(
            theta_for_est, dtype=self._torch_dtype, device=self._torch_device
        )
        theta_torch.requires_grad_(True)
        # theta_torch shape is (1, D)

        # Estimator expects input (sample_dim, batch_dim, *event) and
        # condition (batch_dim, *params). Use sample_dim=n_obs, batch_dim=1.
        obs = self.observation  # (n_obs, *event_shape)
        obs_expanded = obs.unsqueeze(1)

        with torch.set_grad_enabled(True):
            log_prob_batch = self.estimator.log_prob(
                input=obs_expanded, condition=theta_torch
            )
            log_prob_sum_over_samples = log_prob_batch.sum(dim=0)  # (1,)
            # Squeeze batch_dim=1 to get a scalar
            log_prob_scalar = log_prob_sum_over_samples.squeeze()

        # Compute gradients
        grad_np = _compute_gradients(log_prob_scalar, theta_torch)

        # Reshape gradient to match original theta shape
        # grad_np is currently (1, D), need to match original_shape
        if original_ndim == 0:
            # Scalar: (1, 1) -> () — use reshape for explicit 0-d output
            grad_np = grad_np.reshape(())
        elif original_ndim == 1:
            # Vector param: (1, D) -> (D,)
            grad_np = grad_np.squeeze(0)
        elif original_ndim == 2:
            # Keep as (1, D)
            pass

        _store_outputs(output_storage, log_prob_scalar, grad_np, theta_np.dtype)

    def grad(
        self, inputs: list[TensorVariable], output_grads: list[TensorVariable]
    ) -> list[TensorVariable]:
        """Compute gradients of log-likelihood w.r.t. parameters.

        This method is called by PyTensor to get gradients for the backward pass.

        Args:
            inputs: List containing [theta]
            output_grads: Gradients w.r.t. outputs [grad_log_prob, grad_gradients]

        Returns:
            List containing gradient w.r.t. theta
        """
        # Reuse the Apply node corresponding to this Op application and
        # reference the second output (gradients wrt theta). We obtain the
        # node by calling the Op and accessing owner.outputs.
        value = self(*inputs)
        gradients = value.owner.outputs[1]  # type: ignore[union-attr]
        # Chain rule: dC/dtheta = dC/dlog_prob * dlog_prob/dtheta
        return [output_grads[0] * gradients]


class HierarchicalNeuralLikelihoodOp(Op):
    """PyTensor Op for hierarchical models with subject/trial structure.

    This Op handles models with S subjects each having T trials (2-level),
    or G groups each with S subjects and T trials (3-level). The NLE is
    trained on single-subject data, and at inference time we broadcast each
    subject's parameters across their trials.

    Shape conventions (2-level, num_groups=None):
    - theta: (S, D) where S = num_subjects, D = parameter event dimension
    - observations: input as (T, S, *obs_event), stored as (S*T, *obs_event)

    Shape conventions (3-level, num_groups > 0):
    - theta: (G, S, D) where G = num_groups
    - observations: input as (G, S, T, *obs_event), stored as (G*S*T, *obs_event)

    Output: scalar log-likelihood summed over all subjects and trials.
    """

    default_output: int = 0

    __props__ = (
        "observation_shape",
        "observation_digest",
        "num_trials",
        "num_subjects",
        "num_groups",
    )

    def __init__(
        self,
        estimator: ConditionalDensityEstimator,
        observation: np.ndarray,
        num_trials: int,
        num_subjects: int,
        num_groups: int | None = None,
    ):
        """Initialize the hierarchical neural likelihood op.

        Args:
            estimator: Trained ConditionalDensityEstimator from NLE
            observation: Observed data. 2-level: shape (T, S, *event).
                3-level: shape (G, S, T, *event).
            num_trials: Number of trials per subject
            num_subjects: Number of subjects (per group in 3-level)
            num_groups: Number of groups for 3-level hierarchy (None for 2-level)
        """
        self.estimator = estimator
        self.num_trials = num_trials
        self.num_subjects = num_subjects
        self.num_groups = num_groups

        # Infer device & dtype from estimator parameters
        self._torch_device, self._torch_dtype = _get_estimator_device_dtype(estimator)

        # Validate and store observation
        obs_np = np.asarray(observation)
        self.observation_shape = tuple(obs_np.shape)

        if num_groups is not None:
            # 3-level: obs shape (G, S, T, *E) — hierarchy from coarsest to finest
            if obs_np.ndim < 3:
                raise ValueError(
                    f"3-level hierarchy requires observed with at least 3 dimensions "
                    f"(num_groups, num_subjects, num_trials, ...), "
                    f"got shape {obs_np.shape}"
                )
            if (
                obs_np.shape[0] != num_groups
                or obs_np.shape[1] != num_subjects
                or obs_np.shape[2] != num_trials
            ):
                raise ValueError(
                    f"Observation shape {obs_np.shape} doesn't match "
                    f"num_groups={num_groups}, num_subjects={num_subjects}, "
                    f"num_trials={num_trials}. Expected shape "
                    f"(num_groups, num_subjects, num_trials, *event_shape)."
                )

            # Flatten: (G, S, T, *E) -> (G*S*T, *E) — already in correct order
            event_shape = obs_np.shape[3:]
            obs_flat = obs_np.reshape(-1, *event_shape)  # (G*S*T, *E)
        else:
            # 2-level: obs shape (T, S, *E)
            if obs_np.shape[0] != num_trials or obs_np.shape[1] != num_subjects:
                raise ValueError(
                    f"Observation shape {obs_np.shape} doesn't match "
                    f"num_trials={num_trials}, num_subjects={num_subjects}. "
                    f"Expected shape (num_trials, num_subjects, *event_shape)."
                )

            # Flatten: (T, S, *E) -> (S, T, *E) -> (S*T, *E)
            obs_flat = obs_np.transpose(1, 0, *range(2, obs_np.ndim))  # (S, T, *E)
            obs_flat = obs_flat.reshape(-1, *obs_np.shape[2:])  # (S*T, *E)

        self.observation_digest = _compute_observation_digest(obs_np)

        self.observation = torch.as_tensor(
            obs_flat, dtype=self._torch_dtype, device=self._torch_device
        )

    def make_node(self, theta: TensorVariable) -> Apply:
        """Create the Apply node for the computation graph.

        Args:
            theta: Parameter tensor with shape (num_subjects, event_dim)

        Returns:
            Apply node with scalar log-likelihood and gradient outputs
        """
        theta = pt.as_tensor_variable(theta)
        logp_out = pt.scalar(dtype=theta.dtype)
        grad_out = theta.type()
        return Apply(self, [theta], [logp_out, grad_out])

    def perform(
        self,
        node: Apply,
        inputs: list[np.ndarray],
        output_storage: list[list[np.ndarray]],
    ) -> None:
        """Compute the hierarchical log-likelihood and gradients.

        2-level: theta (S, D) -> repeat_interleave(T) -> (S*T, D)
        3-level: theta (G, S, D) -> reshape (G*S, D)
            -> repeat_interleave(T) -> (G*S*T, D)
        """
        self.estimator.eval()

        theta_np = inputs[0]
        original_shape = theta_np.shape

        if self.num_groups is not None:
            # 3-level: expected (G, S, D), may receive (1, G, S, D) from PyMC
            if theta_np.ndim == 4 and theta_np.shape[0] == 1:
                theta_np = theta_np.squeeze(0)
            if theta_np.ndim != 3:
                raise ValueError(
                    f"3-level HierarchicalNeuralLikelihoodOp expects theta with shape "
                    f"(num_groups={self.num_groups}, num_subjects={self.num_subjects}, "
                    f"event_dim), but received shape {original_shape}. "
                    f"Check your PyMC prior:\n"
                    f"  theta = pm.Normal('theta', ..., "
                    f"shape=({self.num_groups}, {self.num_subjects}, event_dim))"
                )

            if theta_np.shape[0] != self.num_groups:
                raise ValueError(
                    f"theta has {theta_np.shape[0]} groups but Op expects "
                    f"{self.num_groups} groups."
                )
            if theta_np.shape[1] != self.num_subjects:
                raise ValueError(
                    f"theta has {theta_np.shape[1]} subjects but Op expects "
                    f"{self.num_subjects} subjects."
                )

            # Convert to torch with gradient tracking
            theta_torch = torch.as_tensor(
                theta_np, dtype=self._torch_dtype, device=self._torch_device
            )
            theta_torch.requires_grad_(True)

            # Flatten: (G, S, D) -> (G*S, D) -> repeat_interleave(T) -> (G*S*T, D)
            theta_flat = theta_torch.reshape(-1, theta_torch.shape[-1])  # (G*S, D)
            theta_expanded = theta_flat.repeat_interleave(self.num_trials, dim=0)
        else:
            # 2-level: expected (S, D), may receive (1, S, D) from PyMC
            if theta_np.ndim == 3 and theta_np.shape[0] == 1:
                theta_np = theta_np.squeeze(0)
            if theta_np.ndim != 2:
                raise ValueError(
                    f"HierarchicalNeuralLikelihoodOp expects theta with shape "
                    f"(num_subjects={self.num_subjects}, event_dim), but received "
                    f"shape {original_shape} with {theta_np.ndim} dimensions. "
                    f"Check your PyMC prior:\n"
                    f"  theta = pm.Normal('theta', ..., "
                    f"shape=({self.num_subjects}, event_dim))"
                )

            if theta_np.shape[0] != self.num_subjects:
                raise ValueError(
                    f"theta has {theta_np.shape[0]} subjects but Op expects "
                    f"{self.num_subjects} subjects."
                )

            # Convert to torch with gradient tracking
            theta_torch = torch.as_tensor(
                theta_np, dtype=self._torch_dtype, device=self._torch_device
            )
            theta_torch.requires_grad_(True)

            # Broadcast: (S, D) -> (S*T, D)
            theta_expanded = theta_torch.repeat_interleave(self.num_trials, dim=0)

        # Observations already flattened in __init__: (S*T, *E) or (G*S*T, *E)
        # Add sample_dim=1 for estimator
        obs_for_est = self.observation.unsqueeze(0)

        with torch.set_grad_enabled(True):
            log_probs = self.estimator.log_prob(
                input=obs_for_est,
                condition=theta_expanded,
            )
            total_logp = log_probs.sum()

        # Compute gradients w.r.t. theta (before expansion)
        grad_np = _compute_gradients(total_logp, theta_torch)

        # Restore original shape if we squeezed a leading dimension
        if self.num_groups is not None:
            if len(original_shape) == 4 and original_shape[0] == 1:
                grad_np = grad_np[np.newaxis, ...]
        else:
            if len(original_shape) == 3 and original_shape[0] == 1:
                grad_np = grad_np[np.newaxis, ...]

        _store_outputs(output_storage, total_logp, grad_np, inputs[0].dtype)

    def grad(
        self, inputs: list[TensorVariable], output_grads: list[TensorVariable]
    ) -> list[TensorVariable]:
        """Compute gradients for the backward pass."""
        value = self(*inputs)
        gradients = value.owner.outputs[1]  # type: ignore[union-attr]
        return [output_grads[0] * gradients]


def _validate_inputs(
    observed: np.ndarray,
    num_trials: int | None,
    num_subjects: int | None,
    num_groups: int | None = None,
) -> np.ndarray:
    """Validate inputs for neural_likelihood_to_pymc.

    Args:
        observed: Observed data array
        num_trials: Number of trials (hierarchical mode)
        num_subjects: Number of subjects (hierarchical mode)
        num_groups: Number of groups (3-level hierarchy)

    Returns:
        Validated observed array

    Raises:
        ValueError: If inputs are invalid
    """
    # Convert to numpy array if needed
    if not isinstance(observed, np.ndarray):
        observed = np.asarray(observed)

    # Check for empty data
    if observed.size == 0:
        raise ValueError("observed data cannot be empty")

    # Check for NaN/Inf
    if np.any(~np.isfinite(observed)):
        raise ValueError(
            "observed data contains NaN or Inf values. "
            "Please check your data preprocessing."
        )

    # Validate num_trials and num_subjects
    if num_trials is not None and num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}")
    if num_subjects is not None and num_subjects <= 0:
        raise ValueError(f"num_subjects must be positive, got {num_subjects}")
    if num_groups is not None and num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")

    # Validate num_groups requires num_trials and num_subjects
    if num_groups is not None and (num_trials is None or num_subjects is None):
        raise ValueError(
            "num_groups requires both num_trials and num_subjects to be set."
        )

    # Shape-specific validation is handled by the Op constructors
    # (NeuralLikelihoodOp / HierarchicalNeuralLikelihoodOp) which know
    # their own shape conventions.

    return observed


def neural_likelihood_to_pymc(
    likelihood_nn: ConditionalDensityEstimator,
    theta: TensorVariable,
    observed: np.ndarray,
    name: str = "likelihood",
    dims: tuple[str, ...] | None = None,
    num_trials: int | None = None,
    num_subjects: int | None = None,
    num_groups: int | None = None,
    **kwargs: Any,
) -> TensorVariable:
    """Create a PyMC CustomDist from a neural likelihood estimator.

    This function wraps a trained NLE network as a PyMC distribution that can
    be used as a likelihood in a PyMC model. The likelihood is conditioned on
    the observed data and evaluates log p(x|theta).

    Supports three modes:
    1. **Simple mode** (default): For single-subject models with i.i.d. observations
    2. **2-level hierarchical**: For multi-subject models with trials per subject
    3. **3-level hierarchical**: For grouped multi-subject models
       (groups > subjects > trials)

    Simple mode shape conventions:
    - Single observation: observed has shape (*event_shape,)
    - I.i.d. observations: observed has shape (n_obs, *event_shape)
    - theta: any shape supported by PyMC (scalar, vector, etc.)

    2-level hierarchical (num_trials + num_subjects):
    - observed: shape (num_trials, num_subjects, *event_shape)
    - theta: shape (num_subjects, event_dim)

    3-level hierarchical (num_trials + num_subjects + num_groups):
    - observed: shape (num_groups, num_subjects, num_trials, *event_shape)
    - theta: shape (num_groups, num_subjects, event_dim)

    Args:
        likelihood_nn: Trained ConditionalDensityEstimator from NLE
        theta: PyMC parameter variable that the likelihood depends on
        observed: Observed data to condition the likelihood on
        name: Name for the PyMC distribution
        dims: PyMC dimension names for the distribution (forwarded to CustomDist)
        num_trials: For hierarchical models, number of observations per subject.
            Despite the name "trials", this refers to independent observations
            (e.g., repeated measurements, time points, or experimental replicates).
        num_subjects: For hierarchical models, number of subjects (or groups/units)
        num_groups: For 3-level hierarchical models, number of groups
        **kwargs: Additional arguments forwarded to pm.CustomDist

    Returns:
        PyMC CustomDist representing the neural likelihood

    Example:
        ```python
        # After training NLE on single-subject data
        likelihood_nn = nle.train()

        # Simple mode: single subject with i.i.d. observations
        with pm.Model() as model:
            theta = pm.Normal("theta", mu=0, sigma=1, shape=2)
            likelihood = neural_likelihood_to_pymc(
                likelihood_nn, theta, x_observed, "x"
            )
            trace = pm.sample()

        # 2-level hierarchical: multiple subjects with trials
        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            tau = pm.InverseGamma("tau", alpha=1, beta=1)
            theta = pm.Normal("theta", mu=mu, sigma=pm.math.sqrt(tau),
                              shape=(num_subjects, event_dim))
            likelihood = neural_likelihood_to_pymc(
                likelihood_nn, theta, x_observed, "x",
                num_trials=num_trials, num_subjects=num_subjects,
            )
            trace = pm.sample()

        # 3-level hierarchical: groups of subjects with trials
        with pm.Model() as model:
            mu_group = pm.Normal("mu_group", mu=0, sigma=1, shape=num_groups)
            theta = pm.Normal("theta", mu=mu_group[:, None, None],
                              sigma=1, shape=(num_groups, num_subjects, event_dim))
            likelihood = neural_likelihood_to_pymc(
                likelihood_nn, theta, x_observed, "x",
                num_trials=num_trials, num_subjects=num_subjects,
                num_groups=num_groups,
            )
            trace = pm.sample()
        ```
    """
    # Validate inputs early
    observed = _validate_inputs(observed, num_trials, num_subjects, num_groups)

    # Validate estimator has parameters (i.e., is trained)
    try:
        next(likelihood_nn.parameters())  # type: ignore[attr-defined]
    except StopIteration:
        raise ValueError(
            "likelihood_nn has no parameters. Is it a trained estimator? "
            "Make sure to call inference.train() before using the estimator."
        ) from None

    # Determine mode based on parameters
    op: NeuralLikelihoodOp | HierarchicalNeuralLikelihoodOp

    if num_trials is not None and num_subjects is not None:
        # Hierarchical mode (2-level or 3-level)
        op = HierarchicalNeuralLikelihoodOp(
            likelihood_nn, observed, num_trials, num_subjects,
            num_groups=num_groups,
        )
    elif num_trials is None and num_subjects is None and num_groups is None:
        # Simple mode
        op = NeuralLikelihoodOp(likelihood_nn, observed)
    else:
        raise ValueError(
            "For hierarchical mode, provide both num_trials and num_subjects "
            "(and optionally num_groups for 3-level). "
            "For simple mode, provide none of these."
        )

    # Cache device/dtype from the Op for use in random() closure
    torch_device = op._torch_device
    torch_dtype = op._torch_dtype

    def logp(
        value: Any, theta: TensorVariable
    ) -> TensorVariable:  # PyMC passes (value, *dist_params)
        """Log-probability function for the CustomDist.

        We ignore ``value`` here because the observation is captured in the Op.
        The returned scalar corresponds to log p(x_o | theta).
        """
        # The Op handles all the computation; default_output=0 returns log_prob
        return cast(TensorVariable, op(theta))

    def random(
        theta_np: np.ndarray,
        rng: np.random.Generator | None = None,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Draw samples x ~ p(x | theta) from the neural likelihood.

        Args:
            theta_np: Numpy array for theta (no batching supported here).
            rng: Optional NumPy Generator. If provided, used to seed torch.
            size: Output size pre-pended to the support shape. Examples:
                - None: returns a single sample with shape equal to obs shape.
                - 5: returns shape (5, *obs_shape)
                - (n, m): returns shape (n, m, *obs_shape)

        Notes:
            - This implementation supports a single unbatched theta per draw.
              For vectorized/batched theta, call this function in a loop.
        """
        if num_trials is not None and num_subjects is not None:
            raise NotImplementedError(
                "Posterior predictive sampling (pm.sample_posterior_predictive) is "
                "not yet supported for hierarchical models. For now, manually simulate "
                "from posterior samples:\n"
                "  for theta_sample in posterior_samples:\n"
                "      x_pred = likelihood_nn.sample((n,), condition=theta_sample)\n"
                "See the SBI tutorials for examples."
            )
        # Optionally seed torch from the provided numpy Generator (PyMC default)
        if isinstance(rng, np.random.Generator):
            seed = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
            torch.manual_seed(seed)

        # Use cached device/dtype from closure (avoids repeated parameter lookup)
        theta_t = torch.as_tensor(theta_np, dtype=torch_dtype, device=torch_device)

        # Validate and normalize theta to (batch=1, event_dim) for estimator
        if theta_t.dim() == 0:
            # Scalar: () -> (1, 1)
            theta_t = theta_t.reshape(1, 1)
        elif theta_t.dim() == 1:
            # Vector param: (D,) -> (1, D)
            theta_t = theta_t.unsqueeze(0)
        elif theta_t.dim() == 2 and theta_t.shape[0] == 1:
            # Already correct: (1, D)
            pass
        else:
            raise ValueError(
                f"random() expects unbatched theta with shape (), (D,), or (1, D), "
                f"got shape {tuple(theta_np.shape)}. Check your PyMC prior definition."
            )

        # Map PyMC `size` to estimator sample_shape and final output shape
        if size is None:
            n_obs = observed.shape[0] if np.ndim(observed) > 1 else 1
            out_shape: tuple[int, ...] = (n_obs,) if n_obs > 1 else ()
        elif isinstance(size, int):
            out_shape = (size,)
        else:  # tuple[int, ...]
            out_shape = size
        sample_shape = torch.Size(out_shape if out_shape else (1,))

        # Draw samples: shape (sample_dim, batch_dim, *obs_event_shape)
        with torch.no_grad():
            x_t = likelihood_nn.sample(sample_shape=sample_shape, condition=theta_t)

        # Remove batch_dim (assumed 1) and reshape to size + obs_shape
        # Current shape: (S, B=1, *E) → (S, *E)
        if x_t.dim() >= 2:
            x_t = x_t[:, 0, ...]
        x_np = x_t.cpu().numpy()

        if not out_shape:
            # return a single sample with shape (*E)
            return np.asarray(x_np[0])
        # reshape from (S, *E) to (*size, *E)
        return np.asarray(x_np).reshape(*out_shape, *x_np.shape[1:])

    # For 3-level hierarchy, rearrange observed so PyMC can broadcast.
    # The Op stores its own flattened copy from the (G, S, T, *E) input.
    # PyMC needs T in the leading position to broadcast with theta (G, S, D):
    # (G, S, T, *E) -> (T, G, S, *E)
    observed_for_pymc = observed
    if num_groups is not None:
        observed_for_pymc = np.moveaxis(observed, 2, 0)

    return pm.CustomDist(
        name,
        theta,
        logp=logp,
        random=random,
        observed=observed_for_pymc,
        dims=dims,
        **kwargs,
    )
