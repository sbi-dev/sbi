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
- **Op equality**: Ops are considered equal if they reference the same estimator
    and have identical observations (compared via SHA-256 hash digest).
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

    __props__ = ("estimator_id", "observation_shape", "observation_digest")

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
        self.estimator_id = id(estimator)  # For props/equality

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
            # Scalar: (1, 1) -> ()
            grad_np = grad_np.squeeze()
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

    This Op handles the case where we have S subjects, each with T trials.
    The NLE is trained on single-subject data, and at inference time we
    broadcast each subject's parameters across their trials.

    Shape conventions:
    - theta: (S, D) where S = num_subjects, D = parameter event dimension
    - observations: stored as (T*S, *obs_event) after flattening
    - Output: scalar log-likelihood summed over all subjects and trials
    """

    default_output: int = 0

    __props__ = (
        "estimator_id",
        "observation_shape",
        "observation_digest",
        "num_trials",
        "num_subjects",
    )

    def __init__(
        self,
        estimator: ConditionalDensityEstimator,
        observation: np.ndarray,
        num_trials: int,
        num_subjects: int,
    ):
        """Initialize the hierarchical neural likelihood op.

        Args:
            estimator: Trained ConditionalDensityEstimator from NLE
            observation: Observed data with shape (num_trials, num_subjects, *event)
            num_trials: Number of trials per subject
            num_subjects: Number of subjects
        """
        self.estimator = estimator
        self.estimator_id = id(estimator)
        self.num_trials = num_trials
        self.num_subjects = num_subjects

        # Infer device & dtype from estimator parameters
        self._torch_device, self._torch_dtype = _get_estimator_device_dtype(estimator)

        # Validate and store observation
        obs_np = np.asarray(observation)
        self.observation_shape = tuple(obs_np.shape)

        if obs_np.shape[0] != num_trials or obs_np.shape[1] != num_subjects:
            raise ValueError(
                f"Observation shape {obs_np.shape} doesn't match "
                f"num_trials={num_trials}, num_subjects={num_subjects}. "
                f"Expected shape (num_trials, num_subjects, *event_shape)."
            )

        self.observation_digest = _compute_observation_digest(obs_np)

        # Flatten observations: (T, S, *E) -> (S*T, *E) in subject-major order
        # After transpose: (S, T, *E) - all trials for each subject are consecutive
        # This matches the repeat_interleave on theta which creates (S*T, D) by
        # repeating each subject's params T times: [θ0, θ0, ..., θ1, θ1, ...]
        obs_flat = obs_np.transpose(1, 0, *range(2, obs_np.ndim))  # (S, T, *E)
        obs_flat = obs_flat.reshape(-1, *obs_np.shape[2:])  # (S*T, *E)
        # Use device parameter directly to avoid unnecessary copy
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

        The computation:
        1. Broadcast theta: (S, D) -> (S*T, D) by repeating each subject T times
        2. Evaluate log p(x | theta) for all S*T pairs
        3. Sum to get total log-likelihood
        """
        theta_np = inputs[0]
        original_shape = theta_np.shape

        # Normalize theta shape: PyMC may add leading dimensions
        # Expected: (S, D) but may receive (1, S, D) where S=num_subjects, D=event_dim
        if theta_np.ndim == 3 and theta_np.shape[0] == 1:
            # Squeeze out the leading sample dimension
            theta_np = theta_np.squeeze(0)
        elif theta_np.ndim != 2:
            raise ValueError(
                f"HierarchicalNeuralLikelihoodOp expects theta with shape "
                f"(num_subjects={self.num_subjects}, event_dim), but received "
                f"shape {original_shape} with {theta_np.ndim} dimensions. "
                f"For hierarchical models, theta should be a 2D array where each "
                f"row represents one subject's parameters. Check your PyMC prior:\n"
                f"  theta = pm.Normal('theta', ..., "
                f"shape=({self.num_subjects}, event_dim))"
            )

        if theta_np.shape[0] != self.num_subjects:
            raise ValueError(
                f"theta has {theta_np.shape[0]} subjects but Op expects "
                f"{self.num_subjects} subjects. Check that your PyMC prior shape "
                f"matches num_subjects:\n"
                f"  theta = pm.Normal('theta', ..., "
                f"shape=({self.num_subjects}, event_dim))"
            )

        # Convert to torch with gradient tracking
        theta_torch = torch.as_tensor(
            theta_np, dtype=self._torch_dtype, device=self._torch_device
        )
        theta_torch.requires_grad_(True)

        # Broadcast theta: (S, D) -> (S*T, D)
        # Each subject's parameters repeated T times to match flattened obs
        theta_expanded = theta_torch.repeat_interleave(self.num_trials, dim=0)

        # Observations are already flattened: (S*T, *E)
        # Add sample_dim=1 for estimator: (1, S*T, *E)
        obs_for_est = self.observation.unsqueeze(0)

        with torch.set_grad_enabled(True):
            # Vectorized forward pass
            log_probs = self.estimator.log_prob(
                input=obs_for_est,  # (1, S*T, *E)
                condition=theta_expanded,  # (S*T, D)
            )  # Returns: (1, S*T)

            # Sum all log probs to get total likelihood
            total_logp = log_probs.sum()

        # Compute gradients w.r.t. theta (before expansion)
        grad_np = _compute_gradients(total_logp, theta_torch)

        # Restore original shape if we squeezed a leading dimension
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
) -> np.ndarray:
    """Validate inputs for neural_likelihood_to_pymc.

    Args:
        observed: Observed data array
        num_trials: Number of trials (hierarchical mode)
        num_subjects: Number of subjects (hierarchical mode)

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

    # Early shape validation for hierarchical mode
    if num_trials is not None and num_subjects is not None:
        if observed.ndim < 2:
            raise ValueError(
                f"Hierarchical mode requires observed with at least 2 dimensions "
                f"(num_trials, num_subjects, ...), got shape {observed.shape}"
            )
        if observed.shape[0] != num_trials:
            raise ValueError(
                f"observed.shape[0]={observed.shape[0]} doesn't match "
                f"num_trials={num_trials}. Expected shape "
                f"(num_trials={num_trials}, num_subjects={num_subjects}, *event_shape)."
            )
        if observed.shape[1] != num_subjects:
            raise ValueError(
                f"observed.shape[1]={observed.shape[1]} doesn't match "
                f"num_subjects={num_subjects}. Expected shape "
                f"(num_trials={num_trials}, num_subjects={num_subjects}, *event_shape)."
            )

    return observed


def neural_likelihood_to_pymc(
    likelihood_nn: ConditionalDensityEstimator,
    theta: TensorVariable,
    observed: np.ndarray,
    name: str = "likelihood",
    dims: tuple[str, ...] | None = None,
    num_trials: int | None = None,
    num_subjects: int | None = None,
    **kwargs: Any,
) -> TensorVariable:
    """Create a PyMC CustomDist from a neural likelihood estimator.

    This function wraps a trained NLE network as a PyMC distribution that can
    be used as a likelihood in a PyMC model. The likelihood is conditioned on
    the observed data and evaluates log p(x|θ).

    Supports two modes:
    1. **Simple mode** (default): For single-subject models with i.i.d. observations
    2. **Hierarchical mode**: For multi-subject models with trials per subject

    Simple mode shape conventions:
    - Single observation: observed has shape (*event_shape,)
    - I.i.d. observations: observed has shape (n_obs, *event_shape)
    - theta: any shape supported by PyMC (scalar, vector, etc.)

    Hierarchical mode shape conventions (when num_trials and num_subjects given):
    - observed: shape (num_trials, num_subjects, *event_shape)
    - theta: shape (num_subjects, event_dim) - one parameter vector per subject

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

        # Hierarchical mode: multiple subjects with trials
        # observed shape: (num_trials, num_subjects, *event_shape)
        with pm.Model() as model:
            # Hyperpriors
            mu = pm.Normal("mu", mu=0, sigma=1)
            tau = pm.InverseGamma("tau", alpha=1, beta=1)

            # Subject-level parameters: shape (num_subjects, event_dim)
            theta = pm.Normal("theta", mu=mu, sigma=pm.math.sqrt(tau),
                              shape=(num_subjects, event_dim))

            # Neural likelihood with hierarchical structure
            likelihood = neural_likelihood_to_pymc(
                likelihood_nn, theta, x_observed, "x",
                num_trials=num_trials,
                num_subjects=num_subjects,
            )
            trace = pm.sample()
        ```
    """
    # Validate inputs early
    observed = _validate_inputs(observed, num_trials, num_subjects)

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
        # Hierarchical mode
        op = HierarchicalNeuralLikelihoodOp(
            likelihood_nn, observed, num_trials, num_subjects
        )
    elif num_trials is None and num_subjects is None:
        # Simple mode
        op = NeuralLikelihoodOp(likelihood_nn, observed)
    else:
        raise ValueError(
            "Both num_trials and num_subjects must be provided for "
            "hierarchical mode, or neither for simple mode."
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

    return pm.CustomDist(
        name, theta, logp=logp, random=random, observed=observed, dims=dims, **kwargs
    )
