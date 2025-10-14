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


class NeuralLikelihoodOp(Op):
    """PyTensor Op wrapping a neural likelihood estimator for use with PyMC.

    This Op evaluates log p(x|θ) using a trained ConditionalDensityEstimator
    and provides gradients for HMC/NUTS sampling.
    """

    # Input: parameter vector θ; Outputs: scalar log-likelihood and gradient vector
    # We define outputs explicitly in make_node and only use default_output here.
    default_output: int = 0  # Return only log-likelihood by default when called

    __props__ = ("estimator_id", "observation_shape", "observation_digest")

    def __init__(self, estimator: ConditionalDensityEstimator, observation: np.ndarray):
        """Initialize the neural likelihood op.

        Args:
            estimator: Trained ConditionalDensityEstimator from NLE
            observation: Observed data x_o to condition on, shape (n_obs, *event_shape)
        """
        self.estimator = estimator
        self.estimator_id = id(estimator)  # For props/equality

        # Infer device & dtype from the first parameter (standard nn.Module contract)
        p0 = next(self.estimator.parameters())  # type: ignore[attr-defined]
        self._torch_device: torch.device = p0.device
        self._torch_dtype: torch.dtype = p0.dtype

        # Store observation as torch tensor on the estimator's device/dtype
        self.observation = torch.as_tensor(observation, dtype=self._torch_dtype).to(
            self._torch_device
        )
        self.observation_shape = tuple(observation.shape)
        # Deterministic hash to encode observation content for Op equality
        obs_bytes = np.asarray(observation).tobytes(order="C")
        self.observation_digest = hashlib.sha1(obs_bytes).hexdigest()

        # Ensure single observation has an explicit leading n_obs dimension.
        # Shapes after this:
        #   - single observation: (1, *event_shape)
        #   - i.i.d. observations: (n_obs, *event_shape)
        if self.observation.dim() == 1:
            self.observation = self.observation.unsqueeze(0)

    def make_node(self, theta: TensorVariable) -> Apply:
        """Create the Apply node for the computation graph.

        Args:
            theta: Parameter tensor variable

        Returns:
            Apply node with input and output variables
        """
        # Ensure theta is a tensor variable
        theta = pt.as_tensor_variable(theta)

        # Outputs are: scalar log-likelihood and gradient vector (match theta dtype)
        outputs = [pt.scalar(dtype=theta.dtype), pt.vector(dtype=theta.dtype)]

        return Apply(self, [theta], outputs)

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
        # Get theta from inputs (NumPy array) and convert to torch on the
        # estimator's device/dtype with gradient tracking enabled.
        theta_np = inputs[0]
        theta_torch = torch.as_tensor(
            theta_np, dtype=self._torch_dtype, device=self._torch_device
        )
        theta_torch.requires_grad_(True)

        # Keep the original shape for gradient computation
        original_shape = theta_torch.shape

        # Ensure theta has batch dimension: (batch_dim, *cond_event_shape)
        if theta_torch.dim() == 1:
            theta_torch = theta_torch.unsqueeze(0)

        # Safeguard: PyMC evaluates one theta per logp call. Prevent accidental
        # passing of multiple thetas that would otherwise be summed.
        if theta_torch.shape[0] != 1:
            raise ValueError(
                "NeuralLikelihoodOp received a batch of thetas with batch_dim="
                f"{theta_torch.shape[0]}. This Op expects exactly one theta per "
                "evaluation. If you intend to evaluate multiple thetas, loop over "
                "them externally. I.i.d. observations are supported by providing "
                "observed with shape (n_obs, *event_shape); this Op sums over the "
                "observation (sample) dimension automatically."
            )

        # Estimator expects input (sample_dim, batch_dim, *event) and
        # condition (batch_dim, *params). Use sample_dim=n_obs, batch_dim=1.
        obs = self.observation  # (n_obs, *event_shape)
        # Insert batch_dim=1 → (n_obs, 1, *event_shape)
        obs_expanded = obs.unsqueeze(1)

        with torch.set_grad_enabled(True):
            log_prob = self.estimator.log_prob(
                input=obs_expanded, condition=theta_torch
            )

        # Reduce to scalar (sum over sample_dim)
        if log_prob.dim() > 0:
            log_prob = log_prob.sum()

        # Compute gradients eagerly (so grad() can reference this output symbolically)
        if log_prob.requires_grad:
            grad_list = torch.autograd.grad(
                outputs=log_prob,
                inputs=theta_torch,
                create_graph=False,
                allow_unused=False,
            )
            grad_t = grad_list[0]
            # Remove batch dimension if it was added
            if len(original_shape) == 1 and grad_t.shape[0] == 1:
                grad_t = grad_t.squeeze(0)
            grad_np = grad_t.detach().cpu().numpy()
        else:
            # Use theta's dtype for numerical outputs if possible
            grad_np = np.zeros(original_shape, dtype=theta_np.dtype)

        # Store outputs using theta dtype (PyMC typically expects floatX)
        out_dtype = (
            theta_np.dtype if np.issubdtype(theta_np.dtype, np.floating) else np.float64
        )
        output_storage[0][0] = log_prob.detach().cpu().numpy().astype(out_dtype)
        output_storage[1][0] = grad_np.astype(out_dtype)

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


def neural_likelihood_to_pymc(
    likelihood_nn: ConditionalDensityEstimator,
    theta: TensorVariable,
    observed: np.ndarray,
    name: str = "likelihood",
) -> TensorVariable:
    """Create a PyMC CustomDist from a neural likelihood estimator.

    This function wraps a trained NLE network as a PyMC distribution that can
    be used as a likelihood in a PyMC model. The likelihood is conditioned on
    the observed data and evaluates log p(x|θ).

    Observed shape conventions:
    - Single observation: observed has shape (*event_shape,)
    - I.i.d. observations: observed has shape (n_obs, *event_shape)

    The Op will construct input with sample_dim=n_obs and batch_dim=1 and
    will sum the log-probabilities over the sample (i.i.d.) dimension so that
    the returned logp is a scalar.

    Args:
        likelihood_nn: Trained ConditionalDensityEstimator from NLE
        theta: PyMC parameter variable that the likelihood depends on
        observed: Observed data to condition the likelihood on. For multiple
            i.i.d. observations, pass shape (n_obs, *event_shape).
        name: Name for the PyMC distribution

    Returns:
        PyMC CustomDist representing the neural likelihood

    Example:
        ```python
        # After training NLE
        likelihood_nn = nle.train()

        # In PyMC model
        with pm.Model() as model:
            # Prior
            theta = pm.Normal("theta", mu=0, sigma=1, shape=2)

            # Neural likelihood
            likelihood = neural_likelihood_to_pymc(
                likelihood_nn, theta, x_observed, "x"
            )

            # Sample posterior
            trace = pm.sample()
        ```
    """
    # Create the Op for this specific observation
    op = NeuralLikelihoodOp(likelihood_nn, observed)

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
              For vectorized/batched theta, wrap over this function externally.
        """
        # Optionally seed torch from the provided numpy Generator (PyMC default)
        if isinstance(rng, np.random.Generator):
            seed = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
            torch.manual_seed(seed)

        # Prepare condition tensor
        # Use estimator's device/dtype as in the Op
        # Device & dtype for sampling from first parameter (simple and robust)
        p0 = next(likelihood_nn.parameters())  # type: ignore[attr-defined]
        torch_device = p0.device
        torch_dtype = p0.dtype

        theta_t = torch.as_tensor(theta_np, dtype=torch_dtype, device=torch_device)
        if theta_t.dim() == 1:
            # add batch dim = 1 to match estimator expectation
            theta_t = theta_t.unsqueeze(0)

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

    return pm.CustomDist(name, theta, logp=logp, random=random, observed=observed)
