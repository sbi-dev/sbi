# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution
from zuko.distributions import NormalizingFlow

from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.score_fn_iid import get_iid_method
from sbi.neural_nets.estimators import ConditionalVectorFieldEstimator
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.samplers.ode_solvers import build_neural_ode
from sbi.sbi_types import TorchTransform
from sbi.utils.sbiutils import mcmc_transform, within_support
from sbi.utils.torchutils import ensure_theta_batched


def vector_field_estimator_based_potential(
    vector_field_estimator: ConditionalVectorFieldEstimator,
    prior: Optional[Distribution],
    x_o: Optional[Tensor],
    enable_transform: bool = True,
    **kwargs,
) -> Tuple["VectorFieldBasedPotential", TorchTransform]:
    r"""Returns the potential function gradient for vector field estimators.

    Args:
        vector_field_estimator: The neural network modelling the vector field.
        prior: The prior distribution.
        x_o: The observed data at which to evaluate the vector field.
        enable_transform: Whether to enable transforms. Not supported yet.
        **kwargs: Additional keyword arguments passed to
            `VectorFieldBasedPotential`.
    """
    device = str(next(vector_field_estimator.parameters()).device)

    potential_fn = VectorFieldBasedPotential(
        vector_field_estimator, prior, x_o, device=device, **kwargs
    )

    if prior is not None:
        theta_transform = mcmc_transform(
            prior, device=device, enable_transform=enable_transform
        )
    else:
        theta_transform = torch.distributions.transforms.identity_transform

    return potential_fn, theta_transform


class VectorFieldBasedPotential(BasePotential):
    def __init__(
        self,
        vector_field_estimator: ConditionalVectorFieldEstimator,
        prior: Optional[Distribution],  # type: ignore
        x_o: Optional[Tensor] = None,
        iid_method: Literal["fnpe", "gauss", "auto_gauss", "jac_gauss"] = "auto_gauss",
        iid_params: Optional[Dict[str, Any]] = None,
        device: Union[str, torch.device] = "cpu",
        neural_ode_backend: str = "zuko",
        neural_ode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Potential class for vector field estimators. Implements the potential function
        via the probability flow ODE and the gradient via the score estimator. If
        the vector field estimator does not define the score (SCORE_DEFINED = False),
        the gradient is not available and an error is raised.

        Note that the potential function is not defined for the iid setting yet.

        Args:
            vector_field_estimator: The neural network modelling the vector field.
            prior: The prior distribution.
            x_o: The observed data at which to evaluate the posterior.
            iid_method: Which method to use for computing the score in the iid setting.
                We currently support "fnpe", "gauss", "auto_gauss", "jac_gauss".
            iid_params: Parameters for the iid method, for arguments see
                `IIDScoreFunction`.
            device: The device on which to evaluate the potential.
            neural_ode_backend: The backend to use for the neural ODE. Currently,
                only "zuko" is supported.
            neural_ode_kwargs: Additional keyword arguments for the neural ODE.
        """
        self.vector_field_estimator = vector_field_estimator
        self.vector_field_estimator.eval()
        self.iid_method = iid_method
        self.iid_params = iid_params

        neural_ode_kwargs = neural_ode_kwargs or {}
        self.neural_ode = build_neural_ode(
            self.vector_field_estimator.ode_fn,
            self.vector_field_estimator.net,
            self.vector_field_estimator.mean_base,
            self.vector_field_estimator.std_base,
            backend=neural_ode_backend,
            t_min=self.vector_field_estimator.t_min,
            t_max=self.vector_field_estimator.t_max,
            **neural_ode_kwargs,
        )

        super().__init__(prior, x_o, device=device)

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Moves score_estimator, prior and x_o to the given device.

        It also sets the device attribute to the given device.

        Args:
            device: Device to move the score_estimator, prior and x_o to.
        """

        self.device = device
        self.vector_field_estimator.to(device)
        if self.prior:
            self.prior.to(device)  # type: ignore
        if self._x_o is not None:
            self._x_o = self._x_o.to(device)

    def set_x(
        self,
        x_o: Optional[Tensor],
        x_is_iid: Optional[bool] = False,
        iid_method: Literal["fnpe", "gauss", "auto_gauss", "jac_gauss"] = "auto_gauss",
        iid_params: Optional[Dict[str, Any]] = None,
        **ode_kwargs,
    ):
        """
        Set the observed data and whether it is IID.

        Rebuilds the continuous normalizing flow if the observed data is set.

        Args:
            x_o: The observed data.
            x_is_iid: Whether the observed data is IID (if batch_dim>1).
            iid_method: Which method to use for computing the score in the iid setting.
                We currently support "fnpe", "gauss", "auto_gauss", "jac_gauss".
            iid_params: Parameters for the iid method, for arguments see
                `IIDScoreFunction`.
            ode_kwargs: Additional keyword arguments for the neural ODE.
        """
        super().set_x(x_o, x_is_iid)
        self.iid_method = iid_method
        self.iid_params = iid_params
        # NOTE: Once IID potential evaluation is supported. This needs to be adapted.
        # See #1450.
        if not x_is_iid and (self._x_o is not None):
            self.flow = self.rebuild_flow(**ode_kwargs)

    def __call__(
        self,
        theta: Tensor,
        track_gradients: bool = False,
    ) -> Tensor:
        """
        Return the potential (posterior log prob) via probability flow ODE.

        Args:
            theta: The parameters at which to evaluate the potential.
            track_gradients: Whether to track gradients. Default is False.

        Returns:
            The potential function, i.e., the log probability of the posterior.
        """
        # TODO: incorporate iid setting. See issue #1450 and PR #1508
        if self.x_is_iid:
            if (
                self.vector_field_estimator.MARGINALS_DEFINED
                and self.vector_field_estimator.SCORE_DEFINED
            ):
                raise NotImplementedError(
                    "Potential function evaluation in the "
                    "IID setting is not yet supported"
                    " for vector field based methods. "
                    "Sampling does however work via `.sample`. "
                    "If you intended to evaluate the posterior "
                    "given a batch of (non-iid) "
                    "x use `log_prob_batched`."
                )
            else:
                raise NotImplementedError(
                    "IID is not supported for this vector field estimator "
                    "since the required methods (marginals or score) are not defined."
                )

        theta = ensure_theta_batched(torch.as_tensor(theta))
        theta_density_estimator = reshape_to_sample_batch_event(
            theta, theta.shape[1:], leading_is_sample=True
        )
        self.vector_field_estimator.eval()

        with torch.set_grad_enabled(track_gradients):
            log_probs = self.flow.log_prob(theta_density_estimator).squeeze(-1)
            # Force probability to be zero outside prior support.
            in_prior_support = within_support(self.prior, theta)

            masked_log_prob = torch.where(
                in_prior_support,
                log_probs,
                torch.tensor(float("-inf"), dtype=torch.float32, device=self.device),
            )
            return masked_log_prob

    def gradient(
        self,
        theta: Tensor,
        time: Optional[Tensor] = None,
        track_gradients: bool = False,
    ) -> Tensor:
        r"""Returns the potential function gradient for score-based methods.

        Args:
            theta: The parameters at which to evaluate the potential gradient.
            time: The diffusion time. If None, then `t_min` of the
                self.vector_field_estimator is used
                (i.e. we evaluate the gradient of the actual data distribution).
            track_gradients: Whether to track gradients. Default is False.

        Returns:
            The gradient of the potential function.

        Raises:
            ValueError: If the score is not defined for this vector field estimator.
        """
        if not self.vector_field_estimator.SCORE_DEFINED:
            raise ValueError(
                "Gradient is not available since the score"
                "is not defined for this vector field estimator."
            )

        if time is None:
            time = torch.tensor([self.vector_field_estimator.t_min])

        if self._x_o is None:
            raise ValueError(
                "No observed data x_o is available. Please reinitialize \
                the potential or manually set self._x_o."
            )

        with torch.set_grad_enabled(track_gradients):
            if not self.x_is_iid or self._x_o.shape[0] == 1:
                score = self.vector_field_estimator.score(
                    input=theta, condition=self.x_o, t=time
                )
            else:
                assert self.prior is not None, "Prior is required for iid methods."

                iid_method = get_iid_method(self.iid_method)
                score_fn_iid = iid_method(
                    self.vector_field_estimator, self.prior, **(self.iid_params or {})
                )

                score = score_fn_iid(theta, self.x_o, time)

        return score

    def rebuild_flow(self, **kwargs) -> NormalizingFlow:
        """
        Rebuilds the continuous normalizing flow. This is used when
        a new default x is set, or to evaluate the log probs at higher precision.
        """
        if self._x_o is None:
            raise ValueError(
                "No observed data x_o is available. Please reinitialize \
                the potential or manually set self._x_o."
            )
        x_density_estimator = reshape_to_batch_event(
            self.x_o, event_shape=self.vector_field_estimator.condition_shape
        )

        flow = self.neural_ode(x_density_estimator, **kwargs)
        return flow


class DifferentiablePotentialFunction(torch.autograd.Function):
    """
    A wrapper of `VectorFieldBasedPotential` with a custom autograd function
    to compute the gradient of log_prob with respect to theta. Instead of
    backpropagating through the continuous normalizing flow, we use the gradient
    of the score estimator.

    """

    @staticmethod
    def forward(ctx, input, call_function, gradient_function):
        """
        Computes the potential normally.
        """
        # Save the methods as callables
        ctx.call_function = call_function
        ctx.gradient_function = gradient_function
        ctx.save_for_backward(input)

        # Perform the forward computation
        output = call_function(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad = ctx.gradient_function(input)
        # Match dims
        while len(grad_output.shape) < len(grad.shape):
            grad_output = grad_output.unsqueeze(-1)
        grad_input = grad_output * grad
        return grad_input, None, None


class CallableDifferentiablePotentialFunction:
    """
    This class handles the forward and backward functions from the potential function
    that can be passed to DifferentiablePotentialFunction, as torch.autograd.Function
    only supports static methods, and so it can't be given the potential class directly.
    """

    def __init__(self, vector_field_based_potential: VectorFieldBasedPotential):
        self.vector_field_based_potential = vector_field_based_potential

    def __call__(self, input):
        return DifferentiablePotentialFunction.apply(
            input,
            self.vector_field_based_potential.__call__,
            self.vector_field_based_potential.gradient,
        )
