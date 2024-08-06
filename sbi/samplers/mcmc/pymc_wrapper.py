from typing import Any, Callable, Optional

import numpy as np
import pymc
import pytensor.tensor as pt
import torch
from arviz.data import InferenceData

from sbi.utils.torchutils import tensor2numpy


class PyMCPotential(pt.Op):  # type: ignore
    """PyTensor Op wrapping a callable potential function"""

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [
        pt.dscalar,
        pt.dvector,
    ]  # outputs a single scalar value (the potential) and gradients for every input
    default_output = 0  # return only potential by default

    def __init__(
        self,
        potential_fn: Callable,
        device: str,
        track_gradients: bool = True,
    ):
        """PyTensor Op wrapping a callable potential function for use
        with PyMC samplers.

        Args:
            potential_fn: Potential function that returns a potential given parameters
            device: The device to which to move the parameters before evaluation.
            track_gradients: Whether to track gradients from potential function
        """
        self.potential_fn = potential_fn
        self.device = device
        self.track_gradients = track_gradients

    def perform(self, node: Any, inputs: Any, outputs: Any) -> None:
        """Compute potential and possibly gradients from input parameters

        Args:
            node: A "node" that represents the computation, handled internally
                by PyTensor.
            inputs: A sequence of inputs to the operation of type `itypes`. In this
                case, the sequence will contain one array containing the
                simulator parameters.
            outputs: A sequence allocated for storing operation outputs. In this
                case, the sequence will contain one scalar for the computed potential
                and an array containing the gradient of the potential with respect
                to the simulator parameters.
        """
        # unpack and handle inputs
        params = inputs[0]
        params = (
            torch.tensor(params)
            .to(device=self.device, dtype=torch.float32)
            .requires_grad_(self.track_gradients)
        )

        # call the potential function
        energy = self.potential_fn(params, track_gradients=self.track_gradients)

        # output the log-likelihood
        outputs[0][0] = tensor2numpy(energy).astype(np.float64)

        # compute and record gradients if desired
        if self.track_gradients:
            energy.backward()
            grads = params.grad
            outputs[1][0] = tensor2numpy(grads).astype(np.float64)
        else:
            outputs[1][0] = np.zeros(params.shape, dtype=np.float64)

    def grad(self, inputs: Any, output_grads: Any) -> list:
        """Get gradients computed from `perform` and return Jacobian-Vector product

        Args:
            inputs: A sequence of inputs to the operation of type `itypes`. In this
                case, the sequence will contain one array containing the
                simulator parameters.
            output_grads: A sequence of the gradients of the output variables. The first
                element will be the gradient of the output of the whole computational
                graph with respect to the output of this specific operation, i.e.,
                the potential.

        Returns:
            A list containing the gradient of the output of the whole computational
            graph with respect to the input of this operation, i.e.,
            the simulator parameters.
        """
        # get outputs from forward pass (but doesn't re-compute it, I think...)
        value = self(*inputs)
        gradients = value.owner.outputs[1:]  # type: ignore
        # compute and return JVP
        return [(output_grads[0] * grad) for grad in gradients]


class PyMCSampler:
    """Interface for PyMC samplers"""

    def __init__(
        self,
        potential_fn: Callable,
        initvals: np.ndarray,
        step: str = "nuts",
        draws: int = 1000,
        tune: int = 1000,
        chains: Optional[int] = None,
        mp_ctx: str = "spawn",
        progressbar: bool = True,
        param_name: str = "theta",
        device: str = "cpu",
    ):
        """Interface for PyMC samplers

        Args:
            potential_fn: Potential function from density estimator.
            initvals: Initial parameters.
            step: One of `"slice"`, `"hmc"`, or `"nuts"`.
            draws: Number of total samples to draw.
            tune: Number of tuning steps to take.
            chains: Number of MCMC chains to run in parallel.
            mp_ctx: Multiprocessing context for parallel sampling.
            progressbar: Whether to show/hide progress bars.
            param_name: Name for parameter variable, for PyMC and ArviZ structures
            device: The device to which to move the parameters for potential_fn.
        """
        self.param_name = param_name
        self._step = step
        self._draws = draws
        self._tune = tune
        self._initvals = [{self.param_name: iv} for iv in initvals]
        self._chains = chains
        self._mp_ctx = mp_ctx
        self._progressbar = progressbar
        self._device = device

        # create PyMC model object
        track_gradients = step in ("nuts", "hmc")
        self._model = pymc.Model()
        potential = PyMCPotential(
            potential_fn, track_gradients=track_gradients, device=device
        )
        with self._model:
            pymc.DensityDist(
                self.param_name, logp=potential, size=(initvals.shape[-1],)
            )

    def run(self) -> np.ndarray:
        """Run MCMC with PyMC

        Returns:
            MCMC samples
        """
        step_class = dict(slice=pymc.Slice, hmc=pymc.HamiltonianMC, nuts=pymc.NUTS)
        with self._model:
            inference_data = pymc.sample(
                step=step_class[self._step](),
                tune=self._tune,
                draws=self._draws,
                initvals=self._initvals,  # type: ignore
                chains=self._chains,
                progressbar=self._progressbar,
                mp_ctx=self._mp_ctx,
            )
        self._inference_data = inference_data
        traces = inference_data.posterior  # type: ignore
        samples = getattr(traces, self.param_name).data
        return samples

    def get_samples(
        self, num_samples: Optional[int] = None, group_by_chain: bool = True
    ) -> np.ndarray:
        """Returns samples from last call to self.run.

        Raises ValueError if no samples have been generated yet.

        Args:
            num_samples: Number of samples to return (for each chain if grouped by
                chain), if too large, all samples are returned (no error).
            group_by_chain: Whether to return samples grouped by chain (chain x samples
                x dim_params) or flattened (all_samples, dim_params).

        Returns:
            samples
        """
        if self._inference_data is None:
            raise ValueError("No samples found from MCMC run.")
        # if not grouped by chain, flatten samples into (all_samples, dim_params)
        traces = self._inference_data.posterior  # type: ignore
        samples = getattr(traces, self.param_name).data
        if not group_by_chain:
            samples = samples.reshape(-1, samples.shape[-1])

        # if not specified return all samples
        if num_samples is None:
            return samples
        # otherwise return last num_samples (for each chain when grouped).
        elif group_by_chain:
            return samples[:, -num_samples:, :]
        else:
            return samples[-num_samples:, :]

    def get_inference_data(self) -> InferenceData:
        """Returns InferenceData from last call to self.run,
        which contains diagnostic information in addition to samples

        Raises ValueError if no samples have been generated yet.

        Returns:
            InferenceData containing samples and sampling run information
        """
        if self._inference_data is None:
            raise ValueError("No samples found from MCMC run.")
        return self._inference_data
