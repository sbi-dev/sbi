from typing import Callable, Optional

import numpy as np
import pymc
import pytensor.tensor as pt
import torch

from sbi.inference.potentials.base_potential import BasePotential
from sbi.utils import tensor2numpy


class PyMCPotential(pt.Op):
    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [
        pt.dscalar,
        pt.dvector,
    ]  # outputs a single scalar value (the potential) and gradients for every input
    default_output = 0  # return only potential by default

    def __init__(
        self,
        potential_fn: BasePotential,
        device: str,
        track_gradients: bool = True,
    ):
        """PyTensor operation wrapping a `BasePotential` callable for use
        with PyMC samplers.

        Args:
            potential_fn: Potential function that returns a potential given parameters
            device: The device to which to move the parameters before evaluation.
            track_gradients: Whether to track gradients from potential function
        """
        self.potential_fn = potential_fn
        self.device = device
        self.track_gradients = track_gradients

    def perform(self, node, inputs, outputs) -> None:
        """Compute potential and possibly gradients from input parameters"""
        # unpack and handle inputs
        params = inputs[0]
        params = (
            torch.tensor(params)
            .to(torch.float32)
            .to(self.device)
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

    def grad(self, inputs, g):
        """Get gradients computed from `perform` and return Jacobian-Vector product"""
        # get outputs from forward pass (but doesn't re-compute it, I think...)
        value = self(*inputs)
        gradients = value.owner.outputs[1:]
        # compute and return JVP
        return [(g[0] * grad) for grad in gradients]


class PyMCSampler:
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
        track_gradients = True if step in (pymc.NUTS, pymc.HamiltonianMC) else False
        self._model = pymc.Model()
        potential = PyMCPotential(
            potential_fn, track_gradients=track_gradients, device=device
        )
        with self._model:
            params = pymc.Normal(
                self.param_name, mu=initvals.mean(axis=0)
            )  # dummy prior
            pymc.Potential("likelihood", potential(params))

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
                initvals=self._initvals,
                chains=self._chains,
                progressbar=self._progressbar,
                mp_ctx=self._mp_ctx,
            )
        self._inference_data = inference_data
        samples = getattr(inference_data.posterior, self.param_name).data
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
        samples = getattr(self._inference_data.posterior, self.param_name).data
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
