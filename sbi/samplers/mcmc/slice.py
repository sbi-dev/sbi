# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Dict, Optional

import torch
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from torch import Tensor


class Slice(MCMCKernel):
    def __init__(
        self,
        model: Optional[Callable] = None,
        potential_fn: Optional[Callable] = None,
        initial_width: float = 0.01,
        max_width=float("inf"),
        transforms: Optional[Dict] = None,
        max_plate_nesting: Optional[int] = None,
        jit_compile: Optional[bool] = False,
        jit_options: Optional[Dict] = None,
        ignore_jit_warnings: bool = False,
    ) -> None:
        """
        Slice sampling kernel [1].

        During the warmup phase, the width of the bracket is adapted, starting from
        the provided initial width.

        **References**

        [1] `Slice Sampling <https://doi.org/10.1214/aos/1056562461>`_,
            Radford M. Neal

        Args:
            model: Python callable containing Pyro primitives.
            potential_fn: Python callable calculating potential energy with input
                is a dict of real support parameters.
            initial_width: Initial bracket width
            max_width: Maximum bracket width
            transforms: Optional dictionary that specifies a transform
                for a sample site with constrained support to unconstrained space. The
                transform should be invertible, and implement `log_abs_det_jacobian`.
                If not specified and the model has sites with constrained support,
                automatic transformations will be applied, as specified in
                :mod:`torch.distributions.constraint_registry`.
            max_plate_nesting: Optional bound on max number of nested
                :func:`pyro.plate` contexts. This is required if model contains
                discrete sample sites that can be enumerated over in parallel.
            jit_compile: Optional parameter denoting whether to use
                the PyTorch JIT to trace the log density computation, and use this
                optimized executable trace in the integrator.
            jit_options: A dictionary contains optional arguments for
                :func:`torch.jit.trace` function.
            ignore_jit_warnings: Flag to ignore warnings from the JIT
                tracer when ``jit_compile=True``. Default is False.
        """
        if not ((model is None) ^ (potential_fn is None)):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        # NB: deprecating args - model, transforms
        self.model = model
        self.transforms = transforms
        self._max_plate_nesting = max_plate_nesting
        self._jit_compile = jit_compile
        self._jit_options = jit_options
        self._ignore_jit_warnings = ignore_jit_warnings

        self.potential_fn = potential_fn

        self._initial_width = initial_width
        self._max_width = max_width

        self._reset()

        super(Slice, self).__init__()

    def _reset(self):
        self._t = 0
        self._width: Optional[Tensor] = None
        self._num_dimensions: Optional[int] = None
        self._initial_params: Optional[Dict] = None
        self._site_name = None

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)

        # TODO: Clean up required for multiple sites
        assert self.initial_params is not None
        self._site_name = next(iter(self.initial_params.keys()))
        self._num_dimensions = next(iter(self.initial_params.values())).shape[-1]

        self._width = torch.full((self._num_dimensions,), self._initial_width)

    @property
    def initial_params(self):
        return deepcopy(self._initial_params)

    @initial_params.setter
    def initial_params(self, params):
        assert (
            isinstance(params, dict) and len(params) == 1
        ), "Slice sampling only implemented for a single site."  # TODO: Implement
        self._initial_params = params

    def _initialize_model_properties(self, model_args, model_kwargs):
        init_params, potential_fn, transforms, trace = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            transforms=self.transforms,
            max_plate_nesting=self._max_plate_nesting,
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
        )
        self.potential_fn = potential_fn
        self.transforms = transforms
        if self._initial_params is None:
            self.initial_params = init_params
        self._prototype_trace = trace

    def cleanup(self):
        self._reset()

    def sample(self, params):
        assert (
            self._num_dimensions is not None and self._width is not None
        ), "Chain not initialized."

        for dim in torch.randperm(self._num_dimensions):
            # cast for pyright.
            idx = int(dim.item())
            (
                params[self._site_name].view(-1)[idx],
                width_d,
            ) = self._sample_from_conditional(params, idx)
            if self._t < self._warmup_steps:
                # TODO: Other schemes for tuning bracket width?
                self._width[idx] += (width_d.item() - self._width[idx]) / (self._t + 1)

        self._t += 1

        return params.copy()

    def _sample_from_conditional(self, params, dim):
        # TODO: Flag for doubling and stepping out procedures, see Neal paper, and also:
        # https://pints.readthedocs.io/en/latest/mcmc_samplers/slice_doubling_mcmc.html
        # https://pints.readthedocs.io/en/latest/mcmc_samplers/slice_stepout_mcmc.html

        def _log_prob_d(x):
            assert self.potential_fn is not None, "Chain not initialized."

            return -self.potential_fn(
                {
                    self._site_name: torch.cat(
                        (
                            params[self._site_name].view(-1)[:dim],
                            x.reshape(1),
                            params[self._site_name].view(-1)[dim + 1 :],
                        )
                    ).unsqueeze(
                        0
                    )  # TODO: The unsqueeze seems to give a speed up, figure out when
                    # this is the case exactly
                }
            )

        assert (
            self._site_name is not None and self._width is not None
        ), "Chain not initialized."

        # Sample uniformly from slice
        log_height = _log_prob_d(params[self._site_name].view(-1)[dim]) + torch.log(
            torch.rand(1, device=params[self._site_name].device)
        )

        # Position the bracket randomly around the current sample
        lower = params[self._site_name].view(-1)[dim] - self._width[dim] * torch.rand(
            1, device=params[self._site_name].device
        )
        upper = lower + self._width[dim]

        # Find lower bracket end
        while (
            _log_prob_d(lower) >= log_height
            and params[self._site_name].view(-1)[dim] - lower < self._max_width
        ):
            lower -= self._width[dim]

        # Find upper bracket end
        while (
            _log_prob_d(upper) >= log_height
            and upper - params[self._site_name].view(-1)[dim] < self._max_width
        ):
            upper += self._width[dim]

        # Sample uniformly from bracket
        new_parameter = (upper - lower) * torch.rand(
            1, device=params[self._site_name].device
        ) + lower

        # If outside slice, reject sample and shrink bracket
        while _log_prob_d(new_parameter) < log_height:
            if new_parameter < params[self._site_name].view(-1)[dim]:
                lower = new_parameter
            else:
                upper = new_parameter
            new_parameter = (upper - lower) * torch.rand(
                1, device=params[self._site_name].device
            ) + lower

        return new_parameter, upper - lower
