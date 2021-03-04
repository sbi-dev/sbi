# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import warnings

from pyro.infer.mcmc import MCMC as BaseMCMC
from pyro.infer.mcmc.api import _MultiSampler, _UnarySampler
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.nuts import NUTS


class MCMC(BaseMCMC):
    """
    Identical to Pyro's MCMC class except for `available_cpu` parameter.

    Wrapper class for Markov Chain Monte Carlo algorithms. Specific MCMC algorithms
    are TraceKernel instances and need to be supplied as a ``kernel`` argument
    to the constructor.

    .. note:: The case of `num_chains > 1` uses python multiprocessing to
        run parallel chains in multiple processes. This goes with the usual
        caveats around multiprocessing in python, e.g. the model used to
        initialize the ``kernel`` must be serializable via `pickle`, and the
        performance / constraints will be platform dependent (e.g. only
        the "spawn" context is available in Windows). This has also not
        been extensively tested on the Windows platform.

    :param kernel: An instance of the ``TraceKernel`` class, which when
        given an execution trace returns another sample trace from the target
        (posterior) distribution.
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int warmup_steps: Number of warmup iterations. The samples generated
        during the warmup phase are discarded. If not provided, default is
        half of `num_samples`.
    :param int num_chains: Number of MCMC chains to run in parallel. Depending on
        whether `num_chains` is 1 or more than 1, this class internally dispatches
        to either `_UnarySampler` or `_MultiSampler`.
    :param dict initial_params: dict containing initial tensors in unconstrained
        space to initiate the markov chain. The leading dimension's size must match
        that of `num_chains`. If not specified, parameter values will be sampled from
        the prior.
    :param hook_fn: Python callable that takes in `(kernel, samples, stage, i)`
        as arguments. stage is either `sample` or `warmup` and i refers to the
        i'th sample for the given stage. This can be used to implement additional
        logging, or more generally, run arbitrary code per generated sample.
    :param str mp_context: Multiprocessing context to use when `num_chains > 1`.
        Only applicable for Python 3.5 and above. Use `mp_context="spawn"` for
        CUDA.
    :param bool disable_progbar: Disable progress bar and diagnostics update.
    :param bool disable_validation: Disables distribution validation check. This is
        disabled by default, since divergent transitions will lead to exceptions.
        Switch to `True` for debugging purposes.
    :param dict transforms: dictionary that specifies a transform for a sample site
        with constrained support to unconstrained space.
    :param int available_cpu: Number of available CPUs, defaults to `mp.cpu_count()-1`.
        Setting it to 1 disables multiprocessing.
    """

    def __init__(
        self,
        kernel,
        num_samples,
        warmup_steps=None,
        initial_params=None,
        num_chains=1,
        hook_fn=None,
        mp_context=None,
        disable_progbar=False,
        disable_validation=True,
        transforms=None,
        available_cpu=mp.cpu_count() - 1,
    ):
        self.warmup_steps = (
            num_samples if warmup_steps is None else warmup_steps
        )  # Stan
        self.num_samples = num_samples
        self.kernel = kernel
        self.transforms = transforms
        self.disable_validation = disable_validation
        self._samples = None
        self._args = None
        self._kwargs = None
        if (
            isinstance(self.kernel, (HMC, NUTS))
            and self.kernel.potential_fn is not None
        ):
            if initial_params is None:
                raise ValueError(
                    "Must provide valid initial parameters to begin sampling"
                    " when using `potential_fn` in HMC/NUTS kernel."
                )
        parallel = False
        if num_chains > 1:
            # check that initial_params is different for each chain
            if initial_params:
                for v in initial_params.values():
                    if v.shape[0] != num_chains:
                        raise ValueError(
                            "The leading dimension of tensors in `initial_params` "
                            "must match the number of chains."
                        )
                # FIXME: probably we want to use "spawn" method by default to avoid the error
                # CUDA initialization error https://github.com/pytorch/pytorch/issues/2517
                # even that we run MCMC in CPU.
                if mp_context is None:
                    # change multiprocessing context to 'spawn' for CUDA tensors.
                    if list(initial_params.values())[0].is_cuda:
                        mp_context = "spawn"

            # verify num_chains is compatible with available CPU.
            available_cpu = max(available_cpu, 1)
            if num_chains <= available_cpu:
                parallel = True
            else:
                warnings.warn(
                    "num_chains={} is more than available_cpu={}. "
                    "Chains will be drawn sequentially.".format(
                        num_chains, available_cpu
                    )
                )
        else:
            if initial_params:
                initial_params = {k: v.unsqueeze(0) for k, v in initial_params.items()}

        self.num_chains = num_chains
        self._diagnostics = [None] * num_chains

        if parallel:
            self.sampler = _MultiSampler(
                kernel,
                num_samples,
                self.warmup_steps,
                num_chains,
                mp_context,
                disable_progbar,
                initial_params=initial_params,
                hook=hook_fn,
            )
        else:
            self.sampler = _UnarySampler(
                kernel,
                num_samples,
                self.warmup_steps,
                num_chains,
                disable_progbar,
                initial_params=initial_params,
                hook=hook_fn,
            )
