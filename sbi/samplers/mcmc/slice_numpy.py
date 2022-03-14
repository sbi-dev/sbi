# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import os
import sys
from math import ceil
from typing import Callable, Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange

from sbi.simulators.simutils import tqdm_joblib
from sbi.utils import tensor2numpy


class MCMCSampler:
    """
    Superclass for MCMC samplers.
    """

    def __init__(self, x, lp_f: Callable, thin: Optional[int], verbose: bool = False):
        """

        Args:
            x: initial state
            lp_f: Function that returns the log prob.
            thin: amount of thinning; if None, no thinning.
            verbose: Whether to show progress bars (False).
        """

        self.x = np.array(x, dtype=float)
        self.lp_f = lp_f
        self.L = lp_f(self.x)
        self.thin = 1 if thin is None else thin
        self.n_dims = self.x.size if self.x.ndim == 1 else self.x.shape[1]
        self.verbose = verbose

    def set_state(self, x):
        """
        Sets the state of the chain to x.
        """

        self.x = np.array(x, dtype=float)
        self.L = self.lp_f(self.x)

    def gen(self, n_samples):
        """
        Generates MCMC samples. Should be implemented in a subclass.
        """
        raise NotImplementedError("Should be implemented as a subclass.")


class SliceSampler(MCMCSampler):
    def __init__(
        self, x, lp_f, max_width=float("inf"), thin=None, verbose: bool = False
    ):
        """Slice sampling for multivariate continuous probability distributions.

        It cycles sampling from each conditional using univariate slice sampling.

        Args:
            x: initial state
            lp_f: Function that returns the log prob.
            max_width: maximum bracket width
            thin: amount of thinning; if None, no thinning.
            verbose: Whether to show progress bars (False).
        """

        MCMCSampler.__init__(self, x, lp_f, thin, verbose=verbose)
        self.max_width = max_width
        self.width = None

    def gen(
        self,
        n_samples: int,
        logger=sys.stdout,
        show_info: bool = False,
        rng=np.random,  # type: ignore
    ):
        """
        Return samples using slice sampling.

        Args:
            n_samples: number of samples
            logger: logger for logging messages. If None, no logging takes place
            show_info: whether to plot info at the end of sampling
            rng: random number generator to use
        Returns:
            sampels: numpy array of samples
        """

        assert n_samples >= 0, "number of samples can't be negative"

        order = list(range(self.n_dims))
        L_trace = []
        samples = np.empty([int(n_samples), int(self.n_dims)])
        logger = open(os.devnull, "w") if logger is None else logger

        if self.width is None:
            # logger.write('tuning bracket width...\n')
            self._tune_bracket_width(rng)

        tbar = trange(int(n_samples), miniters=10, disable=not self.verbose)
        tbar.set_description("Generating samples")
        for n in tbar:
            # for n in range(int(n_samples)):
            for _ in range(self.thin):

                rng.shuffle(order)

                for i in order:
                    self.x[i], _ = self._sample_from_conditional(i, self.x[i], rng)

            samples[n] = self.x.copy()

            self.L = self.lp_f(self.x)
            # logger.write('sample = {0}, log prob = {1:.2}\n'.format(n+1, self.L))

            if show_info:
                L_trace.append(self.L)

        # show trace plot
        if show_info:
            fig, ax = plt.subplots(1, 1)
            ax.plot(L_trace)
            ax.set_ylabel("log probability")
            ax.set_xlabel("samples")
            plt.show(block=False)

        return samples

    def _tune_bracket_width(self, rng):
        """
        Initial test run for tuning bracket width.
        Note that this is not correct sampling; samples are thrown away.

        Args:
            rng: Random number generator to use.
        """

        n_samples = 50
        order = list(range(self.n_dims))
        x = self.x.copy()
        self.width = np.full(self.n_dims, 0.01)

        tbar = trange(n_samples, miniters=10, disable=not self.verbose)
        tbar.set_description("Tuning bracket width...")
        for n in tbar:
            # for n in range(int(n_samples)):
            rng.shuffle(order)

            for i in range(self.n_dims):
                x[i], wi = self._sample_from_conditional(i, x[i], rng)
                self.width[i] += (wi - self.width[i]) / (n + 1)

    def _sample_from_conditional(self, i: int, cxi, rng):
        """
        Samples uniformly from conditional by constructing a bracket.

        Args:
            i: conditional to sample from
            cxi: current state of variable to sample
            rng: random number generator to use

        Returns:
            new state, final bracket width
        """
        assert self.width is not None, "Chain not initialized."

        # conditional log prob
        Li = lambda t: self.lp_f(np.concatenate([self.x[:i], [t], self.x[i + 1 :]]))
        wi = self.width[i]

        # sample a slice uniformly
        logu = Li(cxi) + np.log(1.0 - rng.rand())

        # position the bracket randomly around the current sample
        lx = cxi - wi * rng.rand()
        ux = lx + wi

        # find lower bracket end
        while Li(lx) >= logu and cxi - lx < self.max_width:
            lx -= wi

        # find upper bracket end
        while Li(ux) >= logu and ux - cxi < self.max_width:
            ux += wi

        # sample uniformly from bracket
        xi = (ux - lx) * rng.rand() + lx

        # if outside slice, reject sample and shrink bracket
        while Li(xi) < logu:
            if xi < cxi:
                lx = xi
            else:
                ux = xi
            xi = (ux - lx) * rng.rand() + lx

        return xi, ux - lx


class SliceSamplerVectorized:
    def __init__(
        self,
        log_prob_fn: Callable,
        init_params: np.ndarray,
        num_chains: int = 1,
        tuning: int = 50,
        verbose: bool = True,
        init_width: Union[float, np.ndarray] = 0.01,
        max_width: float = float("inf"),
    ):
        """Slice sampler in pure Numpy, vectorized evaluations across chains.

        Args:
            log_prob_fn: Log prob function.
            init_params: Initial parameters.
            verbose: Show/hide additional info such as progress bars.
            tuning: Number of tuning steps for brackets.
            init_width: Inital width of brackets.
            max_width: Maximum width of brackets.
        """
        self._log_prob_fn = log_prob_fn

        self.x = init_params
        self.num_chains = num_chains
        self.tuning = tuning
        self.verbose = verbose

        self.init_width = init_width
        self.max_width = max_width

        self.n_dims = self.x.size

        self._reset()

    def _reset(self):
        self.rng = np.random  # type: ignore
        self.state = {}
        for c in range(self.num_chains):
            self.state[c] = {}
            self.state[c]["t"] = 0
            self.state[c]["width"] = None
            self.state[c]["x"] = None

    def run(self, num_samples: int) -> np.ndarray:
        """Runs MCMC

        Args:
            num_samples: Number of samples to generate

        Returns:
            MCMC samples
        """
        assert num_samples >= 0

        self.n_dims = self.x.shape[1]

        # Init chains
        for c in range(self.num_chains):
            self.state[c]["x"] = self.x[c, :]

            self.state[c]["i"] = 0
            self.state[c]["order"] = list(range(self.n_dims))
            self.rng.shuffle(self.state[c]["order"])

            self.state[c]["samples"] = np.empty([int(num_samples), int(self.n_dims)])

            self.state[c]["state"] = "BEGIN"

            self.state[c]["width"] = np.full(self.n_dims, self.init_width)

        if self.verbose:
            pbar = tqdm(
                range(self.num_chains * num_samples),
                desc=f"Running vectorized MCMC with {self.num_chains} chains",
            )

        num_chains_finished = 0
        while num_chains_finished != self.num_chains:

            num_chains_finished = 0

            for sc in self.state.values():
                if sc["state"] == "BEGIN":
                    sc["cxi"] = sc["x"][sc["order"][sc["i"]]]
                    sc["wi"] = sc["width"][sc["order"][sc["i"]]]
                    sc["next_param"] = np.concatenate(
                        [
                            sc["x"][: sc["order"][sc["i"]]],
                            [sc["cxi"]],
                            sc["x"][sc["order"][sc["i"]] + 1 :],
                        ]
                    )

            params = np.stack([sc["next_param"] for sc in self.state.values()])
            log_probs = self._log_prob_fn(params)

            for c in range(self.num_chains):
                sc = self.state[c]

                if sc["state"] == "BEGIN":
                    # position the bracket randomly around the current sample
                    sc["logu"] = log_probs[c] + np.log(1.0 - self.rng.rand())
                    sc["lx"] = sc["cxi"] - sc["wi"] * self.rng.rand()
                    sc["ux"] = sc["lx"] + sc["wi"]
                    sc["next_param"] = np.concatenate(
                        [
                            sc["x"][: sc["order"][sc["i"]]],
                            [sc["lx"]],
                            sc["x"][sc["order"][sc["i"]] + 1 :],
                        ]
                    )
                    sc["state"] = "LOWER"

                elif sc["state"] == "LOWER":
                    outside_lower = (
                        log_probs[c] >= sc["logu"]
                        and sc["cxi"] - sc["lx"] < self.max_width
                    )

                    if outside_lower:
                        sc["lx"] -= sc["wi"]
                        sc["next_param"] = np.concatenate(
                            [
                                sc["x"][: sc["order"][sc["i"]]],
                                [sc["lx"]],
                                sc["x"][sc["order"][sc["i"]] + 1 :],
                            ]
                        )

                    else:
                        sc["next_param"] = np.concatenate(
                            [
                                sc["x"][: sc["order"][sc["i"]]],
                                [sc["ux"]],
                                sc["x"][sc["order"][sc["i"]] + 1 :],
                            ]
                        )
                        sc["state"] = "UPPER"

                elif sc["state"] == "UPPER":
                    outside_upper = (
                        log_probs[c] >= sc["logu"]
                        and sc["ux"] - sc["cxi"] < self.max_width
                    )

                    if outside_upper:
                        sc["ux"] += sc["wi"]
                        sc["next_param"] = np.concatenate(
                            [
                                sc["x"][: sc["order"][sc["i"]]],
                                [sc["ux"]],
                                sc["x"][sc["order"][sc["i"]] + 1 :],
                            ]
                        )
                    else:
                        # sample uniformly from bracket
                        sc["xi"] = (sc["ux"] - sc["lx"]) * self.rng.rand() + sc["lx"]
                        sc["next_param"] = np.concatenate(
                            [
                                sc["x"][: sc["order"][sc["i"]]],
                                [sc["xi"]],
                                sc["x"][sc["order"][sc["i"]] + 1 :],
                            ]
                        )
                        sc["state"] = "SAMPLE_SLICE"

                elif sc["state"] == "SAMPLE_SLICE":
                    # if outside slice, reject sample and shrink bracket
                    rejected = log_probs[c] < sc["logu"]

                    if rejected:
                        if sc["xi"] < sc["cxi"]:
                            sc["lx"] = sc["xi"]
                        else:
                            sc["ux"] = sc["xi"]
                        sc["xi"] = (sc["ux"] - sc["lx"]) * self.rng.rand() + sc["lx"]
                        sc["next_param"] = np.concatenate(
                            [
                                sc["x"][: sc["order"][sc["i"]]],
                                [sc["xi"]],
                                sc["x"][sc["order"][sc["i"]] + 1 :],
                            ]
                        )

                    else:
                        if sc["t"] < num_samples:
                            sc["state"] = "BEGIN"

                            sc["x"] = sc["next_param"].copy()

                            if sc["t"] <= (self.tuning):
                                i = sc["order"][sc["i"]]
                                sc["width"][i] += (
                                    (sc["ux"] - sc["lx"]) - sc["width"][i]
                                ) / (sc["t"] + 1)

                            if sc["i"] < len(sc["order"]) - 1:
                                sc["i"] += 1

                            else:
                                if sc["t"] > self.tuning:
                                    sc["samples"][sc["t"]] = sc["x"].copy()

                                sc["t"] += 1

                                self.state[c]["i"] = 0
                                self.state[c]["order"] = list(range(self.n_dims))
                                self.rng.shuffle(self.state[c]["order"])

                                if self.verbose:
                                    if sc["t"] % 10 == 0:
                                        pbar.update(10)  # type: ignore

                        else:
                            sc["state"] = "DONE"

                if sc["state"] == "DONE":
                    num_chains_finished += 1

        samples = np.stack([self.state[c]["samples"] for c in range(self.num_chains)])

        return samples


def slice_np_parallized(
    potential_function: Callable,
    initial_params: torch.Tensor,
    num_samples: int,
    thin: int,
    warmup_steps: int,
    vectorized: bool,
    num_workers: int = 1,
    show_progress_bars: bool = False,
):
    """Run slice np (vectorized) parallized over CPU cores.

    In case of the vectorized version of slice np parallization happens over batches of
    chains to still exploit vectorization.

    MCMC progress bars are omitted if num_workers > 1 to reduce clutter. Instead the
    progress over chains is shown.

    Args:
        potential_function: potential function
        initial_params: initital parameters, one for each chain
        num_samples: number of MCMC samples to produce
        thin: thinning factor
        warmup_steps: number of warmup / burnin steps
        vectorized: whether to use the vectorized version
        num_workers: number of CPU cores to use
        show_progress_bars: whether to show progress bars

    Returns:
        Tensor: final MCMC samples of each chain (num_chains, num_samples, dim_samples)
    """
    num_chains, dim_samples = initial_params.shape

    # Generate seeds for workers from current random state.
    seeds = torch.randint(high=2**31, size=(num_chains,))

    if not vectorized:
        # Define run function for given input.
        def run_slice_np(inits, seed):
            # Seed current job.
            np.random.seed(seed)
            posterior_sampler = SliceSampler(
                tensor2numpy(inits).reshape(-1),
                lp_f=potential_function,
                thin=thin,
                # Show pbars of workers only for single worker
                verbose=show_progress_bars and num_workers == 1,
            )
            if warmup_steps > 0:
                posterior_sampler.gen(int(warmup_steps))
            return posterior_sampler.gen(ceil(num_samples / num_chains))

        # For sequential chains each batch has only a single chain.
        batch_size = 1
        run_fun = run_slice_np

    else:  # Sample all chains at the same time

        # Define local function to run a batch of chains vectorized.
        def run_slice_np_vectorized(inits, seed):
            # Seed current job.
            np.random.seed(seed)
            posterior_sampler = SliceSamplerVectorized(
                init_params=tensor2numpy(inits),
                log_prob_fn=potential_function,
                num_chains=inits.shape[0],
                # Show pbars of workers only for single worker
                verbose=show_progress_bars and num_workers == 1,
            )
            warmup_ = warmup_steps * thin
            num_samples_ = ceil((num_samples * thin) / num_chains)
            samples = posterior_sampler.run(warmup_ + num_samples_)
            samples = samples[:, warmup_:, :]  # discard warmup steps
            samples = samples[:, ::thin, :]  # thin chains
            samples = torch.from_numpy(samples)  # chains x samples x dim
            return samples

        # For vectorized case a batch contains multiple chains to exploit vectorization.
        batch_size = ceil(num_chains / num_workers)
        run_fun = run_slice_np_vectorized

    # Parallize over batch of chains.
    initial_params_in_batches = torch.split(initial_params, batch_size, dim=0)
    num_batches = len(initial_params_in_batches)

    # Show progress bars over batches.
    with tqdm_joblib(
        tqdm(
            range(num_batches),  # type: ignore
            disable=not show_progress_bars or num_workers == 1,
            desc=f"""Running {num_chains} MCMC chains with {num_workers} worker{"s" if
                  num_workers>1 else ""} (batch_size={batch_size}).""",
            total=num_chains,
        )
    ):
        all_samples = Parallel(n_jobs=num_workers)(
            delayed(run_fun)(initial_params_batch, seed)
            for initial_params_batch, seed in zip(initial_params_in_batches, seeds)
        )
        all_samples = np.stack(all_samples).astype(np.float32)
        samples = torch.from_numpy(all_samples)

    return samples.reshape(num_chains, -1, dim_samples)  # chains x samples x dim
