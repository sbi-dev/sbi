# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import os
import sys
from math import ceil
from typing import Callable, Optional, Union
from warnings import warn

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
        self,
        x,
        lp_f,
        max_width=float("inf"),
        init_width: Union[float, np.ndarray] = 0.01,
        thin=None,
        tuning: int = 50,
        verbose: bool = False,
    ):
        """Slice sampling for multivariate continuous probability distributions.

        It cycles sampling from each conditional using univariate slice sampling.

        Args:
            x: Initial state.
            lp_f: Function that returns the log prob.
            max_width: maximum bracket width.
            init_width: Inital width of brackets.
            thin: Amount of thinning; if None, no thinning.
            tuning: Number of tuning steps for brackets.
            verbose: Whether to show progress bars (False).
        """

        MCMCSampler.__init__(self, x, lp_f, thin, verbose=verbose)
        self.max_width = max_width
        self.init_width = init_width
        self.width = None
        self.tuning = tuning

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

        order = list(range(self.n_dims))
        x = self.x.copy()
        self.width = np.full(self.n_dims, self.init_width)

        tbar = trange(self.tuning, miniters=10, disable=not self.verbose)
        tbar.set_description("Tuning bracket width...")
        for n in tbar:
            # for n in range(int(self.tuning)):
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


class SliceSamplerSerial:
    def __init__(
        self,
        log_prob_fn: Callable,
        init_params: np.ndarray,
        num_chains: int = 1,
        thin: Optional[int] = None,
        tuning: int = 50,
        verbose: bool = True,
        init_width: Union[float, np.ndarray] = 0.01,
        max_width: float = float("inf"),
        num_workers: int = 1,
    ):
        """Slice sampler in pure Numpy, running for each chain in serial.

        Parallelization across CPUs is possible by setting num_workers > 1.

        Args:
            log_prob_fn: Log prob function.
            init_params: Initial parameters.
            num_chains: Number of MCMC chains to run in parallel
            thin: amount of thinning; if None, no thinning.
            tuning: Number of tuning steps for brackets.
            verbose: Show/hide additional info such as progress bars.
            init_width: Inital width of brackets.
            max_width: Maximum width of brackets.
            num_workers: Number of parallel workers to use.
        """
        self._log_prob_fn = log_prob_fn

        self.x = init_params
        self.num_chains = num_chains
        self.thin = thin
        self.tuning = tuning
        self.verbose = verbose

        self.init_width = init_width
        self.max_width = max_width

        self.n_dims = self.x.size
        self.num_workers = num_workers
        self._samples = None

    def run(self, num_samples: int) -> np.ndarray:
        """Runs MCMC and returns thinned samples.

        Sampling is performed parallelized across CPUs if self.num_workers > 1.
        Parallelization is seeded across workers.

        Note: Thinning is performed internally.

        Args:
            num_samples: Number of samples to generate
        Returns:
            MCMC samples in shape (num_chains, num_samples_per_chain, num_dim)
        """

        num_chains, dim_samples = self.x.shape

        # Generate seeds for workers from current random state.
        seeds = torch.randint(high=2**31, size=(num_chains,))

        with tqdm_joblib(
            tqdm(
                range(num_chains),  # type: ignore
                disable=not self.verbose or self.num_workers == 1,
                desc=f"""Running {self.num_chains} MCMC chains with
                      {self.num_workers} worker{"s" if self.num_workers>1 else ""}.""",
                total=self.num_chains,
            )
        ):
            all_samples = Parallel(n_jobs=self.num_workers)(
                delayed(self.run_fun)(num_samples, initial_params_batch, seed)
                for initial_params_batch, seed in zip(self.x, seeds)
            )

        samples = np.stack(all_samples).astype(np.float32)
        samples = samples.reshape(num_chains, -1, dim_samples)  # chains, samples, dim
        samples = samples[:, :: self.thin, :]  # thin chains

        # save samples
        self._samples = samples

        return samples

    def run_fun(self, num_samples, inits, seed) -> np.ndarray:
        """Runs MCMC for a given number of samples starting at inits."""
        np.random.seed(seed)
        posterior_sampler = SliceSampler(
            inits,
            lp_f=self._log_prob_fn,
            max_width=self.max_width,
            init_width=self.init_width,
            thin=self.thin,
            tuning=self.tuning,
            # turn off pbars in parallel mode.
            verbose=self.num_workers == 1 and self.verbose,
        )
        return posterior_sampler.gen(num_samples)

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
        if self._samples is None:
            raise ValueError("No samples found from MCMC run.")
        # if not grouped by chain, flatten samples into (all_samples, dim_params)
        if not group_by_chain:
            samples = self._samples.reshape(-1, self._samples.shape[2])
        else:
            samples = self._samples

        # if not specified return all samples
        if num_samples is None:
            return samples
        # otherwise return last num_samples (for each chain when grouped).
        elif group_by_chain:
            return samples[:, -num_samples:, :]
        else:
            return samples[-num_samples:, :]


class SliceSamplerVectorized:
    def __init__(
        self,
        log_prob_fn: Callable,
        init_params: np.ndarray,
        num_chains: int = 1,
        thin: Optional[int] = None,
        tuning: int = 50,
        verbose: bool = True,
        init_width: Union[float, np.ndarray] = 0.01,
        max_width: float = float("inf"),
        num_workers: int = 1,
    ):
        """Slice sampler in pure Numpy, vectorized evaluations across chains.

        Args:
            log_prob_fn: Log prob function.
            init_params: Initial parameters.
            num_chains: Number of MCMC chains to run in parallel
            thin: amount of thinning; if None, no thinning.
            tuning: Number of tuning steps for brackets.
            verbose: Show/hide additional info such as progress bars.
            init_width: Inital width of brackets.
            max_width: Maximum width of brackets.
            num_workers: Number of parallel workers to use (not implemented.)
        """
        self._log_prob_fn = log_prob_fn

        self.x = init_params
        self.num_chains = num_chains
        self.thin = 1 if thin is None else thin
        self.tuning = tuning
        self.verbose = verbose

        self.init_width = init_width
        self.max_width = max_width

        self.n_dims = self.x.size

        self._samples = None

        # TODO: implement parallelization across batches of chains.
        if num_workers > 1:
            warn(
                """Parallelization of vectorized slice sampling not implement, running
                serially."""
            )
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

        samples = samples[:, :: self.thin, :]  # thin chains

        self._samples = samples

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
        if self._samples is None:
            raise ValueError("No samples found from MCMC run.")
        # if not grouped by chain, flatten samples into (all_samples, dim_params)
        if not group_by_chain:
            samples = self._samples.reshape(-1, self._samples.shape[2])
        else:
            samples = self._samples

        # if not specified return all samples
        if num_samples is None:
            return samples
        # otherwise return last num_samples (for each chain when grouped).
        elif group_by_chain:
            return samples[:, -num_samples:, :]
        else:
            return samples[-num_samples:, :]
