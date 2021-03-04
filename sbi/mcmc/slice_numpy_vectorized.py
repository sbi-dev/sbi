# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Callable, Union

import numpy as np
from tqdm import tqdm


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
        self.rng = np.random
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
            pbar = tqdm(range(self.num_chains * num_samples))
            print("Generating MCMC samples")

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
                                        pbar.update(10)

                        else:
                            sc["state"] = "DONE"

                if sc["state"] == "DONE":
                    num_chains_finished += 1

        samples = np.stack([self.state[c]["samples"] for c in range(self.num_chains)])

        return samples
