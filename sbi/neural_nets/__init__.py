# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from sbi.neural_nets.factory import (
    classifier_nn,
    likelihood_nn,
    marginal_nn,
    posterior_flow_nn,
    posterior_nn,
    posterior_score_nn,
)


def __getattr__(name):
    if name == "flowmatching_nn":
        raise ImportError(
            "`flowmatching_nn` has been removed. "
            "Please use `posterior_flow_nn` instead."
        )
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


__all__ = [
    "classifier_nn",
    "likelihood_nn",
    "marginal_nn",
    "posterior_nn",
    "posterior_score_nn",
    "posterior_flow_nn",
]
