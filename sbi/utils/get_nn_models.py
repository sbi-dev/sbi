# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Any, Callable
from sbi.neural_nets.flow import build_made, build_maf, build_nsf
from sbi.neural_nets.mdn import build_mdn
from sbi.neural_nets.classifier import (
    build_linear_classifier,
    build_mlp_classifier,
    build_resnet_classifier,
)


def classifier_nn(model: str, **kwargs: Any) -> Callable:
    def build_fn(batch_theta, batch_x):
        if model == "linear":
            return build_linear_classifier(
                batch_x=batch_x, batch_y=batch_theta, **kwargs
            )
        if model == "mlp":
            return build_mlp_classifier(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        if model == "resnet":
            return build_resnet_classifier(
                batch_x=batch_x, batch_y=batch_theta, **kwargs
            )
            raise NotImplementedError

    return build_fn


def likelihood_nn(model: str, **kwargs: Any) -> Callable:
    def build_fn(batch_theta, batch_x):
        if model == "mdn":
            return build_mdn(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        if model == "made":
            return build_made(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        if model == "maf":
            return build_maf(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        elif model == "nsf":
            return build_nsf(batch_x=batch_x, batch_y=batch_theta, **kwargs)
        else:
            raise NotImplementedError

    return build_fn


def posterior_nn(model: str, **kwargs: Any) -> Callable:
    def build_fn(batch_theta, batch_x):
        if model == "mdn":
            return build_mdn(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        if model == "made":
            return build_made(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        if model == "maf":
            return build_maf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "nsf":
            return build_nsf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        else:
            raise NotImplementedError

    return build_fn
