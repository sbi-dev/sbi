# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.inference import NPE_C
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.estimators import ConditionalDensityEstimator
from sbi.neural_nets.net_builders.estimator_configs import (
    DensityEstimatorBuilder,
)
from sbi.utils.user_input_checks import check_estimator_arg


@pytest.mark.parametrize("model", ("maf", "nsf", "mdn", "zuko_nsf"))
def test_npe_accepts_builder(model):
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    builder = DensityEstimatorBuilder(model=model)
    inference = NPE_C(prior, density_estimator=builder, show_progress_bars=False)
    assert inference._build_neural_net is not None


def test_npe_string_emits_deprecation_warning():
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    with pytest.warns(FutureWarning, match="deprecated"):
        NPE_C(prior, density_estimator="maf", show_progress_bars=False)


def test_npe_callable_no_warning():
    num_dim = 2
    prior = MultivariateNormal(zeros(num_dim), eye(num_dim))
    build_fn = posterior_nn(model="maf")
    inference = NPE_C(prior, density_estimator=build_fn, show_progress_bars=False)
    assert inference._build_neural_net is build_fn


@pytest.mark.parametrize(
    "estimator",
    (
        DensityEstimatorBuilder(model="maf"),
        "maf",
        posterior_nn(model="maf"),
    ),
    ids=["builder", "string", "callable"],
)
def test_check_estimator_arg_accepts_valid_inputs(estimator):
    check_estimator_arg(estimator)


def test_check_estimator_arg_rejects_module():
    with pytest.raises(AssertionError):
        check_estimator_arg(torch.nn.Linear(3, 3))
