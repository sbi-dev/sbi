# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Tuple

import pytest
import torch
from torch import eye, zeros
from torch.distributions import MultivariateNormal

from sbi.neural_nets.density_estimators import NFlowsFlow, ZukoFlow
from sbi.neural_nets.density_estimators.shape_handling import reshape_to_iid_batch_event
from sbi.neural_nets.flow import build_nsf, build_zuko_maf, build_maf
from sbi.neural_nets import (
    posterior_nn,
    PermutationInvariantEmbedding,
    FCEmbedding,
    build_mnle,
    likelihood_nn,
)
from sbi.neural_nets.mnle import MixedDensityEstimator
from sbi.neural_nets.categorial import build_categoricalmassestimator


@pytest.mark.parametrize(
    "theta_or_x_shape, target_shape, event_shape, leading_is_iid",
    (
        ((3,), (1, 1, 3), (3,), False),
        ((3,), (1, 1, 3), (3,), True),
        ((1, 3), (1, 1, 3), (3,), False),
        ((1, 3), (1, 1, 3), (3,), True),
        ((2, 3), (1, 2, 3), (3,), False),
        ((2, 3), (2, 1, 3), (3,), True),
        ((1, 2, 3), (1, 2, 3), (3,), True),
        ((1, 2, 3), (2, 1, 3), (3,), False),
        ((3, 5), (1, 1, 3, 5), (3, 5), False),
        ((3, 5), (1, 1, 3, 5), (3, 5), True),
        ((1, 3, 5), (1, 1, 3, 5), (3, 5), False),
        ((1, 3, 5), (1, 1, 3, 5), (3, 5), True),
        ((2, 3, 5), (1, 2, 3, 5), (3, 5), False),
        ((2, 3, 5), (2, 1, 3, 5), (3, 5), True),
        ((1, 2, 3, 5), (1, 2, 3, 5), (3, 5), True),
        ((1, 2, 3, 5), (2, 1, 3, 5), (3, 5), False),
        pytest.param((1, 2, 3, 5), (1, 2, 3, 5), (5), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3, 5), (1, 2, 3, 5), (3), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (1, 2, 3), (1, 5), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (1, 2, 3), (1, 3), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (2, 1, 3), (3), False, marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (2, 1, 3), (3), True, marks=pytest.mark.xfail),
    ),
)
def test_shape_handling_utility_for_density_estimator(
    theta_or_x_shape: Tuple,
    target_shape: Tuple,
    event_shape: Tuple,
    leading_is_iid: bool,
):
    """Test whether `reshape_to_batch_iid_event` results in expected outputs."""
    input = torch.randn(theta_or_x_shape)
    output = reshape_to_iid_batch_event(
        input, event_shape=event_shape, leading_is_iid=leading_is_iid
    )
    assert output.shape == target_shape, (
        f"Shapes of Output ({output.shape}) and target shape ({target_shape}) do not "
        f"match."
    )


@pytest.mark.parametrize("density_estimator_name", ("maf", "zuko_maf"))
@pytest.mark.parametrize("input_iid_dim", (1, 2))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_density_estimator_loss_shapes(
    density_estimator_name,
    input_iid_dim,
    input_event_shape,
    condition_event_shape,
    batch_dim,
):
    """Test whether `loss` of DensityEstimators follow the shape convention."""
    density_estimator, inputs, conditions = _build_density_estimator_and_tensors(
        density_estimator_name,
        input_event_shape,
        condition_event_shape,
        batch_dim,
        input_iid_dim,
    )

    losses = density_estimator.loss(inputs, condition=conditions)
    assert losses.shape == (input_iid_dim, batch_dim)


@pytest.mark.parametrize("density_estimator_name", ("maf", "zuko_maf"))
@pytest.mark.parametrize("sample_shape", ((1,), (2, 3)))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_density_estimator_sample_shapes(
    density_estimator_name,
    sample_shape,
    input_event_shape,
    condition_event_shape,
    batch_dim,
):
    """Test whether `loss` of DensityEstimators follow the shape convention."""
    density_estimator, _, conditions = _build_density_estimator_and_tensors(
        density_estimator_name, input_event_shape, condition_event_shape, batch_dim
    )
    samples = density_estimator.sample(sample_shape, condition=conditions)
    assert samples.shape == (*sample_shape, batch_dim, *input_event_shape)


@pytest.mark.parametrize(
    "density_estimator_name", ("maf", "zuko_maf", "discrete", "mixed")
)
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_correctness_of_density_estimator_loss(
    density_estimator_name,
    input_event_shape,
    condition_event_shape,
    batch_dim,
):
    """Test whether identical inputs lead to identical loss values."""
    input_iid_dim = 2
    density_estimator, inputs, condition = (
        _build_density_estimator_and_tensors(
            density_estimator_name,
            input_event_shape,
            condition_event_shape,
            batch_dim,
            input_iid_dim,
        )
    )
    losses = density_estimator.loss(inputs, condition=condition)
    assert torch.allclose(losses[0, :], losses[1, :])


def _build_density_estimator_and_tensors(
    density_estimator_name: str,
    input_event_shape: Tuple[int],
    condition_event_shape: Tuple[int],
    batch_dim: int,
    input_iid_dim: int = 1,
):
    if density_estimator_name == "discrete":
        input_event_shape = (1,)
    elif density_estimator_name == "mixed":
        input_event_shape = (
            input_event_shape[0] + 1,
        )  # 1 does not make sense for mixed.

    # Use discrete thetas such that discrete density esitmators can also use them.
    building_thetas = torch.randint(
        0, 4, (1000, *input_event_shape), dtype=torch.float32
    )
    building_xs = torch.randn((1000, *condition_event_shape))
    if density_estimator_name == "maf":
        density_estimator = build_maf(
            torch.randn_like(building_thetas), torch.randn_like(building_xs)
        )
    elif density_estimator_name == "zuko_maf":
        density_estimator = build_zuko_maf(
            torch.randn_like(building_thetas), torch.randn_like(building_xs)
        )
    elif density_estimator_name == "mixed":
        building_thetas[:, :-1] += 5.0  # Make continuous dims positive for log-tf.
        density_estimator = build_mnle(building_thetas, building_xs)
    elif density_estimator_name == "discrete":
        density_estimator = build_categoricalmassestimator(building_thetas, building_xs)
    else:
        raise ValueError

    inputs = building_thetas[:batch_dim]
    condition = building_xs[:batch_dim]

    inputs = inputs.unsqueeze(0)
    inputs = inputs.expand(input_iid_dim, -1, -1)
    condition = condition.unsqueeze(0)
    return density_estimator, inputs, condition
