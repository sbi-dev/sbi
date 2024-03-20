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
from sbi.neural_nets.flow import build_nsf, build_zuko_maf
from sbi.neural_nets import posterior_nn, PermutationInvariantEmbedding, FCEmbedding


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
        ((1, 2, 3), (1, 2, 3), (3,), False),
        ((3, 5), (1, 1, 3, 5), (3, 5), False),
        ((3, 5), (1, 1, 3, 5), (3, 5), True),
        ((1, 3, 5), (1, 1, 3, 5), (3, 5), False),
        ((1, 3, 5), (1, 1, 3, 5), (3, 5), True),
        ((2, 3, 5), (1, 2, 3, 5), (3, 5), False),
        ((2, 3, 5), (2, 1, 3, 5), (3, 5), True),
        ((1, 2, 3, 5), (1, 2, 3, 5), (3, 5), True),
        ((1, 2, 3, 5), (1, 2, 3, 5), (3, 5), False),
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
@pytest.mark.parametrize("condition_iid_dim", (1, 3))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_density_estimator_loss_shapes(
    density_estimator_name,
    input_iid_dim,
    input_event_shape,
    condition_iid_dim,
    condition_event_shape,
    batch_dim,
):
    """Test whether `loss` of DensityEstimators follow the shape convention."""
    output_dim = 5
    embedding_net = FCEmbedding(
        input_dim=condition_event_shape[0], output_dim=output_dim
    )
    if condition_iid_dim > 1:
        embedding_net = PermutationInvariantEmbedding(
            embedding_net,
            trial_net_output_dim=output_dim,
        )

    density_estimator = posterior_nn(
        density_estimator_name, embedding_net=embedding_net
    )
    building_thetas = torch.randn((100, *input_event_shape))
    building_xs = torch.randn((100, condition_iid_dim, *condition_event_shape))
    density_estimator = density_estimator(building_thetas, building_xs)

    inputs = torch.randn((input_iid_dim, batch_dim, *input_event_shape))
    condition = torch.randn((condition_iid_dim, batch_dim, *condition_event_shape))
    losses = density_estimator.loss(inputs, condition=condition)
    assert losses.shape == (input_iid_dim, batch_dim)


@pytest.mark.parametrize("density_estimator_name", ("maf", "zuko_maf"))
@pytest.mark.parametrize("input_iid_dim", (1, 2))
@pytest.mark.parametrize("input_event_shape", ((1,), (4,)))
@pytest.mark.parametrize("condition_iid_dim", (1, 3))
@pytest.mark.parametrize("condition_event_shape", ((1,), (7,)))
@pytest.mark.parametrize("batch_dim", (1, 10))
def test_density_estimator_sample_shapes(
    density_estimator_name,
    input_iid_dim,
    input_event_shape,
    condition_iid_dim,
    condition_event_shape,
    batch_dim,
):
    """Test whether `loss` of DensityEstimators follow the shape convention."""
    output_dim = 5
    embedding_net = FCEmbedding(
        input_dim=condition_event_shape[0], output_dim=output_dim
    )
    if condition_iid_dim > 1:
        embedding_net = PermutationInvariantEmbedding(
            embedding_net,
            trial_net_output_dim=output_dim,
        )

    density_estimator = posterior_nn(
        density_estimator_name, embedding_net=embedding_net
    )
    building_thetas = torch.randn((100, *input_event_shape))
    building_xs = torch.randn((100, condition_iid_dim, *condition_event_shape))
    density_estimator = density_estimator(building_thetas, building_xs)

    inputs = torch.randn((input_iid_dim, batch_dim, *input_event_shape))
    condition = torch.randn((condition_iid_dim, batch_dim, *condition_event_shape))
    losses = density_estimator.loss(inputs, condition=condition)
    assert losses.shape == (input_iid_dim, batch_dim)


@pytest.mark.parametrize("density_estimator_name", ("maf", "zuko_maf"))
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
    density_estimator = posterior_nn(density_estimator_name)
    building_thetas = torch.randn((100, *input_event_shape))
    building_xs = torch.randn((100, *condition_event_shape))
    density_estimator = density_estimator(building_thetas, building_xs)

    condition_iid_dim = 1
    input_iid_dim = 2
    inputs = torch.randn((1, batch_dim, *input_event_shape))
    inputs = inputs.expand(input_iid_dim, -1, -1)
    condition = torch.randn((condition_iid_dim, batch_dim, *condition_event_shape))
    losses = density_estimator.loss(inputs, condition=condition)
    assert torch.allclose(losses[0, :], losses[1, :])