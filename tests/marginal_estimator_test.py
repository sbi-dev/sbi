# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Callable, Union

import pytest
import torch
from torch.distributions import (
    Categorical,
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
    VonMises,
)

from sbi.inference.trainers.marginal import MarginalTrainer
from sbi.neural_nets import MarginalConfigBase, MarginalNCSFConfig, MarginalNSFConfig
from sbi.neural_nets.factory import ZukoFlowType, marginal_nn
from sbi.neural_nets.net_builders.flow import (
    build_zuko_flow,
    build_zuko_ncsf,
    build_zuko_unconditional_flow,
)
from sbi.utils.metrics import check_c2st
from sbi.utils.torchutils import process_device


@pytest.mark.parametrize(
    "dist",
    [
        MultivariateNormal(
            loc=torch.tensor([2.0, 3.0]),
            covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
        ),
        MixtureSameFamily(
            Categorical(torch.ones(2)),
            Normal(torch.tensor([-1.5, 1.5]), torch.tensor([0.5, 0.75])),
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "model", [MarginalNSFConfig(), marginal_nn(model=ZukoFlowType.NSF)]
)
def test_marginal_estimator(
    dist: torch.distributions.Distribution,
    device: str,
    model: Union[MarginalConfigBase, Callable],
):
    """Test the marginal estimator with various distributions and devices."""
    num_training_samples = 2_000
    num_test_samples = 1_000
    device = process_device(device)

    # Generate samples from the true distribution
    x_train = dist.sample((num_training_samples,))
    if len(x_train.shape) == 1:
        x_train = x_train.unsqueeze(1)

    # Instantiate a trainer for the marginal pdf and train it
    trainer = MarginalTrainer(density_estimator=model, device=device)
    trainer.append_samples(x_train)
    est = trainer.train(max_num_epochs=3000)

    # Sample from the marginal pdf estimator
    samples = est.sample(torch.Size((num_test_samples,)))

    # Compute the C2ST score
    x_test = dist.sample(torch.Size((num_test_samples,)))
    if len(x_test.shape) == 1:
        x_test = x_test.unsqueeze(1)

    check_c2st(x_test, samples.cpu(), f'MarginalEstimator-{model}')


@pytest.mark.parametrize("which_nf", ["MAF", "NSF"])
def test_unconditional_flow_is_equivariant_to_affine_data_changes(which_nf: str):
    """With z-scoring, an affine change of the data only subtracts its log-determinant.

    This holds only if the flow transforms see z-scored data, i.e., the z-score is
    applied before them.
    """
    x = torch.randn(500, 2) * torch.tensor([1.0, 3.0])
    shift, scale = 1000.0, 10.0
    y = shift + scale * x

    torch.manual_seed(0)
    flow_x = build_zuko_unconditional_flow(which_nf, x, z_score_x="independent")
    torch.manual_seed(0)
    flow_y = build_zuko_unconditional_flow(which_nf, y, z_score_x="independent")

    with torch.no_grad():
        log_prob_x = flow_x.log_prob(x.unsqueeze(0))
        log_prob_y = flow_y.log_prob(y.unsqueeze(0))
    log_det = x.shape[1] * torch.log(torch.tensor(scale))
    assert torch.allclose(log_prob_y, log_prob_x - log_det, atol=1e-3)


@pytest.mark.parametrize("build", ["marginal_config", "marginal_nn", "conditional"])
def test_ncsf_with_defaults_integrates_to_one(build: str):
    """NCSF takes no z-score by default, so its density integrates to one.

    A z-score would move the data off [-pi, pi). For concentrated data, the
    density would then wrap around the circle several times.
    """
    torch.manual_seed(0)
    x = VonMises(torch.tensor(0.5), torch.tensor(20.0)).sample((500, 2))
    grid_1d = torch.linspace(-torch.pi, torch.pi, 201)[:-1] + torch.pi / 200
    grid = torch.cartesian_prod(grid_1d, grid_1d)
    cell_area = (2 * torch.pi / 200) ** 2

    with torch.no_grad():
        if build == "conditional":
            flow = build_zuko_ncsf(x, torch.randn(500, 1))
            log_prob = flow.log_prob(grid.unsqueeze(1), torch.zeros(1, 1))
        else:
            if build == "marginal_config":
                flow = MarginalNCSFConfig().build(x)
            else:
                flow = marginal_nn(ZukoFlowType.NCSF)(x)
            log_prob = flow.log_prob(grid.unsqueeze(0))
    assert log_prob.exp().sum().item() * cell_area == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("z_score_x", ["independent", "structured"])
def test_ncsf_rejects_a_z_score_in_both_builders(z_score_x: str):
    """A z-score would move NCSF's data off [-pi, pi)."""
    x = torch.pi * (2 * torch.rand(100, 2) - 1)

    with pytest.raises(ValueError, match="NCSF"):
        build_zuko_unconditional_flow("NCSF", x, z_score_x=z_score_x)
    with pytest.raises(ValueError, match="NCSF"):
        build_zuko_flow("NCSF", x, torch.randn(100, 1), z_score_x=z_score_x)


def test_marginal_trainer_exhausted_epoch_budget_returns_the_best_weights():
    """A run that hits `max_num_epochs` must warn and end on its best weights.

    The trainer has its own training loop, whose condition short-circuits like the
    shared one in `NeuralInference`: `_converged` never scores the final epoch.
    """
    torch.manual_seed(0)
    x = torch.randn(200, 2)

    trainer = MarginalTrainer(show_progress_bars=False)
    trainer.append_samples(x)
    with pytest.warns(UserWarning, match="max_num_epochs"):
        trainer.train(max_num_epochs=2, stop_after_epochs=50)

    assert trainer.epoch > 2, "the budget must run out for this to test anything"
    final = trainer._neural_net.state_dict()
    assert all(
        torch.equal(final[k], v) for k, v in trainer._best_model_state_dict.items()
    )
