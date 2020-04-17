import pytest
import torch

import sbi.utils as utils
from sbi.inference.snpe.snpe_c import SnpeC
from sbi.simulators.nonlinear_gaussian import (
    get_ground_truth_posterior_samples_nonlinear_gaussian,
    non_linear_gaussian,
)
from sbi.simulators.simutils import prepare_sbi_problem
from sbi.utils.torchutils import BoxUniform

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")


@pytest.mark.slow
def test_nonlinearGaussian_based_on_mmd(set_seed):
    """Run inference with snpe-c and test against average mmd.

    This test is seeded using the set_seed fixture defined in tests/conftest.py.

    Args:
        set_seed: fixture for manual seeding, see tests/conftest.py
    """

    # Ground truth parameters as specified in 'Sequential Neural Likelihood' paper.
    theta_o = torch.tensor([-0.7, -2.9, -1.0, -0.9, 0.6])
    theta_dim = theta_o.numel()
    # ground truth observation using same seed as 'Sequential Neural Likelihood' paper.
    x_o = torch.tensor(
        [
            -0.97071232,
            -2.94612244,
            -0.44947218,
            -3.42318484,
            -0.13285634,
            -3.36401699,
            -0.85367595,
            -2.42716377,
        ]
    )

    prior = BoxUniform(low=-3 * torch.ones(theta_dim), high=3 * torch.ones(theta_dim),)

    simulator, prior, x_o = prepare_sbi_problem(non_linear_gaussian, prior, x_o)

    infer = SnpeC(
        simulator=simulator,
        x_o=x_o,
        prior=prior,
        num_atoms=-1,
        z_score_x=True,
        use_combined_loss=False,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
    )

    num_rounds, num_simulations_per_round = 2, 1000
    posterior = infer(
        num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round
    )

    samples = posterior.sample(1000)

    # Sample from (analytically tractable) target distribution.
    target_samples = get_ground_truth_posterior_samples_nonlinear_gaussian(
        num_samples=1000
    )

    # Compute and check if MMD is larger than expected.
    mmd = utils.unbiased_mmd_squared(target_samples.float(), samples.float())

    max_mmd = 0.16  # mean mmd plus 2 stds.
    assert mmd < max_mmd, f"MMD={mmd} larger than mean plus 2 stds."
