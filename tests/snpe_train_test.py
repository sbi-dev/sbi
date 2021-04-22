# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch
# from scipy.stats import beta, multivariate_normal, uniform
# from torch import Tensor, eye, nn, ones, zeros
from torch.distributions import Distribution, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import SNPE,  simulate_for_sbi
from sbi.utils.user_input_checks import prepare_for_sbi
from sbi.simulators.linear_gaussian import diagonal_linear_gaussian


@pytest.mark.gpu
@pytest.mark.parametrize("target_device", ("cpu", "cuda:0"))
def test_train_with_different_data_and_training_device(target_device):

    tdir = Path(tempfile.mkdtemp())
    tdevice = torch.device(target_device)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=str(tdir))

    num_dim = 2
    prior_ = MultivariateNormal(loc=torch.zeros(num_dim),
                                covariance_matrix=torch.eye(num_dim))
    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior_)

    inference = SNPE(prior,
                     density_estimator="maf",
                     show_progress_bars=False,
                     summary_writer=writer,
                     device=tdevice)

    # Run inference.
    theta, x = simulate_for_sbi(simulator, prior, 100)
    theta, x = theta.to(tdevice), x.to(tdevice)
    _ = inference.append_simulations(theta, x).train(max_num_epochs=2)
    _ = inference.build_posterior()

    writer.close()

    assert tdir.exists()
    # enforce flush
    del writer

    wfiles = list(Path(tdir).glob("*"))

    assert len(wfiles) > 0

    wsizes = [Path(item).stat().st_size for item in wfiles]
    assert sum(wsizes) > 0

    # check inference object about what has landed
    assert hasattr(inference, "_summary")

    assert "epochs" in inference._summary.keys()
    assert inference._summary["epochs"] == [3]
    assert len(inference._summary["epoch_durations_sec"]) == 3
    assert inference._summary["epoch_durations_sec"][-1] > 0.
    assert len(inference._summary["train_log_probs"]) == 3

    for wf in wfiles:
        wf.unlink()

    tdir.rmdir()
