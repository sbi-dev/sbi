# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
import torch.distributions.transforms as torch_tf
import torch.nn as nn
from torch import eye, ones, zeros
from torch.distributions import Distribution, MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    SNLE,
    SNPE_A,
    SNPE_C,
    SNRE_A,
    SNRE_B,
    DirectPosterior,
    MCMCPosterior,
    NeuralInference,
    RejectionPosterior,
    VIPosterior,
    likelihood_estimator_based_potential,
    posterior_estimator_based_potential,
    ratio_estimator_based_potential,
    simulate_for_sbi,
)
from sbi.inference.potentials.base_potential import BasePotential
from sbi.simulators import diagonal_linear_gaussian, linear_gaussian
from sbi.utils.get_nn_models import classifier_nn, likelihood_nn, posterior_nn
from sbi.utils.torchutils import BoxUniform, process_device
from sbi.utils.user_input_checks import (
    check_embedding_net_device,
    prepare_for_sbi,
    validate_theta_and_x,
)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "method, model, mcmc_method",
    [
        (SNPE_C, "mdn", "rejection"),
        (SNPE_C, "maf", "slice"),
        (SNPE_C, "maf", "direct"),
        (SNLE, "maf", "slice"),
        (SNLE, "nsf", "slice_np"),
        (SNLE, "nsf", "rejection"),
        (SNRE_A, "mlp", "slice_np_vectorized"),
        (SNRE_B, "resnet", "nuts"),
        (SNRE_B, "resnet", "rejection"),
    ],
)
@pytest.mark.parametrize("data_device", ("cpu", "cuda:0"))
@pytest.mark.parametrize(
    "training_device, prior_device",
    [
        pytest.param("cpu", "cuda", marks=pytest.mark.xfail),
        pytest.param("cuda:0", "cpu", marks=pytest.mark.xfail),
        ("cuda:0", "cuda:0"),
        ("cuda:0", "cuda:0"),
        ("cpu", "cpu"),
    ],
)
def test_training_and_mcmc_on_device(
    method,
    model,
    data_device,
    mcmc_method,
    training_device,
    prior_device,
    prior_type="gaussian",
):
    """Test training on devices.

    This test does not check training speeds.

    """

    num_dim = 2
    num_samples = 10
    num_simulations = 100
    max_num_epochs = 5

    x_o = zeros(1, num_dim).to(data_device)
    likelihood_shift = -1.0 * ones(num_dim).to(prior_device)
    likelihood_cov = 0.3 * eye(num_dim).to(prior_device)

    if prior_type == "gaussian":
        prior_mean = zeros(num_dim).to(prior_device)
        prior_cov = eye(num_dim).to(prior_device)
        prior = MultivariateNormal(loc=prior_mean, covariance_matrix=prior_cov)
    else:
        prior = BoxUniform(
            low=-2 * torch.ones(num_dim),
            high=2 * torch.ones(num_dim),
            device=prior_device,
        )

    def simulator(theta):
        return linear_gaussian(theta, likelihood_shift, likelihood_cov)

    training_device = process_device(training_device)

    if method in [SNPE_A, SNPE_C]:
        kwargs = dict(
            density_estimator=utils.posterior_nn(model=model, num_transforms=2)
        )
    elif method == SNLE:
        kwargs = dict(
            density_estimator=utils.likelihood_nn(model=model, num_transforms=2)
        )
    elif method in (SNRE_A, SNRE_B):
        kwargs = dict(classifier=utils.classifier_nn(model=model))
    else:
        raise ValueError()

    inferer = method(show_progress_bars=False, device=training_device, **kwargs)

    proposals = [prior]

    # Test for two rounds.
    for _ in range(2):
        theta, x = simulate_for_sbi(simulator, proposals[-1], num_simulations)
        theta, x = theta.to(data_device), x.to(data_device)

        estimator = inferer.append_simulations(theta, x).train(
            training_batch_size=100, max_num_epochs=max_num_epochs
        )
        if method == SNLE:
            potential_fn, theta_transform = likelihood_estimator_based_potential(
                estimator, prior, x_o
            )
        elif method == SNPE_A or method == SNPE_C:
            potential_fn, theta_transform = posterior_estimator_based_potential(
                estimator, prior, x_o
            )
        elif method == SNRE_A or method == SNRE_B:
            potential_fn, theta_transform = ratio_estimator_based_potential(
                estimator, prior, x_o
            )
        else:
            raise ValueError

        if mcmc_method == "rejection":
            posterior = RejectionPosterior(
                proposal=prior,
                potential_fn=potential_fn,
                device=training_device,
            )
        elif mcmc_method == "direct":
            posterior = DirectPosterior(
                posterior_estimator=estimator, prior=prior
            ).set_default_x(x_o)
        else:
            posterior = MCMCPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                proposal=prior,
                method=mcmc_method,
                device=training_device,
            )
        proposals.append(posterior)

    # Check for default device for inference object
    weights_device = next(inferer._neural_net.parameters()).device
    assert torch.device(training_device) == weights_device
    samples = proposals[-1].sample(sample_shape=(num_samples,))
    proposals[-1].potential(samples)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "device_input, device_target",
    [
        ("cpu", "cpu"),
        ("cuda", "cuda:0"),
        ("cuda:0", "cuda:0"),
        pytest.param("cuda:42", None, marks=pytest.mark.xfail),
        pytest.param("qwerty", None, marks=pytest.mark.xfail),
    ],
)
def test_process_device(device_input: str, device_target: Optional[str]) -> None:
    device_output = process_device(device_input)
    assert device_output == device_target, (
        f"Failure when processing device '{device_input}': "
        f"result should have been '{device_target}' and is "
        f"instead '{device_output}'"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("device_datum", ["cpu", "cuda"])
@pytest.mark.parametrize("device_embedding_net", ["cpu", "cuda"])
def test_check_embedding_net_device(
    device_datum: str, device_embedding_net: str
) -> None:

    datum = torch.zeros((1, 1)).to(device_datum)
    embedding_net = nn.Linear(in_features=1, out_features=1).to(device_embedding_net)

    if device_datum != device_embedding_net:
        with pytest.warns(UserWarning):
            check_embedding_net_device(datum=datum, embedding_net=embedding_net)
    else:
        check_embedding_net_device(datum=datum, embedding_net=embedding_net)

    output_device_net = [p.device for p in embedding_net.parameters()][0]
    assert datum.device == output_device_net, (
        f"Failure when processing embedding_net: "
        f"device should have been set to should have been '{datum.device}' but is "
        f"still '{output_device_net}'"
    )


@pytest.mark.parametrize(
    "shape_x",
    [
        (3, 1),
    ],
)
@pytest.mark.parametrize(
    "shape_theta", [(3, 2), pytest.param((2, 1), marks=pytest.mark.xfail)]
)
def test_validate_theta_and_x_shapes(
    shape_x: Tuple[int], shape_theta: Tuple[int]
) -> None:
    x = torch.empty(shape_x)
    theta = torch.empty(shape_theta)

    validate_theta_and_x(theta, x, training_device="cpu")


def test_validate_theta_and_x_tensor() -> None:
    x = torch.empty((1, 1))
    theta = torch.ones((1, 1)).tolist()

    with pytest.raises(Exception):
        validate_theta_and_x(theta, x, training_device="cpu")


def test_validate_theta_and_x_type() -> None:
    x = torch.empty((1, 1))
    theta = torch.empty((1, 1), dtype=int)

    with pytest.raises(Exception):
        validate_theta_and_x(theta, x, training_device="cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("training_device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("data_device", ["cpu", "cuda:0"])
def test_validate_theta_and_x_device(training_device: str, data_device: str) -> None:
    theta = torch.empty((1, 1)).to(data_device)
    x = torch.empty((1, 1)).to(data_device)

    if training_device != data_device:
        with pytest.warns(UserWarning):
            theta, x = validate_theta_and_x(theta, x, training_device=training_device)
    else:
        theta, x = validate_theta_and_x(theta, x, training_device=training_device)

    assert str(theta.device) == training_device, (
        f"Data should have its device converted from '{data_device}' "
        f"to training_device '{training_device}'."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("inference_method", [SNPE_A, SNPE_C, SNRE_A, SNRE_B, SNLE])
@pytest.mark.parametrize("data_device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("training_device", ("cpu", "cuda:0"))
def test_train_with_different_data_and_training_device(
    inference_method, data_device: str, training_device: str
) -> None:

    assert torch.cuda.is_available(), "this test requires that cuda is available."

    num_dim = 2
    prior_ = BoxUniform(
        -torch.ones(num_dim), torch.ones(num_dim), device=training_device
    )
    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior_)

    inference = inference_method(
        prior,
        **(
            dict(classifier="resnet")
            if inference_method in [SNRE_A, SNRE_B]
            else dict(
                density_estimator=(
                    "mdn_snpe_a" if inference_method == SNPE_A else "maf"
                )
            )
        ),
        show_progress_bars=False,
        device=training_device,
    )

    theta, x = simulate_for_sbi(simulator, prior, 32)
    theta, x = theta.to(data_device), x.to(data_device)
    x_o = torch.zeros(x.shape[1])
    inference = inference.append_simulations(theta, x)

    posterior_estimator = inference.train(max_num_epochs=2)

    # Check for default device for inference object
    weights_device = next(inference._neural_net.parameters()).device
    assert torch.device(training_device) == weights_device

    _ = DirectPosterior(
        posterior_estimator=posterior_estimator, prior=prior
    ).set_default_x(x_o)


@pytest.mark.gpu
@pytest.mark.parametrize("inference_method", [SNPE_A, SNPE_C, SNRE_A, SNRE_B, SNLE])
@pytest.mark.parametrize("prior_device", ("cpu", "cuda"))
@pytest.mark.parametrize("embedding_net_device", ("cpu", "cuda"))
@pytest.mark.parametrize("data_device", ("cpu", "cuda"))
@pytest.mark.parametrize("training_device", ("cpu", "cuda"))
def test_embedding_nets_integration_training_device(
    inference_method,
    prior_device: str,
    embedding_net_device: str,
    data_device: str,
    training_device: str,
) -> None:

    # add other methods

    D_theta = 2
    D_x = 3
    samples_per_round = 32
    num_rounds = 2

    x_o = torch.ones((1, D_x))

    prior = utils.BoxUniform(
        low=-torch.ones((D_theta,)), high=torch.ones((D_theta,)), device=prior_device
    )

    if inference_method in [SNRE_A, SNRE_B]:
        embedding_net_theta = nn.Linear(in_features=D_theta, out_features=2).to(
            embedding_net_device
        )
        embedding_net_x = nn.Linear(in_features=D_x, out_features=2).to(
            embedding_net_device
        )
        nn_kwargs = dict(
            classifier=classifier_nn(
                model="resnet",
                embedding_net_x=embedding_net_x,
                embedding_net_theta=embedding_net_theta,
                hidden_features=4,
            )
        )
    elif inference_method == SNLE:
        embedding_net = nn.Linear(in_features=D_theta, out_features=2).to(
            embedding_net_device
        )
        nn_kwargs = dict(
            density_estimator=likelihood_nn(
                model="maf",
                embedding_net=embedding_net,
                hidden_features=4,
                num_transforms=2,
            )
        )
    else:
        embedding_net = nn.Linear(in_features=D_x, out_features=2).to(
            embedding_net_device
        )
        nn_kwargs = dict(
            density_estimator=posterior_nn(
                model="mdn_snpe_a" if inference_method == SNPE_A else "maf",
                embedding_net=embedding_net,
                hidden_features=4,
                num_transforms=2,
            )
        )

    with pytest.raises(Exception) if prior_device != training_device else nullcontext():
        inference = inference_method(prior=prior, **nn_kwargs, device=training_device)

    if prior_device != training_device:
        pytest.xfail("We do not correct the case of invalid prior device")

    theta = prior.sample((samples_per_round,)).to(data_device)

    proposal = prior
    for round_idx in range(num_rounds):
        X = (
            MultivariateNormal(torch.zeros((D_x,)), torch.eye(D_x))
            .sample((samples_per_round,))
            .to(data_device)
        )

        with pytest.warns(
            UserWarning
        ) if data_device != training_device else nullcontext():
            density_estimator_append = inference.append_simulations(theta, X)

        with pytest.warns(UserWarning) if (round_idx == 0) and (
            embedding_net_device != training_device
        ) else nullcontext():
            density_estimator_train = density_estimator_append.train(max_num_epochs=2)

        posterior = inference.build_posterior(density_estimator_train)
        proposal = posterior.set_default_x(x_o)
        theta = proposal.sample((samples_per_round,))


@pytest.mark.parametrize("inference_method", [SNPE_A, SNPE_C, SNRE_A, SNRE_B, SNLE])
def test_nograd_after_inference_train(inference_method) -> None:

    num_dim = 2
    prior_ = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim))
    simulator, prior = prepare_for_sbi(diagonal_linear_gaussian, prior_)

    inference = inference_method(
        prior,
        **(
            dict(classifier="resnet")
            if inference_method in [SNRE_A, SNRE_B]
            else dict(
                density_estimator=(
                    "mdn_snpe_a" if inference_method == SNPE_A else "maf"
                )
            )
        ),
        show_progress_bars=False,
    )

    theta, x = simulate_for_sbi(simulator, prior, 32)
    inference = inference.append_simulations(theta, x)

    posterior_estimator = inference.train(max_num_epochs=2)

    def check_no_grad(model):
        for p in model.parameters():
            assert p.grad is None

    check_no_grad(posterior_estimator)
    check_no_grad(inference._neural_net)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("num_dim", (1, 2))
@pytest.mark.parametrize("q", ("maf", "nsf", "gaussian_diag", "gaussian", "mcf", "scf"))
@pytest.mark.parametrize("vi_method", ("rKL", "fKL", "IW", "alpha"))
@pytest.mark.parametrize("sampling_method", ("naive", "sir"))
def test_vi_on_gpu(num_dim: int, q: Distribution, vi_method: str, sampling_method: str):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods

    """

    device = "cuda:0"

    if num_dim == 1 and q in ["mcf", "scf"]:
        return

    # Good run where everythink is one the correct device.
    class FakePotential(BasePotential):
        def __call__(self, theta, **kwargs):
            return torch.ones(len(theta), dtype=torch.float32, device=device)

        def allow_iid_x(self) -> bool:
            return True

    potential_fn = FakePotential(
        prior=MultivariateNormal(
            zeros(num_dim, device=device), eye(num_dim, device=device)
        ),
        device=device,
    )
    theta_transform = torch_tf.identity_transform

    posterior = VIPosterior(
        potential_fn=potential_fn, theta_transform=theta_transform, q=q, device=device
    )
    posterior.set_default_x(
        torch.tensor(np.zeros((num_dim,)).astype(np.float32)).to(device)
    )
    posterior.vi_method = vi_method

    posterior.train(min_num_iters=9, max_num_iters=10, warm_up_rounds=10)
    samples = posterior.sample((1,), method=sampling_method)
    logprobs = posterior.log_prob(samples)

    assert str(samples.device) == device, "The devices after training does not match"
    assert str(logprobs.device) == device, "The devices after training does not match"
