# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from contextlib import nullcontext
from typing import Tuple

import pytest
import torch
import torch.distributions.transforms as torch_tf
import torch.nn as nn
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    SNLE,
    SNPE_A,
    SNPE_C,
    SNRE_A,
    SNRE_B,
    SNRE_C,
    DirectPosterior,
    VIPosterior,
    likelihood_estimator_based_potential,
    ratio_estimator_based_potential,
    simulate_for_sbi,
)
from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets import classifier_nn, likelihood_nn, posterior_nn
from sbi.simulators import diagonal_linear_gaussian, linear_gaussian
from sbi.utils.torchutils import BoxUniform, gpu_available, process_device
from sbi.utils.user_input_checks import (
    check_embedding_net_device,
    validate_theta_and_x,
)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "method, model, sampling_method",
    [
        (SNPE_C, "maf", "direct"),
        (SNPE_C, "mdn", "rejection"),
        pytest.param(SNPE_C, "maf", "slice_np_vectorized", marks=pytest.mark.mcmc),
        pytest.param(SNPE_C, "mdn", "slice", marks=pytest.mark.mcmc),
        pytest.param(SNLE, "nsf", "slice_np_vectorized", marks=pytest.mark.mcmc),
        pytest.param(SNLE, "mdn", "slice", marks=pytest.mark.mcmc),
        (SNLE, "nsf", "rejection"),
        (SNLE, "maf", "importance"),
        pytest.param(SNRE_A, "mlp", "slice_np_vectorized", marks=pytest.mark.mcmc),
        pytest.param(SNRE_A, "mlp", "slice", marks=pytest.mark.mcmc),
        (SNRE_B, "resnet", "rejection"),
        (SNRE_B, "resnet", "importance"),
        pytest.param(SNRE_B, "resnet", "slice", marks=pytest.mark.mcmc),
        (SNRE_C, "resnet", "rejection"),
        (SNRE_C, "resnet", "importance"),
        pytest.param(SNRE_C, "resnet", "nuts", marks=pytest.mark.mcmc),
    ],
)
@pytest.mark.parametrize(
    "training_device, prior_device",
    [
        ("gpu", "gpu"),
        pytest.param("cpu", "gpu", marks=pytest.mark.xfail),
        pytest.param("gpu", "cpu", marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize("prior_type", ["gaussian", "uniform"])
def test_training_and_mcmc_on_device(
    method,
    model,
    sampling_method,
    training_device,
    prior_device,
    prior_type,
    mcmc_params_fast: dict,
):
    """Test training on devices.

    This test does not check training speeds.

    """

    training_device = process_device(training_device)
    data_device = "cpu"
    prior_device = process_device(prior_device)

    num_dim = 2
    num_samples = 10
    max_num_epochs = 10
    num_rounds = 2  # test proposal sampling in round 2.
    num_simulations_per_round = [200, num_samples]
    # use more warmup steps to avoid Infs during MCMC in round two.
    mcmc_params_fast["warmup_steps"] = 20

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

    if method in [SNPE_A, SNPE_C]:
        kwargs = dict(
            density_estimator=posterior_nn(
                model=model, num_transforms=2, dtype=torch.float32
            )
        )
        train_kwargs = dict(force_first_round_loss=True)
    elif method == SNLE:
        kwargs = dict(
            density_estimator=likelihood_nn(
                model=model, num_transforms=2, dtype=torch.float32
            )
        )
        train_kwargs = dict()
    elif method in (SNRE_A, SNRE_B, SNRE_C):
        kwargs = dict(classifier=classifier_nn(model=model))
        train_kwargs = dict()
    else:
        raise ValueError()

    inferer = method(
        prior=prior, show_progress_bars=False, device=training_device, **kwargs
    )

    proposals = [prior]

    for _ in range(num_rounds):
        theta = proposals[-1].sample((num_simulations_per_round[_],))
        x = simulator(theta).to(data_device)
        theta = theta.to(data_device)

        estimator = inferer.append_simulations(theta, x, data_device=data_device).train(
            training_batch_size=100, max_num_epochs=max_num_epochs, **train_kwargs
        )

        # mcmc cases
        if sampling_method in ["slice", "slice_np", "slice_np_vectorized", "nuts"]:
            posterior = inferer.build_posterior(
                sample_with="mcmc",
                mcmc_method=sampling_method,
                mcmc_parameters=mcmc_params_fast,
            )
        elif sampling_method in ["rejection", "direct"]:
            # all other cases: rejection, direct
            posterior = inferer.build_posterior(
                sample_with=sampling_method,
                rejection_sampling_parameters=(
                    {"proposal": prior}
                    if sampling_method == "rejection" and method == SNPE_C
                    else {}
                ),
            )
        else:
            # build potential for SNLE or SNRE and construct ImportanceSamplingPosterior
            if method == SNLE:
                potential_fn, theta_transform = likelihood_estimator_based_potential(
                    estimator, prior, x_o
                )
            elif method in [SNRE_A, SNRE_B, SNRE_C]:
                potential_fn, theta_transform = ratio_estimator_based_potential(
                    estimator, prior, x_o
                )
            else:
                raise ValueError()
            posterior = ImportanceSamplingPosterior(
                potential_fn, prior, theta_transform
            )
        proposals.append(posterior.set_default_x(x_o))

    # Check for default device for inference object
    weights_device = next(inferer._neural_net.parameters()).device
    assert torch.device(training_device) == weights_device
    samples = proposals[-1].sample(sample_shape=(num_samples,))
    proposals[-1].potential(samples)


@pytest.mark.gpu
@pytest.mark.parametrize("device_datum", ["cpu", "gpu"])
@pytest.mark.parametrize("device_embedding_net", ["cpu", "gpu"])
def test_check_embedding_net_device(
    device_datum: str, device_embedding_net: str
) -> None:
    device_datum = process_device(device_datum)
    device_embedding_net = process_device(device_embedding_net)

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


@pytest.mark.parametrize("shape_x", [(3, 1)])
@pytest.mark.parametrize(
    "shape_theta", [(3, 2), pytest.param((2, 1), marks=pytest.mark.xfail)]
)
def test_validate_theta_and_x_shapes(
    shape_x: Tuple[int], shape_theta: Tuple[int]
) -> None:
    """Test validate_theta_and_x with different shapes."""
    x = torch.empty(shape_x)
    theta = torch.empty(shape_theta)

    validate_theta_and_x(theta, x, training_device="cpu")


def test_validate_theta_and_x_tensor() -> None:
    """Test whether validate_theta_and_x raises Exceptio when list is passed."""
    x = torch.empty((1, 1))
    theta = torch.ones((1, 1)).tolist()

    with pytest.raises(AssertionError):
        validate_theta_and_x(theta, x, training_device="cpu")


def test_validate_theta_and_x_type() -> None:
    """Test whether validate_theta_and_x raises Exceptio when empty x is passed."""
    x = torch.empty((1, 1))
    theta = torch.empty((1, 1), dtype=int)

    with pytest.raises(AssertionError):
        validate_theta_and_x(theta, x, training_device="cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("training_device", ["cpu", "gpu"])
@pytest.mark.parametrize("data_device", ["cpu", "gpu"])
def test_validate_theta_and_x_device(training_device: str, data_device: str) -> None:
    training_device = process_device(training_device)
    data_device = process_device(data_device)

    theta = torch.empty((1, 1)).to(data_device)
    x = torch.empty((1, 1)).to(data_device)

    theta, x = validate_theta_and_x(
        theta, x, data_device=data_device, training_device=training_device
    )

    assert str(theta.device) == data_device, (
        f"Data and parameters must be on the same device but:"
        f"data device='{data_device}' and training_device='{training_device}'."
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "inference_method", [SNPE_A, SNPE_C, SNRE_A, SNRE_B, SNRE_C, SNLE]
)
@pytest.mark.parametrize("data_device", ("cpu", "gpu"))
@pytest.mark.parametrize("training_device", ("cpu", "gpu"))
def test_train_with_different_data_and_training_device(
    inference_method, data_device: str, training_device: str
) -> None:
    assert gpu_available(), "this test requires that gpu is available."

    data_device = process_device(data_device)
    training_device = process_device(training_device)

    num_dim = 2
    num_simulations = 32
    prior = BoxUniform(
        -torch.ones(num_dim), torch.ones(num_dim), device=training_device
    )
    simulator = diagonal_linear_gaussian

    inference = inference_method(
        prior,
        **(
            dict(classifier="resnet")
            if inference_method in [SNRE_A, SNRE_B, SNRE_C]
            else dict(
                density_estimator=(
                    "mdn_snpe_a" if inference_method == SNPE_A else "maf"
                )
            )
        ),
        show_progress_bars=False,
        device=training_device,
    )

    theta = prior.sample((num_simulations,))
    x = simulator(theta).to(data_device)
    theta = theta.to(data_device)
    x_o = torch.zeros(x.shape[1])
    inference = inference.append_simulations(theta, x, data_device=data_device)

    posterior_estimator = inference.train(max_num_epochs=2)

    # Check for default device for inference object
    weights_device = next(inference._neural_net.parameters()).device
    assert torch.device(training_device) == weights_device

    _ = DirectPosterior(
        posterior_estimator=posterior_estimator, prior=prior
    ).set_default_x(x_o)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "inference_method", [SNPE_A, SNPE_C, SNRE_A, SNRE_B, SNRE_C, SNLE]
)
@pytest.mark.parametrize("prior_device", ("cpu", "gpu"))
@pytest.mark.parametrize("embedding_net_device", ("cpu", "gpu"))
@pytest.mark.parametrize("data_device", ("cpu", "gpu"))
@pytest.mark.parametrize("training_device", ("cpu", "gpu"))
def test_embedding_nets_integration_training_device(
    inference_method,
    prior_device: str,
    embedding_net_device: str,
    data_device: str,
    training_device: str,
    mcmc_params_fast: dict,
) -> None:
    """Test embedding nets integration with different devices, priors and methods."""
    # add other methods

    theta_dim = 2
    x_dim = 3
    # process all device strings
    prior_device = process_device(prior_device)
    embedding_net_device = process_device(embedding_net_device)
    data_device = process_device(data_device)
    training_device = process_device(training_device)

    samples_per_round = 64
    num_rounds = 2

    x_o = torch.ones((1, x_dim))

    prior = utils.BoxUniform(
        low=-torch.ones((theta_dim,)),
        high=torch.ones((theta_dim,)),
        device=prior_device,
    )

    if inference_method in [SNRE_A, SNRE_B, SNRE_C]:
        embedding_net_theta = nn.Linear(in_features=theta_dim, out_features=2).to(
            embedding_net_device
        )
        embedding_net_x = nn.Linear(in_features=x_dim, out_features=2).to(
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
        train_kwargs = dict()
    elif inference_method == SNLE:
        embedding_net = nn.Linear(in_features=theta_dim, out_features=2).to(
            embedding_net_device
        )
        nn_kwargs = dict(
            density_estimator=likelihood_nn(
                model="mdn",
                embedding_net=embedding_net,
                hidden_features=4,
                num_transforms=2,
            )
        )
        train_kwargs = dict()
    else:
        embedding_net = nn.Linear(in_features=x_dim, out_features=2).to(
            embedding_net_device
        )
        nn_kwargs = dict(
            density_estimator=posterior_nn(
                model="mdn_snpe_a" if inference_method == SNPE_A else "mdn",
                embedding_net=embedding_net,
                hidden_features=4,
                num_transforms=2,
            )
        )
        if inference_method == SNPE_A:
            train_kwargs = dict()
        else:
            train_kwargs = dict(force_first_round_loss=True)

    with pytest.raises(Exception) if prior_device != training_device else nullcontext():
        inference = inference_method(prior=prior, **nn_kwargs, device=training_device)

    if prior_device != training_device:
        pytest.xfail("We do not correct the case of invalid prior device")

    theta = prior.sample((samples_per_round,)).to(data_device)

    proposal = prior
    for _ in range(num_rounds):
        # sample theta and x independently - quick way to get 3D simulation data.
        theta = proposal.sample((samples_per_round,))
        x = (
            MultivariateNormal(torch.zeros((x_dim,)), torch.eye(x_dim))
            .sample((samples_per_round,))
            .to(data_device)
        )

        with (
            pytest.warns(UserWarning)
            if data_device != training_device
            else nullcontext()
        ):
            density_estimator_append = inference.append_simulations(theta, x)

        density_estimator_train = density_estimator_append.train(
            max_num_epochs=2, **train_kwargs
        )

        posterior = inference.build_posterior(
            density_estimator_train,
            **(
                {}
                if inference_method == SNPE_A
                else dict(
                    mcmc_method="slice_np_vectorized",
                    mcmc_parameters=mcmc_params_fast,
                )
            ),
        )
        proposal = posterior.set_default_x(x_o)


@pytest.mark.parametrize(
    "inference_method", [SNPE_A, SNPE_C, SNRE_A, SNRE_B, SNRE_C, SNLE]
)
def test_nograd_after_inference_train(inference_method) -> None:
    """Test that no gradients are present after training."""
    num_dim = 2
    prior = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim))
    simulator = diagonal_linear_gaussian

    inference = inference_method(
        prior,
        **(
            dict(classifier="resnet")
            if inference_method in [SNRE_A, SNRE_B, SNRE_C]
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
@pytest.mark.parametrize("num_dim", (1, 3))
# NOTE: macOS MPS fails for nsf with num_dim > 1
# might be related to https://github.com/pytorch/pytorch/issues/89127
@pytest.mark.parametrize("q", ("maf", "nsf", "gaussian_diag", "gaussian", "mcf", "scf"))
@pytest.mark.parametrize("vi_method", ("rKL", "fKL", "IW", "alpha"))
@pytest.mark.parametrize("sampling_method", ("naive", "sir"))
def test_vi_on_gpu(num_dim: int, q: str, vi_method: str, sampling_method: str):
    """Test VI on Gaussian, comparing to ground truth target via c2st.

    Args:
        num_dim: parameter dimension of the gaussian model
        vi_method: different vi methods
        sampling_method: Different sampling methods
    """

    device = process_device("gpu")

    if num_dim == 1 and q in ["mcf", "scf"]:
        return

    # Skip the test for nsf on mps:0 as it results in NaNs.
    if device == "mps:0" and num_dim > 1 and q == "nsf":
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
    posterior.set_default_x(torch.zeros((num_dim,), dtype=torch.float32).to(device))
    posterior.vi_method = vi_method

    posterior.train(min_num_iters=9, max_num_iters=10, warm_up_rounds=10)
    samples = posterior.sample((1,), method=sampling_method)
    logprobs = posterior.log_prob(samples)

    assert (
        str(samples.device) == device
    ), f"The devices after training do not match: {samples.device} vs {device}"
    assert (
        str(logprobs.device) == device
    ), f"The devices after training do not match: {logprobs.device} vs {device}"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "arg_device, device",
    [
        ("cpu", None),
        ("gpu", None),
        ("cpu", "cpu"),
        ("gpu", "gpu"),
        pytest.param("gpu", "cpu", marks=pytest.mark.xfail),
        pytest.param("cpu", "gpu", marks=pytest.mark.xfail),
    ],
)
def test_boxuniform_device_handling(arg_device, device):
    """Test mismatch between device passed via low / high and device kwarg."""

    arg_device = process_device(arg_device)
    device = process_device(device)

    prior = BoxUniform(
        low=zeros(1).to(arg_device), high=ones(1).to(arg_device), device=device
    )
    SNPE_C(prior=prior, device=arg_device)
