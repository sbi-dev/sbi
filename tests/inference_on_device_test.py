# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

from typing import Tuple

import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    NLE,
    NPE_A,
    NPE_C,
    NRE_A,
    NRE_B,
    NRE_C,
    VIPosterior,
    likelihood_estimator_based_potential,
    ratio_estimator_based_potential,
)
from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets.factory import (
    classifier_nn,
    embedding_net_warn_msg,
    likelihood_nn,
    posterior_nn,
)
from sbi.simulators import diagonal_linear_gaussian, linear_gaussian
from sbi.utils.torchutils import BoxUniform, gpu_available, process_device
from sbi.utils.user_input_checks import (
    validate_theta_and_x,
)

# tests in this file are skipped if there is GPU device available
pytestmark = pytest.mark.skipif(
    not gpu_available(), reason="No CUDA or MPS device available."
)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "method, model, sampling_method",
    [
        (NPE_C, "maf", "direct"),
        (NPE_C, "mdn", "rejection"),
        pytest.param(NPE_C, "maf", "slice_np_vectorized", marks=pytest.mark.mcmc),
        pytest.param(NPE_C, "mdn", "slice_np", marks=pytest.mark.mcmc),
        pytest.param(NLE, "nsf", "slice_np_vectorized", marks=pytest.mark.mcmc),
        pytest.param(NLE, "mdn", "slice_np", marks=pytest.mark.mcmc),
        (NLE, "nsf", "rejection"),
        (NLE, "maf", "importance"),
        pytest.param(NRE_A, "mlp", "slice_np_vectorized", marks=pytest.mark.mcmc),
        pytest.param(NRE_A, "mlp", "slice_np", marks=pytest.mark.mcmc),
        (NRE_B, "resnet", "rejection"),
        (NRE_B, "resnet", "importance"),
        pytest.param(NRE_B, "resnet", "slice_np", marks=pytest.mark.mcmc),
        (NRE_C, "resnet", "rejection"),
        (NRE_C, "resnet", "importance"),
        pytest.param(NRE_C, "resnet", "nuts_pymc", marks=pytest.mark.mcmc),
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

    if method in [NPE_A, NPE_C]:
        kwargs = dict(
            density_estimator=posterior_nn(
                model=model, num_transforms=2, dtype=torch.float32
            )
        )
        train_kwargs = dict(force_first_round_loss=True)
    elif method == NLE:
        kwargs = dict(
            density_estimator=likelihood_nn(
                model=model, num_transforms=2, dtype=torch.float32
            )
        )
        train_kwargs = dict()
    elif method in (NRE_A, NRE_B, NRE_C):
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
        if sampling_method in ["slice_np", "slice_np_vectorized", "nuts_pymc"]:
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
                    if sampling_method == "rejection" and method == NPE_C
                    else {}
                ),
            )
        else:
            # build potential for NLE or NRE and construct ImportanceSamplingPosterior
            if method == NLE:
                potential_fn, theta_transform = likelihood_estimator_based_potential(
                    estimator, prior, x_o
                )
            elif method in [NRE_A, NRE_B, NRE_C]:
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
@pytest.mark.parametrize("inference_method", [NPE_A, NPE_C, NRE_A, NRE_B, NRE_C, NLE])
@pytest.mark.parametrize("data_device", ("cpu", "gpu"))
@pytest.mark.parametrize("training_device", ("cpu", "gpu"))
@pytest.mark.parametrize("embedding_device", ("cpu", "gpu"))
def test_train_with_different_data_and_training_device(
    inference_method, data_device: str, training_device: str, embedding_device: str
) -> None:
    data_device = process_device(data_device)
    training_device = process_device(training_device)
    embedding_device = process_device(embedding_device)

    num_dim = 2
    num_simulations = 32
    prior = BoxUniform(
        -torch.ones(num_dim), torch.ones(num_dim), device=training_device
    )
    simulator = diagonal_linear_gaussian

    # moving embedding net to device to mimic user with large custom embedding.
    embedding_net = FCEmbedding(input_dim=num_dim, output_dim=num_dim).to(
        embedding_device
    )

    if inference_method in [NRE_A, NRE_B, NRE_C]:
        net_builder_fun = classifier_nn
        kwargs = dict(model="mlp", embedding_net_x=embedding_net)
    elif inference_method == NLE:
        net_builder_fun = likelihood_nn
        kwargs = dict(model="mdn", embedding_net=embedding_net)
    elif inference_method == NPE_A:
        net_builder_fun = posterior_nn
        kwargs = dict(model="mdn_snpe_a", embedding_net=embedding_net)
    else:
        net_builder_fun = posterior_nn
        kwargs = dict(model="mdn", embedding_net=embedding_net)

    # warning must be issued when embedding not on cpu.
    if embedding_device != "cpu":
        with pytest.warns(UserWarning, match=embedding_net_warn_msg):
            net_builder = net_builder_fun(**kwargs)
    else:
        net_builder = net_builder_fun(**kwargs)

    inference = inference_method(
        prior, net_builder, show_progress_bars=False, device=training_device
    )

    theta = prior.sample((num_simulations,))
    x = simulator(theta).to(data_device)
    theta = theta.to(data_device)
    x_o = torch.zeros(x.shape[1])
    inference = inference.append_simulations(theta, x, data_device=data_device)

    estimator = inference.train(max_num_epochs=2)

    # Check for default device for inference object
    weights_device = next(inference._neural_net.parameters()).device
    assert torch.device(training_device) == weights_device

    # Check device inference in posterior class
    posterior = inference.build_posterior(
        # use data_device as switch to test both device inference cases.
        density_estimator=estimator if data_device == "cpu" else None,
        prior=prior,
    ).set_default_x(x_o)
    assert posterior._device == str(
        weights_device
    ), "inferred posterior device not correct."


@pytest.mark.parametrize("inference_method", [NPE_A, NPE_C, NRE_A, NRE_B, NRE_C, NLE])
def test_nograd_after_inference_train(inference_method) -> None:
    """Test that no gradients are present after training."""
    num_dim = 2
    prior = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim))
    simulator = diagonal_linear_gaussian

    inference = inference_method(
        prior,
        **(
            dict(classifier="resnet")
            if inference_method in [NRE_A, NRE_B, NRE_C]
            else dict(
                density_estimator=("mdn_snpe_a" if inference_method == NPE_A else "maf")
            )
        ),
        show_progress_bars=False,
    )

    num_simulations = 32
    theta = prior.sample((num_simulations,))
    x = simulator(theta)
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
    NPE_C(prior=prior, device=arg_device)
