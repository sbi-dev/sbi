# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import sys
from typing import Tuple, Union

import pymc
import pytest
import torch
import torch.distributions.transforms as torch_tf
from torch import eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils as utils
from sbi.inference import (
    ABC,
    FMPE,
    NLE,
    NPE,
    NPE_A,
    NPE_C,
    NRE_A,
    NRE_B,
    NRE_C,
    VIPosterior,
    likelihood_estimator_based_potential,
    ratio_estimator_based_potential,
)
from sbi.inference.posteriors.ensemble_posterior import (
    EnsemblePotential,
)
from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.inference.potentials.posterior_based_potential import PosteriorBasedPotential
from sbi.inference.potentials.ratio_based_potential import RatioBasedPotential
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.neural_nets.factory import (
    classifier_nn,
    embedding_net_warn_msg,
    likelihood_nn,
    posterior_nn,
)
from sbi.simulators import diagonal_linear_gaussian, linear_gaussian
from sbi.utils.torchutils import (
    BoxUniform,
    gpu_available,
    process_device,
)
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
        pytest.param(
            NRE_C,
            "resnet",
            "nuts_pymc",
            marks=(
                pytest.mark.mcmc,
                pytest.mark.skipif(
                    condition=sys.version_info >= (3, 10)
                    and pymc.__version__ >= "5.20.1",
                    reason="Inconsistent behaviour with pymc>=5.20.1 and python>=3.10",
                ),
            ),
        ),
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
        train_kwargs = dict()
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

        data_kwargs = (
            dict(proposal=proposals[-1]) if method in [NPE_A, NPE_C] else dict()
        )
        estimator = inferer.append_simulations(
            theta, x, data_device=data_device, **data_kwargs
        ).train(max_num_epochs=max_num_epochs, **train_kwargs)

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
    assert posterior._device == str(weights_device), (
        "inferred posterior device not correct."
    )


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

    assert str(samples.device) == device, (
        f"The devices after training do not match: {samples.device} vs {device}"
    )
    assert str(logprobs.device) == device, (
        f"The devices after training do not match: {logprobs.device} vs {device}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "arg_device, device",
    [
        ("cpu", None),
        ("gpu", None),
        ("cpu", "cpu"),
        ("gpu", "gpu"),
        (torch.device("cpu"), torch.device("cpu")),
        pytest.param("gpu", "cpu", marks=pytest.mark.xfail),
        pytest.param("cpu", "gpu", marks=pytest.mark.xfail),
    ],
)
def test_boxuniform_device_handling(arg_device, device):
    """Test mismatch between device passed via low / high and device kwarg.

    Also tests torch.device as argument of process_device."""

    arg_device = process_device(arg_device)
    device = process_device(device)

    prior = BoxUniform(
        low=zeros(1).to(arg_device), high=ones(1).to(arg_device), device=device
    )
    NPE_C(prior=prior, device=arg_device)


@pytest.mark.gpu
@pytest.mark.parametrize("method", [NPE_A, NPE_C])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_multiround_mdn_training_on_device(method: Union[NPE_A, NPE_C], device: str):
    num_dim = 2
    num_rounds = 2
    num_simulations = 100
    device = process_device("gpu")
    prior = BoxUniform(-torch.ones(num_dim), torch.ones(num_dim), device=device)
    simulator = diagonal_linear_gaussian

    estimator = "mdn_snpe_a" if method == NPE_A else "mdn"

    trainer = method(prior, density_estimator=estimator, device=device)

    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    proposal = prior
    for _ in range(num_rounds):
        trainer.append_simulations(theta, x, proposal=proposal).train(max_num_epochs=2)
        proposal = trainer.build_posterior().set_default_x(torch.zeros(num_dim))
        theta = proposal.sample((num_simulations,))
        x = simulator(theta)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "training_device, inference_device", [("cpu", "gpu"), ("gpu", "cpu")]
)
def test_conditioned_posterior_on_gpu(
    training_device: str, inference_device: str, mcmc_params_fast: dict
):
    """Test that training and sampling device can be interchanged

    for conditional posteriors.

    Args:
        training_device: device for trainig
        inference_device: device for inference
        mcmc_params_fast: dictionary for mcmc posterior
    """

    # Training.
    training_device = process_device(training_device)
    num_dims = 3

    proposal = BoxUniform(
        low=-torch.ones(num_dims), high=torch.ones(num_dims), device=training_device
    )

    trainer = NPE_C(device=training_device, show_progress_bars=False)

    num_simulations = 100
    theta = proposal.sample((num_simulations,))
    x = torch.randn_like(theta)
    x_o = torch.zeros(1, num_dims).to(training_device)
    trainer = trainer.append_simulations(theta, x)

    estimator = trainer.train(max_num_epochs=2)

    # Inference.
    inference_device = process_device(inference_device)
    condition_o = torch.ones(1, 1, device=inference_device)
    estimator.to(inference_device)
    proposal.to(inference_device)
    prior = BoxUniform(
        low=-torch.ones(num_dims - 1),
        high=torch.ones(num_dims - 1),
        device=inference_device,
    )

    potential_fn, prior_transform = likelihood_estimator_based_potential(
        estimator,
        proposal,
        x_o.to(inference_device),
    )
    conditioned_potential_fn = potential_fn.condition_on_theta(
        condition_o, dims_global_theta=[0, 1]
    )

    prior_transform = utils.mcmc_transform(prior, device=inference_device)

    conditional_posterior = MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        theta_transform=prior_transform,
        proposal=prior,
        device=inference_device,
        **mcmc_params_fast,
    ).set_default_x(x_o)

    conditional_posterior.to(inference_device)
    samples = conditional_posterior.sample((1,), x=x_o.to(inference_device))
    assert str(samples.device).split(":")[0] == inference_device.split(":")[0], (
        "Samples are not on the correct device"
    )
    conditional_posterior.potential_fn(samples)
    map_ = conditional_posterior.map()
    assert str(map_.device).split(":")[0] == inference_device.split(":")[0], (
        "MAP is not on the correct device"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("device_inference", ["cpu", "gpu"])
def test_direct_posterior_on_gpu(device: str, device_inference: str):
    """Test that training and sampling device can be interchanged.

    Args:
        device: device to train the model on.
        device_inference: device to run the inference on.
    """
    device = process_device(device)
    num_dims = 3

    prior = BoxUniform(
        low=-torch.ones(num_dims, device=device),
        high=torch.ones(num_dims, device=device),
    )
    x_o = torch.zeros(1, num_dims).to(device)

    inference = NPE()
    estimator = inference.append_simulations(
        torch.randn((100, num_dims)), torch.randn((100, num_dims))
    ).train(max_num_epochs=1)
    posterior = inference.build_posterior(
        density_estimator=estimator, prior=prior, sample_with="direct"
    )
    posterior.set_default_x(x_o)

    device_inference = process_device(device_inference)
    posterior.to(device_inference)
    sample = posterior.sample((1,), x=x_o.to(device_inference))
    assert str(sample.device).split(":")[0] == device_inference.split(":")[0], (
        "Samples are not on the correct device."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "potential",
    [
        LikelihoodBasedPotential,
        PosteriorBasedPotential,
        RatioBasedPotential,
        EnsemblePotential,
    ],
)
def test_to_method_on_potentials(device: str, potential: Union[ABC, BasePotential]):
    """Test .to() method on potential.

    Args:
        device: device where to move the model.
        potential: potential to train the model on.
    """

    device = process_device(device)
    prior = BoxUniform(torch.tensor([1.0]), torch.tensor([1.0]))
    inference = NPE()
    estimator = inference.append_simulations(
        torch.randn((100, 3)), torch.randn((100, 2))
    ).train(max_num_epochs=1)

    x_o = torch.tensor([0.1]).to(device)
    if potential == EnsemblePotential:
        potential_fn = potential(
            [
                RatioBasedPotential(estimator, prior),
                PosteriorBasedPotential(estimator, prior),
            ],
            prior=prior,
            x_o=x_o,
            weights=torch.tensor([0.1, 0.9]),
        )
    else:
        potential_fn = potential(estimator, prior)
    potential_fn.to(device)

    assert str(potential_fn.device).split(":")[0] == device.split(":")[0], (
        "Device attribute of potential_fn is not correct"
    )
    if hasattr("potential", "prior"):
        assert str(potential_fn).split(":")[0].prior == device.split(":")[0], (
            "Device attribute of potential_fn.prior is not vcorrect"
        )


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "sampling_method", ["rejection", "importance", "mcmc", "direct"]
)
def test_to_method_on_posteriors(device: str, sampling_method: str):
    """Test .to() method on posteriors.

    Args:
        device: device to train and sample the model on.
        sampling_method: method to sample from the posterior.
    """
    device = process_device(device)
    prior = BoxUniform(torch.zeros(3), torch.ones(3))
    inference = NPE()
    x_o = torch.zeros(2).to(device)
    estimator = inference.append_simulations(
        torch.randn((100, 3)), torch.randn((100, 2))
    ).train(max_num_epochs=1)
    if sampling_method == "rejection":
        posterior = inference.build_posterior(
            density_estimator=estimator,
            prior=prior,
            rejection_sampling_parameters={"proposal": prior},
            sample_with=sampling_method,
        )
    else:
        posterior = inference.build_posterior(
            density_estimator=estimator, prior=prior, sample_with=sampling_method
        )
    posterior.set_default_x(x_o)
    posterior.to(device)

    assert (posterior.device).split(":")[0] == device.split(":")[0], (
        ".to() should change the device attribute"
    )
    sample_device = posterior.sample((10,), x=x_o)
    assert sample_device.device.type == device.split(":")[0], (
        f"sample was not correctly moved to {device}."
    )
    log_probs = posterior.log_prob(sample_device)
    assert log_probs.device.type == device.split(":")[0], (
        f"log_prob was not correctly moved to {device}."
    )

    for trasnf in posterior.theta_transform._inv.base_transform.parts:
        assert (
            str(trasnf(torch.tensor([0.0], device=device)).device).strip(":0")
            == device.split(":")[0]
        ), "Prior transform is on the correct device."


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("device_inference", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "iid_method", ["fnpe", "gauss", "auto_gauss", "jac_gauss", None]
)
def test_VectorFieldPosterior_device_handling(
    device: str, device_inference: str, iid_method: str
):
    """Test VectorFieldPosterior on different devices training and inference devices.

    Args:
        device: device to train the model on.
        device_inference: device to run the inference on.
        iid_method: method to sample from the posterior.
    """
    device = process_device(device)
    device_inference = process_device(device_inference)
    prior = BoxUniform(torch.zeros(3), torch.ones(3), device=device)
    inference = FMPE(score_estimator="mlp", prior=prior, device=device)
    density_estimator = inference.append_simulations(
        torch.randn((100, 3)), torch.randn((100, 2))
    ).train(max_num_epochs=1)
    posterior = inference.build_posterior(density_estimator, prior)

    # faster but inaccurate log_prob computation
    posterior.potential_fn.neural_ode.update_params(exact=False, atol=1e-4, rtol=1e-4)

    posterior.to(device_inference)
    assert posterior.device == device_inference, (
        f"VectorFieldPosterior is not in device {device_inference}."
    )

    x_o = torch.ones(2).to(device_inference)
    samples = posterior.sample((2,), x=x_o, iid_method=iid_method)
    assert samples.device.type == device_inference.split(":")[0], (
        f"Samples are not on device {device_inference}."
    )

    log_probs = posterior.log_prob(samples, x=x_o)
    assert log_probs.device.type == device_inference.split(":")[0], (
        f"log_prob was not correctly moved to {device_inference}."
    )
