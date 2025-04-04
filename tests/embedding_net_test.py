# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from __future__ import annotations

import math
import sys

import pytest
import torch
from torch import Tensor, eye, ones, zeros
from torch.distributions import MultivariateNormal

from sbi import utils
from sbi.inference import NLE, NPE, NRE, simulate_for_sbi
from sbi.neural_nets import classifier_nn, likelihood_nn, posterior_nn
from sbi.neural_nets.embedding_nets import (
    CNNEmbedding,
    CausalCNNEmbedding,
    FCEmbedding,
    LRUEmbedding,
    PermutationInvariantEmbedding,
    ResNetEmbedding1D,
    ResNetEmbedding2D,
    SpectralConvEmbedding,
)
from sbi.neural_nets.embedding_nets.lru import LRU, LRUBlock
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.metrics import check_c2st
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


@pytest.mark.mcmc
@pytest.mark.parametrize("method", ["NPE", "NLE", "NRE"])
@pytest.mark.parametrize("num_dim", [1, 2])
@pytest.mark.parametrize("embedding_net", ["mlp"])
def test_embedding_net_api(
    method, num_dim: int, embedding_net: str, mcmc_params_fast: dict
):
    """Tests the API when using a preconfigured embedding net."""

    x_o = zeros(1, num_dim)

    # likelihood_mean will be likelihood_shift+theta
    likelihood_shift = -1.0 * ones(num_dim)
    likelihood_cov = 0.3 * eye(num_dim)

    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    theta = prior.sample((1000,))
    x = linear_gaussian(theta, likelihood_shift, likelihood_cov)

    if embedding_net == "mlp":
        embedding = FCEmbedding(input_dim=num_dim)
    else:
        raise NameError(f"{embedding_net} not supported.")

    if method == "NPE":
        density_estimator = posterior_nn("maf", embedding_net=embedding)
        inference = NPE(
            prior, density_estimator=density_estimator, show_progress_bars=False
        )
    elif method == "NLE":
        density_estimator = likelihood_nn("maf", embedding_net=embedding)
        inference = NLE(
            prior, density_estimator=density_estimator, show_progress_bars=False
        )
    elif method == "NRE":
        classifier = classifier_nn("resnet", embedding_net_x=embedding)
        inference = NRE(prior, classifier=classifier, show_progress_bars=False)
    else:
        raise NameError

    _ = inference.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = inference.build_posterior(
        mcmc_method="slice_np_vectorized",
        mcmc_parameters=mcmc_params_fast,
    ).set_default_x(x_o)

    s = posterior.sample((1,))
    _ = posterior.potential(s)


@pytest.mark.parametrize("num_xo_batch", [1, 2])
@pytest.mark.parametrize("num_trials", [1, 2])
@pytest.mark.parametrize("num_dim", [1, 2])
@pytest.mark.parametrize("posterior_method", ["direct", "mcmc"])
def test_embedding_api_with_multiple_trials(
    num_xo_batch, num_trials, num_dim, posterior_method
):
    """Tests the API when using iid trial-based data."""
    prior = utils.BoxUniform(-2.0 * ones(num_dim), 2.0 * ones(num_dim))

    num_thetas = 1000
    theta = prior.sample((num_thetas,))

    # simulate iid x.
    iid_theta = theta.reshape(num_thetas, 1, num_dim).repeat(1, num_trials, 1)
    x = torch.randn_like(iid_theta) + iid_theta
    x_o = zeros(num_xo_batch, num_trials, num_dim)

    output_dim = 5
    single_trial_net = FCEmbedding(input_dim=num_dim, output_dim=output_dim)
    embedding_net = PermutationInvariantEmbedding(
        single_trial_net,
        trial_net_output_dim=output_dim,
    )

    density_estimator = posterior_nn("maf", embedding_net=embedding_net)
    inference = NPE(prior, density_estimator=density_estimator)

    _ = inference.append_simulations(theta, x).train(max_num_epochs=5)

    if posterior_method == "direct":
        posterior = inference.build_posterior().set_default_x(x_o)
    elif posterior_method == "mcmc":
        posterior = inference.build_posterior(
            sample_with=posterior_method,
            mcmc_method="slice_np_vectorized",
        ).set_default_x(x_o)
    if num_xo_batch == 1:
        s = posterior.sample((1,), x=x_o)
        _ = posterior.potential(s)
    else:
        s = posterior.sample_batched((1,), x=x_o).squeeze(0)
        # potentials take `theta` as (batch_shape, event_shape), so squeeze sample_dim
        s = s.squeeze(0)
        _ = posterior.potential(s)


@pytest.mark.parametrize("input_shape", [(32,), (32, 32), (32, 64)])
@pytest.mark.parametrize("num_channels", (1, 2, 3))
def test_1d_and_2d_cnn_embedding_net(input_shape, num_channels):
    estimator_provider = posterior_nn(
        "mdn",
        embedding_net=CNNEmbedding(
            input_shape, in_channels=num_channels, output_dim=20
        ),
    )

    num_dim = input_shape[0]

    def simulator2d(theta):
        x = MultivariateNormal(
            loc=theta, covariance_matrix=0.5 * torch.eye(num_dim)
        ).sample()
        return x.unsqueeze(2).repeat(1, 1, input_shape[1])

    def simulator1d(theta):
        return torch.rand_like(theta) + theta

    if len(input_shape) == 1:
        simulator = simulator1d
        xo = torch.ones(1, num_channels, *input_shape).squeeze(1)
    else:
        simulator = simulator2d
        xo = torch.ones(1, num_channels, *input_shape).squeeze(1)

    prior = MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))

    num_simulations = 1000
    theta = prior.sample(torch.Size((num_simulations,)))
    x = simulator(theta)
    if num_channels > 1:
        x = x.unsqueeze(1).repeat(
            1, num_channels, *[1 for _ in range(len(input_shape))]
        )

    trainer = NPE(prior=prior, density_estimator=estimator_provider)
    trainer.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = trainer.build_posterior().set_default_x(xo)

    s = posterior.sample((10,))
    posterior.potential(s)


@pytest.mark.parametrize("input_shape", [(3, 30), (2, 3, 30)])
@pytest.mark.parametrize("modes", (4, 8))
@pytest.mark.parametrize("conv_channels", (8, 5))
@pytest.mark.parametrize("num_layers", (2, 3))
def test_spectral_conf_embedding(input_shape, modes, conv_channels, num_layers):
    n_points = input_shape[-1]
    in_channels = input_shape[-2]
    estimator_provider = posterior_nn(
        "mdn",
        embedding_net=SpectralConvEmbedding(
            modes=modes,
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_layers=num_layers,
        ),
    )

    def simulator(theta, input_shape=input_shape):
        x = torch.rand_like(theta) + theta
        return repeat_to_match_shape(x, input_shape)

    def repeat_to_match_shape(x, input_shape):
        batch_size = x.shape[0]  # First dimension is batch
        target_shape = (batch_size, *input_shape)
        x_expanded = x.view(batch_size, *([1] * (len(input_shape) - 1)), -1)
        return x_expanded.expand(target_shape)

    xo = torch.ones((1, n_points))
    xo = repeat_to_match_shape(xo, input_shape)

    prior = MultivariateNormal(torch.zeros(n_points), torch.eye(n_points))

    num_simulations = 1000
    theta = prior.sample(torch.Size((num_simulations,)))
    x = simulator(theta)

    trainer = NPE(prior=prior, density_estimator=estimator_provider)
    trainer.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = trainer.build_posterior().set_default_x(xo)

    s = posterior.sample((10,))
    posterior.potential(s)


@pytest.mark.parametrize("input_shape", [(32,), (64,)])
@pytest.mark.parametrize("num_channels", (1, 2, 3))
def test_1d_causal_cnn_embedding_net(input_shape, num_channels):
    estimator_provider = posterior_nn(
        "mdn",
        embedding_net=CausalCNNEmbedding(
            input_shape, in_channels=num_channels, pool_kernel_size=2, output_dim=20
        ),
    )

    num_dim = input_shape[0]

    def simulator2d(theta):
        x = MultivariateNormal(
            loc=theta, covariance_matrix=0.5 * torch.eye(num_dim)
        ).sample()
        return x.unsqueeze(2).repeat(1, 1, input_shape[1])

    def simulator1d(theta):
        return torch.rand_like(theta) + theta

    if len(input_shape) == 1:
        simulator = simulator1d
        xo = torch.ones(1, num_channels, *input_shape).squeeze(1)
    else:
        simulator = simulator2d
        xo = torch.ones(1, num_channels, *input_shape).squeeze(1)

    prior = MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))

    num_simulations = 1000
    theta = prior.sample(torch.Size((num_simulations,)))
    x = simulator(theta)

    if num_channels > 1:
        x = x.unsqueeze(1).repeat(
            1, num_channels, *[1 for _ in range(len(input_shape))]
        )

    trainer = NPE(prior=prior, density_estimator=estimator_provider)
    trainer.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = trainer.build_posterior().set_default_x(xo)

    s = posterior.sample((10,))
    posterior.potential(s)


@pytest.mark.slow
def test_npe_with_with_iid_embedding_varying_num_trials(trial_factor=50):
    """Test inference accuracy with embeddings for varying number of trials.

    Test c2st accuracy and permutation invariance for up to 20 trials.
    """
    num_dim = 2
    max_num_trials = 20
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(num_dim), torch.eye(num_dim)
    )

    # Scale number of training samples with num_trials.
    num_thetas = 5000 + trial_factor * max_num_trials

    theta = prior.sample(sample_shape=torch.Size((num_thetas,)))
    num_trials = torch.randint(1, max_num_trials, size=(num_thetas,))

    # simulate iid x, pad smaller number of trials with nans.
    x = ones(num_thetas, max_num_trials, 2) * float("nan")

    for i in range(num_thetas):
        th = theta[i].repeat(num_trials[i], 1)
        x[i, : num_trials[i]] = torch.randn_like(th) + th

    # build embedding net
    output_dim = 5
    single_trial_net = FCEmbedding(input_dim=num_dim, output_dim=output_dim)
    embedding_net = PermutationInvariantEmbedding(
        single_trial_net,
        trial_net_output_dim=output_dim,
        output_dim=output_dim,
        aggregation_fn="sum",
    )

    # test embedding net
    assert embedding_net(x[:3]).shape == (3, output_dim)

    density_estimator = posterior_nn(
        model="mdn",
        embedding_net=embedding_net,
        z_score_x="none",  # turn off z-scoring because of NaN encodings.
        z_score_theta="independent",
    )
    inference = NPE(prior, density_estimator=density_estimator)

    # do not exclude invalid x, as we padded with nans.
    _ = inference.append_simulations(theta, x, exclude_invalid_x=False).train(
        training_batch_size=100
    )
    posterior = inference.build_posterior()

    num_samples = 1000
    # test different number of trials
    num_test_trials = torch.linspace(1, max_num_trials, 5, dtype=torch.int)
    for num_trials in num_test_trials:
        # x_o must have the same number of trials as x, thus we pad with nans.
        x_o = ones(1, max_num_trials, num_dim) * float("nan")
        x_o[:, :num_trials] = 0.0

        # get reference samples from true posterior
        reference_samples = true_posterior_linear_gaussian_mvn_prior(
            x_o[0, :num_trials, :],  # omit nans
            likelihood_shift=torch.zeros(num_dim),
            likelihood_cov=torch.eye(num_dim),
            prior_cov=prior.covariance_matrix,
            prior_mean=prior.loc,
        ).sample((num_samples,))

        # test inference accuracy and permutation invariance
        num_repeats = 2
        for _ in range(num_repeats):
            trial_permutet_x_o = x_o[:, torch.randperm(x_o.shape[1]), :]
            samples = posterior.sample((num_samples,), x=trial_permutet_x_o)
            check_c2st(
                samples, reference_samples, alg=f"iid-NPE with {num_trials} trials"
            )


@pytest.mark.parametrize("input_shape", [(32, 32), (32, 64), (111, 111)])
@pytest.mark.parametrize("num_channels", (1, 2, 3))
@pytest.mark.parametrize("change_c_mode", ["conv", "zeros"])
@pytest.mark.parametrize("n_stages", [1, 3, 4])
def test_2d_ResNet_cnn_embedding_net(
    input_shape, num_channels, change_c_mode, n_stages
):
    c_stages = [16, 32, 64, 128]
    blocks_per_stage = [2, 2, 2, 2]
    estimator_provider = posterior_nn(
        "mdn",
        embedding_net=ResNetEmbedding2D(
            c_in=num_channels,
            n_stages=n_stages,
            change_c_mode=change_c_mode,
            c_out=20,
            c_stages=c_stages[:n_stages],
            blocks_per_stage=blocks_per_stage[:n_stages],
        ),
    )

    num_dim = input_shape[0]

    def simulator2d(theta):
        x = MultivariateNormal(
            loc=theta, covariance_matrix=0.5 * torch.eye(num_dim)
        ).sample()
        return x.unsqueeze(2).repeat(1, 1, input_shape[1])

    simulator = simulator2d
    xo = torch.ones(1, num_channels, *input_shape).squeeze(1)

    prior = MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))

    num_simulations = 1000
    theta = prior.sample(torch.Size((num_simulations,)))
    x = simulator(theta)
    if num_channels > 1:
        x = x.unsqueeze(1).repeat(
            1, num_channels, *[1 for _ in range(len(input_shape))]
        )

    trainer = NPE(prior=prior, density_estimator=estimator_provider)
    trainer.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = trainer.build_posterior().set_default_x(xo)

    s = posterior.sample((10,))
    posterior.potential(s)


@pytest.mark.parametrize("input_shape", [(2,), (128,)])
@pytest.mark.parametrize("n_blocks", (1, 20))
@pytest.mark.parametrize("c_internal", (2, 20))
@pytest.mark.parametrize("c_hidden_final", (2, 20))
def test_1d_ResNet_fc_embedding_net(input_shape, n_blocks, c_internal, c_hidden_final):
    estimator_provider = posterior_nn(
        "mdn",
        embedding_net=ResNetEmbedding1D(
            c_in=input_shape[0],
            c_out=20,
            n_blocks=n_blocks,
            c_internal=c_internal,
        ),
    )

    num_dim = input_shape[0]

    def simulator1d(theta):
        return torch.rand_like(theta) + theta

    if len(input_shape) == 1:
        simulator = simulator1d
        xo = torch.ones(1, *input_shape)

    prior = MultivariateNormal(torch.zeros(num_dim), torch.eye(num_dim))

    num_simulations = 1000
    theta = prior.sample(torch.Size((num_simulations,)))
    x = simulator(theta)

    trainer = NPE(prior=prior, density_estimator=estimator_provider)
    trainer.append_simulations(theta, x).train(max_num_epochs=2)
    posterior = trainer.build_posterior().set_default_x(xo)

    s = posterior.sample((10,))
    posterior.potential(s)


@pytest.mark.parametrize(
    "bidirectional", [True, False], ids=["one-directional", "bi-directional"]
)
@pytest.mark.parametrize(
    "mode",
    [
        "loop",
        pytest.param(
            "scan",
            marks=pytest.mark.xfail(
                condition=sys.version_info >= (3, 13),
                reason="torch.compiler is not yet supported on Python >= 3.13",
                strict=True,
            ),
        ),
    ],
    ids=["loop", "scan"],
)
def test_lru_isolated(
    bidirectional: bool,
    mode: str,
    input_dim: int = 7,
    state_dim: int = 11,
    r_min: float = 0.1,
    r_max: float = 1.0,
    phase_max: float = 2 * torch.pi,
    batch_size: int = 16,
    sequence_len: int = 50,
):
    """Run some random data trough an LRU layer."""
    lru = LRU(
        input_dim=input_dim,
        state_dim=state_dim,
        r_min=r_min,
        r_max=r_max,
        phase_max=phase_max,
        bidirectional=bidirectional,
        mode=mode,
    )

    x = torch.randn(batch_size, sequence_len, input_dim)

    y = lru(x)
    assert isinstance(y, Tensor)
    assert torch.is_floating_point(y), "Output tensor is not a real tensor"
    assert y.shape == (batch_size, sequence_len, input_dim)


@pytest.mark.parametrize(
    "bidirectional", [True, False], ids=["one-directional", "bi-directional"]
)
@pytest.mark.parametrize(
    "mode",
    [
        "loop",
        pytest.param(
            "scan",
            marks=pytest.mark.xfail(
                condition=sys.version_info >= (3, 13),
                reason="torch.compiler is not yet supported on Python >= 3.13",
                strict=True,
            ),
        ),
    ],
    ids=["loop", "scan"],
)
@pytest.mark.parametrize(
    "apply_input_normalization",
    [True, False],
    ids=["input-normalization", "no-input-normalization"],
)
def test_lru_block_isolated(
    bidirectional: bool,
    mode: str,
    apply_input_normalization: bool,
    hidden_dim: int = 7,
    state_dim: int = 11,
    r_min: float = 0.5,
    r_max: float = 1.0,
    phase_max: float = 2 * torch.pi,
    dropout: float = 0.5,
    batch_size: int = 16,
    sequence_len: int = 50,
):
    """Run some random data through an LRUBlock."""

    lru_block = LRUBlock(
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        r_min=r_min,
        r_max=r_max,
        phase_max=phase_max,
        bidirectional=bidirectional,
        mode=mode,
        dropout=dropout,
        apply_input_normalization=apply_input_normalization,
    )

    x = torch.randn(batch_size, sequence_len, hidden_dim)

    y = lru_block(x)
    assert isinstance(y, Tensor)
    assert torch.is_floating_point(y), "Output tensor is not a real tensor"
    assert y.shape == (batch_size, sequence_len, hidden_dim)


@pytest.mark.parametrize(
    "bidirectional", [True, False], ids=["one-directional", "bi-directional"]
)
@pytest.mark.parametrize(
    "mode",
    [
        "loop",
        pytest.param(
            "scan",
            marks=pytest.mark.xfail(
                condition=sys.version_info >= (3, 13),
                reason="torch.compiler is not yet supported on Python >= 3.13",
                strict=True,
            ),
        ),
    ],
    ids=["loop", "scan"],
)
@pytest.mark.parametrize(
    "aggregate_fcn", ["last_step", "mean"], ids=["last-step", "mean"]
)
def test_lru_embedding_net_isolated(
    bidirectional: bool,
    mode: str,
    aggregate_fcn: str,
    output_dim: int = 5,
    input_dim: int = 7,
    state_dim: int = 11,
    hidden_dim: int = 19,
    num_blocks: int = 2,
    r_min: float = 0.0,
    r_max: float = 1.0,
    phase_max: float = 2 * torch.pi,
    dropout: float = 0.5,
    batch_size: int = 16,
    sequence_len: int = 50,
):
    """Run some random data trough an LRUEmbedding network."""
    embedding_net = LRUEmbedding(
        input_dim=input_dim,  # = observation_dim
        output_dim=output_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        r_min=r_min,
        r_max=r_max,
        phase_max=phase_max,
        bidirectional=bidirectional,
        mode=mode,
        dropout=dropout,
        apply_input_normalization=True,
        aggregate_fcn=aggregate_fcn,
    )

    x = torch.randn(batch_size, sequence_len, input_dim)

    x_embed = embedding_net(x)
    assert isinstance(x_embed, Tensor)
    assert torch.is_floating_point(x_embed), "Output tensor is not a real tensor"
    assert x_embed.shape == (batch_size, output_dim)


def test_lru_pipeline(embedding_feat_dim: int = 17):
    """Smoke-test an entire pipeline run using the LRU embedding."""

    def _simulator(thetas: Tensor, num_time_steps=500, dt=0.002, eps=0.05) -> Tensor:
        """Create a simple simulator for a one-mass dampened spring system."""
        assert thetas.shape[-1] == 2, "Expected 2 parameters: k, d"
        init_state = torch.tensor([[0.2], [0.5]])

        xs = []
        # Create the matrices for the ODE, given the parameters.
        k, d = thetas
        m = 1.0
        omega = torch.sqrt(k / m)  # eigen frequency [Hz]
        zeta = d / (2.0 * torch.sqrt(m * k))  # damping ratio [-]
        A = torch.tensor([[0, 1], [-(omega**2), -2.0 * zeta * omega]])
        B = torch.tensor([[0], [1.0 / m]])

        # Set a fixed initial position and velocity.
        x = init_state.clone()
        u = torch.tensor([[1.3]])

        # Simulate.
        for _ in range(num_time_steps):
            # Compute the ODE's right hand side.
            x_dot = A @ x + B @ u

            # Integrate one step (forward Euler Maruyama).
            x = x + x_dot * dt + eps * math.sqrt(dt) * torch.randn((2, 1))
            xs.append(x.T.clone())

        return torch.cat(xs, dim=0)

    traj = _simulator(torch.tensor([15.0, 0.7]))
    assert traj.shape == (500, 2)

    # Create the embedding.
    embedding_net = LRUEmbedding(input_dim=2, output_dim=embedding_feat_dim)

    # DSt prior distribution for the parameters.
    prior = utils.BoxUniform(
        low=torch.tensor([10.0, 0.5]), high=torch.tensor([20.0, 1.0])
    )

    # Make a SBI-wrapper on the simulator object for compatibility.
    prior, _, prior_returns_numpy = process_prior(prior)
    simulator_wrapper = process_simulator(_simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator_wrapper, prior)

    # Instantiate the neural density estimator.
    neural_posterior = posterior_nn(model="maf", embedding_net=embedding_net)

    # Setup the inference procedure with NPE.
    inferer = NPE(prior=prior, density_estimator=neural_posterior)

    # Run the inference procedure on one round.
    theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=10)
    density_estimator = inferer.append_simulations(theta, x).train(
        training_batch_size=5, max_num_epochs=3
    )
    posterior = inferer.build_posterior(density_estimator)

    # Generate posterior samples.
    true_parameter = torch.tensor([15.0, 0.7])
    x_observed = _simulator(true_parameter)
    samples = posterior.set_default_x(x_observed).sample((10,))

    assert samples.shape == (10, 2)


@pytest.mark.xfail(
    condition=sys.version_info >= (3, 13),
    reason="torch.compiler is not yet supported on Python >= 3.13",
    strict=True,
)
def test_scan(
    input_dim: int = 3,
    output_dim: int = 3,
    state_dim: int = 4,
    hidden_dim: int = 2,
    batch_size: int = 5,
    sequence_len: int = 3,
):
    """Test the scan forward pass of the LRU layer, should be equal to the loop."""
    # causal
    torch.compiler.reset()
    embedding = LRUEmbedding(
        input_dim=input_dim,
        output_dim=output_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_blocks=1,
        bidirectional=False,
    )
    x = torch.randn(batch_size, sequence_len, hidden_dim) * 0.1
    init_state = torch.zeros(batch_size, state_dim)
    y_scan = embedding.lru_blocks[0].lru._forward_scan(x, state=init_state)
    y_loop = embedding.lru_blocks[0].lru._forward_loop(x, state=init_state)
    assert torch.allclose(y_scan, y_loop, atol=1e-5)

    # causal non zero initial state
    torch.compiler.reset()
    embedding = LRUEmbedding(
        input_dim=input_dim,
        output_dim=output_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_blocks=1,
        bidirectional=False,
    )
    x = torch.randn(batch_size, sequence_len, hidden_dim) * 0.1
    init_state = torch.randn(batch_size, state_dim)
    y_scan = embedding.lru_blocks[0].lru._forward_scan(x, state=init_state)
    y_loop = embedding.lru_blocks[0].lru._forward_loop(x, state=init_state)
    assert torch.allclose(y_scan, y_loop, atol=1e-5)

    # bidirectional
    torch.compiler.reset()
    embedding = LRUEmbedding(
        input_dim=input_dim,
        output_dim=output_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_blocks=1,
        bidirectional=True,
    )
    x = torch.randn(batch_size, sequence_len, hidden_dim) * 0.1
    init_state = torch.zeros(batch_size, state_dim * 2)
    y_scan = embedding.lru_blocks[0].lru._forward_scan(x, state=init_state)
    y_loop = embedding.lru_blocks[0].lru._forward_loop(x, state=init_state)
    assert torch.allclose(y_scan, y_loop, atol=1e-5)

    # bidirectional non zero initial state
    torch.compiler.reset()
    embedding = LRUEmbedding(
        input_dim=input_dim,
        output_dim=output_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_blocks=1,
        bidirectional=True,
    )
    x = torch.randn(batch_size, sequence_len, hidden_dim) * 0.1
    init_state = torch.randn(batch_size, state_dim * 2)
    y_scan = embedding.lru_blocks[0].lru._forward_scan(x, state=init_state)
    y_loop = embedding.lru_blocks[0].lru._forward_loop(x, state=init_state)
    assert torch.allclose(y_scan, y_loop, atol=1e-5)

    torch.compiler.reset()
