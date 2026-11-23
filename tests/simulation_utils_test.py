# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import random

import numpy as np
import pytest
import torch
from joblib import parallel_config

from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform, simulate_from_theta
from sbi.utils.torchutils import process_device


def double(theta):
    return theta * 2.0


def numpy_total(theta):
    return np.asarray(theta, dtype=np.float64).sum(-1)


def summary_list(theta):
    return [float(theta.sum()), float(theta.max())]


def numpy_noise(theta):
    return np.asarray(theta) + np.random.randn(*np.shape(theta))


def noise_from_all_backends(theta):
    noise = np.random.randn(*np.shape(theta)) + torch.randn(np.shape(theta)).numpy()
    return np.asarray(theta) + noise + random.random()


def fails_for_negative(theta):
    if (theta < 0).any():
        raise ValueError("negative parameter")
    return theta * 2.0


def batch_sizes(theta):
    return torch.full((len(theta), 1), float(len(theta)))


def storage_ratio(theta):
    """Returns the storage size of `theta` relative to its own size."""
    return torch.full((len(theta), 1), theta.untyped_storage().nbytes() / theta.nbytes)


@pytest.fixture
def prior():
    return BoxUniform(-torch.ones(3), torch.ones(3))


@pytest.mark.parametrize("simulator_is_batched", [True, False])
@pytest.mark.parametrize("num_workers", [1, 2])
def test_simulate_from_theta_matches_simulator(simulator_is_batched, num_workers):
    theta = torch.rand(10, 2)

    x = simulate_from_theta(
        double,
        theta,
        num_workers=num_workers,
        simulation_batch_size=3 if simulator_is_batched else 1,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert torch.allclose(x, theta * 2.0)


@pytest.mark.parametrize(
    "simulator, num_workers, simulator_is_batched, x_shape",
    [
        (lambda theta: theta.double(), 1, True, (20, 3)),
        (numpy_total, 2, True, (20, 1)),
        (summary_list, 2, False, (20, 2)),
    ],
)
def test_simulate_for_sbi_returns_float32_tensors(
    prior, simulator, num_workers, simulator_is_batched, x_shape
):
    theta, x = simulate_for_sbi(
        simulator,
        prior,
        20,
        num_workers=num_workers,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert theta.shape == (20, 3)
    assert isinstance(x, torch.Tensor)
    assert x.shape == x_shape
    assert x.dtype == torch.float32


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu), "numpy"]
)
def test_simulator_gets_theta_type_for_any_num_workers(device):
    """`double` returns its input's type, so `x` shows what the simulator got."""
    theta = np.random.rand(6, 2)
    if device != "numpy":
        theta = torch.as_tensor(
            theta, dtype=torch.float32, device=process_device(device)
        )

    for num_workers in [1, 2]:
        x = simulate_from_theta(
            double, theta, num_workers=num_workers, show_progress_bar=False
        )

        assert type(x) is type(theta)
        assert x.dtype == theta.dtype


@pytest.mark.gpu
def test_simulate_for_sbi_returns_x_on_theta_device():
    device = process_device("gpu")
    prior = BoxUniform(-torch.ones(3, device=device), torch.ones(3, device=device))

    theta, x = simulate_for_sbi(
        numpy_total, prior, 6, num_workers=2, show_progress_bar=False
    )

    assert x.device == theta.device


@pytest.mark.parametrize(
    "simulator",
    [lambda theta: theta * 1j, lambda theta: ["sim.npy"] * len(theta)],
    ids=["complex", "paths"],
)
def test_simulate_for_sbi_rejects_outputs_that_are_not_real_numbers(prior, simulator):
    with pytest.raises(TypeError, match="real numeric"):
        simulate_for_sbi(simulator, prior, 4, show_progress_bar=False)


@pytest.mark.parametrize("simulator_is_batched", [False, True])
def test_seeded_outputs_do_not_depend_on_num_workers(simulator_is_batched):
    x, x_with_workers = [
        simulate_from_theta(
            noise_from_all_backends,
            np.zeros((30, 2)),
            num_workers=num_workers,
            simulator_is_batched=simulator_is_batched,
            seed=0,
            show_progress_bar=False,
        )
        for num_workers in [1, 2]
    ]

    assert len(np.unique(x, axis=0)) == 30
    assert np.array_equal(x, x_with_workers)


def test_simulate_for_sbi_reproducible_with_seed(prior):
    def run(seed, num_workers):
        return simulate_for_sbi(
            numpy_noise,
            prior,
            50,
            num_workers=num_workers,
            seed=seed,
            show_progress_bar=False,
        )

    theta, x = run(torch.tensor(0), num_workers=1)
    after = prior.sample((50,))
    theta_with_workers, x_with_workers = run(0, num_workers=2)

    after_with_workers = prior.sample((50,))

    assert torch.equal(x, x_with_workers)
    assert not torch.equal(x, run(1, num_workers=1)[1])
    # Draws after a seeded call do not repeat it, for any number of workers.
    assert not torch.equal(after, theta)
    assert torch.equal(after, after_with_workers)


def test_unseeded_workers_follow_numpy_state():
    def run():
        np.random.seed(0)
        return simulate_from_theta(
            numpy_noise, np.zeros((30, 2)), num_workers=2, show_progress_bar=False
        )

    x = run()

    assert len(np.unique(x, axis=0)) == 30
    assert np.array_equal(x, run())


@pytest.mark.parametrize(
    "single_theta_simulator",
    [
        lambda theta: np.ravel(theta) * 2.0,
        lambda theta: "sim.npy",
        lambda theta: (theta, theta, theta),
    ],
    ids=["flattened", "one_path", "tuple"],
)
def test_wrongly_batched_simulator_raises(single_theta_simulator):
    with pytest.raises(ValueError, match="simulator_is_batched=False"):
        simulate_from_theta(
            single_theta_simulator,
            torch.rand(6, 2),
            simulation_batch_size=3,
            simulator_is_batched=True,
            show_progress_bar=False,
        )


def test_batch_size_without_a_batched_simulator_raises():
    with pytest.raises(ValueError, match="simulator_is_batched=True"):
        simulate_from_theta(double, torch.rand(6, 2), simulation_batch_size=3)


@pytest.mark.parametrize("simulator_is_batched", [True, False])
def test_simulate_from_theta_returns_paths_in_order(simulator_is_batched):
    theta = torch.arange(5.0).unsqueeze(1)

    def write(theta):
        if simulator_is_batched:
            return [f"sim_{int(t.item())}.npy" for t in theta]
        return f"sim_{int(theta.item())}.npy"

    x = simulate_from_theta(
        write,
        theta,
        num_workers=2,
        simulation_batch_size=2 if simulator_is_batched else 1,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert x == [f"sim_{i}.npy" for i in range(5)]


def test_outputs_of_different_shapes_come_back_as_a_list():
    x = simulate_from_theta(
        lambda theta: torch.zeros(int(theta[0])),
        torch.arange(1.0, 4.0).unsqueeze(1),
        show_progress_bar=False,
    )

    assert [len(row) for row in x] == [1, 2, 3]


@pytest.mark.parametrize("num_workers", [1, 2])
def test_simulator_error_keeps_type_and_names_parameter_set(num_workers):
    theta = torch.ones(4, 2)
    theta[2] = -1.0

    with pytest.raises(ValueError, match="negative parameter") as error:
        simulate_from_theta(
            fails_for_negative, theta, num_workers=num_workers, show_progress_bar=False
        )

    assert "theta[2]" in " ".join(error.value.__notes__)


def test_simulator_without_return_raises():
    with pytest.raises(ValueError, match="miss a `return`"):
        simulate_from_theta(lambda t: None, torch.ones(3, 2), show_progress_bar=False)


@pytest.mark.parametrize(
    "simulation_batch_size, num_workers, call_size",
    [(1, 2, 1.0), (7, 1, 7.0), (None, 1, 35.0)],
)
def test_batched_simulator_gets_simulation_batch_size_parameter_sets(
    prior, simulation_batch_size, num_workers, call_size
):
    _, x = simulate_for_sbi(
        batch_sizes,
        prior,
        35,
        num_workers=num_workers,
        simulation_batch_size=simulation_batch_size,
        show_progress_bar=False,
    )

    assert torch.all(x == call_size)


def test_worker_calls_carry_only_their_own_parameter_sets():
    x = simulate_from_theta(
        storage_ratio,
        torch.rand(40, 2),
        num_workers=2,
        simulation_batch_size=10,
        simulator_is_batched=True,
        show_progress_bar=False,
    )

    assert torch.all(x == 1.0)


def test_workers_can_write_to_large_numpy_parameters():
    def simulator(theta):
        theta -= 1.0
        return theta.sum(1)

    x = simulate_from_theta(
        simulator,
        np.ones((70_000, 2)),  # More than joblib's 1 MB limit for memory maps.
        num_workers=2,
        simulation_batch_size=None,
        simulator_is_batched=True,
        show_progress_bar=False,
    )

    assert not x.any()


@pytest.mark.parametrize("simulator_is_batched", [False, True])
@pytest.mark.parametrize("make_buffer", [np.zeros, torch.zeros], ids=["numpy", "torch"])
def test_simulator_that_reuses_its_output_buffer(simulator_is_batched, make_buffer):
    buffer = make_buffer(4 if simulator_is_batched else 1)

    def simulator(theta):
        rows = theta if simulator_is_batched else theta[None]
        for i, row in enumerate(rows):
            buffer[i] = float(row[0])
        return buffer

    x = simulate_from_theta(
        simulator,
        np.arange(16.0).reshape(-1, 1),
        simulation_batch_size=4 if simulator_is_batched else 1,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert np.array_equal(np.asarray(x).ravel(), np.arange(16.0))


def test_thread_workers_run_without_seed_and_reject_one():
    with parallel_config(backend="threading"):
        x = simulate_from_theta(
            numpy_noise, np.zeros((8, 2)), num_workers=2, show_progress_bar=False
        )
        with pytest.raises(ValueError, match="process workers"):
            simulate_from_theta(
                double, np.zeros((8, 2)), num_workers=2, seed=0, show_progress_bar=False
            )

    assert len(np.unique(x, axis=0)) == 8
