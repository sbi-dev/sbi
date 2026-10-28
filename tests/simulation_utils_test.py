# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import random

import numpy as np
import pytest
import torch
from joblib import parallel_config

from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform, parallelize_simulator, simulate_from_theta
from sbi.utils.torchutils import process_device


def double(theta):
    return theta * 2.0


def torch_double(theta):
    return theta.double()


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


class SolverError(Exception):
    """An exception that cannot be unpickled: `args` holds one message."""

    def __init__(self, code, detail):
        super().__init__(f"code {code}: {detail}")


def fails_with_solver_error(theta):
    if (theta < 0).any():
        raise SolverError(3, "diverged")
    return theta * 2.0


def batch_sizes(theta):
    return torch.full((len(theta), 1), float(len(theta)))


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
        simulation_batch_size=3,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert torch.allclose(x, theta * 2.0)


@pytest.mark.parametrize(
    "simulator, num_workers, simulator_is_batched, x_shape",
    [
        (torch_double, 1, True, (20, 3)),
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
        if device != "numpy":
            assert x.device == theta.device

    x = simulate_from_theta(double, theta[:0], show_progress_bar=False)

    assert type(x) is type(theta)
    if device != "numpy":
        assert x.device == theta.device


@pytest.mark.gpu
def test_simulate_for_sbi_returns_numpy_outputs_on_theta_device():
    device = process_device("gpu")
    prior = BoxUniform(-torch.ones(3, device=device), torch.ones(3, device=device))

    theta, x = simulate_for_sbi(
        numpy_total, prior, 6, num_workers=2, show_progress_bar=False
    )

    assert x.device == theta.device


@pytest.mark.gpu
@pytest.mark.parametrize("num_workers", [1, 2])
def test_ragged_tensor_outputs_stay_on_theta_device(num_workers):
    theta = torch.arange(4.0, device=process_device("gpu")).unsqueeze(1)

    x = simulate_from_theta(
        lambda t: torch.zeros(int(t[0]) + 1, device=t.device),
        theta,
        num_workers=num_workers,
        show_progress_bar=False,
    )

    assert [row.shape[0] for row in x] == [1, 2, 3, 4]
    assert {row.device for row in x} == {theta.device}


def test_simulate_for_sbi_reproducible_with_seed(prior):
    def run(seed):
        return simulate_for_sbi(
            numpy_noise,
            prior,
            50,
            num_workers=2,
            seed=seed,
            show_progress_bar=False,
        )[1]

    assert torch.equal(run(seed=0), run(seed=0))
    assert not torch.equal(run(seed=0), run(seed=1))


@pytest.mark.parametrize(
    "simulator_is_batched, runs",
    [
        (False, [(1, 1), (2, 7)]),
        (True, [(1, 7), (2, 7)]),
    ],
)
def test_seeded_outputs_do_not_depend_on_workers_or_batch_size(
    simulator_is_batched, runs
):
    """`runs` are (num_workers, simulation_batch_size) pairs."""
    x, *others = [
        simulate_from_theta(
            noise_from_all_backends,
            np.zeros((30, 2)),
            num_workers=num_workers,
            simulation_batch_size=simulation_batch_size,
            simulator_is_batched=simulator_is_batched,
            seed=0,
            show_progress_bar=False,
        )
        for num_workers, simulation_batch_size in runs
    ]

    assert len(np.unique(x, axis=0)) == 30
    for other in others:
        assert np.array_equal(other, x)


def test_seed_keeps_callers_random_state():
    np.random.seed(1)
    torch.manual_seed(1)
    expected = np.random.rand(), torch.rand(1)
    np.random.seed(1)
    torch.manual_seed(1)

    simulate_from_theta(
        noise_from_all_backends,
        np.zeros((5, 2)),
        seed=0,
        show_progress_bar=False,
    )

    assert (np.random.rand(), torch.rand(1)) == expected


def test_seed_keeps_callers_queued_cuda_seed():
    queue = getattr(torch.cuda, "_lazy_seed_tracker", None)
    if queue is None or torch.cuda.is_initialized():
        pytest.skip("Only before CUDA starts does torch queue CUDA seeds.")
    torch.manual_seed(1)
    expected = queue.get_calls()

    simulate_from_theta(double, np.zeros((2, 2)), seed=0, show_progress_bar=False)

    assert queue.get_calls() == expected


@pytest.mark.gpu
def test_seed_keeps_callers_accelerator_random_state():
    device = process_device("gpu")
    torch.manual_seed(1)
    expected = torch.rand(1, device=device)
    torch.manual_seed(1)

    simulate_from_theta(double, np.zeros((5, 2)), seed=0, show_progress_bar=False)

    assert torch.equal(torch.rand(1, device=device), expected)


@pytest.mark.parametrize(
    "single_theta_simulator",
    [lambda theta: np.ravel(theta) * 2.0, lambda theta: "sim.npy"],
    ids=["flattened", "one_path"],
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
        simulation_batch_size=2,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert x == [f"sim_{i}.npy" for i in range(5)]


@pytest.mark.parametrize("num_workers, simulator_is_batched", [(2, False), (1, True)])
def test_on_error_nan_keeps_finished_simulations(num_workers, simulator_is_batched):
    theta = torch.ones(8, 2)
    theta[3] = -1.0
    failed = torch.zeros(8, dtype=torch.bool)
    failed[[2, 3] if simulator_is_batched else [3]] = True

    with pytest.warns(UserWarning, match="simulations failed") as record:
        x = simulate_from_theta(
            fails_for_negative,
            theta,
            num_workers=num_workers,
            simulation_batch_size=2,
            simulator_is_batched=simulator_is_batched,
            on_error="nan",
            show_progress_bar=False,
        )

    assert len(record) == 1
    assert f"{int(failed.sum())} of 8 simulations failed" in str(record[0].message)
    assert torch.isnan(x[failed]).all()
    assert torch.equal(x[~failed], theta[~failed] * 2.0)


@pytest.mark.parametrize("on_error, num_workers", [("raise", 2), ("nan", 1)])
def test_simulator_error_keeps_type_and_names_parameter_set(on_error, num_workers):
    theta = -torch.ones(4, 2) if on_error == "nan" else torch.ones(4, 2)
    theta[2] = -1.0

    with pytest.raises(ValueError, match="negative parameter") as error:
        simulate_from_theta(
            fails_for_negative,
            theta,
            num_workers=num_workers,
            on_error=on_error,
            show_progress_bar=False,
        )

    assert "theta[" in " ".join(error.value.__notes__)


def test_auto_batch_size_splits_simulations_across_workers():
    x = simulate_from_theta(
        batch_sizes,
        np.zeros((40, 1)),
        num_workers=2,
        simulator_is_batched=True,
        show_progress_bar=False,
    )

    assert torch.all(x == 40 / (4 * 2))


def test_auto_batch_size_is_one_on_one_worker(prior):
    _, x = simulate_for_sbi(batch_sizes, prior, 10, show_progress_bar=False)

    assert torch.all(x == 1.0)


def test_unknown_on_error_raises():
    with pytest.raises(ValueError, match="on_error"):
        simulate_from_theta(double, torch.ones(4, 2), on_error="ignore")


def test_parallelize_simulator_forwards_settings():
    @parallelize_simulator(
        num_workers=2,
        simulation_batch_size=3,
        simulator_is_batched=True,
        on_error="nan",
    )
    def simulator(theta):
        if (theta < 0).any():
            raise ValueError("negative parameter")
        return np.full((len(theta), 1), float(len(theta)))

    theta = np.ones((10, 1))
    theta[9] = -1.0
    with pytest.warns(UserWarning, match="1 of 10 simulations failed"):
        x = simulator(theta)

    assert np.array_equal(x[:9], np.full((9, 1), 3.0))
    assert np.isnan(x[9]).all()


@pytest.mark.parametrize("simulator_is_batched", [False, True])
@pytest.mark.parametrize(
    "make_buffer",
    [np.zeros, torch.zeros, lambda size: [0.0] * size],
    ids=["numpy", "torch", "list"],
)
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
        simulation_batch_size=4,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert np.array_equal(np.asarray(x).ravel(), np.arange(16.0))


def test_unpicklable_simulator_error_keeps_parallel_results():
    theta = torch.ones(8, 2)
    theta[3] = -1.0

    with pytest.warns(UserWarning, match="1 of 8 simulations failed"):
        x = simulate_from_theta(
            fails_with_solver_error,
            theta,
            num_workers=2,
            on_error="nan",
            show_progress_bar=False,
        )

    assert torch.isnan(x[3]).all()
    assert torch.equal(x[torch.arange(8) != 3], theta[torch.arange(8) != 3] * 2.0)


@pytest.mark.parametrize("container", [list, tuple])
def test_simulator_that_reuses_an_array_inside_a_list(container):
    array = np.zeros(1)

    def simulator(theta):
        array[0] = float(theta[0])
        return container([array])

    x = simulate_from_theta(
        simulator,
        np.arange(4.0).reshape(-1, 1),
        simulation_batch_size=4,
        show_progress_bar=False,
    )

    assert np.array_equal(np.asarray(x).ravel(), np.arange(4.0))


def test_failed_simulations_keep_integer_outputs_exact():
    def simulator(theta):
        if theta[0] < 0:
            raise ValueError("negative parameter")
        return torch.tensor([2**24 + 1])

    with pytest.warns(UserWarning, match="1 of 2 simulations failed"):
        x = simulate_from_theta(
            simulator,
            torch.tensor([[1.0], [-1.0]]),
            on_error="nan",
            show_progress_bar=False,
        )

    assert x[0].item() == 2**24 + 1
    assert torch.isnan(x[1]).all()


def test_thread_workers_run_without_seed_and_reject_one():
    with parallel_config(backend="threading"):
        x = simulate_from_theta(
            numpy_noise, np.zeros((8, 2)), num_workers=2, show_progress_bar=False
        )
        with pytest.raises(ValueError, match="process workers"):
            simulate_from_theta(
                double, np.zeros((8, 2)), num_workers=2, seed=0, show_progress_bar=False
            )
        with pytest.raises(ValueError, match="process workers"):
            simulate_for_sbi(
                double,
                BoxUniform(-torch.ones(2), torch.ones(2)),
                8,
                num_workers=2,
                seed=0,
            )

    assert len(np.unique(x, axis=0)) == 8
