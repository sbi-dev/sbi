# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import random
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform, parallelize_simulator, simulate_from_theta
from sbi.utils.torchutils import process_device


def double(theta):
    return theta * 2.0


def total(theta):
    return theta.sum(-1)


def torch_gaussian(theta):
    return theta + torch.randn_like(theta)


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


@pytest.fixture
def prior():
    return BoxUniform(-torch.ones(3), torch.ones(3))


@pytest.mark.parametrize("simulator", [double, total])
@pytest.mark.parametrize("simulator_is_batched", [True, False])
@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("simulation_batch_size", [1, 3])
def test_simulate_from_theta_matches_simulator(
    simulator, simulator_is_batched, num_workers, simulation_batch_size
):
    theta = torch.rand(10, 2)

    x = simulate_from_theta(
        simulator,
        theta,
        num_workers=num_workers,
        simulation_batch_size=simulation_batch_size,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert np.allclose(np.asarray(x), simulator(theta).numpy())


@pytest.mark.parametrize(
    "simulator, num_workers, simulator_is_batched, x_shape",
    [
        (torch_gaussian, 1, True, (20, 3)),
        (torch_double, 1, True, (20, 3)),
        (numpy_total, 1, True, (20,)),
        (numpy_total, 2, True, (20,)),
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


@pytest.mark.parametrize("num_workers", [1, 2])
def test_simulate_for_sbi_reproducible_with_seed(prior, num_workers):
    def run(seed):
        return simulate_for_sbi(
            numpy_noise,
            prior,
            50,
            num_workers=num_workers,
            seed=seed,
            show_progress_bar=False,
        )[1]

    assert torch.equal(run(seed=0), run(seed=0))
    assert not torch.equal(run(seed=0), run(seed=1))


@pytest.mark.parametrize(
    "simulator_is_batched, runs",
    [
        (False, [(1, 1), (1, 7), (2, 1), (4, 7)]),
        (True, [(1, 7), (2, 7), (4, 7)]),
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


@pytest.mark.parametrize("simulator_is_batched", [False, True])
def test_seed_keeps_callers_random_state(simulator_is_batched):
    np.random.seed(1)
    torch.manual_seed(1)
    expected = np.random.rand(), torch.rand(1)
    np.random.seed(1)
    torch.manual_seed(1)

    simulate_from_theta(
        noise_from_all_backends,
        np.zeros((5, 2)),
        simulator_is_batched=simulator_is_batched,
        seed=0,
        show_progress_bar=False,
    )

    assert (np.random.rand(), torch.rand(1)) == expected


@pytest.mark.gpu
def test_seed_keeps_callers_accelerator_random_state():
    device = process_device("gpu")
    torch.manual_seed(1)
    expected = torch.rand(1, device=device)
    torch.manual_seed(1)

    simulate_from_theta(double, np.zeros((5, 2)), seed=0, show_progress_bar=False)

    assert torch.equal(torch.rand(1, device=device), expected)


def test_parallel_batches_draw_distinct_noise():
    num_simulations = 5000

    x = simulate_from_theta(
        lambda theta: np.random.rand(len(theta), 2),
        np.zeros((num_simulations, 1)),
        num_workers=2,
        simulation_batch_size=1,
        simulator_is_batched=True,
        seed=0,
        show_progress_bar=False,
    )

    assert len(np.unique(x, axis=0)) == num_simulations


@pytest.mark.parametrize(
    "single_theta_simulator",
    [lambda theta: np.ravel(theta) * 2.0, lambda theta: "sim.npy"],
    ids=["flattened", "one_path"],
)
@pytest.mark.parametrize("simulation_batch_size", [1, 3])
def test_wrongly_batched_simulator_raises(
    single_theta_simulator, simulation_batch_size
):
    with pytest.raises(ValueError, match="simulator_is_batched=False"):
        simulate_from_theta(
            single_theta_simulator,
            torch.rand(6, 2),
            simulation_batch_size=simulation_batch_size,
            simulator_is_batched=True,
            show_progress_bar=False,
        )


@pytest.mark.parametrize("simulator_is_batched", [True, False])
@pytest.mark.parametrize("path_type", [str, Path])
def test_simulate_from_theta_returns_paths_in_order(simulator_is_batched, path_type):
    theta = torch.arange(5.0).unsqueeze(1)

    def write(theta):
        if simulator_is_batched:
            return [path_type(f"sim_{int(t.item())}.npy") for t in theta]
        return path_type(f"sim_{int(theta.item())}.npy")

    x = simulate_from_theta(
        write,
        theta,
        simulation_batch_size=2,
        simulator_is_batched=simulator_is_batched,
        show_progress_bar=False,
    )

    assert x == [path_type(f"sim_{i}.npy") for i in range(5)]


def test_parallel_error_cancels_remaining_simulations(tmp_path):
    num_simulations = 20

    def simulator(theta):
        time.sleep(0.05)
        (tmp_path / str(time.time_ns())).touch()
        return np.zeros((2 if theta[0, 0] == 0 else 1, 1))

    with pytest.raises(ValueError, match="simulator_is_batched") as error:
        simulate_from_theta(
            simulator,
            np.arange(float(num_simulations)).reshape(-1, 1),
            num_workers=2,
            simulation_batch_size=1,
            simulator_is_batched=True,
            show_progress_bar=False,
        )
    time.sleep(1.0)  # `error` keeps the failed call's frames alive meanwhile.

    assert error.value is not None
    assert len(list(tmp_path.iterdir())) < num_simulations // 2


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("simulator_is_batched", [False, True])
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


@pytest.mark.parametrize("on_error", ["raise", "nan"])
@pytest.mark.parametrize("num_workers", [1, 2])
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


@pytest.mark.parametrize(
    "setting",
    [{"num_workers": 0}, {"simulation_batch_size": 0}, {"on_error": "ignore"}],
    ids=["num_workers", "simulation_batch_size", "on_error"],
)
def test_invalid_settings_raise(setting):
    with pytest.raises(ValueError, match=next(iter(setting))):
        simulate_from_theta(double, torch.ones(4, 2), **setting)


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


@pytest.mark.parametrize("on_error", ["raise", "nan"])
def test_unpicklable_simulator_error_keeps_parallel_results(on_error):
    theta = torch.ones(8, 2)
    theta[3] = -1.0

    if on_error == "raise":
        with pytest.raises(RuntimeError, match="SolverError: code 3: diverged"):
            simulate_from_theta(
                fails_with_solver_error, theta, num_workers=2, show_progress_bar=False
            )
        return
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
