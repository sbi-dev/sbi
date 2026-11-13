# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import re
import warnings

import pytest
import torch

import sbi.inference
from sbi import utils
from sbi.inference import FMPE, NPE, infer
from sbi.inference.trainers import nle, npe, nre
from sbi.utils.removed_names import REMOVED_MODULES, REMOVED_NAMES, RENAMED_METHODS


def test_infer():
    # Example is taken from 00_getting_started.ipynb
    num_dim = 3
    prior = utils.BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

    def simulator(parameter_set):
        return 1.0 + parameter_set + torch.randn(parameter_set.shape) * 0.1

    posterior = infer(simulator, prior, method="NPE_A", num_simulations=10)
    assert posterior is not None, "Most basic use of 'infer' failed"
    posterior = infer(
        simulator,
        prior,
        method="NPE_A",
        num_simulations=10,
        init_kwargs={"num_components": 5},
        train_kwargs={"max_num_epochs": 2},
        build_posterior_kwargs={"prior": prior},
    )
    assert posterior is not None, "Using 'infer' with keyword arguments failed"


@pytest.mark.parametrize("training_batch_size", (1, 10, 100))
def test_get_dataloaders(training_batch_size):
    N = 1000
    validation_fraction = 0.1

    inferer = NPE()
    inferer.append_simulations(torch.ones(N), torch.zeros(N))
    _, val_loader = inferer.get_dataloaders(
        0,
        training_batch_size=training_batch_size,
        validation_fraction=validation_fraction,
    )

    assert len(val_loader) * val_loader.batch_size == int(validation_fraction * N)


@pytest.mark.parametrize("name", ("NPE", "NLE", "NRE"))
def test_shorthands_agree_across_import_paths(name):
    """`sbi.inference` and the trainer sub-package must resolve to the same class."""
    expected = getattr(sbi.inference, name)
    defining_modules = [m for m in (npe, nle, nre) if hasattr(m, name)]

    assert defining_modules, (
        f"No trainer sub-package defines {name}. Either the name moved or it "
        f"should not be parametrized here, which would make this test vacuous."
    )

    for module in defining_modules:
        assert getattr(module, name) is expected, (
            f"{module.__name__}.{name} disagrees with sbi.inference.{name}"
        )


def test_renamed_method_set_is_complete():
    """Pin the renamed names literally.

    The tests below parametrize over the tables, so dropping an entry would shrink
    their collection rather than fail them, and the name would fall back to a bare
    "cannot import name" for users with a green suite.
    """
    assert set(RENAMED_METHODS) == {
        "SNL",
        "SNLE",
        "SNLE_A",
        "SNPE_A",
        "SNPE_B",
        "SNPE",
        "SNPE_C",
        "APT",
        "SRE",
        "SNRE",
        "SNRE_B",
        "AALR",
        "SNRE_A",
        "CNRE",
        "SNRE_C",
        "ABC",
        "SMC",
    }


@pytest.mark.parametrize(("name", "replacement"), REMOVED_NAMES.items())
def test_removed_names_name_their_replacement(name, replacement):
    # The replacement must run as written.
    exec(replacement, {"theta": torch.randn(10, 2), "x": torch.randn(10, 2)})
    with pytest.raises(ImportError, match=re.escape(replacement)):
        exec(f"from sbi.inference import {name}", {})


@pytest.mark.parametrize(
    "package",
    (
        "sbi.utils",
        "sbi.inference.trainers.npe",
        "sbi.inference.trainers.nle",
        "sbi.inference.trainers.nre",
        "sbi.neural_nets.net_builders",
    ),
)
def test_every_package_that_exported_removed_names_raises(package):
    """Each package that exported a removed name needs its own `__getattr__`."""
    with pytest.raises(ImportError, match=r"from sbi\.inference import NPE`"):
        exec(f"from {package} import SNPE", {})


@pytest.mark.parametrize(("module", "replacement"), REMOVED_MODULES.items())
def test_removed_modules_name_their_replacement(module, replacement):
    """Every import form keeps the message.

    `from package import module` would drop it if the placeholder raised a
    `ModuleNotFoundError`.
    """
    exec(replacement, {})  # The replacement must exist.
    package, name = module.rsplit(".", 1)
    for statement in (
        f"import {module}",
        f"from {module} import anything",
        f"from {package} import {name}",
    ):
        with pytest.raises(ImportError, match=re.escape(replacement)):
            exec(statement, {})


def test_unknown_names_still_raise_attribute_error():
    """Only removed names raise `ImportError`, so `hasattr()` keeps working."""
    assert not hasattr(sbi.inference, "not_an_sbi_name")
    assert not hasattr(utils, "not_an_sbi_name")


def test_infer_rejects_removed_method_names():
    prior = utils.BoxUniform(-torch.ones(2), torch.ones(2))
    with pytest.raises(ValueError, match=r"Use method='npe' instead"):
        infer(lambda theta: theta, prior, method="snpe", num_simulations=10)


@pytest.mark.parametrize("method", ("abc", "smcabc"))
def test_infer_rejects_abc_methods(method):
    """`infer` has never been able to run ABC, so it must say so.

    The ABC classes take the simulator at construction and have no
    `append_simulations`/`train`, so every spelling used to end in the same opaque
    `TypeError` about a missing positional argument. One legacy and one canonical
    spelling, since they take different paths to the check.
    """
    prior = utils.BoxUniform(-torch.ones(2), torch.ones(2))
    target = "SMCABC" if "smc" in method else "MCABC"

    with warnings.catch_warnings():
        # Must not advise a spelling `infer` cannot run either.
        warnings.simplefilter("error", FutureWarning)
        # Word boundary: `MCABC` is a substring of `SMCABC`.
        with pytest.raises(ValueError, match=rf"does not support.*\b{target}\b"):
            infer(
                lambda theta: theta,
                prior,
                method=method,
                num_simulations=10,
            )


def test_canonical_shorthands_do_not_warn():
    """`NPE`, `NLE` and `NRE` are the recommended names: exported and silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert sbi.inference.NPE is sbi.inference.NPE_C
        assert sbi.inference.NLE is sbi.inference.NLE_A
        assert sbi.inference.NRE is sbi.inference.NRE_B
    assert {"NPE", "NLE", "NRE"} <= set(sbi.inference.__all__)


@pytest.mark.parametrize(
    "losses",
    (
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        [10.0, 9.0, 8.0, 7.0, 6.0, 6.0],
    ),
    ids=("best-is-the-last-epoch", "best-is-an-earlier-epoch"),
)
def test_exhausted_epoch_budget_returns_the_best_weights(losses):
    """A run that hits `max_num_epochs` must end on the weights of its best epoch.

    Both shapes matter. The loop condition short-circuits, so `_converged` never
    scores the final epoch: a run still improving at the budget must not roll back
    to an earlier, worse checkpoint, and a run that got worse must roll back.
    """
    prior = utils.BoxUniform(-torch.ones(2), torch.ones(2))
    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)

    inference = NPE(prior=prior, show_progress_bars=False)
    inference.append_simulations(theta, x)

    scripted = iter(losses)
    inference._validate_epoch = lambda *_a, **_kw: next(scripted, losses[-1])
    # `stop_after_epochs` is large so the run cannot converge early.
    inference.train(max_num_epochs=5, stop_after_epochs=50, training_batch_size=50)

    assert inference.epoch > 5, "the budget must have run out for this to test anything"
    assert inference._best_val_loss == min(losses)
    final = inference._neural_net.state_dict()
    assert all(
        torch.equal(final[k], v) for k, v in inference._best_model_state_dict.items()
    )


@pytest.mark.parametrize(
    ("losses", "max_num_epochs", "converged"),
    (
        # Flat after the first epoch, so the run converges long before the budget.
        ([1.0], 20, True),
        # Improving every epoch, so only the budget can stop it.
        ([10.0, 9.0, 8.0, 7.0], 3, False),
    ),
    ids=("converged", "max-num-epochs-reached"),
)
def test_summary_records_why_training_stopped(losses, max_num_epochs, converged):
    """`summary["converged"]` says whether early stopping or the budget ended a run."""
    prior = utils.BoxUniform(-torch.ones(2), torch.ones(2))
    theta = prior.sample((200,))
    x = theta + 0.1 * torch.randn_like(theta)

    inference = NPE(prior=prior, show_progress_bars=False)
    inference.append_simulations(theta, x)

    scripted = iter(losses)
    inference._validate_epoch = lambda *_a, **_kw: next(scripted, losses[-1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        inference.train(
            max_num_epochs=max_num_epochs,
            stop_after_epochs=2 if converged else 50,
            training_batch_size=50,
        )

    assert inference.summary["converged"] == [converged]
    budget_warned = any("max_num_epochs" in str(w.message) for w in caught)
    assert budget_warned is not converged


@pytest.mark.parametrize(
    "train_kwargs",
    (
        {"max_num_epochs": 2},  # Stops at the budget.
        {"stop_after_epochs": 1},  # Converges.
    ),
)
def test_training_output_ends_with_a_newline(monkeypatch, capsys, train_kwargs):
    """Output after `train()` must not continue the epoch counter's line."""
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    prior = utils.BoxUniform(-torch.ones(2), torch.ones(2))
    theta = prior.sample((100,))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # The epoch budget warning.
        NPE(prior=prior).append_simulations(theta, theta).train(**train_kwargs)

    out = capsys.readouterr().out
    assert "Epochs trained" in out
    assert out.endswith("\n")


def test_converged_message_is_plain_when_captured(monkeypatch, capsys):
    """Captured output must not start with the `\\r` that overwrites a progress line."""
    monkeypatch.setenv("TQDM_DISABLE", "1")
    prior = utils.BoxUniform(-torch.ones(2), torch.ones(2))

    NPE(prior=prior)._report_convergence_at_end(epoch=3, max_num_epochs=10)

    expected = "Neural network successfully converged after 3 epochs.\n"
    assert capsys.readouterr().out == expected


def test_vector_field_converged_resets_between_runs():
    """A stale best-val-loss from run one must not leak into run two.

    Otherwise run two can converge immediately and restore run one's weights.
    """
    prior = utils.BoxUniform(-torch.ones(2), torch.ones(2))
    theta = prior.sample((100,))
    x = theta + 0.1 * torch.randn_like(theta)

    inference = FMPE(prior=prior, show_progress_bars=False)
    inference.append_simulations(theta, x)
    inference.train(max_num_epochs=1)

    stale_best = inference._best_val_loss
    inference._val_loss = stale_best + 10.0
    inference._converged(epoch=0, stop_after_epochs=20)

    assert inference._best_val_loss == pytest.approx(stale_best + 10.0)
