# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings

import pytest
import torch

import sbi.inference
from sbi import utils
from sbi.inference import NPE, VectorFieldPosterior, infer
from sbi.inference.trainers import nle, npe, nre
from sbi.neural_nets.net_builders import (
    build_flow_matching_estimator,
    build_score_matching_estimator,
    build_vector_field_estimator,
)


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


# Spelled out rather than read from `_DEPRECATED_ALIASES`, so a wrong entry there fails.
LEGACY_ALIASES = {
    "SNL": "NLE_A",
    "SNLE": "NLE_A",
    "SNLE_A": "NLE_A",
    "SNPE_A": "NPE_A",
    "SNPE_B": "NPE_B",
    "SNPE": "NPE_C",
    "SNPE_C": "NPE_C",
    "APT": "NPE_C",
    "SRE": "NRE_B",
    "SNRE": "NRE_B",
    "SNRE_B": "NRE_B",
    "AALR": "NRE_A",
    "SNRE_A": "NRE_A",
    "CNRE": "NRE_C",
    "SNRE_C": "NRE_C",
    "ABC": "MCABC",
    "SMC": "SMCABC",
}


# The aliases warn by design; this test is about cross-path agreement, which
# `test_deprecated_aliases_warn` does not cover.
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "legacy_name",
    (
        "SNPE_A",
        "SNPE_B",
        "SNPE_C",
        "SNPE",
        "NPE",
        "SNLE_A",
        "SNLE",
        "NLE",
        "SNRE_A",
        "SNRE_B",
        "SNRE_C",
        "SRE",
        "NRE",
    ),
)
def test_legacy_aliases_agree_across_import_paths(legacy_name):
    """Deprecated aliases must resolve to the same class from either import path.

    Only aliases that both import paths define are parametrized here. `SNL` and `SNRE`
    exist on `sbi.inference` alone, so there is nothing to cross-check for them.
    """
    expected = getattr(sbi.inference, legacy_name)
    defining_modules = [m for m in (npe, nle, nre) if hasattr(m, legacy_name)]

    assert defining_modules, (
        f"No trainer sub-package defines {legacy_name}. Either the alias moved or it "
        f"should not be parametrized here, which would make this test vacuous."
    )

    for module in defining_modules:
        assert getattr(module, legacy_name) is expected, (
            f"{module.__name__}.{legacy_name} disagrees with "
            f"sbi.inference.{legacy_name}"
        )


def test_infer_warns_on_legacy_method_string():
    """The legacy method strings warn once, attributed to the caller."""
    prior = utils.BoxUniform(-torch.ones(2), torch.ones(2))

    def simulator(theta):
        return theta + 0.1 * torch.randn_like(theta)

    with pytest.warns(FutureWarning, match="npe_c") as record:
        infer(
            simulator,
            prior,
            method="snpe",
            num_simulations=10,
            train_kwargs={"max_num_epochs": 1},
        )

    future_warnings = [w for w in record if w.category is FutureWarning]
    assert len(future_warnings) == 1
    assert __file__ == future_warnings[0].filename


@pytest.mark.parametrize(("alias", "canonical"), LEGACY_ALIASES.items())
def test_deprecated_aliases_warn(alias, canonical):
    """Every legacy alias must emit a FutureWarning that names its replacement.

    Matches the replacement clause, not the bare name: many aliases contain their
    canonical name as a substring (`"NPE_A"` in `"SNPE_A"`).
    """
    with pytest.warns(
        FutureWarning, match=rf"Use `sbi\.inference\.{canonical}` instead"
    ):
        resolved = getattr(sbi.inference, alias)

    assert resolved is getattr(sbi.inference, canonical), (
        f"sbi.inference.{alias} does not resolve to {canonical}"
    )


def test_deprecated_alias_set_is_complete():
    """An alias missing from the warn-list would break in v0.28.0 with no warning."""
    assert sbi.inference._DEPRECATED_ALIASES == LEGACY_ALIASES


@pytest.mark.parametrize("module", (npe, nle, nre))
def test_submodule_aliases_warn_and_agree(module):
    """The trainer sub-packages warn too, and resolve to the same classes."""
    for alias, canonical in module._DEPRECATED_ALIASES.items():
        assert LEGACY_ALIASES[alias] == canonical, (
            f"{module.__name__}.{alias} maps to {canonical}, expected "
            f"{LEGACY_ALIASES[alias]}"
        )
        with pytest.warns(FutureWarning, match=rf"\.{canonical}` instead"):
            resolved = getattr(module, alias)
        assert resolved is getattr(module, canonical)


def test_canonical_shorthands_do_not_warn():
    """`NPE`, `NLE` and `NRE` stay silent: their fate is a separate naming decision."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert sbi.inference.NPE is sbi.inference.NPE_C
        assert sbi.inference.NLE is sbi.inference.NLE_A
        assert sbi.inference.NRE is sbi.inference.NRE_B


def test_deprecated_vector_field_builders_warn():
    """The flow and score wrapper builders warn and still delegate correctly."""
    theta, x = torch.randn(20, 2), torch.randn(20, 2)

    with pytest.warns(FutureWarning, match="build_vector_field_estimator"):
        flow = build_flow_matching_estimator(theta, x)
    assert type(flow) is type(build_vector_field_estimator(theta, x, "flow"))

    with pytest.warns(FutureWarning, match="build_vector_field_estimator"):
        score = build_score_matching_estimator(theta, x)
    assert type(score) is type(build_vector_field_estimator(theta, x, "score"))


def test_vector_field_posterior_sample_with_warns():
    """Passing `sample_with` to `sample()` warns; setting it at construction is the
    supported path and must stay silent."""
    theta, x = torch.randn(20, 2), torch.randn(20, 2)
    posterior = VectorFieldPosterior(
        vector_field_estimator=build_vector_field_estimator(theta, x, "flow"),
        prior=utils.BoxUniform(-torch.ones(2), torch.ones(2)),
        sample_with="ode",
    )

    with pytest.warns(FutureWarning, match="sample_with"):
        posterior.sample(
            (1,),
            x=torch.zeros(1, 2),
            sample_with="ode",
            reject_outside_prior=False,
            show_progress_bars=False,
        )
