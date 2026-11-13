# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Names and modules that sbi removed, and the imports that replace them.

Using one of them raises an `ImportError` that names the replacement, so that code
written for an older sbi fails with the fix instead of a bare "cannot import name".

To remove a public name, delete it and add it to `REMOVED_NAMES`. If no package
that exported it has a `__getattr__` from `removed_names_getattr` yet, add one. To
remove a module, add it to `REMOVED_MODULES` and leave a placeholder file at its
path that calls `raise_removed_module(__name__)`.
"""

from typing import Callable, NoReturn

# Old trainer and ABC names. `infer(method=...)` accepted them as well. Each maps to
# the shortest current name of the same class, e.g. `NPE` rather than `NPE_C`.
RENAMED_METHODS = {
    "SNL": "NLE",
    "SNLE": "NLE",
    "SNLE_A": "NLE",
    "SNPE_A": "NPE_A",
    "SNPE_B": "NPE_B",
    "SNPE": "NPE",
    "SNPE_C": "NPE",
    "APT": "NPE",
    "AALR": "NRE_A",
    "SNRE_A": "NRE_A",
    "SRE": "NRE",
    "SNRE": "NRE",
    "SNRE_B": "NRE",
    "CNRE": "NRE_C",
    "SNRE_C": "NRE_C",
    "ABC": "MCABC",
    "SMC": "SMCABC",
}

REMOVED_NAMES = {
    **{old: f"from sbi.inference import {new}" for old, new in RENAMED_METHODS.items()},
    "prepare_for_sbi": "from sbi.utils import process_prior, process_simulator",
    "posterior_nn": "from sbi.neural_nets import posterior_nn",
    "likelihood_nn": "from sbi.neural_nets import likelihood_nn",
    "classifier_nn": "from sbi.neural_nets import classifier_nn",
    **{
        old: f"from sbi.neural_nets import {new}; {new}().build(theta, x)"
        for old, new in (
            ("build_flow_matching_estimator", "FlowMatchingConfig"),
            ("build_score_matching_estimator", "VEScoreConfig"),
            ("build_vector_field_estimator", "FlowMatchingConfig"),
        )
    },
}

REMOVED_MODULES = {
    "sbi.inference.base": "from sbi.inference import infer",
    "sbi.inference.snpe": "from sbi.inference import NPE",
    "sbi.inference.snle": "from sbi.inference import NLE",
    "sbi.inference.snre": "from sbi.inference import NRE",
    "sbi.utils.posterior_ensemble": "from sbi.inference import EnsemblePosterior",
}

# Removed names and modules of the sequential methods. Only the name is gone: the
# trainer they map to runs the same multi-round inference.
SEQUENTIAL = {
    **{
        old: RENAMED_METHODS[old]
        for old in (
            *("SNL", "SNLE", "SNLE_A", "SNPE", "SNPE_A", "SNPE_B", "SNPE_C", "APT"),
            *("SRE", "SNRE", "SNRE_A", "SNRE_B", "SNRE_C"),
        )
    },
    "sbi.inference.snpe": "NPE",
    "sbi.inference.snle": "NLE",
    "sbi.inference.snre": "NRE",
}

MULTIROUND_GUIDE = (
    "https://sbi.readthedocs.io/en/latest/how_to_guide/02_multiround_inference.html"
)


def _sequential_note(removed: str) -> str:
    if removed not in SEQUENTIAL:
        return ""
    return (
        f" Sequential (multi-round) inference still works: run `{SEQUENTIAL[removed]}` "
        f"over several rounds, see {MULTIROUND_GUIDE}"
    )


def removed_names_getattr(module_name: str) -> Callable[[str], NoReturn]:
    """Returns a module `__getattr__` that names the replacement of a removed name.

    Other unknown names raise `AttributeError` as usual, so that `hasattr()` and
    `getattr()` with a default keep working for them.

    Assign the result under `if not TYPE_CHECKING:`. Type checkers accept any name
    from a module that defines `__getattr__`, so they would no longer flag unknown
    or removed names.
    """

    def __getattr__(name: str) -> NoReturn:
        if name in REMOVED_NAMES:
            raise ImportError(
                f"The name `{name}` was removed from sbi. "
                f"Use `{REMOVED_NAMES[name]}` instead.{_sequential_note(name)}"
            )
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__


def raise_removed_module(module_name: str) -> NoReturn:
    """Raises the `ImportError` for a module that sbi removed.

    Not a `ModuleNotFoundError`: `from package import module` replaces the message of
    that error with a bare "cannot import name".
    """
    raise ImportError(
        f"The module `{module_name}` was removed from sbi. "
        f"Use `{REMOVED_MODULES[module_name]}` instead."
        f"{_sequential_note(module_name)}"
    )
