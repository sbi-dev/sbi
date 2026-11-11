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

# Old trainer and ABC names. `infer(method=...)` accepted them as well.
RENAMED_METHODS = {
    "SNL": "NLE_A",
    "SNLE": "NLE_A",
    "SNLE_A": "NLE_A",
    "SNPE_A": "NPE_A",
    "SNPE_B": "NPE_B",
    "SNPE": "NPE_C",
    "SNPE_C": "NPE_C",
    "APT": "NPE_C",
    "AALR": "NRE_A",
    "SNRE_A": "NRE_A",
    "SRE": "NRE_B",
    "SNRE": "NRE_B",
    "SNRE_B": "NRE_B",
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
}

REMOVED_MODULES = {
    "sbi.inference.base": "from sbi.inference import infer",
    "sbi.inference.snpe": "from sbi.inference import NPE_C",
    "sbi.inference.snle": "from sbi.inference import NLE_A",
    "sbi.inference.snre": "from sbi.inference import NRE_B",
    "sbi.utils.posterior_ensemble": "from sbi.inference import EnsemblePosterior",
}


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
                f"`{name}` was removed from sbi. Use `{REMOVED_NAMES[name]}` instead."
            )
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__


def raise_removed_module(module_name: str) -> NoReturn:
    """Raises the `ImportError` for a module that sbi removed.

    Not a `ModuleNotFoundError`: `from package import module` replaces the message of
    that error with a bare "cannot import name".
    """
    raise ImportError(
        f"`{module_name}` was removed from sbi. "
        f"Use `{REMOVED_MODULES[module_name]}` instead."
    )
