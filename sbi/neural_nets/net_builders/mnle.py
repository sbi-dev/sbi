# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import warnings

from .mixed_nets import build_mnle as _build_mnle


def build_mnle(*args, **kwargs):
    """
    This function is deprecated. Please use
    'from sbi.neural_nets.net_builders.mixed_nets import build_mnle' instead.

    See mixed_nets.build_mnle for full documentation.
    """
    warnings.warn(
        "This import path is deprecated and will be removed in a future version. "
        "Please use 'from sbi.neural_nets.net_builders.mixed_nets import build_mnle' "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_mnle(*args, **kwargs)
