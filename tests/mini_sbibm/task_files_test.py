# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import pytest

from .slcp import Slcp
from .two_moons import TwoMoons


@pytest.mark.parametrize("task_class", [Slcp, TwoMoons])
def test_stored_true_parameters_load(task_class):
    """The tasks backed by files load the true parameters of each observation."""
    task = task_class()
    theta_o = task.get_true_parameters(1)

    assert theta_o.shape[-1] == task.theta_dim()
