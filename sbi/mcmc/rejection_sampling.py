from typing import Any
from warnings import warn

import torch
from torch import Tensor, exp, log, rand

from sbi.utils import (
    optimize_potential_fn,
    rejection_sample_raw,
    sample_posterior_within_prior,
)
