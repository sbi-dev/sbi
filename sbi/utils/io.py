# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

"""Utility functions for input/output."""

import os
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.parent.absolute()


def get_log_root():
    return os.path.join(os.getcwd(), "sbi-logs")


def get_data_root():
    return os.path.join(get_project_root(), "data")
