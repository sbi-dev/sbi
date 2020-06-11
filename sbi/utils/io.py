# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

"""Utility functions for input/output."""

import os
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.parent.absolute()


def get_log_root():
    return os.path.join(os.getcwd(), "sbi-logs")


def get_data_root():
    return os.path.join(get_project_root(), "data")
