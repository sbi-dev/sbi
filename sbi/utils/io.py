"""Utility functions for Input/Output."""

import os
import time


def get_timestamp():
    formatted_time = time.strftime("%d-%b-%y||%H:%M:%S")
    return formatted_time


def get_project_root():
    return Path(__file__).parent.parent.absolute()


def get_log_root():
    return os.path.join(os.getcwd(), "log")


def get_data_root():
    return os.path.join(get_project_root(), "data")
