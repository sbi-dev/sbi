"""Utility functions for Input/Output."""

import os
import socket
import time


def get_timestamp():
    formatted_time = time.strftime("%d-%b-%y||%H:%M:%S")
    return formatted_time


def get_project_root():
    return Path(__file__).parent.parent.absolute()


def get_log_root():
    return os.path.join(get_project_root(), "log")


def get_data_root():
    return os.path.join(get_project_root(), "data")


def get_output_root():
    return os.path.join(get_project_root(), "out")


def get_checkpoint_root():
    return os.path.join(get_project_root(), "checkpoint")
