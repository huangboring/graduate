"""Timing shared by inference modules without importing legacy training code."""
import time
import torch


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
