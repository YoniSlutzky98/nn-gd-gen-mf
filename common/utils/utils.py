from typing import Tuple

import torch
import torch.nn as nn
import GPUtil
import numpy as np


def get_available_device(max_load: float = 0.3,
                         max_memory: float = 0.3) -> torch.device:
# Try to get a GPU that is under the load and memory thresholds.
    available_gpus = GPUtil.getAvailable(order='first', maxLoad=max_load, maxMemory=max_memory, limit=torch.cuda.device_count())
    if available_gpus:
        return torch.device(f'cuda:{available_gpus[0]}')
    else:
        return torch.device('cpu')

def median_iqr(mat: np.ndarray) -> Tuple[float, float]:
    """Return median, lower error, upper error along axis‑1."""
    med = np.median(mat, axis=1)
    q1  = np.percentile(mat, 25, axis=1)
    q3  = np.percentile(mat, 75, axis=1)
    yerr = np.vstack([med - q1, q3 - med])
    return med, yerr

def activation_factory(name: str):

    if name == "Linear":
        return nn.Identity
    elif name == "Tanh":
        return nn.Tanh
    elif name == "Leaky ReLU":
        return nn.LeakyReLU
    else:
        print('Activation not implemented')
        raise NotImplementedError

def filename_extensions(gt_rank: int,
                        activation: str,
                        gnc_init: str,
                        gd_momentum: float,
                        completion: bool) -> str:
    if activation == "Leaky ReLU":
        activation = "LeakyReLU"
    ext = f'_gt_rank={gt_rank}_act={activation}_gnc_init={gnc_init}'
    if gd_momentum != 0:
        ext = ext + f'_gd_momentum={gd_momentum}'
    if completion:
        ext = ext + '_completion'
    return ext