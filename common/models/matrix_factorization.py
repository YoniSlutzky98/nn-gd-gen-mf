from typing import Type

import torch
import torch.nn as nn


class NonlinearMatrixFactorization(nn.Module):
    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 width: int,
                 depth: int,
                 activation: Type[nn.Module],
                 device: torch.device,
                 negative_slope: float = 0.2):
        super().__init__()
        self.register_buffer("_I", torch.eye(n_rows, device=device))
        layers = []
        dim = n_rows
        for _ in range(depth - 1):
            if activation is nn.LeakyReLU:
                act = activation(negative_slope)
            else:
                act = activation()
            layers += [nn.Linear(dim, width, bias=False), act]
            dim = width
        layers.append(nn.Linear(dim, n_cols, bias=False))
        self.net = nn.Sequential(*layers).to(device)

    def forward(self):
        Y = self.net(self._I).t()
        return Y

    def near_zero_init(self,
             init_scale: float):
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(init_scale)