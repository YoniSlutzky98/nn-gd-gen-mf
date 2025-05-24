import math

from typing import Type, Tuple

import torch
import torch.nn as nn

from common.models.matrix_factorization import NonlinearMatrixFactorization
from common.models.group_rmsprop import GroupRMSprop
from common.utils.matrix_utils import gd_sensing_loss, gnc_sensing_loss


def train_gd(seed: int,
             n_rows: int,
             n_cols: int,
             width: int,
             depth: int,
             activation: Type[nn.Module],
             device: torch.device,
             A_train: torch.Tensor,
             b_train: torch.Tensor,
             A_test: torch.Tensor,
             b_test: torch.Tensor,
             init_scale: float,
             lr: float,
             epochs: int,
             log_period: int,
             print_period: int,
             momentum: float = 0.0,
             negative_slope: float = 0.2,
             verbose: bool = False):
    torch.manual_seed(seed)
    model = NonlinearMatrixFactorization(n_rows, n_cols, width, depth, activation, device, negative_slope)
    model.near_zero_init(init_scale)
    optimizer = GroupRMSprop(model.parameters(), lr=lr, momentum=momentum)
    train_hist, test_hist = [], []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_train = gd_sensing_loss(model(), A_train, b_train)
        loss_train.backward()
        optimizer.step()
        if epoch % log_period == 0:
            with torch.no_grad():
                loss_test = gd_sensing_loss(model(), A_test, b_test)
                train_hist.append(loss_train.item())
                test_hist.append(loss_test.item())
                if (epoch % print_period == 0) and verbose:
                    print(f'epoch {epoch}')
                    print(f'train loss={loss_train.item()}, test loss={loss_test.item()}')
    return test_hist[-1]

def train_gnc(seed: int,
             n_rows: int,
             n_cols: int,
             width: int,
             depth: int,
             activation: Type[nn.Module],
             device: torch.device,
             A_train: torch.Tensor,
             b_train: torch.Tensor,
             A_test: torch.Tensor,
             b_test: torch.Tensor,
             eps_train: float,
             num_samples: int,
             batch_size: int,
             initalization: str,
             normalize: bool,
             softening: float = 1e-6,
             negative_slope: float = 0.2):
    if initalization == 'gauss':
        init = torch.randn
    elif initalization == 'unif':
        init = lambda bs, r, c, device: 2 * torch.rand(bs, r, c, device=device) - 1
    else:
        print('Initialization not implemented')
        raise NotImplementedError

    if activation is nn.LeakyReLU:
        act = activation(negative_slope)
    else:
        act = activation()

    torch.manual_seed(seed)
    prior_gen_losses = []
    gnc_gen_losses = []
    for batch in range(math.ceil(num_samples / batch_size)):
        bs = min(batch_size, num_samples - batch * batch_size)
        Y = torch.eye(n_rows, device=device).unsqueeze(0).expand(bs, n_rows, n_rows)
        for _ in range(depth - 1):
            fan_in = Y.size(-1)
            W = init(bs, fan_in, width, device=device) / math.sqrt(width)
            Y = torch.bmm(Y, W)
            Y = act(Y)
        W = init(bs, width, n_cols, device=device) / math.sqrt(n_cols)
        Y = torch.bmm(Y, W)
        if normalize:
            norms = Y.norm(p='fro', dim=(1, 2), keepdim=True)
            Y = Y / (norms + softening)
        train_losses = gnc_sensing_loss(Y, A_train, b_train)
        succ_mask = train_losses < eps_train
        gen_losses = gnc_sensing_loss(Y, A_test, b_test)
        prior_gen_losses.extend(gen_losses.cpu().tolist())
        gnc_gen_losses.extend(gen_losses[succ_mask].cpu().tolist())
    mean_prior = sum(prior_gen_losses) / len(prior_gen_losses)
    mean_gnc = sum(gnc_gen_losses) / len(gnc_gen_losses) if gnc_gen_losses else 1
    return mean_prior, mean_gnc