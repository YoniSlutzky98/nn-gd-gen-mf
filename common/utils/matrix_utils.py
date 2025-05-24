from typing import Tuple

import torch


def random_fixed_rank_matrix(n_rows: int,
                             n_cols: int,
                             rank: int = 1,
                             norm: float = 1.0,
                             device: torch.device = torch.device("cpu")) -> torch.Tensor:
    U = torch.randn(n_rows, rank, device=device)
    V = torch.randn(rank, n_cols, device=device)
    Y = U @ V
    return Y * (norm / Y.norm())

def orthonormal_complement_basis(tensors: torch.Tensor) -> torch.Tensor:
    bs, n_rows, n_cols = tensors.shape
    dim = n_rows * n_cols
    V = tensors.reshape(bs, dim)
    Q, _ = torch.linalg.qr(V.T, mode='complete')
    rank = torch.linalg.matrix_rank(V).item()
    if rank == dim:
        return tensors.new_empty((0, n_rows, n_cols))
    complement = Q[:, rank:]
    return complement.T.reshape(dim - rank, n_rows, n_cols)

def gd_sensing_loss(Y: torch.Tensor,
                 A: torch.Tensor,
                 b: torch.Tensor) -> torch.Tensor:
    preds = (A * Y).view(A.size(0), -1).sum(-1)
    return (preds - b).pow(2).mean()

def gnc_sensing_loss(Y: torch.Tensor,
                     A: torch.Tensor,
                     b: torch.Tensor) -> torch.Tensor:
    preds = (A.unsqueeze(0) * Y.unsqueeze(1)).view(Y.size(0), A.size(0), -1).sum(-1)
    return ((preds - b.unsqueeze(0)) ** 2).mean(1)

def random_one_hot(n_rows: int,
                   n_cols: int,
                   num_measurements: int,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    basis = torch.eye(n_rows * n_cols, device=device).reshape(n_rows * n_cols, n_rows, n_cols)
    perm = torch.randperm(n_rows * n_cols, device=device)
    train_idx, test_idx = perm[:num_measurements], perm[num_measurements:]
    A_train = basis[train_idx]
    A_test  = basis[test_idx]
    return A_train, A_test

def generate_data(n_rows: int,
                  n_cols: int,
                  gt_rank: int,
                  gt_norm: float,
                  num_measurements: int,
                  device: torch.device,
                  completion: bool = False):
    Y_true = random_fixed_rank_matrix(n_rows, n_cols, gt_rank, gt_norm, device=device)
    if completion:
        A_train, A_test = random_one_hot(n_rows, n_cols, num_measurements, device=device)
    else:
        A_train = torch.randn(num_measurements, n_rows, n_cols, device=device)
        A_train = A_train / A_train.norm(dim=(1, 2), keepdim=True)
        A_test = orthonormal_complement_basis(A_train)
    b_train = (A_train * Y_true).view(A_train.size(0), -1).sum(-1)
    b_test = (A_test * Y_true).view(A_test.size(0), -1).sum(-1)

    return A_train, b_train, A_test, b_test