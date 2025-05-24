import time

import numpy as np
import pandas as pd

from common.utils.matrix_utils import generate_data
from common.utils.utils import get_available_device, activation_factory, filename_extensions
from common.parser import parse_args
from common.training import train_gnc, train_gd
from common.plotting import plot_depth


def run_depth_experiment(args, device):
    activation = activation_factory(args.activation)

    gd_gen_losses = np.zeros((len(args.depths), args.num_seeds))
    gnc_gen_losses = np.zeros((len(args.depths), args.num_seeds))

    for depth_idx, depth in enumerate(args.depths):
        print(f'Starting depth={depth}')
        gnc_bs = args.gnc_batch_sizes[depth_idx]
        gd_lr = args.gd_lrs[depth_idx]
        gd_init_scale = args.gd_init_scales[depth_idx]
        for seed in range(args.num_seeds):
            print('-----------------------------------------------------------------------')
            print(f'Starting seed={seed}')
            A_train, b_train, A_test, b_test = generate_data(args.n_rows,
                                                             args.n_cols,
                                                             args.gt_rank,
                                                             args.gt_norm,
                                                             args.num_measurements,
                                                             device,
                                                             args.completion)
            print("Starting G&C")
            t0 = time.time()
            _, gnc_gen_loss = train_gnc(seed,
                                        args.n_rows,
                                        args.n_cols,
                                        args.width,
                                        depth,
                                        activation,
                                        device,
                                        A_train,
                                        b_train,
                                        A_test,
                                        b_test,
                                        args.gnc_eps_train,
                                        args.gnc_num_samples,
                                        gnc_bs,
                                        args.gnc_init,
                                        args.gnc_normalize,
                                        args.gnc_softening,
                                        args.negative_slope)
            gnc_gen_losses[depth_idx, seed] = gnc_gen_loss
            t1 = time.time()
            print(f'Finished G&C, time elapsed={t1 - t0}s')

            print("Starting GD")
            t0 = time.time()
            gd_gen_loss = train_gd(seed,
                                   args.n_rows,
                                   args.n_cols,
                                   args.width,
                                   depth,
                                   activation,
                                   device,
                                   A_train,
                                   b_train,
                                   A_test,
                                   b_test,
                                   gd_init_scale,
                                   gd_lr,
                                   args.gd_epochs,
                                   args.gd_log_period,
                                   args.gd_print_period,
                                   args.gd_momentum,
                                   args.negative_slope,
                                   args.gd_verbose)
            gd_gen_losses[depth_idx, seed] = gd_gen_loss
            t1 = time.time()
            print(f'Finished GD, time elapsed={t1 - t0}s')
            print(f'Finished seed={seed}')
        print('----------------------------------------------------------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------------------------------------------------')

    return gnc_gen_losses, gd_gen_losses

def main():
    args = parse_args("depth")
    device = get_available_device()
    print(f'Using device {device}.')
    gnc_gen_losses, gd_gen_losses = run_depth_experiment(args, device)

    results_filename = 'depth_results' + filename_extensions(args.gt_rank, args.activation, args.gnc_init, args.gd_momentum, args.completion) + '.csv'
    csv_path = args.results_dir / results_filename
    cols_interleaved = np.empty((len(args.depths), 2 * args.num_seeds))
    cols_interleaved[:, 0::2] = gnc_gen_losses
    cols_interleaved[:, 1::2] = gd_gen_losses
    col_names = []
    for s in range(args.num_seeds):
        col_names.extend([f"gnc_gen_seed={s}", f"gd_gen_seed={s}"])
    results = pd.DataFrame(
        np.column_stack([args.depths, cols_interleaved]),
        columns=["depth"] + col_names,
    )
    results.to_csv(csv_path, index=False)

    plot_depth(args.depths,
               gnc_gen_losses,
               gd_gen_losses,
               args.gt_rank,
               args.activation,
               args.gnc_init,
               args.gd_momentum,
               args.completion,
               args.figures_dir)

    print(f"Finished depth experiments, results saved to {args.results_dir}, figures saved to {args.figures_dir}")

if __name__ == "__main__":
    main()