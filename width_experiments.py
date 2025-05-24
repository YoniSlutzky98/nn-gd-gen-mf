import time

import numpy as np
import pandas as pd

from common.utils.matrix_utils import generate_data
from common.utils.utils import get_available_device, activation_factory, filename_extensions
from common.parser import parse_args
from common.training import train_gnc, train_gd
from common.plotting import plot_width


def run_width_experiment(args, device):
    activation = activation_factory(args.activation)

    gnc_gen_losses = np.zeros((len(args.widths), args.num_seeds))
    gd_gen_losses = np.zeros((len(args.widths), args.num_seeds))
    prior_gen_losses = np.zeros((len(args.widths), args.num_seeds))

    for width_idx, width in enumerate(args.widths):
        print(f'Starting width={width}')
        gnc_bs = args.gnc_batch_sizes[width_idx]
        gd_lr = args.gd_lrs[width_idx]
        gd_init_scale = args.gd_init_scales[width_idx]
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
            prior_gen_loss, gnc_gen_loss = train_gnc(seed,
                                                     args.n_rows,
                                                     args.n_cols,
                                                     width,
                                                     args.depth,
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
            gnc_gen_losses[width_idx, seed] = gnc_gen_loss
            prior_gen_losses[width_idx, seed] = prior_gen_loss
            t1 = time.time()
            print(f'Finished G&C, time elapsed={t1 - t0}s')

            print("Starting GD")
            t0 = time.time()
            gd_gen_loss = train_gd(seed,
                                   args.n_rows,
                                   args.n_cols,
                                   width,
                                   args.depth,
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
            gd_gen_losses[width_idx, seed] = gd_gen_loss
            t1 = time.time()
            print(f'Finished GD, time elapsed={t1 - t0}s')
            print(f'Finished seed={seed}')
        print('----------------------------------------------------------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------------------------------------------------')

    return gnc_gen_losses, gd_gen_losses, prior_gen_losses

def main():
    args = parse_args("width")
    device = get_available_device()
    print(f'Using device {device}.')
    gnc_gen_losses, gd_gen_losses, prior_gen_losses = run_width_experiment(args, device)

    results_filename = 'width_results' + filename_extensions(args.gt_rank, args.activation, args.gnc_init, args.gd_momentum, args.completion) + '.csv'
    csv_path = args.results_dir / results_filename
    cols_interleaved = np.empty((len(args.widths), 3 * args.num_seeds))
    cols_interleaved[:, 0::3] = gnc_gen_losses
    cols_interleaved[:, 1::3] = prior_gen_losses
    cols_interleaved[:, 2::3] = gd_gen_losses
    col_names = []
    for s in range(args.num_seeds):
        col_names.extend([f"gnc_gen_seed={s}", f"prior_gen_seed={s}", f"gd_gen_seed={s}"])
    results = pd.DataFrame(
        np.column_stack([args.widths, cols_interleaved]),
        columns=["width"] + col_names,
    )
    results.to_csv(csv_path, index=False)

    plot_width(args.widths,
               gnc_gen_losses,
               gd_gen_losses,
               prior_gen_losses,
               args.gt_rank,
               args.activation,
               args.gnc_init,
               args.gd_momentum,
               args.completion,
               args.figures_dir)

    print(f"Finished width experiments, results saved to {args.results_dir}, figures saved to {args.figures_dir}")

if __name__ == "__main__":
    main()