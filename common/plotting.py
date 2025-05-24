import os

import numpy as np
import matplotlib.pyplot as plt

from common.utils.utils import median_iqr, filename_extensions


def plot_depth(depths: list,
               gnc_gen_losses: np.ndarray,
               gd_gen_losses: np.ndarray,
               gt_rank: int,
               activation: str,
               gnc_init: str,
               gd_momentum: bool,
               completion: bool,
               figures_dir: str = './figures'):
    os.makedirs(figures_dir, exist_ok=True)
    plot_filename = 'depth_plot' + filename_extensions(gt_rank, activation, gnc_init, gd_momentum, completion)

    gnc_med, gnc_iqr = median_iqr(gnc_gen_losses)
    gd_med, gd_iqr = median_iqr(gd_gen_losses)

    if gd_momentum:
        gd_label = 'Momentum'
    else:
        gd_label = 'GD'

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    height = 3
    width = 6

    fig, ax = plt.subplots(figsize=(width, height))
    ax.errorbar(
        depths, gnc_med, yerr=gnc_iqr,
        fmt="o-", capsize=3, label="G&C",
        linewidth=2.5, elinewidth=1.5
    )
    ax.errorbar(
        depths, gd_med, yerr=gd_iqr,
        fmt="s--", capsize=3, label=gd_label,
        linewidth=2.5, elinewidth=1.5
    )
    ax.set_title(activation, fontsize="xx-large")
    ax.set_xlabel("Depth", fontsize="xx-large")
    ax.set_ylabel("Generalization Loss", fontsize="xx-large")
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    order = [gd_label, "G&C"]
    handles = [handles[labels.index(l)] for l in order if l in labels]
    labels = [l for l in order if l in labels]
    ax.legend(handles, labels, fontsize="x-large", loc="upper right")

    outfile_base = os.path.join(figures_dir, plot_filename)
    fig.savefig(outfile_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outfile_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_width(widths: list,
               gnc_gen_losses: np.ndarray,
               gd_gen_losses: np.ndarray,
               prior_gen_losses: np.ndarray,
               gt_rank: int,
               activation: str,
               gnc_init: str,
               gd_momentum: bool,
               completion: bool,
               figures_dir: str = './figures'):
    os.makedirs(figures_dir, exist_ok=True)
    plot_filename = 'width_plot' + filename_extensions(gt_rank, activation, gnc_init, gd_momentum, completion)

    gd_med, gd_iqr = median_iqr(gd_gen_losses)
    gnc_med, gnc_iqr = median_iqr(gnc_gen_losses)
    prior_med, prior_iqr = median_iqr(prior_gen_losses)

    if gd_momentum:
        gd_label = 'Momentum'
    else:
        gd_label = 'GD'

    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    height = 3
    width = 6

    fig, ax = plt.subplots(figsize=(width, height))
    ax.errorbar(
        widths, gnc_med, yerr=gnc_iqr,
        fmt="o-", capsize=3, label="G&C",
        linewidth=2.5, elinewidth=1.5
    )
    ax.errorbar(
        widths, gd_med, yerr=gd_iqr,
        fmt="s--", capsize=3, label=gd_label,
        linewidth=2.5, elinewidth=1.5
    )
    ax.errorbar(
        widths, prior_med, yerr=prior_iqr,
        fmt="s--", capsize=3, label="Prior",
        linewidth=2.5, elinewidth=1.5
    )
    ax.set_title(activation, fontsize="xx-large")
    ax.set_xlabel("Width", fontsize="xx-large")
    ax.set_ylabel("Generalization Loss", fontsize="xx-large")
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    order = [gd_label, "Prior", "G&C"]
    handles = [handles[labels.index(l)] for l in order if l in labels]
    labels = [l for l in order if l in labels]
    ax.legend(handles, labels, fontsize="x-large", loc="upper right")

    outfile_base = os.path.join(figures_dir, plot_filename)
    fig.savefig(outfile_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outfile_base + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)