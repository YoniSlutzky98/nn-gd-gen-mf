from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict

try:
    import yaml  # optional dependency; falls back to JSON if missing
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

__all__ = ["parse_args"]


# ------------------------------------------------------------------
# Helper ----------------------------------------------------------------


def _load_config(path: pathlib.Path) -> Dict[str, Any]:
    """Load a *YAML* or *JSON* config file into a dict."""
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text()

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed – `pip install pyyaml` or use JSON config.")
        return yaml.safe_load(text) or {}

    if path.suffix.lower() == ".json":
        return json.loads(text)

    raise ValueError("Unsupported config file type – use .yaml/.yml or .json")


# ------------------------------------------------------------------
# Public ----------------------------------------------------------------


def parse_args(experiment: str) -> argparse.Namespace:  # noqa: C901 – a bit long but flat
    """Parse CLI & optional YAML/JSON config.

    Parameters
    ----------
    experiment: ``"depth"`` or ``"width"`` – which driver is calling us.
    """
    if experiment not in {"depth", "width"}:
        raise ValueError("experiment must be 'depth' or 'width'")

    parser = argparse.ArgumentParser(
        description=f"Non‑linear Matrix Factorisation – {experiment} sweep"
    )

    # ------------------------------------------------------------------
    # Generic experiment parameters                                     |
    # ------------------------------------------------------------------
    g = parser.add_argument_group("Problem dimensions & data generation")
    g.add_argument("--n_rows", type=int, default=5, help="Number of rows of ground truth matrix")
    g.add_argument("--n_cols", type=int, default=5, help="Number of columns of ground truth matrix")
    g.add_argument("--gt_rank", type=int, default=1, help="Rank of ground truth matrix")
    g.add_argument("--gt_norm", type=float, default=1.0, help="Frobenius norm of ground truth matrix")
    g.add_argument("--num_seeds", type=int, default=8, help="Number of random seeds per setting")
    g.add_argument("--num_measurements", type=int, default=15, help="Number of measurements")
    g.add_argument("--completion", action="store_true", default=False, help="Should task be matrix completion")

    g.add_argument(
        "--activation",
        choices=["Linear", "Tanh", "Leaky ReLU"],
        default="Linear",
        help="Activation function for the matrix factorization",
    )
    g.add_argument(
        "--negative_slope",
        type=float,
        default=0.2,
        help="Negative slope for the LeakyReLU activation (ignored otherwise)",
    )

    g = parser.add_argument_group("Guess & Check hyperparameters")
    g.add_argument("--gnc_num_samples", type=int, default=int(1e8), help="Number of G&C samples")
    g.add_argument("--gnc_batch_sizes", type=int, nargs="+", default=[int(1e7)], help="Batch sizes for G&C")
    g.add_argument("--gnc_eps_train", type=float, default=0.01, help="Training loss threshold for successful G&C trial")
    g.add_argument("--gnc_normalize", action="store_true", default=False, help="Should G&C matrix factorization be normalized")
    g.add_argument("--gnc_softening", type=float, default=1e-6, help="Normalization softening constant")
    g.add_argument(
        "--gnc_init",
        type=str,
        choices=["gauss", "unif"],
        default="gauss",
        help="Prior initialization distribution for G&C"
    )

    g = parser.add_argument_group("Gradient Descent hyper‑parameters")
    g.add_argument("--gd_lrs", type=float, nargs="+", default=[1e-2], help="Learning rates for GD")
    g.add_argument("--gd_epochs", type=int, default=int(1e5), help="Number of epochs for GD")
    g.add_argument("--gd_init_scales", type=float, nargs="+", default=[1e-2], help="Initialization scales for GD")
    g.add_argument("--gd_log_period", type=int, default=int(1e2), help="Log period for GD")
    g.add_argument("--gd_print_period", type=int, default=int(1e4), help="Print period for GD (if gd_verbose=True)")
    g.add_argument("--gd_momentum", type=float, default=0.0, help="Momentum for GD")
    g.add_argument("--gd_verbose", action="store_true", default=False, help="Should GD training be verbose")

    # ------------------------------------------------------------------
    # Sweep‑specific                                                     |
    # ------------------------------------------------------------------
    if experiment == "depth":
        g = parser.add_argument_group("Depth sweep")
        g.add_argument("--width", type=int, default=5, help="Fixed width for depth experiments")
        g.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9, 10], help="Depths for depth experiments")
    else:  # width sweep
        g = parser.add_argument_group("Width sweep")
        g.add_argument("--depth", type=int, default=2, help="Fixed depth for width experiments")
        g.add_argument("--widths", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160], help="Widths for width experiments")

    # ------------------------------------------------------------------
    # Meta                                                               |
    # ------------------------------------------------------------------
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="Optional YAML or JSON config file (CLI flags override)",
    )
    parser.add_argument("--results_dir", type=pathlib.Path, default=pathlib.Path("./results"), help="Results directory")
    parser.add_argument("--figures_dir", type=pathlib.Path, default=pathlib.Path("./figures"), help="Figures directory")

    # ---- first round parse (just to grab --config) --------------------
    if "--config" in parser.parse_known_args()[0]._get_args():  # type: ignore
        # weird, but keeps mypy happy; we only want to sniff existence
        pass

    args, unknown = parser.parse_known_args()

    # ------------------------------------------------------------------
    # Config file overrides                                              |
    # ------------------------------------------------------------------
    if args.config is not None:
        cfg_dict = _load_config(args.config)
        parser.set_defaults(**cfg_dict)
        # re‑parse now with new defaults (and original CLI flags win)
        args = parser.parse_args()

    # Expand width/depth‑wise lists to correct length -------------------
    if experiment == "depth":
        expected = len(args.depths)
    else:
        expected = len(args.widths)

    # Helper to pad/validate per‑depth/width lists
    def _check_list(name: str, lst: list[Any]):
        if len(lst) == 1:
            lst *= expected
        if len(lst) != expected:
            raise ValueError(f"{name} expects {expected} values, got {len(lst)}")
        return lst

    args.gnc_batch_sizes = _check_list("gnc_batch_sizes", args.gnc_batch_sizes)
    args.gd_lrs = _check_list("gd_lrs", args.gd_lrs)
    args.gd_init_scales = _check_list("gd_init_scales", args.gd_init_scales)

    # Final tweaks ------------------------------------------------------
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    return args