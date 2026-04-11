"""CLI entry point: python -m fetm.optimize

Runs walk-forward Optuna optimization over the full FETM parameter search
space and writes results to ``output/optimization/<timestamp>/`` (or to the
directory supplied via ``--output``).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from fetm.config import load_config
from fetm.optimize.walk_forward_optuna import WalkForwardOptunaOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


BUDGET_PRESETS = {
    "quick": 50,
    "full": 200,
}


def _best_params_as_config(best: dict) -> dict:
    """Expand dot-path best_params into a nested dict ready for settings.yaml."""
    out: dict = {}
    for path, value in best.items():
        keys = path.split(".")
        cursor = out
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[keys[-1]] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward Optuna optimization for the FETM strategy."
    )
    parser.add_argument("--config", default=None, help="Path to base config YAML.")
    parser.add_argument(
        "--budget", choices=list(BUDGET_PRESETS.keys()), default="quick",
        help="Optuna trials per walk-forward window.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override the trials-per-window count from --budget.",
    )
    parser.add_argument(
        "--lambda-dd", type=float, default=0.5,
        help="Drawdown penalty weight in the scalar objective (sharpe - lam*|dd|).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the Optuna TPE sampler.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory. Defaults to output/optimization/<timestamp>/.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    n_trials = args.n_trials if args.n_trials is not None else BUDGET_PRESETS[args.budget]

    # Deferred import so --help / module imports don't drag in yfinance.
    from fetm.data.download import load_raw_data

    ticker = config["data"]["ticker"]
    logger.info("Loading data for %s", ticker)
    data = load_raw_data(ticker, config["data"]["storage_dir"])

    wf_cfg = config.get("backtest", {}).get("walk_forward", {})
    initial_window_years = int(wf_cfg.get("initial_window_years", 5))
    refit_freq_years = int(wf_cfg.get("refit_frequency_years", 1))

    optimizer = WalkForwardOptunaOptimizer(
        base_config=config,
        initial_window_years=initial_window_years,
        refit_freq_years=refit_freq_years,
        n_trials=n_trials,
        lambda_dd=args.lambda_dd,
        seed=args.seed,
    )

    logger.info(
        "Starting optimization: budget=%s (%d trials/window), lambda_dd=%.2f",
        args.budget, n_trials, args.lambda_dd,
    )
    result = optimizer.run(data)

    # Resolve output directory.
    if args.output:
        out_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("output/optimization") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist artifacts.
    result.per_window.to_parquet(out_dir / "walk_forward_results.parquet")
    result.trials.to_parquet(out_dir / "trials.parquet")

    best_nested = _best_params_as_config(result.best_params)
    with open(out_dir / "best_params.yaml", "w") as f:
        yaml.dump(best_nested, f, default_flow_style=False, sort_keys=False)

    run_config = {
        "budget": args.budget,
        "n_trials_per_window": n_trials,
        "lambda_dd": args.lambda_dd,
        "seed": args.seed,
        "initial_window_years": initial_window_years,
        "refit_freq_years": refit_freq_years,
        "base_config": config,
    }
    with open(out_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(result.summary, f, indent=2, default=str)

    # Update a "latest" convenience symlink next to the timestamped dir.
    latest = out_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.name)
    except OSError:
        # Symlinks may be unavailable on some filesystems; ignore silently.
        pass

    logger.info("Optimization complete. Output: %s", out_dir)
    print("\n" + "=" * 70)
    print("FETM Walk-Forward Optimization Summary")
    print("=" * 70)
    for k, v in result.summary.items():
        print(f"{k:<30} {v}")
    print("=" * 70)
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
