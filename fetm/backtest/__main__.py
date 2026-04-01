"""CLI entry point: python -m fetm.backtest"""

import argparse
import logging

from fetm.config import load_config
from fetm.data.download import load_raw_data
from fetm.backtest.engine import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="FETM backtest runner")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--ticker", default=None, help="Override ticker")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.ticker:
        config["data"]["ticker"] = args.ticker

    ticker = config["data"]["ticker"]
    logger.info("Running Phase 1 backtest for %s", ticker)

    data = load_raw_data(ticker, config["data"]["storage_dir"])
    engine = BacktestEngine(config)
    results = engine.run(data)
    run_dir = engine.save_results(results)

    # Print summary
    import json
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    print("\n" + "=" * 70)
    print("FETM Phase 1 Backtest Results (Out-of-Sample)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Buy&Hold':>10} {'Linear':>10} {'Binary':>10} {'FETM':>10}")
    print("-" * 70)
    for metric in ["annualized_return", "annualized_vol", "sharpe_ratio",
                    "sortino_ratio", "max_drawdown", "calmar_ratio",
                    "annual_turnover", "win_rate_monthly"]:
        vals = []
        for strat in ["buyhold", "linear", "binary", "fetm"]:
            v = metrics["strategies"][strat].get(metric, "N/A")
            if isinstance(v, float):
                vals.append(f"{v:>10.4f}")
            else:
                vals.append(f"{v!s:>10}")
        print(f"{metric:<25} {''.join(vals)}")
    print("=" * 70)
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
