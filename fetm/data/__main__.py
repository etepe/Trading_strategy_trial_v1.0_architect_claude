"""CLI entry point: python -m fetm.data"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fetm.config import load_config
from fetm.data.download import download_ticker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="FETM data downloader")
    parser.add_argument("--ticker", default=config["data"]["ticker"])
    parser.add_argument("--start-date", default=config["data"]["start_date"])
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead of downloading")
    args = parser.parse_args()

    output_dir = Path(config["data"]["storage_dir"])

    if args.synthetic:
        from fetm.data.synthetic import generate_spy_like_data
        logger.info("Generating synthetic %s data", args.ticker)
        df = generate_spy_like_data(start_date=args.start_date, end_date=args.end_date or "2026-03-28")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{args.ticker}.parquet"
        df.to_parquet(path)
        logger.info("Saved synthetic data to %s (%d rows)", path, len(df))
        return

    try:
        download_ticker(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
        )
    except Exception:
        logger.warning("Download failed. Falling back to synthetic data generation.")
        from fetm.data.synthetic import generate_spy_like_data
        df = generate_spy_like_data(start_date=args.start_date, end_date=args.end_date or "2026-03-28")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{args.ticker}.parquet"
        df.to_parquet(path)
        logger.info("Saved synthetic data to %s (%d rows)", path, len(df))


if __name__ == "__main__":
    main()
