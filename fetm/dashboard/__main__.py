"""CLI entry point: python -m fetm.dashboard"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FETM dashboard")
    parser.add_argument("--run-dir", default="output/runs/latest",
                        help="Path to backtest run directory")
    parser.add_argument("--port", default="8501", help="Streamlit port")
    args = parser.parse_args()

    app_path = Path(__file__).parent / "app.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", args.port,
        "--",
        "--run-dir", args.run_dir,
    ]

    print(f"Launching FETM dashboard at http://localhost:{args.port}")
    print(f"Run directory: {args.run_dir}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
