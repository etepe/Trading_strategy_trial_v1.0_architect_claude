"""CLI entry point: python -m fetm.dashboard"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _find_latest_run(runs_dir: Path) -> str | None:
    """Find the actual latest run directory (no symlinks needed)."""
    # 1. Try latest.txt pointer
    latest_txt = runs_dir / "latest.txt"
    if latest_txt.is_file():
        target = latest_txt.read_text().strip()
        if Path(target).exists():
            return str(Path(target).resolve())

    # 2. Pick the most recent timestamped directory
    if runs_dir.is_dir():
        subdirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
            key=lambda d: d.name,
            reverse=True,
        )
        if subdirs:
            return str(subdirs[0].resolve())

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FETM dashboard")
    parser.add_argument("--run-dir", default="output/runs/latest",
                        help="Path to backtest run directory")
    parser.add_argument("--port", default="8501", help="Streamlit port")
    args = parser.parse_args()

    run_dir = args.run_dir

    # Resolve "latest" to the actual timestamped directory BEFORE launching Streamlit
    p = Path(run_dir)
    if not (p / "results.parquet").exists():
        # Path doesn't contain results — try to find the real latest run
        if p.name == "latest":
            resolved = _find_latest_run(p.parent)
        else:
            resolved = _find_latest_run(p)

        if resolved:
            run_dir = resolved
        else:
            print("ERROR: No backtest runs found. Run 'python -m fetm.backtest' first.")
            sys.exit(1)

    app_path = Path(__file__).parent / "app.py"

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", args.port,
        "--",
        "--run-dir", run_dir,
    ]

    print(f"Launching FETM dashboard at http://localhost:{args.port}")
    print(f"Run directory: {run_dir}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
