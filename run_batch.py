import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiments.batch_config import load_batch_config
from experiments.batch_runner import BatchRunner
from logger import setup_logger


def parse_args():
    """Parse CLI arguments for a batch experiment run."""

    parser = argparse.ArgumentParser(description="Run a batch of ML experiments")
    parser.add_argument("batch_config_path", type=str, help="Path to batch YAML config file")
    return parser.parse_args()


def main():
    """Run a configured experiment batch."""

    setup_logger()
    args = parse_args()
    batch_config = load_batch_config(args.batch_config_path)
    summary = BatchRunner(batch_config).run()

    print("Batch complete")
    print(f"  Output directory : {summary['batch_dir']}")
    print(f"  Succeeded        : {summary['succeeded_count']}")
    print(f"  Failed           : {summary['failed_count']}")
    print(f"  Combined results : {summary['combined_results_path']}")
    print(f"  Failed runs      : {summary['failed_runs_path']}")
    print(f"  Manifest         : {summary['manifest_path']}")


if __name__ == "__main__":
    main()
