import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logger import setup_logger
from overfit_analysis import OverfitAnalysisRunner
from overfit_config import load_overfit_analysis_config
from experiments.override_utils import load_config_dict, validate_resolved_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run an overfit analysis sweep")
    parser.add_argument("analysis_config_path", type=str, help="Path to overfit analysis YAML config file")
    return parser.parse_args()


def main():
    setup_logger()
    args = parse_args()

    analysis_config = load_overfit_analysis_config(args.analysis_config_path)
    base_config_dict = load_config_dict(analysis_config.base_config)
    experiment_config = validate_resolved_config(base_config_dict)

    summary = OverfitAnalysisRunner(
        analysis_config=analysis_config,
        experiment_config=experiment_config,
        base_config_dict=base_config_dict,
    ).run()

    print("Overfit analysis complete")
    print(f"  Analysis directory : {summary['analysis_dir']}")
    print(f"  Results CSV        : {summary['results_path']}")
    print(f"  Report             : {summary['report_path']}")


if __name__ == "__main__":
    main()
