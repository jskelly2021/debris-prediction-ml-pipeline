import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logger import setup_logger
from reporting import ResearchPacketBuilder


def parse_args():
    """Parse CLI arguments for research packet generation."""

    parser = argparse.ArgumentParser(description="Build a research packet from a batch output directory")
    parser.add_argument("batch_dir", type=str, help="Path to batch output directory")
    return parser.parse_args()


def main():
    """Build the research packet and print generated artifact paths."""

    setup_logger()
    args = parse_args()
    summary = ResearchPacketBuilder(Path(args.batch_dir)).build()

    print("Research packet complete")
    print(f"  Batch directory  : {summary['batch_dir']}")
    print(f"  Report           : {summary['report_path']}")
    print(f"  Packet directory : {summary['packet_dir']}")
    print(f"  Packet manifest  : {summary['packet_manifest_path']}")
    print(f"  Figures          : {summary['figures_dir']}")
    print(f"  Tables           : {summary['tables_dir']}")


if __name__ == "__main__":
    main()
