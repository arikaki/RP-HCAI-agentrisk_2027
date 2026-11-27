"""Download AgentRisk train/dev and the split manifest into ./data as JSONL.

    python scripts/download_data.py --data-dir data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.data import download_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    download_all(ap.parse_args().data_dir)
