"""Paired dense vs bounded cross-view candidate protection; default three epochs per arm."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

from tools.owod.run_sowodb_task2_routing import main


if __name__ == '__main__':
    main(comparison='unknown')
