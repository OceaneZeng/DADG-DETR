"""Run the S-OWODB unknown-to-new adapter transfer ablation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]

from tools.owod.run_sowodb_task2_routing import main

if __name__ == '__main__':
    main(comparison='transfer')
