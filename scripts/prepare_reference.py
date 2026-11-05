#!/usr/bin/env python3
"""Entry point for prepare-reference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from factual_eval.cli import main

if __name__ == '__main__':
    raise SystemExit(main('prepare-reference'))
