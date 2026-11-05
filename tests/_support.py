from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def frozen(name, **namespace):
    snapshots=json.loads((ROOT/'tests/fixtures/original_functions.json').read_text())
    exec(compile(snapshots[name]['code'],snapshots[name]['source'],'exec'),namespace)
    return namespace
