"""Run agentDCA from a source checkout without installing its entry point."""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    importlib.import_module("agentdca.server").main()
