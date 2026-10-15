"""Copy the adabmdca agent skill from an adabmDCApy checkout into this plugin.

The skill is maintained in adabmDCApy (skills/adabmdca). agentDCA ships a copy so that
plugin users, including clients without a shell, get the workflow and interpretation
guidance that the dca-analysis skill builds on. Run this after the skill changes upstream:

    python scripts/sync_adabmdca_skill.py ../adabmDCApy
    python scripts/sync_adabmdca_skill.py ../adabmDCApy --check   # exit 1 if out of date
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "skills" / "adabmdca"


def _files(folder: Path) -> set[Path]:
    return {path.relative_to(folder) for path in folder.rglob("*") if path.is_file()}


def differences(source: Path, target: Path) -> list[str]:
    source_files, target_files = _files(source), _files(target) if target.exists() else set()
    changed = [
        str(name)
        for name in sorted(source_files & target_files)
        if not filecmp.cmp(source / name, target / name, shallow=False)
    ]
    return (
        [f"missing: {name}" for name in sorted(source_files - target_files)]
        + [f"extra: {name}" for name in sorted(target_files - source_files)]
        + [f"changed: {name}" for name in changed]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("adabmdcapy", type=Path, help="Path to an adabmDCApy checkout.")
    parser.add_argument("--check", action="store_true", help="Only report differences.")
    args = parser.parse_args()

    source = args.adabmdcapy.expanduser().resolve() / "skills" / "adabmdca"
    if not (source / "SKILL.md").is_file():
        parser.error(f"{source} has no SKILL.md; is this an adabmDCApy checkout with the skill?")
    found = differences(source, TARGET)
    if args.check:
        print("\n".join(found) if found else "skills/adabmdca is up to date.")
        return 1 if found else 0
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(source, TARGET)
    print(f"Copied {source} -> {TARGET} ({len(found)} difference(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
