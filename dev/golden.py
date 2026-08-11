"""Rewrite ``tests/chart_schema.json``, the key-path golden for the chart spec.

The renderer is JavaScript and the gate has no browser, so the one thing that
cannot be asserted there is the contract between the two halves: a key renamed in
``_chart.py`` passes every Python test and draws a blank chart. The golden closes
that by recording every key path the spec carries, which makes a rename a visible
one-line diff in the same commit.

Run it whenever the spec gains, loses, or renames a key, and bump ``schema`` in
``_chart.py`` in the same commit. It imports the test module rather than
rebuilding the fixture, so the golden can never be recorded against a different
chart from the one the test checks:

    docker build --target builder -t emsl-builder .
    docker run --rm -v "${PWD}:/repo" emsl-builder sh -c \
      "pip install --no-cache-dir /wheels/*.whl numpy pandas pyarrow pytest \
       && python /repo/dev/golden.py"

No arguments, no options. It prints what changed.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"


def main():
    sys.path.insert(0, str(TESTS))
    import test_chart

    path = TESTS / "chart_schema.json"
    before = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    after = sorted(set(test_chart.key_paths(test_chart.everything().spec())))

    added = [k for k in after if k not in before]
    dropped = [k for k in before if k not in after]
    if not added and not dropped:
        print(f"{path.name} is already current, {len(after)} key paths")
        return 0

    path.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
    for key in added:
        print(f"  + {key}")
    for key in dropped:
        print(f"  - {key}")
    print(f"{path.name}: {len(before)} key paths -> {len(after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
