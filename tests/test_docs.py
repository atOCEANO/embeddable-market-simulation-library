"""The documented examples must be real code against the real signatures.

An example that names a parameter the library does not have is worse than no
example: it reads as authoritative, it is what a reader copies first, and it
fails on their first attempt rather than on ours. Nothing here runs the examples,
because most of them stand on a frame and a strategy the page describes in prose;
what it checks is that every block parses and every keyword exists.
"""

import ast
import inspect
import pathlib

import emsl
from emsl import plot
from emsl.backtest import Backtester

# the gate copies the pages to /docs; a source checkout has them beside tests
_CANDIDATES = (
    pathlib.Path("/docs"),
    pathlib.Path(__file__).resolve().parent.parent / ".Documentation",
)

_API = {
    "chart": emsl.chart,
    "chart_defaults": emsl.chart_defaults,
    "to_ohlcv": emsl.to_ohlcv,
    "Backtester": Backtester,
    "Line": plot.Line,
    "Histogram": plot.Histogram,
    "Band": plot.Band,
    "Level": plot.Level,
    "Marker": plot.Marker,
    "Markers": plot.Markers,
    "Background": plot.Background,
    "Panel": plot.Panel,
    "Recorder": plot.Recorder,
    "ramp": plot.ramp,
    "at_bar": plot.at_bar,
    "at_next": plot.at_next,
}

# a bare name, or an attribute hanging off emsl or plot, is the library function.
# self.log.at_bar is Recorder.at_bar, which takes **values and shares its name
# with the module-level helper, so matching on the last word alone is wrong
_ROOTS = {"emsl", "plot"}


def pages():
    # the README counts, and counts most: it is the page nearly everybody reads
    # and the only one whose examples nobody arrives at through a guide
    for root in _CANDIDATES:
        if root.is_dir():
            found = sorted(root.glob("*.md"))
            outside = root.parent / "README.md"
            if outside.is_file():
                found.append(outside)
            return found
    return []


def blocks(text):
    out, buf, inside = [], [], False
    for line in text.splitlines():
        if line.startswith("```python"):
            inside, buf = True, []
        elif line.startswith("```") and inside:
            out.append("\n".join(buf))
            inside = False
        elif inside:
            buf.append(line)
    return out


def called(func):
    if isinstance(func, ast.Name):
        return func.id
    if not isinstance(func, ast.Attribute):
        return None
    base = func.value
    if isinstance(base, ast.Name) and base.id in _ROOTS:
        return func.attr
    if (isinstance(base, ast.Attribute) and base.attr in _ROOTS
            and isinstance(base.value, ast.Name) and base.value.id in _ROOTS):
        return func.attr
    return None


def test_the_documentation_has_pages_to_check():
    assert pages(), "no .Documentation found in the gate or beside tests"


def test_every_documented_example_parses():
    broken = []
    for page in pages():
        for body in blocks(page.read_text(encoding="utf-8")):
            try:
                ast.parse(body)
            except SyntaxError as exc:
                first = (body.strip().splitlines() or [""])[0]
                broken.append(f"{page.name}: {exc.msg} near {first[:60]!r}")
    assert not broken, "\n".join(broken)


def test_every_documented_example_names_real_parameters():
    wrong = []
    for page in pages():
        for body in blocks(page.read_text(encoding="utf-8")):
            try:
                tree = ast.parse(body)
            except SyntaxError:
                continue                      # reported by the test above
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = _API.get(called(node.func))
                if target is None:
                    continue
                allowed = set(inspect.signature(target).parameters)
                for keyword in node.keywords:
                    if keyword.arg is not None and keyword.arg not in allowed:
                        wrong.append(
                            f"{page.name}: {called(node.func)}"
                            f"({keyword.arg}=) is not a parameter"
                        )
    assert not wrong, "\n".join(wrong)
