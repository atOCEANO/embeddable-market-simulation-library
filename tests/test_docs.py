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
import re

import emsl
from emsl import metrics, plot, ta
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
    "tune": emsl.tune,
    "walk_forward": emsl.walk_forward,
    "Market": emsl.Market,
}
# every public function of the two modules whose examples are all calls. Listed by
# reflection rather than by hand: the hand-written dict above had gone four
# releases without gaining an entry, so the newest examples were the least checked
_API.update({name: getattr(metrics, name) for name in metrics.__all__
             if callable(getattr(metrics, name))})
_API.update({name: getattr(ta, name) for name in ta.__all__
             if callable(getattr(ta, name)) and not isinstance(getattr(ta, name), type)})

# a bare name, or an attribute hanging off one of these, is the library function.
# self.log.at_bar is Recorder.at_bar, which takes **values and shares its name
# with the module-level helper, so matching on the last word alone is wrong
_ROOTS = {"emsl", "plot", "metrics", "ta"}


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


def api_page():
    for page in pages():
        if page.name == "Python_API.md":
            return page.read_text(encoding="utf-8")
    raise AssertionError("Python_API.md is not among the pages")


def test_every_export_the_table_names_actually_exists():
    # a table row for something that is not there is the worst kind of
    # documentation: it is authoritative, it is the first thing anybody reads,
    # and it is wrong
    listed = re.findall(r"^\| `emsl\.(\w+)`", api_page(), re.M)
    assert listed, "the export table was not found"
    for name in listed:
        assert hasattr(emsl, name) or name == "sb3", (
            f"the export table names emsl.{name}, which the package does not have"
        )


def test_nothing_is_exported_without_being_documented():
    # the other direction: a name added to __all__ and never written about is a
    # feature nobody will find
    page = api_page()
    missing = [name for name in emsl.__all__ if f"`emsl.{name}`" not in page
               and f"`{name}`" not in page]
    assert not missing, f"exported but undocumented: {missing}"


def test_the_indicator_count_the_docs_claim_is_the_count_there_is():
    # "fourteen" is written in three places and would quietly rot the first time
    # one was added
    functions = [n for n in ta.__all__ if not isinstance(getattr(ta, n), type)]
    assert len(functions) == 14, (
        f"ta has {len(functions)} functions but three pages say fourteen; either "
        f"the count or the prose is now wrong"
    )
    said = [p.name for p in pages() if "fourteen" in p.read_text(encoding="utf-8")]
    assert "Python_API.md" in said and "README.md" in said
