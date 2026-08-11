"""The README's install commands must name the version that is actually shipping.

A tag hardcoded into an install line is correct on the day of a release and wrong
the day after, and nothing catches it: the old URL still resolves, to an old
wheel, so the reader installs a version behind and no error is raised anywhere.
This turns that into a failing build at the moment the version is bumped.
"""

import pathlib
import re

import emsl

# the gate copies it beside the docs; a source checkout has it above tests
_CANDIDATES = (
    pathlib.Path("/README.md"),
    pathlib.Path(__file__).resolve().parent.parent / "README.md",
)

# every place a release tag can hide in an install command
_PINNED = re.compile(r"(?:expanded_assets/|releases/download/|\.git@)v(\d+\.\d+\.\d+)")


def readme():
    for path in _CANDIDATES:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def test_the_readme_is_where_the_tests_can_see_it():
    assert readme() is not None, "no README.md in the gate or above tests"


def test_every_pinned_version_in_the_readme_is_the_one_shipping():
    found = sorted(set(_PINNED.findall(readme())))
    assert found, "no pinned version found; the install commands may have moved"
    assert found == [emsl.__version__], (
        f"README pins {found} and the package is {emsl.__version__}; "
        f"bump the install commands in the same commit as the version"
    )
