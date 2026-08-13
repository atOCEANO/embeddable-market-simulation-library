"""The one test file that opens a chart in a real browser.

Everything in ``test_chart.py`` reads ``Chart.spec()``, because the correctness
gate has no browser and a spec assertion is the sharper test of what Python
decided (ADR 0043). What that cannot reach is the other half of the seam: whether
the shipped JavaScript parses, runs, and turns that document into a picture. This
file is the half that needs a browser, so it skips wherever one is absent and is
run by its own Docker stage rather than by the gate.

It asserts structure and not appearance. A screenshot comparison would fail on a
font, and the questions worth asking here are whether the bundle threw, whether
every panel became a pane, whether every trade became a row, and whether anything
was drawn at all.
"""

import pathlib

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
playwright = pytest.importorskip("playwright.sync_api")

import emsl
from emsl.plot import Band, Level, Line, Panel


# distinct colours per canvas: a canvas nothing drew on carries one, and a chart
# carries hundreds, so this separates "drew" from "mounted without complaint"
_COLOURS = """
() => Array.from(document.querySelectorAll('#chart canvas')).map(c => {
  const ctx = c.getContext('2d');
  if (!ctx || !c.width || !c.height) return 0;
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const seen = new Set();
  for (let i = 0; i < d.length; i += 4) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
  return seen.size;
})
"""


def frame(n=60):
    step = np.sin(np.arange(n, dtype=np.float64) / 4.0) * 5.0
    close = 100.0 + np.arange(n, dtype=np.float64) * 0.4 + step
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0 + np.arange(n, dtype=np.float64),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )


class Swing(emsl.Strategy):
    def next(self, state, engine):
        if state["tick_index"] % 12 == 1:
            engine.market_buy(1.0)
        elif state["tick_index"] % 12 == 7:
            engine.close()


def run(candles):
    return emsl.backtest.Backtester(candles).run(Swing())


def observe(built, tmp_path, name="chart.html", act=None):
    # one browser session per call, gathering everything at once: launching
    # chromium is the expensive part and a test that asked one question per launch
    # would pay it a dozen times
    path = built.save(str(tmp_path / name))
    errors = []
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
        page.on(
            "console",
            lambda m: errors.append("console: " + m.text) if m.type == "error" else None,
        )
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(400)
        if act is not None:
            act(page)
            page.wait_for_timeout(300)
        seen = {
            "errors": errors,
            "panes": page.locator("#paneCtls .panectl").count(),
            "labels": page.locator("#paneCtls .panectl").evaluate_all(
                "gs => gs.map(g => g.getAttribute('aria-label'))"
            ),
            "canvases": page.locator("#chart canvas").count(),
            "colours": page.evaluate(_COLOURS),
            "rows": page.locator("#tbody tr").evaluate_all(
                "rs => rs.map(r => Array.from(r.cells).map(c => c.textContent))"
            ),
            "title": page.locator("#title").text_content(),
            "hint": page.locator("#hint").text_content(),
            "theme": page.evaluate("document.documentElement.getAttribute('data-theme')"),
            "ink": page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--ink')"
            ),
            "table_open": page.evaluate("!document.getElementById('tablePanel').hidden"),
        }
        browser.close()
    return seen


def test_the_shipped_bundle_runs_in_a_browser_without_throwing(tmp_path):
    # the whole point of this file. Until it existed, roughly 1,250 lines of
    # shipped JavaScript had never been executed by anything in the project
    candles = frame()
    seen = observe(emsl.chart(candles, run(candles)), tmp_path)
    assert seen["errors"] == []


def test_the_renderer_draws_rather_than_mounting_an_empty_canvas(tmp_path):
    # a chart that throws nothing and paints nothing passes every other check here
    candles = frame()
    seen = observe(emsl.chart(candles, run(candles)), tmp_path)
    assert seen["canvases"] > 0
    assert max(seen["colours"]) > 20


def test_every_panel_becomes_a_pane_carrying_its_own_name(tmp_path):
    candles = frame()
    close = candles["close"].to_numpy()
    built = emsl.chart(
        candles,
        [
            Line(close, "trend"),
            Line(close - 100.0, "momentum", panel="momentum"),
            Level(0.0, panel="momentum"),
        ],
        run(candles),
        panels=[Panel("momentum", weight=1.0)],
    )
    seen = observe(built, tmp_path)
    assert seen["errors"] == []
    assert seen["panes"] == len(built.spec()["panels"])
    assert [p["name"] for p in built.spec()["panels"]] == [
        label.removesuffix(" scale") for label in seen["labels"]
    ]


def test_every_closed_trade_becomes_a_row_carrying_its_numbers(tmp_path):
    candles = frame()
    result = run(candles)
    built = emsl.chart(candles, result)
    seen = observe(built, tmp_path)
    assert seen["errors"] == []
    assert len(seen["rows"]) == len(result.trades)
    assert len(result.trades) > 0
    for row, trade in zip(seen["rows"], result.trades):
        assert row[1] == ("buy" if trade["side"] == "buy" else "sell")
        assert row[2] == str(trade["entry_tick"])
        assert row[3] == str(trade["exit_tick"])
        assert row[9] == str(trade["bars_held"])


def test_clicking_a_trade_row_selects_it_and_says_which(tmp_path):
    candles = frame()
    result = run(candles)
    seen = observe(
        emsl.chart(candles, result),
        tmp_path,
        act=lambda page: (
            page.click("#tbl"),
            page.click("#tbody tr:first-child"),
        ),
    )
    assert seen["errors"] == []
    assert seen["table_open"]
    # the row says which trade it is and the hint has to agree, which is the whole
    # of the two-way selection: the numbering is the table's own, from one
    assert seen["hint"].startswith("trade " + seen["rows"][0][0])
    assert str(result.trades[0]["bars_held"]) + " bars" in seen["hint"]


def test_the_theme_toggle_rewrites_the_palette_in_a_saved_file(tmp_path):
    # a saved chart carries both palettes so the toggle works with no process
    # behind it (ADR 0041), and the toggle writes the css variables the stylesheet
    # reads, which is what keeps the chart and its frame from drifting apart
    candles = frame()
    built = emsl.chart(candles, run(candles))
    dark = observe(built, tmp_path, name="dark.html")
    light = observe(built, tmp_path, name="light.html", act=lambda page: page.click("#theme"))
    assert dark["theme"] == "dark"
    assert light["theme"] == "light"
    assert dark["ink"].strip() != light["ink"].strip()
    assert light["errors"] == []


def test_a_title_reaches_the_document_and_survives_angle_brackets(tmp_path):
    candles = frame()
    built = emsl.chart(candles, run(candles), title="BTC <perp> 10x")
    seen = observe(built, tmp_path)
    assert seen["errors"] == []
    assert seen["title"] == "BTC <perp> 10x"


def drawn_colours(built, tmp_path, name, want):
    # how many pixels of one colour the chart actually painted. The counted colour
    # is unique to the mark under test, so zero means it was not drawn at all
    path = built.save(str(tmp_path / name))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(400)
        hits = page.evaluate(
            """
            (want) => {
              let total = 0;
              document.querySelectorAll('#chart canvas').forEach(c => {
                const ctx = c.getContext('2d');
                if (!ctx || !c.width || !c.height) return;
                const d = ctx.getImageData(0, 0, c.width, c.height).data;
                for (let i = 0; i < d.length; i += 4) {
                  if (Math.abs(d[i] - want[0]) < 40 && Math.abs(d[i+1] - want[1]) < 40
                      && Math.abs(d[i+2] - want[2]) < 40) total += 1;
                }
              });
              return total;
            }
            """,
            want,
        )
        browser.close()
    return hits


def test_a_level_alone_on_the_price_panel_is_actually_drawn(tmp_path):
    # the price line hung on whichever line or histogram reached the panel first,
    # and the candles were never a candidate, so the simplest call on the page drew
    # the level onto a whitespace anchor whose first value is null and the renderer
    # painted nothing at all (ADR 0075)
    candles = frame()
    built = emsl.chart(candles, Level(float(candles["close"].mean()), "mid", color="#ff00ff"))
    assert drawn_colours(built, tmp_path, "level.html", [255, 0, 255]) > 50


def test_a_band_alone_on_its_own_panel_is_actually_drawn(tmp_path):
    # a primitive converts prices through a series, and a panel drawn only by
    # primitives has none carrying values, so it mounted cleanly and painted
    # nothing. The anchor now takes the panel's own extent (ADR 0075)
    candles = frame()
    close = candles["close"].to_numpy()
    built = emsl.chart(
        candles,
        Band(close - 100.0 + 2.0, close - 100.0 - 2.0, "channel",
             panel="spread", fill="#00ff00"),
        panels=[Panel("spread", weight=1.0)],
    )
    assert drawn_colours(built, tmp_path, "band.html", [0, 255, 0]) > 50


def test_a_ramped_line_never_shows_the_vendored_default_colour(tmp_path):
    # a ramp passed `color: undefined` to the renderer, and its merge skips an
    # undefined key, so lightweight-charts' own #2196f3 survived: a bar the ramp
    # left uncoloured was painted library blue while the legend reported it grey.
    # The asset grep for colours cannot see that, because the literal is in the
    # vendored bundle rather than in ours (ADRs 0043, 0076)
    candles = frame()
    close = candles["close"].to_numpy()
    tint = np.where(np.arange(len(close)) % 2 == 0, "#ff00ff", None).astype(object)
    built = emsl.chart(candles, Line(close, "tinted", color=tint))
    assert drawn_colours(built, tmp_path, "ramp.html", [33, 150, 243]) == 0


def test_a_gap_is_drawn_as_whitespace_rather_than_bridged(tmp_path):
    # ADR 0038's claim is about pixels, and this is the only place it can be
    # checked as pixels. A flat line with a hole in the middle: the columns either
    # side carry the line's colour and a run of columns between them carries none,
    # which is exactly what "the line stops" means. Counted per column rather than
    # per third, because the plot area is inset by the price axis and the thirds of
    # a canvas are not the thirds of a series
    n = 60
    flat = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    values = np.full(n, 100.0)
    values[20:40] = np.nan
    built = emsl.chart(flat, Line(values, "gapped", color="#ff00ff", width=3))
    path = built.save(str(tmp_path / "gap.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(400)
        # magenta pixels per column: the line is the only thing on the chart in
        # that colour, so a column with none is a column the line does not cross
        columns = page.evaluate(
            """
            () => {
              let best = null;
              document.querySelectorAll('#chart canvas').forEach(c => {
                const ctx = c.getContext('2d');
                if (!ctx || !c.width || !c.height) return;
                const d = ctx.getImageData(0, 0, c.width, c.height).data;
                const hit = new Array(c.width).fill(0);
                for (let y = 0; y < c.height; y++) {
                  for (let x = 0; x < c.width; x++) {
                    const i = (y * c.width + x) * 4;
                    if (d[i] > 180 && d[i + 1] < 90 && d[i + 2] > 180) hit[x] += 1;
                  }
                }
                if (hit.some(v => v > 0)) best = hit;
              });
              return best;
            }
            """
        )
        browser.close()

    assert columns is not None, "the line was never drawn, so the gap proves nothing"
    drawn = [x for x, count in enumerate(columns) if count > 0]
    first, last = drawn[0], drawn[-1]
    longest, current = 0, 0
    for x in range(first, last + 1):
        current = 0 if columns[x] else current + 1
        longest = max(longest, current)
    # the hole is a third of the series, so anything close to that is the line
    # stopping; a bridged line leaves no empty column between its ends at all
    assert longest > 0.2 * (last - first), f"the gap was bridged: {longest} of {last - first}"
