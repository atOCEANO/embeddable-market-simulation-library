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
import re

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
playwright = pytest.importorskip("playwright.sync_api")

import emsl
from emsl.plot import Background, Band, Level, Line, Marker, Panel


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


# the legend and every caption are painted on canvas, so there is no text to
# read: the drawing calls are recorded instead. Installed before the document
# runs, because the first paint happens on mount
_RECORDER = """
window.__text = [];
window.__at = [];
const real = CanvasRenderingContext2D.prototype.fillText;
CanvasRenderingContext2D.prototype.fillText = function (s, x, y) {
  window.__text.push(String(s));
  window.__at.push([String(s), y]);
  return real.apply(this, arguments);
};
"""

# the widest canvas is a pane's plot area; the price axis has its own, narrower
_WIDEST = """
() => {
  let best = null;
  document.querySelectorAll('#chart canvas').forEach(c => {
    const r = c.getBoundingClientRect();
    if (!best || r.width > best.width) best = {
      left: r.left, top: r.top, width: r.width, height: r.height };
  });
  return best;
}
"""

_STAMP = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d")


def stamp_under(page, at):
    """The bar the legend reports with the pointer ``at`` across the plot.

    Reading the chart the way a person does, which is the only way to ask it what
    it is showing: the bundle runs inside an IIFE and nothing in it is reachable.
    """
    box = page.evaluate(_WIDEST)
    page.evaluate("window.__text = []")
    page.mouse.move(box["left"] + box["width"] * at, box["top"] + box["height"] * 0.5)
    page.wait_for_timeout(250)
    found = [s for s in page.evaluate("window.__text") if _STAMP.fullmatch(s)]
    return found[0] if found else None


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
            # the bar index moved out of the cell and into its title when the
            # column started carrying a timestamp, and it is what focus= and every
            # guard message are phrased in, so it is worth holding on to
            "titles": page.locator("#tbody tr").evaluate_all(
                "rs => rs.map(r => Array.from(r.cells).map(c => c.getAttribute('title')))"
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
    # the two bar columns read as a time, because the axis over them does and a
    # row saying 4856 under an axis saying November has to be counted back by
    # hand. The tick is still reachable, in the cell's title
    for row, title, trade in zip(seen["rows"], seen["titles"], result.trades):
        assert row[1] == ("buy" if trade["side"] == "buy" else "sell")
        assert row[2] == candles.index[trade["entry_tick"]].strftime("%Y-%m-%d %H:%M")
        assert row[3] == candles.index[trade["exit_tick"]].strftime("%Y-%m-%d %H:%M")
        assert title[2] == "bar " + str(trade["entry_tick"])
        assert title[3] == "bar " + str(trade["exit_tick"])
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


def test_clicking_the_chart_at_a_trade_reveals_its_row(tmp_path):
    # the other half of the two-way link, and the untested one. You do not click
    # the arrow: it is painted on canvas and the handler fires for a click anywhere
    # in that bar's column, so the x comes from the axis rather than from the glyph
    candles = frame(24)
    result = run(candles)
    assert len(result.trades) > 0
    entry = result.trades[0]["entry_tick"]
    bars = len(candles)

    path = emsl.chart(candles, result).save(str(tmp_path / "click.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(500)
        # the widest canvas is the pane's plot area; the price axis has its own
        box = page.evaluate(
            """
            () => {
              let best = null;
              document.querySelectorAll('#chart canvas').forEach(c => {
                const r = c.getBoundingClientRect();
                if (!best || r.width > best.width) best = {
                  left: r.left, top: r.top, width: r.width, height: r.height };
              });
              return best;
            }
            """
        )
        x = box["left"] + (entry + 0.5) * box["width"] / bars
        y = box["top"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.click(x, y)
        page.wait_for_timeout(300)
        seen = {
            "open": page.evaluate("!document.getElementById('tablePanel').hidden"),
            "selected": page.locator("tr.sel").get_attribute("data-n"),
            "hint": page.locator("#hint").text_content(),
        }
        browser.close()

    # the table opening without anyone touching #tbl is the signature of this path
    assert seen["open"]
    assert seen["selected"] == "1"
    assert seen["hint"].startswith("trade 1")


def test_a_legend_too_wide_for_its_panel_counts_what_it_dropped(tmp_path):
    # the legend is painted on canvas, so there is no text to read: the drawing
    # calls are recorded instead. A series it cannot fit has to be counted, because
    # a legend that quietly drops one lies about what is plotted
    candles = frame()
    close = candles["close"].to_numpy()
    marks = [
        Line(close - 100.0 + i, f"a rather long series name number {i}", panel="crowded")
        for i in range(8)
    ]
    built = emsl.chart(candles, marks, panels=[Panel("crowded", weight=0.35)])
    path = built.save(str(tmp_path / "legend.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 620, "height": 700})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(500)
        page.evaluate("window.__text = []")
        page.mouse.move(300, 500)
        page.mouse.move(310, 505)
        page.wait_for_timeout(400)
        drawn = page.evaluate("window.__text")
        browser.close()

    counters = [s for s in drawn if re.fullmatch(r"\+\d+", s)]
    assert counters, f"nothing was counted as dropped; drew {drawn[:20]}"
    named = sum(1 for s in drawn if "long series name" in s)
    assert named + int(counters[0][1:]) >= 8


def test_a_gap_in_a_series_leaves_the_legend_quiet_rather_than_reporting_a_fault(tmp_path):
    # the legend and the trade table share one formatter and it answers "n/a" for a
    # value that is not there. That is right in a table cell, where a column has to
    # line up and a blank one reads as data that went missing on the way. Over the
    # candles the same three characters read as a broken chart, on the bars where
    # the honest answer is that there was no stop.
    #
    # The cursor sits on the last bar until something moves it, so a series whose
    # tail is a gap is in that state the moment it mounts and needs no gesture
    candles = frame()
    stop = candles["close"].to_numpy() - 5.0
    stop[30:] = np.nan
    built = emsl.chart(candles, Line(stop, "trailing stop"))
    path = built.save(str(tmp_path / "gap.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 700})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(600)
        drawn = page.evaluate("window.__text")
        browser.close()

    assert "trailing stop" in drawn, f"the row is not there at all; drew {drawn[:20]}"
    assert "n/a" not in drawn, "the legend reported a gap as a missing value"


def test_a_dense_chart_is_aggregated_without_moving_what_the_crosshair_reads(tmp_path):
    # a year of hourly candles into a notebook cell is a fifth of a pixel a bar,
    # and drawing every one of them is a smear with no OHLC left in it. The
    # renderer aggregates to the pixel below one device pixel of bar spacing,
    # which cut the canvas work on the flagship chart from 105,186 fillRect calls
    # to 13,206 and made the year legible.
    #
    # The hazard it brings is this test's subject. legend.js reads SPEC by logical
    # index while the candle beside it is now a group of bars, so if the renderer
    # started answering with a conflated index the legend would report a bar
    # nowhere near the one under the pointer, quietly and on every dense chart
    n = 4000
    candles = frame(n)
    stamps = [t.strftime("%Y-%m-%d %H:%M") for t in candles.index]
    path = emsl.chart(candles).save(str(tmp_path / "dense.html"))

    read = {}
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 900, "height": 600})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(700)
        for at in (0.1, 0.5, 0.9):
            read[at] = stamp_under(page, at)
        browser.close()

    # the whole series is fitted, so the fraction across the plot is the fraction
    # through the bars. Two percent of the span is the tolerance, which is far
    # tighter than the aggregation and far looser than a rounding
    for at, printed in read.items():
        assert printed, f"no bar was stamped at {at}; nothing reached the legend"
        landed = stamps.index(printed)
        assert abs(landed - at * n) < n * 0.02, (
            f"the pointer was {at:.0%} across and the legend said bar {landed} of {n}"
        )


def test_notes_reach_the_panel_under_the_plot_and_survive_angle_brackets(tmp_path):
    # what stops a saved file carrying a claim and not the evidence for it. Every
    # cell is written with textContent, so the second row here is a cell rather
    # than a decision anybody has to think about (ADR 0113)
    candles = frame()
    built = emsl.chart(candles, run(candles), notes=[
        ["window", "traded", "sharpe"],
        ["1", "<script>alert(1)</script>", 1.94],
        ["2", "2024-01-02 to 2024-01-03", -0.31],
    ])
    path = built.save(str(tmp_path / "notes.html"))
    errors = []
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(400)
        page.click("#tbl")
        page.wait_for_timeout(300)
        seen = {
            "head": page.locator("#nhead th").all_text_contents(),
            "rows": page.locator("#nbody tr").evaluate_all(
                "rs => rs.map(r => Array.from(r.cells).map(c => c.textContent))"
            ),
            "scripts": page.locator("#nbody script").count(),
            # both tables are open together, behind the one button
            "trades_shown": page.evaluate("!document.getElementById('trades').hidden"),
            "notes_shown": page.evaluate("!document.getElementById('notes').hidden"),
        }
        browser.close()

    assert errors == []
    assert seen["head"] == ["window", "traded", "sharpe"]
    assert seen["rows"][0][1] == "<script>alert(1)</script>"
    assert seen["rows"][1][2] == "-0.31"
    assert seen["scripts"] == 0
    assert seen["trades_shown"] and seen["notes_shown"]


def test_notes_keep_the_table_button_on_a_chart_that_never_traded(tmp_path):
    # the button is hidden when there are no fills, because opening an empty
    # trade log reads as a chart that lost its data. A chart with notes and no
    # run has something to show and had no way to show it
    built = emsl.chart(frame(), notes=[["what", "value"], ["bars", 60]])
    path = built.save(str(tmp_path / "onlynotes.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(400)
        hidden = page.evaluate("document.getElementById('tbl').hidden")
        page.click("#tbl")
        page.wait_for_timeout(300)
        seen = {
            "rows": page.locator("#nbody tr").count(),
            "trades_shown": page.evaluate("!document.getElementById('trades').hidden"),
        }
        browser.close()

    assert not hidden, "the button stayed hidden, so the notes were unreachable"
    assert seen["rows"] == 1
    assert not seen["trades_shown"], "an empty trade log opened alongside them"


def test_fit_puts_the_whole_series_back_after_a_row_framed_one_trade(tmp_path):
    # FIT had never been pressed by anything. It is the one control that undoes
    # every other gesture, so a chart whose FIT is broken is a chart a reader can
    # get lost in with no way back
    candles = frame(400)
    result = run(candles)
    assert len(result.trades) > 4
    stamps = [t.strftime("%Y-%m-%d %H:%M") for t in candles.index]

    path = emsl.chart(candles, result).save(str(tmp_path / "fit.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1100, "height": 700})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(500)
        whole = stamp_under(page, 0.02)

        # clicking a row frames that trade, which is the zoom being undone
        page.click("#tbl")
        page.click('#tbody tr[data-n="4"]')
        page.wait_for_timeout(500)
        framed = stamp_under(page, 0.02)

        page.click("#fit")
        page.wait_for_timeout(500)
        back = stamp_under(page, 0.02)
        browser.close()

    # two percent across a fitted 400 bar chart is bar 8, not bar 0, so this is
    # about where the viewport sits rather than about an exact bar
    assert stamps.index(whole) < len(stamps) * 0.05, "the chart did not open fitted"
    assert framed != whole, "clicking the row framed nothing, so FIT is untested"
    assert back == whole, "FIT left the chart where the row had put it"


def test_the_wheel_zooms_the_time_axis(tmp_path):
    # the renderer's own handler, which a synthetic event never reaches and no
    # test had ever driven with a real one
    candles = frame(400)
    path = emsl.chart(candles).save(str(tmp_path / "wheel.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1100, "height": 700})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(500)
        before = stamp_under(page, 0.02)

        box = page.evaluate(_WIDEST)
        page.mouse.move(box["left"] + box["width"] * 0.5,
                        box["top"] + box["height"] * 0.5)
        for _ in range(6):
            page.mouse.wheel(0, -240)
            page.wait_for_timeout(80)
        page.wait_for_timeout(400)
        after = stamp_under(page, 0.02)
        browser.close()

    stamps = [t.strftime("%Y-%m-%d %H:%M") for t in candles.index]
    assert before is not None and after is not None, "the legend stopped reporting"
    # zooming in at the middle walks the left edge forward through the series, so
    # the bar two percent across the plot is a later one than it was
    assert stamps.index(after) > stamps.index(before), "the wheel moved nothing"


def test_dragging_a_pane_separator_resizes_the_panes_and_moves_their_controls(tmp_path):
    # the separator drag fires no event of its own, which is why controls.js
    # catches it on mouseup; that arrangement had never been exercised, so a
    # dragged chart could have left its per-pane buttons behind
    candles = frame(200)
    built = emsl.chart(candles, run(candles))
    path = built.save(str(tmp_path / "drag.html"))
    tops = "gs => gs.map(g => g.style.top)"
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#paneCtls .panectl", timeout=20_000)
        page.wait_for_timeout(600)
        before = page.locator("#paneCtls .panectl").evaluate_all(tops)

        # the separator sits in the gap between two pane canvases
        gap = page.evaluate(
            """
            () => {
              const rs = Array.from(document.querySelectorAll('#chart canvas'))
                .map(c => c.getBoundingClientRect())
                .filter(r => r.width > 200)
                .sort((a, b) => a.top - b.top);
              if (rs.length < 2) return null;
              return { x: rs[0].left + rs[0].width / 2,
                       y: (rs[0].bottom + rs[1].top) / 2 };
            }
            """
        )
        assert gap is not None, "the fixture has only one pane to drag between"
        page.mouse.move(gap["x"], gap["y"])
        page.mouse.down()
        page.mouse.move(gap["x"], gap["y"] - 90, steps=12)
        page.mouse.up()
        page.wait_for_timeout(600)
        after = page.locator("#paneCtls .panectl").evaluate_all(tops)
        browser.close()

    assert len(before) == len(after) > 1
    assert before != after, "the drag changed nothing, or the controls did not follow"


def test_the_candles_put_no_badge_on_the_price_axis(tmp_path):
    # every other series in the file turned its last value badge off and the
    # candles were the one that did not, which read as the omission it was. It
    # takes the candle's own colour, so in the light theme it is a near black
    # block covering an axis label rather than replacing one, and it says nothing
    # new: at rest the cursor is the last bar, so the legend's own C is the same
    # number, and while the pointer moves the crosshair labels the axis itself.
    #
    # Prices are scaled up because the two formats only differ above a thousand:
    # the badge goes through the series price format and carries no separator,
    # while the legend's C is a locale string and does
    candles = frame() * 1000.0
    built = emsl.chart(candles)
    digits = built.spec()["panels"][0]["digits"]
    last = float(candles["close"].to_numpy()[-1])
    badge = f"{last:.{digits}f}"
    legend = f"{last:,.{digits}f}"
    assert badge != legend, "the fixture cannot tell the two formats apart"

    path = built.save(str(tmp_path / "badge.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1000, "height": 600})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(600)
        drawn = page.evaluate("window.__text")
        browser.close()

    assert legend in drawn, "the legend stopped reporting the close as well"
    assert badge not in drawn, "the badge is still stamped on the axis"


def test_a_caption_near_the_top_moves_rather_than_printing_through_the_legend(tmp_path):
    # the legend owns the top of every pane and reserves it through the price
    # scale's top margin. A caption is placed from a price and knew nothing about
    # that reservation, so a Marker with a large offset anchored near the high
    # printed its words straight across the OHLC row
    candles = frame()
    high = candles["high"].to_numpy()
    peak = int(np.argmax(high))
    built = emsl.chart(
        candles,
        Marker(peak, value=float(high[peak]), text="the high", shape="arrow_down",
               offset=30),
    )
    path = built.save(str(tmp_path / "caption.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1000, "height": 600})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(600)
        at = page.evaluate("window.__at")
        browser.close()

    ys = {}
    for text, y in at:
        ys.setdefault(text, y)
    assert "the high" in ys, f"the caption was not drawn; drew {list(ys)[:12]}"
    # the panel's own name is the first thing the legend paints, so its baseline
    # is where the band the legend owns is
    assert "price" in ys
    assert ys["the high"] > ys["price"], (
        "the caption landed at or above the legend it is supposed to clear"
    )


def test_the_axis_drops_a_grain_the_span_does_not_deserve_and_keeps_one_it_does(tmp_path):
    # the renderer weighs every tick on its own and will straddle two of its own
    # thresholds, so 1920 pixels of a two month chart read 9, 17, 12:00, Sept, 9,
    # 17, Oct: one intraday tick between two day numbers, which is noise.
    #
    # Both directions, because suppressing is the easy half and the expensive
    # mistake is taking away detail that was doing its job. Three days of hourly
    # candles want their hours, and an empty label is how a tick is dropped, so
    # its presence is what says the formatter is engaged at all (ADR 0111)
    def labels(candles, name):
        path = emsl.chart(candles).save(str(tmp_path / name))
        page = browser.new_page(viewport={"width": 1280, "height": 700})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(600)
        drawn = page.evaluate("window.__text")
        page.close()
        return drawn

    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        # 3000 hourly bars is four months, past every threshold in the rule
        wide = labels(frame(3000), "wide.html")
        near = labels(frame(72), "near.html")
        browser.close()

    clock = re.compile(r"\d\d:\d\d")
    assert not [s for s in wide if clock.fullmatch(s)], (
        "a time of day survived on a four month axis"
    )
    assert "" in wide, "nothing was suppressed, so the formatter never ran"
    assert [s for s in near if clock.fullmatch(s)], (
        "three days of hourly candles lost the hours they are about"
    )


def test_auto_opens_on_the_scheme_the_reader_asked_for(tmp_path):
    # one saved file, opened twice by readers whose machines disagree. The palette
    # for both has always been in the document; nothing was asking (ADR 0109).
    #
    # The button is checked in the same breath because it names the mode it will
    # switch TO, and it was named once in the markup and then only on a click, so
    # a chart that opened light announced LIGHT and went on announcing it
    path = emsl.chart(frame(), theme="auto").save(str(tmp_path / "auto.html"))
    seen = {}
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        for scheme in ("dark", "light"):
            page = browser.new_page(
                viewport={"width": 900, "height": 600}, color_scheme=scheme
            )
            page.goto(pathlib.Path(path).as_uri())
            page.wait_for_selector("#chart canvas", timeout=20_000)
            page.wait_for_timeout(400)
            seen[scheme] = (
                page.evaluate("document.documentElement.getAttribute('data-theme')"),
                page.locator("#theme").text_content(),
            )
            page.close()
        browser.close()

    assert seen["dark"] == ("dark", "LIGHT")
    assert seen["light"] == ("light", "DARK")


def test_a_named_background_names_the_region_under_the_crosshair(tmp_path):
    # a three-regime shading was three washes and a guess, and the workaround in
    # the wild was stacked Level calls, which draws a horizontal line to label a
    # vertical region (ADR 0110)
    candles = frame()
    labels = np.where(np.arange(len(candles)) % 20 < 10, "calm", "wild")
    built = emsl.chart(
        candles,
        Background(labels, "regime",
                   fill={"calm": "#2fe0a822", "wild": "#ff547022"}),
    )
    path = built.save(str(tmp_path / "regime.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 700})
        page.add_init_script(_RECORDER)
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(600)
        drawn = page.evaluate("window.__text")
        browser.close()

    # the cursor sits on the last bar until something moves it, and bar 59 is 19
    # into its window of twenty, which is the second region
    assert "regime" in drawn, f"the row is not there at all; drew {drawn[:20]}"
    assert "wild" in drawn
    assert "calm" not in drawn


def test_the_log_button_toggles_its_panel_and_says_so(tmp_path):
    # the controls are invisible until the pointer is inside their pane's band, so
    # a plain click fails the actionability check: the hover is part of the feature
    candles = frame()
    built = emsl.chart(candles, run(candles))
    path = built.save(str(tmp_path / "log.html"))
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#paneCtls .panectl", timeout=20_000)
        page.wait_for_timeout(400)
        group = page.locator("#paneCtls .panectl").first
        log = group.locator('[data-act="log"]')

        def aim():
            # re-read every time: the handler calls placePaneCtls, which can move
            # the group, and hovering is what makes it reachable at all
            spot = log.bounding_box()
            x, y = spot["x"] + spot["width"] / 2, spot["y"] + spot["height"] / 2
            page.mouse.move(x, y)
            page.wait_for_timeout(250)
            return x, y

        x, y = aim()
        # the hover is part of the feature, so it is asserted rather than worked
        # around: the group is invisible and untouchable until the pointer is
        # inside its pane's band
        assert group.get_attribute("data-active") == "true"
        assert page.evaluate(
            "([x, y]) => (document.elementFromPoint(x, y) || {}).tagName", [x, y]
        ) == "BUTTON"

        # real coordinates rather than locator.click, whose actionability check
        # races the very class that makes the button reachable
        before = log.get_attribute("aria-pressed")
        page.mouse.click(x, y)
        page.wait_for_timeout(250)
        after = log.get_attribute("aria-pressed")
        percent = group.locator('[data-act="pct"]').get_attribute("aria-pressed")
        x, y = aim()
        page.mouse.click(x, y)
        page.wait_for_timeout(250)
        again = log.get_attribute("aria-pressed")
        browser.close()

    assert before == "false" and after == "true" and again == "false"
    # log and percent are two modes of one scale, never both
    assert percent == "false"


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
                // below the legend band, because the legend paints a swatch in
                // the series' own colour and this scan cannot tell a swatch from
                // the line. It went unnoticed while the legend was narrow enough
                // to sit left of the hole; a wider type scale slid that swatch
                // into the hole's own columns and split the empty run in two,
                // failing a chart that draws the gap perfectly
                for (let y = Math.floor(c.height * 0.25); y < c.height; y++) {
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


def test_a_level_on_the_engines_own_panels_is_actually_drawn(tmp_path):
    # ADR 0075 moved a Level onto frameFor so it could hang on the candles, and
    # left the engine's two panels behind: the equity and drawdown curves are
    # pushed into SERIES after the level loop ran, so a Level there found no
    # series to scale by and painted nothing at all (ADR 0097)
    candles = frame()
    result = run(candles)
    built = emsl.chart(
        candles,
        [Level(float(result.initial), "start", panel="equity", color="#ff00ff")],
        result,
    )
    assert drawn_colours(built, tmp_path, "level-equity.html", [255, 0, 255]) > 50


def test_a_level_on_the_drawdown_panel_is_actually_drawn(tmp_path):
    # halfway down this run's own worst fall, so the reference is inside the panel
    # it is drawn on: a price line off the visible range paints nothing either,
    # and that would pass for the defect without testing it
    candles = frame()
    result = run(candles)
    depth = result.stats["max_drawdown_pct"]
    assert depth > 0.0
    built = emsl.chart(
        candles,
        [Level(-depth / 2.0, "watch", panel="drawdown", color="#ff00ff")],
        result,
    )
    assert drawn_colours(built, tmp_path, "level-drawdown.html", [255, 0, 255]) > 50


def test_an_equity_panel_asked_for_as_a_percent_mounts_clean(tmp_path):
    # the percent axis bases itself on the track's first value, which is the
    # opening balance now rather than the first advance (ADR 0098). The labels are
    # painted into a canvas and cannot be read back, so what a browser can add
    # here is that the documented composition still mounts and paints
    candles = frame()
    result = run(candles)
    built = emsl.chart(
        candles, result,
        panels=[Panel("equity", weight=2.0, scale="percent")],
    )
    seen = observe(built, tmp_path, "percent-equity.html")
    assert seen["errors"] == []
    assert seen["canvases"] > 0
    assert max(seen["colours"]) > 20


def test_the_room_asked_for_with_future_is_on_screen_on_first_paint(tmp_path):
    # fitContent fits to the data, and the projected times carry whitespace, so a
    # projection drawn by a primitive framed to the last real bar and sat off the
    # right edge with nothing on screen saying it was there (ADR 0099). The band
    # here exists ONLY past the last candle, so any pixel of it is proof the
    # viewport reached that far
    candles = frame(40)
    ahead = 6
    n = len(candles) + ahead
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    last = float(candles["close"].iloc[-1])
    upper[len(candles):] = last + 4.0
    lower[len(candles):] = last - 4.0
    built = emsl.chart(
        candles,
        [Band(upper, lower, "projected", fill="#ff00ff")],
        future=ahead,
    )
    assert drawn_colours(built, tmp_path, "future.html", [255, 0, 255]) > 50


def test_asking_for_no_room_still_frames_on_the_data(tmp_path):
    # the reservation is conditional, so a chart with no projection must frame
    # exactly as it always did rather than growing a margin of empty bars
    candles = frame(40)
    seen = observe(emsl.chart(candles, run(candles)), tmp_path, "nofuture.html")
    assert seen["errors"] == []
    assert max(seen["colours"]) > 20


def test_a_background_span_covers_the_bars_its_mask_is_true_on(tmp_path):
    # logicalToCoordinate returns the CENTRE of a bar and a span is half open over
    # whole bars, so painting centre to centre put the shading half a candle right
    # of the bars it belongs to. The width was always correct, which is why it
    # read as right until someone looked at an edge (ADR 0101)
    n = 40
    candles = frame(n)
    mask = np.zeros(n, dtype=bool)
    mask[20:30] = True
    built = emsl.chart(candles, [Background(mask, fill="#ff00ff")])
    path = built.save(str(tmp_path / "span.html"))

    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(pathlib.Path(path).as_uri())
        page.wait_for_selector("#chart canvas", timeout=20_000)
        page.wait_for_timeout(500)
        # the first and last x of the shaded band, and the pixel width of one bar,
        # both read off the same canvas so the comparison needs no scaling
        seen = page.evaluate(
            """
            () => {
              let best = null;
              document.querySelectorAll('#chart canvas').forEach(c => {
                const r = c.getBoundingClientRect();
                if (!best || r.width > best.width) best = c;
              });
              const ctx = best.getContext('2d');
              const d = ctx.getImageData(0, 0, best.width, best.height).data;
              let lo = null, hi = null;
              for (let x = 0; x < best.width; x++) {
                let found = false;
                for (let y = 0; y < best.height; y++) {
                  const i = (y * best.width + x) * 4;
                  if (d[i] > 150 && d[i + 1] < 100 && d[i + 2] > 150) { found = true; break; }
                }
                if (found) { if (lo === null) lo = x; hi = x; }
              }
              return {lo: lo, hi: hi, width: best.width, ratio: window.devicePixelRatio};
            }
            """
        )
        browser.close()

    assert seen["lo"] is not None, "the span painted nothing at all"
    bar = seen["width"] / n
    # the tolerance has to be tighter than the error it is looking for: the defect
    # is exactly half a bar, so anything from a quarter of one upward cannot tell
    # the two apart and passes either way
    assert (seen["hi"] - seen["lo"]) / bar == pytest.approx(10.0, abs=0.2)
    assert seen["lo"] / bar == pytest.approx(20.0, abs=0.2)
