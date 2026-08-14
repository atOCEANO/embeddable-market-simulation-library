"""Build every chart the documentation shows, from data that cannot move.

The pictures in the guide are the output of the snippets printed beside them, so
this is where those snippets run for real. No simulation logic lives here and
nothing is drawn by hand: it loads a frozen parquet, works the features out the
way a reader would, and calls ``emsl.chart``.

The frozen part is the point (ADR 0085). The chart on the front page of the
README used to come from a year of live candles pulled off the router, so a
rerun drew a different year and the picture nobody could reproduce was the one
nearly everybody sees. Every chart here reads the same published five-minute
parquet, so a rerun draws the same picture on any machine for as long as the file
exists.

Keep the documented examples character for character the same as the guide. That
is the whole contract between a snippet and the picture under it, and it is worth
being fussy about. Run it through the charts stage, which also takes the shots:

  docker build --target charts -t emsl-charts .
  docker run --rm -v "<sample-market-data>/data:/data:ro" \
      -v "${PWD}/.Documentation:/out" emsl-charts
"""

from __future__ import annotations

import json
import pathlib

import numpy
import pandas

import emsl
from emsl.plot import (
    Background, Band, Histogram, Level, Line, Markers, Panel, Recorder, ramp,
)

DATA = pathlib.Path("/data/binance_spot_BTCUSDT_5m.parquet")
OUT = pathlib.Path("/work/charts")

GREEN = "#2fe0a8"
RED = "#ff5470"
GREY = "#8b97a5"
BLUE = "#4d9fff"

AHEAD = 26

# the whole charts carry four or five panels over a year of bars, so they need
# the width or they read as a strip. A one-idea example is narrower for the same
# reason in reverse: at 1440 a single pane comes out a letterbox
WIDE = 1440
NARROW = 1180

SHOTS = {}


def resample_to(bars, rule):
    grouped = bars.resample(rule)
    frame = grouped.last()
    frame["open"] = grouped.open.first()
    frame["high"] = grouped.high.max()
    frame["low"] = grouped.low.min()
    frame["volume"] = grouped.volume.sum()
    return frame.dropna()


def relative_strength(close, length):
    change = close.diff()
    gain = change.clip(lower=0.0).ewm(alpha=1.0 / length, adjust=False).mean()
    loss = (-change.clip(upper=0.0)).ewm(alpha=1.0 / length, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / loss)


def vam(close, n):
    logret = numpy.log(close / close.shift(1))
    return numpy.log(close / close.shift(n)) / (logret.rolling(n).std() * numpy.sqrt(n))


def midpoint(frame, n):
    return (frame.high.rolling(n).max() + frame.low.rolling(n).min()) / 2


def keep(name, chart, width, caption):
    # the height is recorded here and read by the shooter rather than written out
    # in both, because it drifted the first time either was edited. A chart sizes
    # itself to its window, so a shot taken at a height the chart was not built
    # for is either padded with empty plot or squeezed
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(path=str(OUT / f"{name}.html"))
    SHOTS[name] = {
        "height": chart.spec()["height"],
        "width": width,
        "caption": caption,
    }


# ------------------------------------------------------------------ the data


def load():
    raw = pandas.read_parquet(DATA)
    stamped = pandas.to_datetime(raw.pop("timestamp"), unit="ms")
    return raw.set_index(stamped).sort_index()


# ------------------------------------------------------- the flagship charts


class Channel(emsl.Strategy):
    """The README's strategy: hold above the channel basis while MACD agrees.

    Every indicator is shifted a bar before it is read, so a decision on bar `i`
    sees only what closed at `i - 1` and fills at the open of `i + 1`.
    """

    def __init__(self, length, fast, slow, signal, trend):
        self.length = length
        self.fast_n = fast
        self.slow_n = slow
        self.signal_n = signal
        self.trend_n = trend

    def init(self, engine):
        close = pandas.Series(engine.data[:, 3])
        basis = close.rolling(self.length).mean()
        width = close.rolling(self.length).std()
        self.basis = basis.shift(1).to_numpy()
        self.upper = (basis + 2.0 * width).shift(1).to_numpy()
        self.lower = (basis - 2.0 * width).shift(1).to_numpy()

        # the regime gate. A breakout system is long only, so the half of the year
        # the market spends below its own long average is the half it has no
        # business trading, and the shading behind the candles is that half
        slower = close.rolling(self.trend_n).mean().shift(1)
        self.uptrend = (close.shift(1) > slower).fillna(False).to_numpy()

        macd = (close.ewm(span=self.fast_n, adjust=False).mean()
                - close.ewm(span=self.slow_n, adjust=False).mean())
        self.hist = (macd - macd.ewm(span=self.signal_n, adjust=False).mean()) \
            .shift(1).to_numpy()

        self.close = close.shift(1).to_numpy()
        self.blocked = numpy.zeros(len(close), dtype=bool)

    def next(self, state, engine):
        bar = state["tick_index"]
        if bar < self.length + 1:
            return
        # in on a break of the upper edge, out on a loss of the basis. The two
        # thresholds are deliberately different: entering and leaving on the same
        # line whipsaws on every bar that straddles it, which on hourly candles
        # was 181 trades in a year and a chart nobody could read
        if state["position"] == 0.0 and self.uptrend[bar] \
                and self.close[bar] > self.upper[bar]:
            # the break has to arrive while momentum is still building; one taken
            # into a fading impulse is what this filters, and the arrow marks it
            if self.hist[bar] > 0.0:
                engine.market_buy(engine.qty_from_weight(1.0))
            else:
                self.blocked[bar] = True
        elif state["position"] > 0.0 and self.close[bar] < self.basis[bar]:
            engine.close()

    def marks(self):
        return [
            Background(values=self.uptrend, fill="#2fe0a814"),
            Band(upper=self.upper, lower=self.lower, name=f"channel {self.length}",
                 fill="#8b97a514"),
            Line(values=self.basis, name=f"basis {self.length}", color=GREY,
                 width=1, style="dashed"),
            Markers(mask=self.blocked, shape="arrow_down", offset=22),
            Histogram(
                values=self.hist,
                name=f"MACD {self.fast_n}/{self.slow_n}/{self.signal_n}",
                panel="momentum",
                color=numpy.where(self.hist >= 0.0, GREEN, RED),
            ),
            Level(value=0.0, panel="momentum", style="dotted"),
        ]


class CloudBreak(emsl.Strategy):
    """Long above the cloud, out below its lower edge."""

    def __init__(self, top, low):
        self.top = top
        self.low = low

    def init(self, engine):
        self.log = Recorder(bars=engine)

    def next(self, state, engine):
        bar = state["tick_index"]
        close = state["bar_close"]
        edge = self.top[bar]
        above = bool(close > edge) if numpy.isfinite(edge) else False

        self.log.at_bar(state, above=above, edge=edge)

        if not numpy.isfinite(edge):
            return
        if state["position"] == 0.0 and above:
            engine.market_buy(engine.qty_from_weight(1.0))
        elif state["position"] > 0.0 and close < self.low[bar]:
            engine.close()

    def marks(self):
        return [Background(values=self.log["above"], fill=BLUE + "14")]


class ExampleCross(emsl.Strategy):
    """The plain cross the guide's examples are drawn against."""

    def __init__(self, fast, slow):
        self.fast_n = fast
        self.slow_n = slow

    def init(self, engine):
        prices = pandas.Series(engine.data[:, 3])
        self.fast = prices.rolling(self.fast_n).mean().shift(1).to_numpy()
        self.slow = prices.rolling(self.slow_n).mean().shift(1).to_numpy()

    def next(self, state, engine):
        bar = state["tick_index"]
        if bar < self.slow_n + 1:
            return
        if state["position"] == 0.0 and self.fast[bar] > self.slow[bar]:
            engine.market_buy(engine.qty_from_weight(1.0))
        elif state["position"] > 0.0 and self.fast[bar] < self.slow[bar]:
            engine.close()

    def marks(self):
        return [
            Line(values=self.fast, name=f"SMA {self.fast_n}"),
            Line(values=self.slow, name=f"SMA {self.slow_n}", style="dashed"),
        ]


def start_here(raw):
    # a calendar year of hourly candles, which is what the router call this
    # replaced asked for: `limit=8760` against a live feed, so it drew whatever
    # year it happened to be run in
    frame = resample_to(bars=raw, rule="1h").loc["2025"]

    strategy = Channel(length=96, fast=12, slow=26, signal=9, trend=480)
    result = emsl.backtest.Backtester(
        candles=frame,
        market="spot",
        quote=10_000.0,
        fee_taker=0.0005,
        fee_maker=0.0002,
        slippage_bps=2.0,
        periods_per_year=8760,
    ).run(strategy)

    held = numpy.zeros(len(frame), dtype=bool)
    for trade in result.trades:
        held[trade["entry_tick"]:trade["exit_tick"] + 1] = True

    keep("chart-backtest", emsl.chart(
        frame=frame,
        marks=strategy.marks(),
        run=result,
        candle_color=numpy.where(held, BLUE, None),
        # deliberately no `panels=`. The README prints this call and claims the
        # placement is the library's, so the claim has to be true of the picture
        # beside it: five panels, their order and their weights are all chosen here
        title="A channel breakout with a MACD filter, BTCUSDT 1h, binance spot, 2025",
        height=760,
    ), WIDE, "a channel, a momentum histogram, and the bars it was holding")

    return frame, result, strategy


def cloud_break(raw):
    frame = resample_to(bars=raw, rule="4h").loc["2024":"2025"]

    tenkan = midpoint(frame=frame, n=9)
    kijun = midpoint(frame=frame, n=26)
    base = (tenkan + kijun) / 2
    wide = midpoint(frame=frame, n=52)

    # senkou a and b belong AHEAD bars later than they were computed, so the
    # shift is done here and the array is simply longer than the frame
    lead = numpy.full(AHEAD, numpy.nan)
    senkou_a = numpy.concatenate([lead, base.to_numpy()])
    senkou_b = numpy.concatenate([lead, wide.to_numpy()])

    cloud = numpy.where(senkou_a >= senkou_b, GREEN + "33", RED + "33")

    strategy = CloudBreak(
        top=numpy.maximum(senkou_a[:len(frame)], senkou_b[:len(frame)]),
        low=numpy.minimum(senkou_a[:len(frame)], senkou_b[:len(frame)]),
    )
    result = emsl.backtest.Backtester(
        candles=frame,
        market="spot",
        quote=10_000.0,
        fee_taker=0.0005,
        periods_per_year=2190,
    ).run(strategy)

    keep("chart-projection", emsl.chart(
        frame=frame,
        marks=strategy.marks() + [
            Line(values=tenkan, name="tenkan", color=BLUE, width=1),
            Line(values=kijun, name="kijun", color=RED, width=1),
            Band(upper=senkou_a, lower=senkou_b, name="cloud", fill=cloud),
        ],
        run=result,
        panels=[
            Panel(name="equity", weight=1.6),
            Panel(name="volume", show=False),
        ],
        future=AHEAD,
        # framed on the tail, because that is the whole point of the picture. Over
        # two years of 4h candles the twenty-six projected bars are one percent of
        # the width, so the image demonstrating `future=` did not show it. The
        # integer form rather than a pair: it means the last N bars PLUS whatever
        # was projected past them, and a pair ending on the last bar caps the
        # window there and clips off exactly the projection being demonstrated
        focus=180,
        title="Ichimoku cloud break, BTCUSDT 4h, the cloud projected past the last candle",
        height=780,
    ), WIDE, "a cloud drawn twenty-six bars past the last candle")


# --------------------------------------------------- the documented examples


def documented_examples(raw):
    frame = resample_to(bars=raw, rule="1h").loc["2025-01":"2025-06"]
    close = frame.close

    rsi = relative_strength(close=close, length=14)
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    upper = mid + 2 * sd
    lower = mid - 2 * sd

    atr = (frame.high - frame.low).rolling(14).mean()
    risk_off = (atr > atr.rolling(100).mean()).to_numpy()

    fast = close.rolling(20).mean().to_numpy()
    slow = close.rolling(50).mean().to_numpy()
    crossed = numpy.zeros(len(frame), dtype=bool)
    crossed[1:] = (fast[1:] > slow[1:]) & (fast[:-1] <= slow[:-1])

    trending = (close > close.rolling(50).mean()).to_numpy()

    fast_z = vam(close=close, n=24)
    slow_z = vam(close=close, n=168)

    vam72 = vam(close=close, n=72)
    shade = ramp(values=vam72, colors=[RED, GREY, GREEN], domain=(-3.0, 3.0))
    # two sigma either way. It was 2.5, and over this window vam 72 runs -2.46 to
    # +2.75, so the "lo" half of the fill map fired on none of the 4,344 bars: the
    # picture teaching a two-colour label map could only ever show one colour
    regime = numpy.where(
        vam72 > 2.0, "hi", numpy.where(vam72 < -2.0, "lo", "flat")
    ).astype(object)

    hours = 4
    prices = close.to_numpy()
    forward = numpy.concatenate([
        prices[hours:] / prices[:-hours] - 1.0, numpy.full(hours, numpy.nan),
    ])

    strategy = ExampleCross(fast=20, slow=50)
    result = emsl.backtest.Backtester(
        candles=frame,
        market="spot",
        quote=10_000.0,
        fee_taker=0.0005,
        periods_per_year=8760,
    ).run(strategy)

    held = numpy.zeros(len(frame), dtype=bool)
    for trade in result.trades:
        held[trade["entry_tick"]:trade["exit_tick"] + 1] = True

    worst = min(result.trades, key=lambda trade: trade["net_pnl"])

    keep("ex-line", emsl.chart(
        frame=frame,
        marks=Line(values=frame.close.rolling(50).mean(), name="SMA 50"),
        height=520,
    ), NARROW, "a moving average over the candles")

    keep("ex-oscillator", emsl.chart(frame=frame, marks=[
        Line(values=rsi, name="RSI 14", panel="rsi"),
        Level(value=70.0, panel="rsi", style="dotted"),
        Level(value=30.0, panel="rsi", style="dotted"),
    ], height=620), NARROW, "an oscillator on its own panel")

    keep("ex-band", emsl.chart(
        frame=frame,
        marks=Band(upper=upper, lower=lower, name="bollinger", fill="#8b97a51a"),
        height=520,
    ), NARROW, "a band between two arrays")

    keep("ex-background", emsl.chart(
        frame=frame,
        marks=Background(values=risk_off, fill="#ff547026"),
        height=520,
    ), NARROW, "a shaded regime behind the candles")

    keep("ex-markers", emsl.chart(
        frame=frame,
        marks=Markers(mask=crossed, shape="arrow_up", offset=-16),
        height=520,
    ), NARROW, "one marker per bar a mask is true on")

    keep("ex-candle-color", emsl.chart(
        frame=frame,
        candle_color=numpy.where(trending, "#4d9fff", None),
        height=520,
    ), NARROW, "the candles tinted by a condition")

    keep("ex-panels", emsl.chart(frame=frame, marks=[
        Line(values=fast_z, name="z 24", panel="z24"),
        Line(values=slow_z, name="z 168", panel="z168"),
    ], panels=[
        Panel(name="z24", range=(-5.0, 5.0)),
        Panel(name="z168", range=(-5.0, 5.0)),
    ], height=760), NARROW, "two panels pinned to the same range")

    keep("ex-feature", emsl.chart(
        frame=frame,
        marks=[
            Line(values=vam(close=frame.close, n=72), name="vam 72", panel="vam"),
            Level(value=0.0, panel="vam", style="dotted"),
        ],
        height=620,
    ), NARROW, "your own feature beside the price")

    keep("ex-verdict", emsl.chart(
        frame=frame,
        marks=[
            Background(values=regime, fill={"hi": "#2fe0a81f", "lo": "#ff54701f"}),
            Line(values=vam72, name="vam 72", panel="vam", color=shade),
            Band(upper=vam72, lower=2.0, only="above", panel="vam",
                 fill=("#2fe0a800", "#2fe0a859")),
            Line(values=forward, name="fwd 4h", panel="fwd"),
            Level(value=0.0, panel="fwd", style="dotted"),
        ],
        candle_color=numpy.where(numpy.abs(vam72) > 2.0, "#4d9fff", None),
        title="vam 72 extremes against the next four hours",
        height=820,
    ), NARROW, "a feature judged against what came next")

    keep("ex-zones", emsl.chart(
        frame=frame,
        marks=[
            Line(values=rsi, name="RSI 14", panel="momentum"),
            Level(value=70.0, panel="momentum", style="dotted"),
            Band(upper=rsi, lower=70.0, only="above", panel="momentum",
                 fill=("#ff547000", "#ff547059")),
            Band(upper=rsi, lower=30.0, only="below", panel="momentum",
                 fill=("#2fe0a800", "#2fe0a859")),
        ],
        run=result,
        height=820,
    ), NARROW, "conditional fills where a line passes its level")

    keep("ex-held", emsl.chart(
        frame=frame,
        marks=strategy.marks(),
        run=result,
        candle_color=numpy.where(held, "#4d9fff", None),
        height=720,
    ), NARROW, "the bars the strategy was holding, tinted")

    keep("ex-focus", emsl.chart(
        frame=frame,
        marks=strategy.marks(),
        run=result,
        focus=worst,
        title=f"worst trade, net {worst['net_pnl']:,.0f}",
        height=660,
    ), NARROW, "the worst trade framed")


def main():
    raw = load()
    print(f"{len(raw):,} five-minute bars, {raw.index[0]} to {raw.index[-1]}\n")

    start_here(raw=raw)
    cloud_break(raw=raw)
    documented_examples(raw=raw)

    (OUT / "manifest.json").write_text(
        json.dumps(SHOTS, indent=2, sort_keys=True), encoding="utf-8"
    )

    for name in sorted(SHOTS):
        size = (OUT / f"{name}.html").stat().st_size
        print(f"{name:<18} {SHOTS[name]['width']:>5}x{SHOTS[name]['height']:<4} "
              f"{size / 1_000_000:5.1f} MB")

    print(f"\n{len(SHOTS)} charts written to {OUT}")


if __name__ == "__main__":
    main()
