"use strict";

// spec to chart. Everything drawn is a key in the document: this file decides no
// colour, no threshold and no number format, which is what lets a Python test in
// a gate with no browser assert what a chart shows (ADR 0043)

let SPEC = null;
let THEME = null;
let MODE = "dark";
let chart = null;
let anchors = [];                      // one series per panel, the coordinate frame
let redraws = [];                      // primitive invalidators
let cursor = 0;                        // bar under the crosshair
let selection = null;                  // {from, to} of the selected trade
let UI = 1;

const LWC = LightweightCharts;
const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';

const STYLE = { solid: 0, dotted: 1, dashed: 2 };
const SCALE = { linear: 0, log: 1, percent: 2 };

// the palette keys chart.css reads. Written on mount and on every toggle, so the
// stylesheet and the chart cannot drift apart
const CSS_VARS = {
  plane: "--plane", surface: "--surface", grid: "--grid", axis: "--axis",
  ink: "--ink", ink2: "--ink-2", muted: "--muted", hairline: "--hairline",
  win: "--win", loss: "--loss", s1: "--s1",
};

const T = function () {
  return THEME[MODE];
};

const invalidate = function () {
  redraws.forEach(function (f) { f(); });
};

// one sub-linear scale factor for everything: a chart three times wider does not
// want three times bigger type. Both dimensions feed it and the tighter one wins,
// since a notebook cell is wide and short and several panes stacked into 400px
// leave the labels eating the plot
const scaleFor = function (w, h) {
  const byW = 0.82 + w / 4600;
  const byH = 0.76 + h / 2300;
  return Math.max(0.82, Math.min(1.30, Math.min(byW, byH)));
};

const fmt = function (v, d) {
  return (v === null || v === undefined || Number.isNaN(v)) ? "n/a"
    : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
};

const stamp = function (i) {
  return new Date(SPEC.t[i] * 1000).toISOString().slice(0, 16).replace("T", " ");
};

// a Track carries i0 and a value array, so the alignment contract arrives as an
// integer and this file never learns the rule. A null becomes whitespace, {time}
// with no value, rather than a dropped row, because dropping would make the
// neighbours adjacent. That is necessary and it is not sufficient: the renderer
// joins straight across whitespace too, so the hole is cut in `paint` by splitting
// the track into one series per run (ADRs 0038, 0073)
const track = function (tr) {
  if (!tr) return [];
  const out = new Array(tr.v.length);
  for (let j = 0; j < tr.v.length; j++) {
    const v = tr.v[j];
    const time = SPEC.t[tr.i0 + j];
    out[j] = v === null ? { time: time } : { time: time, value: v };
  }
  return out;
};

// palette lookup. p[0] is always null, so an absent colour needs no branch here
// and no sentinel on the wire
const colorAt = function (ca, i) {
  if (!ca) return null;
  const j = i - ca.i0;
  if (j < 0 || j >= ca.k.length) return null;
  return ca.p[ca.k[j]];
};

const isRamp = function (c) {
  return c !== null && c !== undefined && typeof c === "object";
};

const priceFormat = function (digits) {
  return { type: "price", precision: digits, minMove: Math.pow(10, -digits) };
};

// a pinned range is honoured by every series on that panel rather than by the
// panel: v5 has no setVisibleRange on a price scale, and autoscaleInfoProvider is
// only consulted for a series that carries data
const pinned = function (panel) {
  if (!panel.range) return {};
  const lo = panel.range[0], hi = panel.range[1];
  return {
    autoscaleInfoProvider: function () {
      return { priceRange: { minValue: lo, maxValue: hi } };
    },
  };
};

const panelIndex = function (name) {
  for (let i = 0; i < SPEC.panels.length; i++) {
    if (SPEC.panels[i].name === name) return i;
  }
  return 0;
};


// ------------------------------------------------------------------ series

// the coordinate frame for a panel: an invisible line covering every bar, so the
// pane exists in panel order even before anything is drawn on it, levels have
// something to hang off, and a primitive has a price scale to convert against
const addAnchor = function (panel, index) {
  const s = chart.addSeries(LWC.LineSeries, Object.assign({
    color: "rgba(0,0,0,0)", lineWidth: 1,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    priceFormat: priceFormat(panel.digits),
  }, pinned(panel)), index);
  s.setData(SPEC.t.map(function (time) { return { time: time }; }));
  return s;
};

const addCandles = function (panel, index) {
  const s = chart.addSeries(LWC.CandlestickSeries, Object.assign({
    borderVisible: true, priceLineVisible: false,
    priceFormat: priceFormat(panel.digits),
  }, pinned(panel)), index);
  return s;
};

const addLine = function (spec, panel, index) {
  const shared = Object.assign({
    lineWidth: spec.width,
    lineStyle: STYLE[spec.style],
    lineType: spec.step ? LWC.LineType.WithSteps : LWC.LineType.Simple,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    priceFormat: priceFormat(spec.digits),
  }, pinned(panel));

  // a fill under a line is an area, which is a different series type rather than
  // an option on this one. Stops run bottom to top like every other fill, so the
  // first sits on the panel floor and the last touches the line. An area carries
  // one lineColor and ignores a per-point colour entirely, so a ramp here was
  // painted flat while the legend went on reporting a colour per bar; that pair is
  // refused at construction rather than resolved in favour of one half (ADR 0076)
  if (spec.fill) {
    const stops = spec.fill;
    return chart.addSeries(LWC.AreaSeries, Object.assign({
      lineColor: spec.color || T().s1,
      bottomColor: stops[0],
      topColor: stops[stops.length - 1],
    }, shared), index);
  }

  // `undefined` is not "no opinion", it is the vendored bundle's opinion: its
  // merge skips an undefined key, so the library's own default blue survives and
  // a bar the ramp left uncoloured is painted in it while the legend reports it
  // grey. That is the colour rule of ADR 0043 leaking out through a hole the
  // asset grep cannot see, since the literal lives in the vendored bundle rather
  // than in ours (ADR 0076)
  return chart.addSeries(LWC.LineSeries, Object.assign({
    color: isRamp(spec.color) ? T().muted : (spec.color || T().s1),
  }, shared), index);
};

const addHistogram = function (spec, panel, index) {
  const s = chart.addSeries(LWC.HistogramSeries, Object.assign({
    // not undefined: see addLine. The bundle's own default would show through
    color: isRamp(spec.color) ? T().muted : (spec.color || T().s4),
    base: spec.base,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: priceFormat(spec.digits),
  }, pinned(panel)), index);
  return s;
};

// the runs of real values in a row list, split at the whitespace. A run is what
// can be drawn as one continuous line; the holes between them are the gaps
const runsOf = function (rows) {
  const runs = [];
  let open = null;
  for (let j = 0; j < rows.length; j++) {
    if (rows[j].value === undefined) {
      open = null;
      continue;
    }
    if (open === null) {
      open = [];
      runs.push(open);
    }
    open.push(rows[j]);
  }
  return runs;
};

// per-item colour is data, so it is written into the point rather than into the
// series options, and it has to be rewritten on a theme change.
//
// The split is here because the renderer joins a line straight across whitespace.
// ADR 0038 says a value that does not exist is drawn as whitespace and the line
// stops, and the document says exactly that: the point carries a time and no
// value. lightweight-charts draws through it anyway, so a trailing stop that
// existed on bars 0 to 40 and again on 900 to 940 was drawn as a smooth line
// across the 860 bars where there was no stop, which is the precise lie the
// decision was written to prevent, arriving from the renderer instead of from
// the data. A break between two series is the only break it honours, so a gapped
// track is drawn as one series per run. The spec is unchanged, because the spec
// was never wrong; this is how it is painted and nothing else (ADR 0073).
const paint = function (entry) {
  const spec = entry.spec;
  const rows = track(spec);
  if (isRamp(spec.color)) {
    for (let j = 0; j < rows.length; j++) {
      const c = colorAt(spec.color, spec.i0 + j);
      if (c !== null && rows[j].value !== undefined) rows[j].color = c;
    }
  }
  if (!entry.make) {
    entry.series.setData(rows);
    return;
  }
  const runs = runsOf(rows);
  if (!entry.siblings) entry.siblings = [];
  while (entry.siblings.length < runs.length - 1) entry.siblings.push(entry.make());
  const drawn = [entry.series].concat(entry.siblings);
  for (let k = 0; k < drawn.length; k++) drawn[k].setData(runs[k] || []);
};

const paintCandles = function () {
  const tint = SPEC.candles.color;
  const rows = new Array(SPEC.n);
  for (let i = 0; i < SPEC.n; i++) {
    const o = SPEC.ohlc[i];
    const row = { time: SPEC.t[i], open: o[0], high: o[1], low: o[2], close: o[3] };
    const c = colorAt(tint, i);
    if (c !== null) {
      row.borderColor = c;
      row.wickColor = c;
      row.color = o[3] >= o[0] ? "rgba(0,0,0,0)" : c;
    }
    rows[i] = row;
  }
  CANDLES.setData(rows);
};

const paintVolume = function () {
  if (!VOLUME) return;
  const rows = track(SPEC.vol);
  for (let j = 0; j < rows.length; j++) {
    const i = SPEC.vol.i0 + j;
    rows[j].color = (SPEC.ohlc[i][3] >= SPEC.ohlc[i][0] ? T().up : T().down) + "66";
  }
  VOLUME.setData(rows);
};

let CANDLES = null;
let VOLUME = null;
let SERIES = [];                       // {spec, series} for every value-carrying mark


// ------------------------------------------------------------------- theme

const applyTheme = function (mode) {
  MODE = mode;
  const t = T();
  const root = document.documentElement;
  root.setAttribute("data-theme", mode);
  root.style.setProperty("color-scheme", mode);
  Object.keys(CSS_VARS).forEach(function (key) {
    if (t[key] !== undefined) root.style.setProperty(CSS_VARS[key], t[key]);
  });

  chart.applyOptions({
    layout: {
      background: { type: LWC.ColorType.Solid, color: t.surface },
      textColor: t.ink2,
      panes: { separatorColor: t.axis, separatorHoverColor: t.grid },
    },
    grid: { vertLines: { color: t.grid }, horzLines: { color: t.grid } },
    crosshair: {
      vertLine: { color: t.muted, labelBackgroundColor: t.axis },
      horzLine: { color: t.muted, labelBackgroundColor: t.axis },
    },
    rightPriceScale: { borderColor: t.axis },
    timeScale: { borderColor: t.axis },
  });

  CANDLES.applyOptions({
    upColor: "rgba(0,0,0,0)", downColor: t.down,
    borderUpColor: t.up, borderDownColor: t.down,
    wickUpColor: t.up, wickDownColor: t.down,
  });

  paintCandles();
  paintVolume();
  SERIES.forEach(paint);
  if (typeof repaintTrades === "function") repaintTrades();
  invalidate();
};


// -------------------------------------------------------------- primitives

// a primitive converts prices through a series, and a series with only
// whitespace has no values for the scale to fit, so the anchor is the last
// resort rather than the first choice
const frameFor = function (index) {
  const panel = SPEC.panels[index];
  if (panel.candles && CANDLES) return CANDLES;
  if (panel.volume && VOLUME) return VOLUME;
  for (let k = 0; k < SERIES.length; k++) {
    if (SERIES[k].spec.panel === panel.name) return SERIES[k].series;
  }
  return anchors[index];
};

// Give a panel's anchor the extent of whatever is drawn on it, when nothing else
// on that panel carries values.
//
// A Band, a Marker or a Level alone on its own panel is drawn entirely by
// primitives, and a primitive converts prices through a series. The anchor is the
// only series there and it holds whitespace, so the price scale has no first
// value; the renderer then skips the source before it ever asks for a range, and
// every conversion off it comes back null. Such a panel mounted cleanly, threw
// nothing, and painted nothing at all, which is the worst way for a chart to be
// wrong. It also swallowed `Panel(range=)` on that panel, since the provider is
// only consulted for a series carrying data. The anchor is fully transparent, so
// seeding it costs no pixels (ADR 0075).
const seedAnchors = function () {
  SPEC.panels.forEach(function (panel, index) {
    if (panel.candles || panel.volume) return;
    const carried = SERIES.some(function (entry) { return entry.spec.panel === panel.name; });
    if (carried) return;

    const seen = [];
    const take = function (v) { if (typeof v === "number" && isFinite(v)) seen.push(v); };
    SPEC.series.forEach(function (s) {
      if (s.panel !== panel.name) return;
      if (s.kind === "level") take(s.value);
      if (s.kind === "marker" && s.values) s.values.forEach(take);
      if (s.kind === "band") {
        if (s.v) s.v.forEach(take);
        if (s.v2) s.v2.forEach(take);
      }
    });
    if (!seen.length) return;

    const lo = Math.min.apply(null, seen);
    const hi = Math.max.apply(null, seen);
    anchors[index].setData(SPEC.t.map(function (time) {
      return { time: time, value: lo };
    }));
    // a pinned panel already carries its own provider from `pinned`, and it wins
    if (!panel.range) {
      anchors[index].applyOptions({
        autoscaleInfoProvider: function () {
          return { priceRange: { minValue: lo, maxValue: hi } };
        },
      });
    }
  });
};

// band, background and marker are the kinds no native series can draw, so they
// arrive here rather than in the series dispatch
const mountPrimitives = function () {
  const panes = chart.panes();

  SPEC.series.forEach(function (s) {
    const index = panelIndex(s.panel);
    if (s.kind === "band") {
      panes[index].attachPrimitive(bandPrimitive(frameFor(index), s));
    } else if (s.kind === "background") {
      panes[index].attachPrimitive(spanFill(
        function () { return s.spans; },
        function () { return s.fills; }
      ));
    } else if (s.kind === "marker") {
      panes[index].attachPrimitive(calloutPrimitive(frameFor(index), s));
    }
  });

  panes.forEach(function (pane, i) {
    if (SPEC.panels[i]) pane.attachPrimitive(legendPrimitive(i));
  });
};


// ------------------------------------------------------------------- mount

const mount = function (spec, root) {
  SPEC = spec;
  THEME = spec.theme;
  MODE = spec.theme.mode;
  cursor = spec.n - 1;

  const t = T();
  chart = LWC.createChart(root, {
    autoSize: true,
    layout: {
      background: { type: LWC.ColorType.Solid, color: t.surface },
      textColor: t.ink2, fontFamily: spec.theme.font || FONT, attributionLogo: false,
      panes: { separatorColor: t.axis, separatorHoverColor: t.grid, enableResize: true },
    },
    grid: { vertLines: { color: t.grid }, horzLines: { color: t.grid } },
    crosshair: {
      mode: LWC.CrosshairMode.Normal,
      vertLine: { color: t.muted, width: 1, style: LWC.LineStyle.Solid, labelBackgroundColor: t.axis },
      horzLine: { color: t.muted, width: 1, style: LWC.LineStyle.Solid, labelBackgroundColor: t.axis },
    },
    // the top margin is where the legend lives. A pane primitive draws beneath
    // the series whatever its zOrder says, so a backing behind the text does
    // not work: an equity curve and a drawdown both start at the top left of
    // their own pane and drew straight through their own label. Reserving the
    // room is what actually keeps it readable
    rightPriceScale: { borderColor: t.axis, scaleMargins: { top: 0.2, bottom: 0.08 } },
    timeScale: { borderColor: t.axis, timeVisible: true, secondsVisible: false },
  });

  // panes come into being in panel order, because the anchor for panel k is the
  // first series asking for pane k
  anchors = spec.panels.map(addAnchor);

  spec.panels.forEach(function (panel, index) {
    if (panel.candles) CANDLES = addCandles(panel, index);
    if (panel.volume) {
      VOLUME = chart.addSeries(LWC.HistogramSeries, Object.assign({
        priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false,
      }, pinned(panel)), index);
    }
  });

  SERIES = [];

  // the drawn series first, so a level can hang on one. A price line attached
  // to a series carrying only whitespace does not render, and the anchor is
  // exactly that: every Level in the library was reaching the renderer
  // correctly and drawing nothing at all
  spec.series.forEach(function (s) {
    const index = panelIndex(s.panel);
    const panel = spec.panels[index];
    if (s.kind === "line") {
      // the factory is what lets a gapped line grow a sibling per run in paint;
      // a histogram needs none, since its bars are already separate marks
      const make = function () { return addLine(s, panel, index); };
      SERIES.push({ spec: s, series: make(), make: make, siblings: [] });
    } else if (s.kind === "histogram") {
      SERIES.push({ spec: s, series: addHistogram(s, panel, index) });
    }
  });

  spec.series.forEach(function (s) {
    if (s.kind !== "level") return;
    const index = panelIndex(s.panel);
    // muted, not axis: the axis colour is chosen to sit almost invisibly
    // against the plane, which is right for a gridline and wrong for a
    // reference the reader is meant to see
    // frameFor, not a line-or-histogram carrier. The carrier it used to keep was
    // filled only by the line and histogram branches, so it never considered the
    // candles: a Level on a price panel with no other series hung on the
    // whitespace anchor, whose firstValue is null, and the renderer drew nothing
    // at all. `emsl.chart(frame, Level(110.0))` is about the simplest call on the
    // page and it drew no level (ADR 0075)
    frameFor(index).createPriceLine({
      price: s.value,
      color: s.color || T().muted,
      lineWidth: s.width,
      lineStyle: STYLE[s.style],
      axisLabelVisible: true,
      title: s.name || "",
    });
  });

  // the engine's own two curves. They carry a `make` like any other line, because
  // the gap split in `paint` only runs for an entry that has one: both are dense
  // by construction today, so this is a guard rather than a fix, and the reason to
  // write it is that the next curve added here will not be (ADRs 0073, 0075)
  const addCurve = function (name, label, track, options) {
    const index = panelIndex(name);
    const panel = spec.panels[index];
    if (!panel) return;
    const make = function () {
      return chart.addSeries(LWC.AreaSeries, Object.assign({
        priceLineVisible: false, lastValueVisible: false,
        priceFormat: priceFormat(panel.digits),
      }, options, pinned(panel)), index);
    };
    SERIES.push({
      spec: {
        kind: "line", panel: name, name: label, color: options.lineColor,
        i0: track.i0, v: track.v, digits: panel.digits,
      },
      series: make(),
      make: make,
      siblings: [],
    });
  };

  if (spec.equity) {
    addCurve("equity", "equity", spec.equity, {
      lineColor: t.s1, topColor: t.s1 + "44", bottomColor: t.s1 + "00", lineWidth: 2,
    });
  }
  if (spec.drawdown) {
    addCurve("drawdown", "drawdown %", spec.drawdown, {
      lineColor: t.loss, topColor: t.loss + "00", bottomColor: t.loss + "44",
      lineWidth: 1, invertFilledArea: true,
    });
  }

  chart.panes().forEach(function (pane, i) {
    if (spec.panels[i]) pane.setStretchFactor(spec.panels[i].weight);
    const scale = pane.priceScale("right");
    if (scale) scale.applyOptions({ mode: SCALE[spec.panels[i].scale] });
  });

  paintCandles();
  paintVolume();
  SERIES.forEach(paint);

  // the cursor travels the whole axis rather than the candles, because with
  // future= the axis is longer and a series drawn past the last candle still has
  // values to read. legend.js is what knows a bar past the candles has no OHLC
  chart.subscribeCrosshairMove(function (p) {
    const last = spec.t.length - 1;
    const i = (p && p.logical !== null && p.logical !== undefined)
      ? Math.max(0, Math.min(last, p.logical)) : spec.n - 1;
    if (i !== cursor) { cursor = i; invalidate(); }
  });

  // before the primitives, because each of them converts through a series and
  // this is what gives a primitive-only panel one that carries values
  seedAnchors();

  if (typeof mountPrimitives === "function") mountPrimitives();
  if (typeof mountTrades === "function") mountTrades();
  if (typeof mountControls === "function") mountControls();

  applyTheme(MODE);

  if (spec.focus) {
    chart.timeScale().setVisibleLogicalRange({ from: spec.focus[0], to: spec.focus[1] });
  } else {
    chart.timeScale().fitContent();
  }
};
