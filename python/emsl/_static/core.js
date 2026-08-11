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
// integer and this file never learns the rule. A null is whitespace, {time} with
// no value, never a dropped row: dropping makes the neighbours adjacent and the
// renderer draws one straight segment across the hole (ADR 0038)
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
  // first sits on the panel floor and the last touches the line
  if (spec.fill) {
    const stops = spec.fill;
    return chart.addSeries(LWC.AreaSeries, Object.assign({
      lineColor: isRamp(spec.color) ? T().s1 : (spec.color || T().s1),
      bottomColor: stops[0],
      topColor: stops[stops.length - 1],
    }, shared), index);
  }

  return chart.addSeries(LWC.LineSeries, Object.assign({
    color: isRamp(spec.color) ? undefined : (spec.color || T().s1),
  }, shared), index);
};

const addHistogram = function (spec, panel, index) {
  const s = chart.addSeries(LWC.HistogramSeries, Object.assign({
    color: isRamp(spec.color) ? undefined : (spec.color || T().s4),
    base: spec.base,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: priceFormat(spec.digits),
  }, pinned(panel)), index);
  return s;
};

// per-item colour is data, so it is written into the point rather than into the
// series options, and it has to be rewritten on a theme change
const paint = function (entry) {
  const spec = entry.spec;
  const rows = track(spec);
  if (isRamp(spec.color)) {
    for (let j = 0; j < rows.length; j++) {
      const c = colorAt(spec.color, spec.i0 + j);
      if (c !== null && rows[j].value !== undefined) rows[j].color = c;
    }
  }
  entry.series.setData(rows);
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
  const carrier = [];
  spec.series.forEach(function (s) {
    const index = panelIndex(s.panel);
    const panel = spec.panels[index];
    if (s.kind === "line") {
      const made = addLine(s, panel, index);
      SERIES.push({ spec: s, series: made });
      if (!carrier[index]) carrier[index] = made;
    } else if (s.kind === "histogram") {
      const made = addHistogram(s, panel, index);
      SERIES.push({ spec: s, series: made });
      if (!carrier[index]) carrier[index] = made;
    }
  });

  spec.series.forEach(function (s) {
    if (s.kind !== "level") return;
    const index = panelIndex(s.panel);
    // muted, not axis: the axis colour is chosen to sit almost invisibly
    // against the plane, which is right for a gridline and wrong for a
    // reference the reader is meant to see
    (carrier[index] || anchors[index]).createPriceLine({
      price: s.value,
      color: s.color || T().muted,
      lineWidth: s.width,
      lineStyle: STYLE[s.style],
      axisLabelVisible: true,
      title: s.name || "",
    });
  });

  if (spec.equity) {
    const index = panelIndex("equity");
    SERIES.push({
      spec: {
        kind: "line", panel: "equity", name: "equity", color: t.s1,
        i0: spec.equity.i0, v: spec.equity.v, digits: spec.panels[index].digits,
      },
      series: chart.addSeries(LWC.AreaSeries, Object.assign({
        lineColor: t.s1, topColor: t.s1 + "44", bottomColor: t.s1 + "00", lineWidth: 2,
        priceLineVisible: false, lastValueVisible: false,
        priceFormat: priceFormat(spec.panels[index].digits),
      }, pinned(spec.panels[index])), index),
    });
  }
  if (spec.drawdown) {
    const index = panelIndex("drawdown");
    SERIES.push({
      spec: {
        kind: "line", panel: "drawdown", name: "drawdown %", color: t.loss,
        i0: spec.drawdown.i0, v: spec.drawdown.v, digits: spec.panels[index].digits,
      },
      series: chart.addSeries(LWC.AreaSeries, Object.assign({
        lineColor: t.loss, topColor: t.loss + "00", bottomColor: t.loss + "44", lineWidth: 1,
        priceLineVisible: false, lastValueVisible: false, invertFilledArea: true,
        priceFormat: priceFormat(spec.panels[index].digits),
      }, pinned(spec.panels[index])), index),
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
