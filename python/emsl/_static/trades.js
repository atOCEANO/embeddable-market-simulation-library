"use strict";

// the engine's own fills: arrows on the bars they happened on, a table, and a
// selection that runs both ways. The arrows use the native marker api, which
// stacks them above and below a bar for free; a user's Marker is a canvas
// primitive instead, because only that can express a pixel offset.

let markerApi = null;
let tradeByTime = null;

// the most trades that can carry a caption before the captions carry nothing.
// A year of hourly bars holds fifty-odd fills, and every one labelled with its
// size and its pnl turns the price panel into a wall of overlapping grey text
// that hides the candles it is annotating. The arrows always draw; the words
// arrive when you have zoomed in far enough to read them
const LABEL_AT_MOST = 12;

const visibleTrades = function () {
  const range = chart.timeScale().getVisibleLogicalRange();
  if (!range) return SPEC.trades.length;
  let seen = 0;
  SPEC.trades.forEach(function (tr) {
    if (tr.out >= range.from && tr.in <= range.to) seen += 1;
  });
  return seen;
};

const tradeMarkers = function () {
  const t = T();
  const words = visibleTrades() <= LABEL_AT_MOST;
  const out = [];
  SPEC.trades.forEach(function (tr) {
    out.push({
      time: SPEC.t[tr.in], position: "belowBar", shape: "arrowUp",
      color: t.muted, size: 1,
      text: words ? tr.side + " " + tr.size.toFixed(3) : "",
    });
    out.push({
      time: SPEC.t[tr.out], position: "aboveBar", shape: "arrowDown",
      color: tr.net >= 0 ? t.win : t.loss, size: 1,
      text: words ? (tr.net >= 0 ? "+" : "") + tr.net.toFixed(0) : "",
    });
  });
  return out;
};

const repaintTrades = function () {
  if (markerApi) markerApi.setMarkers(tradeMarkers());
};

const selectTrade = function (n, opts) {
  let hit = null;
  SPEC.trades.forEach(function (tr) { if (tr.i === n) hit = tr; });
  if (!hit) return;

  selection = [hit.in, hit.out, 0];
  document.querySelectorAll("tr.sel").forEach(function (r) { r.classList.remove("sel"); });
  const row = document.querySelector('tr[data-n="' + n + '"]');
  if (row) row.classList.add("sel");

  if (opts.reveal) {
    document.getElementById("tablePanel").hidden = false;
    document.getElementById("tbl").setAttribute("aria-pressed", "true");
    if (row) row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  if (opts.frame) {
    // pad by the trade's own length so a 2-bar scalp and a 300-bar hold both
    // land with usable context; the floor stops short trades zooming to a wall
    const pad = Math.max(18, Math.max(1, hit.out - hit.in) * 0.7);
    chart.timeScale().setVisibleLogicalRange({ from: hit.in - pad, to: hit.out + pad });
  }
  document.getElementById("hint").textContent =
    "trade " + hit.i + "  ·  " + hit.bars + " bars  ·  net " + fmt(hit.net, 2);
  invalidate();
};

const mountTrades = function () {
  const priceIndex = panelIndex(SPEC.candles.panel);
  const digits = SPEC.panels[priceIndex].digits;

  // the selection rides on the same span mechanism as any other background
  chart.panes()[priceIndex].attachPrimitive(spanFill(
    function () { return selection ? [selection] : []; },
    function () { return [T().selected]; }
  ));

  // no run, a run that never traded, or trades=False. The button is in the
  // template unconditionally, so leaving it would open an empty table with
  // headings and no rows, which reads as a chart that lost its data
  if (!SPEC.trades.length) {
    document.getElementById("tbl").hidden = true;
    return;
  }

  markerApi = LWC.createSeriesMarkers(CANDLES, tradeMarkers());

  // the captions appear and disappear as you zoom, so the count has to be
  // re-read whenever the visible range moves. Only redraw when the answer
  // actually changed, or every pan repaints every marker
  let labelled = null;
  chart.timeScale().subscribeVisibleLogicalRangeChange(function () {
    const now = visibleTrades() <= LABEL_AT_MOST;
    if (now !== labelled) {
      labelled = now;
      repaintTrades();
    }
  });

  // A bar can belong to two trades, because one can exit on the bar the next
  // enters, and that is the normal shape of a rule that is always in the market.
  // Setting each bar to one trade meant the later one overwrote the earlier and
  // clicking that bar could never reach the trade that ENDED there. Keeping the
  // first claim is the useful half: an exit is the bar you are looking at when
  // you ask what just happened (ADR 0077).
  tradeByTime = new Map();
  const claim = function (time, tr) {
    if (!tradeByTime.has(time)) tradeByTime.set(time, tr);
  };
  SPEC.trades.forEach(function (tr) { claim(SPEC.t[tr.out], tr); });
  SPEC.trades.forEach(function (tr) { claim(SPEC.t[tr.in], tr); });

  document.getElementById("tbody").innerHTML = SPEC.trades.map(function (tr) {
    return '<tr data-n="' + tr.i + '"><td>' + tr.i + '</td><td>' + tr.side +
      '</td><td>' + tr.in + '</td><td>' + tr.out +
      '</td><td>' + fmt(tr.size, 4) + '</td><td>' + fmt(tr.px_in, digits) +
      '</td><td>' + fmt(tr.px_out, digits) + '</td><td>' + fmt(tr.fees, 2) +
      '</td><td class="' + (tr.net >= 0 ? "win" : "loss") + '">' + fmt(tr.net, 2) +
      '</td><td>' + tr.bars + '</td></tr>';
  }).join("");

  chart.subscribeClick(function (p) {
    if (!p || !p.time) return;
    const hit = tradeByTime.get(p.time);
    if (hit) selectTrade(hit.i, { reveal: true });
  });

  document.getElementById("tbody").addEventListener("click", function (ev) {
    const row = ev.target.closest("tr[data-n]");
    if (row) selectTrade(Number(row.dataset.n), { frame: true });
  });
};
