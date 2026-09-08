"use strict";

// the engine's own fills: arrows on the bars they happened on, a table, and a
// selection that runs both ways. The arrows use the native marker api, which
// stacks them above and below a bar for free; a user's Marker is a canvas
// primitive instead, because only that can express a pixel offset.

let markerApi = null;
let tradeByTime = null;

// a year of hourly bars holds fifty-odd fills, and every one labelled with its
// size and its pnl turns the price panel into a wall of overlapping grey text
// that hides the candles it is annotating. The arrows always draw; the words
// arrive when you have zoomed in far enough to read them.
//
// Far enough is a question about pixels. Counting the trades was the first answer
// and a count cannot tell twelve fills spread over five months, which read fine,
// from twelve inside one month, which are a pile: every collision this guard
// exists to prevent was happening at or under the old ceiling of twelve. What
// decides it is the gap between neighbouring labels. Entries and exits are
// measured apart, because one caption sits below its bar and the other above, so
// each only ever collides inside its own band.
const CAPTION_GAP = 80;
const CAPTION_CEILING = 60;

// the caption over a bar and the row in the table are one number seen twice, so
// the two precisions are declared together and the short one abbreviates the long
// one instead of being a second opinion about it. They were four scattered
// literals, half toFixed and half toLocaleString, so a caption read -246 beside a
// row reading -245.54 and a five-figure pnl lost its separator on the way up
const TABLE_DP = { size: 4, money: 2 };
const CAPTION_DP = { size: 3, money: 0 };

const captionsFit = function () {
  const ts = chart.timeScale();
  const range = ts.getVisibleLogicalRange();
  if (!range) return false;

  const entries = [], exits = [];
  SPEC.trades.forEach(function (tr) {
    if (tr.in >= range.from && tr.in <= range.to) entries.push(tr.in);
    if (tr.out >= range.from && tr.out <= range.to) exits.push(tr.out);
  });
  // no amount of spacing rescues this many, and the coordinate lookups below are
  // a call into the renderer apiece
  if (entries.length + exits.length > CAPTION_CEILING) return false;

  const gap = CAPTION_GAP * UI;
  const crowded = function (bars) {
    let last = null;
    for (let k = 0; k < bars.length; k++) {
      const x = ts.logicalToCoordinate(bars[k]);
      if (x === null) continue;
      if (last !== null && x - last < gap) return true;
      last = x;
    }
    return false;
  };
  return !crowded(entries) && !crowded(exits);
};

const tradeMarkers = function () {
  const t = T();
  const words = captionsFit();
  const out = [];
  SPEC.trades.forEach(function (tr) {
    // an entry stays neutral while the exits are green and red, because it does
    // not know its outcome yet and colouring it by the exit paints hindsight onto
    // the bar. Neutral is not the same as faint though: muted is the colour
    // chosen to sit almost out of sight against the plane, which is right for a
    // gridline and wrong for the mark that says a position opened here
    out.push({
      time: SPEC.t[tr.in], position: "belowBar", shape: "arrowUp",
      color: t.s4, size: 1,
      text: words ? tr.side + " " + fmt(tr.size, CAPTION_DP.size) : "",
    });
    out.push({
      time: SPEC.t[tr.out], position: "aboveBar", shape: "arrowDown",
      color: tr.net >= 0 ? t.win : t.loss, size: 1,
      text: words ? (tr.net >= 0 ? "+" : "") + fmt(tr.net, CAPTION_DP.money) : "",
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

  // half open, like every span the same primitive paints: this pair is inclusive
  // of the exit bar, so under the shifted geometry it has to name the bar AFTER
  // it. The highlight now covers the entry and the exit bars whole, which it did
  // not before either (ADR 0101)
  selection = [hit.in, hit.out + 1, 0];
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
    "trade " + hit.i + "  ·  " + hit.bars + " bars  ·  net " +
    fmt(hit.net, TABLE_DP.money);
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
    const now = captionsFit();
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
    // a forced close reads as an ordinary exit otherwise, so the side says which
    // it was: the account ended this one, not the strategy
    const side = tr.liq ? tr.side + ' <span class="liq">liq</span>' : tr.side;
    // the axis over the plot says November and the row under it used to say 4856,
    // so reading a row back to the candle it happened on meant counting bars. Both
    // numbers were already here: the click map two blocks up indexes SPEC.t twice
    // and the legend has stamped its own bar since it was written. The tick keeps
    // a title, because it is what focus= and every guard message are phrased in
    return '<tr data-n="' + tr.i + '"><td>' + tr.i + '</td><td>' + side +
      '</td><td title="bar ' + tr.in + '">' + stamp(tr.in) +
      '</td><td title="bar ' + tr.out + '">' + stamp(tr.out) +
      '</td><td>' + fmt(tr.size, TABLE_DP.size) + '</td><td>' + fmt(tr.px_in, digits) +
      '</td><td>' + fmt(tr.px_out, digits) + '</td><td>' + fmt(tr.fees, TABLE_DP.money) +
      '</td><td class="' + (tr.net >= 0 ? "win" : "loss") + '">' +
      fmt(tr.net, TABLE_DP.money) +
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
