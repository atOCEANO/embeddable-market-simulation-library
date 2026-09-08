"use strict";

// every pane owns its scale, so every pane owns its own A / L / %. Log on the
// equity pane turns compounding into a straight line and percent turns it into
// return; neither means anything as a chart-wide setting.

const CTL_H = 46;                      // backdrop height in UI units
let paneBands = [];                    // vertical extent of each pane, for hover
let paneState = [];
// the band at the top of every pane the legend has reserved, in css pixels.
// Published rather than local because callout.js places a caption from a price
// and has no other way of knowing the room is spoken for
let legendRoom = 0;

const placePaneCtls = function () {
  const el = document.getElementById("chart");
  const paneCtls = document.getElementById("paneCtls");
  const panes = chart.panes();
  const gutter = chart.priceScale("right").width();
  const axis = chart.timeScale().height();
  const hs = panes.map(function (p) { return p.getHeight(); });
  const sum = hs.reduce(function (a, b) { return a + b; }, 0);
  // pane heights come from the api but the separator size does not, so it is
  // derived: whatever vertical space the panes do not account for is divided
  // between the gaps
  const sep = panes.length > 1 ? Math.max(0, (el.clientHeight - axis - sum) / (panes.length - 1)) : 0;
  const bw = Math.max(14, Math.min(21 * UI, (gutter - 8) / 3));
  const h = CTL_H * UI;

  // the legend is painted into the top of its own pane, so the room it needs is a
  // count of pixels while the margin reserving that room is a fraction of the
  // pane. One fraction for all of them meant the price pane held back far more
  // than its label used, and the drawdown pane, at a fifth of the height, held
  // back less than a single line: its own curve was drawn through its own name
  // a narrow pane wraps its legend onto a second line and the room has to follow
  // it. This is a threshold rather than a measurement, because the wrap happens
  // in a canvas this file cannot measure into; it is set where the OHLC row stops
  // fitting on one line
  const lines = el.clientWidth < 620 ? 2 : 1;
  legendRoom = Math.round(11 * UI) + lines * Math.round(12.5 * UI * 1.5);
  panes.forEach(function (p, i) {
    const scale = p.priceScale("right");
    if (!scale || !hs[i]) return;
    scale.applyOptions({
      // the floor is not the legend's height. The renderer labels round prices
      // across the whole pane rather than across the data, so the topmost one
      // lands wherever the scale puts it and is clipped by the pane edge if the
      // margin is only as deep as the text below it: at 0.08 the price axis
      // opened on a half-drawn 140000.
      //
      // The same argument at the other end, which went unmade the first time and
      // left the equity panel of the held-bars documentation image ending on a
      // 6000 cut through the middle. A pane is bounded below by the next pane or
      // by the time axis, so there is nowhere for a low label to overhang into,
      // and half a number is worse than a slightly shorter plot
      scaleMargins: {
        top: Math.min(0.34, Math.max(0.15, legendRoom / hs[i])),
        bottom: 0.12,
      },
    });
  });

  paneBands = [];
  let y = 0;
  paneCtls.querySelectorAll(".panectl").forEach(function (g, i) {
    if (i >= hs.length) { g.style.display = "none"; return; }
    paneBands.push({ top: y, bottom: y + hs[i] });
    // the backdrop spans the whole gutter so the fade covers every label it
    // sits over, rather than leaving a lit strip either side of the buttons
    g.style.display = hs[i] < h + 6 ? "none" : "flex";
    g.style.setProperty("--bw", bw + "px");
    g.style.right = "0px";
    g.style.width = Math.max(48, gutter) + "px";
    g.style.height = h + "px";
    g.style.top = (y + hs[i] - h) + "px";
    y += hs[i] + sep;
  });
};

const applyScale = function () {
  const frame = document.getElementById("frame");
  const el = document.getElementById("chart");
  const next = scaleFor(frame.clientWidth || 900, el.clientHeight || 500);
  if (Math.abs(next - UI) < 0.005) return;
  UI = next;
  document.documentElement.style.setProperty("--ui", String(UI));
  chart.applyOptions({ layout: { fontSize: Math.round(12 * UI) } });
  placePaneCtls();
  invalidate();
};

// what the chart is and what it did, above the plot. Built with textContent
// rather than markup, so a title carrying angle brackets is a title and not a
// decision anyone has to think about
const mountHead = function () {
  const head = document.getElementById("head");
  if (!head) return;

  const rows = SPEC.headline || [];
  document.getElementById("title").textContent = SPEC.title || "";

  const stats = document.getElementById("stats");
  stats.textContent = "";
  rows.forEach(function (row) {
    const cell = document.createElement("span");
    const label = document.createElement("i");
    const value = document.createElement("b");
    label.textContent = row[0];
    value.textContent = row[1];
    cell.appendChild(label);
    cell.appendChild(value);
    stats.appendChild(cell);
  });

  head.hidden = !SPEC.title && !rows.length;
};

// the caller's own table, under the fills. Built with textContent rather than
// markup for the same reason the head row is: a cell carrying angle brackets is
// a cell, and not a decision anybody has to think about (ADR 0113)
const mountNotes = function () {
  if (!SPEC.notes) return;
  const head = document.getElementById("nhead");
  const body = document.getElementById("nbody");

  SPEC.notes.head.forEach(function (label) {
    const th = document.createElement("th");
    th.textContent = label;
    head.appendChild(th);
  });
  SPEC.notes.rows.forEach(function (row) {
    const tr = document.createElement("tr");
    row.forEach(function (value) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });

  document.getElementById("notes").hidden = false;
  // a run is not required to have notes, and mountTrades hides the button when
  // there are no fills, so this puts it back
  document.getElementById("tbl").hidden = false;
};

const mountControls = function () {
  mountHead();
  const el = document.getElementById("chart");
  const plot = document.getElementById("plot");
  const paneCtls = document.getElementById("paneCtls");

  paneState = SPEC.panels.map(function (p) {
    return { auto: true, mode: SCALE[p.scale] };
  });

  paneCtls.innerHTML = SPEC.panels.map(function (p, i) {
    return '<div class="panectl" data-pane="' + i + '" role="group" aria-label="' + p.name + ' scale">' +
      '<button data-act="auto" aria-pressed="true" title="Auto-fit the ' + p.name + ' scale">A</button>' +
      '<button data-act="log" aria-pressed="' + (paneState[i].mode === 1) + '" title="Logarithmic ' + p.name + ' scale">L</button>' +
      '<button data-act="pct" aria-pressed="' + (paneState[i].mode === 2) + '" title="Percent ' + p.name + ' scale">%</button></div>';
  }).join("");

  // only the hovered pane shows its buttons, driven by pointer position against
  // the measured bands rather than by CSS :hover. The backdrops have to stay
  // pointer-events:none while invisible or they swallow the drag that resizes
  // the price axis
  const setActivePane = function (index) {
    paneCtls.querySelectorAll(".panectl").forEach(function (g, i) {
      g.setAttribute("data-active", String(i === index));
    });
  };
  plot.addEventListener("mousemove", function (ev) {
    const y = ev.clientY - plot.getBoundingClientRect().top;
    let hit = -1;
    for (let i = 0; i < paneBands.length; i++) {
      if (y >= paneBands[i].top && y < paneBands[i].bottom) { hit = i; break; }
    }
    setActivePane(hit);
  });
  plot.addEventListener("mouseleave", function () { setActivePane(-1); });

  paneCtls.addEventListener("click", function (ev) {
    const btn = ev.target.closest("button");
    if (!btn) return;
    const group = btn.closest(".panectl");
    const i = Number(group.dataset.pane);
    const st = paneState[i];
    const scale = chart.panes()[i].priceScale("right");

    if (btn.dataset.act === "auto") {
      st.auto = !st.auto;
      scale.applyOptions({ autoScale: st.auto });
    } else {
      const want = btn.dataset.act === "log" ? 1 : 2;
      st.mode = st.mode === want ? 0 : want;      // log and percent are exclusive
      scale.applyOptions({ mode: st.mode });
    }
    group.querySelector('[data-act="auto"]').setAttribute("aria-pressed", String(st.auto));
    group.querySelector('[data-act="log"]').setAttribute("aria-pressed", String(st.mode === 1));
    group.querySelector('[data-act="pct"]').setAttribute("aria-pressed", String(st.mode === 2));
    placePaneCtls();
  });

  document.getElementById("tools").addEventListener("click", function (ev) {
    const btn = ev.target.closest("button");
    if (!btn) return;
    if (btn.id === "fit") {
      chart.timeScale().fitContent();
    } else if (btn.id === "tbl") {
      const p = document.getElementById("tablePanel");
      p.hidden = !p.hidden;
      btn.setAttribute("aria-pressed", String(!p.hidden));
    } else if (btn.id === "theme") {
      // the label follows the mode inside applyTheme, because mount changes it
      // too and setting it here left a chart opened light saying LIGHT
      applyTheme(MODE === "dark" ? "light" : "dark");
    }
  });

  // after mountTrades, which is what hides the button on a chart with no fills
  mountNotes();

  applyScale();
  placePaneCtls();
  requestAnimationFrame(function () { applyScale(); placePaneCtls(); });

  // three things move these: the axis widening when the number format changes,
  // the chart resizing, and a pane separator being dragged. Only the first two
  // fire an event, so the drag is caught on mouseup
  chart.timeScale().subscribeSizeChange(function () {
    placePaneCtls();
    // the axis rule is in pixels and a resize changes them without moving a bar,
    // so this is the only thing that tells it the width moved
    remeasureAxis();
  });
  el.addEventListener("mouseup", function () { requestAnimationFrame(placePaneCtls); });
  // a notebook cell dragged wider fires no window resize, so this observes the
  // element instead
  const frame = document.getElementById("frame");
  if (window.ResizeObserver) new ResizeObserver(applyScale).observe(frame);
  else window.addEventListener("resize", applyScale);
};
