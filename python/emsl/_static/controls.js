"use strict";

// every pane owns its scale, so every pane owns its own A / L / %. Log on the
// equity pane turns compounding into a straight line and percent turns it into
// return; neither means anything as a chart-wide setting.

const CTL_H = 46;                      // backdrop height in UI units
let paneBands = [];                    // vertical extent of each pane, for hover
let paneState = [];

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
      const next = MODE === "dark" ? "light" : "dark";
      document.getElementById("theme").textContent = next === "dark" ? "LIGHT" : "DARK";
      applyTheme(next);
    }
  });

  applyScale();
  placePaneCtls();
  requestAnimationFrame(function () { applyScale(); placePaneCtls(); });

  // three things move these: the axis widening when the number format changes,
  // the chart resizing, and a pane separator being dragged. Only the first two
  // fire an event, so the drag is caught on mouseup
  chart.timeScale().subscribeSizeChange(placePaneCtls);
  el.addEventListener("mouseup", function () { requestAnimationFrame(placePaneCtls); });
  // a notebook cell dragged wider fires no window resize, so this observes the
  // element instead
  const frame = document.getElementById("frame");
  if (window.ResizeObserver) new ResizeObserver(applyScale).observe(frame);
  else window.addEventListener("resize", applyScale);
};
