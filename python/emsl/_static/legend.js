"use strict";

// legend: one per pane, painted on canvas rather than built as DOM. Canvas needs
// no layout space, and pane separators are draggable, which would leave a DOM
// overlay needing repositioning on a drag that fires no event.
//
// Every panel prints its own name first. A panel is created the first time a
// panel= mentions it, so a typo makes a stray panel, and showing the name is what
// turns that into something you cannot miss on screen.

const legendValueAt = function (spec, i) {
  const j = i - spec.i0;
  if (j < 0 || j >= spec.v.length) return null;
  return spec.v[j];
};

const legendRows = function (index) {
  const panel = SPEC.panels[index];
  const i = cursor;
  const rows = [{ label: panel.name, strong: true }];

  // a bar past the last candle exists on the axis but has no OHLC, because
  // future= lengthens the axis and never invents a bar that traded
  const traded = i < SPEC.n;

  if (panel.candles) {
    if (SPEC.interval) rows.push({ label: SPEC.interval });
    rows.push({ label: stamp(i) });
    if (traded) {
      const bar = SPEC.ohlc[i];
      const up = bar[3] >= bar[0];
      rows.push({ swatch: up ? T().up : T().down, hollow: up, label: "O", value: fmt(bar[0], panel.digits) });
      rows.push({ label: "H", value: fmt(bar[1], panel.digits) });
      rows.push({ label: "L", value: fmt(bar[2], panel.digits) });
      rows.push({ label: "C", value: fmt(bar[3], panel.digits), valueColor: up ? T().up : T().down });
    } else {
      rows.push({ label: "projected" });
    }
  }

  if (panel.volume && SPEC.vol && traded) {
    const bar = SPEC.ohlc[i];
    rows.push({
      swatch: (bar[3] >= bar[0] ? T().up : T().down) + "66",
      label: "Volume", value: fmt(legendValueAt(SPEC.vol, i), panel.digits),
    });
  }

  SERIES.forEach(function (entry) {
    const spec = entry.spec;
    if (spec.panel !== panel.name) return;
    const swatch = isRamp(spec.color) ? colorAt(spec.color, i) : (spec.color || T().s1);
    rows.push({
      swatch: swatch || T().muted,
      label: spec.name || "",
      value: fmt(legendValueAt(spec, i), spec.digits === undefined ? panel.digits : spec.digits),
    });
  });

  if (panel.name === "equity" && SPEC.equity) {
    const v = legendValueAt(SPEC.equity, i);
    const base = SPEC.equity.v[0];
    if (v !== null && base) {
      const r = (v / base - 1) * 100;
      rows.push({
        label: "ret", value: (r >= 0 ? "+" : "") + fmt(r, 2) + "%",
        valueColor: r >= 0 ? T().win : T().loss,
      });
    }
  }

  return rows;
};

const legendPrimitive = function (index) {
  return {
    updateAllViews: function () {},
    attached: function (p) { redraws.push(p.requestUpdate); },
    paneViews: function () {
      return [{
        zOrder: function () { return "top"; },
        renderer: function () {
          return {
            draw: function (target) {
              target.useMediaCoordinateSpace(function (scope) {
                const ctx = scope.context, t = T();
                // a pane too short to hold a label without covering its own data
                // shows none; the value is still on the crosshair axis label
                if (scope.mediaSize.height < 26 * UI) return;

                const fs = Math.round(12.5 * UI);
                const pad = Math.round(11 * UI), gap = Math.round(6 * UI);
                const swS = Math.round(fs * 0.6), swGap = Math.round(5 * UI);
                const sepW = Math.round(15 * UI), lineH = Math.round(fs * 1.5);
                const maxX = scope.mediaSize.width - Math.round(6 * UI);
                // canvas text does not wrap, so it is done by hand: an entry that
                // does not fit moves to a second line, and if the pane is too
                // short for one the remainder is counted rather than dropped
                const maxLines = Math.max(1, Math.floor((scope.mediaSize.height * 0.42) / lineH));

                ctx.save();
                ctx.textBaseline = "alphabetic";

                const measure = function (r) {
                  ctx.font = (r.strong ? "600 " : "400 ") + fs + "px " + FONT;
                  let w = ctx.measureText(r.label).width;
                  if (r.swatch) w += swS + swGap;
                  if (r.value !== null && r.value !== undefined) {
                    ctx.font = "400 " + fs + "px " + FONT;
                    w += (r.label ? gap : 0) + ctx.measureText(String(r.value)).width;
                  }
                  return w;
                };

                const paintRow = function (r, x, y) {
                  if (r.swatch) {
                    if (r.hollow) {
                      ctx.strokeStyle = r.swatch;
                      ctx.lineWidth = Math.max(1, Math.round(1.5 * UI));
                      ctx.strokeRect(x + 0.5, y - swS + 0.5, swS - 1, swS - 1);
                    } else {
                      ctx.fillStyle = r.swatch;
                      ctx.fillRect(x, y - swS, swS, swS);
                    }
                    x += swS + swGap;
                  }
                  if (r.label) {
                    ctx.font = (r.strong ? "600 " : "400 ") + fs + "px " + FONT;
                    ctx.fillStyle = r.strong ? t.ink : t.muted;
                    ctx.fillText(r.label, x, y);
                    x += ctx.measureText(r.label).width + gap;
                  }
                  if (r.value !== null && r.value !== undefined) {
                    ctx.font = "400 " + fs + "px " + FONT;
                    ctx.fillStyle = r.valueColor || t.ink2;
                    ctx.fillText(String(r.value), x, y);
                  }
                };

                const rows = legendRows(index);
                // the counter has to fit too, or the count vanishes exactly when
                // it is the only thing left saying something is missing
                const more = function (left) { return "+" + left; };
                ctx.font = "600 " + fs + "px " + FONT;
                const moreW = ctx.measureText(more(rows.length)).width + sepW;

                let line = 0, x = pad, lastX = pad;
                for (let k = 0; k < rows.length; k++) {
                  const w = measure(rows[k]);
                  const last = line + 1 >= maxLines;
                  // `x > pad` used to guard this, which let the FIRST entry on a
                  // line paint however wide it was: canvas clipped it at the pane
                  // edge and no counter was drawn, so a narrow panel dropped a
                  // series in silence. That is the exact failure this block is
                  // here to prevent, produced by the block itself (ADR 0076)
                  const overflows = x + w > (last ? maxX - moreW : maxX);
                  if (overflows) {
                    if (last) {
                      ctx.font = "600 " + fs + "px " + FONT;
                      ctx.fillStyle = t.muted;
                      ctx.fillText(more(rows.length - k), lastX, pad + fs * 0.85 + line * lineH);
                      break;
                    }
                    line++; x = pad;
                    // still too wide on a fresh line, and this is now the last
                    // one: report the remainder rather than clipping it
                    if (line + 1 >= maxLines && x + w > maxX - moreW) {
                      ctx.font = "600 " + fs + "px " + FONT;
                      ctx.fillStyle = t.muted;
                      ctx.fillText(more(rows.length - k), x, pad + fs * 0.85 + line * lineH);
                      break;
                    }
                  }
                  paintRow(rows[k], x, pad + fs * 0.85 + line * lineH);
                  x += w + sepW;
                  lastX = x;
                }
                ctx.restore();
              });
            },
          };
        },
      }];
    },
  };
};
