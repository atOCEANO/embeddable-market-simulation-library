"use strict";

// band: fill between two edges, the one plot type with no native call. Trace the
// upper edge left to right, the reference edge back, close, fill. Prices go
// through priceToCoordinate so the shape tracks the scale, log and percent
// included.
//
// only= makes the same primitive conditional: the fill exists solely where the
// upper edge is past the reference, and its gradient runs from that edge toward
// each excursion's own extreme rather than across the whole shape. AreaSeries
// fills to the pane edge and BaselineSeries fills both sides of a fixed value,
// so neither can express that.
//
// A scalar reference arrives as spec.level and an array one as spec.v2, which is
// why Band(up, lo) and Band(rsi, 70, only="above") are one code path.

const bandPrimitive = function (anchor, spec) {

  const refValue = function (j) {
    return spec.v2 ? spec.v2[j] : spec.level;
  };

  // a per-bar fill is a palette rather than a list of stops, and one canvas path
  // carries one fillStyle, so the shape is cut where the colour changes
  const perBar = spec.fill && !Array.isArray(spec.fill);

  // a run is {pts: [[x, yUpper, yRef, bar], ...], fill}. Without only= the whole
  // finite stretch is one run; with it, a run opens where the edge is passed and
  // closes where it is not; with a per-bar fill, also at every colour change
  const collect = function (ts) {
    const runs = [];
    let pts = [];
    let colour = null;
    const flush = function () {
      if (pts.length >= 2 && (!perBar || colour !== null)) {
        runs.push({ pts: pts, fill: colour });
      }
      pts = [];
    };
    const above = spec.only === "above";

    for (let j = 0; j < spec.v.length; j++) {
      const u = spec.v[j];
      const r = refValue(j);
      if (u === null || r === null || u === undefined || r === undefined) { flush(); continue; }
      if (spec.only && (above ? u <= r : u >= r)) { flush(); continue; }

      // the bar index travels with the point: a run skips the bars where the
      // condition is not met, so its position is not its bar
      const bar = spec.i0 + j;
      const x = ts.logicalToCoordinate(bar);
      const yu = anchor.priceToCoordinate(u);
      const yr = anchor.priceToCoordinate(r);
      if (x === null || yu === null || yr === null) { flush(); continue; }

      if (perBar) {
        const here = colorAt(spec.fill, bar);
        if (pts.length && here !== colour) {
          // close on this point and reopen from it, so the two regions meet
          // rather than leaving a one bar hole between them
          pts.push([x, yu, yr, bar]);
          flush();
          pts.push([x, yu, yr, bar]);
        }
        colour = here;
      }
      pts.push([x, yu, yr, bar]);
    }
    flush();
    return runs;
  };

  // without only=, fill[0] sits at the bottom of the shape and fill[last] at the
  // top, which is screen order. With it, fill[0] sits on the reference edge and
  // fill[last] at the excursion's extreme, so a brief poke past the level shades
  // faintly and a deep one shades hard. Python has already flipped the list for
  // only="below", so this never reasons about direction
  const paint = function (ctx, run) {
    if (perBar) {
      ctx.fillStyle = run.fill;
      return;
    }
    const pts = run.pts;
    if (spec.fill.length === 1) {
      ctx.fillStyle = spec.fill[0];
      return;
    }
    let y0, y1;
    if (spec.only) {
      y0 = pts[Math.floor(pts.length / 2)][2];
      y1 = y0;
      for (let k = 0; k < pts.length; k++) {
        if (spec.only === "above") y1 = Math.min(y1, pts[k][1]);
        else y1 = Math.max(y1, pts[k][1]);
      }
      if (Math.abs(y1 - y0) < 0.5) y1 = y0 + (spec.only === "above" ? -1 : 1);
    } else {
      y0 = -Infinity; y1 = Infinity;
      for (let k = 0; k < pts.length; k++) {
        y0 = Math.max(y0, pts[k][1], pts[k][2]);
        y1 = Math.min(y1, pts[k][1], pts[k][2]);
      }
      if (Math.abs(y1 - y0) < 0.5) return void (ctx.fillStyle = spec.fill[0]);
    }
    const g = ctx.createLinearGradient(0, y0, 0, y1);
    for (let k = 0; k < spec.fill.length; k++) {
      g.addColorStop(k / (spec.fill.length - 1), spec.fill[k]);
    }
    ctx.fillStyle = g;
  };

  // an edge is stroked segment by segment when its colour is per bar, because a
  // canvas path carries one strokeStyle. Assigning a palette object to
  // strokeStyle is a silent no-op, which is what this used to do
  const stroke = function (ctx, run) {
    const pts = run.pts;
    const ramped = isRamp(spec.color);
    const edges = spec.v2 ? [1, 2] : [1];
    ctx.lineWidth = spec.width || 1;

    for (let e = 0; e < edges.length; e++) {
      const at = edges[e];
      if (!ramped) {
        ctx.strokeStyle = spec.color;
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][at]);
        for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0], pts[k][at]);
        ctx.stroke();
        continue;
      }
      for (let k = 1; k < pts.length; k++) {
        const colour = colorAt(spec.color, pts[k][3]);
        if (colour === null) continue;
        ctx.strokeStyle = colour;
        ctx.beginPath();
        ctx.moveTo(pts[k - 1][0], pts[k - 1][at]);
        ctx.lineTo(pts[k][0], pts[k][at]);
        ctx.stroke();
      }
    }
  };

  return {
    updateAllViews: function () {},
    attached: function (p) { redraws.push(p.requestUpdate); },
    paneViews: function () {
      return [{
        zOrder: function () { return "bottom"; },
        renderer: function () {
          return {
            draw: function (target) {
              target.useMediaCoordinateSpace(function (scope) {
                const ts = chart.timeScale(), ctx = scope.context;
                const runs = collect(ts);
                if (!runs.length) return;
                ctx.save();
                for (let n = 0; n < runs.length; n++) {
                  const run = runs[n];
                  const pts = run.pts;
                  ctx.beginPath();
                  ctx.moveTo(pts[0][0], pts[0][1]);
                  for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0], pts[k][1]);
                  for (let k = pts.length - 1; k >= 0; k--) ctx.lineTo(pts[k][0], pts[k][2]);
                  ctx.closePath();
                  paint(ctx, run);
                  ctx.fill();
                  if (spec.color) stroke(ctx, run);
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
