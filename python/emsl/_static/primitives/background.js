"use strict";

// background: shading anchored to bar index, so a span holds through zoom, pan
// and resize. One primitive covers three cases that differ only in where the fill
// comes from: a regime map, a single condition, and the current trade selection.
// A span carries a fill index, so the number of regimes is not fixed at two, and
// a label with no colour produced no span at all back in Python.
//
// Stops run bottom to top, so a three-stop fill is strongest at the floor and
// gone before the ceiling, which reads as ground under the candles rather than
// as a tinted pane they are sitting inside.

const spanFill = function (getSpans, getFills) {
  return {
    updateAllViews: function () {},
    attached: function (p) { redraws.push(p.requestUpdate); },
    paneViews: function () {
      return [{
        zOrder: function () { return "bottom"; },
        renderer: function () {
          return {
            draw: function () {},
            drawBackground: function (target) {
              target.useBitmapCoordinateSpace(function (scope) {
                const spans = getSpans();
                if (!spans || !spans.length) return;
                const ts = chart.timeScale();
                const ctx = scope.context;
                const h = scope.bitmapSize.height;
                const fills = getFills();

                // built once per frame, not once per span
                const cache = fills.map(function (stops) {
                  if (stops.length === 1) return stops[0];
                  const g = ctx.createLinearGradient(0, h, 0, 0);
                  for (let k = 0; k < stops.length; k++) {
                    g.addColorStop(k / (stops.length - 1), stops[k]);
                  }
                  return g;
                });

                // half a bar, in pixels, taken from two whole indices rather than
                // by asking for a fractional one: logicalToCoordinate returns the
                // CENTRE of a bar, and a span is half open over whole bars, so
                // painting centre to centre put the shading half a candle right
                // of the bars it belongs to. The width was unaffected, which is
                // why it read as correct until someone looked at an edge. The
                // converter does not take a fraction, and asking it for one drops
                // the span rather than nudging it (ADR 0101)
                const c0 = ts.logicalToCoordinate(0);
                const c1 = ts.logicalToCoordinate(1);
                const half = (c0 === null || c1 === null) ? 0 : (c1 - c0) / 2;

                for (let n = 0; n < spans.length; n++) {
                  const s = spans[n];
                  const left = ts.logicalToCoordinate(s[0]);
                  const right = ts.logicalToCoordinate(s[1]);
                  if (left === null || right === null) continue;
                  const a = left - half;
                  const b = right - half;
                  ctx.fillStyle = cache[s[2]];
                  ctx.fillRect(a * scope.horizontalPixelRatio, 0,
                    Math.max(1, (b - a) * scope.horizontalPixelRatio), h);
                }
              });
            },
          };
        },
      }];
    },
  };
};
