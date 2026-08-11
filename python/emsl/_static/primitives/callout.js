"use strict";

// marker: anchored to a data point but displaced in PIXELS, so the glyph holds
// its distance from the bar at every zoom level instead of drifting as the scale
// changes. Native markers take aboveBar, belowBar or an exact price, none of
// which is a pixel offset, which is why the engine's own trade arrows use
// createSeriesMarkers and a user's Marker does not. Nobody should unify them.
//
// spec.bars is a list and spec.values is the matching list of anchors or absent,
// so one Marker and a Markers over four hundred bars are one primitive and one
// loop rather than one attached primitive per glyph.

const calloutPrimitive = function (anchor, spec) {

  const glyph = function (ctx, x, y, size, shape) {
    ctx.beginPath();
    if (shape === "square") {
      ctx.rect(x - size, y - size, size * 2, size * 2);
    } else if (shape === "arrow_up") {
      ctx.moveTo(x, y - size);
      ctx.lineTo(x + size, y + size);
      ctx.lineTo(x - size, y + size);
      ctx.closePath();
    } else if (shape === "arrow_down") {
      ctx.moveTo(x, y + size);
      ctx.lineTo(x + size, y - size);
      ctx.lineTo(x - size, y - size);
      ctx.closePath();
    } else {
      ctx.arc(x, y, size, 0, Math.PI * 2);
    }
  };

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
                const ts = chart.timeScale();
                const ctx = scope.context;
                const color = spec.color || T().s1;
                const size = Math.round(4 * UI);

                ctx.save();

                for (let j = 0; j < spec.bars.length; j++) {
                  const at = spec.bars[j];
                  const x = ts.logicalToCoordinate(at);
                  if (x === null || x < -80 || x > scope.mediaSize.width + 80) continue;

                  // an absent value anchors to the bar's own extreme, on the side
                  // the offset points, so the glyph never lands inside the candle.
                  // A projected bar has no candle to read, so it needs one given
                  let price = spec.values ? spec.values[j] : null;
                  if (price === null || price === undefined) {
                    const bar = SPEC.ohlc[at];
                    if (!bar) continue;
                    price = spec.offset >= 0 ? bar[1] : bar[2];
                  }
                  const y = anchor.priceToCoordinate(price);
                  if (y === null) continue;

                  const cy = y - spec.offset * UI;

                  // the leader line only earns its place when the glyph has been
                  // pushed far enough away to look detached from its bar
                  if (Math.abs(spec.offset) > 12) {
                    ctx.strokeStyle = color;
                    ctx.lineWidth = Math.max(1, UI);
                    ctx.beginPath();
                    ctx.moveTo(x, y - Math.sign(spec.offset) * Math.round(4 * UI));
                    ctx.lineTo(x, cy + Math.sign(spec.offset) * size);
                    ctx.stroke();
                  }

                  glyph(ctx, x, cy, size, spec.shape);
                  ctx.fillStyle = color;
                  ctx.fill();

                  if (spec.text) {
                    const fs = Math.round(11 * UI);
                    const padX = Math.round(7 * UI), padY = Math.round(4 * UI);
                    ctx.font = "600 " + fs + "px " + FONT;
                    const w = ctx.measureText(spec.text).width + padX * 2;
                    const h = fs + padY * 2;
                    const ty = cy - Math.sign(spec.offset || 1) * (size + h / 2 + Math.round(3 * UI));

                    ctx.beginPath();
                    if (ctx.roundRect) ctx.roundRect(x - w / 2, ty - h / 2, w, h, Math.round(3 * UI));
                    else ctx.rect(x - w / 2, ty - h / 2, w, h);
                    ctx.fillStyle = T().surface;
                    ctx.fill();
                    ctx.strokeStyle = color;
                    ctx.lineWidth = Math.max(1, UI);
                    ctx.stroke();

                    ctx.fillStyle = color;
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(spec.text, x, ty + 0.5);
                  }
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
