"""Turn the built charts into the images the documentation shows.

Every picture in the docs is a screenshot of a document ``build.py`` actually
produced, so regenerating the set is one command and a stale one is hard to keep
by accident. Nothing here crops or retouches: what the reader sees is what the
library drew, at the size it was drawn. No chart is described here either, which
is why the manifest exists; this file only knows how to point a browser at one.
"""

from __future__ import annotations

import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

CHARTS = pathlib.Path("/work/charts")

# their own folder under imgs: these are generated and there are fourteen of
# them, so they do not get mixed in with the hand-made diagrams beside them
IMAGES = pathlib.Path("/out/imgs/charts")

# twice the scale, because these are read on displays that have had twice the
# pixels for a decade
SCALE = 2

# the settle is for the renderer, not for the page: lightweight-charts sizes
# itself from the container on the first frame and paints the primitives on the
# next, so a shot taken at load catches an empty canvas
SETTLE_MS = 2500


def main():
    manifest = CHARTS / "manifest.json"
    if not manifest.is_file():
        print(f"no {manifest}, so there is nothing to shoot; run build.py first")
        return 1

    shots = json.loads(manifest.read_text(encoding="utf-8"))
    missing = sorted(name for name in shots
                     if not (CHARTS / f"{name}.html").is_file())
    if missing:
        print(f"the manifest names charts that were not built: {missing}")
        return 1

    IMAGES.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as play:
        browser = play.chromium.launch()
        for name in sorted(shots):
            shot = shots[name]
            width, height = shot["width"], shot["height"]
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=SCALE,
            )
            page.goto((CHARTS / f"{name}.html").as_uri())
            page.wait_for_timeout(SETTLE_MS)
            page.screenshot(path=str(IMAGES / f"{name}.png"))
            page.close()
            size = (IMAGES / f"{name}.png").stat().st_size
            print(f"{name + '.png':<22} {width}x{height} "
                  f"({width / height:.1f}:1) at {SCALE}x {size / 1000:>6.0f} kB"
                  f"   {shot['caption']}")
        browser.close()

    print(f"\n{len(shots)} images written to {IMAGES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
