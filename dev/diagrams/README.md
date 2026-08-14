# Diagram sources

The mermaid sources for `.Documentation/imgs/`. They live here because the
originals were not kept and had to be reconstructed from the rendered PNGs, which
is a thing to do once.

Render one file per `docker run`, from PowerShell, never a `sh -c` loop:

```bash
docker run --rm --shm-size=1g -v "${PWD}/dev/diagrams:/data" minlag/mermaid-cli \
  -i /data/205314.mmd -o /data/205314.png \
  -c /data/config.json -p /data/puppeteer.json -b transparent -s 3
```

Then copy the PNG into `.Documentation/imgs/`.

Two things the recipe depends on. The image's bundled headless-shell is broken
with an ENOENT, so `puppeteer.json` must point `executablePath` at
`/usr/bin/chromium`. And the background must be `transparent` rather than white,
or the PNGs invert badly in GitHub's dark mode.

The palette is the repo's: node fill `#16232e`, teal border `#2ee6a6` for the
engine stages, blue border `#4d9feb` for the entry point, text `#e6edf3`, arrows
`#8b949e`, trebuchet sans.

The images are named with six-digit numbers matching the router's convention, and
the descriptive name survives only in the alt text: 205310 is the README hero,
205312 the crate layering, 205314 the step lifecycle, 205316 no-lookahead, 205318
the RL loop. Only 205314 has its source here so far; the rest are still PNG only.
