"""App-Icon nach der Handskizze (zwei überlagerte X zwischen zwei Strichen) als gemx.ico + SVG.

Run: python make_icon.py  -> gemx.ico (16-256 px) und app_icon.svg (gleiche Linien wie das
Favicon/Logo in compare_view.html). Nur numpy, PNG/ICO per Hand geschrieben.
"""
import struct
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
# Striche im 64er-Raster, aus dem Foto abgenommen: Strich links, \ \ / /, Strich rechts
SEGS = [(5.4, 37.0, 19.4, 34.2), (43.2, 30.0, 58.6, 27.8),
        (18.8, 21.4, 34.8, 40.4), (30.6, 18.0, 41.8, 39.8),
        (33.4, 17.4, 24.2, 46.0), (45.2, 17.2, 31.2, 43.2)]
SEGS = [tuple(round(32 + (v - 32) * 0.84, 1) for v in s) for s in SEGS]  # etwas Rand zur Kachel
BG, BG2, INK = (28, 28, 30), (44, 44, 48), (245, 245, 247)


def svg():
    lines = "".join(f'<path d="M{a} {b}L{c} {d}"/>' for a, b, c, d in SEGS)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#1c1c1e"/>'
            f'<g stroke="#f5f5f7" stroke-width="3.4" stroke-linecap="round" fill="none">{lines}</g></svg>')


def render(n):
    ss = 4  # supersampling
    g = (np.arange(n * ss) + 0.5) / (n * ss) * 64
    x, y = np.meshgrid(g, g)
    r = 14
    qx, qy = np.clip(x, r, 64 - r), np.clip(y, r, 64 - r)
    tile = (np.hypot(x - qx, y - qy) <= r).astype(float)
    w = max(3.4, 64 / n * 1.6) / 2  # small sizes need thicker lines to stay visible
    ink = np.zeros_like(x)
    for a, b, c, d in SEGS:
        vx, vy = c - a, d - b
        t = np.clip(((x - a) * vx + (y - b) * vy) / (vx * vx + vy * vy), 0, 1)
        ink = np.maximum(ink, (np.hypot(x - a - t * vx, y - b - t * vy) <= w).astype(float))
    down = lambda m: m.reshape(n, ss, n, ss).mean(axis=(1, 3))
    tile, ink = down(tile), down(ink)
    shade = (np.arange(n)[:, None] / max(n - 1, 1)).repeat(n, 1)
    rgb = np.stack([BG2[i] + (BG[i] - BG2[i]) * shade for i in range(3)], -1)
    rgb = rgb * (1 - ink[..., None]) + np.array(INK) * ink[..., None]
    return np.dstack([rgb, tile * 255]).round().astype(np.uint8)


def png(img):
    h, w, _ = img.shape
    raw = b"".join(b"\0" + img[i].tobytes() for i in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def ico(sizes=(16, 24, 32, 48, 64, 128, 256)):
    pngs = [png(render(s)) for s in sizes]
    out, off = struct.pack("<HHH", 0, 1, len(sizes)), 6 + 16 * len(sizes)
    for s, p in zip(sizes, pngs):
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(p), off)
        off += len(p)
    return out + b"".join(pngs)


if __name__ == "__main__":
    data = ico()
    assert data[:4] == b"\0\0\1\0" and render(32)[16, 0, 3] == 255 and render(32)[0, 0, 3] == 0  # Kante voll, Ecke rund
    (HERE / "gemx.ico").write_bytes(data)
    (HERE / "app_icon.svg").write_text(svg(), encoding="utf-8")
    (HERE / "vorschau").mkdir(exist_ok=True)
    (HERE / "vorschau" / "icon_256.png").write_bytes(png(render(256)))
    print(svg())
