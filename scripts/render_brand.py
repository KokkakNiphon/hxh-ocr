"""Render the raster brand files from their SVG/HTML sources with headless Chrome (or Edge).

    web/brand/mark.svg -> web/apple-touch-icon.png (180), web/brand/icon-192.png, web/brand/icon-512.png
    web/brand/og.html  -> web/og.png (1200 x 630, the social card)

Needs the static server on http://127.0.0.1:8765 serving web/ (python -m http.server 8765 --directory web),
so the card's web fonts and images load.

    python scripts/render_brand.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
BASE = "http://127.0.0.1:8765"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "msedge",
]


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("no Chrome or Edge found")


def shot(url: str, out: Path, w: int, h: int):
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={prof}",
                        "--default-background-color=00000000", f"--window-size={w},{h}", "--virtual-time-budget=6000",
                        f"--screenshot={out}", url], check=True, capture_output=True, timeout=120)
    print(f"{out.relative_to(ROOT)}  {w}x{h}")


def main():
    tmp = WEB / "brand" / "_icon.html"
    for size, out in [(180, WEB / "apple-touch-icon.png"), (192, WEB / "brand" / "icon-192.png"), (512, WEB / "brand" / "icon-512.png")]:
        # app icons are full-bleed squares (the platform rounds them), so the mark's corners are squared off
        svg = (WEB / "brand" / "mark.svg").read_text(encoding="utf-8").replace('rx="8"', 'rx="0"')
        tmp.write_text(f'<!doctype html><style>html,body{{margin:0;background:transparent}}svg{{display:block;width:{size}px;height:{size}px}}</style>{svg}', encoding="utf-8")
        shot(f"{BASE}/brand/_icon.html", out, size, size)
    tmp.unlink()
    shot(f"{BASE}/brand/og.html", WEB / "og.png", 1200, 630)


if __name__ == "__main__":
    main()
