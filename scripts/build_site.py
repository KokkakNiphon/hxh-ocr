"""Assemble the browser demo (web/) into site/dist for static hosting (Cloudflare Workers assets).

Copies only what the page loads: index.html, the JS/CSS, both wasm builds, every model listed
in web/models and web/models/accurate (*.json + the raw .hxhm it names; no base64 copies) and the examples listed in
web/examples/examples.json. Run scripts/build_web.ps1 first if the wasm or models changed.

    python scripts/build_site.py        # -> site/dist
    python scripts/build_site.py --translations --head   # what is committed, not the working tree

--head takes every file from git HEAD, so a deploy never picks up work in progress (other
agents or translators editing pages at the same time).
"""
from __future__ import annotations

import json
import re
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DIST = ROOT / "site" / "dist"


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "models").mkdir(parents=True)
    (DIST / "examples").mkdir()
    files = ["index.html", "theme.js", "lang.js", "site.css", "app.js", "ocr.js", "style.css", "hxh_infer.wasm", "hxh_infer_nosimd.wasm",
             # brand, icons, social card and crawler files (scripts/render_brand.py, scripts/site_meta.py)
             "favicon.svg", "apple-touch-icon.png", "og.png", "site.webmanifest", "robots.txt", "sitemap.xml",
             "brand/mark.svg", "brand/mark-dark.svg", "brand/icon-192.png", "brand/icon-512.png"]
    with_tr = "--translations" in sys.argv
    # the reader in other languages: its page plus its dictionary, once translated
    for lang in ("th", "ja"):
        for f in (f"index-{lang}.html", f"i18n-reader-{lang}.js"):
            if with_tr and (WEB / f).exists():
                files.append(f)
    if any(f.startswith("index-") for f in files):
        files.append("i18n.js")
    (DIST / "brand").mkdir()
    # two model sets: models/ (Fast: the live camera, and pictures on Fast) and models/accurate/
    for folder in ("models", "models/accurate"):
        for name in ("model", "model_v", "det"):
            meta_path = WEB / folder / f"{name}.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            if meta["file"].endswith(".b64.txt"):
                raise SystemExit(f"{meta_path} points at a base64 copy; it should name the raw .hxhm")
            files += [f"{folder}/{name}.json", f"{folder}/{meta['file']}"]
    examples = json.loads((WEB / "examples" / "examples.json").read_text(encoding="utf-8"))
    files += ["examples/examples.json"] + [f"examples/{e['file']}" for e in examples]
    # the blog: the page, its CSS/JS, img/ and data/ (helper notes and scripts stay out); the Thai and
    # Japanese pages only with --translations (they are published once they are complete)
    blog = WEB / "blog"
    if blog.exists():
        for src in sorted(blog.rglob("*")):
            if not with_tr and re.search(r"(-th|-ja)\.html$|^i18n-", src.name):
                continue
            if src.is_file() and src.suffix.lower() not in (".md", ".py", ".mjs"):
                rel = src.relative_to(WEB).as_posix()
                (DIST / rel).parent.mkdir(parents=True, exist_ok=True)
                files.append(rel)
    total = 0
    head = "--head" in sys.argv
    for f in files:
        src, dst = WEB / f, DIST / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if head:
            r = subprocess.run(["git", "show", f"HEAD:web/{f}"], cwd=ROOT, capture_output=True)
            if r.returncode:  # not committed yet: leave it out
                print(f"  skipped (not committed): web/{f}")
                continue
            dst.write_bytes(r.stdout)
        else:
            shutil.copyfile(src, dst)
        total += dst.stat().st_size
    print(f"{len(files)} files, {total / 1e6:.2f} MB -> {DIST}")


if __name__ == "__main__":
    main()
