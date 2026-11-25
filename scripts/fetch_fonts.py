"""Openly licensed fonts for the page generator (rust/hxh-synth) -> fonts/open/ (git-ignored).

Since generator v5 (pages_v5, lines_v5) every non-Hunter distractor (Latin, Japanese, Korean,
digits, icons) is rendered only from the fonts listed in FONTS below, all under the SIL Open Font
License 1.1 or the Apache License 2.0, so rendered datasets can be published under CC BY 4.0. The
Windows system fonts used for pages_v3 / pages_foreign are no longer used.

    python scripts/fetch_fonts.py                 # download from the official sources, check sha256
    python scripts/fetch_fonts.py --pin           # download, then write the hashes into fonts/open/fonts.lock.json
    python scripts/fetch_fonts.py --from DIR ...  # copy files with these names from local dirs instead
    python scripts/fetch_fonts.py --list          # print the table (name, licence, source)

Then: hxh-synth pages ... --font-dir fonts/open

`used_sha256` is the hash of the exact file that generated pages_v5 / lines_v5 (copies of the same
OFL / Apache fonts that ship inside other applications on the build machine, converted from WOFF2
where needed). The official URLs may serve a newer version of the same font; `--pin` records what
they serve. woff2 sources are decompressed to sfnt with fontTools (needs brotli).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fonts" / "open"
GF = "https://github.com/google/fonts/raw/main"

# file name (as hxh-synth's font table knows it), licence, family, official source, sha256 of the file used
FONTS = [
    ("NotoSans-Regular.ttf", "OFL-1.1", "Noto Sans", f"{GF}/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf", "6b04c8dd65af6b73"),
    ("NotoSansJP-Regular.otf", "OFL-1.1", "Noto Sans JP", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/JP/NotoSansJP-Regular.otf", "3e8146c4ce0945f2"),
    ("NotoSansKR-Regular.otf", "OFL-1.1", "Noto Sans KR", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/KR/NotoSansKR-Regular.otf", "b21817200abcb37a"),
    ("NotoSansTC-Light.otf", "OFL-1.1", "Noto Sans TC", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/TC/NotoSansTC-Light.otf", None),
    ("NotoSansTC-Regular.otf", "OFL-1.1", "Noto Sans TC", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/TC/NotoSansTC-Regular.otf", None),
    ("NotoSansTC-Bold.otf", "OFL-1.1", "Noto Sans TC", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/TC/NotoSansTC-Bold.otf", "8f9f70225bcbfe7f"),
    ("NotoSansTC-Black.otf", "OFL-1.1", "Noto Sans TC", "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/TC/NotoSansTC-Black.otf", None),
    ("NotoSansCJKsc-Regular.otf", "OFL-1.1", "Noto Sans CJK SC", "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf", None),
    ("NotoSansSymbols2-Regular.ttf", "OFL-1.1", "Noto Sans Symbols 2", f"{GF}/ofl/notosanssymbols2/NotoSansSymbols2-Regular.ttf", None),
    ("NotoEmoji-Regular.ttf", "OFL-1.1", "Noto Emoji (monochrome outlines)", f"{GF}/ofl/notoemoji/NotoEmoji%5Bwght%5D.ttf", None),
    ("GoNotoKurrent-Regular.ttf", "OFL-1.1", "Go Noto Kurrent", "https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf", "2f2cee5fbb2403df"),
    ("GoNotoKurrent-Bold.ttf", "OFL-1.1", "Go Noto Kurrent", "https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Bold.ttf", None),
    ("GenEiGothicPro-Light.otf", "Apache-2.0 (Source Han Sans derivative)", "GenEi Gothic Pro", "https://okoneya.jp/font/genei-gothic.html", None),
    ("GenEiGothicPro-Regular.otf", "Apache-2.0 (Source Han Sans derivative)", "GenEi Gothic Pro", "https://okoneya.jp/font/genei-gothic.html", "98c59a80695784cd"),
    ("GenEiGothicPro-Semibold.otf", "Apache-2.0 (Source Han Sans derivative)", "GenEi Gothic Pro", "https://okoneya.jp/font/genei-gothic.html", None),
    ("GenEiGothicPro-Bold.otf", "Apache-2.0 (Source Han Sans derivative)", "GenEi Gothic Pro", "https://okoneya.jp/font/genei-gothic.html", None),
    ("GenEiGothicPro-Heavy.otf", "Apache-2.0 (Source Han Sans derivative)", "GenEi Gothic Pro", "https://okoneya.jp/font/genei-gothic.html", "4f9576279a3b6011"),
    ("DroidSansFallback.ttf", "Apache-2.0", "Droid Sans Fallback", "https://android.googlesource.com/platform/frameworks/base/+/refs/tags/android-4.4_r1/data/fonts/DroidSansFallback.ttf", "05d71b179ef97b82"),
    ("Roboto-Light.ttf", "Apache-2.0", "Roboto", f"{GF}/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf", None),
    ("Roboto-Regular.ttf", "Apache-2.0", "Roboto", f"{GF}/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf", None),
    ("Roboto-Italic.ttf", "Apache-2.0", "Roboto", f"{GF}/ofl/roboto/Roboto-Italic%5Bwdth%2Cwght%5D.ttf", None),
    ("Roboto-Medium.ttf", "Apache-2.0", "Roboto", f"{GF}/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf", None),
    ("Roboto-Bold.ttf", "Apache-2.0", "Roboto", f"{GF}/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf", "2457150314076024"),
    ("Roboto-BoldItalic.ttf", "Apache-2.0", "Roboto", f"{GF}/ofl/roboto/Roboto-Italic%5Bwdth%2Cwght%5D.ttf", None),
    ("Roboto-Black.ttf", "Apache-2.0", "Roboto", f"{GF}/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf", None),
    ("Roboto-BoldCondensed.ttf", "Apache-2.0", "Roboto Condensed", f"{GF}/ofl/robotocondensed/RobotoCondensed%5Bwght%5D.ttf", None),
    ("RobotoMono-Regular.ttf", "Apache-2.0", "Roboto Mono", f"{GF}/apache/robotomono/RobotoMono%5Bwght%5D.ttf", "c3c88e6a2a6b5989"),
    ("RobotoSlab-Regular.ttf", "Apache-2.0", "Roboto Slab", f"{GF}/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf", None),
    ("RobotoSlab-Bold.ttf", "Apache-2.0", "Roboto Slab", f"{GF}/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf", None),
    ("Lato-Light.ttf", "OFL-1.1", "Lato", f"{GF}/ofl/lato/Lato-Light.ttf", None),
    ("Lato-Regular.ttf", "OFL-1.1", "Lato", f"{GF}/ofl/lato/Lato-Regular.ttf", "7ae714b63c2c8b94"),
    ("Lato-Italic.ttf", "OFL-1.1", "Lato", f"{GF}/ofl/lato/Lato-Italic.ttf", None),
    ("Lato-Bold.ttf", "OFL-1.1", "Lato", f"{GF}/ofl/lato/Lato-Bold.ttf", None),
    ("Lato-Black.ttf", "OFL-1.1", "Lato", f"{GF}/ofl/lato/Lato-Black.ttf", None),
    ("OpenSans-Light.ttf", "Apache-2.0", "Open Sans", f"{GF}/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf", None),
    ("OpenSans-Regular.ttf", "Apache-2.0", "Open Sans", f"{GF}/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf", "e64e508b2aa2880f"),
    ("OpenSans-Semibold.ttf", "Apache-2.0", "Open Sans", f"{GF}/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf", None),
    ("OpenSans-Bold.ttf", "Apache-2.0", "Open Sans", f"{GF}/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf", None),
    ("Inter.ttf", "OFL-1.1", "Inter", f"{GF}/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf", None),
    ("InterTight-Light.ttf", "OFL-1.1", "Inter Tight", f"{GF}/ofl/intertight/InterTight%5Bwght%5D.ttf", "ec12a8fe0f32e6d7"),
    ("fontawesome-webfont.ttf", "OFL-1.1 (font files)", "Font Awesome 4.3 (icons)", "https://github.com/FortAwesome/Font-Awesome/raw/v4.3.0/fonts/fontawesome-webfont.ttf", "9e540a087924a6e6"),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def to_sfnt(data: bytes) -> bytes:
    if data[:4] != b"wOF2":
        return data
    from fontTools.ttLib import TTFont
    f = TTFont(io.BytesIO(data))
    f.flavor = None
    out = io.BytesIO()
    f.save(out)
    return out.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", nargs="+", default=None, help="copy from local dirs (files matched by name)")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for name, lic, fam, url, used in FONTS:
            print(f"{name:32s} {lic:42s} {fam:30s} {url}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "fonts.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    bad = 0
    for name, lic, fam, url, used in FONTS:
        if args.src:
            cands = [p for d in args.src for p in Path(d).rglob("*") if p.is_file() and p.name.lower() == name.lower()]
            if not cands:
                print(f"missing locally: {name}")
                continue
            data = to_sfnt(cands[0].read_bytes())
        else:
            if "okoneya.jp" in url or "googlesource" in url:
                print(f"{name}: fetch by hand from {url} (no direct file URL)")
                continue
            with urllib.request.urlopen(url) as r:
                data = to_sfnt(r.read())
        h = sha(data)
        pinned = lock.get(name)
        if pinned and pinned != h and not args.pin:
            print(f"HASH MISMATCH {name}: {h} != pinned {pinned}")
            bad += 1
            continue
        if args.pin or not pinned:
            lock[name] = h
        (OUT / name).write_bytes(data)
        print(f"{name:32s} {len(data) / 1e6:6.2f} MB  sha256 {h[:16]}  {lic}")
    lock_path.write_text(json.dumps(lock, indent=1, sort_keys=True))
    (OUT / "LICENSES.md").write_text("# fonts/open\n\n| file | licence | family | source |\n|---|---|---|---|\n" +
                                     "\n".join(f"| {n} | {l} | {f} | {u} |" for n, l, f, u, _ in FONTS) + "\n", encoding="utf-8")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
