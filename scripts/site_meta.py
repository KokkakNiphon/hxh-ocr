"""Keep every page's <head> metadata, logo and language plumbing in step, and write sitemap.xml/robots.txt.

For each page in PAGES that exists, it:
  - rewrites a managed block before </head> (between <!-- site:begin --> and <!-- site:end -->):
    canonical and hreflang links, Open Graph and Twitter cards built from the page's own <title> and
    meta description, author, icons, manifest, theme colour, JSON-LD, plus site.css and lang.js;
  - drops what the block replaces (the old data-URI favicon, old hreflang links);
  - swaps the old CSS logo (<span class="mark">) for the SVG mark;
  - removes the old in-article language links (<p class="lang-switch">); the nav menu (lang.js) replaces them.
Titles and descriptions stay in the pages themselves, so translators edit them there.

    python scripts/site_meta.py            # all pages
    python scripts/site_meta.py --only blog
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SITE = "https://hxh-ocr.kokkakniphon.workers.dev"
AUTHOR = {"@type": "Person", "name": "KokkakNiphon", "url": "https://github.com/KokkakNiphon"}
PUBLISHED, MODIFIED = "2026-11-25", "2026-11-25"
DATES = {"dsi": ("2026-11-26", "2026-11-26")}  # (published, modified) for pages that differ from the above
LOCALE = {"en": "en_US", "th": "th_TH", "ja": "ja_JP"}
MARK = (WEB / "brand" / "mark-inline.txt").read_text(encoding="utf-8").strip()

# file, family, language, URL path on the live site (it drops ".html")
FAMILIES = {
    "reader": {"en": ("index.html", "/"), "th": ("index-th.html", "/index-th"), "ja": ("index-ja.html", "/index-ja")},
    "explainer": {"en": ("blog/index.html", "/blog/"), "th": ("blog/index-th.html", "/blog/index-th"), "ja": ("blog/index-ja.html", "/blog/index-ja")},
    "deep": {"en": ("blog/deep-dive.html", "/blog/deep-dive"), "th": ("blog/deep-dive-th.html", "/blog/deep-dive-th"), "ja": ("blog/deep-dive-ja.html", "/blog/deep-dive-ja")},
    "dsi": {"en": ("blog/dsi.html", "/blog/dsi"), "th": ("blog/dsi-th.html", "/blog/dsi-th"), "ja": ("blog/dsi-ja.html", "/blog/dsi-ja")},
}
KIND = {"reader": "WebApplication", "explainer": "BlogPosting", "deep": "TechArticle", "dsi": "BlogPosting"}
# the social card: the site's own, or a page family's (path on the site, alt text)
OG = {"dsi": ("/blog/img/dsi/og.jpg", "HxH Translator for DSi: the app's pick-a-mode screen with the 3D beetle opened, and はんたー in Hunter script.")}
OG_DEFAULT = ("/og.png", "HxH OCR: the first OCR for the Hunter × Hunter alphabet, reading a manga page.")
ABOUT = [{"@type": "Thing", "name": "Hunter × Hunter alphabet"}, {"@type": "Thing", "name": "Optical character recognition"}]


def published(family: str, lang: str) -> bool:
    """A page is listed (hreflang, sitemap) only if its file exists and, for translations, is ready."""
    return (WEB / FAMILIES[family][lang][0]).exists()


def block(family: str, lang: str, title: str, desc: str, prefix: str) -> str:
    url = SITE + FAMILIES[family][lang][1]
    alts = [(l, SITE + FAMILIES[family][l][1]) for l in ("en", "th", "ja") if published(family, l)]
    og_path, og_alt = OG.get(family, OG_DEFAULT)
    og_image = SITE + og_path
    ld: dict = {"@context": "https://schema.org", "@type": KIND[family], "url": url, "inLanguage": lang,
                "description": desc, "image": og_image, "author": AUTHOR}
    if family == "reader":
        ld.update({"name": "HxH OCR", "alternateName": "Hunter × Hunter alphabet reader",
                   "applicationCategory": "UtilitiesApplication", "operatingSystem": "Any (runs in a web browser)",
                   "browserRequirements": "Requires JavaScript and WebAssembly", "isAccessibleForFree": True,
                   "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}, "about": ABOUT})
    else:
        pub, mod = DATES.get(family, (PUBLISHED, MODIFIED))
        about = ABOUT + ([{"@type": "Thing", "name": "Nintendo DSi"}] if family == "dsi" else [])
        ld.update({"headline": title, "datePublished": pub, "dateModified": mod, "publisher": AUTHOR,
                   "mainEntityOfPage": url, "about": about, "isPartOf": {"@type": "WebSite", "name": "HxH OCR", "url": SITE + "/"}})
    e = lambda s: html.escape(s, quote=True)
    lines = [
        "<!-- site:begin (scripts/site_meta.py) -->",
        f'<link rel="canonical" href="{url}">',
        *[f'<link rel="alternate" hreflang="{l}" href="{u}">' for l, u in alts],
        f'<link rel="alternate" hreflang="x-default" href="{SITE + FAMILIES[family]["en"][1]}">',
        '<meta name="author" content="KokkakNiphon">',
        '<meta name="robots" content="index, follow, max-image-preview:large">',
        '<meta name="theme-color" content="#2340d8" media="(prefers-color-scheme: light)">',
        '<meta name="theme-color" content="#0b0c0f" media="(prefers-color-scheme: dark)">',
        f'<meta property="og:type" content="{"website" if family == "reader" else "article"}">',
        '<meta property="og:site_name" content="HxH OCR">',
        f'<meta property="og:title" content="{e(title)}">',
        f'<meta property="og:description" content="{e(desc)}">',
        f'<meta property="og:url" content="{url}">',
        f'<meta property="og:image" content="{og_image}">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        f'<meta property="og:image:alt" content="{e(og_alt)}">',
        f'<meta property="og:locale" content="{LOCALE[lang]}">',
        *[f'<meta property="og:locale:alternate" content="{LOCALE[l]}">' for l, _ in alts if l != lang],
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{e(title)}">',
        f'<meta name="twitter:description" content="{e(desc)}">',
        f'<meta name="twitter:image" content="{og_image}">',
        f'<link rel="icon" href="{prefix}favicon.svg" type="image/svg+xml">',
        f'<link rel="apple-touch-icon" href="{prefix}apple-touch-icon.png">',
        f'<link rel="manifest" href="{prefix}site.webmanifest">',
        f'<link rel="stylesheet" href="{prefix}site.css">',
        f'<script src="{prefix}lang.js" defer></script>',
        '<script type="application/ld+json">' + json.dumps(ld, ensure_ascii=False).replace("<", "\\u003c") + "</script>",
        "<!-- site:end -->",
    ]
    return "\n".join(lines) + "\n"


def process(family: str, lang: str):
    rel, _ = FAMILIES[family][lang]
    p = WEB / rel
    if not p.exists():
        return
    s = p.read_text(encoding="utf-8")
    prefix = "../" if rel.startswith("blog/") else ""
    title = html.unescape(re.search(r"<title>(.*?)</title>", s, re.S).group(1).strip())
    m = re.search(r'<meta name="description" content="(.*?)">', s, re.S)
    desc = html.unescape(m.group(1)) if m else title
    s = re.sub(r"<!-- site:begin.*?<!-- site:end -->\n?", "", s, flags=re.S)
    s = re.sub(r'<link rel="icon" href="data:[^"]*">\n?', "", s)
    s = re.sub(r'<link rel="alternate" hreflang="[^"]*" href="[^"]*">\n?', "", s)
    s = s.replace("</head>", block(family, lang, title, desc, prefix) + "</head>", 1)
    s = s.replace('<span class="mark" aria-hidden="true"></span>', MARK)
    s = re.sub(r'\n?[ \t]*<p class="lang-switch"[^>]*>.*?</p>', "", s, flags=re.S)
    p.write_text(s, encoding="utf-8", newline="\n")
    print(f"{rel}: {title}")


def sitemap():
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">']
    for family, langs in FAMILIES.items():
        alts = [(l, SITE + langs[l][1]) for l in ("en", "th", "ja") if published(family, l)]
        for l, u in alts:
            out.append(f"  <url><loc>{u}</loc><lastmod>{DATES.get(family, (PUBLISHED, MODIFIED))[1]}</lastmod>")
            out += [f'    <xhtml:link rel="alternate" hreflang="{l2}" href="{u2}"/>' for l2, u2 in alts]
            out.append(f'    <xhtml:link rel="alternate" hreflang="x-default" href="{SITE + langs["en"][1]}"/>')
            out.append("  </url>")
    out.append("</urlset>")
    (WEB / "sitemap.xml").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    (WEB / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {SITE}/sitemap.xml\n", encoding="utf-8", newline="\n")
    print("sitemap.xml, robots.txt")


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for family in FAMILIES:
        if only == "blog" and family == "reader":
            continue
        for lang in ("en", "th", "ja"):
            process(family, lang)
    sitemap()


if __name__ == "__main__":
    main()
