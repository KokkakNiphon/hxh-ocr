# Foreign-text and icon distractors

`det_v1` was trained on `pages_v2`, which held only Hunter glyphs and non-text art. On a
screenshot of Omniglot's Hunter × Hunter page, it boxed every line of text: the headings,
the Japanese with kanji, the romaji, the English prose, the URLs. It also boxed rows of icons.
The reader then turned that Latin and Japanese into garbage kana. Since `pages_v3`, pages also
hold ordinary writing in real fonts and icons. These are never labelled, so the DBNet targets
treat them as background, and the detector has to learn to fire only on Hunter script.

## Fonts (`src/fonts.rs`)

**Since v5 (2026-11-25) only openly licensed fonts are used**, so rendered sets can be published
under CC BY 4.0: `--font-dir fonts/open`, filled by `scripts/fetch_fonts.py` (git-ignored; every
file's name, version, licence string and sha256 are in `fonts.lock.json` next to this file). The
Windows system fonts that pages_v3 and pages_foreign were rendered with (Segoe UI, Segoe MDL2 /
Fluent icons, MS Gothic, Yu Gothic, ...) have unclear terms for redistributing rendered output;
`pages_v3o` and `pages_foreign_open` are the same recipes re-rendered with the open set.

| family | files | licence |
|---|---|---|
| Noto Sans | NotoSans-Regular | OFL-1.1 |
| Noto Sans JP / KR | NotoSansJP-Regular, NotoSansKR-Regular | OFL-1.1 |
| Noto Sans TC | Light, Regular, Bold, Black (kana included) | OFL-1.1 |
| Noto Sans CJK SC | Regular | OFL-1.1 |
| Noto Sans Symbols 2 | Regular (icon class) | OFL-1.1 |
| Noto Emoji | monochrome outlines (icon class) | OFL-1.1 |
| Go Noto Kurrent | Regular, Bold | OFL-1.1 |
| GenEi Gothic Pro | Light, Regular, Semibold, Bold, Heavy (Source Han Sans derivative) | Apache-2.0 (as stated in the font) |
| Droid Sans Fallback | Regular (kana, kanji, hangul) | Apache-2.0 |
| Roboto | Light, Regular, Italic, Medium, Bold, Bold Italic, Black, Bold Condensed | Apache-2.0 |
| Roboto Slab, Roboto Mono | Regular, Bold / Regular | Apache-2.0 |
| Lato | Light, Regular, Italic, Bold, Black | OFL-1.1 |
| Open Sans | Light, Regular, Semibold, Bold | Apache-2.0 |
| Inter, Inter Tight | Inter (variable, default instance), Inter Tight Light | OFL-1.1 |
| Font Awesome 4.3 | fontawesome-webfont (icon class) | OFL-1.1 (font files) |

Where the files came from: the copies used for pages_v5, pages_v3o, pages_foreign_open and
lines_v5 were taken from other software installed on the build machine that bundles these
open fonts (WOFF2 decompressed with fontTools). Each file was checked by its own licence fields
(OpenType name IDs 13 / 14) before use, except seven with no licence text of their own (six
Roboto files from 2011, Apache-2.0 upstream; Font Awesome 4.3, whose field links to its licence
page; its font files are OFL-1.1), which were checked against the upstream licence; fonts whose
own fields gave other terms (Droid Sans Mono: Ascender EULA, Brutal Type,
Motiva Sans) were left out. `scripts/fetch_fonts.py` can fetch the official files from their
upstream sources (google/fonts, notofonts/noto-cjk, go-noto-universal, Font Awesome) and pin
them by hash; the official files may be newer versions, so their hashes can differ from the
lock file.

There is no open handwriting or mincho (serif Japanese) face in the set yet, so the `hand`
class and Japanese serif text are absent from v5 data. **Font Awesome 4.3 stands in for
Material Symbols** (Apache-2.0), which is not on the build machine; it would be the natural
second icon font once fetched.

- **Read at run time** from `--font-dir` (default `C:\Windows\Fonts`, or `$HXH_FONT_DIR`; use
  `;` to give several directories). The binary knows only a list of file names (`KNOWN`).
  Nothing is copied, committed or embedded. A directory with none of the known files gives
  every `.ttf` / `.otf` / `.ttc` in it (up to 64 files, first face). `hxh-synth fonts` lists
  what would be used, and `meta.json` records the face list of each set.
- **Rasterising:** `ab_glyph` 0.2.32 (pinned; pure Rust, `ttf-parser` underneath) supplies
  cmap, advances, kerning and outlines for TrueType, CFF and collections. The outlines are
  flattened to polygons in pixels (0.15 px tolerance) and filled with tiny-skia, which also
  draws the Hunter strokes. Foreign text therefore goes through the same homographies,
  outlines, shadows, patchy ink and page degradations. Fake bold is outline dilation (a
  stroke of the outline, 0.02-0.07 em). Synthetic italic is a shear of 0.12-0.25.
- **Scripts:** coverage bits come from probe strings, so a face is used only for the scripts it
  really covers: Latin, Latin-ext (macrons), Japanese, Korean, Greek, Cyrillic. A string the
  face does not fully cover loses its missing characters.
- **Faces found on this machine (80):**
  - Latin: Arial (4 styles), Arial Narrow (2), Verdana (2), Tahoma (2), Segoe UI (4),
    Calibri (3), Trebuchet (2), Candara, Corbel, Bahnschrift, Franklin Gothic, MS Sans
    Serif, Ebrima, Gadugi.
  - Serif: Times (3), Georgia (3), Cambria (2), Palatino (3), Garamond (2), Bookman, Century,
    Constantia, Sylfaen.
  - Mono: Consolas (2), Courier (2), Lucida Console.
  - Display: Impact, Franklin Gothic Condensed.
  - Hand: Comic Sans (2), Ink Free, Segoe Print (2), Segoe Script, MV Boli, Gabriola.
  - Japanese: MS Gothic, MS UI Gothic, MS PGothic, Yu Gothic (L, R, M, B) and Noto Sans JP
    Bold. Meiryo, MS Mincho, Yu Mincho and BIZ UD are in the list but not installed. SimSun,
    Microsoft YaHei and YaHei Bold (Chinese designs that include kana) stand in for the
    missing mincho shapes.
  - Korean: Malgun Gothic (2).
  - Icons: Segoe MDL2 Assets (1,525 glyphs), Segoe Fluent Icons (1,624), Segoe UI Emoji
    (outlines of the colour glyphs' base layer; 22k glyphs, only the cmap-mapped ones are
    used), Webdings, Wingdings 1-3, Marlett, Symbol. Segoe UI Symbol serves as both a text face
    and a symbol source.
- **Icons are picked by glyph id**, because the symbol-encoded fonts (Wingdings and the rest)
  have no Unicode cmap that ab_glyph reads. A glyph must have an outline at least 0.25 em on
  its long side. Text faces also supply the Unicode symbols they cover from `SYMBOLS`: arrows,
  ★☆✓✗, ●○■□▲▼◆, ♪♥♦♣♠, ①②③, ©®™, ±×÷, box drawing, math, ⚙⚠✉ and so on. Many Hunter glyphs
  look like such symbols (∀, ∇, Z-like strokes, rings, dots), so these are the hard negatives.

## Text (`src/foreign.rs`)

| `script` | content |
|---|---|
| `latin` | English-like sentences from a ~300-word list plus HxH names; headings in Title Case or ALL CAPS; SFX (BAM!, WHOOSH); labels (© 2011 Shueisha, Page 157); links (Read more », ← Previous) |
| `romaji` | Hepburn from the kana sampler: yōon, sokuon, n', macrons for ー and ou / uu (hantā, Tōkyō) |
| `url` | https:// / www. + domain + TLD + paths |
| `digits` | dates (2011-10-02, 10/02/2011, March 3, 2011), times, prices ($ € £ ¥), %, phone numbers, No. / Vol. / p. / Ch., versions, ISBNs |
| `japanese` | kanji sentences from templates (noun / particle / verb / ending lists, 「」、。, years and months), headings (概要, 文字一覧), labels (第3話, 36巻, 出口, 営業中), now and then a Latin insert |
| `kana` | hiragana or katakana from the Hunter text sampler, katakana SFX (ドン, ゴゴゴ) |
| `korean`, `greek`, `cyrillic` | words from small lists, about 3% each |
| `icon` | icon glyphs; the text is the character, or `#<glyph id>` for private-use and unmapped glyphs |

Japanese also runs vertically: top-to-bottom columns, glyphs upright, ー / brackets / dashes
turned 90°, 、。 moved to the top right of their cell, small kana nudged. Latin is always
horizontal.

## Placement

- **Inside the existing layouts** (`page.rs`):
  - **Whole groups:** a bubble, caption, sign, block, free title / SFX, filler or table cell is
    written in foreign text with probability `--foreign-prob` (0.15), times 1.6 on manga
    pages. The group keeps the same containers, styles, ±12° rotation and perspective. Foreign
    bubbles and captions are vertical Japanese 40% of the time, as in Japanese manga.
  - **Mixed groups** (`--mixed-prob` 0.08): a bubble, caption, sign or block with Hunter
    lines and foreign lines.
  - **Extra foreign elements** (`--foreign-extra` scales them):
    - manga: labels, translated SFX, extra bubbles, page numbers;
    - frame: background signage, credits, timestamps, subtitles in English, romaji or
      Japanese, text blocks;
    - doc: paragraphs, labels, signs.
  - **Icons** (`--icon-extra` scales them): single icons, rows of 3-12 icons, grids of rows,
    and icon + label buttons on signs, 10-64 px.
  - **Ordering:** elements that may hold Hunter text are placed first. The foreign-only and
    icon extras then take the space left over instead of crowding out labelled lines.
- **The web layout** (`web.rs`, `--mode-p` 4th weight, 15% of pages): a screenshot on a plain
  white, light or (12%) dark background, with sharp axis-aligned text.
  - Page elements: nav bar (site name, links, icon links, icon cluster), breadcrumb, H1 / H2
    (some with an icon), meta line, paragraphs (6-19 px text), bullet, number and icon lists,
    link and URL lines, ruled / zebra tables, images with captions, star ratings, toolbars,
    icon grids with labels, sidebar (contents, numbered or icon links) and footer (social-icon
    row, copyright, link rows).
  - The Hunter content (left out on negatives): a sample block of 2-6 labelled lines, often
    framed like an embedded image and followed by its romaji / kana transliteration; or an
    Omniglot-style chart of Hunter glyph cells, each with its kana and / or romaji label.
  - `--ui-prob` (0.25) of web pages are app / UI screenshots, heavy on toolbars, icon grids
    and icon + label lists.
  - 70% of web pages get the light screenshot degradation (occasional rescale, slight blur,
    contrast, noise, JPEG 50-95), 30% the full camera-like pipeline.
- **Negatives** (`--empty-prob`, 3%): pages with no Hunter text. With fonts, every element on
  them is foreign text or icons; without fonts they are art-only, as before.
- **Sizes:** foreign groups use the Hunter size ranges per element kind, and 30% of labels,
  cells, blocks and captions go down to `--foreign-em-min` (6 px). Web body text is 6-19 px.

## The keep-clear rule

A foreign or icon run must stay at least `keep_margin` × max(its line height, the Hunter
line's height) away from every labelled quad, plus the run's outline or shadow halo. The
default `keep_margin` is 0.3. The check (`Composer::clear_of`) uses exact convex-polygon
distances on the final quads and runs at two points:

- in `try_place`, for every candidate position of a group, against the other class;
- in `draw_lines`, where a line that would break it is skipped.

In mixed groups, the lines of different classes are stacked (keep + 0.15) line heights apart,
so rotation or perspective rarely makes the check drop a line. Containers and occupancy work
as before, so no foreign ink sits on a labelled quad.

The test `foreign_keeps_clear_of_labelled_lines` checks this on 320 pages with more mixing
than the defaults.

## Records

`labels.jsonl` gets two new per-page fields; `lines` is unchanged.

- `"layout"`: `manga`, `frame`, `doc` or `web`.
- `"foreign"`: a list of `{"quad", "text", "script", "font", "orient", "kind"}`, one per run.
  - The quad is the tight box of the run's flattened outlines (plus dilation and a link's
    underline), in the same corner order as the Hunter quads.
  - A line made of an icon and a label gives two runs.
  - A row of icons is one run, with its glyphs separated by spaces in `text`.
  - `kind` is the element: a group kind (`bubble`, `sign`, ...), or a web element (`nav`,
    `para`, `heading`, `table`, `list`, `link`, `sidebar`, `footer`, `toolbar`, `icon_grid`,
    `chart`, `caption`, `rating`, ...).

`meta.json` adds per-split stats: layouts, foreign runs per page, runs by script, kind and font,
foreign text heights, and the quad areas of Hunter lines, foreign text and icons.

## Caveats

- **No complex shaping:** no GPOS kerning, ligatures or vertical alternates beyond the
  rotation and shift rules above. Japanese and Latin do not need more for this purpose.
- **No hinting:** small text is grayscale anti-aliased, unhinted outlines.
- **Emoji are monochrome silhouettes** (the base glyph's outline), not colour layers.
- **The output depends on the installed fonts.** The same seed with a different font set gives
  different pages, so `meta.json` records the faces that were used.
- **Words are sometimes cut:** long words are clipped by character count when there is no
  space to cut at. A distractor can end mid-word ("WELCO").
