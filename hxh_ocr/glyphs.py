"""Hunter x Hunter alphabet: glyph stroke definitions.

The Hunter alphabet (ハンター文字) is a one-to-one substitution cipher for Japanese
kana. Glyphs were traced by hand from the fan chart on Hunterpedia
(File:Hunterjap.png, 966x526 px). Coordinates below are pixel positions read off
zoomed crops of that chart; each entry carries the zoom factor of its crop so it
can be mapped back to chart pixels, then normalised to em units
(EM_PX chart pixels = 1 em) centred on the glyph's bounding box. Relative glyph
sizes are kept as drawn in the chart (e.g. い is a lone dot, て is short).

Voiced kana are the base glyph plus a small filled dot at the lower right
(dakuten); p-kana add a small hollow ring there (handakuten). Small kana and
katakana are written with the same glyphs, so labels are normalised to
full-size hiragana.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Zoom factors of the crops the coordinates were read from (crop zoom x screenshot scale).
F1 = 4.6 * 0.78125
F2 = 4.4 * 0.78125
EM_PX = 32.0  # chart pixels per em; a full-height glyph is ~0.9 em
REF_STROKE_EM = 2.3 / EM_PX  # stroke width in the chart, in em


def L(*pts):  # open polyline
    return ("path", list(pts))


def C(*pts):  # closed polygon outline
    return ("closed", list(pts))


def O(cx, cy, rx, ry=None):  # hollow circle / ellipse
    return ("ellipse", cx, cy, rx, rx if ry is None else ry)


def D(cx, cy, r):  # filled dot
    return ("dot", cx, cy, r)


def S(*pts):  # smooth curve through points (Catmull-Rom)
    return ("spline", list(pts))


def A(cx, cy, rx, ry, a0, a1):
    """Arc inside a path. Angles in degrees, y axis down (90 = down, 270 = up)."""
    return ("arc", cx, cy, rx, ry, a0, a1)


def P(*items):  # path made of points and arcs
    return ("path", list(items))


# kana -> (zoom factor, primitives). Order follows the source chart.
_RAW: dict[str, tuple[float, list]] = {
    "ー": (F1, [L((8, 40), (40, 84), (8, 127)), L((40, 84), (105, 84)), L((138, 40), (105, 84), (138, 127))]),
    "あ": (F1, [L((237, 35), (285, 140), (333, 35)), L((253, 70), (317, 70))]),
    "ち": (F1, [L((500, 35), (500, 110)), L((455, 78), (548, 78)), D(500, 123, 22)]),
    "え": (F1, [C((718, 35), (657, 138), (780, 138))]),
    "は": (F1, [L((22, 215), (132, 215)), L((22, 302), (132, 302)), L((77, 215), (77, 302))]),
    "へ": (F1, [L((240, 212), (335, 212)), C((287, 237), (240, 285), (335, 285))]),
    "ひ": (F1, [L((461, 215), (461, 302), (550, 302)), D(522, 240, 22)]),
    "ほ": (F1, [O(728, 260, 57), L((671, 260), (785, 260)), L((728, 203), (728, 317))]),
    "ふ": (F1, [P((22, 390), (107, 390), A(107, 432.5, 42.5, 42.5, -90, 90), (22, 475))]),
    "い": (F1, [D(310, 432, 22)]),
    "か": (F1, [L((462, 463), (566, 463)), L((465, 400), (512, 460), (558, 400))]),
    "け": (F1, [L((690, 380), (690, 485)), L((690, 435), (725, 435)), C((725, 435), (770, 388), (770, 483))]),
    "き": (F2, [L((48, 35), (48, 133)), D(75, 83, 22)]),
    "こ": (F2, [L((240, 88), (240, 38), (323, 38), (323, 108)), O(298, 108, 23)]),
    "く": (F2, [P((430, 38), (430, 115), A(446, 115, 16, 16, 180, 0), (462, 38)),
               L((462, 88), (513, 38)), L((466, 92), (513, 130))]),
    # ま's zigzag points LEFT (Ƹ) in both official sources (printed guidebook chart and the 2011
    # anime chart); the Hunterpedia fan chart traced here showed it pointing right (Σ), kept below
    # as a rare variant. Coordinates are the Hunterpedia trace mirrored about its centre.
    "ま": (F2, [L((712, 42), (620, 42)), L((710, 42), (642, 85), (710, 127)), L((712, 127), (620, 127))]),
    "め": (F2, [L((35, 203), (128, 297)), L((38, 297), (128, 203)), L((128, 203), (128, 297))]),
    "み": (F2, [L((238, 240), (238, 297)), L((275, 240), (275, 297)), L((315, 240), (315, 297)), D(275, 213, 15)]),
    "も": (F2, [L((425, 213), (513, 213)), L((425, 246), (513, 246)), L((468, 188), (468, 275)), D(468, 290, 22)]),
    "む": (F2, [L((640, 190), (596, 287), (660, 287)),
               P((660, 240), (660, 278), A(682.5, 278, 22.5, 22.5, 180, 0), (705, 240))]),
    "ん": (F2, [D(50, 388, 22), L((45, 465), (127, 360))]),
    "な": (F2, [L((218, 370), (320, 370), (220, 457), (320, 457))]),
    "ね": (F2, [C((413, 380), (413, 448), (522, 380), (522, 448))]),
    "に": (F2, [L((630, 373), (690, 373)), D(660, 413, 22), L((617, 453), (705, 453))]),
    "の": (F2, [O(66, 63, 24), S((138, 35), (128, 80), (100, 112), (40, 130))]),
    "ぬ": (F2, [P((225, 143), (225, 58), A(266, 58, 41, 41, 180, 360), (307, 143)), L((227, 55), (307, 143))]),
    "を": (F2, [L((498, 35), (402, 130), (522, 130)), L((444, 88), (522, 88))]),
    "お": (F2, [O(666, 82, 55)]),
    "ら": (F2, [L((128, 205), (128, 292), (43, 292))]),
    "れ": (F2, [C((230, 201), (318, 201), (272, 240)), L((272, 240), (272, 297), (320, 250))]),
    "り": (F2, [D(488, 225, 32), L((465, 270), (432, 302)), L((513, 270), (546, 302))]),
    "ろ": (F2, [O(680, 222, 31), L((628, 302), (652, 254), (708, 254), (735, 302))]),
    "る": (F2, [P((78, 365), (78, 440), A(58, 440, 20, 20, 0, 90), (42, 460)),
               P((105, 365), (105, 440), A(125, 440, 20, 20, 180, 0), (145, 365))]),
    "さ": (F2, [P((246, 365), (246, 445), A(265.5, 445, 19.5, 19.5, 180, 0), (285, 385),
                 A(311, 385, 26, 26, 180, 360), (337, 460))]),
    "せ": (F2, [L((405, 413), (440, 362), (487, 465), (525, 413)), L((405, 413), (527, 413))]),
    "し": (F2, [L((37, 32), (125, 32)), L((37, 67), (125, 67)), D(80, 110, 22)]),
    "そ": (F2, [O(265, 57, 28), O(265, 108, 29)]),
    "す": (F2, [L((410, 38), (458, 38), (513, 132)), S((410, 128), (445, 127), (463, 110), (477, 82))]),
    "た": (F2, [L((643, 40), (643, 125)), L((593, 125), (695, 125))]),
    "て": (F2, [L((25, 203), (118, 203)), C((25, 226), (118, 226), (72, 273))]),
    "と": (F2, [O(272, 248, 34, 36), L((272, 197), (272, 293))]),
    "つ": (F2, [P((424, 188), (424, 263), A(470.5, 263, 46.5, 46.5, 180, 0), (517, 188)),
               L((456, 192), (456, 255)), L((487, 192), (487, 255))]),
    "う": (F2, [P((621, 310), (621, 229), A(665, 229, 44, 44, 180, 360), (709, 310))]),
    "わ": (F2, [L((38, 465), (76, 362), (120, 465)), L((80, 465), (118, 362), (162, 465))]),
    "や": (F2, [L((243, 362), (296, 413), (243, 465)), L((296, 413), (342, 413))]),
    "よ": (F2, [O(472, 388, 33), L((472, 421), (472, 470))]),
    "ゆ": (F2, [L((632, 368), (688, 368), (688, 398)),
               P((637, 460), (637, 425), A(684.5, 425, 47.5, 27, 180, 360), (732, 460))]),
}

# Alternative forms seen in other references. Omniglot's chart (omniglot.com/conscripts/hunter.htm,
# drawn from a fan font) ends も in a hollow ring where the Hunterpedia chart has a filled dot.
#
# Official references (data/real/reference/): the printed guidebook chart draws も with a filled
# dot and せ wide, like Hunterpedia; the 2011 anime chart restyles も with a ring and せ narrow
# (the long diagonal near vertical, a "4" look). Both forms of each are generated. ま points left
# in both official sources; the right-pointing Σ from the fan chart is kept as a rare variant.
_VARIANTS: dict[str, list[tuple[float, list]]] = {
    "も": [(F2, [L((425, 213), (513, 213)), L((425, 246), (513, 246)), L((468, 188), (468, 274)), O(468, 292, 18)])],
    "ま": [(F2, [L((620, 42), (712, 42)), L((622, 42), (690, 85), (622, 127)), L((620, 127), (712, 127))])],
    "せ": [(F2, [L((429.4, 413), (450.4, 362), (478.6, 465), (501.4, 413)), L((427, 413), (505, 413))])],
    # ー as fan fonts draw it: smaller than
    # the chart's >—<, about 0.55 em tall, with a short bar, so it reads almost as "x". The second
    # form has no bar left at all. The chart form stays the main one.
    "ー": [(F1, [L((30, 52), (58, 84), (30, 116)), L((58, 84), (86, 84)), L((114, 52), (86, 84), (114, 116))]),
          (F1, [L((44, 58), (70, 84), (44, 110)), L((70, 84), (76, 84)), L((102, 58), (76, 84), (102, 110))])],
}
# sampling weights per form (main form first); unlisted glyphs pick forms uniformly
_VARIANT_WEIGHTS: dict[str, list[float]] = {"ま": [0.85, 0.15], "ー": [0.7, 0.2, 0.1]}

DAKUTEN = {
    "が": "か", "ぎ": "き", "ぐ": "く", "げ": "け", "ご": "こ",
    "ざ": "さ", "じ": "し", "ず": "す", "ぜ": "せ", "ぞ": "そ",
    "だ": "た", "ぢ": "ち", "づ": "つ", "で": "て", "ど": "と",
    "ば": "は", "び": "ひ", "ぶ": "ふ", "べ": "へ", "ぼ": "ほ",
    "ゔ": "う",
}
HANDAKUTEN = {"ぱ": "は", "ぴ": "ひ", "ぷ": "ふ", "ぺ": "へ", "ぽ": "ほ"}

# Gojuon order, used for the model vocabulary. The last token is a word space
# (Omniglot's sample text separates words with spaces).
SPACE = " "
VOCAB: list[str] = list(
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
    "がぎぐげござじずぜぞだぢづでどばびぶべぼゔぱぴぷぺぽー"
) + [SPACE]
GLYPH_CHARS = [c for c in VOCAB if c != SPACE]
BASE_CHARS = [c for c in VOCAB if c in _RAW]

# Mark geometry relative to the base glyph's bounding box (em), measured on the chart.
DAKUTEN_DX, DAKUTEN_DY, DAKUTEN_R = 0.21, -0.07, 0.065
HANDAKUTEN_DX, HANDAKUTEN_DY, HANDAKUTEN_R = 0.24, -0.09, 0.095


@dataclass
class Stroke:
    kind: str                 # "line" or "dot"
    pts: np.ndarray           # (N, 2) polyline, or (1, 2) centre for a dot
    closed: bool = False
    r: float = 0.0            # dot radius
    width_scale: float = 1.0  # relative stroke width (handakuten ring is thinner)
    hollow: bool = False      # dot drawn as a ring (handakuten)

    def copy(self) -> "Stroke":
        return Stroke(self.kind, self.pts.copy(), self.closed, self.r, self.width_scale, self.hollow)


@dataclass
class Glyph:
    char: str
    base: str
    mark: str | None          # None, "dakuten" or "handakuten"
    strokes: list[Stroke] = field(default_factory=list)
    variants: list[list[Stroke]] = field(default_factory=list)  # all forms, strokes == variants[0]
    weights: list[float] | None = None  # sampling probability per form (None = uniform)


def _arc_pts(cx, cy, rx, ry, a0, a1, step_deg=4.0):
    n = max(2, int(abs(a1 - a0) / step_deg) + 1)
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def _spline(pts, samples=10):
    p = np.asarray(pts, float)
    p = np.vstack([p[:1], p, p[-1:]])
    out = []
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    for i in range(len(p) - 3):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2:-1])
    return np.vstack(out)


def _prims_to_strokes(prims) -> list[Stroke]:
    strokes = []
    for p in prims:
        kind = p[0]
        if kind == "path":
            pts = []
            for item in p[1]:
                if isinstance(item, tuple) and item and item[0] == "arc":
                    pts.extend(_arc_pts(*item[1:]))
                else:
                    pts.append(item)
            strokes.append(Stroke("line", np.asarray(pts, float)))
        elif kind == "closed":
            strokes.append(Stroke("line", np.asarray(p[1], float), closed=True))
        elif kind == "ellipse":
            _, cx, cy, rx, ry = p
            pts = _arc_pts(cx, cy, rx, ry, 0, 360, step_deg=6)[:-1]
            strokes.append(Stroke("line", pts, closed=True))
        elif kind == "spline":
            strokes.append(Stroke("line", _spline(p[1])))
        elif kind == "dot":
            _, cx, cy, r = p
            strokes.append(Stroke("dot", np.array([[cx, cy]], float), r=float(r)))
        else:
            raise ValueError(kind)
    return strokes


def strokes_bbox(strokes: list[Stroke]) -> tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for s in strokes:
        if s.kind == "dot":
            (cx, cy), r = s.pts[0], s.r
            xs0.append(cx - r); xs1.append(cx + r); ys0.append(cy - r); ys1.append(cy + r)
        else:
            xs0.append(s.pts[:, 0].min()); xs1.append(s.pts[:, 0].max())
            ys0.append(s.pts[:, 1].min()); ys1.append(s.pts[:, 1].max())
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _build_base(f: float, prims) -> list[Stroke]:
    strokes = _prims_to_strokes(prims)
    x0, y0, x1, y1 = strokes_bbox(strokes)
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    k = 1.0 / (f * EM_PX)
    for s in strokes:
        s.pts = (s.pts - c) * k
        s.r *= k
    return strokes


def mark_stroke(mark: str, bbox, scale: float = 1.0) -> Stroke:
    """Dakuten / handakuten placed at the lower right of a base glyph bbox (em)."""
    _, _, x1, y1 = bbox
    if mark == "dakuten":
        c = (x1 + DAKUTEN_DX * scale, y1 + DAKUTEN_DY * scale)
        return Stroke("dot", np.array([c], float), r=DAKUTEN_R * scale)
    c = (x1 + HANDAKUTEN_DX * scale, y1 + HANDAKUTEN_DY * scale)
    return Stroke("dot", np.array([c], float), r=HANDAKUTEN_R * scale, hollow=True, width_scale=0.65)


def _build_all() -> dict[str, Glyph]:
    base = {ch: [_build_base(*_RAW[ch])] + [_build_base(*v) for v in _VARIANTS.get(ch, [])] for ch in _RAW}
    w = _VARIANT_WEIGHTS.get
    glyphs = {ch: Glyph(ch, ch, None, base[ch][0], base[ch], w(ch)) for ch in _RAW}
    for ch, b in DAKUTEN.items():
        glyphs[ch] = Glyph(ch, b, "dakuten", base[b][0], base[b], w(b))
    for ch, b in HANDAKUTEN.items():
        glyphs[ch] = Glyph(ch, b, "handakuten", base[b][0], base[b], w(b))
    return glyphs


GLYPHS: dict[str, Glyph] = _build_all()
assert set(GLYPHS) == set(GLYPH_CHARS), set(GLYPH_CHARS) ^ set(GLYPHS)
assert len(VOCAB) == 74

# --- label normalisation -------------------------------------------------------
# Small kana are drawn with the same glyph (often smaller, as in Omniglot's chart);
# labels always use the full-size kana.
SMALL_TO_FULL = dict(zip("ぁぃぅぇぉっゃゅょゎゕゖ", "あいうえおつやゆよわかけ"))
_OLD = {"ゐ": "い", "ゑ": "え"}


def normalize(text: str, keep_small: bool = False) -> str:
    """Katakana -> hiragana, small kana -> full size (unless keep_small), whitespace -> single space.

    Characters outside the vocab are kept; callers decide whether to drop them.
    """
    out = []
    for ch in text:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:  # katakana block
            ch = chr(o - 0x60)
        elif ch == "ヷ":
            ch = "わ"
        if not keep_small:
            ch = SMALL_TO_FULL.get(ch, ch)
        ch = _OLD.get(ch, ch)
        if ch in "‐－−～〜":
            ch = "ー"
        if ch.isspace():
            ch = SPACE
            if not out or out[-1] == SPACE:
                continue
        out.append(ch)
    return "".join(out).strip()


_MACRON = dict(zip("aiueo", "āīūēō"))


def to_romaji(text: str) -> str:
    """Rough Hepburn romanisation, handy for previews and the demo UI.

    The long-vowel mark ー puts a macron on the vowel before it (はんたー -> hantā), as in
    modified Hepburn. Doubling the vowel instead (hantaa) read like a misrecognised あ.
    """
    table = {}
    rows = {
        "": "あいうえお", "k": "かきくけこ", "s": "さしすせそ", "t": "たちつてと", "n": "なにぬねの",
        "h": "はひふへほ", "m": "まみむめも", "r": "らりるれろ", "g": "がぎぐげご", "z": "ざじずぜぞ",
        "d": "だぢづでど", "b": "ばびぶべぼ", "p": "ぱぴぷぺぽ",
    }
    for cons, kana in rows.items():
        for v, ch in zip("aiueo", kana):
            table[ch] = cons + v
    table.update({"し": "shi", "ち": "chi", "つ": "tsu", "ふ": "fu", "じ": "ji", "ぢ": "ji", "づ": "zu",
                  "や": "ya", "ゆ": "yu", "よ": "yo", "わ": "wa", "を": "wo", "ん": "n", "ゔ": "vu"})
    out = []
    for ch in text:
        if ch == "ー":
            prev = out[-1] if out else ""
            if prev and prev[-1] in _MACRON:
                out[-1] = prev[:-1] + _MACRON[prev[-1]]
            elif not (prev and prev[-1] in "āīūēō"):  # a second ー adds nothing
                out.append("-")
        else:
            out.append(table.get(ch, ch))
    return "".join(out)
