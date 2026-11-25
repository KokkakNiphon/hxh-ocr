"""Text sources for synthetic lines: uniform kana, frequency-weighted kana, and real words.

A mix keeps every class well represented (uniform) while also showing the model
realistic sequences (words, names, loanwords with ー). Users can add a corpus
file of kana/katakana text (one entry per line) with load_corpus().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .glyphs import GLYPH_CHARS, SMALL_TO_FULL, SPACE, normalize

# Rough relative frequencies of kana in Japanese text (small っ folded into つ).
_FREQ = {
    "い": 50, "う": 45, "の": 45, "し": 40, "か": 35, "ん": 35, "た": 33, "に": 33, "て": 30, "と": 30,
    "な": 28, "る": 28, "こ": 25, "す": 24, "は": 24, "つ": 22, "を": 20, "ま": 20, "も": 18, "き": 18,
    "く": 18, "で": 17, "ら": 16, "が": 16, "れ": 14, "り": 14, "あ": 13, "よ": 12, "け": 12, "せ": 12,
    "さ": 12, "お": 12, "だ": 11, "ー": 10, "ど": 9, "ち": 9, "そ": 9, "や": 9, "え": 8, "め": 8, "わ": 8,
    "ろ": 7, "み": 7, "じ": 7, "ご": 6, "ひ": 6, "ね": 6, "ほ": 5, "ば": 5, "ふ": 5, "べ": 4, "ぶ": 4,
    "ゆ": 4, "む": 4, "げ": 3, "び": 3, "ぎ": 3, "ず": 3, "ぐ": 3, "ぼ": 3, "へ": 3, "ざ": 2, "ぜ": 2,
    "ぞ": 2, "ぱ": 2, "ぬ": 1, "ぴ": 1, "ぷ": 1, "ぺ": 1, "ぽ": 1, "づ": 1, "ぢ": 0.5, "ゔ": 0.3,
}
assert set(_FREQ) == set(GLYPH_CHARS)

# Names and terms from the series plus everyday words, signs and loanwords (katakana is fine).
_WORDS = """
ハンター ゴン キルア クラピカ レオリオ ヒソカ イルミ クロロ ネテロ ジン ビスケ カイト パリストン メルエム コムギ
ゾルディック ネン テン ゼツ レン ハツ ギョウ ケン リュウ イン エン グリードアイランド クモ ゲンエイリョダン
キメラアント ハンターライセンス ハンターしけん くじらじま ヨークシン てんくうとうぎじょう カキン アルカ ナニカ
ゴトー ミルキ キキョウ シルバ ゼノ ウイング ズシ ノブナガ フェイタン シャルナーク フィンクス マチ パクノダ
ウボォーギン ボノレノフ コルトピ シズク フランクリン カルト ツェリードニヒ センリツ バショウ ネオン モラウ ノヴ
こんにちは ありがとう さようなら おはよう こんばんは ともだち なまえ しけん ぼうけん せかい やくそく ゆめ
ちから いのち こころ そら うみ やま かわ みち まち くに ひと こども おとうさん おかあさん せんせい がっこう
でんしゃ くるま えき みせ しょくどう ほん しんぶん てがみ じかん きょう あした きのう いま まえ うしろ みぎ
ひだり うえ した なか そと いりぐち でぐち きけん ちゅうい たちいりきんし ようこそ ホテル レストラン カフェ
バー スーパー コーヒー ケーキ ラーメン うどん すし おかね ぎんこう びょういん けいさつ ゆうびんきょく
としょかん こうえん どうぶつえん はくぶつかん くうこう ひこうき ふね しま もり ほうこくしょ ちず たから
かぎ とびら へや まど いえ にわ はな き くさ いぬ ねこ とり さかな くま おおかみ りゅう へび むし
あか あお きいろ みどり しろ くろ ひかり やみ かぜ あめ ゆき ほし つき たいよう ひ みず つち かね
ひとつ ふたつ みっつ よっつ いつつ いち に さん よん ご ろく なな はち きゅう じゅう ひゃく せん まん
すき きらい たのしい かなしい つよい よわい はやい おそい おおきい ちいさい あたらしい ふるい
いく くる みる きく はなす よむ かく たべる のむ ねる おきる はしる あるく たたかう まもる さがす
けいやく せいやく じょうけん ちかい かくご しゅぎょう しあい かち まけ しょうぶ なかま てき しんゆう
ゲーム カード ルール ポイント ボス レベル スキル アイテム ショップ チケット パーティー オークション
メール ニュース テレビ ラジオ パソコン ケータイ アプリ データ パスワード セーブ ロード スタート
""".split()


_I_ROW = set("きしちにひみりぎじぢびぴ")
_SMALLABLE = {"や": "ゃ", "ゆ": "ゅ", "よ": "ょ"}
_FULL_TO_SMALL = {v: k for k, v in SMALL_TO_FULL.items()}


class TextSampler:
    """Samples raw line text. Raw text may contain small kana and spaces; the training
    label is normalize(raw), which maps small kana to full size."""

    def __init__(self, extra_corpus: list[str] | None = None):
        self.chars = GLYPH_CHARS
        w = np.array([_FREQ[c] for c in self.chars], float)
        self.freq_p = w / w.sum()
        words = [normalize(x, keep_small=True) for x in _WORDS]
        words += [normalize(x, keep_small=True) for x in (extra_corpus or [])]
        ok = set(GLYPH_CHARS) | set(SMALL_TO_FULL) | {SPACE}
        self.words = [x for x in words if x and all(c in ok for c in x)]
        self.non_bar = [c for c in self.chars if c != "ー"]

    def _length(self, rng, lo, hi) -> int:
        return int(np.clip(round(rng.gamma(2.2, 3.0)), lo, hi))

    def sample(self, rng: np.random.Generator, min_len: int = 1, max_len: int = 20) -> str:
        u = rng.random()
        n = self._length(rng, min_len, max_len)
        if u < 0.6:
            if u < 0.35:
                s = [self.chars[i] for i in rng.integers(0, len(self.chars), n)]
            else:
                s = [self.chars[i] for i in rng.choice(len(self.chars), n, p=self.freq_p)]
            for i in range(1, len(s)):  # sprinkle small kana where Japanese would have them
                if s[i] in _SMALLABLE and s[i - 1] in _I_ROW and rng.random() < 0.4:
                    s[i] = _SMALLABLE[s[i]]
                elif s[i] == "つ" and rng.random() < 0.2:
                    s[i] = "っ"
            if n >= 4 and rng.random() < 0.15:  # random word breaks
                for _ in range(rng.integers(1, max(2, n // 4))):
                    s.insert(int(rng.integers(1, len(s))), SPACE)
            s = "".join(s)[:max_len]
        else:
            sep = SPACE if rng.random() < 0.35 else ""
            parts, total = [], 0
            while total < n:
                w = self.words[rng.integers(len(self.words))]
                parts.append(w)
                total += len(w) + len(sep)
            s = sep.join(parts)
            if len(s) > max_len:
                start = rng.integers(0, len(s) - max_len + 1) if rng.random() < 0.3 else 0
                s = s[start:start + max_len]
        return self._fix(s, rng)

    def _fix(self, s: str, rng) -> str:
        out = []
        for c in s:
            if c == SPACE and (not out or out[-1] == SPACE):
                continue
            if c == "ー" and (not out or out[-1] in ("ー", SPACE)):
                c = self.non_bar[rng.integers(len(self.non_bar))]
            out.append(c)
        s = "".join(out).strip()
        return s or self.non_bar[rng.integers(len(self.non_bar))]


def label_of(raw: str) -> str:
    """Training label for a raw sampled string."""
    return normalize(raw)


def load_corpus(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [normalize(x, keep_small=True) for x in lines if x.strip()]
