"""Edit distance, alignment ops and CER for CTC evaluation."""
from __future__ import annotations

from collections import Counter


def align(ref: str, hyp: str):
    """Levenshtein distance and one optimal alignment.

    Returns (distance, ops) with ops a list of (ref_char|None, hyp_char|None):
    (a, a) match, (a, b) substitution, (a, None) deletion, (None, b) insertion.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append((ref[i - 1], None)); i -= 1
        else:
            ops.append((None, hyp[j - 1])); j -= 1
    return d[n][m], ops[::-1]


def edit_distance(ref: str, hyp: str) -> int:
    return align(ref, hyp)[0]


def score(refs, hyps):
    """Corpus CER (sum of edits / sum of ref chars) and exact-line accuracy."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    chars = max(1, sum(len(r) for r in refs))
    acc = sum(r == h for r, h in zip(refs, hyps)) / max(1, len(refs))
    return edits / chars, acc


def confusions(refs, hyps, top: int = 20):
    """Most common (ref, hyp) errors; hyp/ref '<none>' marks insertion/deletion."""
    c = Counter()
    for r, h in zip(refs, hyps):
        for a, b in align(r, h)[1]:
            if a != b:
                c[(a or "<none>", b or "<none>")] += 1
    return c.most_common(top)


CROSS = "×"
# the glyphs that look like the cross: ー is a stretched X (>-<), め an X closed on one side, ね a bow tie
CROSS_LOOKALIKES = "ーめね"


def cross_metrics(refs, hyps) -> dict:
    """The cross `×` (readers v7) on positive lines: recall (reference crosses read as crosses),
    precision (read crosses that are crosses), what the missed ones were read as, the look-alike
    glyphs read as crosses, the lines with crosses (their line accuracy) and crosses read on lines
    that have none."""
    ref_x = hit = false = 0
    read_as: Counter = Counter()
    alike = {c: [0, 0] for c in CROSS_LOOKALIKES}  # glyph -> [in the references, read as a cross]
    for r, h in zip(refs, hyps):
        for a, b in align(r, h)[1]:
            if a == CROSS:
                ref_x += 1
                hit += b == CROSS
                if b != CROSS:
                    read_as[b or "<none>"] += 1
            elif b == CROSS:
                false += 1
            if a in alike:
                alike[a][0] += 1
                alike[a][1] += b == CROSS
    xl = [i for i, r in enumerate(refs) if CROSS in r]
    plain = [i for i, r in enumerate(refs) if CROSS not in r]
    return {
        "cross_ref": ref_x, "cross_hit": hit, "cross_recall": hit / max(1, ref_x),
        "cross_false": false, "cross_precision": hit / max(1, hit + false),
        "cross_read_as": dict(read_as.most_common(8)),
        "cross_lines": len(xl), "cross_lines_acc": sum(refs[i] == hyps[i] for i in xl) / max(1, len(xl)),
        "cross_on_plain_lines": sum(CROSS in hyps[i] for i in plain),
        "lookalikes_read_as_cross": {c: {"ref": n, "as_cross": k} for c, (n, k) in alike.items()},
    }
