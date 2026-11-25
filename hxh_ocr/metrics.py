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
