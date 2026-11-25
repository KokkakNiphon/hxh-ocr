"""Greedy CTC decoding.  Kept dependency-free and loop-based so it ports 1:1 to JS / Rust."""
from __future__ import annotations

BLANK = 0


def greedy_decode(ids, blank: int = BLANK) -> list[int]:
    """`ids`: per-timestep argmax class ids.  Collapse repeats, then drop blanks."""
    out = []
    prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def decode_text(ids, tokens: list[str], blank: int = BLANK) -> str:
    """Class ids -> string (class c >= 1 is tokens[c - 1])."""
    return "".join(tokens[c - 1] for c in greedy_decode(ids, blank))


def logits_to_ids(logits):
    """Model output [B, C, 1, T] -> argmax ids as nested lists [B][T]."""
    return logits.argmax(dim=1)[:, 0, :].tolist()


def decode_batch(logits, tokens: list[str], lengths=None) -> list[str]:
    """Decode a batch; `lengths` (optional) = valid timesteps per sample."""
    ids = logits_to_ids(logits)
    if lengths is not None:
        ids = [row[: int(n)] for row, n in zip(ids, lengths)]
    return [decode_text(row, tokens) for row in ids]
