"""Collect out/*/metrics.json into a markdown table (printed to stdout)."""
from __future__ import annotations

import json

from spike_common import SPIKE


def row(tag: str, m: dict) -> str:
    c, h = m["calib"], m["holdout"]
    return (
        f"| {tag} | {m['sizes']['espdl']:,} | {m['quantize_seconds']:.0f} | "
        f"{c['cos_mean']:.5f} | {c['snr_db_mean']:.1f} | {c['argmax_agree_mean'] * 100:.2f} | {c['greedy_seq_identical'] * 100:.1f} | "
        f"{h['cos_mean']:.5f} | {h['snr_db_mean']:.1f} | {h['argmax_agree_mean'] * 100:.2f} | {h['greedy_seq_identical'] * 100:.1f} | "
        f"{h['fp32_top1_margin_median']:.3f} | {_cer(m)} |"
    )


def _cer(m: dict) -> str:
    c = m.get("holdout_cer")
    if not c:
        return "-"
    return " / ".join(f"{c[k] * 100:.1f}" for k in ("fp32", "ppq_int8", "qdq_ort") if k in c)


def main():
    print(
        "| run | .espdl B | quant s | calib cos | calib SNR dB | calib argmax % | calib seq % | "
        "hold cos | hold SNR dB | hold argmax % | hold seq % | fp32 margin (med) | hold CER % fp32 / int8 / QDQ |"
    )
    print("|" + "---|" * 13)
    for p in sorted((SPIKE / "out").glob("*/metrics.json")):
        print(row(p.parent.name, json.loads(p.read_text())))
    print()
    print("| run | QDQ ORT vs ESP-PPQ sim: hold argmax % | max abs diff | QDQ ORT vs fp32: hold argmax % | seq % |")
    print("|---|---|---|---|---|")
    for p in sorted((SPIKE / "out").glob("*/metrics.json")):
        m = json.loads(p.read_text())
        if "qdq_ort_vs_ppq_holdout" in m:
            a, b = m["qdq_ort_vs_ppq_holdout"], m["qdq_ort_vs_fp32_holdout"]
            print(
                f"| {p.parent.name} | {a['argmax_agree_mean'] * 100:.2f} | {a['max_abs_err']:.4f} | "
                f"{b['argmax_agree_mean'] * 100:.2f} | {b['greedy_seq_identical'] * 100:.1f} |"
            )


if __name__ == "__main__":
    main()
