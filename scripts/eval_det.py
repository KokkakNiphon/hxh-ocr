"""Evaluate the line detector on hxh-synth pages (640 letterbox), detection and end-to-end.

  python scripts/eval_det.py --ckpt runs/det_v1/best.pt                    # PyTorch fp32
  python scripts/eval_det.py --onnx exports/det_v1.onnx                      # onnxruntime fp32
  python scripts/eval_det.py --outputs int8.npy --offset 800 --n 100         # saved detector outputs
  python scripts/eval_det.py --ckpt runs/det_v1/best.pt --tune               # sweep post-processing

Detection: precision / recall / F1 at quad IoU >= 0.5 (cv2.intersectConvexConvex, greedy one-to-one
matching), orientation accuracy on matched lines, mean IoU.

End to end: every GT line counts. A matched GT line is read from its matched predicted quad (crop:
perspective warp of the quad, grown by --pad-x / --pad-y line heights, to an upright strip; vertical
quads come out rotated 90 deg counter-clockwise, because their first corner is the column's top-right).
Strips go through the recogniser's own preprocessing (height 48, INTER_AREA, width capped, right pad by
edge replication). A missed GT line counts as len(text) deletions. CER = edits / GT characters, for
horizontal and vertical lines separately. "oracle" = the same, read from the GT quads (recogniser-only
error with these crops).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hxh_ocr.det_data import LETTERBOX, PageIndex, letterbox, normalize_u8  # noqa: E402
from hxh_ocr.det_post import PostParams, decode, match, reading_order  # noqa: E402
from hxh_ocr.metrics import edit_distance  # noqa: E402


# ----------------------------------------------------------------------------
# detector outputs
# ----------------------------------------------------------------------------
class TorchDet:
    def __init__(self, ckpt, device):
        from hxh_ocr.det_model import load_det_checkpoint
        self.model, self.cfg = load_det_checkpoint(ckpt, device)
        self.device = device

    @torch.no_grad()
    def __call__(self, xs_u8):
        x = torch.from_numpy(normalize_u8(xs_u8))[:, None].to(self.device)
        return self.model(x).float().cpu().numpy()


class OnnxDet:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, xs_u8):
        x = normalize_u8(xs_u8)[:, None]
        return np.concatenate([self.sess.run(None, {self.name: x[i:i + 1]})[0] for i in range(len(x))])


# ----------------------------------------------------------------------------
# recogniser
# ----------------------------------------------------------------------------
def crop_strip(img: np.ndarray, q: np.ndarray, pad_x: float, pad_y: float) -> np.ndarray:
    """Upright strip of quad q (clockwise from the reading top-left), grown by pad_x / pad_y line
    heights along / across the reading direction, at the quad's native resolution."""
    q = q.astype(np.float64)
    top, bot = q[1] - q[0], q[2] - q[3]
    lw = 0.5 * (np.hypot(*top) + np.hypot(*bot))
    lh = 0.5 * (np.hypot(*(q[3] - q[0])) + np.hypot(*(q[2] - q[1])))
    u = top + bot
    u = u / max(np.hypot(*u), 1e-9)
    v = (q[3] - q[0]) + (q[2] - q[1])
    v = v / max(np.hypot(*v), 1e-9)
    ex, ey = pad_x * lh, pad_y * lh
    src = np.array([q[0] - u * ex - v * ey, q[1] + u * ex - v * ey,
                    q[2] + u * ex + v * ey, q[3] - u * ex + v * ey], np.float32)
    W = max(2, int(round(lw + 2 * ex)))
    H = max(2, int(round(lh + 2 * ey)))
    # pixel-centre convention: the strip's outer edges are at -0.5 and W - 0.5
    dst = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class Recognizer:
    def __init__(self, ckpt, device):
        from hxh_ocr.engine import load_checkpoint
        self.model, cfg = load_checkpoint(ckpt, device)
        self.tokens = cfg["tokens"]
        self.H = cfg.get("height", 32)
        self.max_w = cfg.get("max_width") or 16 * self.H
        self.device = device

    def prep(self, strip: np.ndarray) -> np.ndarray:
        """hxh_ocr/data.py eval preprocessing: height H (INTER_AREA), aspect kept, width capped."""
        h, w = strip.shape
        nw = int(max(8, min(round(w * self.H / h), self.max_w)))
        img = cv2.resize(strip, (nw, self.H), interpolation=cv2.INTER_AREA)
        return normalize_u8(img)

    @torch.no_grad()
    def __call__(self, strips: list[np.ndarray], bs: int = 64) -> list[str]:
        from hxh_ocr.decode import decode_batch
        xs = [self.prep(s) for s in strips]
        order = np.argsort([x.shape[1] for x in xs])
        out = [""] * len(xs)
        for s in range(0, len(xs), bs):
            ids = order[s:s + bs]
            ws = [xs[i].shape[1] for i in ids]
            W = int(math.ceil(max(ws) / 8) * 8)
            batch = np.empty((len(ids), 1, self.H, W), np.float32)
            for k, i in enumerate(ids):
                w = xs[i].shape[1]
                batch[k, 0, :, :w] = xs[i]
                batch[k, 0, :, w:] = xs[i][:, -1:]
            logits = self.model(torch.from_numpy(batch).to(self.device)).float().cpu()
            lens = [min(math.ceil(w / 4), logits.shape[-1]) for w in ws]
            for i, t in zip(ids, decode_batch(logits, self.tokens, lens)):
                out[i] = t
        return out


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def detection_stats(pages, iou=0.5):
    """pages: list of (gt_quads, gt_vert, dets) -> metrics dict + per-page matches."""
    tp = fp = fn = 0
    o_ok = o_n = 0
    ious = []
    by = {"h": [0, 0], "v": [0, 0]}  # [matched, total] by GT orientation
    matches = []
    for gq, gv, dets in pages:
        m = match(gq.astype(np.float64), [d.quad for d in dets], iou)
        matches.append(m)
        tp += len(m); fp += len(dets) - len(m); fn += len(gq) - len(m)
        for g, j, v in m:
            ious.append(v)
            o_ok += int(dets[j].vertical == bool(gv[g])); o_n += 1
        for o in ("h", "v"):
            sel = gv if o == "v" else ~gv
            by[o][1] += int(sel.sum())
            by[o][0] += sum(1 for g, _, _ in m if bool(gv[g]) == (o == "v"))
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-9),
            "orient_acc": o_ok / max(o_n, 1), "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "recall_h": by["h"][0] / max(by["h"][1], 1), "recall_v": by["v"][0] / max(by["v"][1], 1),
            "gt_h": by["h"][1], "gt_v": by["v"][1]}, matches


def e2e(idx: PageIndex, pages, matches, rec_h, rec_v, pad_x, pad_y, oracle=True):
    """End-to-end CER over all GT lines, split by GT orientation (missed lines = deletions).

    A matched line is read by the recogniser of its *predicted* orientation (as the deployed
    pipeline would); the oracle reads the GT quad with the recogniser of the GT orientation."""
    texts = idx.texts
    recs = {"h": rec_h, "v": rec_v}
    jobs = {"h": [], "v": []}  # recogniser -> [(kind, gt_global_index, strip, page, det_index)]
    for p, ((gq, gv, dets), m) in enumerate(zip(pages, matches)):
        if not m and not oracle:
            continue
        img = idx.read(p)
        a, _ = idx.line_range(p)
        for g, j, _ in m:
            jobs["v" if dets[j].vertical else "h"].append(("pred", a + g, crop_strip(img, dets[j].quad, pad_x, pad_y), p, j))
        if oracle:
            for g in range(len(gq)):
                jobs["v" if gv[g] else "h"].append(("gt", a + g, crop_strip(img, gq[g], pad_x, pad_y), p, -1))
    read = {}  # (kind, gt_global_index) -> hyp ; None if no recogniser for it
    hyps = {}
    for o in ("h", "v"):
        if recs[o] is None or not jobs[o]:
            for kind, gi, _, p, j in jobs[o]:
                read[(kind, gi)] = None
            continue
        out = recs[o]([job[2] for job in jobs[o]])
        for (kind, gi, _, p, j), hyp in zip(jobs[o], out):
            read[(kind, gi)] = hyp
            if kind == "pred":
                hyps[(p, j)] = hyp
    res = {}
    for o in ("h", "v"):
        st = {"chars": 0, "lines": 0, "err": 0, "missed": 0, "exact": 0, "oerr": 0, "oexact": 0,
              "unread": 0, "ounread": 0}
        for p, (gq, gv, _) in enumerate(pages):
            a, _ = idx.line_range(p)
            for g in range(len(gq)):
                if bool(gv[g]) != (o == "v"):
                    continue
                ref = str(texts[a + g])
                st["chars"] += len(ref); st["lines"] += 1
                if ("pred", a + g) not in read:
                    st["missed"] += len(ref); st["err"] += len(ref)
                elif read[("pred", a + g)] is None:
                    st["unread"] += 1
                else:
                    e = edit_distance(ref, read[("pred", a + g)])
                    st["err"] += e; st["exact"] += int(e == 0)
                if oracle:
                    hyp = read.get(("gt", a + g))
                    if hyp is None:
                        st["ounread"] += 1
                    else:
                        e = edit_distance(ref, hyp)
                        st["oerr"] += e; st["oexact"] += int(e == 0)
        ok = st["unread"] == 0 and st["chars"] > 0
        ook = oracle and st["ounread"] == 0 and st["chars"] > 0
        res[o] = {"chars": st["chars"], "lines": st["lines"],
                  "cer": st["err"] / st["chars"] if ok else None,
                  "cer_matched_only": (st["err"] - st["missed"]) / max(st["chars"] - st["missed"], 1) if ok else None,
                  "deleted_chars_missed_lines": st["missed"],
                  "exact_lines": st["exact"] / max(st["lines"], 1) if ok else None,
                  "oracle_cer": st["oerr"] / st["chars"] if ook else None,
                  "oracle_exact_lines": st["oexact"] / max(st["lines"], 1) if ook else None,
                  "unread_lines": st["unread"]}
    return res, hyps


def run_detector(det, idx: PageIndex, bs=8, store=False):
    """-> list of (outputs or None, sx, sy, u8)"""
    res = []
    for s in range(0, len(idx), bs):
        batch, sc = [], []
        for i in range(s, min(s + bs, len(idx))):
            u8, sx, sy = letterbox(idx.read(i), LETTERBOX)
            batch.append(u8); sc.append((sx, sy))
        outs = det(np.stack(batch))
        for o, (sx, sy) in zip(outs, sc):
            res.append((o.astype(np.float16) if store else o, sx, sy))
    return res


def figure(idx, pages, path, rec_h, rec_v, pad_x, pad_y, n=6, cols=2):
    """Val pages with every predicted quad (green horizontal, magenta vertical, red dot = first
    corner, number = reading order) and what the recognisers read from each crop (GT in brackets
    when the detection matches a GT line at IoU >= 0.5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Yu Gothic", "MS Gothic", "Meiryo", "Noto Sans CJK JP"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 11, rows * 8),
                             gridspec_kw={"width_ratios": [3, 1.6] * cols})
    pick = [i for i in range(len(pages)) if 8 <= len(pages[i][0]) <= 20][:n]
    for k, p in enumerate(pick):
        ax, tx = axes.flat[2 * k], axes.flat[2 * k + 1]
        img = idx.read(p)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        gq, gv, dets = pages[p]
        m = {j: g for g, j, _ in match(gq.astype(np.float64), [d.quad for d in dets], 0.5)}
        a, _ = idx.line_range(p)
        texts = {}
        for o, rec in (("h", rec_h), ("v", rec_v)):
            js = [j for j, d in enumerate(dets) if d.vertical == (o == "v")]
            if rec is not None and js:
                for j, t in zip(js, rec([crop_strip(img, dets[j].quad, pad_x, pad_y) for j in js])):
                    texts[j] = t
        order, blocks = reading_order(dets)
        lines = []
        for r, j in enumerate(order):
            d = dets[j]
            q = np.vstack([d.quad, d.quad[:1]])
            col = "#d020d0" if d.vertical else "#10b010"
            ax.plot(q[:, 0], q[:, 1], color=col, lw=1.2)
            ax.plot(d.quad[0, 0], d.quad[0, 1], "o", color="red", ms=2.5)
            ax.text(d.quad[0, 0], d.quad[0, 1] - 2, str(r + 1), color="red", fontsize=8,
                    ha="left", va="bottom", fontweight="bold")
            ref = str(idx.texts[a + m[j]]) if j in m else None
            mark = "" if ref is None else ("  ok" if texts.get(j) == ref else f"  [{ref}]")
            lines.append(f"{r + 1:2d} {'v' if d.vertical else 'h'} {texts.get(j, '?')}{mark if ref else '  (no GT match)'}")
        ax.set_title(f"val page {p}: {len(dets)} detected, {len(gq)} GT lines, {len(m)} matched", fontsize=11)
        ax.axis("off")
        tx.axis("off")
        tx.text(0, 1, "\n".join(lines[:48]), va="top", ha="left", fontsize=9, transform=tx.transAxes)
    for ax_ in axes.flat[2 * len(pick):]:
        ax_.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--onnx")
    src.add_argument("--outputs", help=".npy [N, 2, 320, 320] detector outputs for pages offset..offset+N")
    ap.add_argument("--data", default="data/pages_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rec", default="runs/v4_mnv4s_h48/best.pt")
    ap.add_argument("--rec-vert", default="runs/v4_mnv4s_h48_vert/best.pt")
    ap.add_argument("--pad-x", type=float, default=0.3)
    ap.add_argument("--pad-y", type=float, default=0.2)
    ap.add_argument("--box-thresh", type=float, default=None)
    ap.add_argument("--unclip-scale", type=float, default=None)
    ap.add_argument("--unclip", default=None)
    ap.add_argument("--unclip-ratio", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--tune", action="store_true", help="sweep post-processing on the pages, print a table")
    ap.add_argument("--tune-crop", action="store_true", help="sweep crop padding with GT quads")
    ap.add_argument("--figure", default=None, help="e.g. previews/det_val.png")
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    idx = PageIndex([Path(args.data) / args.split], limit=args.n, offset=args.offset, keep_text=True)
    t0 = time.time()
    if args.outputs:
        arr = np.load(args.outputs, mmap_mode="r")
        assert len(arr) >= len(idx), (arr.shape, len(idx))
        outs = []
        for i in range(len(idx)):
            w, h = idx.sizes[i]
            s = LETTERBOX / max(w, h)
            nw, nh = max(1, min(LETTERBOX, round(w * s))), max(1, min(LETTERBOX, round(h * s)))
            outs.append((np.asarray(arr[i], np.float32), nw / w, nh / h))
    else:
        det = TorchDet(args.ckpt, device) if args.ckpt else OnnxDet(args.onnx)
        outs = run_detector(det, idx, store=args.tune)
    t_det = time.time() - t0

    post = PostParams()
    for k in ("box_thresh", "unclip_scale", "unclip", "unclip_ratio", "min_area"):
        if getattr(args, k) is not None:
            setattr(post, k, getattr(args, k))

    def gt(i):
        return idx.lines(i)

    if args.tune:
        grid = [("exact", s, b, 3) for s in (0.65, 0.7, 0.75) for b in (0.35, 0.4, 0.45, 0.5, 0.55)]
        grid += [("ratio", r, b, 3) for r in (1.25, 1.5) for b in (0.4, 0.5)]
        grid += [("exact", 0.8, 0.6, 2), ("exact", 0.8, 0.6, 5)]
        for mode, sv, bt, ma in grid:
            pp = PostParams(unclip=mode, unclip_scale=sv if mode == "exact" else 1.0, unclip_ratio=sv,
                            box_thresh=bt, min_area=ma)
            pages = [(*gt(i), decode(np.asarray(o, np.float32), sx, sy, pp)) for i, (o, sx, sy) in enumerate(outs)]
            m, _ = detection_stats(pages)
            print(f"{mode:6s} {sv:6.2f}       {bt:4.2f}      {ma:2d}     | {m['precision']:.4f} {m['recall']:.4f} "
                  f"{m['f1']:.4f} {m['mean_iou']:.3f} {m['orient_acc']:.4f}", flush=True)
        return

    pages = [(*gt(i), decode(o, sx, sy, post)) for i, (o, sx, sy) in enumerate(outs)]
    det_m, matches = detection_stats(pages)
    result = {"pages": len(idx), "offset": args.offset, "post": vars(post), "detection": det_m,
              "detector_seconds": t_det}
    print(f"pages {len(idx)} (offset {args.offset}); detector {t_det:.0f}s")
    print(f"detection @IoU0.5: P {det_m['precision']:.4f} R {det_m['recall']:.4f} F1 {det_m['f1']:.4f} | "
          f"orientation acc {det_m['orient_acc']:.4f} | mean IoU {det_m['mean_iou']:.3f} | "
          f"recall h {det_m['recall_h']:.4f} ({det_m['gt_h']}) v {det_m['recall_v']:.4f} ({det_m['gt_v']}) | "
          f"tp {det_m['tp']} fp {det_m['fp']} fn {det_m['fn']}", flush=True)

    if args.tune_crop:
        rec = Recognizer(args.rec, device)
        hp = [(gq, gv, []) for gq, gv, _ in pages]
        for px_, py_ in [(0.1, 0.1), (0.2, 0.15), (0.3, 0.2), (0.4, 0.25), (0.3, 0.3), (0.5, 0.3)]:
            r, _ = e2e(idx, hp, [[] for _ in pages], rec, None, px_, py_, oracle=True)
            print(f"pad_x {px_} pad_y {py_}: oracle CER h {r['h']['oracle_cer']:.4f} exact {r['h']['oracle_exact_lines']:.4f}",
                  flush=True)
        return

    hyps = {}
    rec_h = rec_v = None
    if not args.no_e2e or args.figure:
        rec_h = Recognizer(args.rec, device) if Path(args.rec).exists() else None
        rec_v = Recognizer(args.rec_vert, device) if args.rec_vert and Path(args.rec_vert).exists() else None
    if not args.no_e2e:
        result["recognizers"] = {"h": args.rec if rec_h else None, "v": args.rec_vert if rec_v else None}
        r, hyps = e2e(idx, pages, matches, rec_h, rec_v, args.pad_x, args.pad_y, oracle=not args.no_oracle)
        result["e2e"] = r
        for o in ("h", "v"):
            x = r[o]
            if x.get("cer") is None:
                print(f"e2e {o}: no recogniser ({x['chars']} chars)")
                continue
            print(f"e2e {o}: CER {x['cer']:.4f} (matched lines only {x['cer_matched_only']:.4f}, "
                  f"{x['deleted_chars_missed_lines']} chars in missed lines) exact lines {x['exact_lines']:.4f} | "
                  f"oracle (GT quads) CER {x['oracle_cer'] if x['oracle_cer'] is None else round(x['oracle_cer'], 4)} | "
                  f"{x['lines']} lines, {x['chars']} chars", flush=True)
    if args.figure:
        figure(idx, pages, args.figure, rec_h, rec_v, args.pad_x, args.pad_y)
        print(f"figure -> {args.figure}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"detection": det_m, "e2e": result.get("e2e")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
