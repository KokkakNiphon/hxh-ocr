"""Convert the release datasets to Parquet shards for the Hugging Face Hub (nothing is uploaded).

Only data rendered without fonts or with openly licensed fonts is released: the guard below
refuses a source whose meta.json lists any font outside rust/hxh-synth/fonts.lock.json (every
file there is OFL-1.1 or Apache-2.0) or any font directory other than fonts/open, and every
page's distractor runs must name a locked font. pages_v3 and pages_foreign (Windows system
fonts) are therefore not released; pages_v3o and pages_foreign_open are the same recipes
re-rendered with the open fonts.

Two dataset repos are built, each as a folder that matches the repo exactly, so
`hf upload-large-folder` can push it as is (see scripts/hf_upload.md):

    release/hf-datasets/hxh-ocr-lines/        line images for the readers
        README.md                              (copied from oss/hf/datasets/hxh-ocr-lines/)
        data/synth_v4/{train,validation}-xxxxx-of-yyyyy.parquet   <- data/synth_v4 (default config)
        data/v5/{train,validation}-...                            <- data/lines_v5b (cut from v5 pages)
    release/hf-datasets/hxh-ocr-pages/        pages for the detector
        README.md
        data/v3o/{train,validation}-...        <- data/pages_v3o (default config)
        data/v2/{train,validation}-...         <- data/pages_v2
        data/foreign_open/validation-...       <- data/pages_foreign_open
        data/v5/{train,validation}-...         <- data/pages_v5 (photo-style)

Anything else under the two data/ folders (shards of sets that are no longer released) is
deleted. Image bytes are embedded unchanged (the generators' JPEGs) in the Hugging Face `Image`
layout (struct<bytes, path>), and the schema carries the `huggingface` features metadata, so
`datasets.load_dataset("parquet", ...)` decodes `image` to a PIL image.

Columns
    lines: image, text, vertical (bool), style        (labels.tsv: path, label, h|v, style; in v5
           the style is the layout of the page the line was cut from, and an empty text marks a
           negative crop: non-Hunter writing or an icon)
    pages: image, width, height, layout, photo, lines, foreign   (labels.jsonl; pages_v2 predates
           `layout` and `foreign`, so they are null / empty there; `photo` is true for v5 pages
           composed as a phone photo of a screen)

Usage (from the repo root, .venv with pyarrow and datasets):

    python scripts/hf_package_datasets.py                       # every source, then verify
    python scripts/hf_package_datasets.py --only v5,lines_v5    # some: synth_v4, lines_v5, v3o, v2, foreign_open, v5
    python scripts/hf_package_datasets.py --verify-only         # re-check existing shards
    python scripts/hf_package_datasets.py --cards-only --namespace <HF_NAMESPACE>

Memory stays low: page labels are kept as raw JSON lines and parsed per row, shards are written
one row group at a time and verified one shard at a time. Shards that already exist with the
expected row count and schema are kept, so an interrupted run resumes. Progress and the summary
go to stdout and release/logs/hf_datasets.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
FONT_LOCK = ROOT / "rust" / "hxh-synth" / "fonts.lock.json"

# (key, repo folder, config, source dataset dir, kind)
SOURCES = [
    ("synth_v4", "hxh-ocr-lines", "synth_v4", "synth_v4", "lines"),
    ("lines_v5", "hxh-ocr-lines", "v5", "lines_v5b", "lines"),
    ("v3o", "hxh-ocr-pages", "v3o", "pages_v3o", "pages"),
    ("v2", "hxh-ocr-pages", "v2", "pages_v2", "pages"),
    ("foreign_open", "hxh-ocr-pages", "foreign_open", "pages_foreign_open", "pages"),
    ("v5", "hxh-ocr-pages", "v5", "pages_v5", "pages"),
]
# rendered with Windows system fonts: never released (open-font replacements: pages_v3o, pages_foreign_open)
NOT_OPEN = {"pages_v3", "pages_foreign"}
# pages_v5 was written by two runs with the same seed (pages are seeded by (seed, split, index)):
# the validation split by pages_v5 itself, the train split by pages_v5_tr (chunked, --resume 1),
# then moved into data/pages_v5/train. Both runs' meta.json are checked.
META_EXTRA = {"pages_v5": ["pages_v5_tr"]}
SPLITS = [("train", "train"), ("val", "validation")]  # source split -> Hub split
LINE_KEYS = {"quad", "text", "orient", "kind"}
FOREIGN_KEYS = {"quad", "text", "script", "font", "orient", "kind"}
PAGE_KEYS = {"image", "width", "height", "layout", "photo", "lines", "foreign"}


def features(kind: str):
    from datasets import Features, Image, List, Value

    quad = List(List(Value("float32"), length=2), length=4)
    if kind == "lines":
        return Features({"image": Image(), "text": Value("string"), "vertical": Value("bool"),
                         "style": Value("string")})
    return Features({
        "image": Image(),
        "width": Value("int32"),
        "height": Value("int32"),
        "layout": Value("string"),
        "photo": Value("bool"),
        "lines": List({"quad": quad, "text": Value("string"), "orient": Value("string"),
                       "kind": Value("string")}),
        "foreign": List({"quad": quad, "text": Value("string"), "script": Value("string"),
                         "font": Value("string"), "orient": Value("string"), "kind": Value("string")}),
    })


# ---------------------------------------------------------------- open-font guard

def font_lock() -> dict:
    return json.loads(FONT_LOCK.read_text(encoding="utf-8"))


def check_open_fonts(name: str, lock: dict) -> dict:
    """What rendered this set: no fonts at all, or only locked open fonts from fonts/open."""
    if name in NOT_OPEN:
        sys.exit(f"{name} was rendered with Windows system fonts and is not released")
    metas = [json.loads((ROOT / "data" / n / "meta.json").read_text(encoding="utf-8"))
             for n in [name] + META_EXTRA.get(name, [])]
    if len(metas) > 1:  # every run that wrote part of the set must agree on its fonts
        assert all(m.get("font_dirs") == metas[0].get("font_dirs") for m in metas), f"{name}: font dirs differ"
    meta = metas[0]
    dirs = meta.get("font_dirs")
    fonts = sorted({f["name"].split("#")[0] for m in metas for f in m.get("fonts", [])})
    if not fonts and not dirs:
        # font-free: pages_v2 (hxh-synth before fonts.rs existed) and synth_v4 (the Python line
        # generator draws only traced strokes; no background images, no corpus)
        args = meta.get("args", {})
        assert not args.get("font-dir") and not args.get("bg_dir"), f"{name}: unexpected font / image inputs {args}"
        return {"fonts": "none (glyph strokes only)"}
    assert dirs == ["fonts/open"], f"{name}: font dirs {dirs}, expected only fonts/open"
    bad = sorted(set(fonts) - set(lock))
    assert not bad, f"{name}: fonts not in {FONT_LOCK.name}: {bad}"
    return {"fonts": f"{len(set(fonts))} open faces from fonts/open, all in fonts.lock.json"}


# ---------------------------------------------------------------- source readers

class LineSource:
    """labels.tsv rows; image bytes from pack.bin (verified against the files) or the files."""

    def __init__(self, split_dir: Path):
        self.dir = split_dir
        self.rows = []
        for ln in (split_dir / "labels.tsv").read_text(encoding="utf-8").splitlines():
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) != 4 or parts[2] not in ("h", "v"):
                raise ValueError(f"{split_dir}/labels.tsv: unexpected row {ln!r}")
            self.rows.append(parts)
        self.pack = None
        idx = split_dir / "pack_index.npy"
        if idx.exists() and (split_dir / "pack.bin").exists():
            index = np.load(idx)
            if len(index) == len(self.rows) and self._pack_ok(index):
                self.index = index
                self.pack = np.memmap(split_dir / "pack.bin", dtype=np.uint8, mode="r")
        neg = sum(1 for r in self.rows if not r[1])
        vert = sum(1 for r in self.rows if r[2] == "v")
        self.stats = {"rows": len(self.rows), "vertical": vert, "negatives": neg}
        print(f"  {split_dir.relative_to(ROOT)}: {len(self.rows)} rows ({vert} vertical, {neg} negatives), "
              f"images from {'pack.bin (spot-checked)' if self.pack is not None else 'files'}", flush=True)

    def _pack_ok(self, index) -> bool:
        mm = np.memmap(self.dir / "pack.bin", dtype=np.uint8, mode="r")
        if int(index[-1, 0] + index[-1, 1]) != mm.shape[0]:
            return False
        rng = random.Random(0)
        for i in [0, len(self.rows) - 1] + rng.sample(range(len(self.rows)), min(64, len(self.rows))):
            off, n = index[i]
            if bytes(mm[off:off + n]) != (self.dir / self.rows[i][0]).read_bytes():
                return False
        return True

    def __len__(self):
        return len(self.rows)

    def nbytes(self) -> int:
        if self.pack is not None:
            return int(self.index[:, 1].sum())
        return sum((self.dir / r[0]).stat().st_size for r in self.rows)

    def image_bytes(self, i: int) -> bytes:
        if self.pack is not None:
            off, n = self.index[i]
            return bytes(self.pack[off:off + n])
        return (self.dir / self.rows[i][0]).read_bytes()

    def image_path(self, i: int) -> str:
        return self.rows[i][0]

    def labels(self, i: int) -> dict:
        _, label, orient, style = self.rows[i]
        return {"text": label, "vertical": orient == "v", "style": style}

    def row(self, i: int) -> dict:
        return {"image": {"bytes": self.image_bytes(i), "path": self.rows[i][0]}, **self.labels(i)}


class PageSource:
    """labels.jsonl kept as raw lines (parsed per row, to keep memory low)."""

    def __init__(self, split_dir: Path, lock: dict):
        self.dir = split_dir
        self.raw = [ln for ln in (split_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        n_lines = n_foreign = n_photo = 0
        for ln in self.raw:
            p = json.loads(ln)
            extra = set(p) - PAGE_KEYS
            if extra:
                raise ValueError(f"{split_dir}: unexpected page keys {extra}")
            for l in p["lines"]:
                if set(l) != LINE_KEYS:
                    raise ValueError(f"{split_dir}: unexpected line keys {set(l)}")
            for f in p.get("foreign", []):
                if set(f) != FOREIGN_KEYS:
                    raise ValueError(f"{split_dir}: unexpected foreign keys {set(f)}")
                if f["font"] and f["font"].split("#")[0] not in lock:  # "" = drawn shape, no font
                    raise ValueError(f"{split_dir}: distractor font {f['font']!r} is not in {FONT_LOCK.name}")
            n_lines += len(p["lines"])
            n_foreign += len(p.get("foreign", []))
            n_photo += bool(p.get("photo"))
        self.stats = {"rows": len(self.raw), "lines": n_lines, "foreign": n_foreign, "photo": n_photo}
        print(f"  {split_dir.relative_to(ROOT)}: {len(self.raw)} pages, {n_lines} lines, {n_foreign} foreign runs, "
              f"{n_photo} photo pages", flush=True)

    def __len__(self):
        return len(self.raw)

    def page(self, i: int) -> dict:
        return json.loads(self.raw[i])

    def nbytes(self) -> int:
        return sum((self.dir / json.loads(ln)["image"]).stat().st_size for ln in self.raw)

    def image_path(self, i: int) -> str:
        return self.page(i)["image"]

    def labels(self, i: int, p: dict | None = None) -> dict:
        p = p or self.page(i)
        return {
            "width": p["width"], "height": p["height"], "layout": p.get("layout"), "photo": bool(p.get("photo", False)),
            "lines": [{"quad": l["quad"], "text": l["text"], "orient": l["orient"], "kind": l["kind"]}
                      for l in p["lines"]],
            "foreign": [{k: f[k] for k in ("quad", "text", "script", "font", "orient", "kind")}
                        for f in p.get("foreign", [])],
        }

    def row(self, i: int) -> dict:
        p = self.page(i)
        return {"image": {"bytes": (self.dir / p["image"]).read_bytes(), "path": p["image"]}, **self.labels(i, p)}


def open_source(kind: str, split_dir: Path, lock: dict):
    return LineSource(split_dir) if kind == "lines" else PageSource(split_dir, lock)


# ---------------------------------------------------------------- writing

def same_schema(a: pa.Schema, b: pa.Schema) -> bool:
    """Fields and the schema-level metadata (the `huggingface` features) equal."""
    return a.equals(b, check_metadata=False) and dict(a.metadata or {}) == dict(b.metadata or {})


def out_dir(out_root: Path, repo: str, config: str) -> Path:
    return out_root / repo / "data" / config


def shard_paths(d: Path, hub_split: str, n: int) -> list[Path]:
    return [d / f"{hub_split}-{i:05d}-of-{n:05d}.parquet" for i in range(n)]


def write_split(src, kind: str, d: Path, hub_split: str, shard_mb: float, log: dict) -> list[Path]:
    total = src.nbytes()
    n = max(1, math.ceil(total / (shard_mb * 1e6)))
    paths = shard_paths(d, hub_split, n)
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob(f"{hub_split}-*"):  # shards from a run with another shard count, temp files
        if stale not in paths:
            stale.unlink()
    schema = features(kind).arrow_schema
    rg = 1000 if kind == "lines" else 100  # row-group size: a few MB, good for the Hub viewer
    bounds = np.linspace(0, len(src), n + 1).astype(int)
    t0, done = time.time(), 0
    for k, path in enumerate(paths):
        a, b = int(bounds[k]), int(bounds[k + 1])
        if path.exists():
            try:
                with pq.ParquetFile(path) as pf:  # closed before a rewrite (Windows cannot replace an open file)
                    ok = pf.metadata.num_rows == b - a and same_schema(pf.schema_arrow, schema)
            except Exception:
                ok = False
            if ok:
                print(f"    keep {path.name} ({b - a} rows)", flush=True)
                done += b - a
                continue
        tmp = path.with_suffix(".parquet.tmp")
        with pq.ParquetWriter(tmp, schema, compression="snappy") as w:
            for s in range(a, b, rg):
                rows = [src.row(i) for i in range(s, min(b, s + rg))]
                w.write_table(pa.Table.from_pylist(rows, schema=schema), row_group_size=rg)
                done += len(rows)
                del rows
                if (s - a) % (rg * 20) == 0:
                    el = time.time() - t0
                    print(f"    {path.name}: {done}/{len(src)} rows, {el:.0f}s, "
                          f"{done / max(el, 1e-6):.0f} rows/s", flush=True)
        tmp.replace(path)
        print(f"    wrote {path.name}: {b - a} rows, {path.stat().st_size / 1e6:.0f} MB", flush=True)
    log.update(rows=len(src), image_bytes=total, shards=n, seconds=round(time.time() - t0, 1),
               parquet_bytes=sum(p.stat().st_size for p in paths), stats=src.stats)
    return paths


def prune(out_root: Path, keep_dirs: dict[str, set[str]]):
    """Delete shards of configs that are no longer released (e.g. v3 and foreign, Windows fonts)."""
    for repo, configs in keep_dirs.items():
        data = out_root / repo / "data"
        if not data.exists():
            continue
        for p in sorted(data.iterdir()):
            if p.is_dir() and p.name in configs:
                continue
            print(f"  prune {p.relative_to(out_root)}", flush=True)
            shutil.rmtree(p) if p.is_dir() else p.unlink()


# ---------------------------------------------------------------- verification

def verify_split(src, kind: str, paths: list[Path], log: dict, sample: int = 200):
    """Row counts, schema, every label, and a sample of image bytes against the source, one shard
    at a time; then load the shards with `datasets` (streaming) and decode an image."""
    counts = []
    for p in paths:
        with pq.ParquetFile(p) as pf:
            counts.append(pf.metadata.num_rows)
    assert sum(counts) == len(src), f"row count {sum(counts)} != source {len(src)}"
    schema = features(kind).arrow_schema
    cols = ["text", "vertical", "style"] if kind == "lines" else ["width", "height", "layout", "photo", "lines", "foreign"]
    rng = random.Random(1)
    sample_idx = set([0, len(src) - 1] + rng.sample(range(len(src)), min(sample, len(src))))
    off = 0
    for path, cnt in zip(paths, counts):
        pf = pq.ParquetFile(path)
        assert same_schema(pf.schema_arrow, schema), f"{path.name}: schema differs"
        got = pq.read_table(path, columns=cols).to_pylist()
        for j, g in enumerate(got):
            i = off + j
            exp = src.labels(i)
            if kind == "lines":
                assert g == exp, f"row {i}: {g} != {exp}"
            else:
                assert (g["width"], g["height"], g["layout"], g["photo"]) == \
                    (exp["width"], exp["height"], exp["layout"], exp["photo"]), i
                for key in ("lines", "foreign"):
                    assert len(g[key]) == len(exp[key]), (i, key)
                    for gl, el in zip(g[key], exp[key]):
                        assert {k: v for k, v in gl.items() if k != "quad"} == \
                            {k: v for k, v in el.items() if k != "quad"}, (i, key)
                        assert np.allclose(np.array(gl["quad"]), np.array(el["quad"]), atol=1e-3), (i, key)
        del got
        rg_rows = pf.metadata.row_group(0).num_rows
        for i in sorted(k for k in sample_idx if off <= k < off + cnt):
            j = i - off
            img = pf.read_row_group(j // rg_rows, columns=["image"]).column("image")[j % rg_rows].as_py()
            rel = src.image_path(i)
            assert img["path"] == rel, (i, img["path"], rel)
            assert hashlib.sha1(img["bytes"]).digest() == hashlib.sha1((src.dir / rel).read_bytes()).digest(), i
        pf.close()
        off += cnt
    import datasets
    ds = datasets.load_dataset("parquet", data_files={"x": [str(p) for p in paths]}, split="x", streaming=True)
    first = next(iter(ds))
    im = first["image"]
    assert hasattr(im, "size"), "image did not decode to PIL"
    log.update(verified=True, rows_checked=len(src), images_checked=len(sample_idx),
               features=list(ds.features), first_image_size=list(im.size), shard_rows=counts)
    print(f"    verified: {sum(counts)} rows = source, labels equal, {len(sample_idx)} images byte-equal, "
          f"datasets decodes {list(ds.features)} (first image {im.size})", flush=True)


# ---------------------------------------------------------------- cards and summary

def copy_cards(out_root: Path, namespace: str | None):
    for repo in sorted({s[1] for s in SOURCES}):
        card = ROOT / "oss" / "hf" / "datasets" / repo / "README.md"
        if not card.exists():
            print(f"  (no card yet: {card.relative_to(ROOT)})")
            continue
        text = card.read_text(encoding="utf-8")
        if namespace:
            text = text.replace("<HF_NAMESPACE>", namespace)
        dst = out_root / repo / "README.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        print(f"  card -> {dst.relative_to(ROOT)}")


def summary(out_root: Path) -> dict:
    res, total = {}, 0
    for repo in sorted({s[1] for s in SOURCES}):
        files = [p for p in (out_root / repo).rglob("*") if p.is_file()]
        b = sum(p.stat().st_size for p in files)
        per = {}
        for p in files:
            if p.suffix == ".parquet":
                key = f"{p.parent.name}/{p.name.split('-')[0]}"
                per[key] = per.get(key, 0) + p.stat().st_size
        res[repo] = {"bytes": b, "files": len(files), "per_split": per}
        total += b
        print(f"  {repo}: {b / 1e9:.2f} GB in {len(files)} files: "
              + ", ".join(f"{k} {v / 1e9:.2f} GB" for k, v in sorted(per.items())), flush=True)
    res["total_bytes"] = total
    print(f"  total: {total / 1e9:.2f} GB", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "release" / "hf-datasets"))
    ap.add_argument("--only", default="", help="comma list of: " + ", ".join(s[0] for s in SOURCES))
    ap.add_argument("--shard-mb", type=float, default=400.0, help="target shard size (image bytes)")
    ap.add_argument("--namespace", default=None, help="Hugging Face user or org; fills <HF_NAMESPACE> in the cards")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--cards-only", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    args = ap.parse_args()

    out_root = Path(args.out)
    data_root = Path(args.data_root)
    if args.cards_only:
        copy_cards(out_root, args.namespace)
        return
    keys = [k for k in args.only.split(",") if k] or [s[0] for s in SOURCES]
    unknown = set(keys) - {s[0] for s in SOURCES}
    if unknown:
        sys.exit(f"unknown --only keys: {sorted(unknown)}")
    out_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_root).free / 1e9
    print(f"free space on the output drive: {free:.0f} GB (need about 15 GB)", flush=True)
    if free < args.min_free_gb and not args.verify_only:
        sys.exit(f"not enough free space: {free:.0f} GB < {args.min_free_gb} GB")

    lock = font_lock()
    log_path = ROOT / "release" / "logs" / "hf_datasets.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    keep: dict[str, set[str]] = {}
    for s in SOURCES:
        keep.setdefault(s[1], set()).add(s[2])
    # drop log entries of sets that are no longer released
    report = {k: v for k, v in report.items() if k.count("/") == 2 and k.split("/")[1] in keep.get(k.split("/")[0], ())}
    if not args.verify_only:
        prune(out_root, keep)
    for key, repo, config, name, kind in SOURCES:
        if key not in keys:
            continue
        fonts = check_open_fonts(name, lock)
        print(f"== {name} -> {repo} (config {config}); fonts: {fonts['fonts']}", flush=True)
        for src_split, hub_split in SPLITS:
            sd = data_root / name / src_split
            if not sd.exists():
                continue
            src = open_source(kind, sd, lock)
            if len(src) == 0:
                continue
            log = report.setdefault(f"{repo}/{config}/{hub_split}", {})
            log.update(source=f"data/{name}/{src_split}", **fonts)
            d = out_dir(out_root, repo, config)
            if args.verify_only:
                n = len(list(d.glob(f"{hub_split}-*-of-*.parquet")))
                paths = shard_paths(d, hub_split, n)
            else:
                paths = write_split(src, kind, d, hub_split, args.shard_mb, log)
            if not args.no_verify:
                verify_split(src, kind, paths, log)
            del src
            log_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    copy_cards(out_root, args.namespace)
    report["_summary"] = summary(out_root)
    log_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
