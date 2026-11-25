// Figures for the non-technical explainer (blog/index.html). Shared helpers come from blog.js
// (window.HXH); glyph shapes from data/glyphs.json (the training generator's own strokes).

import { tinyNet, insideReader } from "./nn.js";

const H = window.HXH;
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
};
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const visibleWatch = (node) => {
  const s = { on: false };
  new IntersectionObserver((es) => { s.on = es.some((e) => e.isIntersecting); }, { threshold: 0.25 }).observe(node);
  return s;
};
const whenGlyphs = () => new Promise((res) => {
  const t = setInterval(() => { const g = H.glyphs(); if (g) { clearInterval(t); res(g); } }, 60);
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// a tiny seeded random generator, so the "wobble" is the same on every visit
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

let G;

/** Glyph (with an optional voiced mark) for a kana; null if it isn't in the alphabet. */
function shapeOf(ch) {
  const comp = G.composed[ch];
  const g = G.glyphs[comp ? comp.base : ch];
  return g ? { form: g.forms[0], mark: comp ? comp.mark : null } : null;
}

/** Draw a glyph form centred at (cx, cy), em size S; jitter (em) wobbles every point. */
function drawShape(parent, shape, S, cx, cy, { jitter = 0, seed = 1, widthScale = 1, cls = "" } = {}) {
  const r = rng(seed), sw = G.stroke_width_em * widthScale;
  const j = () => (r() - 0.5) * 2 * jitter;
  const out = [];
  for (const st of shape.form.strokes) {
    if (st.type === "dot") {
      out.push(el("circle", { cx: cx + (st.c[0] + j()) * S, cy: cy + (st.c[1] + j()) * S, r: Math.max(st.r, sw / 2) * S, class: `glyph-fill ${cls}` }, parent));
    } else {
      const pts = st.pts.map(([x, y]) => [x + j(), y + j()]);
      const d = pts.map((p, i) => `${i ? "L" : "M"}${(cx + p[0] * S).toFixed(1)},${(cy + p[1] * S).toFixed(1)}`).join("") + (st.closed ? "Z" : "");
      out.push(el("path", { d, class: `glyph-stroke ${cls}`, "stroke-width": (sw * S).toFixed(2), pathLength: 1 }, parent));
    }
  }
  if (shape.mark) {
    const m = G.marks[shape.mark], [, , x1, y1] = shape.form.bbox;
    const mx = cx + (x1 + m.dx + j()) * S, my = cy + (y1 + m.dy + j()) * S;
    out.push(m.hollow
      ? el("circle", { cx: mx, cy: my, r: Math.max(m.r, sw * m.width_scale) * S, class: `glyph-stroke ${cls}`, "stroke-width": (sw * m.width_scale * S).toFixed(2) }, parent)
      : el("circle", { cx: mx, cy: my, r: Math.max(m.r, sw / 2) * S, class: `glyph-fill ${cls}` }, parent));
  }
  return out;
}

// ---------------------------------------------------------------- 1. decoder (hero)

async function decoder() {
  const host = $("decoder");
  if (!host) return;
  const phrase = "はんたーらいせんす";
  const roma = ["ha", "n", "ta", "ー", "ra", "i", "se", "n", "su"];
  const chars = [...phrase], n = chars.length, cell = 70, W = n * cell + 20, Hh = 228;
  const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}`, class: "decoder-svg" });
  const cols = chars.map((ch, i) => {
    const x = 10 + cell * (i + 0.5);
    const g = el("g", { class: "dec-col" }, svg);
    const strokes = drawShape(g, shapeOf(ch), 50, x, 44, { cls: "dec-ink" });
    const arrow = el("path", { d: `M${x},${84} v14 m-5,-5 l5,5 l5,-5`, class: "dec-arrow" }, g);
    const kana = el("text", { x, y: 136, class: "dec-kana", "text-anchor": "middle" }, g);
    kana.textContent = ch;
    const r = el("text", { x, y: 166, class: "dec-roma", "text-anchor": "middle" }, g);
    r.textContent = roma[i] === "ー" ? "(long)" : roma[i];
    return { strokes, arrow, kana, r };
  });
  const out = el("text", { x: W / 2, y: 212, class: "dec-out", "text-anchor": "middle" }, svg);
  out.textContent = "hantā raisensu  →  “Hunter License”";
  host.append(svg);
  const all = (on) => cols.forEach((c) => {
    c.strokes.forEach((s) => s.classList.toggle("on", on));
    [c.arrow, c.kana, c.r].forEach((e) => e.classList.toggle("on", on));
  });
  if (reduced()) { all(true); out.classList.add("on"); return; }
  const vis = visibleWatch(host);
  for (;;) {
    while (!vis.on) await sleep(300);
    all(false);
    out.classList.remove("on");
    await sleep(500);
    for (const c of cols) { c.strokes.forEach((s) => s.classList.add("on")); await sleep(170); }
    await sleep(400);
    for (const c of cols) { c.arrow.classList.add("on"); c.kana.classList.add("on"); await sleep(170); }
    await sleep(300);
    for (const c of cols) { c.r.classList.add("on"); await sleep(110); }
    out.classList.add("on");
    await sleep(4200);
  }
}

// ---------------------------------------------------------------- 2. find / cut / read

async function threeSteps() {
  const strips = $("step-strips"), reads = $("step-reads");
  if (!strips || !reads) return;
  const d = await H.getJson("img/pipeline/crops.json").catch(() => null);
  const list = d ? d.crops.filter((c) => c.featured).slice(0, 4) : [];
  for (const c of list) {
    const im = document.createElement("img");
    im.src = `img/pipeline/${c.file}`;
    im.alt = `Straightened strip of line ${c.line}`;
    im.loading = "lazy";
    strips.append(im);
    const r = document.createElement("div");
    r.className = "rd";
    r.innerHTML = `<span class="k" lang="ja"></span><span class="r"></span>`;
    r.children[0].textContent = c.text;
    r.children[1].textContent = c.romaji;
    reads.append(r);
  }
}

// ---------------------------------------------------------------- 3. make a training picture

async function maker() {
  const host = $("maker");
  if (!host) return;
  const steps = [
    ["Draw the symbols", "Clean strokes from the alphabet: ご (go) is こ plus the voicing dot, then ん (n)."],
    ["Draw them by hand", "Every point is nudged a little and the pen gets thicker, like lettering drawn by hand."],
    ["Put them somewhere", "A random background: here the dotted screentone used for shading in manga."],
    ["Tilt it", "A slight slant and squash, as if photographed at an angle."],
    ["Blur and add grain", "Out of focus, a little noisy, slightly faded: like a phone photo of a page."],
  ];
  const text = "ごん";
  const W = 520, Hh = 180;
  host.innerHTML = `<div class="mk-stage"></div>
    <div class="mk-controls">
      <button type="button" class="mk-prev" aria-label="Previous step">‹</button>
      <div class="mk-dots" role="tablist"></div>
      <button type="button" class="mk-next" aria-label="Next step">›</button>
      <button type="button" class="mk-play" aria-label="Pause">❚❚</button>
    </div>
    <div class="mk-caption"><span class="mk-n label"></span><strong class="mk-title"></strong><span class="mk-text"></span></div>`;
  const stage = host.querySelector(".mk-stage");
  const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "A training picture being made" });
  const defs = el("defs", {}, svg);
  const pat = el("pattern", { id: "mk-tone", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
  el("circle", { cx: 3.5, cy: 3.5, r: 1.35, fill: "#9aa0aa" }, pat);
  const lg = el("linearGradient", { id: "mk-grad", x1: 0, y1: 0, x2: 1, y2: 1 }, defs);
  el("stop", { offset: "0", "stop-color": "#e9ecf2" }, lg);
  el("stop", { offset: "1", "stop-color": "#c3c8d2" }, lg);
  const f = el("filter", { id: "mk-photo", x: "-5%", y: "-5%", width: "110%", height: "110%" }, defs);
  el("feGaussianBlur", { in: "SourceGraphic", stdDeviation: 1.6, result: "b" }, f);
  el("feTurbulence", { type: "fractalNoise", baseFrequency: 0.9, numOctaves: 2, seed: 7, result: "n" }, f);
  el("feColorMatrix", { in: "n", type: "saturate", values: 0, result: "ng" }, f);
  el("feComposite", { in: "b", in2: "ng", operator: "arithmetic", k1: 0, k2: 0.86, k3: 0.22, k4: -0.06 }, f);

  const paper = el("rect", { x: 0, y: 0, width: W, height: Hh, fill: "#fbfbf8" }, svg);
  const scene = el("g", {}, svg);
  const bg = el("g", { class: "mk-bg" }, scene);
  el("rect", { x: 0, y: 0, width: W, height: Hh, fill: "url(#mk-grad)" }, bg);
  el("rect", { x: 0, y: 0, width: W, height: Hh, fill: "url(#mk-tone)", opacity: 0.55 }, bg);
  el("path", { d: `M-20,${Hh - 30} C120,${Hh - 90} 300,${Hh + 10} ${W + 20},${Hh - 70}`, fill: "none", stroke: "#5b616c", "stroke-width": 3, opacity: 0.45 }, bg);
  const bubble = el("ellipse", { cx: W / 2, cy: Hh / 2, rx: 150, ry: 66, fill: "#fbfbf8", stroke: "#20232a", "stroke-width": 3 }, bg);
  const ink = el("g", { class: "mk-ink" }, scene);
  const clean = el("g", {}, ink), hand = el("g", {}, ink);
  const chars = [...text];
  chars.forEach((ch, i) => {
    const x = W / 2 + (i - (chars.length - 1) / 2) * 96;
    drawShape(clean, shapeOf(ch), 74, x, Hh / 2, {});
    drawShape(hand, shapeOf(ch), 74, x, Hh / 2, { jitter: 0.045, seed: 11 + i * 7, widthScale: 1.45 });
  });
  stage.append(svg);
  void paper; void bubble;

  const dots = host.querySelector(".mk-dots");
  steps.forEach((_, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-label", `Step ${i + 1}: ${steps[i][0]}`);
    b.addEventListener("click", () => { setPlaying(false); show(i); });
    dots.append(b);
  });
  let cur = 0;
  const show = (i) => {
    cur = (i + steps.length) % steps.length;
    clean.style.display = cur >= 1 ? "none" : "";
    hand.style.display = cur >= 1 ? "" : "none";
    bg.style.opacity = cur >= 2 ? "1" : "0";
    scene.style.transform = cur >= 3 ? "translate(14px, 6px) rotate(-5deg) skewX(9deg) scale(0.94, 0.9)" : "none";
    scene.setAttribute("filter", cur >= 4 ? "url(#mk-photo)" : "");
    ink.style.opacity = cur >= 4 ? "0.86" : "1";
    [...dots.children].forEach((b, k) => b.setAttribute("aria-selected", String(k === cur)));
    host.querySelector(".mk-n").textContent = `Step ${cur + 1} of ${steps.length}`;
    host.querySelector(".mk-title").textContent = steps[cur][0];
    host.querySelector(".mk-text").textContent = steps[cur][1];
  };
  let playing = !reduced();
  const play = host.querySelector(".mk-play");
  const setPlaying = (p) => { playing = p; play.textContent = p ? "❚❚" : "▶"; play.setAttribute("aria-label", p ? "Pause" : "Play"); };
  host.querySelector(".mk-prev").addEventListener("click", () => { setPlaying(false); show(cur - 1); });
  host.querySelector(".mk-next").addEventListener("click", () => { setPlaying(false); show(cur + 1); });
  play.addEventListener("click", () => setPlaying(!playing));
  setPlaying(playing);
  show(reduced() ? steps.length - 1 : 0);
  const vis = visibleWatch(host);
  setInterval(() => { if (playing && vis.on) show(cur + 1); }, 2600);
}

// ---------------------------------------------------------------- 4. before / after swipe

function swipe() {
  const host = $("swipe");
  if (!host) return;
  const top = host.querySelector(".swipe-top"), handle = host.querySelector(".swipe-handle"), range = host.querySelector("input");
  const set = (v) => {
    host.style.setProperty("--pos", `${v}%`);
    host.style.setProperty("--w", `${host.clientWidth}px`);
  };
  range.addEventListener("input", () => set(range.value));
  new ResizeObserver(() => set(range.value)).observe(host);
  set(range.value);
  void top; void handle;
  if (reduced()) return;
  // a gentle hint of the gesture once, when it first scrolls into view
  let done = false;
  new IntersectionObserver(async (es) => {
    if (done || !es.some((e) => e.isIntersecting)) return;
    done = true;
    for (const v of [50, 70, 30, 50]) { range.value = v; set(v); await sleep(380); }
  }, { threshold: 0.6 }).observe(host);
}

// ---------------------------------------------------------------- 5. one number, two ways

function bytes() {
  const host = $("bytes");
  if (!host) return;
  const x = 0.7853;
  const dv = new DataView(new ArrayBuffer(4));
  dv.setFloat32(0, x);
  const f32 = dv.getFloat32(0);
  const bits32 = [0, 1, 2, 3].map((i) => dv.getUint8(i).toString(2).padStart(8, "0")).join("");
  const q = Math.max(-128, Math.min(127, Math.round(x * 128)));
  const bits8 = (q & 0xff).toString(2).padStart(8, "0");
  const row = (label, bits, value, note, cls) => {
    const r = document.createElement("div");
    r.className = `byte-row ${cls}`;
    const lab = document.createElement("div");
    lab.className = "byte-label";
    lab.innerHTML = `<strong></strong><span class="label"></span>`;
    lab.children[0].textContent = label;
    lab.children[1].textContent = `${bits.length / 8} byte${bits.length > 8 ? "s" : ""}`;
    const cells = document.createElement("div");
    cells.className = "byte-cells";
    for (let i = 0; i < bits.length; i += 8) {
      const b = document.createElement("div");
      b.className = "byte";
      b.textContent = bits.slice(i, i + 8);
      cells.append(b);
    }
    const v = document.createElement("div");
    v.className = "byte-value";
    v.innerHTML = `<span class="mono"></span><span class="note"></span>`;
    v.children[0].textContent = value;
    v.children[1].textContent = note;
    r.append(lab, cells, v);
    return r;
  };
  host.replaceChildren(
    row("float32", bits32, f32.toPrecision(9), "what models are usually trained with", "f32"),
    row("int8", bits8, `${q} × 1/128 = ${(q / 128).toFixed(4)}`, "what runs on your phone: close enough, four times smaller", "i8"),
  );
}

// ---------------------------------------------------------------- 6. how long it takes

async function timings() {
  const host = $("chart-sizes");
  if (!host) return;
  const p = await H.getJson("img/pipeline/pipeline.json").catch(() => null);
  const t = p && p.timings_ms;
  if (!t) { host.closest("figure").hidden = true; H.numberFigures(); return; }
  H.barChart(host, {
    title: "Time for each step",
    sub: `The example page in this article, ${p.counts.shown} lines; milliseconds on one core of a desktop processor (lower is faster)`,
    unit: " ms", digits: 0,
    rows: [
      { label: "find every line\n(line finder)", value: t.detector },
      { label: "read one line\n(reader)", value: t.read_per_line_mean, hi: true },
      { label: "the whole page,\nevery line read", value: t.page_total },
    ],
  });
}

// ---------------------------------------------------------------- 7. privacy diagram

function privacy() {
  const host = $("privacy");
  if (!host) return;
  const W = 620, Hh = 250;
  const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "Diagram: the photo, the line finder, the reader and the text all stay inside the phone; nothing goes to the internet." });
  // phone
  el("rect", { x: 20, y: 10, width: 400, height: 230, rx: 22, class: "pv-phone" }, svg);
  el("rect", { x: 34, y: 34, width: 372, height: 182, rx: 8, class: "pv-screen" }, svg);
  el("circle", { cx: 220, cy: 22, r: 3, class: "pv-dot" }, svg);
  const box = (x, label, sub) => {
    const g = el("g", {}, svg);
    el("rect", { x, y: 92, width: 78, height: 66, rx: 6, class: "pv-box" }, g);
    const t = el("text", { x: x + 39, y: 121, "text-anchor": "middle", class: "pv-t" }, g);
    t.textContent = label;
    const s = el("text", { x: x + 39, y: 140, "text-anchor": "middle", class: "pv-s" }, g);
    s.textContent = sub;
    return g;
  };
  box(46, "photo", "your picture");
  box(140, "find", "line finder");
  box(234, "read", "two readers");
  const out = el("g", {}, svg);
  el("rect", { x: 328, y: 92, width: 66, height: 66, rx: 6, class: "pv-box pv-out" }, out);
  const k = el("text", { x: 361, y: 130, "text-anchor": "middle", class: "pv-k" }, out);
  k.textContent = "ごん";
  for (const x of [124, 218, 312]) el("path", { d: `M${x},125 h12 m-5,-5 l5,5 l-5,5`, class: "pv-arrow" }, svg);
  const cap = el("text", { x: 220, y: 70, "text-anchor": "middle", class: "pv-cap" }, svg);
  cap.textContent = "inside your browser";
  // the internet, crossed out
  const cl = el("g", {}, svg);
  el("path", { d: "M488,120 a26,26 0 0 1 50,-10 a20,20 0 0 1 36,14 a18,18 0 0 1 -4,36 h-80 a22,22 0 0 1 -2,-40 z", class: "pv-cloud" }, cl);
  el("path", { d: "M470,170 L590,90", class: "pv-cross" }, cl);
  const ct = el("text", { x: 530, y: 196, "text-anchor": "middle", class: "pv-s" }, cl);
  ct.textContent = "no upload";
  el("path", { d: "M420,125 h46", class: "pv-gap" }, svg);
  host.append(svg);
}

// ---------------------------------------------------------------- 8. sideways writing

function turn() {
  const host = $("turn");
  if (!host) return;
  const word = [..."はんたー"];
  const W = 620, Hh = 300, S = 46, cell = 58;
  const svg = el("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "A vertical column of four glyphs turns a quarter turn to the left and becomes a horizontal strip." });
  const colX = 150, colTop = 30;
  // the column, drawn upright (as on the page)
  const g = el("g", { class: "turn-col" }, svg);
  el("rect", { x: colX - cell / 2 - 6, y: colTop - 6, width: cell + 12, height: cell * word.length + 12, rx: 4, class: "turn-frame" }, g);
  word.forEach((ch, i) => drawShape(g, shapeOf(ch), S, colX, colTop + cell * (i + 0.5), { cls: "dec-ink on" }));
  const first = el("circle", { cx: colX + cell / 2 + 14, cy: colTop + cell / 2, r: 4, class: "turn-first" }, g);
  void first;
  // labels
  const t1 = el("text", { x: colX, y: Hh - 12, "text-anchor": "middle", class: "pv-s" }, svg);
  t1.textContent = "on the page: top to bottom";
  const arrow = el("path", { d: `M${colX + 70},${Hh / 2} h60 m-8,-8 l8,8 l-8,8`, class: "pv-arrow turn-arrow" }, svg);
  void arrow;
  const reader = el("g", { class: "turn-reader" }, svg);
  el("rect", { x: 300, y: Hh / 2 - 26, width: 300, height: 52, rx: 4, class: "turn-strip" }, reader);
  const t2 = el("text", { x: 450, y: Hh / 2 + 56, "text-anchor": "middle", class: "pv-s" }, reader);
  t2.textContent = "the vertical reader sees: はんたー";
  host.append(svg);
  // the turn: rotate the column -90° about its centre and move it into the strip
  const cx = colX, cy = colTop + (cell * word.length) / 2;
  const tx = 450 - cx, ty = Hh / 2 - cy;
  g.style.transformOrigin = `${cx}px ${cy}px`;
  g.style.transformBox = "view-box";
  const set = (on) => {
    g.style.transform = on ? `translate(${tx}px, ${ty}px) rotate(-90deg)` : "none";
    reader.classList.toggle("on", on);
  };
  if (reduced()) { set(true); return; }
  const vis = visibleWatch(host);
  let on = false;
  setInterval(() => { if (vis.on) { on = !on; set(on); } }, 2400);
}

// ---------------------------------------------------------------- 9. the vote

function vote() {
  const host = $("vote");
  if (!host) return;
  const looks = ["はんたー", "はんたー", "はだたー", "はんたー", "はんたー"];
  host.innerHTML = `<div class="vote-grid"></div><div class="vote-shown"><span class="label">label shown</span><span class="k" lang="ja"></span></div>
    <p class="vote-note label">illustration</p>`;
  const grid = host.querySelector(".vote-grid"), shown = host.querySelector(".vote-shown .k");
  const cells = looks.map((s, i) => {
    const c = document.createElement("div");
    c.className = "vote-look" + (s !== looks[0] ? " wrong" : "");
    c.innerHTML = `<span class="label">look ${i + 1}</span><span class="k" lang="ja"></span>`;
    c.children[1].textContent = s;
    grid.append(c);
    return c;
  });
  const majority = (k) => {
    const n = new Map();
    let best = looks[0], bn = 0;
    for (const s of looks.slice(0, k)) n.set(s, (n.get(s) || 0) + 1);
    for (const s of looks.slice(0, k)) if (n.get(s) >= bn) { best = s; bn = n.get(s); }
    // ties go to the latest, as in the app; with one wrong look the majority holds
    return [...n.entries()].sort((a, b) => b[1] - a[1])[0][0] || best;
  };
  const showUpTo = (k) => {
    cells.forEach((c, i) => c.classList.toggle("on", i < k));
    shown.textContent = k ? majority(k) : "…";
  };
  if (reduced()) { showUpTo(looks.length); return; }
  const vis = visibleWatch(host);
  let k = 0;
  setInterval(() => { if (vis.on) { k = (k % (looks.length + 2)) + 1; showUpTo(Math.min(k, looks.length)); } }, 900);
}

// ---------------------------------------------------------------- computer-vision lesson figures

/** The real strip the reader saw (img/ctc_strip.png): a 44 x 44 crop around its first symbol,
 *  averaged over 2 x 2 blocks into a 22 x 22 grid of brightness values (0 black .. 255 white). */
let CROP = null;
function loadCrop() {
  if (CROP) return CROP;
  CROP = new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0);
      const X0 = 20, Y0 = 2, S = 44, N = S / 2;
      const d = ctx.getImageData(X0, Y0, S, S).data;
      const grid = [];
      for (let r = 0; r < N; r++) {
        const row = [];
        for (let k = 0; k < N; k++) {
          let s = 0;
          for (const [dy, dx] of [[0, 0], [0, 1], [1, 0], [1, 1]]) s += d[((2 * r + dy) * S + (2 * k + dx)) * 4];
          row.push(Math.round(s / 4));
        }
        grid.push(row);
      }
      res({ grid, n: N, box: { x: X0, y: Y0, w: S, h: S }, stripW: img.naturalWidth, stripH: img.naturalHeight });
    };
    img.onerror = rej;
    img.src = "img/ctc_strip.png";
  });
  return CROP;
}

const ICONS = {
  camera: "M4 8h3l2-3h6l2 3h3v11H4z M12 10.5a3 3 0 1 0 0 6a3 3 0 1 0 0-6",
  grid: "M4 4h16v16H4z M4 9.3h16 M4 14.6h16 M9.3 4v16 M14.6 4v16",
  search: "M10 4a6 6 0 1 0 0 12a6 6 0 1 0 0-12 M14.5 14.5L20 20",
  cut: "M6 7a2 2 0 1 0 0 .01 M6 17a2 2 0 1 0 0 .01 M7.6 8.2L20 17 M7.6 15.8L20 7",
  eye: "M2 12s4-7 10-7s10 7 10 7s-4 7-10 7S2 12 2 12z M12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6",
  text: "M5 6h14 M12 6v13 M8 19h8",
};

function journey() {
  const host = $("journey");
  if (!host) return;
  const stops = [
    ["camera", "Photo", "You take a picture.", "see"],
    ["grid", "Numbers", "The computer sees a grid of numbers.", "see"],
    ["search", "Find lines", "The line finder spots every line of writing.", "find"],
    ["cut", "Cut out", "Each line is cut out and straightened.", "jobs"],
    ["eye", "Read", "The reader turns each strip into kana.", "read"],
    ["text", "Words", "Kana become Latin letters you can read.", "code"],
  ];
  const ol = document.createElement("ol");
  ol.className = "journey-list";
  const items = stops.map(([icon, title, text, target], i) => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `#${target}`;
    a.className = "journey-stop";
    const svg = el("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" });
    el("path", { d: ICONS[icon], class: "journey-ic" }, svg);
    const n = document.createElement("span");
    n.className = "journey-n";
    n.textContent = String(i + 1);
    const t = document.createElement("strong");
    t.textContent = title;
    const p = document.createElement("span");
    p.className = "journey-t";
    p.textContent = text;
    a.append(n, svg, t, p);
    li.append(a);
    ol.append(li);
    return li;
  });
  host.append(ol);
  if (reduced()) return;
  const vis = visibleWatch(host);
  let k = 0;
  setInterval(() => {
    if (!vis.on) return;
    items.forEach((li, i) => li.classList.toggle("on", i === k));
    k = (k + 1) % items.length;
  }, 1100);
}

async function pixels() {
  const host = $("pixels");
  if (!host) return;
  const C = await loadCrop().catch(() => null);
  if (!C) { host.closest("figure").hidden = true; H.numberFigures(); return; }
  host.innerHTML = `<div class="px-strip"></div>
    <div class="px-zoom"><div class="px-grid" role="img" aria-label="A 22 by 22 grid of pixels showing one symbol"></div>
    <div class="px-side"><button type="button" class="px-toggle" aria-pressed="false">Show the numbers</button>
    <p class="px-read label">Point at a pixel</p>
    <div class="px-scale"><span class="px-sw" style="background:#000"></span><span>0 = black</span>
      <span class="px-sw" style="background:#808080"></span><span>128 = gray</span>
      <span class="px-sw" style="background:#fff"></span><span>255 = white</span></div></div></div>`;
  // the whole strip, with the zoomed area marked
  const strip = host.querySelector(".px-strip");
  const sv = el("svg", { viewBox: `0 0 ${C.stripW} ${C.stripH}`, preserveAspectRatio: "none", "aria-hidden": "true" });
  el("image", { href: "img/ctc_strip.png", x: 0, y: 0, width: C.stripW, height: C.stripH, style: "image-rendering: pixelated" }, sv);
  el("rect", { x: C.box.x, y: C.box.y, width: C.box.w, height: C.box.h, class: "px-box" }, sv);
  strip.append(sv);
  const grid = host.querySelector(".px-grid");
  grid.style.setProperty("--n", C.n);
  const read = host.querySelector(".px-read");
  C.grid.forEach((row, r) => row.forEach((v, k) => {
    const c = document.createElement("span");
    c.className = "px";
    c.style.background = `rgb(${v},${v},${v})`;
    c.style.setProperty("--fg", v < 128 ? "#fff" : "#111");
    c.textContent = String(v);
    c.dataset.r = r;
    c.dataset.c = k;
    grid.append(c);
  }));
  grid.addEventListener("pointermove", (e) => {
    const t = e.target.closest(".px");
    if (t) read.textContent = `row ${+t.dataset.r + 1}, column ${+t.dataset.c + 1}: ${t.textContent}`;
  });
  const btn = host.querySelector(".px-toggle");
  btn.addEventListener("click", () => {
    const on = !grid.classList.contains("nums");
    grid.classList.toggle("nums", on);
    btn.setAttribute("aria-pressed", String(on));
    btn.textContent = on ? "Hide the numbers" : "Show the numbers";
  });
}

const FILTERS = [
  { name: "Up-and-down edges", k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], note: "big where dark meets light from left to right" },
  { name: "Across edges", k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], note: "big where dark meets light from top to bottom" },
  { name: "Blur", k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]], note: "adds up the neighbourhood: softens everything" },
];

async function conv() {
  const host = $("conv");
  if (!host) return;
  const C = await loadCrop().catch(() => null);
  if (!C) { host.closest("figure").hidden = true; H.numberFigures(); return; }
  // darkness score 0..9 per pixel: easier numbers for the arithmetic (0 = background, 9 = darkest ink)
  const lo = Math.min(...C.grid.flat()), hi = Math.max(...C.grid.flat());
  const D = C.grid.map((row) => row.map((v) => Math.round(((hi - v) / (hi - lo)) * 9)));
  const N = C.n, M = N - 2;
  host.innerHTML = `<div class="cv-pick" role="radiogroup" aria-label="Filter"></div>
    <div class="cv-row">
      <div class="cv-col"><span class="label">Picture (darkness 0–9)</span><div class="cv-grid cv-in"></div></div>
      <div class="cv-mid"><span class="label">Filter</span><div class="cv-k"></div><div class="cv-math"></div></div>
      <div class="cv-col"><span class="label">Result</span><div class="cv-grid cv-out"></div></div>
    </div>
    <div class="cv-controls"><button type="button" class="cv-play" aria-label="Pause">❚❚</button><span class="cv-note"></span></div>`;
  const gin = host.querySelector(".cv-in"), gout = host.querySelector(".cv-out");
  gin.style.setProperty("--n", N);
  gout.style.setProperty("--n", M);
  const inCells = [], outCells = [];
  D.forEach((row) => row.forEach((d) => {
    const c = document.createElement("span");
    c.style.setProperty("--d", d / 9);
    gin.append(c);
    inCells.push(c);
  }));
  for (let i = 0; i < M * M; i++) {
    const c = document.createElement("span");
    gout.append(c);
    outCells.push(c);
  }
  const win = document.createElement("div");
  win.className = "cv-win";
  gin.append(win);
  const kEl = host.querySelector(".cv-k"), math = host.querySelector(".cv-math"), note = host.querySelector(".cv-note");
  let F = FILTERS[0], out = [], maxAbs = 1, pos = 0, playing = !reduced();
  const compute = () => {
    out = [];
    for (let r = 0; r < M; r++) for (let c = 0; c < M; c++) {
      let s = 0;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += D[r + i][c + j] * F.k[i][j];
      out.push(s);
    }
    maxAbs = Math.max(1, ...out.map(Math.abs));
  };
  const showK = () => {
    kEl.replaceChildren(...F.k.flat().map((w) => {
      const s = document.createElement("span");
      s.textContent = String(w);
      s.className = w > 0 ? "pos" : w < 0 ? "neg" : "";
      return s;
    }));
    note.textContent = F.note;
  };
  const paint = (p) => {
    pos = p;
    const r = Math.floor(p / M), c = p % M;
    win.style.setProperty("--r", r);
    win.style.setProperty("--c", c);
    outCells.forEach((cell, i) => {
      const on = i <= p;
      cell.style.setProperty("--v", on ? Math.abs(out[i]) / maxAbs : 0);
      cell.classList.toggle("cur", i === p);
    });
    const terms = [];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) terms.push(`${D[r + i][c + j]}×${F.k[i][j] < 0 ? `(${F.k[i][j]})` : F.k[i][j]}`);
    math.innerHTML = "";
    const t = document.createElement("div");
    t.className = "cv-terms";
    t.textContent = terms.join(" + ");
    const s = document.createElement("div");
    s.className = "cv-sum";
    s.textContent = `= ${out[p]}`;
    math.append(t, s);
  };
  FILTERS.forEach((f, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = f.name;
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(i === 0));
    b.addEventListener("click", () => {
      F = f;
      host.querySelectorAll(".cv-pick button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
      compute();
      showK();
      paint(reduced() ? M * M - 1 : 0);
    });
    host.querySelector(".cv-pick").append(b);
  });
  // point at the picture or the result to move the window there
  const hover = (e, grid, n, off) => {
    const rc = grid.getBoundingClientRect();
    const c = Math.floor(((e.clientX - rc.left) / rc.width) * n) - off, r = Math.floor(((e.clientY - rc.top) / rc.height) * n) - off;
    if (r >= 0 && c >= 0 && r < M && c < M) { setPlaying(false); paint(r * M + c); }
  };
  gin.addEventListener("pointermove", (e) => hover(e, gin, N, 1));
  gout.addEventListener("pointermove", (e) => hover(e, gout, M, 0));
  const playBtn = host.querySelector(".cv-play");
  const setPlaying = (p) => { playing = p; playBtn.textContent = p ? "❚❚" : "▶"; playBtn.setAttribute("aria-label", p ? "Pause" : "Play"); };
  playBtn.addEventListener("click", () => setPlaying(!playing));
  compute();
  showK();
  setPlaying(playing);
  paint(reduced() ? M * M - 1 : 0);
  const vis = visibleWatch(host);
  let hold = 0;
  setInterval(() => {
    if (!playing || !vis.on) return;
    if (pos >= M * M - 1) { if (++hold > 40) { hold = 0; paint(0); } return; }
    paint(pos + 1);
  }, 45);
}

async function layers() {
  const host = $("layers");
  if (!host) return;
  const C = await loadCrop().catch(() => null);
  const g = G.glyphs["く"], form = g && g.forms[0];
  host.innerHTML = "";
  const row = document.createElement("div");
  row.className = "ly-row";
  const panel = (title, sub, node) => {
    const p = document.createElement("div");
    p.className = "ly-panel";
    const box = document.createElement("div");
    box.className = "ly-box";
    box.append(node);
    const t = document.createElement("strong");
    t.textContent = title;
    const s = document.createElement("span");
    s.className = "label";
    s.textContent = sub;
    p.append(box, t, s);
    return p;
  };
  const canvasOf = (vals, n, color) => {
    const cv = document.createElement("canvas");
    cv.width = n;
    cv.height = n;
    const ctx = cv.getContext("2d");
    const im = ctx.createImageData(n, n);
    vals.forEach((v, i) => {
      const [r, gg, b] = color(v);
      im.data.set([r, gg, b, 255], i * 4);
    });
    ctx.putImageData(im, 0, 0);
    return cv;
  };
  const arrow = () => {
    const a = document.createElement("span");
    a.className = "ly-arrow";
    a.textContent = "→";
    a.setAttribute("aria-hidden", "true");
    return a;
  };
  const parts = [];
  if (C) {
    const N = C.n, flat = C.grid.flat();
    parts.push(panel("Pixels", "the input", canvasOf(flat, N, (v) => [v, v, v])));
    // edge strength: both Sobel filters, combined
    const E = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const at = (y, x) => C.grid[Math.max(0, Math.min(N - 1, y))][Math.max(0, Math.min(N - 1, x))];
      const gx = -at(r - 1, c - 1) - 2 * at(r, c - 1) - at(r + 1, c - 1) + at(r - 1, c + 1) + 2 * at(r, c + 1) + at(r + 1, c + 1);
      const gy = -at(r - 1, c - 1) - 2 * at(r - 1, c) - at(r - 1, c + 1) + at(r + 1, c - 1) + 2 * at(r + 1, c) + at(r + 1, c + 1);
      E.push(Math.hypot(gx, gy));
    }
    const mx = Math.max(...E);
    parts.push(panel("Edges", "early layers", canvasOf(E.map((e) => e / mx), N, (t) => [255 - 220 * t, 255 - 190 * t, 255 - 40 * t])));
  }
  // strokes: little line detectors at several angles, plus a dot detector (a drawing)
  const st = el("svg", { viewBox: "0 0 60 60", "aria-hidden": "true" });
  [[10, 12, 22, 12], [36, 8, 48, 20], [12, 34, 12, 48], [34, 48, 46, 36]].forEach(([a, b, c, d]) => el("path", { d: `M${a},${b}L${c},${d}`, class: "ly-stroke" }, st));
  el("circle", { cx: 44, cy: 46, r: 3.5, class: "ly-dot" }, st);
  parts.push(panel("Strokes", "middle layers", st));
  // parts: the real strokes of く, pulled apart, and the voicing dot
  const pt = el("svg", { viewBox: "0 0 60 60", "aria-hidden": "true" });
  if (form) {
    form.strokes.forEach((s, i) => {
      const d = (s.pts || []).map((p, k) => `${k ? "L" : "M"}${(22 + i * 14 + p[0] * 34).toFixed(1)},${(30 + p[1] * 34).toFixed(1)}`).join("");
      if (d) el("path", { d, class: "ly-stroke" }, pt);
    });
    el("circle", { cx: 52, cy: 50, r: 3.5, class: "ly-dot" }, pt);
  }
  parts.push(panel("Parts", "later layers", pt));
  const sym = el("svg", { viewBox: "0 0 60 60", "aria-hidden": "true" });
  if (form) {
    const shape = { form, mark: "dakuten" };
    drawShape(sym, shape, 40, 28, 30, { cls: "dec-ink on" });
  }
  const symPanel = panel("ぐ (gu)", "the last layer", sym);
  symPanel.classList.add("final");
  parts.push(symPanel);
  parts.forEach((p, i) => { if (i) row.append(arrow()); row.append(p); });
  host.append(row);
}

async function learn() {
  const host = $("learn");
  if (!host) return;
  const d = await H.getJson("data/charts.json").catch(() => null);
  const grp = d && d.curves.find((c) => c.id === "recognizers_v4");
  const ser = grp && grp.series.find((s) => s.id === "v4_mnv4s_h48");
  const pts = ser ? ser.points : [];
  host.innerHTML = `<div class="lr-row"><div class="lr-loop"></div><div class="lr-chart"></div></div>`;
  // the loop
  const L = el("svg", { viewBox: "0 0 240 240", role: "img", "aria-label": "The practice loop: show, guess, check, adjust" });
  el("circle", { cx: 120, cy: 120, r: 80, class: "lr-ring" }, L);
  const steps = [["Show", "examples", 120, 40], ["Guess", "what they say", 200, 120], ["Check", "against answers", 120, 200], ["Adjust", "the numbers", 40, 120]];
  const nodes = steps.map(([t, s, x, y]) => {
    const g = el("g", { class: "lr-node" }, L);
    el("circle", { cx: x, cy: y, r: 27 }, g);
    const a = el("text", { x, y: y - 1, "text-anchor": "middle", class: "lr-t" }, g);
    a.textContent = t;
    const b = el("text", { x, y: y + 11, "text-anchor": "middle", class: "lr-s" }, g);
    b.textContent = s;
    return g;
  });
  const dot = el("circle", { cx: 120, cy: 40, r: 6, class: "lr-dot" }, L);
  const round = el("text", { x: 120, y: 116, "text-anchor": "middle", class: "lr-round" }, L);
  const roundLab = el("text", { x: 120, y: 134, "text-anchor": "middle", class: "lr-s" }, L);
  roundLab.textContent = "practice rounds";
  host.querySelector(".lr-loop").append(L);
  // the chart: real validation error during training
  const W = 360, Hh = 240, l = 40, r = 10, t = 14, b = 34;
  const S = el("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "How many symbols the reader got wrong while it trained" });
  const xMax = pts.length ? pts[pts.length - 1][0] : 30000, yMax = 4;
  const sx = (v) => l + (v / xMax) * (W - l - r), sy = (v) => t + (1 - Math.min(v, yMax) / yMax) * (Hh - t - b);
  for (const v of [0, 1, 2, 3, 4]) {
    el("line", { x1: l, x2: W - r, y1: sy(v), y2: sy(v), class: v ? "lr-ax" : "lr-ax0" }, S);
    const tx = el("text", { x: l - 6, y: sy(v) + 4, "text-anchor": "end", class: "lr-s" }, S);
    tx.textContent = String(v);
  }
  for (const v of [0, 10000, 20000, 30000]) {
    const tx = el("text", { x: sx(v), y: Hh - b + 16, "text-anchor": "middle", class: "lr-s" }, S);
    tx.textContent = v ? `${v / 1000}k` : "0";
  }
  const yl = el("text", { x: l, y: 9, class: "lr-s" }, S);
  yl.textContent = "symbols wrong per 100";
  const xl = el("text", { x: W - r, y: Hh - 2, "text-anchor": "end", class: "lr-s" }, S);
  xl.textContent = "practice rounds";
  const path = el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(""), class: "lr-line" }, S);
  void path;
  const mark = el("circle", { r: 5, class: "lr-dot" }, S);
  const tag = el("text", { class: "lr-tag" }, S);
  host.querySelector(".lr-chart").append(S);
  const showPoint = (i) => {
    if (!pts.length) return;
    const [x, y] = pts[i];
    mark.setAttribute("cx", sx(x));
    mark.setAttribute("cy", sy(y));
    tag.setAttribute("x", Math.min(sx(x) + 8, W - 70));
    tag.setAttribute("y", sy(y) - 10);
    tag.textContent = `${y.toFixed(1)} wrong`;
    round.textContent = x.toLocaleString("en-US");
  };
  let step = 0, i = 0;
  const place = () => {
    const [, , x, y] = steps[step];
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", y - 34);
    nodes.forEach((n, k) => n.classList.toggle("on", k === step));
  };
  showPoint(reduced() ? pts.length - 1 : 0);
  place();
  if (reduced()) return;
  const vis = visibleWatch(host);
  setInterval(() => {
    if (!vis.on) return;
    step = (step + 1) % steps.length;
    place();
    if (step === 0) { i = (i + 1) % Math.max(1, pts.length); showPoint(i); }
  }, 650);
}

// ---------------------------------------------------------------- go

(async () => {
  journey();
  threeSteps();
  swipe();
  bytes();
  timings();
  pixels();
  conv();
  learn();
  insideReader(H);
  G = await whenGlyphs();
  decoder();
  maker();
  layers();
  tinyNet(H, G);
  privacy();
  turn();
  vote();
  H.numberFigures();
})();
