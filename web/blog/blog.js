// HxH OCR blog: table of contents, charts (SVG), the glyph chart and the encoder, the pipeline crops.
// Numbers in the charts are copied from docs/research/JOURNAL.md (entry named in each `source`).

const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
};
const text = (parent, x, y, s, cls, anchor = "start") => {
  const t = el("text", { x, y, class: cls, "text-anchor": anchor }, parent);
  t.textContent = s;
  return t;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- table of contents + scrollspy

function buildToc() {
  const ol = $("toc");
  if (!ol) return;
  const heads = [...document.querySelectorAll("article h2[id]")];
  const links = new Map();
  for (const h of heads) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `#${h.id}`;
    a.textContent = h.textContent;
    li.append(a);
    ol.append(li);
    links.set(h, a);
  }
  const visible = new Set();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) e.isIntersecting ? visible.add(e.target) : visible.delete(e.target);
    let current = null;
    for (const h of heads) {
      if (h.getBoundingClientRect().top < window.innerHeight * 0.35) current = h;
    }
    links.forEach((a, h) => a.classList.toggle("active", h === current));
  }, { rootMargin: "0px 0px -50% 0px", threshold: [0, 1] });
  heads.forEach((h) => io.observe(h));
  window.addEventListener("scroll", () => {
    let current = null;
    for (const h of heads) if (h.getBoundingClientRect().top < window.innerHeight * 0.35) current = h;
    links.forEach((a, h) => a.classList.toggle("active", h === current));
  }, { passive: true });
}

// ---------------------------------------------------------------- charts

/** Horizontal bars, optionally in labelled groups: rows = [{label, value, hi?, group?}]. */
function barChart(host, { title, sub, unit = "", rows, max, digits = 2, series }) {
  if (!host) return;
  host.classList.add("chart");
  host.replaceChildren();
  const head = document.createElement("div");
  head.innerHTML = `<div class="fig-title"></div><div class="fig-sub"></div>`;
  head.children[0].textContent = title;
  head.children[1].textContent = sub || "";
  host.append(head);
  if (series) { // legend for paired bars
    const lg = document.createElement("div");
    lg.className = "legend";
    for (const s of series) {
      const i = document.createElement("span");
      const sw = document.createElement("i");
      sw.style.background = s.color;
      i.append(sw, s.name);
      lg.append(i);
    }
    host.append(lg);
  }
  const draw = () => {
    host.querySelector("svg")?.remove();
    const W = Math.max(280, host.clientWidth);
    const narrow = W < 520;
    const labelW = narrow ? Math.min(150, W * 0.42) : 210;
    const valueW = 64;
    const barH = 14, gapRow = 10, gapPair = 4, gapGroup = 20;
    const nVals = series ? series.length : 1;
    const rowH = nVals * barH + (nVals - 1) * gapPair;
    const top = max ?? Math.max(...rows.flatMap((r) => (series ? r.values : [r.value])));
    let y = 4;
    const svg = el("svg", { role: "img", "aria-label": title });
    let lastGroup;
    const items = [];
    for (const r of rows) {
      if (r.group && r.group !== lastGroup) {
        if (lastGroup !== undefined) y += gapGroup - gapRow;
        text(svg, 0, y + 10, r.group, "t-group");
        y += 20;
        lastGroup = r.group;
      }
      items.push({ r, y });
      y += rowH + gapRow;
    }
    const H = y;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const x0 = labelW + 8, x1 = W - valueW;
    for (const t of [0.25, 0.5, 0.75, 1]) el("line", { x1: x0 + (x1 - x0) * t, x2: x0 + (x1 - x0) * t, y1: 0, y2: H - gapRow, class: "ax" }, svg);
    el("line", { x1: x0, x2: x0, y1: 0, y2: H - gapRow, class: "ax-strong" }, svg);
    for (const { r, y: ry } of items) {
      const lines = String(r.label).split("\n");
      lines.forEach((ln, i) => text(svg, 0, ry + rowH / 2 + 4 + (i - (lines.length - 1) / 2) * 13, ln, i ? "t-muted" : "t-label"));
      const vals = series ? r.values : [r.value];
      vals.forEach((v, k) => {
        const by = ry + k * (barH + gapPair);
        const w = Math.max(1.5, ((x1 - x0) * v) / top);
        const b = el("rect", { x: x0, y: by, width: w, height: barH, rx: 1.5, class: "bar" + (r.hi && !series ? " hi" : "") }, svg);
        if (series) b.style.fill = series[k].color;
        text(svg, x0 + w + 6, by + barH - 3, `${v.toFixed(digits)}${unit}`, "t-value");
      });
    }
    host.append(svg);
  };
  draw();
  new ResizeObserver(draw).observe(host);
}

/** Line chart: series = [{name, color, points: [[x, y], ...]}]. */
function lineChart(host, { title, sub, series, xLabel, yLabel, yMax, yUnit = "", xFmt = (v) => v }) {
  if (!host) return;
  host.classList.add("chart");
  host.replaceChildren();
  const head = document.createElement("div");
  head.innerHTML = `<div class="fig-title"></div><div class="fig-sub"></div>`;
  head.children[0].textContent = title;
  head.children[1].textContent = sub || "";
  const lg = document.createElement("div");
  lg.className = "legend";
  for (const s of series) {
    const i = document.createElement("span");
    const sw = document.createElement("i");
    sw.style.background = s.color;
    i.append(sw, s.name);
    lg.append(i);
  }
  host.append(head, lg);
  const draw = () => {
    host.querySelector("svg")?.remove();
    const W = Math.max(280, host.clientWidth), H = Math.round(Math.min(320, Math.max(200, W * 0.5)));
    const L = 44, R = 12, T = 8, B = 30;
    const xs = series.flatMap((s) => s.points.map((p) => p[0]));
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const ys = series.flatMap((s) => s.points.map((p) => p[1]));
    const top = yMax ?? Math.max(...ys) * 1.05;
    const sx = (v) => L + ((v - xMin) / (xMax - xMin || 1)) * (W - L - R);
    const sy = (v) => T + (1 - Math.min(v, top) / top) * (H - T - B);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const v = (top * i) / ticks, y = sy(v);
      el("line", { x1: L, x2: W - R, y1: y, y2: y, class: i ? "ax" : "ax-strong" }, svg);
      text(svg, L - 6, y + 3.5, `${v.toFixed(top < 2 ? 1 : 0)}${yUnit}`, "t-muted", "end");
    }
    for (let i = 0; i <= 4; i++) {
      const v = xMin + ((xMax - xMin) * i) / 4;
      text(svg, sx(v), H - B + 16, xFmt(v), "t-muted", i === 0 ? "start" : i === 4 ? "end" : "middle");
    }
    if (xLabel) text(svg, W - R, H - 2, xLabel, "t-muted", "end");
    if (yLabel) text(svg, L, T + 2, "", "t-muted");
    for (const s of series) {
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
      const path = el("path", { d, class: "line" }, svg);
      path.style.stroke = s.color;
      const last = s.points[s.points.length - 1];
      const dot = el("circle", { cx: sx(last[0]), cy: sy(last[1]), r: 3 }, svg);
      dot.style.fill = s.color;
    }
    host.append(svg);
  };
  draw();
  new ResizeObserver(draw).observe(host);
}

function renderCharts() {
  // colours as CSS variables, so a theme switch recolours the charts without redrawing them
  const accent = "var(--accent)", vert = "var(--vert)", bar = "var(--bar)", alt = "var(--line-strong)";

  barChart($("chart-cross"), {
    title: "Character error rate on each validation set",
    sub: "48 px reader, fp32, 3,000 lines per set (lower is better)",
    unit: "%", digits: 2,
    series: [{ name: "v3 val (clean)", color: alt }, { name: "v4 val (official forms, art, bold)", color: accent }],
    rows: [
      { label: "trained on v3", values: [0.63, 3.07] },
      { label: "trained on v4", values: [0.69, 0.90] },
    ],
  });

  barChart($("chart-ablation"), {
    title: "Validation CER by reader variant",
    sub: "fp32; each round on its own data version, so compare within a round (lower is better)",
    unit: "%", digits: 2,
    rows: [
      { group: "Round 1 · 32 px, v1 data", label: "MobileNetV4, ImageNet", value: 1.16, hi: true },
      { group: "Round 1 · 32 px, v1 data", label: "tiny CNN", value: 1.27 },
      { group: "Round 1 · 32 px, v1 data", label: "MobileNetV4, scratch", value: 1.47 },
      { group: "Round 2 · v3 data", label: "48 px", value: 0.64, hi: true },
      { group: "Round 2 · v3 data", label: "32 px", value: 0.84 },
      { group: "Round 2 · v3 data", label: "32 px, three-stage\n96k params", value: 1.14 },
    ],
  });

  barChart($("chart-speed"), {
    title: "Milliseconds per inference",
    sub: "node 24, one thread of a Ryzen 7 3700X (lower is better)",
    unit: " ms", digits: 1,
    series: [{ name: "scalar", color: alt }, { name: "WebAssembly SIMD", color: accent }],
    rows: [
      { label: "48 px reader\n154M MACs", values: [65.5, 9.5] },
      { label: "32 px reader\n60.5M MACs", values: [25.6, 3.7] },
      { label: "three-stage reader\n30.5M MACs", values: [14.9, 2.0] },
    ],
  });

  barChart($("chart-recall"), {
    title: "Detector recall by line thickness",
    sub: "share of ground-truth lines found at IoU ≥ 0.5; thickness in the 640 px letterbox",
    unit: "%", digits: 0, max: 100,
    rows: [
      { label: "6–9 px", value: 73 },
      { label: "9–14 px", value: 91 },
      { label: "14–25 px", value: 95, hi: true },
      { label: "≥ 25 px", value: 96, hi: true },
    ],
  });

  // training curves come from data/charts.json (scripts/blog/make_assets.py)
  fetch("data/charts.json").then((r) => (r.ok ? r.json() : null)).then((d) => {
    const host = $("chart-curves");
    if (!host) return;
    const group = d && d.curves && d.curves.find((c) => c.id === "recognizers_v4");
    const colors = { v4_mnv4s_h48: accent, v4_mnv4s_h48_vert: vert, v4_mnv4s3_h32: bar };
    const names = { v4_mnv4s_h48: "48 px horizontal", v4_mnv4s_h48_vert: "48 px vertical", v4_mnv4s3_h32: "32 px three-stage" };
    const series = group ? group.series.filter((s) => s.points && s.points.length > 1)
      .map((s) => ({ name: names[s.id] || s.label, color: colors[s.id] || bar, points: s.points })) : [];
    if (!series.length) {
      host.closest("figure").hidden = true;
      numberFigures();
      return;
    }
    lineChart(host, {
      title: "Validation CER during training",
      sub: "synth_v4, fp32 EMA weights with re-estimated BatchNorm, every 1,000 steps (lower is better)",
      series, yMax: 4, yUnit: "%", xLabel: "step",
      xFmt: (v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(Math.round(v))),
    });
  }).catch(() => { hide("chart-curves"); numberFigures(); });
}

// ---------------------------------------------------------------- glyphs (data/glyphs.json)

let G = null; // data/glyphs.json as written by scripts/blog/make_assets.py (see data/README.md)

/** Draw one glyph form centred at (cx, cy), em size S px, into an SVG parent. */
function drawGlyph(parent, form, S, cx, cy, mark = null) {
  const sw = G.stroke_width_em;
  for (const st of form.strokes) {
    if (st.type === "dot") {
      el("circle", { cx: cx + st.c[0] * S, cy: cy + st.c[1] * S, r: Math.max(st.r, sw / 2) * S, class: "glyph-fill" }, parent);
    } else {
      const d = st.pts.map((p, i) => `${i ? "L" : "M"}${(cx + p[0] * S).toFixed(2)},${(cy + p[1] * S).toFixed(2)}`).join("") + (st.closed ? "Z" : "");
      el("path", { d, class: "glyph-stroke", "stroke-width": (sw * S).toFixed(2) }, parent);
    }
  }
  if (mark) { // dakuten dot / handakuten ring at the lower right of the form actually drawn
    const m = G.marks[mark], [, , x1, y1] = form.bbox;
    const mx = cx + (x1 + m.dx) * S, my = cy + (y1 + m.dy) * S;
    if (m.hollow) {
      el("circle", { cx: mx, cy: my, r: Math.max(m.r, sw * m.width_scale) * S, class: "glyph-stroke", "stroke-width": (sw * m.width_scale * S).toFixed(2) }, parent);
    } else {
      el("circle", { cx: mx, cy: my, r: Math.max(m.r, sw / 2) * S, class: "glyph-fill" }, parent);
    }
  }
}

function glyphSvg(kana, size = 48, formIndex = 0) {
  const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, "aria-hidden": "true" });
  const g = G && G.glyphs[kana];
  if (g) drawGlyph(svg, g.forms[formIndex] || g.forms[0], size * 0.8, size / 2, size / 2);
  return svg;
}

function renderGlyphChart() {
  const host = $("glyph-chart"), detail = $("glyph-detail");
  if (!host || !detail) return;
  host.replaceChildren();
  let selected = null;
  const show = (kana, btn) => {
    if (selected) selected.setAttribute("aria-pressed", "false");
    selected = btn;
    btn.setAttribute("aria-pressed", "true");
    const g = G.glyphs[kana];
    detail.replaceChildren(glyphSvg(kana, 92));
    const info = document.createElement("div");
    const big = document.createElement("div");
    big.className = "big";
    big.lang = "ja";
    const kata = kana === "ー" ? "" : String.fromCharCode(kana.charCodeAt(0) + 0x60);
    big.textContent = kata ? `${kana} ${kata}` : kana;
    const p = document.createElement("div");
    p.className = "info";
    const voiced = Object.entries(G.composed).filter(([, c]) => c.base === kana).map(([k]) => k);
    p.textContent = kana === "ー"
      ? "The long-vowel mark (chōon): it lengthens the vowel before it, so はんたー reads hantā."
      : `Romaji “${g.romaji}”.` + (voiced.length ? ` With a mark it also writes ${voiced.join(" ")}.` : "");
    info.append(big, p);
    if (g.forms.length > 1) {
      const v = document.createElement("div");
      v.className = "variants";
      g.forms.forEach((_, i) => v.append(glyphSvg(kana, 40, i)));
      const note = document.createElement("div");
      note.className = "info";
      note.textContent = g.forms.map((f) => `${f.name} ${Math.round(f.weight * 100)}%`).join(" · ");
      info.append(v, note);
    }
    if (voiced.length) {
      const v = document.createElement("div");
      v.className = "variants";
      for (const k of voiced.slice(0, 2)) {
        const svg = el("svg", { viewBox: "0 0 40 40", "aria-hidden": "true" });
        drawGlyph(svg, g.forms[0], 30, 18, 19, G.composed[k].mark);
        v.append(svg);
      }
      info.append(v);
    }
    detail.append(info);
  };
  let first = null;
  for (const row of G.rows) {
    const cells = [...row.kana, null, null, null, null].slice(0, 5);
    for (const k of cells) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "glyph";
      if (!k || !G.glyphs[k]) {
        b.classList.add("empty");
        b.tabIndex = -1;
        b.setAttribute("aria-hidden", "true");
        host.append(b);
        continue;
      }
      b.setAttribute("role", "listitem");
      b.setAttribute("aria-pressed", "false");
      b.setAttribute("aria-label", `${k} (${G.glyphs[k].romaji})`);
      b.append(glyphSvg(k, 46));
      const kk = document.createElement("span");
      kk.className = "k";
      kk.lang = "ja";
      kk.textContent = k;
      const r = document.createElement("span");
      r.className = "r";
      r.textContent = G.glyphs[k].romaji;
      b.append(kk, r);
      for (const ev of ["mouseenter", "focus", "click"]) b.addEventListener(ev, () => show(k, b));
      host.append(b);
      first ||= [k, b];
    }
  }
  if (first) show(...first);
}

// ---------------------------------------------------------------- encoder

const toHira = (c) => {
  const o = c.charCodeAt(0);
  return o >= 0x30a1 && o <= 0x30f6 ? String.fromCharCode(o - 0x60) : c;
};

function toRomaji(text) {
  const out = [];
  const MAC = { a: "ā", i: "ī", u: "ū", e: "ē", o: "ō" };
  for (const ch of text) {
    if (ch === "ー") {
      const last = out.length ? out[out.length - 1] : "", v = last.slice(-1);
      if (MAC[v]) out[out.length - 1] = last.slice(0, -1) + MAC[v];
      continue;
    }
    const g = G && (G.glyphs[ch] || G.composed[ch]);
    out.push(ch === " " ? " " : g && g.romaji ? g.romaji : ch);
  }
  return out.join("");
}

function renderEncoder() {
  const input = $("enc-input"), out = $("enc-out"), rom = $("enc-romaji"), presets = $("enc-presets");
  if (!input || !out) return;
  for (const p of ["ハンター", "ゴン", "キルア", "クラピカ", "レオリオ", "ハンターしけん", "グリードアイランド"]) {
    const b = document.createElement("button");
    b.type = "button";
    b.lang = "ja";
    b.textContent = p;
    b.addEventListener("click", () => { input.value = p; draw(); });
    presets.append(b);
  }
  const draw = () => {
    const s = [...input.value].map((c) => { const h = toHira(c); return (G.small_to_full && G.small_to_full[h]) || h; }).join("");
    const chars = [...s].slice(0, 40);
    const cell = 54, S = 42, H = 60;
    const W = Math.max(1, chars.length) * cell + 8;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${s} in Hunter script` });
    svg.style.width = `${W}px`;
    let x = 4;
    for (const ch of chars) {
      if (ch === " " || ch === "　") { x += cell * 0.5; continue; }
      const comp = G.composed[ch];
      const g = G.glyphs[comp ? comp.base : ch];
      if (!g) {
        text(svg, x + cell / 2, H * 0.62, ch, "t-muted", "middle");
      } else {
        drawGlyph(el("g", {}, svg), g.forms[0], S, x + cell / 2 - 3, H / 2, comp ? comp.mark : null);
      }
      x += cell;
    }
    out.replaceChildren(svg);
    rom.textContent = s ? `${s} · ${toRomaji(s)}` : "";
  };
  input.addEventListener("input", draw);
  draw();
}

// ---------------------------------------------------------------- CTC animation (data/ctc.json, data/nn.json)

const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const pctOf = (p) => (p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);

/**
 * Greedy CTC, shown as three rows that fill in as a cursor walks along the strip: what each step
 * says, repeats glued together, blanks dropped (then the symbols close up into the reading).
 * Below: a close-up of the steps around the cursor and the top three guesses there. nn (optional,
 * data/nn.json) adds the span of the strip each step pays most attention to.
 */
function renderCtc(d, nn) {
  const host = $("ctc");
  if (!host) return;
  host.classList.add("chart");
  const T = d.timesteps, stride = d.stride || d.width / T;
  const Tc = Math.min(T, Math.ceil((d.contentWidth + 24) / stride)); // the steps over the line; the rest say blank
  const pxW = Tc * stride;
  const spans = nn && nn.spans && nn.spans.length === T ? nn.spans : null;
  // runs of the same class (blank runs included); each symbol gets a colour, the same one every time
  const runs = [];
  d.columns.slice(0, Tc).forEach((c, t) => {
    const last = runs[runs.length - 1];
    if (last && last.id === c.id) last.t1 = t; else runs.push({ id: c.id, ch: c.ch, t0: t, t1: t });
  });
  const hue = new Map();
  runs.forEach((r) => { if (r.id && !hue.has(r.id)) hue.set(r.id, hue.size % 8); });
  const symCls = (id) => (id ? `sym s${hue.get(id)}` : "sym blank");
  const emits = runs.filter((r) => r.id !== 0);
  const reading = emits.map((r) => r.ch).join("");

  host.innerHTML = `<div class="fig-title">Reading one line, step by step</div>
    <div class="fig-sub">The real reader on ${d.label} (${d.romaji}). It answers once for every 4 pixels of the strip: ${T} steps. The ${T - Tc} steps after the end of the line all say "nothing here" and are cut off.</div>`;
  const stage = document.createElement("div");
  stage.className = "ctc-stage";
  host.append(stage);
  const ctl = document.createElement("div");
  ctl.className = "ctc-controls";
  ctl.innerHTML = `<button type="button" class="ctc-restart" aria-label="Back to the start">⏮</button>
    <button type="button" class="ctc-prev" aria-label="One step back">◀</button>
    <button type="button" class="ctc-play" aria-label="Pause">❚❚</button>
    <button type="button" class="ctc-next" aria-label="One step forward">▶</button>
    <input type="range" min="0" max="${Tc - 1}" value="0" aria-label="Step">
    <button type="button" class="ctc-speed" aria-pressed="false">Slow</button>`;
  host.append(ctl);
  const readout = document.createElement("div");
  readout.className = "ctc-readout";
  host.append(readout);
  const [restartBtn, prevBtn, playBtn, nextBtn] = ctl.querySelectorAll("button");
  const speedBtn = ctl.querySelector(".ctc-speed"), slider = ctl.querySelector("input");

  // close-up and top guesses
  const below = document.createElement("div");
  below.className = "ctc-below";
  host.append(below);
  const zoomBox = document.createElement("div");
  zoomBox.className = "ctc-zoom";
  zoomBox.innerHTML = `<span class="label">Close-up: the steps around the cursor</span>`;
  below.append(zoomBox);
  const topBox = document.createElement("div");
  topBox.className = "ctc-top";
  below.append(topBox);
  if (d.columns[0] && d.columns[0].top) {
    topBox.innerHTML = `<span class="label">Top guesses at this step</span>`;
    for (let i = 0; i < 3; i++) {
      const r = document.createElement("div");
      r.className = "ctc-guess";
      r.innerHTML = `<span class="g-k"></span><span class="g-bar"><i></i></span><span class="g-p"></span>`;
      topBox.append(r);
    }
  }
  const guesses = [...topBox.querySelectorAll(".ctc-guess")];

  // the close-up: 45 steps of the strip at 2x, the cursor step and its attention span, and 9 big tiles
  const ZW = 360, ZS = 2, ZN = 45, NT = 9, TW = 36;
  const zsvg = el("svg", { viewBox: `0 0 ${ZW} 166`, role: "img", "aria-label": "Close-up of the steps around the cursor" }, zoomBox);
  const zclip = el("clipPath", { id: "ctc-zclip" }, el("defs", {}, zsvg));
  el("rect", { x: 0, y: 0, width: ZW, height: d.height * ZS }, zclip);
  const zg = el("g", { "clip-path": "url(#ctc-zclip)" }, zsvg);
  const zimg = el("image", { href: "img/ctc_strip.png", x: 0, y: 0, width: d.width * ZS, height: d.height * ZS, preserveAspectRatio: "none", style: "image-rendering: pixelated" }, zg);
  const zspan = el("rect", { y: 1, height: d.height * ZS - 2, rx: 3, class: "ctc-span" }, zg);
  const zcur = el("rect", { y: 0, width: stride * ZS, height: d.height * ZS, class: "ctc-zcur" }, zg);
  el("rect", { x: 0.5, y: 0.5, width: ZW - 1, height: d.height * ZS - 1, fill: "none", class: "ax-strong" }, zsvg);
  const fans = [], tiles = [];
  for (let i = 0; i < NT; i++) {
    fans.push(el("path", { class: "ctc-fan" }, zsvg));
    const g = el("g", {}, zsvg);
    const tx = 2 + i * (TW + 4);
    el("rect", { x: tx, y: 126, width: TW, height: 36, rx: 4 }, g);
    const t = text(g, tx + TW / 2, 149, "", "ctc-tk", "middle");
    tiles.push({ g, t, cx: tx + TW / 2 });
  }

  let W, cw, refs;
  const build = () => {
    stage.replaceChildren();
    W = Math.max(300, stage.clientWidth);
    cw = W / Tc;
    const sh = (W * d.height) / pxW;
    const y1 = sh + 30, h1 = 16, y2 = y1 + h1 + 34, h2 = 20, y3 = y2 + h2 + 46, h3 = 30, yR = y3 + h3 + 22, H = yR + 8;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `CTC decoding of ${d.label}` });
    const clip = el("clipPath", { id: "ctc-clip" }, el("defs", {}, svg));
    el("rect", { x: 0, y: 0, width: W, height: sh }, clip);
    const sg = el("g", { "clip-path": "url(#ctc-clip)" }, svg);
    el("image", { href: "img/ctc_strip.png", x: 0, y: 0, width: (W * d.width) / pxW, height: sh, preserveAspectRatio: "none", style: "image-rendering: pixelated" }, sg);
    const span = el("rect", { y: 1, height: sh - 2, rx: 2, class: "ctc-span" }, sg);
    el("rect", { x: 0.5, y: 0.5, width: W - 1, height: sh - 1, fill: "none", class: "ax-strong" }, svg);
    // row 1: what each step says
    text(svg, 0, y1 - 9, "1 · What each step says", "t-group");
    const cells = d.columns.slice(0, Tc).map((c, t) => el("rect", {
      x: t * cw + 0.3, y: y1, width: Math.max(0.6, cw - 0.6), height: h1, rx: Math.min(2, cw / 3), class: symCls(c.id),
    }, svg));
    // row 2: repeats glued together (blank runs too)
    text(svg, 0, y2 - 9, "2 · Glue repeats together", "t-group");
    const r2 = runs.map((r) => {
      const g = el("g", { class: `ctc-run ${symCls(r.id)}` }, svg);
      const rect = el("rect", { x: r.t0 * cw + 0.5, y: y2, width: Math.max(1, (r.t1 - r.t0 + 1) * cw - 1), height: h2, rx: 3 }, g);
      if (r.id) text(g, ((r.t0 + r.t1 + 1) / 2) * cw, y2 + h2 + 14, r.ch, "ctc-k", "middle");
      return { g, rect };
    });
    // row 3: blanks dropped; the symbols fall into place, then close up into the reading
    text(svg, 0, y3 - 14, "3 · Drop the blanks", "t-group");
    const ow = Math.min(30, (W - 8) / Math.max(1, emits.length)) - 3, ox0 = (W - (ow + 3) * emits.length) / 2;
    const r3 = emits.map((r, i) => {
      const at = ((r.t0 + r.t1 + 1) / 2) * cw, packed = ox0 + (ow + 3) * i + ow / 2 + 1.5;
      const g = el("g", { class: `ctc-out3 ${symCls(r.id)}` }, svg);
      el("rect", { x: -ow / 2, y: y3, width: ow, height: h3, rx: 4 }, g);
      text(g, 0, y3 + h3 / 2 + 5.5, r.ch, "ctc-o", "middle");
      g.style.transformOrigin = `0px ${(y3 + h3 / 2).toFixed(1)}px`;
      return { g, at, packed };
    });
    // before they close up, the tiles sit over their runs: shrink them if two runs are close
    let gap = Infinity;
    r3.forEach((o, i) => { if (i) gap = Math.min(gap, o.at - r3[i - 1].at); });
    const fit = Math.max(0.45, Math.min(1, (gap - 3) / ow));
    const rom = text(svg, W / 2, yR, d.romaji, "ctc-rom", "middle");
    // the cursor
    const head = el("rect", { x: 0, y: -3, width: Math.max(2, cw), height: y1 + h1 + 6, rx: 1, class: "ctc-head" }, svg);
    stage.append(svg);
    refs = { cells, r2, r3, rom, head, span, fit };
    paint(col);
  };

  let col = reduced() ? Tc - 1 : 0;
  const paint = (t) => {
    col = t;
    slider.value = String(t);
    if (!refs) return;
    const done = t >= Tc - 1;
    refs.head.setAttribute("x", (t * cw).toFixed(1));
    refs.cells.forEach((r, i) => r.classList.toggle("seen", i <= t));
    runs.forEach((r, i) => {
      const { g, rect } = refs.r2[i];
      g.classList.toggle("seen", r.t0 <= t);
      rect.setAttribute("width", Math.max(1, (Math.min(r.t1, t) - r.t0 + 1) * cw - 1).toFixed(1));
    });
    let read = "";
    emits.forEach((r, i) => {
      const o = refs.r3[i], on = r.t0 <= t;
      if (on) read += r.ch;
      o.g.classList.toggle("seen", on);
      o.g.style.transform = done ? `translateX(${o.packed.toFixed(1)}px)` : `translateX(${o.at.toFixed(1)}px) scale(${refs.fit.toFixed(3)})`;
    });
    refs.rom.classList.toggle("seen", done);
    // what this step looks at
    const sp = spans ? spans[t] : null;
    refs.span.setAttribute("x", sp ? ((sp[0] / pxW) * W).toFixed(1) : 0);
    refs.span.setAttribute("width", sp ? ((Math.max(4, sp[1] - sp[0]) / pxW) * W).toFixed(1) : 0);
    // close-up around the step
    const c0 = Math.max(0, Math.min(T - ZN, t - (ZN >> 1)));
    zimg.setAttribute("x", (-c0 * stride * ZS).toFixed(1));
    zcur.setAttribute("x", ((t - c0) * stride * ZS).toFixed(1));
    zspan.setAttribute("x", sp ? ((sp[0] - c0 * stride) * ZS).toFixed(1) : 0);
    zspan.setAttribute("width", sp ? (Math.max(4, sp[1] - sp[0]) * ZS).toFixed(1) : 0);
    tiles.forEach((tl, i) => {
      const s = t - (NT >> 1) + i, ok = s >= 0 && s < T;
      const c = ok ? d.columns[s] : null;
      tl.g.setAttribute("class", `ctc-tile ${ok ? symCls(c.id) : "off"}${i === NT >> 1 ? " cur" : ""}`);
      tl.t.textContent = ok ? (c.id ? c.ch : "–") : "";
      const sx = ((s - c0) * stride + stride / 2) * ZS, y0 = d.height * ZS;
      fans[i].setAttribute("d", ok && sx >= 0 && sx <= ZW ? `M${sx.toFixed(1)},${y0} C${sx.toFixed(1)},${y0 + 16} ${tl.cx},110 ${tl.cx},126` : "M0,0");
      fans[i].classList.toggle("cur", i === NT >> 1);
      fans[i].classList.toggle("off", !(ok && sx >= 0 && sx <= ZW));
    });
    const c = d.columns[t];
    readout.textContent = `step ${t + 1} / ${T} · ${c.id ? c.ch : "nothing here"} ${pctOf(c.p)} · read so far: ${read || "…"}${done ? ` (${d.romaji})` : ""}`;
    if (c.top) c.top.forEach(([ch, p], i) => {
      const g = guesses[i];
      if (!g) return;
      g.children[0].textContent = ch === "" ? "nothing" : ch === " " ? "space" : ch;
      g.children[0].classList.toggle("blank", ch === "" || ch === " ");
      g.children[1].firstChild.style.width = `${Math.max(0.5, p * 100)}%`;
      g.children[2].textContent = pctOf(p);
      g.classList.toggle("first", i === 0);
    });
  };

  // play: one step per 70 ms (280 ms when slow), a pause at the end, only while visible
  let playing = !reduced(), slow = false, visible = false, last = 0, hold = 0;
  const tick = (now) => {
    if (playing && visible) {
      if (!last) last = now;
      if (now - last > (slow ? 280 : 70)) {
        last = now;
        if (col >= Tc - 1) {
          if (++hold > (slow ? 10 : 36)) { hold = 0; paint(0); }
        } else paint(col + 1);
      }
    }
    requestAnimationFrame(tick);
  };
  const setPlaying = (p) => {
    playing = p;
    playBtn.textContent = p ? "❚❚" : "▶";
    playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
  };
  const step = (k) => { setPlaying(false); paint(Math.max(0, Math.min(Tc - 1, col + k))); };
  playBtn.addEventListener("click", () => { if (!playing && col >= Tc - 1) paint(0); setPlaying(!playing); });
  prevBtn.addEventListener("click", () => step(-1));
  nextBtn.addEventListener("click", () => step(1));
  restartBtn.addEventListener("click", () => { hold = 0; paint(0); });
  speedBtn.addEventListener("click", () => { slow = !slow; speedBtn.setAttribute("aria-pressed", String(slow)); });
  slider.addEventListener("input", () => { setPlaying(false); paint(Number(slider.value)); });
  new IntersectionObserver((es) => { visible = es.some((e) => e.isIntersecting); }).observe(host);
  setPlaying(playing);
  build();
  new ResizeObserver(() => { if (Math.abs(Math.max(300, stage.clientWidth) - W) > 1) build(); }).observe(stage);
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- pipeline flow (img/pipeline/*.json)

function renderFlow(pipe, crops) {
  const host = $("flow");
  if (!host) return;
  const t = pipe && pipe.timings_ms ? pipe.timings_ms : (pipe && pipe.timings) || {};
  const ms = (v) => (v == null ? "" : `${Number(v).toFixed(v < 10 ? 1 : 0)} ms`);
  const featured = (crops || []).filter((c) => c.featured).slice(0, 4);
  const shown = pipe && pipe.counts && pipe.counts.shown;
  const steps = [
    { title: "Page", tag: `${pipe ? `${pipe.page?.width ?? pipe.width ?? 901} × ${pipe.page?.height ?? pipe.height ?? 554} px` : "any size"} · gray`, text: "Any photo, screenshot or scan, turned gray in the browser.", img: "img/pipeline/01_page.webp" },
    { title: "Letterbox", tag: `640 × 640 · ${ms(t.letterbox ?? t.letterbox_ms)}`, text: "Shrunk so the long side is 640 px, placed top-left, padded white. Rust, area averaging.", img: "img/pipeline/02_letterbox.webp" },
    { title: "Detector", tag: `int8 · 1.57 G MACs · ${ms(t.detector ?? t.detect ?? t.run)}`, text: "Two maps at 320 × 320: where each line's shrunken core is, and whether it runs across or down.", imgs: ["img/pipeline/03_kernel.webp", "img/pipeline/04_orient.webp"] },
    { title: "Lines", tag: `Rust · ${ms(t.post ?? t.postprocess ?? t.post_processing)}`, text: "Connected cores → minimum-area rectangles → grown back (unclip 1.5) → reading order, columns right to left.", img: "img/pipeline/05_boxes.webp" },
    { title: "Crops", tag: "48 × 768 strips", text: "Each line cut out straight with margins at the page's own resolution; columns turned a quarter turn.", crops: featured },
    { title: "Readers", tag: `int8 · 154 M MACs per line${t.read_per_line_mean ? ` · ${ms(t.read_per_line_mean)} each` : ""}`, text: "A horizontal reader and a vertical reader turn each strip into kana; unsure lines are set aside.", out: featured },
  ];
  host.replaceChildren();
  const ol = document.createElement("ol");
  ol.className = "flow-list";
  steps.forEach((s, i) => {
    const li = document.createElement("li");
    li.className = "flow-step";
    li.style.setProperty("--i", i);
    li.innerHTML = `<span class="flow-node" aria-hidden="true">${i + 1}</span>
      <div class="flow-body"><div class="flow-head"><span class="flow-title"></span><span class="label flow-tag"></span></div>
      <p class="flow-text"></p><div class="flow-media"></div></div>`;
    li.querySelector(".flow-title").textContent = s.title;
    li.querySelector(".flow-tag").textContent = s.tag;
    li.querySelector(".flow-text").textContent = s.text;
    const media = li.querySelector(".flow-media");
    for (const src of s.imgs || (s.img ? [s.img] : [])) {
      const im = document.createElement("img");
      im.src = src;
      im.alt = "";
      im.loading = "lazy";
      media.append(im);
    }
    if (s.crops) {
      media.classList.add("strips");
      for (const c of s.crops) {
        const im = document.createElement("img");
        im.src = `img/pipeline/${c.file}`;
        im.alt = `Strip: ${c.text}`;
        im.loading = "lazy";
        media.append(im);
      }
    }
    if (s.out) {
      media.classList.add("reads");
      for (const c of s.out) {
        const r = document.createElement("div");
        r.className = "flow-read";
        r.innerHTML = `<span class="k" lang="ja"></span><span class="r"></span><span class="label"></span>`;
        r.children[0].textContent = c.text;
        r.children[1].textContent = c.romaji;
        r.children[2].textContent = `${c.vertical ? "vertical" : "horizontal"} · ${Math.round((c.confidence ?? 1) * 100)}%`;
        media.append(r);
      }
    }
    if (!media.childElementCount) media.remove();
    ol.append(li);
  });
  host.append(ol);
  if (shown) {
    const foot = document.createElement("p");
    foot.className = "flow-foot label";
    foot.textContent = `${shown} lines shown on this page · about ${Math.round(pipe.timings_ms?.page_total ?? 180)} ms end to end in node`;
    host.append(foot);
  }
  // a pulse walks down the steps while the figure is on screen
  if (reduced()) return;
  let visible = false, i = 0;
  new IntersectionObserver((es) => { visible = es.some((e) => e.isIntersecting); }).observe(host);
  const items = [...ol.children];
  setInterval(() => {
    if (!visible) return;
    items.forEach((li, k) => li.classList.toggle("active", k === i));
    ol.style.setProperty("--progress", String(i / (items.length - 1)));
    i = (i + 1) % items.length;
  }, 1400);
}

// ---------------------------------------------------------------- figure numbers

/** Number the visible figure captions in document order ("Fig. # —" in the HTML). */
function numberFigures() {
  // captions start "<label> # —"; the label is the page's own word for a figure (<html data-fig>)
  const label = document.documentElement.dataset.fig || "Fig.";
  let n = 0;
  for (const fc of document.querySelectorAll("article figure figcaption")) {
    if (fc.closest("figure").hidden) continue;
    fc.dataset.raw ??= fc.textContent;
    if (fc.dataset.raw.startsWith(`${label} # —`)) fc.textContent = fc.dataset.raw.replace("#", String(++n));
  }
}

// ---------------------------------------------------------------- translations of text the scripts draw

/**
 * Translated pages load i18n-<lang>.js first, which sets window.HXH_I18N = { strings: {english:
 * translation}, patterns: [[regex, replacement], ...] }. Text the scripts put into the article
 * (chart titles, labels, animation readouts, SVG text, aria-labels) is translated as it appears.
 */
function startTranslator() {
  const T = window.HXH_I18N;
  const art = document.querySelector("article");
  if (!T || !art) return;
  const strings = T.strings || {};
  const pats = (T.patterns || []).map(([re, out]) => [new RegExp(re), out]);
  const made = new Set();
  const tr = (s) => {
    const t = s.trim();
    if (!t || made.has(t)) return null;
    let out = strings[t];
    if (out == null) for (const [re, rep] of pats) if (re.test(t)) { out = t.replace(re, rep); break; }
    if (out == null || out === t) return null;
    made.add(out);
    return s.replace(t, out);
  };
  const ATTRS = ["aria-label", "alt", "title", "placeholder"];
  const walk = (node) => {
    if (node.nodeType === 3) {
      const v = tr(node.data);
      if (v != null) node.data = v;
      return;
    }
    if (node.nodeType !== 1 || node.closest?.("[data-no-i18n]")) return;
    for (const a of ATTRS) {
      const v = node.getAttribute?.(a);
      if (v) { const o = tr(v); if (o != null) node.setAttribute(a, o); }
    }
    for (const c of node.childNodes) walk(c);
  };
  walk(art);
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === "characterData") walk(m.target);
      else if (m.type === "attributes") walk(m.target);
      else m.addedNodes.forEach(walk);
    }
  }).observe(art, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
}

// ---------------------------------------------------------------- go

const getJson = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${url}: HTTP ${r.status}`))));
const hide = (id) => { const f = $(id) && $(id).closest("figure"); if (f) f.hidden = true; };

/** Language links whose page isn't published yet stay visible but inert ("soon"). */

startTranslator();
buildToc();
renderCharts();
numberFigures();
getJson("data/glyphs.json").then((d) => {
  G = d;
  renderGlyphChart();
  renderEncoder();
}).catch((err) => {
  console.warn("glyph data not loaded:", err);
  hide("glyph-chart");
  hide("enc-out");
  numberFigures();
});
Promise.all([getJson("data/ctc.json"), getJson("data/nn.json").catch(() => null)]).then(([d, nn]) => renderCtc(d, nn)).catch((err) => { console.warn(err); hide("ctc"); numberFigures(); });
Promise.all([getJson("img/pipeline/pipeline.json").catch(() => null), getJson("img/pipeline/crops.json").catch(() => null)])
  .then(([pipe, crops]) => renderFlow(pipe, crops && crops.crops))
  .catch((err) => { console.warn(err); hide("flow"); numberFigures(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => renderCharts());

// building blocks for page-specific scripts (explain.js, deep.js); they load after this module
window.HXH = { el, text, css, barChart, lineChart, drawGlyph, glyphSvg, getJson, numberFigures, toRomaji,
  glyphs: () => G };
document.dispatchEvent(new Event("hxh:ready"));
