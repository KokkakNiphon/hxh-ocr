// HxH OCR deep dive (deep-dive.html): the figures that blog.js does not draw.
//
// Data, never typed in where a file has it:
//   data/charts.json, data/glyphs.json     scripts/blog/make_assets.py (journal tables, run logs, glyph strokes)
//   img/pipeline/pipeline.json             scripts/blog/pipeline_assets.mjs (one page through the browser code)
//   img/deep/data.json                     scripts/blog/deep_data.py (QAT logs, det_v1 log, reader exponents)
// Numbers that are typed in below are copied from docs/research/JOURNAL.md, the run configs or
// rust/hxh-synth/FOREIGN.md, and each block names its source. Schematics say so in their captions.

const HX = window.HXH || null;
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}
function tx(parent, x, y, s, cls = "tm", anchor = "start") {
  const t = el("text", { x, y, class: cls, "text-anchor": anchor }, parent);
  t.textContent = s;
  return t;
}
function html(tag, cls, text, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}
function svgRoot(w, h, label) {
  return el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img", "aria-label": label });
}
function arrow(svg, x1, y1, x2, y2, cls = "d-arrow", head = "d-head", s = 6) {
  el("line", { x1, y1, x2, y2, class: cls }, svg);
  const a = Math.atan2(y2 - y1, x2 - x1);
  const p = (da) => `${(x2 - s * Math.cos(a + da)).toFixed(1)},${(y2 - s * Math.sin(a + da)).toFixed(1)}`;
  el("path", { d: `M${x2},${y2}L${p(-0.45)}L${p(0.45)}Z`, class: head }, svg);
}
/** Redraw on width changes only (a redraw that changes the height does not loop). */
function responsive(host, draw) {
  let last = 0;
  const run = () => {
    const w = Math.round(host.clientWidth);
    if (!w || w === last) return;
    last = w;
    draw(w);
  };
  new ResizeObserver(run).observe(host);
  run();
}
const getJson = (u) => fetch(u).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${u}: HTTP ${r.status}`))));
const hideFig = (id) => { const f = $(id) && $(id).closest("figure"); if (f) f.hidden = true; };
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (v, d = 2) => `${v.toFixed(d)}%`;
const fmtStep = (v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(Math.round(v)));

/** Segmented buttons; calls onPick(index) and returns a setter. */
function segmented(host, labels, onPick, start = 0) {
  const seg = html("div", "seg", null, host);
  seg.setAttribute("role", "group");
  const btns = labels.map((l, i) => {
    const b = html("button", null, l, seg);
    b.type = "button";
    b.addEventListener("click", () => pick(i));
    return b;
  });
  const pick = (i) => {
    btns.forEach((b, k) => b.setAttribute("aria-pressed", String(k === i)));
    onPick(i);
  };
  pick(start);
  return pick;
}

// ---------------------------------------------------------------- line chart (y range, log axis, hover)

function lineChart(host, o) {
  if (!host) return;
  host.classList.add("chart");
  host.replaceChildren();
  if (o.title) {
    html("div", "fig-title", o.title, host);
    html("div", "fig-sub", o.sub || "", host);
  }
  const lg = html("div", "legend", null, host);
  for (const s of o.series) {
    const i = html("span", null, null, lg);
    const sw = html("i", null, null, i);
    sw.style.background = s.dash ? "transparent" : s.color;
    if (s.dash) { sw.style.border = `1.5px dashed ${s.color}`; sw.style.height = "0"; sw.style.width = "14px"; sw.style.verticalAlign = "3px"; }
    i.append(s.name);
  }
  const box = html("div", null, null, host);
  const out = html("div", "readout", null, host);
  out.setAttribute("aria-live", "off");
  const xs = [...new Set(o.series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const xMin = o.xMin ?? xs[0], xMax = o.xMax ?? xs[xs.length - 1];
  const ys = o.series.flatMap((s) => s.points.map((p) => p[1]));
  const yMin = o.yMin ?? 0;
  let yMax = o.yMax ?? Math.max(...ys) * 1.04, autoTicks = null;
  if (!o.yTicks && !o.log) {
    const step = [0.1, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50].find((st) => (yMax - yMin) / st <= 6) || 100;
    yMax = yMin + Math.ceil((yMax - yMin) / step - 1e-9) * step;
    autoTicks = [];
    for (let v = yMin; v <= yMax + 1e-9; v += step) autoTicks.push(+v.toFixed(4));
  }
  const f = o.log ? Math.log10 : (v) => v;
  const xFmt = o.xFmt || fmtStep, yFmt = o.yFmt || ((v) => `${+v.toFixed(2)}${o.yUnit || ""}`);
  const say = (x) => {
    const parts = o.series.map((s) => {
      const p = s.points.find((q) => q[0] === x);
      return p ? `${s.short || s.name} ${yFmt(p[1])}` : null;
    }).filter(Boolean);
    out.textContent = `${o.xName || "step"} ${xFmt(x)} · ${parts.join(" · ")}`;
  };
  responsive(box, (W) => {
    box.replaceChildren();
    const H = o.height || Math.round(Math.min(300, Math.max(200, W * 0.46)));
    const L = 46, R = 14, T = 10, B = 30;
    const sx = (v) => L + ((v - xMin) / (xMax - xMin || 1)) * (W - L - R);
    const sy = (v) => T + (1 - (f(Math.min(Math.max(v, yMin), yMax)) - f(yMin)) / (f(yMax) - f(yMin))) * (H - T - B);
    const svg = svgRoot(W, H, o.title || o.label || "chart");
    const yt = o.yTicks || autoTicks;
    yt.forEach((v, i) => {
      el("line", { x1: L, x2: W - R, y1: sy(v), y2: sy(v), class: i ? "ax" : "ax-strong" }, svg);
      tx(svg, L - 6, sy(v) + 3.5, `${+v.toFixed(o.yDigits ?? 2)}${o.yUnit || ""}`, "t-muted", "end");
    });
    const xt = o.xTicks || Array.from({ length: 5 }, (_, i) => xMin + ((xMax - xMin) * i) / 4);
    xt.forEach((v, i) => tx(svg, sx(v), H - B + 16, xFmt(v), "t-muted", o.xTicks ? "middle" : i === 0 ? "start" : i === xt.length - 1 ? "end" : "middle"));
    if (o.xLabel) tx(svg, W - R, H - 2, o.xLabel, "t-muted", "end");
    for (const s of o.series) {
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
      const path = el("path", { d, class: "line" }, svg);
      path.style.stroke = s.color;
      if (s.dash) path.style.strokeDasharray = "5 4";
      if (s.points.length <= 16) {
        for (const p of s.points) {
          const c = el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 2.6 }, svg);
          c.style.fill = s.dash ? "var(--bg)" : s.color;
          c.style.stroke = s.color;
        }
      }
    }
    const guide = el("line", { x1: 0, x2: 0, y1: T, y2: H - B, class: "ax-strong", visibility: "hidden" }, svg);
    const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
    const near = (evt) => {
      const r = svg.getBoundingClientRect();
      const px = ((evt.clientX - r.left) / r.width) * W;
      let best = xs[0];
      for (const x of xs) if (Math.abs(sx(x) - px) < Math.abs(sx(best) - px)) best = x;
      return best;
    };
    hit.addEventListener("pointermove", (e) => {
      const x = near(e);
      guide.setAttribute("x1", sx(x)); guide.setAttribute("x2", sx(x)); guide.setAttribute("visibility", "visible");
      say(x);
    });
    hit.addEventListener("pointerleave", () => { guide.setAttribute("visibility", "hidden"); say(xs[xs.length - 1]); });
    box.append(svg);
  });
  say(xs[xs.length - 1]);
}

// ---------------------------------------------------------------- glyph drawing (data/glyphs.json)

function drawForm(parent, G, form, S, cx, cy, mark = null) {
  const sw = G.stroke_width_em;
  for (const st of form.strokes) {
    if (st.type === "dot") {
      el("circle", { cx: cx + st.c[0] * S, cy: cy + st.c[1] * S, r: Math.max(st.r, sw / 2) * S, class: "glyph-fill" }, parent);
    } else {
      const d = st.pts.map((p, i) => `${i ? "L" : "M"}${(cx + p[0] * S).toFixed(2)},${(cy + p[1] * S).toFixed(2)}`).join("") + (st.closed ? "Z" : "");
      el("path", { d, class: "glyph-stroke", "stroke-width": (sw * S).toFixed(2) }, parent);
    }
  }
  if (mark) {
    const m = G.marks[mark], [, , x1, y1] = form.bbox;
    const mx = cx + (x1 + m.dx) * S, my = cy + (y1 + m.dy) * S;
    if (m.hollow) el("circle", { cx: mx, cy: my, r: Math.max(m.r, sw * m.width_scale) * S, class: "glyph-stroke", "stroke-width": (sw * m.width_scale * S).toFixed(2) }, parent);
    else el("circle", { cx: mx, cy: my, r: Math.max(m.r, sw / 2) * S, class: "glyph-fill" }, parent);
  }
}
function glyphOf(G, k) {
  const comp = G.composed[k];
  const g = G.glyphs[comp ? comp.base : k];
  return g ? { form: g.forms[0], mark: comp ? comp.mark : null, romaji: comp ? comp.romaji : g.romaji } : null;
}
function drawKana(parent, G, k, S, cx, cy) {
  const g = glyphOf(G, k);
  if (g) drawForm(parent, G, g.form, S, cx, cy, g.mark);
}

// ---------------------------------------------------------------- 1 system overview (HTML)

function overview() {
  const host = $("d-overview");
  if (!host) return;
  const lanes = [
    { title: "Offline · Python, PyTorch and Rust on one PC", nodes: [
      ["Glyph strokes", "47 traced glyphs, 3 with two forms"],
      ["Generators", "lines (Python), pages (Rust)"],
      ["Train fp32", "PyTorch; EMA + BN re-estimation"],
      ["QAT", "ESP-DL-exact fake quantisation"],
      ["ESP-PPQ", "int8 .espdl, power-of-two scales", true],
      [".hxhm", "converter, checked per filter", true],
    ] },
    { title: "Online · the browser, Rust compiled to WebAssembly", nodes: [
      ["Image", "photo, screenshot or camera"],
      ["Letterbox", "gray, 640 × 640"],
      ["Detector", "int8, 1.57 G MACs", true],
      ["Post-processing", "lines, order, crops"],
      ["Readers h / v", "int8, 154 M MACs per line", true],
      ["Greedy CTC", "kana, romaji, confidence"],
    ] },
  ];
  host.replaceChildren();
  for (const lane of lanes) {
    const wrap = html("div", null, null, host);
    html("p", "label lane-title", lane.title, wrap);
    const row = html("div", "lane", null, wrap);
    lane.nodes.forEach(([t, s, hi], i) => {
      if (i) html("span", "arr", "→", row).setAttribute("aria-hidden", "true");
      const n = html("div", hi ? "node hi" : "node", null, row);
      html("span", "n-t", t, n);
      html("span", "n-s", s, n);
    });
  }
}

// ---------------------------------------------------------------- 2 glyph variants and marks (glyphs.json)

function variants(G) {
  const host = $("d-variants");
  if (!host) return;
  host.replaceChildren();
  for (const k of ["ま", "せ", "も"]) {
    const g = G.glyphs[k];
    const card = html("div", "gcard", null, host);
    const svg = svgRoot(150, 86, `${k}: ${g.forms.map((f) => f.name).join(", ")}`);
    g.forms.forEach((f, i) => {
      const cx = 40 + i * 70;
      drawForm(el("g", i === 0 ? { class: "g-acc" } : {}, svg), G, f, 44, cx, 34);
      tx(svg, cx, 80, `${Math.round(f.weight * 100)}%`, i === 0 ? "ta" : "tm", "middle");
    });
    card.append(svg);
    html("div", "c-t", `${k} ${g.romaji}`, card);
    html("div", "c-s", g.forms.map((f) => f.name).join(" · "), card);
  }
}

function marks(G) {
  const host = $("d-marks");
  if (!host) return;
  const sw = G.stroke_width_em, dk = G.marks.dakuten, hk = G.marks.handakuten;
  const W = 640, H = 176;
  const svg = svgRoot(W, H, "Where the generator places voicing marks");
  svg.style.maxWidth = "700px";
  // か with the v4 range of dot centres: x1 + (0.55..1.35) dx, y1 + dy + U(-0.06, 0.03)
  const S = 150, ka = G.glyphs["か"].forms[0], [x0, y0, x1, y1] = ka.bbox, cx = 120, cy = 86;
  el("rect", { x: cx + x0 * S, y: cy + y0 * S, width: (x1 - x0) * S, height: (y1 - y0) * S, class: "d-dash" }, svg);
  const bx0 = cx + (x1 + 0.55 * dk.dx) * S, bx1 = cx + (x1 + 1.35 * dk.dx) * S;
  const by0 = cy + (y1 + dk.dy - 0.06) * S, by1 = cy + (y1 + dk.dy + 0.03) * S;
  el("rect", { x: bx0 - 2, y: by0 - 2, width: bx1 - bx0 + 4, height: by1 - by0 + 4, rx: 3, class: "d-soft", style: "stroke: var(--accent); stroke-width: 1; stroke-dasharray: 3 2" }, svg);
  drawForm(svg, G, ka, S, cx, cy);
  const g = el("g", { class: "g-acc" }, svg);
  el("circle", { cx: cx + (x1 + dk.dx) * S, cy: cy + (y1 + dk.dy) * S, r: Math.max(dk.r, sw / 2) * S, class: "glyph-fill" }, g);
  tx(svg, cx + x0 * S, cy + y0 * S - 8, "bbox of the drawn form", "tm");
  tx(svg, bx1 + 8, by0 + 4, "dot centres, v4:", "tm2");
  tx(svg, bx1 + 8, by0 + 18, `x: (0.55–1.35) × dx`, "tm");
  tx(svg, bx1 + 8, by0 + 32, `y: dy + U(−0.06, 0.03) em`, "tm");
  tx(svg, bx1 + 8, by0 + 46, `chart: dx ${dk.dx}, dy ${dk.dy}, r ${dk.r} em`, "ta");
  // は + handakuten at the chart position
  const ha = G.glyphs["は"].forms[0], S2 = 92, cx2 = 572, cy2 = 74;
  drawForm(svg, G, ha, S2, cx2, cy2);
  const g2 = el("g", { class: "g-acc" }, svg);
  const [, , hx1, hy1] = ha.bbox;
  el("circle", { cx: cx2 + (hx1 + hk.dx) * S2, cy: cy2 + (hy1 + hk.dy) * S2, r: Math.max(hk.r, sw * hk.width_scale) * S2, class: "glyph-stroke", "stroke-width": (sw * hk.width_scale * S2).toFixed(2) }, g2);
  tx(svg, cx2, 150, `ぱ: ring r ${hk.r} em`, "tm", "middle");
  tx(svg, cx2, 164, `line ${hk.width_scale} × stroke`, "tm", "middle");
  host.replaceChildren(svg);
}

// ---------------------------------------------------------------- 3 turned glyphs (vertical reader)

function rotateFig(G) {
  const host = $("d-rotate");
  if (!host) return;
  host.replaceChildren();
  host.style.gridTemplateColumns = "minmax(0, 1fr)";
  const card = html("div", "gcard", null, host);
  const W = 420, H = 112;
  const svg = svgRoot(W, H, "ふ turned a quarter turn counter-clockwise looks like う");
  svg.style.maxWidth = "460px";
  const S = 64, y = 50;
  drawKana(svg, G, "ふ", S, 50, y);
  tx(svg, 50, 104, "ふ upright", "tm", "middle");
  arrow(svg, 92, y, 140, y);
  tx(svg, 116, y - 8, "90° ccw", "tm", "middle");
  const g = el("g", { transform: `rotate(-90 190 ${y})`, class: "g-acc" }, svg);
  drawKana(g, G, "ふ", S, 190, y);
  tx(svg, 190, 104, "ふ turned", "ta", "middle");
  tx(svg, 262, y + 6, "≈", "tb", "middle");
  drawKana(svg, G, "う", S, 330, y);
  tx(svg, 330, 104, "う upright", "tm", "middle");
  card.append(svg);
  html("div", "c-s", "A shared reader would have to be told the orientation; a separate vertical reader avoids the collision.", card);
}

// ---------------------------------------------------------------- 4 keep-clear rule (schematic; FOREIGN.md)

function keepClear(G) {
  const host = $("d-keep");
  if (!host) return;
  const W = 600, H = 214;
  const svg = svgRoot(W, H, "Keep-clear rule between labelled Hunter lines and foreign text");
  svg.style.maxWidth = "640px";
  const qx = 60, qy = 84, qw = 200, qh = 44, m = 0.3 * qh + 8;
  el("rect", { x: qx - m, y: qy - m, width: qw + 2 * m, height: qh + 2 * m, rx: m, class: "d-warm" }, svg);
  el("rect", { x: qx - m, y: qy - m, width: qw + 2 * m, height: qh + 2 * m, rx: m, class: "d-dash" }, svg);
  el("rect", { x: qx, y: qy, width: qw, height: qh, class: "d-hi" }, svg);
  if (G) [..."はんたー"].forEach((k, i) => drawKana(svg, G, k, 36, qx + 30 + i * 46, qy + qh / 2));
  tx(svg, qx - m, qy - m - 10, "labelled Hunter quad, height h₁", "ta");
  // a foreign run inside the margin (rejected) and one outside it (placed)
  const rx = 262, ry = 42, rw = 80, rh = 22;
  el("rect", { x: rx, y: ry, width: rw, height: rh, class: "d-foreign-s", "stroke-dasharray": "3 3" }, svg);
  tx(svg, rx + 8, ry + 16, "© 2011", "tf");
  tx(svg, rx + rw + 8, ry + 16, "overlaps the margin: rejected ✗", "tm2");
  const fx = 380, fy = 128, fw = 120, fh = 26;
  el("rect", { x: fx, y: fy, width: fw, height: fh, class: "d-foreign-s" }, svg);
  tx(svg, fx + 8, fy + 18, "Read more »", "tf");
  tx(svg, fx, fy + fh + 16, "foreign run, height h₂: placed ✓", "tm2");
  arrow(svg, qx + qw + 2, qy + qh / 2, qx + qw + m - 1, qy + qh / 2, "d-arrow", "d-head", 5);
  tx(svg, qx + qw + m + 6, qy + qh / 2 + 4, "margin", "tm");
  tx(svg, 14, 204, "margin = 0.3 × max(h₁, h₂) + the foreign run's outline or shadow halo", "tm2");
  host.replaceChildren(svg);
}

// ---------------------------------------------------------------- 5 reader architecture (shapes from hxh_ocr/model.py)

const READER = [ // channels, height, width for a 48 x 768 input (MNV4S, keep_stages = 4)
  { name: "input", op: "gray", st: "", c: 1, h: 48, w: 768 },
  { name: "stem", op: "3×3 conv", st: "s2,2", c: 32, h: 24, w: 384 },
  { name: "stage 1", op: "2 convs", st: "s2,2", c: 16, h: 12, w: 192 },
  { name: "stage 2", op: "2 convs", st: "s2,1", c: 32, h: 6, w: 192 },
  { name: "stage 3", op: "6 UIB", st: "s2,1", c: 48, h: 3, w: 192 },
  { name: "stage 4", op: "6 UIB", st: "s2,1", c: 64, h: 2, w: 192 },
  { name: "mean H", op: "avg pool", st: "", c: 64, h: 1, w: 192 },
  { name: "head", op: "2 × res.", st: "dw + 1×1", c: 64, h: 1, w: 192 },
  { name: "classes", op: "1×1 conv", st: "", c: 75, h: 1, w: 192, hi: true },
];

function slab(svg, cx, cy, n, k = 1) {
  const fh = Math.max(4, 58 * (n.h / 48)) * k, fw = (4 + 46 * (n.w / 768)) * k, d = (3 + 22 * (n.c / 75)) * k;
  const ddx = d * 0.8, ddy = d * 0.55;
  const x0 = cx - (fw + ddx) / 2, y0 = cy - fh / 2 + ddy / 2;
  const g = n.hi ? "" : n.name === "input" ? " g" : "";
  el("path", { d: `M${x0},${y0}L${x0 + ddx},${y0 - ddy}L${x0 + fw + ddx},${y0 - ddy}L${x0 + fw},${y0}Z`, class: `d-slab-t${g}` }, svg);
  el("path", { d: `M${x0 + fw},${y0}L${x0 + fw + ddx},${y0 - ddy}L${x0 + fw + ddx},${y0 - ddy + fh}L${x0 + fw},${y0 + fh}Z`, class: `d-slab-s${g}` }, svg);
  el("rect", { x: x0, y: y0, width: fw, height: fh, class: `d-slab-f${g}` }, svg);
  return { left: x0, right: x0 + fw + ddx, top: y0 - ddy, bottom: y0 + fh };
}

function readerArch() {
  const host = $("d-reader-arch");
  if (!host) return;
  responsive(host, (W) => {
    const n = READER.length;
    let svg;
    if (W >= 600) {
      const H = 178, cw = W / n, cy = 52;
      svg = svgRoot(W, H, "Reader architecture with tensor shapes");
      const boxes = READER.map((nd, i) => slab(svg, cw * (i + 0.5), cy, nd));
      READER.forEach((nd, i) => {
        const x = cw * (i + 0.5);
        tx(svg, x, 114, nd.name, "tb", "middle");
        tx(svg, x, 128, nd.op, "tm", "middle");
        if (nd.st) tx(svg, x, 141, nd.st, "tm", "middle");
        tx(svg, x, 156, `${nd.c}×${nd.h}×${nd.w}`, nd.hi ? "ta" : "tm2", "middle");
        if (i) {
          const a = boxes[i - 1].right + 3, b = boxes[i].left - 3;
          if (b - a > 6) arrow(svg, a, cy, b, cy, "d-arrow", "d-head", 4.5);
        }
      });
      tx(svg, W - 4, 176, "T = 192 columns → greedy CTC", "ta", "end");
    } else {
      const rowH = 62, H = n * rowH + 18;
      svg = svgRoot(W, H, "Reader architecture with tensor shapes");
      const boxes = READER.map((nd, i) => slab(svg, 52, 32 + i * rowH, nd, 0.72));
      READER.forEach((nd, i) => {
        const y = 32 + i * rowH;
        tx(svg, 112, y - 4, nd.name, "tb");
        tx(svg, 112, y + 12, `${nd.op}${nd.st ? ` ${nd.st}` : ""} · ${nd.c}×${nd.h}×${nd.w}`, nd.hi ? "ta" : "tm2");
        if (i) arrow(svg, 52, boxes[i - 1].bottom + 3, 52, boxes[i].top - 3, "d-arrow", "d-head", 4.5);
      });
      tx(svg, 112, H - 4, "T = 192 columns → greedy CTC", "ta");
    }
    host.replaceChildren(svg);
  });
}

// ---------------------------------------------------------------- 6 detector architecture (hxh_ocr/det_model.py; MACs from the journal)

function detArch() {
  const host = $("d-det-arch");
  if (!host) return;
  const W = 790, H = 372;
  const svg = svgRoot(W, H, "DBNet-lite detector architecture");
  const box = (x, y, w, h, a, b, cls = "d-box") => {
    el("rect", { x, y, width: w, height: h, rx: 3, class: cls }, svg);
    tx(svg, x + w / 2, y + h / 2 - 2, a, "t", "middle");
    if (b) tx(svg, x + w / 2, y + h / 2 + 12, b, cls === "d-hi" ? "ta" : "tm", "middle");
    return { x, y, w, h, cx: x + w / 2, cy: y + h / 2 };
  };
  tx(svg, 14, 14, "backbone", "tm");
  tx(svg, 176, 14, "FPN-lite", "tm");
  tx(svg, 640, 14, "head", "tm");
  const bb = [
    box(14, 24, 124, 36, "input", "640² × 1"),
    box(14, 74, 124, 36, "stem 3×3 s2", "320² × 32"),
  ];
  const lv = [["stage 1 · s4", "160² × 16"], ["stage 2 · s8", "80² × 32"], ["stage 3 · s16", "40² × 48"], ["stage 4 · s32", "20² × 64"]];
  const ys = [124, 178, 232, 286];
  lv.forEach(([a, b], i) => bb.push(box(14, ys[i], 124, 40, a, b)));
  for (let i = 1; i < bb.length; i++) arrow(svg, 76, bb[i - 1].y + bb[i - 1].h, 76, bb[i].y - 1, "d-arrow", "d-head", 4.5);
  const lat = ys.map((y) => box(176, y, 96, 40, "1×1 → 48", null, "d-box2"));
  const sm = ys.map((y) => box(310, y, 108, 40, "dw3×3 + 1×1", "→ 16 ch", "d-box2"));
  ys.forEach((y, i) => {
    arrow(svg, 138, y + 20, 175, y + 20, "d-arrow", "d-head", 4.5);
    arrow(svg, 272, y + 20, 309, y + 20, "d-arrow", "d-head", 4.5);
  });
  for (let i = 3; i > 0; i--) { // top-down: up x2 and add
    const x = 260;
    arrow(svg, x, ys[i], x, ys[i - 1] + 41, "d-arrow-a", "d-head-a", 4.5);
    tx(svg, x - 6, (ys[i] + ys[i - 1] + 40) / 2 + 4, "↑2 +", "ta", "end");
  }
  const ups = ["×1", "×2", "×4", "×8"];
  const cat = box(500, 150, 92, 44, "concat", "160² × 64", "d-box2");
  const fuse = box(500, 222, 92, 40, "1×1 conv", "160² × 64", "d-box2");
  ys.forEach((y, i) => {
    tx(svg, 438, y + 16, `up ${ups[i]}`, "ta");
    el("path", { d: `M418,${y + 20} C460,${y + 20} 470,${cat.cy} 499,${cat.cy}`, class: "d-arrow" }, svg);
  });
  arrow(svg, 499, cat.cy, 500, cat.cy, "d-arrow", "d-head", 4.5);
  arrow(svg, cat.cx, cat.y + cat.h, cat.cx, fuse.y - 1, "d-arrow", "d-head", 4.5);
  const hx = 638, hw = 140;
  const hd = [
    box(hx, 24, hw, 40, "3×3 → 32", "160² × 32"),
    box(hx, 84, hw, 40, "nearest up ×2", "320² × 32"),
    box(hx, 144, hw, 40, "3×3 → 16", "320² × 16"),
    box(hx, 204, hw, 44, "1×1 → 2 · stride 2", "320² × 2", "d-hi"),
  ];
  for (let i = 1; i < hd.length; i++) arrow(svg, hd[i].cx, hd[i - 1].y + hd[i - 1].h, hd[i].cx, hd[i].y - 1, "d-arrow", "d-head", 4.5);
  el("path", { d: `M592,${fuse.cy} C618,${fuse.cy} 612,${hd[0].cy} 636,${hd[0].cy}`, class: "d-arrow" }, svg);
  arrow(svg, 635, hd[0].cy, 637, hd[0].cy, "d-arrow", "d-head", 4.5);
  tx(svg, hx + hw / 2, 266, "ch 0: text-kernel logit", "tm2", "middle");
  tx(svg, hx + hw / 2, 280, "ch 1: vertical logit", "tm2", "middle");
  el("rect", { x: hx, y: 294, width: hw, height: 34, rx: 3, class: "d-dash" }, svg);
  tx(svg, hx + hw / 2, 308, "threshold head", "tm2", "middle");
  tx(svg, hx + hw / 2, 321, "training only", "tm", "middle");
  el("path", { d: `M546,${fuse.y + fuse.h} C546,312 600,311 636,311`, class: "d-dash" }, svg);
  // MACs (journal: backbone 442 M, FPN 180 M, head 947 M; total 1.57 G at 640 x 640)
  const brace = (x0, x1, label) => {
    el("path", { d: `M${x0},340 L${x0},346 L${x1},346 L${x1},340`, class: "d-line" }, svg);
    tx(svg, (x0 + x1) / 2, 362, label, "tm2", "middle");
  };
  brace(14, 138, "backbone 442 M MACs");
  brace(176, 592, "FPN-lite 180 M");
  brace(hx, hx + hw, "head 947 M");
  host.replaceChildren(svg);
}

// ---------------------------------------------------------------- 7 target geometry (equations 3 and 7)

function geometry() {
  const host = $("d-geom"), ctl = $("d-geom-ctl"), out = $("d-geom-out");
  if (!host || !ctl) return;
  const r = 0.4, rho = 1.5;
  let ratio = 8;
  const lab = html("label", null, "length / thickness ", ctl);
  const input = html("input", null, null, lab);
  Object.assign(input, { type: "range", min: "1", max: "16", step: "0.5", value: String(ratio) });
  input.setAttribute("aria-label", "Line length as a multiple of its thickness");
  const val = html("span", null, `${ratio}`, lab);
  const draw = (W) => {
    const l = ratio, A = l, L = 2 * (l + 1), D = (A * (1 - r * r)) / L;
    const lk = l - 2 * D, hk = 1 - 2 * D;
    const Dp = (rho * lk * hk) / (2 * (lk + hk)), lu = lk + 2 * Dp, hu = hk + 2 * Dp;
    const pad = 24, maxW = W - 2 * pad;
    const h = Math.min(64, maxW / (l + 2 * D)), narrow = W < 520;
    const drawH = h * (1 + 2 * D), H = Math.round(drawH + 24 + (narrow ? 74 : 30));
    const cx = W / 2, cy = 12 + drawH / 2;
    const svg = svgRoot(W, H, "Shrink, threshold band and unclip of one line");
    const rect = (w, hh, attrs) => el("rect", { x: cx - (w * h) / 2, y: cy - (hh * h) / 2, width: w * h, height: hh * h, ...attrs }, svg);
    for (let k = 0; k < 6; k++) { // threshold band: peaks at the edge (0.7), falls to 0.3 at distance D
      const sw = 2 * D * h * (1 - k / 6);
      rect(l, 1, { fill: "none", "stroke-width": sw.toFixed(2), "stroke-linejoin": "round", style: "stroke: color-mix(in srgb, var(--vert) 13%, transparent)" });
    }
    rect(lk, hk, { class: "d-accent", opacity: 0.85 });
    rect(lu, hu, { class: "d-accent-s", "stroke-dasharray": "5 4" });
    rect(l, 1, { fill: "none", style: "stroke: var(--text); stroke-width: 1.5" });
    const ly = 12 + drawH + 26, items = [
      ["ground truth = exact inverse", (x, y) => el("rect", { x, y: y - 8, width: 14, height: 9, fill: "none", style: "stroke: var(--text); stroke-width: 1.5" }, svg)],
      ["kernel", (x, y) => el("rect", { x, y: y - 8, width: 14, height: 9, class: "d-accent" }, svg)],
      ["unclip 1.5", (x, y) => el("rect", { x, y: y - 8, width: 14, height: 9, class: "d-accent-s", "stroke-dasharray": "3 2" }, svg)],
      ["threshold band ±D", (x, y) => el("rect", { x, y: y - 8, width: 14, height: 9, style: "fill: color-mix(in srgb, var(--vert) 35%, transparent)" }, svg)],
    ];
    items.forEach(([name, sw], i) => {
      const x = narrow ? 4 : 4 + [0, 0.34, 0.54, 0.76][i] * W, y = ly + (narrow ? i * 20 : 0);
      sw(x, y);
      tx(svg, x + 20, y, name, "tm2");
    });
    host.replaceChildren(svg);
    out.textContent = `D = ${D.toFixed(3)} h · kernel ${hk.toFixed(3)} h thick · unclip 1.5 → ${hu.toFixed(3)} h · exact inverse → 1.000 h`;
  };
  let width = 0;
  responsive(host, (W) => { width = W; draw(W); });
  input.addEventListener("input", () => { ratio = Number(input.value); val.textContent = `${ratio}`; draw(width || host.clientWidth); });
}

// ---------------------------------------------------------------- 8 int8 grids (img/deep/data.json: exponents.sites)

function siteName(n) {
  if (n === "x") return "input (fixed)";
  if (n === "bn1") return "stem";
  if (n === "mean") return "mean over H";
  if (n === "head_cls") return "classifier";
  let m = n.match(/^blocks_(\d)_(\d)_(dw_start|dw_mid|pw_exp|pw_proj)/);
  if (m) return `blocks.${m[1]}.${m[2]} ${m[3]}`;
  m = n.match(/^blocks_(\d)_(\d)_bn1$/);
  if (m) return `blocks.${m[1]}.${m[2]} conv`;
  m = n.match(/^head_blocks_(\d)_act(\d)$/);
  if (m) return `head.${m[1]} ${m[2] === "1" ? "dw" : "pw"}`;
  m = n.match(/^add(?:_(\d+))?$/);
  if (m) return `add ${m[1] ?? 0}`;
  return n;
}

function int8Grid(D) {
  const host = $("d-int8"), ctl = $("d-int8-ctl"), out = $("d-int8-out");
  if (!host || !ctl) return;
  const sites = D.exponents.sites;
  const lab = html("label", null, "tensor ", ctl);
  const sel = html("select", null, null, lab);
  sel.setAttribute("aria-label", "Quantised tensor of the 48 px reader");
  sites.forEach((s, i) => {
    const o = html("option", null, `${siteName(s.node)} · e = ${s.exp}`, sel);
    o.value = String(i);
  });
  let cur = Math.max(0, sites.findIndex((s) => s.node === "bn1"));
  sel.value = String(cur);
  const draw = (W) => {
    const s = sites[cur], e = s.exp;
    const rows = [e - 1, e, e + 1];
    const xMax = Math.max(127 * 2 ** (e + 1), s.clip) * 1.06;
    const L = 64, R = 16, T = 22, rowH = 40, H = T + rows.length * rowH + 36;
    const sx = (v) => L + (v / xMax) * (W - L - R);
    const svg = svgRoot(W, H, `int8 grids around exponent ${e}`);
    rows.forEach((k, i) => {
      const y = T + i * rowH + 8, top = 127 * 2 ** k, step = 2 ** k;
      const chosen = k === e;
      el("rect", { x: sx(0), y, width: sx(top) - sx(0), height: 16, rx: 2, class: chosen ? "d-hi" : "d-box" }, svg);
      let m = 1;
      while (((sx(step * m) - sx(0)) < 4) && m < 128) m *= 2;
      for (let q = 0; q <= 127; q += m) {
        const x = sx(q * step);
        el("line", { x1: x, x2: x, y1: y + (q % (m * 8) ? 10 : 4), y2: y + 16, class: chosen ? "d-accent-s" : "d-line", "stroke-width": 1 }, svg);
      }
      tx(svg, L - 8, y + 12, `e = ${k}`, chosen ? "ta" : "tm", "end");
      const lbl = `step ${step >= 1 ? step : `2^${k}`} · max ${+top.toFixed(4)}${m > 1 ? ` · every ${m === 2 ? "2nd" : `${m}th`} tick` : ""}`;
      tx(svg, L, y + 30, lbl, chosen ? "tm2" : "tm");
    });
    const axY = T + rows.length * rowH + 6;
    el("line", { x1: sx(0), x2: W - R, y1: axY, y2: axY, class: "d-axis" }, svg);
    const nice = [0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200].find((n) => xMax / n <= 6) || 500;
    for (let v = 0; v <= xMax; v += nice) {
      el("line", { x1: sx(v), x2: sx(v), y1: axY, y2: axY + 4, class: "d-axis" }, svg);
      tx(svg, sx(v), axY + 16, `${+v.toFixed(2)}`, "tm", "middle");
    }
    if (s.clip > 0) {
      el("line", { x1: sx(s.clip), x2: sx(s.clip), y1: T, y2: axY, style: "stroke: var(--foreign); stroke-width: 1.5; stroke-dasharray: 4 3" }, svg);
      el("line", { x1: sx(s.clip), x2: sx(s.clip), y1: 4, y2: T, style: "stroke: var(--foreign); stroke-width: 1.5; stroke-dasharray: 4 3" }, svg);
      tx(svg, sx(s.clip) + (sx(s.clip) > W - 110 ? -4 : 4), 12, `c = ${s.clip}`, "tf", sx(s.clip) > W - 110 ? "end" : "start");
    }
    host.replaceChildren(svg);
    const l2 = s.clip > 0 ? Math.log2(s.clip / 128) : null;
    out.textContent = `${siteName(s.node)} · e = ${e} · step 2^${e} = ${2 ** e} · int8 range [${-128 * 2 ** e}, ${127 * 2 ** e}]` +
      (l2 === null ? " · input exponent fixed at −7" : ` · KL clip c = ${s.clip} · log2(c/128) = ${l2.toFixed(2)} → ${e}`);
  };
  let width = 0;
  responsive(host, (W) => { width = W; draw(W); });
  sel.addEventListener("change", () => { cur = Number(sel.value); draw(width || host.clientWidth); });
}

// ---------------------------------------------------------------- 9 requantisation chain

function requant() {
  const host = $("d-requant");
  if (!host) return;
  const nodes = [
    ["int8 x, w", "exponents", "e_x, e_w"],
    ["Σ x·w + bias", "int32 at", "e_x + e_w"],
    ["shift s", "s = e_y", "− e_x − e_w"],
    ["round half up", "(a+2^(s−1))", "≫ s"],
    ["saturate", "clamp to", "[−128, 127]"],
    ["ReLU", "if fused", ""],
    ["int8 y", "exponent", "e_y", true],
  ];
  responsive(host, (W) => {
    let svg;
    if (W >= 600) {
      const n = nodes.length, gap = 10, bw = (W - gap * (n - 1)) / n, H = 74;
      svg = svgRoot(W, H, "Integer path of one convolution");
      nodes.forEach(([a, b, c, hi], i) => {
        const x = i * (bw + gap);
        el("rect", { x, y: 6, width: bw, height: 62, rx: 3, class: hi ? "d-hi" : "d-box" }, svg);
        tx(svg, x + bw / 2, 26, a, "t", "middle");
        tx(svg, x + bw / 2, 42, b, hi ? "ta" : "tm", "middle");
        if (c) tx(svg, x + bw / 2, 56, c, hi ? "ta" : "tm", "middle");
        if (i) arrow(svg, x - gap + 1, 37, x - 1, 37, "d-arrow", "d-head", 4);
      });
    } else {
      const rowH = 50, H = nodes.length * rowH;
      svg = svgRoot(W, H, "Integer path of one convolution");
      const bw = Math.min(W, 320), x = (W - bw) / 2;
      nodes.forEach(([a, b, c, hi], i) => {
        const y = i * rowH;
        el("rect", { x, y: y + 2, width: bw, height: 36, rx: 3, class: hi ? "d-hi" : "d-box" }, svg);
        tx(svg, x + 12, y + 24, a, "t");
        tx(svg, x + bw - 12, y + 24, `${b} ${c}`.trim(), hi ? "ta" : "tm", "end");
        if (i) arrow(svg, W / 2, y - 10, W / 2, y + 1, "d-arrow", "d-head", 4.5);
      });
    }
    host.replaceChildren(svg);
  });
}

// ---------------------------------------------------------------- 10 exponents per conv (data.json: exponents.convs)

function exponents(D) {
  const host = $("d-exponents"), out = $("d-exp-out");
  if (!host) return;
  const convs = D.exponents.convs;
  const kinds = { full: ["3×3 conv weights", "var(--bar)"], depthwise: ["depthwise weights", "var(--vert)"], pointwise: ["1×1 weights", "var(--accent)"] };
  const legend = html("div", "legend-row", null);
  for (const [, [name, color]] of Object.entries(kinds)) {
    const s = html("span", null, null, legend);
    const i = html("i", null, null, s);
    i.style.cssText = `border:0;width:9px;height:9px;border-radius:50%;background:${color};vertical-align:0`;
    s.append(name);
  }
  const sq = html("span", null, null, legend);
  const i2 = html("i", null, null, sq);
  i2.style.cssText = "border:1.5px solid var(--text-2);width:8px;height:8px;border-radius:1px;vertical-align:0";
  sq.append("output activation exponent");
  legend.style.margin = "0 0 8px";
  host.before(legend);
  const say = (k) => {
    const c = convs[k], s = c.out - c.in - c.w;
    out.textContent = `${c.name} · ${c.kind} · weights e_w = ${c.w} · input e_x = ${c.in} · output e_y = ${c.out} · shift s = ${s}${s <= 0 ? " (left shift)" : ""}`;
  };
  responsive(host, (W) => {
    const H = 250, L = 34, R = 10, T = 22, B = 24, yMin = -9, yMax = 1;
    const sx = (i) => L + ((i + 0.5) / convs.length) * (W - L - R);
    const sy = (v) => T + ((yMax - v) / (yMax - yMin)) * (H - T - B);
    const svg = svgRoot(W, H, "Weight and output exponents of every convolution");
    for (let v = -8; v <= 0; v += 1) {
      el("line", { x1: L, x2: W - R, y1: sy(v), y2: sy(v), class: v === 0 ? "d-axis" : "d-grid" }, svg);
      if (v % 2 === 0) tx(svg, L - 6, sy(v) + 3.5, String(v), "tm", "end");
    }
    // stage separators from the names
    const stageOf = (n) => (n.startsWith("head") ? "head" : n.startsWith("blocks.3") ? "stage 4" : n.startsWith("blocks.2") ? "stage 3" : "stem–2");
    const groups = [];
    convs.forEach((c, i) => {
      const st = stageOf(c.name);
      if (!groups.length || groups[groups.length - 1].st !== st) groups.push({ st, i0: i, n: 0 });
      groups[groups.length - 1].n++;
    });
    const cw = sx(1) - sx(0);
    groups.forEach((g, k) => {
      const x = sx(g.i0) - cw / 2;
      if (k) el("line", { x1: x, x2: x, y1: T - 14, y2: H - B, class: "d-grid" }, svg);
      if (g.n * cw >= g.st.length * 6.4 + 6) tx(svg, x + 4, T - 6, g.st, "tm");
    });
    convs.forEach((c, i) => {
      const s = 7;
      el("rect", { x: sx(i) - s / 2, y: sy(c.out) - s / 2, width: s, height: s, rx: 1, style: "fill: none; stroke: var(--text-2); stroke-width: 1.2" }, svg);
      const dot = el("circle", { cx: sx(i), cy: sy(c.w), r: 3.8 }, svg);
      dot.style.fill = kinds[c.kind][1];
    });
    tx(svg, W - R, H - 6, "convolutions in network order →", "tm", "end");
    if (W >= 480) tx(svg, L, H - 6, "exponent e", "tm");
    const guide = el("line", { x1: 0, x2: 0, y1: T, y2: H - B, class: "ax-strong", visibility: "hidden", style: "stroke: var(--line-strong)" }, svg);
    const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect(), px = ((e.clientX - r.left) / r.width) * W;
      const k = Math.max(0, Math.min(convs.length - 1, Math.round(((px - L) / (W - L - R)) * convs.length - 0.5)));
      guide.setAttribute("x1", sx(k)); guide.setAttribute("x2", sx(k)); guide.setAttribute("visibility", "visible");
      say(k);
    });
    host.replaceChildren(svg);
  });
  say(convs.findIndex((c) => c.name === "blocks.3.2.dw_mid.conv"));
}

// ---------------------------------------------------------------- 11 QAT curves (data.json: qat_curves)

function qatCurves(D) {
  const host = $("d-qat"), ctl = $("d-qat-ctl");
  if (!host || !ctl) return;
  const S = D.qat_curves.series;
  html("span", null, "model ", ctl);
  segmented(ctl, S.map((s) => s.label), (i) => {
    const s = S[i];
    const top = Math.max(...s.int8.map((p) => p[1]), ...s.float.map((p) => p[1]));
    lineChart(host, {
      title: `QAT fine-tune: ${s.label} (${s.id})`,
      sub: "validation CER on 2,000 lines; synth_v4; lower is better",
      series: [
        { name: "int8 (fake-quantised)", short: "int8", color: "var(--accent)", points: s.int8 },
        { name: "same weights in float", short: "float", color: "var(--muted)", points: s.float, dash: true },
      ],
      yMin: 0, yMax: top * 1.04, yUnit: "%",
      xTicks: [0, 1000, 2000, 3000, 4000],
    });
  });
}

// ---------------------------------------------------------------- 12 training curves per round (charts.json)

function roundCurves(C) {
  const host = $("d-curves"), ctl = $("d-curves-ctl");
  if (!host || !ctl) return;
  const ids = [["ablation_round1", "Round 1"], ["round2_v3", "Round 2"], ["recognizers_v4", "Round 3"]];
  const groups = ids.map(([id]) => C.curves.find((c) => c.id === id)).filter(Boolean);
  if (!groups.length) { hideFig("d-curves"); return; }
  const colors = ["var(--accent)", "var(--vert)", "var(--bar)"];
  html("span", null, "round ", ctl);
  segmented(ctl, groups.map((g) => ids.find(([id]) => id === g.id)[1]), (i) => {
    const g = groups[i];
    lineChart(host, {
      title: g.title,
      sub: "val CER (%) on a fixed 2,000-line subset, log scale; lower is better",
      series: g.series.map((s, k) => ({ name: s.label, short: s.label.replace("MobileNetV4-small", "mnv4s"), color: colors[k % 3], points: s.points })),
      log: true, yMin: 0.5, yMax: 20, yTicks: [0.5, 1, 2, 5, 10, 20], yUnit: "%", yDigits: 1,
      xTicks: [1000, 10000, 20000, 30000], xLabel: "",
    });
  });
}

// ---------------------------------------------------------------- 13 confusions (journal: round 2, mnv4s 32 px, full v3 val)

function confusions(G) {
  const host = $("d-confusions");
  if (!host) return;
  const pairs = [["が", "か", 12], ["ぽ", "ぼ", 10], ["ぴ", "び", 9], ["ば", "は", 9], ["だ", "た", 9], ["ぱ", "ば", 9],
    ["ぷ", "ぶ", 7], ["ぼ", "ぽ", 7], ["ぺ", "べ", 7], ["げ", "け", 7], ["べ", "へ", 7]];
  host.replaceChildren();
  for (const [a, b, n] of pairs) {
    const card = html("div", "gcard", null, host);
    const svg = svgRoot(150, 62, `${a} read as ${b}, ${n} times`);
    drawKana(svg, G, a, 36, 30, 31);
    arrow(svg, 66, 31, 84, 31, "d-arrow", "d-head", 4.5);
    drawKana(el("g", { class: "g-acc" }, svg), G, b, 36, 114, 31);
    card.append(svg);
    html("div", "c-t", `${a} → ${b}`, card);
    const s = html("div", "c-s", `${n}×`, card);
    const bar = html("div", null, null, card);
    bar.style.cssText = `height:3px;margin-top:4px;border-radius:2px;background:var(--accent);width:${(n / 12) * 100}%`;
    s.setAttribute("aria-label", `${n} times`);
  }
}

// ---------------------------------------------------------------- 14 detector curve (charts.json: detector_det_v1)

function detCurve(C) {
  const host = $("d-det-curve");
  const g = C.curves.find((c) => c.id === "detector_det_v1");
  if (!host || !g) { hideFig("d-det-curve"); return; }
  const s = g.series[0];
  lineChart(host, {
    title: "det_v1: validation detection while training",
    sub: "300 pages_v2 val pages, quad IoU ≥ 0.5, every 2,000 steps; higher is better",
    series: [
      { name: "F1", color: "var(--accent)", points: s.points },
      { name: "precision", color: "var(--bar)", points: s.precision },
      { name: "recall", color: "var(--vert)", points: s.recall },
    ],
    yMin: 75, yMax: 95, yUnit: "%", yTicks: [75, 80, 85, 90, 95],
    xTicks: [2000, 12000, 24000, 36000],
  });
}

// ---------------------------------------------------------------- 15 confidence filter (charts.json: confidence_filter)

function confSweep(C) {
  const t = C.tables.find((x) => x.id === "confidence_filter");
  if (!t) { hideFig("d-conf-a"); return; }
  const rows = t.rows;
  const lab = (i) => (rows[i].min_conf ? String(rows[i].min_conf) : "none");
  const pts = (k) => rows.map((r, i) => [i, r[k]]);
  const common = { xFmt: (i) => lab(Math.round(i)), xTicks: rows.map((_, i) => i), xName: "min confidence" };
  lineChart($("d-conf-a"), {
    title: "Detection with a minimum line confidence", sub: "precision, recall and F1 (%)",
    series: [
      { name: "precision", color: "var(--bar)", points: pts("precision") },
      { name: "recall", color: "var(--vert)", points: pts("recall") },
      { name: "F1", color: "var(--accent)", points: pts("f1") },
    ],
    yMin: 80, yMax: 96, yTicks: [80, 84, 88, 92, 96], yUnit: "%", height: 220, ...common,
  });
  lineChart($("d-conf-b"), {
    title: "Character error rates", sub: "end-to-end CER counts hidden lines as missed; matched-line CER does not",
    series: [
      { name: "CER h, end to end", short: "CER h", color: "var(--accent)", points: pts("cer_h") },
      { name: "CER v, end to end", short: "CER v", color: "var(--vert)", points: pts("cer_v") },
      { name: "CER of matched lines", short: "matched", color: "var(--bar)", points: pts("cer_matched") },
    ],
    yMin: 0, yMax: 25, yTicks: [0, 5, 10, 15, 20, 25], yUnit: "%", height: 220, ...common,
  });
  const body = $("d-conf-rows");
  if (body) {
    body.replaceChildren();
    for (const r of rows) {
      const tr = html("tr", r.min_conf === 0.85 ? "hi" : null, null, body);
      const cells = [r.min_conf ? String(r.min_conf) : "none", ...["precision", "recall", "f1", "cer_h", "cer_v", "cer_matched", "exact_h", "exact_v"].map((k) => pct(r[k]))];
      cells.forEach((c, i) => {
        const td = html("td", "num", null, tr);
        if (r.min_conf === 0.85 && i === 0) { const b = html("strong", null, `${c} (demo)`, td); b.title = "deployed"; } else td.textContent = c;
      });
    }
  }
}

// ---------------------------------------------------------------- 16 timings (img/pipeline/pipeline.json)

function timings(P) {
  const host = $("d-timings");
  if (!host) return;
  const t = P.timings_ms;
  if (!t) { hideFig("d-timings"); return; }
  const parts = [
    ["letterbox", t.letterbox, "var(--line-strong)"],
    ["detector (int8)", t.detector, "var(--accent)"],
    ["post-processing", t.post, "var(--vert)"],
    [`read ${P.line_count} lines`, t.read_all_lines, "var(--bar)"],
  ];
  const sum = parts.reduce((s, p) => s + p[1], 0);
  const sub = $("d-timings-sub");
  if (sub) sub.textContent = `${P.page.width} × ${P.page.height} px page, ${P.line_count} detected lines; node, one thread, wasm SIMD, CPU shared; median of ${t.runs} runs`;
  host.replaceChildren();
  const bar = html("div", "stack", null, host);
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", parts.map((p) => `${p[0]} ${p[1]} ms`).join(", "));
  for (const [, v, c] of parts) {
    const d = html("div", null, null, bar);
    d.style.flex = `${v} 0 0`;
    d.style.background = c;
  }
  const lg = html("div", "stack-legend", null, host);
  for (const [name, v, c] of parts) {
    const s = html("span", null, null, lg);
    html("i", null, null, s).style.background = c;
    s.append(name);
    const val = html("span", "v", null, s);
    html("b", null, `${v.toFixed(v < 10 ? 2 : 1)} ms`, val);
    val.append(` · ${Math.round((v / sum) * 100)}%`);
  }
  const foot = html("p", "table-note", `Total ${t.page_total.toFixed(1)} ms; one line (crop, prepare, read) ${t.read_per_line_mean} ms on its own. Source: img/pipeline/pipeline.json.`, host);
  foot.style.marginTop = "12px";
}

// ---------------------------------------------------------------- 17 tap a line (schematic; numbers from the journal)

function tapFig() {
  const host = $("d-tap");
  if (!host) return;
  const W = 600, H = 300;
  const svg = svgRoot(W, H, "Tap-a-line windows on a 3,000 px photo");
  svg.style.maxWidth = "660px";
  const P = 260, ox = 16, oy = 20, k = P / 3000;
  el("rect", { x: ox, y: oy, width: P, height: P, class: "d-box" }, svg);
  tx(svg, ox, oy - 6, "photo 3,000 × 3,000 px", "tm2");
  // a small page pasted into it, with tiny lines
  const px = ox + 120, py = oy + 120, pw = 70, ph = 52;
  el("rect", { x: px, y: py, width: pw, height: ph, class: "d-box2" }, svg);
  for (let i = 0; i < 5; i++) el("rect", { x: px + 6, y: py + 7 + i * 9, width: 34 + ((i * 13) % 22), height: 3, class: "d-faint" }, svg);
  const tapX = px + 22, tapY = py + 26;
  const win = (side, n, cls, where) => {
    const s = side * k, x = Math.max(ox, Math.min(ox + P - s, tapX - s / 2)), y = Math.max(oy, Math.min(oy + P - s, tapY - s / 2));
    el("rect", { x, y, width: s, height: s, class: cls }, svg);
    const cl = cls === "d-accent-s" ? "ta" : "tm";
    if (where === "top") tx(svg, x + s + 4, y + 9, n, cl);
    else if (where === "right") tx(svg, x + s + 4, y + s - 3, n, cl);
    else tx(svg, x - 3, y + s + 12, n, cl, "end");
  };
  win(1280, "2 · 1,280 px", "d-dash", "top");
  win(640, "1 · 640 px", "d-accent-s", "right");
  win(320, "3 · 320 px", "d-dash", "below");
  el("circle", { cx: tapX, cy: tapY, r: 4.5, class: "d-accent" }, svg);
  el("circle", { cx: tapX, cy: tapY, r: 9, class: "d-accent-s", opacity: 0.5 }, svg);
  // right panel
  const x = 312;
  tx(svg, x, 40, "Page pass", "tb");
  tx(svg, x, 58, "whole photo shrunk to 640 px (× 0.21)", "tm2");
  tx(svg, x, 74, "found 4 of the page's 14 lines", "tm2");
  tx(svg, x, 112, "Tap", "tb");
  tx(svg, x, 130, "windows around the tap, cut from the", "tm2");
  tx(svg, x, 146, "original: 640 px first (no shrinking),", "tm2");
  tx(svg, x, 162, "then 1,280 px, then 320 px", "tm2");
  tx(svg, x, 178, "read 13 of 14 lines exactly", "ta");
  tx(svg, x, 194, "75–110 ms per tap", "ta");
  tx(svg, x, 232, "The first window with a line under the tap", "tm");
  tx(svg, x, 246, "(or within 1.5 line thicknesses) wins.", "tm");
  tx(svg, ox, H - 6, "schematic; window sizes to scale", "tm");
  host.replaceChildren(svg);
}

// ---------------------------------------------------------------- 18 false positives (numbers measured with page_e2e.mjs --conf-sweep)

function fpChart() {
  const host = $("d-fp");
  if (!host || !HX) { hideFig("d-fp"); return; }
  HX.barChart(host, {
    title: "Shown false lines per page, det_v1 vs det_v2",
    sub: "browser pipeline, unclip 1.5, margins 0.3 / 0.2; a false line is on text or icons if ≥ 40% of it overlaps them",
    unit: "", digits: 2,
    series: [
      { name: "on ordinary text", color: "var(--foreign)" },
      { name: "on icons", color: "var(--icon)" },
      { name: "elsewhere", color: "var(--line-strong)" },
    ],
    rows: [
      { group: "pages_foreign · 150 pages", label: "det_v1, no filter", values: [25.0, 2.3, 2.67] },
      { group: "pages_foreign · 150 pages", label: "det_v1, min conf 0.85", values: [2.39, 0.95, 1.17] },
      { group: "pages_foreign · 150 pages", label: "det_v2, no filter", values: [0.43, 0.16, 0.10] },
      { group: "pages_foreign · 150 pages", label: "det_v2, min conf 0.8", values: [0.15, 0.11, 0.07] },
      { group: "pages_v3 val · 100 mixed pages", label: "det_v1, no filter", values: [13.59, 0.63, 1.35] },
      { group: "pages_v3 val · 100 mixed pages", label: "det_v1, min conf 0.85", values: [1.96, 0.28, 0.6] },
      { group: "pages_v3 val · 100 mixed pages", label: "det_v2, no filter", values: [0.80, 0.14, 0.59] },
      { group: "pages_v3 val · 100 mixed pages", label: "det_v2, min conf 0.8", values: [0.29, 0.08, 0.46] },
    ],
  });
}

// ---------------------------------------------------------------- 19 live tracker (schematic of web/ocr.js LiveTracker)

function tracker() {
  const host = $("d-track");
  if (!host) return;
  const A = "はんたー", B = "はんだー";
  // per frame: the reader's output (null = reused, not read) and what the track shows
  const frames = [
    { read: A, shown: null }, { read: A, shown: A }, { read: B, shown: A }, { read: A, shown: A },
    { read: A, shown: A }, { read: A, shown: A }, { read: null, shown: A }, { read: null, shown: A },
  ];
  const W = 740, H = 224, x0 = 132, fw = 75;
  const svg = svgRoot(W, H, "Track and vote across frames");
  [["frame", 22, "tm"], ["detector", 46, "tm2"], ["reader", 90, "tm2"], ["vote over last 5", 142, "tm2"], ["shown", 174, "tm2"]]
    .forEach(([n, y, c]) => tx(svg, 10, y, n, c));
  frames.forEach((f, i) => {
    const cx = x0 + i * fw + fw / 2;
    tx(svg, cx, 22, `t${i + 1}`, "tm", "middle");
    el("rect", { x: cx - 24, y: 32, width: 48, height: 19, rx: 2, class: "d-hi" }, svg);
    tx(svg, cx, 45, "box", "ta", "middle");
    if (f.read) {
      el("rect", { x: cx - 32, y: 74, width: 64, height: 26, rx: 3, class: f.read === B ? "d-warm" : "d-box" }, svg);
      tx(svg, cx, 92, f.read, "tk-s", "middle");
    } else {
      el("rect", { x: cx - 32, y: 74, width: 64, height: 26, rx: 3, class: "d-dash" }, svg);
      tx(svg, cx, 91, "reused", "tm", "middle");
    }
    if (f.shown) {
      el("rect", { x: cx - 32, y: 158, width: 64, height: 26, rx: 3, class: "d-hi" }, svg);
      tx(svg, cx, 176, f.shown, "tk-s", "middle");
    } else {
      tx(svg, cx, 175, "hidden", "tm", "middle");
    }
  });
  tx(svg, x0 + 2 * fw + fw / 2, 114, "misread", "tf", "middle");
  // reuse: three equal reads (t4-t6) let t7 and t8 skip the reader
  const rx0 = x0 + 3 * fw + 6, rx1 = x0 + 6 * fw - 6;
  el("path", { d: `M${rx0},70 L${rx0},64 L${rx1},64 L${rx1},70`, class: "d-accent-s" }, svg);
  tx(svg, rx0, 60, "3 equal reads in a row → reuse for 2 frames", "ta");
  // vote window at t6: t2..t6
  const vx0 = x0 + fw + 6, vx1 = x0 + 6 * fw - 6;
  el("path", { d: `M${vx0},118 L${vx0},124 L${vx1},124 L${vx1},118`, class: "d-line" }, svg);
  tx(svg, (vx0 + vx1) / 2, 142, `at t6: ${A} ×4, ${B} ×1 → ${A}`, "tk-s", "middle");
  tx(svg, x0 + fw / 2, 200, "seen once", "tm", "middle");
  tx(svg, W - 8, 218, "schematic, readings illustrative", "tm", "end");
  host.replaceChildren(svg);
}

// ---------------------------------------------------------------- figure and table numbers, cross-references

function numberAll() {
  // the page's own words for "Fig." and "Table" (<html data-fig data-table>), e.g. 図 / 表
  const FIG = document.documentElement.dataset.fig || "Fig.", TAB = document.documentElement.dataset.table || "Table";
  let f = 0, t = 0;
  const map = {};
  for (const fig of document.querySelectorAll("article figure")) {
    const fc = fig.querySelector("figcaption");
    if (!fc || fig.hidden) continue;
    fc.dataset.raw ??= fc.textContent;
    const raw = fc.dataset.raw;
    if (raw.startsWith(`${FIG} # —`)) { f++; fc.textContent = raw.replace("#", String(f)); if (fig.id) map[fig.id] = `${FIG} ${f}`; }
    else if (raw.startsWith(`${TAB} # —`)) { t++; fc.textContent = raw.replace("#", String(t)); if (fig.id) map[fig.id] = `${TAB} ${t}`; }
  }
  for (const a of document.querySelectorAll("a.xref")) {
    const id = (a.getAttribute("href") || "").slice(1);
    if (map[id]) a.textContent = map[id];
  }
}

// ---------------------------------------------------------------- go

function safe(name, fn) {
  try { fn(); } catch (e) { console.warn(`deep.js: ${name} failed`, e); }
}

overview();
safe("reader", readerArch);
safe("detector", detArch);
safe("geometry", geometry);
safe("requant", requant);
safe("tap", tapFig);
safe("tracker", tracker);
safe("fp", fpChart);
numberAll();

const jobs = [
  getJson("data/glyphs.json").then((G) => {
    safe("variants", () => variants(G));
    safe("marks", () => marks(G));
    safe("rotate", () => rotateFig(G));
    safe("keep", () => keepClear(G));
    safe("confusions", () => confusions(G));
  }).catch((e) => { console.warn(e); hideFig("d-variants"); hideFig("d-rotate"); hideFig("d-confusions"); safe("keep", () => keepClear(null)); }),
  getJson("img/deep/data.json").then((D) => {
    safe("int8", () => int8Grid(D));
    safe("exponents", () => exponents(D));
    safe("qat", () => qatCurves(D));
  }).catch((e) => { console.warn(e); hideFig("d-int8"); hideFig("d-exponents"); hideFig("d-qat"); }),
  getJson("data/charts.json").then((C) => {
    safe("curves", () => roundCurves(C));
    safe("det curve", () => detCurve(C));
    safe("conf", () => confSweep(C));
  }).catch((e) => { console.warn(e); hideFig("d-curves"); hideFig("d-det-curve"); hideFig("d-conf-a"); }),
  getJson("img/pipeline/pipeline.json").then((P) => safe("timings", () => timings(P)))
    .catch((e) => { console.warn(e); hideFig("d-timings"); }),
];
Promise.allSettled(jobs).then(() => {
  numberAll();
  setTimeout(numberAll, 1500); // blog.js may hide a figure after its own fetches fail
});
if (reduced()) document.documentElement.classList.add("reduced-motion");
