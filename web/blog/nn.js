// Neural-network figures for the explainer (blog/index.html), imported by explain.js:
//   tinyNet      a small real network you can poke (data/tinynet.json, scripts/blog/tiny_net.py)
//   insideReader the real line reader's feature maps, layer by layer (data/nn.json, img/nn/,
//                scripts/blog/nn_activations.py)

const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
};
const html = (tag, cls, parent, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  if (parent) parent.append(e);
  return e;
};
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const visibleWatch = (node) => {
  const s = { on: false };
  new IntersectionObserver((es) => { s.on = es.some((e) => e.isIntersecting); }, { threshold: 0.2 }).observe(node);
  return s;
};
const hideFig = (node) => { const f = node && node.closest("figure"); if (f) f.hidden = true; };
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}
const pct = (p) => (p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);

// any CSS colour (tokens included) as [r, g, b]
const rgbOf = (() => {
  let x;
  return (col) => {
    if (!x) { const c = document.createElement("canvas"); c.width = c.height = 1; x = c.getContext("2d", { willReadFrequently: true }); }
    x.clearRect(0, 0, 1, 1);
    x.fillStyle = "#000";
    x.fillStyle = col;
    x.fillRect(0, 0, 1, 1);
    const d = x.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2]];
  };
})();
const token = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function shapeOf(G, ch) {
  const comp = G.composed[ch];
  const g = G.glyphs[comp ? comp.base : ch];
  return g ? { form: g.forms[0], mark: comp ? comp.mark : null } : null;
}

/** A Hunter glyph as SVG strokes, centred at (cx, cy), em size S. */
function glyphInto(G, parent, ch, S, cx, cy, cls = "") {
  const sh = shapeOf(G, ch);
  if (!sh) return;
  const sw = G.stroke_width_em;
  for (const st of sh.form.strokes) {
    if (st.type === "dot") el("circle", { cx: cx + st.c[0] * S, cy: cy + st.c[1] * S, r: Math.max(st.r, sw / 2) * S, class: `glyph-fill ${cls}` }, parent);
    else el("path", { d: st.pts.map((p, i) => `${i ? "L" : "M"}${(cx + p[0] * S).toFixed(1)},${(cy + p[1] * S).toFixed(1)}`).join("") + (st.closed ? "Z" : ""), class: `glyph-stroke ${cls}`, "stroke-width": (sw * S).toFixed(2) }, parent);
  }
  if (sh.mark) {
    const m = G.marks[sh.mark], [, , x1, y1] = sh.form.bbox;
    const mx = cx + (x1 + m.dx) * S, my = cy + (y1 + m.dy) * S;
    if (m.hollow) el("circle", { cx: mx, cy: my, r: Math.max(m.r, sw * m.width_scale) * S, class: `glyph-stroke ${cls}`, "stroke-width": (sw * m.width_scale * S).toFixed(2) }, parent);
    else el("circle", { cx: mx, cy: my, r: Math.max(m.r, sw / 2) * S, class: `glyph-fill ${cls}` }, parent);
  }
}

/** Ink coverage of a symbol on the network's grid, drawn exactly as scripts/blog/tiny_net.py does. */
function rasterise(G, net, ch, seed = 0) {
  const R = net.canvas * 4, N = net.grid, S = net.em * R, c = R / 2;
  const cv = document.createElement("canvas");
  cv.width = cv.height = R;
  const x = cv.getContext("2d", { willReadFrequently: true });
  const sh = shapeOf(G, ch), sw = G.stroke_width_em * S;
  const r = seed ? rng(seed) : null;
  const j = () => (r ? (r() - 0.5) * 0.07 : 0);
  x.lineCap = x.lineJoin = "round";
  x.lineWidth = sw;
  x.fillStyle = x.strokeStyle = "#000";
  for (const st of sh.form.strokes) {
    x.beginPath();
    if (st.type === "dot") {
      x.arc(c + (st.c[0] + j()) * S, c + (st.c[1] + j()) * S, Math.max(st.r * S, sw / 2), 0, 7);
      x.fill();
    } else {
      st.pts.forEach(([px, py], i) => { const X = c + (px + j()) * S, Y = c + (py + j()) * S; if (i) x.lineTo(X, Y); else x.moveTo(X, Y); });
      if (st.closed) x.closePath();
      x.stroke();
    }
  }
  if (sh.mark) {
    const m = G.marks[sh.mark], [, , x1, y1] = sh.form.bbox;
    const mx = c + (x1 + m.dx + j()) * S, my = c + (y1 + m.dy + j()) * S;
    x.beginPath();
    if (m.hollow) {
      x.lineWidth = G.stroke_width_em * m.width_scale * S;
      x.arc(mx, my, Math.max(m.r, G.stroke_width_em * m.width_scale) * S, 0, 7);
      x.stroke();
    } else {
      x.arc(mx, my, Math.max(m.r * S, sw / 2), 0, 7);
      x.fill();
    }
  }
  const a = x.getImageData(0, 0, R, R).data, b = R / N, out = new Float32Array(N * N);
  for (let gy = 0; gy < N; gy++) for (let gx = 0; gx < N; gx++) {
    let s = 0;
    for (let yy = gy * b; yy < (gy + 1) * b; yy++) for (let xx = gx * b; xx < (gx + 1) * b; xx++) s += a[(yy * R + xx) * 4 + 3];
    out[gy * N + gx] = s / (b * b * 255);
  }
  return out;
}

function forward(net, x) {
  const acts = [Array.from(x)];
  let a = acts[0];
  net.layers.forEach((L, li) => {
    const o = L.w.map((row, j) => row.reduce((s, w, i) => s + w * a[i], L.b[j]));
    a = li < net.layers.length - 1 ? o.map((v) => Math.max(0, v)) : o;
    acts.push(a);
  });
  const z = acts[acts.length - 1], m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((p, q) => p + q, 0);
  return { acts, p: e.map((v) => v / s) };
}

// ---------------------------------------------------------------- a tiny network you can poke

export async function tinyNet(H, G) {
  const host = $("tinynet");
  if (!host) return;
  const net = await H.getJson("data/tinynet.json").catch(() => null);
  if (!net || !G) { hideFig(host); return; }
  host.classList.add("chart", "tn");
  const N = net.grid, C = net.classes;
  html("div", "fig-title", host, "A real neural network, small enough to see");
  html("div", "fig-sub", host, `It knows only ${C.length} symbols. Pick one, or paint and erase pixels on the grid, and watch the signal flow through it.`);
  const bar = html("div", "tn-bar", host);
  const pick = html("div", "tn-pick", bar);
  pick.setAttribute("role", "radiogroup");
  pick.setAttribute("aria-label", "Symbol to draw");
  const picks = C.map((ch) => {
    const b = html("button", "", pick);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-label", `${ch} (${H.toRomaji(ch)})`);
    const s = el("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }, b);
    glyphInto(G, s, ch, 17, 12, 12);
    html("span", "", b, ch);
    b.addEventListener("click", () => choose(ch, 0));
    return b;
  });
  const tools = html("div", "tn-tools", bar);
  const wob = html("button", "", tools, "Wobble it");
  wob.type = "button";
  const clr = html("button", "", tools, "Clear");
  clr.type = "button";

  // ---- the drawing: left to right on wide screens, top to bottom on phones
  const read = html("p", "tn-read", host);
  read.setAttribute("aria-live", "polite");
  html("p", "tn-hint", host, "Try this: pick ぐ, then erase its little dot. Then hover over (or tap) a neuron in the first layer to see which pixels it looks for.");
  const nH = [net.layers[0].b.length, net.layers[1].b.length, C.length];
  const spreadAt = (n, a, b) => Array.from({ length: n }, (_, i) => a + (i * (b - a)) / Math.max(1, n - 1));
  const maxW = (L) => Math.max(...L.w.flat().map(Math.abs));
  const K = 4; // the input -> first-layer weights are too many to draw (256 x 12): the strongest few per neuron
  let R = null, narrow = null;

  const build = (nar) => {
    narrow = nar;
    if (R) R.svg.remove();
    let cell, gx0, gy0, P, VB, labs;
    if (!nar) {
      cell = 17; gx0 = 14; gy0 = 46;
      const GW = N * cell, ys = (n) => spreadAt(n, gy0 + 10, gy0 + GW - 10);
      P = [[432, ys(nH[0])], [556, ys(nH[1])], [660, ys(nH[2])]].map(([x, yy]) => yy.map((y) => [x, y]));
      VB = [820, 336];
      labs = [[gx0, 30, `${N * N} pixels in`, "start"], [432, 30, `${nH[0]} neurons`, "middle"], [556, 30, `${nH[1]} neurons`, "middle"], [710, 30, `${C.length} answers`, "middle"]];
    } else {
      cell = 15; gx0 = 80; gy0 = 30;
      const xs = (n, a, b) => spreadAt(n, a, b);
      P = [[xs(nH[0], 22, 378), 346], [xs(nH[1], 50, 350), 424], [xs(nH[2], 45, 355), 502]].map(([xx, y]) => xx.map((x) => [x, y]));
      VB = [400, 584];
      labs = [[gx0, 20, `${N * N} pixels in`, "start"], [6, 322, `${nH[0]} neurons`, "start"], [6, 400, `${nH[1]} neurons`, "start"], [6, 478, `${C.length} answers`, "start"]];
    }
    const GW = N * cell, dx = nar ? 0 : 10, dy = nar ? 10 : 0;
    const svg = el("svg", { viewBox: `0 0 ${VB[0]} ${VB[1]}`, role: "img" });
    const inG = el("g", { class: "tn-edges d0" }, svg);
    const inEdges = P[0].map(() => Array.from({ length: K }, () => el("line", { class: "tn-e pos", "stroke-width": 1 }, inG)));
    // weights between the hidden layers and to the answers: drawn once, their brightness follows the signal
    const edgeLayer = (li, A, B) => {
      const g = el("g", { class: `tn-edges d${li * 2}` }, svg);
      const L = net.layers[li], mw = maxW(L);
      return L.w.map((row, j) => row.map((w, i) => el("line", {
        x1: A[i][0] + dx, y1: A[i][1] + dy, x2: B[j][0] - dx, y2: B[j][1] - dy,
        class: `tn-e ${w >= 0 ? "pos" : "neg"}`, "stroke-width": (0.4 + 2.4 * Math.abs(w) / mw).toFixed(2),
      }, g)));
    };
    const e2 = edgeLayer(1, P[0], P[1]), e3 = edgeLayer(2, P[1], P[2]);
    // the input grid: weights (shown for a neuron in focus) under the ink
    el("rect", { x: gx0 - 1, y: gy0 - 1, width: GW + 2, height: GW + 2, class: "tn-gridbg" }, svg);
    const over = el("g", { class: "tn-over" }, svg);
    const px = [], ov = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const at = { x: gx0 + c * cell + 0.5, y: gy0 + r * cell + 0.5, width: cell - 1, height: cell - 1 };
      ov.push(el("rect", at, over));
      px.push(el("rect", { ...at, class: "tn-px" }, svg));
    }
    const overLab = text(svg, nar ? 6 : gx0, gy0 + GW + (nar ? 20 : 22), "", "t-muted tn-halo");
    const hit = el("rect", { x: gx0, y: gy0, width: GW, height: GW, class: "tn-hit" }, svg);
    labs.forEach(([x, y, s, a]) => text(svg, x, y, s, "t-group tn-halo", a));
    // neurons
    const nodeLayer = (pts, li) => {
      const g = el("g", { class: `tn-nodes d${li * 2 - 1}` }, svg);
      return pts.map(([x, y], i) => {
        const n = el("g", { class: "tn-n" }, g);
        if (li === 1) { n.setAttribute("tabindex", "0"); n.setAttribute("role", "button"); n.setAttribute("aria-label", `First-layer neuron ${i + 1}: show the pixels it looks for`); }
        el("circle", { cx: x, cy: y, r: 9, class: "tn-ring" }, n);
        const f = el("circle", { cx: x, cy: y, r: 9, class: "tn-fill" }, n);
        return { n, f };
      });
    };
    const n1 = nodeLayer(P[0], 1), n2 = nodeLayer(P[1], 2), n3 = nodeLayer(P[2], 3);
    // answers: glyph, kana and a bar (just the number on phones)
    const outs = C.map((ch, i) => {
      const [x, y] = P[2][i], g = el("g", { class: "tn-ans d5" }, svg);
      if (!nar) {
        glyphInto(G, g, ch, 22, x + 30, y, "tn-glyph");
        text(g, x + 50, y + 5, ch, "tn-k");
        el("rect", { x: x + 68, y: y - 5, width: 52, height: 10, rx: 2, class: "tn-track" }, g);
        const b = el("rect", { x: x + 68, y: y - 5, width: 0, height: 10, rx: 2, class: "tn-barfill" }, g);
        return { g, b, t: text(g, VB[0] - 2, y + 4, "", "t-value", "end") };
      }
      glyphInto(G, g, ch, 20, x, y + 30, "tn-glyph");
      text(g, x, y + 60, ch, "tn-k", "middle");
      return { g, b: null, t: text(g, x, y + 76, "", "t-value", "middle") };
    });
    host.insertBefore(svg, read);

    n1.forEach((n, i) => {
      n.n.addEventListener("pointerenter", (e) => { if (e.pointerType === "mouse") { focus = i; showFocus(); } });
      n.n.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") { focus = -1; showFocus(); } });
      n.n.addEventListener("click", () => { focus = focus === i ? -1 : i; showFocus(); });
      n.n.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); focus = focus === i ? -1 : i; showFocus(); } });
      n.n.addEventListener("blur", () => { if (focus === i) { focus = -1; showFocus(); } });
    });
    // painting: a press on an inked pixel erases, on an empty one paints; dragging keeps doing the same
    const cellAt = (e) => {
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const q = pt.matrixTransform(svg.getScreenCTM().inverse());
      const c = Math.floor((q.x - gx0) / cell), r = Math.floor((q.y - gy0) / cell);
      return c >= 0 && r >= 0 && c < N && r < N ? r * N + c : -1;
    };
    hit.addEventListener("pointerdown", (e) => {
      const i = cellAt(e);
      if (i < 0) return;
      e.preventDefault();
      mode = x[i] > 0.5 ? 0 : 1;
      x[i] = mode;
      try { hit.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
      update(false);
    });
    hit.addEventListener("pointermove", (e) => {
      if (mode < 0) return;
      const i = cellAt(e);
      if (i >= 0 && x[i] !== mode) { x[i] = mode; update(false); }
    });
    const end = () => { mode = -1; };
    hit.addEventListener("pointerup", end);
    hit.addEventListener("pointercancel", end);
    R = { svg, P, gx0, gy0, cell, dx, dy, inEdges, e2, e3, px, ov, over, overLab, n1, n2, n3, outs };
  };

  // ---- state
  let x = new Float32Array(N * N), cur = "ぐ", focus = -1, mode = -1;
  const update = (animate) => {
    host.classList.toggle("tn-anim", !!animate && !reduced());
    R.px.forEach((p, i) => { p.style.fillOpacity = x[i].toFixed(2); });
    const { acts, p } = forward(net, x);
    // input -> layer 1: the K pixels that push each neuron up the most
    let m0 = 1e-6;
    const tops = net.layers[0].w.map((row) => {
      const c = row.map((w, i) => [w * x[i], i]).sort((a, b) => b[0] - a[0]).slice(0, K);
      c.forEach(([v]) => { m0 = Math.max(m0, v); });
      return c;
    });
    tops.forEach((c, j) => c.forEach(([v, i], k) => {
      const ln = R.inEdges[j][k], r = Math.floor(i / N), cc = i % N;
      ln.setAttribute("x1", R.gx0 + (cc + 0.5) * R.cell);
      ln.setAttribute("y1", R.gy0 + (r + 0.5) * R.cell);
      ln.setAttribute("x2", R.P[0][j][0] - R.dx);
      ln.setAttribute("y2", R.P[0][j][1] - R.dy);
      ln.style.opacity = v > 0 ? (0.08 + 0.75 * v / m0).toFixed(3) : 0;
    }));
    const glow = (edges, L, a) => {
      let m = 1e-6;
      L.w.forEach((row) => row.forEach((w, i) => { m = Math.max(m, Math.abs(w * a[i])); }));
      edges.forEach((row, j) => row.forEach((ln, i) => { ln.style.opacity = (0.05 + 0.9 * Math.abs(L.w[j][i] * a[i]) / m).toFixed(3); }));
    };
    glow(R.e2, net.layers[1], acts[1]);
    glow(R.e3, net.layers[2], acts[2]);
    const fill = (nodes, a) => { const m = Math.max(1e-6, ...a); nodes.forEach((n, i) => { n.f.style.opacity = (a[i] / m).toFixed(3); }); };
    fill(R.n1, acts[1]);
    fill(R.n2, acts[2]);
    R.n3.forEach((n, i) => { n.f.style.opacity = p[i].toFixed(3); });
    const best = p.indexOf(Math.max(...p));
    R.outs.forEach((o, i) => {
      if (o.b) o.b.setAttribute("width", (52 * p[i]).toFixed(1));
      o.t.textContent = pct(p[i]);
      o.g.classList.toggle("best", i === best);
    });
    const ink = x.reduce((s, v) => s + v, 0);
    read.textContent = ink < 0.5
      ? "The grid is empty: paint something."
      : `It thinks: ${C[best]} (${H.toRomaji(C[best])}), ${pct(p[best])} sure`;
    R.svg.setAttribute("aria-label", `A small neural network. ${read.textContent}`);
    showFocus();
  };
  // what a first-layer neuron looks for: its 256 weights, painted on the grid
  const showFocus = () => {
    R.over.classList.toggle("on", focus >= 0);
    host.classList.toggle("tn-focus", focus >= 0);
    R.n1.forEach((n, i) => n.n.classList.toggle("focus", i === focus));
    if (focus < 0) { R.overLab.textContent = ""; return; }
    const w = net.layers[0].w[focus], m = Math.max(...w.map(Math.abs));
    R.ov.forEach((r, i) => {
      r.setAttribute("class", w[i] >= 0 ? "pos" : "neg");
      r.style.opacity = (0.9 * Math.abs(w[i]) / m).toFixed(3);
    });
    R.overLab.textContent = `Neuron ${focus + 1} looks for: blue = ink wanted, purple = ink unwanted`;
  };

  const choose = (ch, s) => {
    cur = ch;
    x = rasterise(G, net, ch, s);
    picks.forEach((b, i) => b.setAttribute("aria-checked", String(C[i] === ch)));
    update(true);
  };
  wob.addEventListener("click", () => choose(cur, 1 + Math.floor(Math.random() * 1e6)));
  clr.addEventListener("click", () => { x = new Float32Array(N * N); picks.forEach((b) => b.setAttribute("aria-checked", "false")); update(true); });

  const wantNarrow = () => host.clientWidth < 520;
  build(wantNarrow());
  new ResizeObserver(() => { if (wantNarrow() !== narrow) { build(wantNarrow()); update(false); } }).observe(host);
  choose("ぐ", 0);
}

function text(parent, x, y, s, cls, anchor = "start") {
  const t = el("text", { x, y, class: cls, "text-anchor": anchor }, parent);
  t.textContent = s;
  return t;
}

// ---------------------------------------------------------------- inside the real reader

export async function insideReader(H) {
  const host = $("inside");
  if (!host) return;
  const [d, ctc] = await Promise.all([H.getJson("data/nn.json").catch(() => null), H.getJson("data/ctc.json").catch(() => null)]);
  if (!d) { hideFig(host); return; }
  host.classList.add("chart", "in");
  const T = d.timesteps, content = ctc ? ctc.contentWidth : d.input.width;
  const Tc = Math.min(T, Math.ceil((content + 24) / (d.input.width / T))); // steps over the line; the rest is padding
  const pxW = Tc * (d.input.width / T);
  html("div", "fig-title", host, "Inside the real reader");
  html("div", "fig-sub", host, `What the reader computes for ${ctc ? ctc.label : d.text}, stage by stage. Move along the figure to pick a step.`);

  const load = (f) => new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = `img/nn/${f}`; });
  const rows = [
    { name: "Input", sub: `the strip · ${d.input.height} × ${d.input.width} pixels`, files: [d.input.file], h: 54, raw: true },
    ...d.stages.map((s, i) => ({
      name: `Stage ${i + 1}`, sub: `${s.channels} feature maps · ${s.height} × ${s.width} · 3 shown`,
      files: s.maps.slice(0, 3), h: Math.max(11, Math.min(36, Math.round(s.height * 1.5))),
    })),
    { name: "One list per step", sub: `${d.features.channels} numbers for each of the ${T} steps`, files: [d.features.file], h: 64 },
    { name: "Scores", sub: `${d.classes} possible answers · the ${d.rows.length} strongest shown`, scores: true, h: d.rows.length * 15 },
  ];
  const imgs = await Promise.all(rows.map((r) => Promise.all((r.files || []).map(load))));
  if (imgs.some((a) => a.some((im) => !im))) { hideFig(host); return; }

  const grid = html("div", "in-grid", host);
  const cursors = [];
  let spanBox = null, scoreLabs = [];
  const canvases = [];
  rows.forEach((r, ri) => {
    const row = html("div", `in-row${r.scores ? " in-scores" : ""}`, grid);
    const lab = html("div", "in-lab", row);
    html("b", "", lab, r.name);
    html("span", "", lab, r.sub);
    const maps = html("div", "in-maps", row);
    if (r.scores) {
      const labs = html("div", "in-slabs", maps);
      scoreLabs = d.rows.map((s) => {
        const word = s.id === 0 ? "blank" : s.label === " " ? "space" : null;
        return html("span", word ? "word" : "", labs, word || s.label);
      });
      labs.style.setProperty("--rh", `${r.h / d.rows.length}px`);
      const cv = html("canvas", "", maps);
      cv.width = Tc;
      cv.height = d.rows.length;
      cv.style.height = `${r.h}px`;
      canvases.push({ cv, scores: true });
    } else {
      imgs[ri].forEach((im) => {
        const cw = Math.round(im.width * (Tc / T));
        const cv = html("canvas", "", maps);
        cv.width = cw;
        cv.height = im.height;
        cv.style.height = `${r.h}px`;
        canvases.push({ cv, im, raw: r.raw });
      });
    }
    if (r.raw) spanBox = html("div", "in-span", maps);
    cursors.push(html("div", "in-cur", maps));
  });
  const read = html("p", "in-read", host);

  // colour the maps with the page's own palette: background -> accent -> text
  const paint = () => {
    const a = rgbOf(token("--bg")), b = rgbOf(token("--accent")), c = rgbOf(token("--text"));
    const ramp = (t) => {
      const [p, q, u] = t < 0.6 ? [a, b, t / 0.6] : [b, c, (t - 0.6) / 0.4];
      return [0, 1, 2].map((k) => Math.round(p[k] + (q[k] - p[k]) * u));
    };
    const lut = Array.from({ length: 256 }, (_, v) => ramp(v / 255));
    for (const k of canvases) {
      const x = k.cv.getContext("2d", { willReadFrequently: true });
      if (k.scores) {
        const im = x.createImageData(Tc, d.rows.length);
        d.rows.forEach((s, ri) => { for (let t = 0; t < Tc; t++) { const [r, g, bb] = lut[Math.round(Math.min(1, s.p[t]) * 255)]; im.data.set([r, g, bb, 255], (ri * Tc + t) * 4); } });
        x.putImageData(im, 0, 0);
        continue;
      }
      x.drawImage(k.im, 0, 0);
      if (k.raw) continue;
      const im = x.getImageData(0, 0, k.cv.width, k.cv.height);
      for (let i = 0; i < im.data.length; i += 4) { const [r, g, bb] = lut[im.data[i]]; im.data[i] = r; im.data[i + 1] = g; im.data[i + 2] = bb; }
      x.putImageData(im, 0, 0);
    }
  };
  paint();
  window.addEventListener("hxh:theme", () => requestAnimationFrame(paint));
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => requestAnimationFrame(paint));

  const labelOf = (id) => { const r = d.rows.find((s) => s.id === id); return r ? r.label : "?"; };
  let t = 0;
  const show = (nt) => {
    t = Math.max(0, Math.min(Tc - 1, nt));
    const f = (t + 0.5) / Tc;
    cursors.forEach((c) => { c.style.left = `calc(var(--gut) + (100% - var(--gut)) * ${f.toFixed(4)})`; });
    const [s0, s1] = d.spans[t];
    spanBox.style.left = `calc(var(--gut) + (100% - var(--gut)) * ${(s0 / pxW).toFixed(4)})`;
    spanBox.style.width = `calc((100% - var(--gut)) * ${(Math.max(4, s1 - s0) / pxW).toFixed(4)})`;
    const id = d.best[t], row = d.rows.find((s) => s.id === id), p = row ? row.p[t] : 0;
    scoreLabs.forEach((l, i) => l.classList.toggle("on", d.rows[i].id === id));
    read.textContent = `Step ${t + 1} of ${T} · it pays most attention to pixels ${s0}–${s1} of the strip · best answer: ${!id ? "nothing here" : labelOf(id) === " " ? "space" : labelOf(id)} (${pct(p)})`;
  };

  // pointer picks the step; otherwise a slow sweep while the figure is on screen
  let user = 0;
  const pickAt = (e) => {
    const m = grid.querySelector(".in-maps").getBoundingClientRect();
    const gut = parseFloat(getComputedStyle(grid).getPropertyValue("--gut")) || 0;
    const f = (e.clientX - m.left - gut) / (m.width - gut);
    if (f < 0 || f > 1) return;
    user = performance.now();
    show(Math.floor(f * Tc));
  };
  grid.addEventListener("pointermove", pickAt);
  grid.addEventListener("pointerdown", pickAt);
  const vis = visibleWatch(host);
  show(reduced() ? Math.round(Tc * 0.45) : 0);
  if (!reduced()) {
    setInterval(() => {
      if (!vis.on || performance.now() - user < 4000) return;
      show(t >= Tc - 1 ? 0 : t + 1);
    }, 90);
  }
}
