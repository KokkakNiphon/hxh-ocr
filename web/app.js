// HxH alphabet reader: UI. All OCR logic lives in ocr.js (shared with the node tests).
import {
  HxhDetector, HxhEngine, LiveTracker, base64ToBytes, ctcDetails, idsToText, pointInQuad, quadIoU, readAt,
  readLine, readLines, readPage, rgbaToGray, toRomaji, wasmSimdSupported,
} from "./ocr.js";

const $ = (id) => document.getElementById(id);
const els = {
  file: $("file"), examples: $("examples"), work: $("work"), stage: $("stage"), view: $("view"),
  hint: $("hint"), page: $("btn-page"), zoom: $("btn-zoom"), fit: $("opt-fit"),
  invert: $("opt-invert"), summary: $("page-summary"), lines: $("lines"), result: $("result"),
  kana: $("out-kana"), romaji: $("out-romaji"), expected: $("out-expected"), strip: $("strip"),
  stripCaption: $("strip-caption"), padMark: $("pad-mark"), notices: $("out-notices"), timing: $("out-timing"),
  status: $("status"), modelInfo: $("model-info"),
  liveBtn: $("btn-live"), live: $("live"), liveStage: $("live-stage"), liveVideo: $("live-video"),
  liveOverlay: $("live-overlay"), liveNote: $("live-note"), liveStats: $("live-stats"),
  liveStop: $("btn-live-stop"), liveFreeze: $("btn-live-freeze"), liveLines: $("live-lines"),
  pasteBtn: $("btn-paste"), pasteKey: $("paste-key"), loading: $("loading"), stageBusy: $("stage-busy"),
  copyAll: $("btn-copy-all"), copyLine: $("btn-copy-line"), lowToggle: $("low-toggle"), lowCount: $("low-count"),
  optLow: $("opt-low"), dropOverlay: $("drop-overlay"), toast: $("toast"), srStatus: $("sr-status"),
  quality: $("quality"), empty: $("empty"), transcript: $("transcript"), options: $("options"),
};

// Sentences this script shows that are not in index.html. Each one is whole, so a translation can
// replace it as one piece.
const MSG = {
  cantOpen: "This browser could not open that image. Try a JPEG or PNG.",
  notImage: "That file isn't an image. Choose, paste or drop a JPEG, PNG or WebP picture.",
  noClipboardImage: "There's no image on the clipboard. Copy an image first, then try again.",
  clipboardBlocked: "The browser didn't let this page read the clipboard. Paste with the keyboard (Ctrl+V or ⌘V), or choose the image instead.",
  linkNotImage: "That link isn't an image. Drag the picture itself, or save it and choose the file.",
  crossSite: "The site that image comes from doesn't let other pages read it. Save the image to your device, then drop or choose the file.",
  copyFailed: "Couldn't copy. Select the text and copy it yourself.",
};

const MAX_SIDE = 3072; // big camera photos are scaled down once on load
const UNSURE = 0.6; // characters below this softmax probability get underlined

const QUALITY_KEY = "hxh-quality"; // the viewer's model choice for still images, remembered in this browser

const state = {
  // model sets: { name, det: HxhDetector | null, rec: { h: HxhEngine, v?: HxhEngine }, cfg (det.json), vocab, models }
  //   fast: web/models (640 px detector, small readers): the live camera, and still images on Fast
  //   accurate: web/models/accurate (960 px detector, wider readers): still images on Accurate
  sets: { fast: null, accurate: null },
  quality: "accurate", // still images: "fast" or "accurate"
  accurate: null, // promise of the accurate set once its loading has started (resolves null if it failed)
  accurateFailed: false,
  waitingModels: false, // a page read is waiting for the accurate models
  mod: null, // the compiled wasm module
  simd: false, examples: [],
  image: null, // { canvas, w, h, gray, example? }
  lines: [], // lines on show: page lines in reading order, then tapped lines and boxes
  pageLines: [], // every page line read, low-confidence ones included
  showLow: false, // also show page lines that don't read like Hunter script
  pageInfo: null, // timings of the last page read
  selected: -1,
  hover: -1, // the line under the pointer, on the picture or in the text (both light up)
  view: null, // image region shown in the stage
  disp: null, // css <-> image mapping
  drag: null,
  busy: false,
  reading: false, // a whole-page read is running (the stage shows a progress overlay)
  readAgain: false, // a page read was asked for while busy: run it when the current work ends
  loadFailed: false, // the models could not be loaded
  touched: false, // the user picked something (don't auto-load an example over it)
  note: "", // a one-off message under the stage
};

const setStatus = (msg, error = false) => {
  els.status.textContent = msg;
  els.status.classList.toggle("error", error);
};
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const full = () => ({ x: 0, y: 0, w: state.image.w, h: state.image.h });
const same = (a, b) => a && b && a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
// let the page paint once before blocking work; a hidden tab gets no frames, so don't wait for one there
const nextFrame = () => new Promise((r) => {
  if (document.hidden) { setTimeout(r, 0); return; }
  let done = false;
  const go = () => { if (!done) { done = true; setTimeout(r, 0); } };
  requestAnimationFrame(go);
  setTimeout(go, 200); // the tab may be hidden right after this call
});
const kb = (n) => `${Math.round(n / 1000)} KB`;
const rectQuad = (b) => [b.x - 0.5, b.y - 0.5, b.x + b.w - 0.5, b.y - 0.5, b.x + b.w - 0.5, b.y + b.h - 0.5, b.x - 0.5, b.y + b.h - 0.5];
const quadBox = (q) => {
  const xs = [q[0], q[2], q[4], q[6]], ys = [q[1], q[3], q[5], q[7]];
  const x0 = Math.min(...xs), y0 = Math.min(...ys);
  return { x: x0, y: y0, w: Math.max(...xs) - x0, h: Math.max(...ys) - y0 };
};
const quadArea = (q) => Math.abs((q[0] - q[4]) * (q[3] - q[7]) - (q[2] - q[6]) * (q[1] - q[5])) / 2;

// ---------------------------------------------------------------- assets + engines

const getUrl = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r;
};

/**
 * One model set from a folder (models/ or models/accurate/): { h, v?, det? }, each { json, bytes }.
 * The horizontal reader is required; without the vertical reader or the detector the app still
 * reads boxes. A model file ending in .b64.txt is the .hxhm base64-encoded (hosts that only serve
 * text, like the claude.ai Artifact build); anything else is the raw binary.
 */
async function fetchSet(dir) {
  const model = async (name, required) => {
    try {
      const json = await (await getUrl(`${dir}${name}`)).json();
      const r = await getUrl(`${dir}${json.file}`);
      const bytes = /\.b64\.txt$/.test(json.file) ? base64ToBytes((await r.text()).trim()) : new Uint8Array(await r.arrayBuffer());
      return { json, bytes };
    } catch (err) {
      if (required) throw err;
      console.warn(`optional model ${dir}${name} not loaded: ${err.message}`);
      return null;
    }
  };
  const [h, v, det] = await Promise.all([model("model.json", true), model("model_v.json", false), model("det.json", false)]);
  const models = { h };
  if (v) models.v = v;
  if (det) models.det = det;
  return models;
}

async function loadAssets() {
  const simd = wasmSimdSupported();
  const inl = globalThis.HXH_INLINE; // set by the single-file build (dist/hxh-ocr.html)
  if (inl) {
    const unpack = (o) => Object.fromEntries(Object.entries(o).map(([k, m]) => [k, { json: m.json, bytes: base64ToBytes(m.b64) }]));
    return {
      simd, wasm: base64ToBytes(simd ? inl.wasm : inl.wasmNoSimd), models: unpack(inl.models), examples: inl.examples,
      accurate: inl.accurate ? async () => unpack(inl.accurate) : null, // the single-file build may leave it out
    };
  }
  const [wasm, models, examples] = await Promise.all([
    getUrl(simd ? "hxh_infer.wasm" : "hxh_infer_nosimd.wasm").then((r) => r.arrayBuffer()),
    fetchSet("models/"),
    getUrl("examples/examples.json").then((r) => r.json()).catch(() => []),
  ]);
  return {
    simd, wasm: new Uint8Array(wasm), models, accurate: () => fetchSet("models/accurate/"),
    examples: examples.map((e) => ({ ...e, src: `examples/${e.file}` })),
  };
}

/** Engines for one model set. The horizontal reader must load; the others are optional. */
async function makeSet(name, models) {
  const set = { name, det: null, rec: { h: await HxhEngine.create(state.mod, models.h.bytes) }, cfg: {}, vocab: models.h.json.vocab, models };
  for (const [k, create] of [["v", (b) => HxhEngine.create(state.mod, b)], ["det", (b) => HxhDetector.create(state.mod, b)]]) {
    if (!models[k]) continue;
    try {
      const e = await create(models[k].bytes);
      if (k === "v") set.rec.v = e;
      else { set.det = e; set.cfg = models.det.json; }
    } catch (err) {
      console.warn(`${name} ${k} model failed to load: ${err.message}`);
      delete models[k];
    }
  }
  if (set.vocab.length + 1 !== set.rec.h.numClasses) console.warn(`${name}: vocab has ${set.vocab.length} tokens, model ${set.rec.h.numClasses} classes`);
  return set;
}

/** The set still images are read with: the viewer's choice, or Fast while Accurate is missing. */
const stillSet = () => (state.quality === "accurate" && state.sets.accurate) || state.sets.fast;

/** The progress line in the picker while the models load: "download", "start", "failed" or null. */
function setLoading(step) {
  els.loading.hidden = !step;
  els.loading.classList.toggle("error", step === "failed");
  els.loading.querySelectorAll("[data-step]").forEach((s) => (s.hidden = s.dataset.step !== step));
}

async function init() {
  let a;
  setLoading("download");
  try {
    a = await loadAssets();
    setLoading("start");
    await new Promise((r) => setTimeout(r, 0)); // let it paint (not rAF: that waits while the tab is hidden)
    state.mod = await WebAssembly.compile(a.wasm);
    state.sets.fast = await makeSet("fast", a.models);
  } catch (err) {
    console.error(err);
    const hint = location.protocol === "file:" ? " Open this page through a web server, or use the single-file build." : "";
    setStatus(`Could not load the model: ${err.message}.${hint}`, true);
    state.loadFailed = true;
    setLoading("failed");
    renderBusy();
    return;
  }
  setLoading(null);
  const fast = state.sets.fast, { h } = fast.rec;
  for (const [id, v] of [["about-h", h.height], ["about-w", h.width]]) {
    const el = document.getElementById(id);
    if (el) el.textContent = String(v);
  }
  state.simd = a.simd;
  renderModelInfo();
  if (!fast.det) els.hint.textContent = "Drag a box around one line of text to read it.";
  els.liveBtn.disabled = !fast.det;
  if (!fast.det) els.liveBtn.title = "Live mode needs the line detector, which did not load.";
  setStatus("Ready. Everything runs on this device.");
  // the accurate set comes after the fast one, so the live camera and Fast are ready first
  if (a.accurate) {
    state.accurate = a.accurate()
      .then((models) => makeSet("accurate", models))
      .then((set) => {
        if (!set.det) throw new Error("its line detector did not load");
        state.sets.accurate = set;
        renderModelInfo();
        return set;
      })
      .catch((err) => {
        console.warn(`accurate models not loaded: ${err.message}`);
        state.accurateFailed = true;
        renderQuality();
        return null;
      });
  } else state.accurateFailed = true;
  renderQuality();
  renderExamples(a.examples);
  if (state.image) readWholePage();
  else if (!state.touched && a.examples.length) selectExample(0, false);
}

/** The footer line naming the models of each set, and the total size in the lede. */
function renderModelInfo() {
  const describe = (set) => {
    const m = set.models, parts = [];
    if (set.det) parts.push(`line detector ${m.det.json.name} (${kb(m.det.bytes.length)}, ${(set.det.macs / 1e9).toFixed(2)} G MACs at ${set.det.size}×${set.det.size})`);
    parts.push(`horizontal reader ${m.h.json.name} (${kb(m.h.bytes.length)})`);
    if (set.rec.v) parts.push(`vertical reader ${m.v.json.name} (${kb(m.v.bytes.length)})`);
    return `${parts.join(", ")}; readers ${(set.rec.h.macs / 1e6).toFixed(0)} M MACs at ${set.rec.h.height}×${set.rec.h.width}.`;
  };
  const { fast, accurate } = state.sets;
  const sets = [fast, accurate].filter(Boolean);
  const total = sets.reduce((s, set) => s + Object.values(set.models).reduce((t, m) => t + m.bytes.length, 0), 0);
  $("model-size").textContent = kb(total);
  const lines = [`Fast models: ${describe(fast)}`];
  if (accurate) lines.push(`Accurate models: ${describe(accurate)}`);
  lines.push(`All int8, WebAssembly${state.simd ? " SIMD" : " (no SIMD)"}.`);
  els.modelInfo.replaceChildren(...lines.map((t) => Object.assign(document.createElement("span"), { textContent: t })));
}

// ---------------------------------------------------------------- model choice (Fast / Accurate)

try {
  const q = localStorage.getItem(QUALITY_KEY);
  if (q === "fast" || q === "accurate") state.quality = q;
} catch { /* storage may be blocked: keep the default */ }

/** The Fast / Accurate switch: shows the choice, and is disabled when the accurate set can't load. */
function renderQuality() {
  const noAccurate = state.accurateFailed && !state.sets.accurate;
  for (const input of els.quality.querySelectorAll("input")) {
    input.checked = input.value === (noAccurate ? "fast" : state.quality);
    if (input.value === "accurate") input.disabled = noAccurate;
    input.closest("label").classList.toggle("on", input.checked);
  }
  els.quality.title = noAccurate ? "The accurate models could not be loaded, so Fast is used." : "";
}

els.quality.addEventListener("change", (e) => {
  const q = e.target.value;
  if (q !== "fast" && q !== "accurate") return;
  state.quality = q;
  try { localStorage.setItem(QUALITY_KEY, q); } catch { /* not remembered: fine */ }
  renderQuality();
  if (state.image && state.sets.fast) readWholePage();
});

// ---------------------------------------------------------------- examples + file input

function renderExamples(list) {
  state.examples = list;
  els.examples.replaceChildren(...list.map((ex, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = ex.kind === "page" ? "page" : "line";
    b.setAttribute("aria-pressed", "false");
    b.setAttribute("aria-label", ex.kind === "page" ? `Example page ${i + 1}` : `Example line ${i + 1}`);
    const img = document.createElement("img");
    img.src = ex.src;
    img.alt = "";
    img.draggable = false; // a slipped click must not start a drag (and show the drop overlay)
    b.append(img);
    b.addEventListener("click", () => selectExample(i, true));
    return b;
  }));
}

function markExample(i) {
  [...els.examples.children].forEach((b, k) => b.setAttribute("aria-pressed", String(k === i)));
}

async function selectExample(i, byUser) {
  if (byUser) state.touched = true;
  const ex = state.examples[i];
  markExample(i);
  await openImage(ex.src, ex);
}

els.file.addEventListener("change", () => {
  const f = els.file.files && els.file.files[0];
  els.file.value = ""; // picking the same file again should fire again
  if (f) openFile(f);
});

const IMAGE_NAME = /\.(png|jpe?g|jfif|gif|webp|bmp|avif|heic|heif|tiff?)$/i;
const isImageFile = (f) => !!f && (/^image\//i.test(f.type || "") || (!f.type && IMAGE_NAME.test(f.name || "")));

/**
 * Open an image the user gave us: chosen, pasted or dropped. All three come through here and read
 * the whole page, the same way.
 */
async function openFile(file) {
  if (!isImageFile(file)) {
    toast(MSG.notImage);
    return;
  }
  state.touched = true;
  markExample(-1);
  const url = URL.createObjectURL(file);
  try {
    await openImage(url, null, true);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function openImage(src, example, reveal = false) {
  const img = new Image();
  img.decoding = "async";
  img.src = src;
  try {
    await img.decode();
  } catch {
    toast(MSG.cantOpen);
    return;
  }
  stopLive();
  await openDrawable(img, img.naturalWidth, img.naturalHeight, example, reveal);
}

/** One of the three things under the source bar: the drop area, the picture with its text, or the camera. */
function showPane(which) {
  els.empty.hidden = which !== "empty";
  els.work.hidden = which !== "work";
  els.live.hidden = which !== "live";
}

/** Bring the work card into view when it is off screen (a phone, or a paste while scrolled down). */
function revealWork() {
  const top = els.work.getBoundingClientRect().top;
  if (top >= 0 && top <= window.innerHeight * 0.7) return;
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  els.work.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
}

// ---------------------------------------------------------------- paste + drop

let toastTimer = 0;
/** A short message at the bottom of the window (paste and drop happen anywhere on the page). */
function toast(msg) {
  clearTimeout(toastTimer);
  els.toast.textContent = msg;
  els.toast.classList.add("show");
  toastTimer = setTimeout(hideToast, 6000);
}
function hideToast() {
  clearTimeout(toastTimer);
  els.toast.classList.remove("show");
  toastTimer = setTimeout(() => (els.toast.textContent = ""), 250);
}
els.toast.addEventListener("click", hideToast);

/** The first image file a clipboard or drag carries; else its first file (to say it isn't one); else null. */
function fileFrom(dt) {
  if (!dt) return null;
  const files = [];
  for (const it of dt.items || []) {
    if (it.kind !== "file") continue;
    const f = it.getAsFile();
    if (f) files.push(f);
  }
  if (!files.length && dt.files) files.push(...dt.files);
  return files.find(isImageFile) || files[0] || null;
}

const TEXT_INPUT = /^(text|search|url|tel|email|password|number|date|datetime-local|month|week|time)$/i;
const isEditable = (t) => t instanceof Element &&
  (t.isContentEditable || t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && TEXT_INPUT.test(t.type || "text")));

// Ctrl/Cmd+V anywhere on the page. Text (or a paste into a text field) is left to the browser.
document.addEventListener("paste", (e) => {
  if (isEditable(e.target) || isEditable(document.activeElement)) return;
  const file = fileFrom(e.clipboardData);
  if (!file) return;
  e.preventDefault();
  openFile(file);
});

// The Paste button: the async clipboard API, where the browser has it.
const canReadClipboard = !!(window.isSecureContext && navigator.clipboard && typeof navigator.clipboard.read === "function");
els.pasteBtn.hidden = !canReadClipboard;
if (/Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent || "")) els.pasteKey.textContent = "⌘V";
els.pasteBtn.addEventListener("click", async () => {
  let items;
  try {
    items = await navigator.clipboard.read();
  } catch (err) {
    console.warn(`clipboard.read: ${err && err.name}: ${err && err.message}`);
    toast(err && err.name === "DataError" ? MSG.noClipboardImage : MSG.clipboardBlocked);
    return;
  }
  for (const item of items) {
    const type = item.types.find((t) => /^image\//i.test(t));
    if (!type) continue;
    try {
      const blob = await item.getType(type);
      openFile(new File([blob], "clipboard", { type: blob.type || type }));
    } catch (err) {
      console.warn(err);
      toast(MSG.clipboardBlocked);
    }
    return;
  }
  toast(MSG.noClipboardImage);
});

// Drag and drop anywhere on the window. dragenter/dragleave fire for every element the pointer
// crosses, so count them: the overlay goes away when the count is back to zero (the pointer left
// the window) or on drop. Drags that start on this page (text, links) are left alone.
const dnd = { depth: 0, internal: false, lastOver: 0, watch: 0 };
const showDrop = (on) => {
  els.dropOverlay.hidden = !on;
  document.documentElement.classList.toggle("dragging", on);
  clearInterval(dnd.watch);
  if (!on) return;
  // browsers send dragover again and again while a drag is over the page: when they stop, the
  // drag has ended somewhere we were not told about, so don't leave the overlay up
  dnd.lastOver = performance.now();
  dnd.watch = setInterval(() => {
    if (performance.now() - dnd.lastOver < 1500) return;
    dnd.depth = 0;
    showDrop(false);
  }, 500);
};
const dragsImage = (e) => !dnd.internal && !!e.dataTransfer &&
  [...(e.dataTransfer.types || [])].some((t) => t === "Files" || t === "text/uri-list");

document.addEventListener("dragstart", () => (dnd.internal = true));
document.addEventListener("dragend", () => {
  dnd.internal = false;
  dnd.depth = 0;
  showDrop(false);
});
document.addEventListener("dragenter", (e) => {
  if (!dragsImage(e)) return;
  e.preventDefault();
  dnd.depth++;
  showDrop(true);
});
document.addEventListener("dragover", (e) => {
  if (!dragsImage(e)) return;
  e.preventDefault(); // allow the drop (and keep the browser from opening the file itself)
  e.dataTransfer.dropEffect = "copy";
  dnd.lastOver = performance.now();
  if (els.dropOverlay.hidden) {
    dnd.depth = Math.max(dnd.depth, 1);
    showDrop(true);
  }
});
document.addEventListener("dragleave", (e) => {
  if (!dragsImage(e)) return;
  dnd.depth = Math.max(0, dnd.depth - 1);
  if (!dnd.depth) showDrop(false);
});
document.addEventListener("drop", (e) => {
  if (!dragsImage(e)) return;
  e.preventDefault();
  dnd.depth = 0;
  showDrop(false);
  dropped(e.dataTransfer);
});
window.addEventListener("blur", () => {
  if (els.dropOverlay.hidden) return;
  dnd.depth = 0;
  showDrop(false);
});

/** The address of the image in a drag from another page: an <img> in its HTML, else its first link. */
function droppedUrl(dt) {
  const ok = (s, base) => {
    try {
      const u = new URL(s, base);
      return /^(https?|data|blob):$/.test(u.protocol) ? u.href : null;
    } catch {
      return null;
    }
  };
  const list = (dt.getData("text/uri-list") || "").split(/\r?\n/).map((s) => s.trim()).filter((s) => s && !s.startsWith("#"));
  const html = dt.getData("text/html");
  if (html) {
    const img = new DOMParser().parseFromString(html, "text/html").querySelector("img[src]");
    const u = img && ok(img.getAttribute("src"), list[0] || location.href);
    if (u) return u;
  }
  if (list.length) return ok(list[0]);
  const text = (dt.getData("text/plain") || "").trim();
  return /^https?:\/\/\S+$/i.test(text) ? ok(text) : null;
}

async function dropped(dt) {
  const file = fileFrom(dt);
  if (file) {
    openFile(file);
    return;
  }
  // an image dragged from another tab may come as an address only: fetch it (works when its site allows)
  const url = droppedUrl(dt);
  if (!url) {
    toast(MSG.notImage);
    return;
  }
  let blob;
  try {
    const r = await fetch(url, { credentials: "omit" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    blob = await r.blob();
  } catch (err) {
    console.warn(`could not fetch the dropped image (${err.message})`);
    toast(MSG.crossSite);
    return;
  }
  if (!/^image\//i.test(blob.type)) {
    toast(MSG.linkNotImage);
    return;
  }
  openFile(new File([blob], "dropped", { type: blob.type }));
}

/** Show an image (or video frame) in the work card and read the whole page. */
async function openDrawable(img, naturalW, naturalH, example, reveal = false) {
  const k = Math.min(1, MAX_SIDE / Math.max(naturalW, naturalH));
  const w = Math.max(1, Math.round(naturalW * k));
  const h = Math.max(1, Math.round(naturalH * k));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff"; // transparent pixels read as white paper
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, w, h);
  const gray = rgbaToGray(ctx.getImageData(0, 0, w, h).data);
  state.image = { canvas, w, h, gray, example };
  state.view = full();
  state.lines = [];
  state.pageLines = [];
  state.pageInfo = null;
  state.selected = -1;
  state.hover = -1;
  showPane("work");
  layout();
  renderBusy();
  if (reveal) revealWork();
  await readWholePage();
}

// ---------------------------------------------------------------- live camera

const LIVE_SIDE = 960; // frames are processed at this long side (the detector letterboxes to 640)
const LIVE_MAX_LINES = 10; // read at most this many lines per frame, the biggest first

const live = {
  on: false, stream: null, simulated: false, anim: 0, tracker: new LiveTracker(),
  frame: document.createElement("canvas"), W: 0, H: 0, tracks: [],
  fps: 0, last: 0, tDetect: 0, tRead: 0, tapAt: null,
};

els.liveBtn.addEventListener("click", () => startLive());
els.liveStop.addEventListener("click", () => stopLive(true));
els.liveFreeze.addEventListener("click", () => freezeLive());

async function startLive() {
  if (!state.sets.fast?.det || live.on) return;
  state.touched = true;
  markExample(-1);
  let stream = null, why = "";
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
      });
    } catch (err) {
      why = err && err.name === "NotAllowedError"
        ? "The camera isn't allowed here (blocked by the browser or by the page that hosts this demo)."
        : err && err.name === "NotFoundError" ? "No camera was found." : `The camera could not be opened (${err && err.name}).`;
    }
  } else {
    why = "This browser can't open a camera on this page (it needs a secure https:// address).";
  }
  live.simulated = !stream;
  if (!stream) stream = await simulatedCamera();
  if (!stream) {
    setStatus(`${why} Live mode needs a camera.`, true);
    return;
  }
  live.stream = stream;
  live.tracker.reset();
  live.tracks = [];
  live.fps = 0;
  live.last = 0;
  els.liveNote.textContent = live.simulated
    ? `${why} Showing live mode on a moving example page instead, as if it were the camera.`
    : "Point the camera at Hunter × Hunter script. Tap a line (or Freeze) to keep the picture.";
  els.liveNote.classList.toggle("note", live.simulated);
  els.liveVideo.srcObject = stream;
  showPane("live");
  try {
    await els.liveVideo.play();
  } catch (err) {
    console.error(err);
  }
  live.on = true;
  liveLayout();
  els.live.scrollIntoView({ behavior: "smooth", block: "start" });
  liveLoop();
}

/** A stand-in camera: an example page drifting and turning slightly, as a MediaStream. */
async function simulatedCamera() {
  const ex = state.examples.find((e) => e.kind === "page");
  const c = document.createElement("canvas");
  if (!ex || !c.captureStream) return null;
  const img = new Image();
  img.src = ex.src;
  try {
    await img.decode();
  } catch {
    return null;
  }
  c.width = 1280;
  c.height = 720;
  const ctx = c.getContext("2d");
  const t0 = performance.now();
  const tick = () => {
    const t = (performance.now() - t0) / 1000;
    const base = Math.min(c.width / img.naturalWidth, c.height / img.naturalHeight);
    const z = base * (1.25 + 0.2 * Math.sin(t / 3.1));
    ctx.fillStyle = "#3a3a3a";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.save();
    ctx.translate(c.width / 2 + 60 * Math.sin(t / 2.3), c.height / 2 + 40 * Math.sin(t / 1.7));
    ctx.rotate((2.5 * Math.PI / 180) * Math.sin(t / 2.9));
    ctx.scale(z, z);
    ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
    ctx.restore();
    live.anim = requestAnimationFrame(tick);
  };
  tick();
  return c.captureStream(30);
}

function stopLive(backToPicker = false) {
  if (!live.on && !live.stream) return;
  live.on = false;
  cancelAnimationFrame(live.anim);
  if (live.stream) live.stream.getTracks().forEach((t) => t.stop());
  live.stream = null;
  els.liveVideo.srcObject = null;
  showPane(state.image ? "work" : "empty");
  if (backToPicker && state.image) layout();
}

/** Keep the current camera frame as a still image: the page view, with every line's details. */
async function freezeLive(tap = null) {
  const v = els.liveVideo;
  if (!live.on || !v.videoWidth) return;
  const W = v.videoWidth, H = v.videoHeight;
  const still = document.createElement("canvas");
  still.width = W;
  still.height = H;
  still.getContext("2d").drawImage(v, 0, 0, W, H);
  stopLive();
  await openDrawable(still, W, H, null);
  if (tap && state.image) {
    const x = tap.x * state.image.w - 0.5, y = tap.y * state.image.h - 0.5; // tap as a fraction of the frame
    const hits = state.lines.map((l, i) => [i, l]).filter(([, l]) => pointInQuad(l.quad, x, y));
    if (hits.length) select(hits[0][0], true);
    else await tapAt({ x, y });
  }
}

function liveLayout() {
  const v = els.liveVideo;
  const ar = v.videoWidth && v.videoHeight ? v.videoWidth / v.videoHeight : 16 / 9;
  const maxW = Math.max(1, Math.floor(els.liveStage.clientWidth));
  const maxH = wideLayout() ? Math.max(320, window.innerHeight - navHeight() - 40) : Math.max(200, Math.round(window.innerHeight * 0.6));
  const cssW = Math.min(maxW, Math.round(maxH * ar)), cssH = Math.round(cssW / ar);
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  for (const el of [v, els.liveOverlay]) {
    el.style.width = `${cssW}px`;
    el.style.height = `${cssH}px`;
  }
  els.liveOverlay.width = Math.round(cssW * dpr);
  els.liveOverlay.height = Math.round(cssH * dpr);
  live.css = { w: cssW, h: cssH, dpr };
  drawLive();
}
new ResizeObserver(() => live.on && liveLayout()).observe(els.liveStage);
els.liveVideo.addEventListener("loadedmetadata", () => live.on && liveLayout());

async function liveLoop() {
  const v = els.liveVideo;
  const ctx = live.frame.getContext("2d", { willReadFrequently: true });
  while (live.on) {
    await nextFrame();
    if (!live.on) break;
    if (document.hidden || v.readyState < 2 || !v.videoWidth) continue;
    const t0 = performance.now();
    const k = Math.min(1, LIVE_SIDE / Math.max(v.videoWidth, v.videoHeight));
    const W = Math.round(v.videoWidth * k), H = Math.round(v.videoHeight * k);
    if (live.frame.width !== W || live.frame.height !== H) {
      live.frame.width = W;
      live.frame.height = H;
      liveLayout();
    }
    ctx.drawImage(v, 0, 0, W, H);
    const gray = rgbaToGray(ctx.getImageData(0, 0, W, H).data);
    let lines = [], reuse = new Map();
    try {
      const S = state.sets.fast; // the live camera always uses the fast models
      const d = S.det.detect(gray, W, H, readOpts(S).params);
      const t1 = performance.now();
      // the biggest lines first, read in reading order
      const keep = new Set(d.lines.map((l, i) => [l.length * l.thickness, i]).sort((a, b) => b[0] - a[0])
        .slice(0, LIVE_MAX_LINES).map(([, i]) => i));
      const found = d.lines.filter((_, i) => keep.has(i));
      // a line whose track has read the same text 3 times running keeps it for up to 2 frames
      // instead of being read again (reading costs more than detecting on a busy page)
      for (const [i, l] of found.entries()) {
        let best = null, bestV = 0.5;
        for (const t of live.tracks) {
          const v = quadIoU(l.quad, t.quad);
          if (v > bestV) { best = t; bestV = v; }
        }
        const tail = best ? best.texts.slice(-3) : [];
        if (best && tail.length === 3 && tail.every((x) => x === tail[0]) && (best.skips || 0) < 2) {
          best.skips = (best.skips || 0) + 1;
          reuse.set(i, best);
        } else if (best) best.skips = 0;
      }
      const toRead = found.filter((_, i) => !reuse.has(i));
      const read = readLines(S.det, S.rec, toRead, readOpts(S));
      let k = 0;
      lines = found.map((l, i) => (reuse.has(i) ? { ...reuse.get(i).line, ...l, reusedText: reuse.get(i).text } : read[k++]));
      live.tDetect = t1 - t0;
      live.tRead = performance.now() - t1;
    } catch (err) {
      console.error(err);
      els.liveNote.textContent = `Reading failed: ${err.message}`;
      continue;
    }
    live.W = W;
    live.H = H;
    live.tracks = live.tracker.update(lines, (l) => l.reusedText ?? idsToText(l.ids, state.sets.fast.vocab).trim());
    live.reused = reuse.size;
    const now = performance.now();
    if (live.last) live.fps = live.fps ? 0.8 * live.fps + 0.2 * (1000 / (now - live.last)) : 1000 / (now - live.last);
    live.last = now;
    drawLive();
    renderLiveLines();
  }
}

/** Tracks worth showing: seen at least twice, still current, with some text. */
const shownTracks = () => live.tracks.filter((t) => t.hits >= 2 && t.missed === 0 && t.text && (t.line.conf ?? 1) >= minConf(state.sets.fast));

function drawLive() {
  const c = live.css;
  if (!c) return;
  const ctx = els.liveOverlay.getContext("2d");
  ctx.setTransform(c.dpr, 0, 0, c.dpr, 0, 0);
  ctx.clearRect(0, 0, c.w, c.h);
  if (!live.W) return;
  const sx = c.w / live.W, sy = c.h / live.H;
  const P = (x, y) => [(x + 0.5) * sx, (y + 0.5) * sy];
  const colH = cssVar("--accent") || "#2340d8";
  const colV = cssVar("--vert") || "#6a4bc4";
  for (const t of shownTracks()) {
    const col = t.vertical ? colV : colH;
    const pts = [0, 1, 2, 3].map((j) => P(t.quad[2 * j], t.quad[2 * j + 1]));
    ctx.beginPath();
    pts.forEach(([x, y], j) => (j ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = col;
    ctx.stroke();
    // label: kana over romaji, above the line's box (below it when there is no room)
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const left = Math.min(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
    const kana = t.text, rom = toRomaji(t.text);
    ctx.font = `600 17px ${cssVar("--kana-font") || "sans-serif"}`;
    const wk = ctx.measureText(kana).width;
    ctx.font = "12px system-ui, sans-serif";
    const wr = ctx.measureText(rom).width;
    const bw = Math.max(wk, wr) + 12, bh = 38;
    let bx = Math.max(2, Math.min(c.w - bw - 2, left));
    let by = top - bh - 4;
    if (by < 2) by = Math.min(c.h - bh - 2, bottom + 4);
    ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(bx, by, bw, bh, 6) : ctx.rect(bx, by, bw, bh);
    ctx.fill();
    ctx.fillStyle = col;
    ctx.fillRect(bx, by, 3, bh);
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.font = `600 17px ${cssVar("--kana-font") || "sans-serif"}`;
    ctx.fillText(kana, bx + 8, by + 3);
    ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText(rom, bx + 8, by + 22);
  }
  els.liveStats.textContent = live.fps
    ? `${live.fps.toFixed(1)} frames/s · detect ${live.tDetect.toFixed(0)} ms · read ${live.tRead.toFixed(0)} ms` +
      (live.reused ? ` (${live.reused} kept)` : "")
    : "Starting…";
}

function renderLiveLines() {
  const tracks = shownTracks();
  // reading order of the latest frame: the order the detector gave the lines
  tracks.sort((a, b) => (a.line.order ?? 0) - (b.line.order ?? 0));
  const key = tracks.map((t) => `${t.id}:${t.text}`).join("|");
  if (key === live.listKey) return;
  live.listKey = key;
  els.liveLines.replaceChildren(...tracks.map((t, i) => {
    const li = document.createElement("li");
    const row = document.createElement("div");
    row.className = "line" + (t.vertical ? " vertical" : "");
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = String(i + 1);
    const body = document.createElement("span");
    body.className = "body";
    const k = document.createElement("span");
    k.className = "lk";
    k.setAttribute("data-no-i18n", ""); // a reading: never run it through the page translator
    k.lang = "ja";
    k.textContent = t.text;
    const r = document.createElement("span");
    r.className = "lr";
    r.setAttribute("data-no-i18n", "");
    r.textContent = toRomaji(t.text);
    body.append(k, r);
    row.append(num, body);
    li.append(row);
    return li;
  }));
}

els.liveOverlay.addEventListener("click", (e) => {
  if (!live.on || !live.css) return;
  const rect = els.liveOverlay.getBoundingClientRect();
  freezeLive({ x: (e.clientX - rect.left) / rect.width, y: (e.clientY - rect.top) / rect.height });
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && live.on && !live.simulated) els.liveNote.textContent = "Paused while the page is hidden.";
  else if (live.on && !live.simulated) els.liveNote.textContent = "Point the camera at Hunter × Hunter script. Tap a line (or Freeze) to keep the picture.";
});


// ---------------------------------------------------------------- reading

const readOpts = (S = stillSet()) => ({ params: S.cfg.params || {}, pad: S.cfg.pad || {}, invert: els.invert.checked });
// Page lines whose reading is less sure than this are other writing (Latin, Japanese, art)
// that the detector picked up: hidden unless the viewer asks to see them.
const minConf = (S = stillSet()) => S.cfg.min_conf ?? 0.85;
const minShowScore = (S = stillSet()) => S.cfg.min_show_score ?? 0;
const setOf = (l) => state.sets[l.set] || stillSet();
const isLow = (l) => l.source === "page" && (l.conf < minConf(setOf(l)) || (l.score ?? 1) < minShowScore(setOf(l)));

/** Rebuild the lines on show from the page lines and the extra (tapped / boxed) ones. */
function applyLineFilter() {
  const extra = state.lines.filter((l) => l.source !== "page");
  const page = state.showLow ? state.pageLines : state.pageLines.filter((l) => !isLow(l));
  const keep = state.lines[state.selected];
  state.lines = [...page, ...extra];
  const i = keep ? state.lines.indexOf(keep) : -1;
  state.selected = i >= 0 ? i : state.lines.length ? 0 : -1;
}

/** A read line ready to show: where it came from and which model set read it. */
function decorate(l, source, S) {
  return { ...l, source, set: S.name, text: idsToText(l.ids, S.vocab) };
}

/** Read the whole image: every line the detector finds, in reading order. */
async function readWholePage() {
  const { image } = state;
  if (!state.sets.fast || !image) return;
  if (state.busy) {
    // a page read still waiting to start picks up the current image and settings by itself;
    // a tap being read has to be followed by a page read
    if (!state.reading) state.readAgain = true;
    return;
  }
  state.readAgain = false;
  state.lines = [];
  state.pageLines = [];
  state.selected = -1;
  state.pageInfo = null;
  state.note = "";
  let S = stillSet();
  if (S.det) {
    state.busy = true;
    state.reading = true;
    els.summary.textContent = "Finding the lines…";
    renderLines();
    renderSelected();
    // Accurate chosen but still downloading: wait for it (Fast is used if it fails)
    if (state.quality === "accurate" && !state.sets.accurate && state.accurate) {
      state.waitingModels = true;
      renderBusy();
      await state.accurate;
      state.waitingModels = false;
      S = stillSet();
    }
    renderBusy();
    draw();
    await nextFrame(); // let the progress overlay paint: the read below blocks the page
    try {
      if (state.image === image) {
        // Fast also looks at zoomed-in tiles, so small text in a big photo is found; the accurate
        // detector sees the page at 960 px and finds it without them (tiles there cost ~2.5x and
        // added false lines: docs/research/JOURNAL.md, 2026-11-25 model family)
        const tiles = S.name === "fast";
        const r = readPage(S.det, S.rec, image.gray, image.w, image.h, { ...readOpts(S), tiles, minConf: minConf(S) });
        // a box the reader finds no glyph in is almost always a false detection: leave it out
        state.pageLines = r.lines.map((l) => decorate(l, "page", S)).filter((l) => l.text.trim());
        state.pageInfo = { tDetect: r.tLetterbox + r.tDetect + r.tPost, tRead: r.tRead, nRead: r.lines.length };
        applyLineFilter();
      }
    } catch (err) {
      console.error(err);
      setStatus(`Reading failed: ${err.message}`, true);
    } finally {
      state.busy = false;
      state.reading = false;
      renderBusy();
    }
    if (state.image !== image) { // another image came in while this read waited to start
      readWholePage();
      return;
    }
  }
  // nothing found in a short, wide image (e.g. a cropped line): read the whole image as one line
  if (!state.pageLines.length && (!S.det || image.w >= 1.5 * image.h)) readBox(full(), false);
  if (state.lines.length && state.selected < 0) state.selected = 0;
  renderAll();
}

/** Read a box as one horizontal line (the manual fallback). */
function readBox(box, select = true) {
  const { image } = state, S = stillSet(), { rec } = S;
  let r;
  try {
    r = readLine(rec.h, image.gray, image.w, image.h, box, { fit: els.fit.checked, invert: els.invert.checked });
  } catch (err) {
    console.error(err);
    setStatus(`Reading failed: ${err.message}`, true);
    return;
  }
  const line = decorate({ ...r, quad: rectQuad(r.box), vertical: false, engine: rec.h, drawn: box }, "box", S);
  state.lines.push(line);
  if (select) state.selected = state.lines.length - 1;
}

/** A tap: select the line under it, or look for one around it at a higher resolution. */
async function tapAt(p) {
  const hits = state.lines.map((l, i) => [i, l]).filter(([, l]) => pointInQuad(l.quad, p.x, p.y));
  if (hits.length) {
    hits.sort((a, b) => quadArea(a[1].quad) - quadArea(b[1].quad));
    select(hits[0][0]);
    return;
  }
  const { image } = state, S = stillSet();
  if (!S.det || state.busy) return;
  state.busy = true;
  state.note = "Looking for a line here…";
  renderNote();
  await nextFrame();
  let r = null;
  try {
    if (state.image === image) r = readAt(S.det, S.rec, image.gray, image.w, image.h, p.x, p.y, readOpts(S));
  } catch (err) {
    console.error(err);
    setStatus(`Reading failed: ${err.message}`, true);
  } finally {
    state.busy = false;
  }
  if (state.readAgain || state.image !== image) { // a new image (or a page read) was asked for meanwhile
    state.note = "";
    readWholePage();
    return;
  }
  if (!r) {
    state.note = "No line found at that spot. Drag a box around the text to read it anyway.";
    renderNote();
    return;
  }
  state.note = "";
  const q = r.quad;
  const cx = (q[0] + q[2] + q[4] + q[6]) / 4, cy = (q[1] + q[3] + q[5] + q[7]) / 4;
  const dup = state.lines.findIndex((l) => pointInQuad(l.quad, cx, cy));
  if (dup >= 0) state.lines[dup] = decorate(r, state.lines[dup].source === "page" ? "page" : "tap", S);
  else state.lines.push(decorate(r, "tap", S));
  state.selected = dup >= 0 ? dup : state.lines.length - 1;
  renderAll(true);
}

function select(i, scroll = false) {
  state.selected = i;
  draw();
  renderLines();
  renderSelected(scroll);
}

/** The overlay on the stage: the models are still loading, or the page is being read. */
function renderBusy() {
  const kind = state.image && !state.sets.fast && !state.loadFailed ? "models"
    : state.waitingModels ? "accurate" : state.reading ? "page" : null;
  els.stageBusy.hidden = !kind;
  els.stageBusy.querySelectorAll("[data-busy]").forEach((s) => (s.hidden = s.dataset.busy !== kind));
  els.stage.setAttribute("aria-busy", String(!!kind));
}

// ---------------------------------------------------------------- stage: layout + drawing

function layout() {
  if (!state.image) return;
  const v = state.view;
  const cssW = Math.max(1, Math.floor(els.stage.clientWidth));
  // wide screens: the picture column is sticky and may use the window's height; narrow ones keep
  // the text in view under the (sticky) picture
  const maxH = wideLayout() ? Math.max(320, window.innerHeight - navHeight() - 40)
    : Math.max(200, Math.min(520, Math.round(window.innerHeight * 0.42)));
  const pad = 12;
  let k = (cssW - 2 * pad) / v.w;
  if (v.h * k > maxH - 2 * pad) k = (maxH - 2 * pad) / v.h;
  const cssH = Math.round(Math.min(maxH, Math.max(140, v.h * k + 2 * pad)));
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const c = els.view;
  c.style.width = `${cssW}px`;
  c.style.height = `${cssH}px`;
  c.width = Math.round(cssW * dpr);
  c.height = Math.round(cssH * dpr);
  state.disp = {
    k, dpr, cssW, cssH,
    ox: (cssW - v.w * k) / 2 - v.x * k,
    oy: (cssH - v.h * k) / 2 - v.y * k,
  };
  const zoomed = !same(state.view, full());
  els.zoom.textContent = zoomed ? "Show all" : "Zoom to line";
  els.zoom.disabled = !zoomed && state.selected < 0;
  document.documentElement.style.setProperty("--stage-h", `${cssH + 16}px`);
  draw();
}

const wideLayout = () => matchMedia("(min-width: 960px)").matches;
const navHeight = () => document.querySelector(".site-nav")?.offsetHeight || 56;
window.addEventListener("resize", () => { layout(); if (live.on) liveLayout(); });

const toCss = (x, y) => [state.disp.ox + (x + 0.5) * state.disp.k, state.disp.oy + (y + 0.5) * state.disp.k];

let drawQueued = false;
function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => {
    drawQueued = false;
    draw();
  });
}

function draw() {
  const im = state.image, d = state.disp;
  if (!im || !d) return;
  const ctx = els.view.getContext("2d");
  ctx.setTransform(d.dpr, 0, 0, d.dpr, 0, 0);
  ctx.clearRect(0, 0, d.cssW, d.cssH);
  ctx.imageSmoothingEnabled = d.k < 3; // show pixels when zoomed far in
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(im.canvas, d.ox, d.oy, im.w * d.k, im.h * d.k);

  const colH = cssVar("--accent") || "#2340d8";
  const colV = cssVar("--vert") || "#6a4bc4";
  const drag = state.drag && state.drag.moved ? state.drag.box : null;
  if (drag) { // dim everything outside the box being drawn
    const [x0, y0] = toCss(drag.x - 0.5, drag.y - 0.5), [x1, y1] = toCss(drag.x + drag.w - 0.5, drag.y + drag.h - 0.5);
    ctx.fillStyle = "rgba(0, 0, 0, 0.42)";
    ctx.beginPath();
    ctx.rect(0, 0, d.cssW, d.cssH);
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.fill("evenodd");
    ctx.lineWidth = 2;
    ctx.strokeStyle = colH;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    return;
  }
  // unselected lines first, then the one under the pointer, the selected one on top
  const hov = state.hover !== state.selected && state.lines[state.hover] ? state.hover : -1;
  const order = state.lines.map((_, i) => i).filter((i) => i !== state.selected && i !== hov);
  if (hov >= 0) order.push(hov);
  if (state.selected >= 0) order.push(state.selected);
  for (const i of order) {
    const l = state.lines[i];
    const sel = i === state.selected, hot = i === hov;
    const col = isLow(l) ? cssVar("--muted") || "#888" : l.vertical ? colV : colH;
    const pts = [0, 1, 2, 3].map((j) => toCss(l.quad[2 * j], l.quad[2 * j + 1]));
    ctx.beginPath();
    pts.forEach(([x, y], j) => (j ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    if (sel || hot) {
      ctx.save();
      ctx.globalAlpha = sel ? 0.18 : 0.1;
      ctx.fillStyle = col;
      ctx.fill();
      ctx.restore();
    }
    ctx.lineWidth = sel ? 3 : hot ? 2.5 : 1.5;
    ctx.strokeStyle = col;
    ctx.setLineDash(l.source === "box" || isLow(l) ? [6, 4] : []);
    ctx.stroke();
    ctx.setLineDash([]);
    // reading-order badge at the line's start (q0)
    const r = sel ? 10 : 8;
    const [bx, by] = pts[0];
    ctx.beginPath();
    ctx.arc(bx, by, r, 0, 2 * Math.PI);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = `600 ${sel ? 12 : 10}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(i + 1), bx, by + 0.5);
  }
}

new ResizeObserver(() => layout()).observe(els.stage);
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => draw());
window.addEventListener("hxh:theme", () => { draw(); if (typeof drawLive === "function") drawLive(); });

// ---------------------------------------------------------------- stage: tap and drag

function pointerImage(e) {
  const rect = els.view.getBoundingClientRect();
  const { k, ox, oy } = state.disp;
  const x = (e.clientX - rect.left - ox) / k - 0.5;
  const y = (e.clientY - rect.top - oy) / k - 0.5;
  const im = state.image;
  return { x: Math.min(im.w - 0.5, Math.max(-0.5, x)), y: Math.min(im.h - 0.5, Math.max(-0.5, y)) };
}

function boxFrom(a, b) {
  const im = state.image;
  const x0 = Math.max(0, Math.floor(Math.min(a.x, b.x) + 0.5)), x1 = Math.min(im.w, Math.ceil(Math.max(a.x, b.x) + 0.5));
  const y0 = Math.max(0, Math.floor(Math.min(a.y, b.y) + 0.5)), y1 = Math.min(im.h, Math.ceil(Math.max(a.y, b.y) + 0.5));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

els.view.addEventListener("pointerdown", (e) => {
  if (!state.image || !state.disp || (e.pointerType === "mouse" && e.button !== 0)) return;
  if (state.drag) { // a second finger: give the gesture back to the browser (pinch zoom)
    state.drag = null;
    draw();
    return;
  }
  state.drag = { id: e.pointerId, anchor: pointerImage(e), x: e.clientX, y: e.clientY, moved: false, box: null };
  try {
    els.view.setPointerCapture(e.pointerId);
  } catch { /* synthetic events have no active pointer */ }
});

/** The line under a point of the picture: the smallest box that holds it, or -1. */
function lineAt(p) {
  let best = -1, area = Infinity;
  state.lines.forEach((l, i) => {
    if (!pointInQuad(l.quad, p.x, p.y)) return;
    const a = quadArea(l.quad);
    if (a < area) { best = i; area = a; }
  });
  return best;
}

/** Light up one line on the picture and in the text at once (-1: none). */
function setHover(i) {
  if (state.hover === i) return;
  state.hover = i;
  els.lines.querySelectorAll("button.line").forEach((b, k) => b.classList.toggle("hover", k === i));
  els.transcript.querySelectorAll(".tl").forEach((t) => t.classList.toggle("hover", Number(t.dataset.i) === i));
  requestDraw();
}

els.view.addEventListener("pointermove", (e) => {
  const d = state.drag;
  if (!d && state.disp && e.pointerType === "mouse") { // hovering, not dragging: point at the line under the mouse
    const i = lineAt(pointerImage(e));
    setHover(i);
    els.view.style.cursor = i >= 0 ? "pointer" : "crosshair";
    return;
  }
  if (!d || e.pointerId !== d.id) return;
  if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 8) return;
  d.moved = true;
  d.box = boxFrom(d.anchor, pointerImage(e));
  requestDraw();
});

els.view.addEventListener("pointerup", (e) => {
  const d = state.drag;
  if (!d || e.pointerId !== d.id) return;
  state.drag = null;
  if (!d.moved) {
    draw();
    tapAt(d.anchor);
    return;
  }
  const b = d.box;
  if (!b || b.w < 4 || b.h < 4 || b.w * state.disp.k < 10 || b.h * state.disp.k < 8) {
    draw(); // a tiny box: ignore it
    return;
  }
  state.note = "";
  readBox(b);
  renderAll(true);
});

els.view.addEventListener("pointercancel", () => {
  state.drag = null;
  draw();
});
els.view.addEventListener("pointerleave", () => setHover(-1));

// ---------------------------------------------------------------- tools

els.page.addEventListener("click", () => {
  els.options.open = false;
  readWholePage();
});

// the options menu closes on a click elsewhere or on Escape, like any menu
document.addEventListener("click", (e) => {
  if (els.options.open && !els.options.contains(e.target)) els.options.open = false;
});
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || !els.options.open) return;
  els.options.open = false;
  els.options.querySelector("summary").focus();
});

els.zoom.addEventListener("click", () => {
  if (!state.image) return;
  const im = state.image;
  if (!same(state.view, full()) || state.selected < 0) {
    state.view = full();
  } else {
    const s = quadBox(state.lines[state.selected].quad);
    const m = Math.max(8, Math.min(s.w, s.h) * 0.6);
    let x0 = s.x - m, x1 = s.x + s.w + m, y0 = s.y - m, y1 = s.y + s.h + m;
    const minH = (x1 - x0) / 3, minW = (y1 - y0) / 3; // keep a usable shape for long lines
    if (y1 - y0 < minH) { const c = (y0 + y1) / 2; y0 = c - minH / 2; y1 = c + minH / 2; }
    if (x1 - x0 < minW) { const c = (x0 + x1) / 2; x0 = c - minW / 2; x1 = c + minW / 2; }
    const fit = (a0, a1, n) => {
      if (a1 - a0 >= n) return [0, n];
      if (a0 < 0) return [0, a1 - a0];
      if (a1 > n) return [n - (a1 - a0), n];
      return [a0, a1];
    };
    [x0, x1] = fit(x0, x1, im.w);
    [y0, y1] = fit(y0, y1, im.h);
    state.view = { x: Math.round(x0), y: Math.round(y0), w: Math.round(x1 - x0), h: Math.round(y1 - y0) };
  }
  layout();
});

els.invert.addEventListener("change", () => readWholePage());

els.optLow.addEventListener("change", () => {
  state.showLow = els.optLow.checked;
  applyLineFilter();
  renderAll();
});

// ---------------------------------------------------------------- copy

/** Copy text; the old execCommand route covers pages where the async clipboard is not allowed. */
async function writeClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch { /* not allowed here (e.g. inside a frame): try the old way */ }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.cssText = "position:fixed;top:0;left:0;width:1px;height:1px;opacity:0;";
  const back = document.activeElement;
  document.body.append(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch { /* nothing left to try */ }
  ta.remove();
  if (back && back.focus) back.focus({ preventScroll: true });
  return ok;
}

async function copyFrom(btn, text) {
  if (!text) return;
  if (!(await writeClipboard(text))) {
    toast(MSG.copyFailed);
    return;
  }
  btn.classList.add("done");
  els.srStatus.textContent = btn.querySelector(".copy-done").textContent; // "Copied", in the page's language
  clearTimeout(btn.doneTimer);
  btn.doneTimer = setTimeout(() => {
    btn.classList.remove("done");
    els.srStatus.textContent = "";
  }, 1800);
}

// one line: its kana, then its romaji
els.copyLine.addEventListener("click", () => {
  const l = state.lines[state.selected];
  const t = l && l.text ? l.text.trim() : "";
  copyFrom(els.copyLine, t && `${t}\n${toRomaji(t)}`);
});

// every line on show, in reading order: the kana lines, a blank line, then the romaji lines
els.copyAll.addEventListener("click", () => {
  const texts = state.lines.map((l) => (l.text || "").trim()).filter(Boolean);
  copyFrom(els.copyAll, texts.length ? `${texts.join("\n")}\n\n${texts.map(toRomaji).join("\n")}` : "");
});

// ---------------------------------------------------------------- results

function renderAll(scroll = false) {
  layout();
  renderLines();
  renderSelected(scroll);
  renderNote();
}

function renderNote() {
  els.hint.classList.toggle("note", !!state.note);
  els.hint.textContent = state.note || (state.sets.fast?.det
    ? "Tap a numbered line to see it. Tap unmarked text to look closer, or drag a box around one line to read just that."
    : "Drag a box around one line of text to read it.");
}

function renderLines() {
  const n = state.lines.length;
  const pi = state.pageInfo;
  if (state.busy && !n) { // a page is being read: placeholder rows where its lines will be
    els.lines.replaceChildren(...[58, 34, 72, 46].map((w) => {
      const li = document.createElement("li");
      li.className = "skeleton";
      li.setAttribute("aria-hidden", "true");
      const a = document.createElement("span"), b = document.createElement("span");
      a.className = "sk sk-num";
      b.className = "sk sk-text";
      b.style.setProperty("--w", `${w}%`);
      li.append(a, b);
      return li;
    }));
    els.transcript.hidden = true;
    els.lowToggle.hidden = true;
    els.copyAll.hidden = true;
    return;
  }
  renderTranscript();
  const pageN = state.pageLines.filter((l) => !isLow(l)).length;
  const lowN = state.pageLines.length - pageN;
  if (pi) {
    els.summary.textContent = pageN
      ? `${pageN} line${pageN === 1 ? "" : "s"} found in ${pi.tDetect.toFixed(0)} ms, read in ${pi.tRead.toFixed(0)} ms ` +
        `(${(pi.tRead / pi.nRead).toFixed(1)} ms per line).`
      : `No Hunter script found (${pi.tDetect.toFixed(0)} ms).${n ? "" : " Tap the text to look closer, or drag a box around one line."}`;
  } else {
    els.summary.textContent = n ? "" : "Drag a box around one line of text.";
  }
  // page lines that don't read like Hunter script: hidden unless this box is ticked
  els.lowToggle.hidden = !(pi && lowN);
  els.lowCount.textContent = String(lowN);
  els.optLow.checked = state.showLow;
  els.copyAll.hidden = !state.lines.some((l) => l.text && l.text.trim());
  els.lines.replaceChildren(...state.lines.map((l, i) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.className = "line" + (l.vertical ? " vertical" : "");
    b.setAttribute("aria-pressed", String(i === state.selected));
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = String(i + 1);
    const body = document.createElement("span");
    body.className = "body";
    const k = document.createElement("span");
    k.className = "lk";
    k.setAttribute("data-no-i18n", ""); // a reading: never run it through the page translator
    k.lang = "ja";
    k.textContent = l.text || "…";
    const r = document.createElement("span");
    r.className = "lr";
    r.setAttribute("data-no-i18n", "");
    r.textContent = l.text ? toRomaji(l.text) : "(nothing recognised)";
    body.append(k, r);
    b.append(num, body);
    const tags = [isLow(l) ? "not Hunter script?" : "", l.vertical ? "vertical" : "",
      l.source === "tap" ? "tapped" : l.source === "box" ? "box" : ""].filter(Boolean);
    if (isLow(l)) b.classList.add("low");
    if (tags.length) {
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = tags.join(" · ");
      b.append(t);
    }
    if (i === state.hover) b.classList.add("hover");
    b.addEventListener("click", () => select(i, true));
    // pointing at a line here lights it up on the picture too
    b.addEventListener("pointerenter", () => setHover(i));
    b.addEventListener("pointerleave", () => state.hover === i && setHover(-1));
    b.addEventListener("focus", () => setHover(i));
    b.addEventListener("blur", () => state.hover === i && setHover(-1));
    li.append(b);
    if (i === state.selected) {
      li.classList.add("open");
      li.append(els.result); // the selected line opens in place, under its row
    }
    return li;
  }));
}

/** The whole reading as plain text, one line each, in reading order; a click on a line selects it. */
function renderTranscript() {
  const rows = state.lines.map((l, i) => [i, (l.text || "").trim()]).filter(([, t]) => t);
  els.transcript.hidden = !rows.length;
  els.transcript.replaceChildren(...rows.map(([i, t]) => {
    const s = document.createElement("span");
    s.className = "tl" + (i === state.selected ? " on" : "") + (i === state.hover ? " hover" : "");
    s.dataset.i = String(i);
    s.textContent = t;
    s.addEventListener("click", () => {
      if (!window.getSelection()?.isCollapsed) return; // selecting text to copy, not picking a line
      select(i, true);
    });
    s.addEventListener("pointerenter", () => setHover(i));
    s.addEventListener("pointerleave", () => state.hover === i && setHover(-1));
    return s;
  }));
}

function renderSelected(scroll = false) {
  const l = state.lines[state.selected];
  if (!l) {
    els.result.hidden = true;
    return;
  }
  renderResult(l);
  const li = els.lines.children[state.selected];
  if (li && els.result.parentElement !== li) li.append(els.result);
  if (scroll && li) {
    const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    li.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "nearest" });
  }
}

function renderResult(l) {
  const { image } = state, S = setOf(l), { vocab } = S;
  const engine = l.engine;
  const text = l.text;
  const det = ctcDetails(l.logits, engine.timesteps, engine.channels, engine.numClasses, engine.logitsExp);
  const detOk = det.length === l.ids.length && det.every((c, i) => c.id === l.ids[i]);
  const pct = (p) => `${Math.round(p * 100)}%`;
  const tok = (id) => vocab[id - 1] ?? "?";

  els.result.hidden = false;
  els.kana.replaceChildren();
  let detail = null;
  if (!text) {
    const span = document.createElement("span");
    span.className = "empty";
    span.textContent = "No text recognised in this line.";
    els.kana.append(span);
  } else {
    l.ids.forEach((id, i) => {
      const span = document.createElement("span");
      span.setAttribute("data-no-i18n", "");
      span.className = "ch";
      span.textContent = tok(id);
      const c = detOk ? det[i] : null;
      if (c && c.prob < UNSURE && tok(id) !== " ") {
        span.classList.add("unsure");
        span.tabIndex = 0;
        span.setAttribute("role", "button");
        const alt = c.alt > 0 ? `, maybe ${tok(c.alt)} (${pct(c.altProb)})` : "";
        const msg = `${tok(id)} ${pct(c.prob)}${alt}`;
        span.setAttribute("aria-label", `Unsure: ${msg}`);
        span.title = msg;
        const show = () => {
          els.kana.querySelectorAll(".ch.on").forEach((x) => x.classList.remove("on"));
          span.classList.add("on");
          detail ||= document.createElement("div");
          detail.className = "detail";
          detail.innerHTML = "";
          const b = document.createElement("b");
          b.textContent = tok(id);
          detail.append(b, ` ${pct(c.prob)} sure`);
          if (c.alt > 0) {
            const b2 = document.createElement("b");
            b2.textContent = tok(c.alt);
            detail.append(", runner-up ", b2, ` ${pct(c.altProb)}`);
          }
          els.romaji.after(detail);
        };
        span.addEventListener("click", show);
        span.addEventListener("keydown", (ev) => (ev.key === "Enter" || ev.key === " ") && (ev.preventDefault(), show()));
      }
      els.kana.append(span);
    });
  }
  els.result.querySelectorAll(".detail").forEach((x) => x.remove());
  els.romaji.textContent = text ? toRomaji(text) : "";
  els.romaji.setAttribute("data-no-i18n", "");
  els.copyLine.hidden = !(text && text.trim());
  if (els.copyLine.dataset.line !== String(state.selected)) els.copyLine.classList.remove("done"); // "Copied" was for another line
  els.copyLine.dataset.line = String(state.selected);

  // example lines carry their label; compare when the whole example was read as one line
  const ex = image.example;
  if (ex && ex.kind !== "page" && ex.text && state.lines.length === 1) {
    els.expected.hidden = false;
    els.expected.innerHTML = "";
    const k = document.createElement("span");
    k.className = "kana-inline";
    k.textContent = ex.text;
    const verdict = document.createElement("span");
    verdict.className = text === ex.text ? "ok" : "no";
    verdict.textContent = text === ex.text ? " ✓ match" : " ✗ differs";
    els.expected.append("Label: ", k, ` (${toRomaji(ex.text)})`, verdict);
  } else {
    els.expected.hidden = true;
  }

  // the strip the model saw
  const { pixels, width, height, contentWidth } = l.prep;
  const id = new ImageData(width, height);
  for (let i = 0; i < pixels.length; i++) {
    id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = pixels[i];
    id.data[4 * i + 3] = 255;
  }
  els.strip.width = width;
  els.strip.height = height;
  els.strip.style.aspectRatio = `${width} / ${height}`;
  els.strip.getContext("2d").putImageData(id, 0, 0);
  els.padMark.hidden = contentWidth >= width;
  els.padMark.style.left = `${(100 * contentWidth) / width}%`;
  els.stripCaption.textContent = `What the ${l.vertical && S.rec.v ? "vertical" : "horizontal"} reader saw: ` +
    (l.vertical ? "the column turned on its side, " : "the line ") + `as a ${height} × ${width} px grayscale strip.`;

  const notes = [];
  if (l.prep.squeezed) {
    notes.push(["warn", `This line is ${l.prep.naturalWidth} px long at ${height} px height, more than the reader's ${width} px, ` +
      `so it was squeezed ${(l.prep.naturalWidth / width).toFixed(1)}× to fit. Expect errors: drag boxes around about ` +
      `${Math.round(width / height)} glyphs at a time.`]);
  }
  if (l.prep.upscaled) {
    notes.push(["warn", "The text is small in this image, so it was scaled up. Get closer (or use a sharper image) so the small " +
      "dakuten marks survive."]);
  }
  if (l.vertical && !S.rec.v) notes.push(["warn", "No vertical reader is loaded, so this column was read by the horizontal one."]);
  if (l.source === "box" && l.box && l.box.fitted) notes.push(["info", "Auto-fit: read the box tightened around the text."]);
  if (l.source === "tap") notes.push(["info", "Found by looking closer around your tap."]);
  if (l.conf < minConf(S) && l.ids.length) {
    notes.push(["warn", `The reader is unsure of this line (${Math.round(100 * l.conf)}% on average): it may not be Hunter script at all ` +
      "(Latin or Japanese text, art)."]);
  }
  els.notices.replaceChildren(...notes.map(([kind, t]) => {
    const li = document.createElement("li");
    if (kind === "info") li.className = "info";
    li.textContent = t;
    return li;
  }));
  els.timing.textContent = `Line ${state.selected + 1} of ${state.lines.length} · WebAssembly${state.simd ? " SIMD" : ""}`;
}

init();
