// Record a scripted session of a page in headless Chrome and encode it as a GIF (for the README).
//
//   node scripts/readme/capture.mjs <scenario.json> <out.gif> [--width 1024] [--height 640]
//        [--scale 1] [--mobile] [--theme dark|light] [--cursor arrow|dot] [--fps 12] [--colors 160]
//        [--out-width N (default: the recorded width)] [--keep]
//
// The scenario is a JSON list of steps:
//   ["goto", url]                 load a page (fresh profile, so a first visit)
//   ["record", sel?, pad?, settleMs?]  start recording frames (Chrome's screencast); with sel, only that
//                                 element plus pad px, scrolled into the window first
//   ["stop"]                      stop recording
//   ["wait", ms]
//   ["until", js, timeoutMs?]     poll a JS expression until it is truthy
//   ["move", sel, ms?, fx?, fy?]  glide the pointer to a point of an element (centre by default)
//   ["click", sel, ms?, fx?, fy?] glide there, then press and release
//   ["eval", js]                  run JS in the page
//   ["hold", ms]                  show the last recorded frame this long (default 50 ms: seamless loops)
// Headless screenshots show no pointer, so a drawn one (an arrow, or a dot for a phone) follows the
// mouse and rings on each press. Frames keep their real timing; ffmpeg resamples them to --fps and
// builds one palette for the whole clip. Needs Chrome (or CHROME_PATH) and ffmpeg on PATH.
//
// The README's GIFs (docs/readme/) are figures of the "How it works" page, one loop each, recorded
// from one reset of the figure's animation to the next so they play seamlessly:
//   node scripts/readme/capture.mjs scripts/readme/decoder.json docs/readme/decoder.gif --width 1200 --height 1000 --fps 15 --colors 128
//   node scripts/readme/capture.mjs scripts/readme/maker.json docs/readme/training_picture.gif --width 1200 --height 1000 --fps 15 --colors 128
//   node scripts/readme/capture.mjs scripts/readme/ctc.json docs/readme/ctc_reading.gif --width 1200 --height 1000 --fps 15 --colors 128
//   node scripts/readme/capture.mjs scripts/readme/turn.json docs/readme/vertical_turn.gif --width 1200 --height 1000 --fps 15 --colors 128
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(`--${k}`); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes(`--${k}`);
const VALUED = new Set(["--width", "--height", "--scale", "--theme", "--cursor", "--fps", "--colors", "--out-width"]);
const [scenarioPath, outGif] = argv.filter((a, i) => !a.startsWith("--") && !VALUED.has(argv[i - 1]));
if (!scenarioPath || !outGif) { console.error("usage: node scripts/readme/capture.mjs <scenario.json> <out.gif> [options]"); process.exit(2); }
const W = +opt("width", 1024), H = +opt("height", 640), SCALE = +opt("scale", 1), FPS = +opt("fps", 12);
const COLORS = +opt("colors", 160), OUTW = +opt("out-width", 0), THEME = opt("theme", "dark"), CURSOR = opt("cursor", "arrow");
const steps = JSON.parse(readFileSync(scenarioPath, "utf8"));

function chromePath() {
  const c = [process.env.CHROME_PATH, "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    process.env.LOCALAPPDATA && join(process.env.LOCALAPPDATA, "Google\\Chrome\\Application\\chrome.exe"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"];
  const p = c.find((x) => x && existsSync(x));
  if (!p) throw new Error("no Chrome found; set CHROME_PATH");
  return p;
}

const CURSOR_JS = `(() => {
  const dot = ${JSON.stringify(CURSOR === "dot")};
  const mk = () => {
    if (document.getElementById("__rc")) return;
    const c = document.createElement("div"); c.id = "__rc";
    c.innerHTML = dot
      ? '<div style="width:34px;height:34px;border-radius:50%;background:rgba(255,255,255,.55);border:2px solid rgba(0,0,0,.55);box-shadow:0 1px 6px rgba(0,0,0,.35)"></div>'
      : '<svg width="24" height="24" viewBox="0 0 24 24"><path d="M4 2.5v17.2l4.6-4.3 3.2 6.9 2.9-1.3-3.2-6.8h6.4z" fill="#fff" stroke="#111" stroke-width="1.5" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, { position: "fixed", left: "-60px", top: "-60px", zIndex: 2147483647, pointerEvents: "none",
      transform: dot ? "translate(-17px,-17px)" : "translate(-4px,-2px)", transition: "transform .08s" });
    const r = document.createElement("div"); r.id = "__rr";
    Object.assign(r.style, { position: "fixed", zIndex: 2147483646, pointerEvents: "none", width: "36px", height: "36px",
      margin: "-18px 0 0 -18px", borderRadius: "50%", border: "3px solid rgba(110,135,255,.95)", opacity: 0 });
    document.documentElement.append(r, c);
    addEventListener("mousemove", (e) => { c.style.left = e.clientX + "px"; c.style.top = e.clientY + "px"; }, true);
    addEventListener("mousedown", (e) => {
      r.style.left = e.clientX + "px"; r.style.top = e.clientY + "px";
      r.animate([{ opacity: 1, transform: "scale(.35)" }, { opacity: 0, transform: "scale(1.5)" }], { duration: 520, easing: "ease-out" });
      if (dot) c.firstChild.animate([{ transform: "scale(1)" }, { transform: "scale(.8)" }, { transform: "scale(1)" }], { duration: 260 });
    }, true);
  };
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", mk); else mk();
})();`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const profile = mkdtempSync(join(tmpdir(), "readme-cap-"));
  const port = 9300 + Math.floor(Math.random() * 500);
  const chrome = spawn(chromePath(), ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
    "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required", "about:blank"], { stdio: "ignore" });
  let target;
  for (let i = 0; i < 100 && !target; i++) {
    await sleep(100);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === "page"); } catch {}
  }
  if (!target) throw new Error("Chrome did not start");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0; const pending = new Map(); const listeners = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { const { r, j } = pending.get(d.id); pending.delete(d.id); d.error ? j(new Error(d.error.message)) : r(d.result); }
    else if (d.method) listeners.forEach((f) => f(d));
  };
  const send = (method, params = {}) => new Promise((r, j) => { const i = ++id; pending.set(i, { r, j }); ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async (js) => {
    const r = await send("Runtime.evaluate", { expression: js, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(`${js.slice(0, 80)}: ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
    return r.result.value;
  };

  await send("Page.enable"); await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: SCALE, mobile: flag("mobile") });
  if (flag("mobile")) await send("Emulation.setTouchEmulationEnabled", { enabled: false });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: THEME }, { name: "prefers-reduced-motion", value: "no-preference" }] });
  await send("Page.addScriptToEvaluateOnNewDocument", { source: CURSOR_JS });

  const tmp = mkdtempSync(join(tmpdir(), "readme-frames-"));
  const frames = []; let recording = false; let holdExtra = 0; let crop = null;
  listeners.push((d) => {
    if (d.method !== "Page.screencastFrame") return;
    send("Page.screencastFrameAck", { sessionId: d.params.sessionId }).catch(() => {});
    if (!recording) return;
    const f = join(tmp, `${String(frames.length).padStart(5, "0")}.png`);
    writeFileSync(f, Buffer.from(d.params.data, "base64"));
    frames.push({ f, t: d.params.metadata.timestamp });
  });
  let mouse = [W / 2, H + 40];
  // the point of an element; one outside the window is first scrolled smoothly to the middle
  const point = async (sel, fx = 0.5, fy = 0.5) => {
    const js = `(() => { const e = document.querySelector(${JSON.stringify(sel)});
      if (!e) throw new Error("no element " + ${JSON.stringify(sel)}); const r = e.getBoundingClientRect();
      return [r.left + r.width * ${fx}, r.top + r.height * ${fy}, r.top < 60 || r.bottom > innerHeight]; })()`;
    const p = await evaluate(js);
    if (!p[2]) return p;
    await evaluate(`document.querySelector(${JSON.stringify(sel)}).scrollIntoView({ block: "center", behavior: "smooth" })`);
    await sleep(800);
    return evaluate(js);
  };
  const glide = async (to, ms) => {
    const [x0, y0] = mouse, n = Math.max(1, Math.round(ms / 16));
    for (let k = 1; k <= n; k++) {
      const s = k / n, e = s < 0.5 ? 2 * s * s : 1 - (-2 * s + 2) ** 2 / 2;
      mouse = [x0 + (to[0] - x0) * e, y0 + (to[1] - y0) * e];
      await send("Input.dispatchMouseEvent", { type: "mouseMoved", x: mouse[0], y: mouse[1] });
      await sleep(16);
    }
  };
  const loaded = () => new Promise((r) => { const f = (d) => { if (d.method === "Page.loadEventFired") { listeners.splice(listeners.indexOf(f), 1); r(); } }; listeners.push(f); });

  for (const [cmd, ...a] of steps) {
    if (cmd === "goto") { const l = loaded(); await send("Page.navigate", { url: a[0] }); await l; }
    else if (cmd === "record") {
      if (a[0]) {
        // record only this element (plus a margin): bring it into the window first, then fix the crop
        const pad = a[1] ?? 0;
        crop = await evaluate(`(() => { const e = document.querySelector(${JSON.stringify(a[0])});
          if (!e) throw new Error("no element " + ${JSON.stringify(a[0])});
          window.scrollTo({ top: e.getBoundingClientRect().top + scrollY - Math.max(0, (innerHeight - e.offsetHeight) / 2), behavior: "instant" });
          const r = e.getBoundingClientRect(), p = ${pad};
          const x = Math.max(0, r.left - p), y = Math.max(0, r.top - p);
          return [x, y, Math.min(innerWidth, r.right + p) - x, Math.min(innerHeight, r.bottom + p) - y]; })()`);
        await sleep(a[2] ?? 600);
      }
      recording = true;
      await send("Page.startScreencast", { format: "png", everyNthFrame: 1, maxWidth: W * SCALE, maxHeight: H * SCALE });
    } else if (cmd === "stop") { await sleep(80); recording = false; await send("Page.stopScreencast"); }
    else if (cmd === "wait") await sleep(a[0]);
    else if (cmd === "hold") holdExtra += a[0];
    else if (cmd === "until") {
      const t0 = Date.now();
      while (!(await evaluate(a[0]))) { if (Date.now() - t0 > (a[1] || 20000)) throw new Error(`timeout: ${a[0]}`); await sleep(100); }
    } else if (cmd === "move") await glide(await point(a[0], a[2], a[3]), a[1] ?? 500);
    else if (cmd === "click") {
      await glide(await point(a[0], a[2], a[3]), a[1] ?? 500); await sleep(90);
      await send("Input.dispatchMouseEvent", { type: "mousePressed", x: mouse[0], y: mouse[1], button: "left", clickCount: 1 });
      await sleep(70);
      await send("Input.dispatchMouseEvent", { type: "mouseReleased", x: mouse[0], y: mouse[1], button: "left", clickCount: 1 });
    } else if (cmd === "scroll") {
      // put an element's top this many px below the window's top (instant before recording, smooth after)
      await evaluate(`window.scrollTo({ top: document.querySelector(${JSON.stringify(a[0])}).getBoundingClientRect().top + scrollY - ${a[1] || 0},
        behavior: ${JSON.stringify(recording ? "smooth" : "instant")} })`);
      await sleep(recording ? 800 : 100);
    } else if (cmd === "eval") console.log("eval:", JSON.stringify(await evaluate(a[0])));
    else throw new Error(`unknown step ${cmd}`);
  }
  if (recording) { recording = false; await send("Page.stopScreencast"); }
  ws.close(); chrome.kill(); await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch {}

  if (frames.length < 2) throw new Error(`only ${frames.length} frames recorded`);
  // concat list with each frame's real duration (the last one shown for "hold", so a loop recorded
  // from one reset to the next plays seamlessly)
  let list = "";
  frames.forEach((fr, i) => {
    const dur = i + 1 < frames.length ? Math.max(0.001, frames[i + 1].t - fr.t) : Math.max(0.05, holdExtra / 1000);
    list += `file '${fr.f.replace(/\\/g, "/")}'\nduration ${dur.toFixed(4)}\n`;
  });
  list += `file '${frames.at(-1).f.replace(/\\/g, "/")}'\n`;
  writeFileSync(join(tmp, "list.txt"), list);
  const total = frames.at(-1).t - frames[0].t + Math.max(0.05, holdExtra / 1000);
  // the frames' real pixel size (a screencast may come back at CSS size): read it from the PNG header
  const head = readFileSync(frames[0].f).subarray(16, 24);
  const fw = head.readUInt32BE(0), fh = head.readUInt32BE(4), k = fw / W;
  console.log(`frames ${fw} x ${fh}${crop ? `, crop (css px) ${crop.map((v) => v.toFixed(0)).join(" ")}` : ""}`);
  const c = crop && crop.map((v) => Math.floor(v * k / 2) * 2);
  if (c) { c[2] = Math.min(c[2], fw - c[0]); c[3] = Math.min(c[3], fh - c[1]); }
  const vf = (c ? `crop=${c[2]}:${c[3]}:${c[0]}:${c[1]},` : "") + `fps=${FPS},scale=${OUTW || (c ? c[2] : fw)}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=${COLORS}:stats_mode=full[p];` +
    `[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle`;
  const r = spawnSync("ffmpeg", ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", join(tmp, "list.txt"), "-vf", vf, "-loop", "0", resolve(outGif)], { stdio: "inherit" });
  if (r.status !== 0) throw new Error("ffmpeg failed");
  if (flag("keep")) console.log("frames kept in", tmp); else rmSync(tmp, { recursive: true, force: true });
  console.log(`${outGif}: ${frames.length} frames, ${total.toFixed(1)} s, ${(statSync(outGif).size / 1024 / 1024).toFixed(2)} MB`);
}

main().catch((e) => { console.error(e.message || e); process.exit(1); });
