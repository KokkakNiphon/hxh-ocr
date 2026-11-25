// Read one photo through exactly the browser code path (web/ocr.js readPage) and print JSON:
// every detected line with its quad, orientation, reading and confidence, plus its rectified crop
// (written next to the output as <out>_crop<i>.gray, width x height in the JSON).
// Driven by scripts/eval_photos.py, which makes the gray input and draws the overlays.
//
//   node web/test/read_photo.mjs <gray.u8> <W> <H> <out-prefix> [--min-conf 0.8] [--det path] [--tile]
//        [--rec path] [--rec-v path] [--det-json path] [--rec-json path]
// --det-json / --rec-json default to the .json next to --det / --rec when it exists (model family
// layout, experiments/family/<size>/), else web/models/det.json / model.json.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { HxhDetector, HxhEngine, idsToText, readPage } from "../ocr.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const [grayPath, Ws, Hs, outPrefix] = process.argv.slice(2);
const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const W = Number(Ws), H = Number(Hs);
const sibling = (p, d) => (p && existsSync(p.replace(/\.hxhm$/, ".json")) ? p.replace(/\.hxhm$/, ".json") : d);
const detMeta = JSON.parse(readFileSync(opt("--det-json", sibling(opt("--det"), join(root, "web", "models", "det.json"))), "utf8"));
const vocab = JSON.parse(readFileSync(opt("--rec-json", sibling(opt("--rec"), join(root, "web", "models", "model.json"))), "utf8")).vocab;
const mod = await WebAssembly.compile(readFileSync(join(root, "web", "hxh_infer.wasm")));
const det = await HxhDetector.create(mod, readFileSync(opt("--det", join(root, "web", "models", "det.hxhm"))));
const rec = {
  h: await HxhEngine.create(mod, readFileSync(opt("--rec", join(root, "web", "models", "model.hxhm")))),
  v: await HxhEngine.create(mod, readFileSync(opt("--rec-v", join(root, "web", "models", "model_v.hxhm")))),
};
const gray = new Uint8Array(readFileSync(grayPath));
const params = detMeta.params || {};
const pad = detMeta.pad || { along: 0.3, across: 0.2 };
const r = readPage(det, rec, gray, W, H, { params, pad, tiles: argv.includes("--tiles"), minConf: detMeta.min_conf ?? 0.8 });
const lines = r.lines.map((l, i) => {
  const f = `${outPrefix}_crop${i}.gray`;
  writeFileSync(f, l.crop.pixels);
  return {
    quad: l.quad, vertical: l.vertical, score: l.score, text: idsToText(l.ids, vocab), conf: l.conf,
    crop: { file: f, w: l.crop.w, h: l.crop.h },
  };
});
console.log(JSON.stringify({ W, H, minConf: detMeta.min_conf ?? null, minShowScore: detMeta.min_show_score ?? 0, ms: Math.round(r.tDetect + r.tRead + r.tLetterbox + r.tPost), lines }));
