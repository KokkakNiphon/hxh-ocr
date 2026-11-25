// Cloudflare deployment of the browser demo: an assets-only Worker (no server code).
// Build the files with `python ../scripts/build_site.py`, then `npx cf deploy ./dist`
// (or `npm run deploy`, which does both).
import { defineConfig } from "cf/config";

export default defineConfig({
  worker: {
    name: "hxh-ocr",
    compatibilityDate: "2026-11-25",
  },
});
